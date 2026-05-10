"""Compile + forward/backward validate mined chain candidates.

Phase 1 (build + validate_graph + CompiledLayer construct): structural gate.
Phase 2 (forward + backward smoke test): runs a random tensor through the
compiled layer, checks output is finite, and verifies backprop produces
finite parameter gradients. Filters out chains that compile but die at
runtime — the gate auto-registration relies on.

Phase 3+ (holdout training) is still deferred; chains that pass Phase 2
are *eligible* to be registered as live templates but should still go
through the standard screening pipeline before promotion to validation.

The validator is read-only on the project state: builds graphs in memory,
records pass/fail, returns annotated candidate dicts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..synthesis.graph import ComputationGraph
from ..synthesis.primitives import PRIMITIVE_REGISTRY
from ..synthesis.validator import validate_graph


_DEFAULT_MODEL_DIM = 64
_DEFAULT_MAX_OPS = 20
_DEFAULT_MAX_DEPTH = 15
_DEFAULT_SMOKE_BATCH = 2
_DEFAULT_SMOKE_SEQ = 8


def _add_chain_op(graph: ComputationGraph, op_name: str, current: int) -> int:
    """Append an op from the chain. Skip ops with non-1 input arity since the
    mined chains are linear sequences and we have no second-input source."""
    prim = PRIMITIVE_REGISTRY.get(op_name)
    if prim is None:
        raise ValueError(f"unknown op: {op_name}")
    if prim.n_inputs != 1:
        # Mined chains are flat sequences; multi-input ops need a router or
        # signal source we don't synthesize here. Mark these for Phase 2.
        raise ValueError(
            f"op {op_name} requires {prim.n_inputs} inputs; chain validator handles arity-1 only"
        )
    return graph.add_op(op_name, [current])


def _build_chain_graph(
    chain: Iterable[str],
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> ComputationGraph:
    """Build a graph: input → rmsnorm → [chain ops] → fix_dim → output."""
    graph = ComputationGraph(model_dim=model_dim)
    current = graph.add_input()
    current = graph.add_op("rmsnorm", [current])
    for op_name in chain:
        current = _add_chain_op(graph, op_name, current)
    cur_dim = graph.nodes[current].output_shape.dim
    if cur_dim != model_dim:
        op = "linear_proj_down" if cur_dim > model_dim else "linear_proj_up"
        current = graph.add_op(op, [current], config={"out_dim": model_dim})
    graph.set_output(current)
    return graph


def _run_smoke_test(
    compiled_layer: Any,
    *,
    model_dim: int,
    batch: int = _DEFAULT_SMOKE_BATCH,
    seq: int = _DEFAULT_SMOKE_SEQ,
) -> Dict[str, Any]:
    """Run forward + backward on the compiled layer. Returns finite/NaN flags."""
    import torch

    out: Dict[str, Any] = {
        "forward_passed": False,
        "backward_passed": False,
        "output_has_nan": False,
        "output_has_inf": False,
        "param_grad_finite": True,
        "smoke_error": None,
    }
    x = torch.randn(batch, seq, model_dim, requires_grad=True)
    try:
        y = compiled_layer(x)
    except Exception as exc:
        out["smoke_error"] = f"forward: {type(exc).__name__}: {exc}"
        return out
    out["forward_passed"] = True
    out["output_has_nan"] = bool(torch.isnan(y).any())
    out["output_has_inf"] = bool(torch.isinf(y).any())
    if out["output_has_nan"] or out["output_has_inf"]:
        out["smoke_error"] = "output non-finite"
        return out
    try:
        y.sum().backward()
    except Exception as exc:
        out["smoke_error"] = f"backward: {type(exc).__name__}: {exc}"
        return out
    # Verify parameter gradients are finite when present.
    for p in compiled_layer.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            out["param_grad_finite"] = False
            out["smoke_error"] = "param grad non-finite"
            return out
    out["backward_passed"] = True
    return out


def validate_chain(
    chain: Iterable[str],
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
    max_ops: int = _DEFAULT_MAX_OPS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    run_smoke: bool = True,
) -> Dict[str, Any]:
    """Try to build + validate + compile + smoke-test a chain.

    Phase 1 fields:
        compile_passed, validate_passed, n_ops, failure_mode, error
    Phase 2 fields (when ``run_smoke=True`` and compile passed):
        forward_passed, backward_passed, output_has_nan, output_has_inf,
        param_grad_finite, smoke_error
    """
    chain_list = [str(op) for op in chain]
    result: Dict[str, Any] = {
        "compile_passed": False,
        "validate_passed": False,
        "forward_passed": False,
        "backward_passed": False,
        "n_ops": 0,
        "failure_mode": None,
        "error": None,
    }
    try:
        graph = _build_chain_graph(chain_list, model_dim=model_dim)
    except ValueError as exc:
        result["failure_mode"] = "build"
        result["error"] = str(exc)
        return result
    except KeyError as exc:
        result["failure_mode"] = "build_missing_op"
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["failure_mode"] = "build_exception"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["n_ops"] = len(graph.nodes) - 1  # exclude input

    validation = validate_graph(graph, max_ops=max_ops, max_depth=max_depth)
    if validation.errors:
        result["failure_mode"] = "validate"
        result["error"] = "; ".join(validation.errors[:3])
        return result
    result["validate_passed"] = True

    # Lazy compile via the layer-level helper (wires dispatch handlers).
    # CompiledLayer alone leaves ops without execute_fn handlers, so we use
    # the same path the runner uses to assemble a single trainable layer.
    try:
        from ..synthesis.compiler import _compile_layer_module
    except ImportError as exc:
        result["failure_mode"] = "compile_import"
        result["error"] = str(exc)
        return result
    try:
        compiled = _compile_layer_module(graph, prefer_fast_path=True)
    except Exception as exc:
        result["failure_mode"] = "compile"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["compile_passed"] = True

    if not run_smoke:
        return result

    smoke = _run_smoke_test(compiled, model_dim=model_dim)
    result.update(smoke)
    if not smoke["backward_passed"] and result["failure_mode"] is None:
        result["failure_mode"] = "smoke"
        result["error"] = smoke.get("smoke_error")
    return result


def annotate_candidates_with_validation(
    candidates: List[Dict[str, Any]],
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
) -> List[Dict[str, Any]]:
    """Add a ``validation`` block to each candidate.

    Mutates each input dict for memory efficiency, then returns the same
    list. The block uses the same shape returned by ``validate_chain``.
    """
    for candidate in candidates:
        chain = candidate.get("chain") or []
        candidate["validation"] = validate_chain(chain, model_dim=model_dim)
    return candidates


def filter_to_passing(
    candidates: List[Dict[str, Any]],
    *,
    require_validate: bool = True,
    require_compile: bool = True,
    require_forward: bool = False,
    require_backward: bool = False,
) -> List[Dict[str, Any]]:
    """Return only candidates whose validation passed the requested gates."""
    out: List[Dict[str, Any]] = []
    for candidate in candidates:
        v = candidate.get("validation") or {}
        if require_validate and not v.get("validate_passed"):
            continue
        if require_compile and not v.get("compile_passed"):
            continue
        if require_forward and not v.get("forward_passed"):
            continue
        if require_backward and not v.get("backward_passed"):
            continue
        out.append(candidate)
    return out
