"""Compile-validate mined chain candidates emitted by the template promoter.

Phase 1 of the auto-promotion pipeline: take a candidate chain
``(op_a, op_b, ..., op_k)`` and check that it can actually be assembled
into a valid ComputationGraph that compiles into a CompiledLayer.

This is a structural gate, not a quality gate. A chain that fails to
compile cannot become a useful template even if its mining stats are
strong; conversely, a chain that compiles is not yet proven useful (forward
correctness, gradient health, training holdout — all Phase 2+).

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


def validate_chain(
    chain: Iterable[str],
    *,
    model_dim: int = _DEFAULT_MODEL_DIM,
    max_ops: int = _DEFAULT_MAX_OPS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> Dict[str, Any]:
    """Try to build + validate a graph from the chain.

    Returns a dict:
        compile_passed: bool
        validate_passed: bool
        n_ops: int — final op count of the wrapper graph
        failure_mode: str | None — first error category
        error: str | None — first error message
    """
    chain_list = [str(op) for op in chain]
    result: Dict[str, Any] = {
        "compile_passed": False,
        "validate_passed": False,
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

    # Lazy compile check: importing CompiledLayer pulls torch + heavy deps.
    # The promoter pipeline calls this many times, so guard the import.
    try:
        from ..synthesis.compiled_model import CompiledLayer
    except ImportError as exc:
        result["failure_mode"] = "compile_import"
        result["error"] = str(exc)
        return result
    try:
        CompiledLayer(graph)
    except Exception as exc:
        result["failure_mode"] = "compile"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["compile_passed"] = True
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
) -> List[Dict[str, Any]]:
    """Return only candidates whose validation passed the requested gates."""
    out: List[Dict[str, Any]] = []
    for candidate in candidates:
        v = candidate.get("validation") or {}
        if require_validate and not v.get("validate_passed"):
            continue
        if require_compile and not v.get("compile_passed"):
            continue
        out.append(candidate)
    return out
