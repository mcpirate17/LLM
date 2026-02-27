"""
Runtime Bridge: aria-designer workflow → research/ eval pipeline.

Converts aria-designer WorkflowGraphModel JSON into research ComputationGraph,
then drives compilation, sandbox evaluation, fingerprinting, and novelty scoring.

Usage:
    from runtime.bridge import evaluate_workflow, workflow_to_graph

    result = evaluate_workflow(workflow_json, model_dim=256, device="cuda")
"""

from __future__ import annotations

import copy
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Ensure research/ is importable
_RESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research"))
if _RESEARCH_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_RESEARCH_ROOT))

from research.synthesis.graph import ComputationGraph
from research.synthesis.primitives import PRIMITIVE_REGISTRY, get_primitive
from research.mathspaces.registry import register_all_mathspaces
from research.synthesis.component_registry import registry, fe_type_to_op_name
from research.synthesis.workflow_converter import workflow_to_computation_graph as _w2cg

# Ensure mathspace primitives are available for resolution.
register_all_mathspaces()


# ── Component ID → Primitive Name mapping ────────────────────────────

def _execution_class(component_leaf: str, category: Optional[str]) -> str:
    if component_leaf in registry.component_execution_class:
        return str(registry.component_execution_class[component_leaf])
    if category and category in registry.category_execution_class:
        return str(registry.category_execution_class[category])
    return "primitive"


def _is_passthrough_component(component_leaf: str) -> bool:
    return component_leaf in registry.passthrough_components


def _is_source_component(component_leaf: str) -> bool:
    return component_leaf in registry.source_components


def _is_template_lowered_component(component_leaf: str) -> bool:
    return component_leaf in registry.template_lowered_components


def _alias_semantic_info(component_leaf: str, primitive_name: str) -> Dict[str, Any]:
    """Return semantic fidelity metadata for alias mappings."""
    if component_leaf == primitive_name:
        return {"semantic_fidelity": "exact", "warnings": []}
    note = registry.approximate_alias_notes.get(component_leaf)
    if note:
        return {"semantic_fidelity": "approximate", "warnings": [str(note)]}
    return {"semantic_fidelity": "exact", "warnings": []}


def get_component_execution_capability(component_type: str) -> Dict[str, Any]:
    """Return bridge execution capability metadata for a component type."""
    parts = component_type.split("/")
    cid = parts[-1]
    category = parts[0] if len(parts) > 1 else None

    if cid in _IO_COMPONENTS:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "io",
            "execution_class": "io",
            "semantic_fidelity": "exact",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": [],
            "reason": "IO passthrough node.",
        }

    if _is_source_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "source",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Data-source lowered to deterministic graph input in bridge mode."],
            "reason": "Supported via data-source lowering in bridge (deterministic eval fallback).",
        }

    if _is_passthrough_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "passthrough",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Component is currently wire-through passthrough in bridge mode."],
            "reason": "Supported via passthrough lowering (wire-through).",
        }

    if _is_template_lowered_component(cid):
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "template",
            "execution_class": _execution_class(cid, category),
            "semantic_fidelity": "approximate",
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": ["Template component expands to an approximate primitive subgraph."],
            "reason": "Supported via template lowering (expanded primitive subgraph).",
        }

    if cid in PRIMITIVE_REGISTRY:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "direct",
            "execution_class": "primitive",
            "semantic_fidelity": "exact",
            "bridge_supported": True,
            "primitive_name": cid,
            "warnings": [],
            "reason": "Direct primitive mapping.",
        }

    if cid in registry.aliases:
        alias_target = registry.aliases[cid]
        semantic = _alias_semantic_info(cid, alias_target)
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "alias",
            "execution_class": "primitive",
            "semantic_fidelity": semantic["semantic_fidelity"],
            "bridge_supported": True,
            "primitive_name": alias_target,
            "warnings": semantic["warnings"],
            "reason": "Mapped via component alias.",
        }

    exec_class = _execution_class(cid, category)
    return {
        "component_type": component_type,
        "component_leaf": cid,
        "mapping_kind": "unsupported",
        "execution_class": exec_class,
        "semantic_fidelity": "unsupported",
        "bridge_supported": False,
        "primitive_name": None,
        "warnings": [],
        "reason": (
            "No primitive mapping. Requires lowering/expansion path."
            if exec_class in {"composite", "data_control", "control"}
            else "No primitive mapping registered."
        ),
    }


# ── Workflow → ComputationGraph conversion ───────────────────────────

def workflow_to_graph(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
    return_id_map: bool = False,
) -> ComputationGraph | Tuple[ComputationGraph, Dict[str, int]]:
    """Convert an aria-designer workflow JSON to a research ComputationGraph."""
    return _w2cg(workflow_json, model_dim, return_id_map)


# ── Compression / Efficiency Analysis ─────────────────────────────────

# Sparse ops and their effective density factors (fraction of weights actually used)
_SPARSE_OP_DENSITY = {
    "nm_sparse_linear": 0.50,        # 2:4 sparsity → 50% density
    "block_sparse_linear": 0.25,     # default block_density
    "semi_structured_2_4_linear": 0.50,
}


@dataclass
class CompressionResult:
    """Result of compression & efficiency analysis."""
    # Pruning curve: list of {sparsity, loss, loss_ratio}
    pruning_curve: List[Dict[str, float]] = field(default_factory=list)
    baseline_loss: float = 0.0
    pruning_tolerance: float = 0.0  # 0-1, how gracefully it degrades

    # Param compression
    dense_params: int = 0
    effective_params: int = 0
    compression_ratio: float = 1.0  # dense / effective

    # Sparse op coverage
    sparse_ops: int = 0
    total_ops: int = 0
    sparse_op_coverage: float = 0.0
    sparse_op_names: List[str] = field(default_factory=list)

    # Theoretical sizes
    theoretical_size_fp16_mb: float = 0.0
    theoretical_size_int8_mb: float = 0.0
    theoretical_size_int4_mb: float = 0.0

    # Composite scores
    memory_efficiency_score: float = 0.0  # 0-1, smaller → higher
    triton_compatibility_score: float = 0.0
    efficiency_score: float = 0.0         # composite 0-1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        return d


def analyze_compression(
    model,
    graph,
    vocab_size: int = 32000,
    device: str = "cpu",
    batch_size: int = 2,
    seq_len: int = 64,
) -> CompressionResult:
    """Analyze compression characteristics: pruning tolerance, sparse coverage, sizes."""
    import torch

    result = CompressionResult()

    # --- 1. Walk graph → sparse op coverage & effective params ---
    sparse_names = []
    total_op_count = 0
    effective_param_count = 0

    for node in graph.nodes.values():
        if node.is_input:
            continue
        total_op_count += 1
        op_name = node.op_name
        if op_name in _SPARSE_OP_DENSITY:
            sparse_names.append(op_name)
            density = node.config.get("block_density", _SPARSE_OP_DENSITY[op_name])
        else:
            density = 1.0

        # Estimate params for this op
        if op_name in PRIMITIVE_REGISTRY:
            prim = PRIMITIVE_REGISTRY[op_name]
            if prim.has_params:
                D = graph.model_dim
                out_dim = node.config.get("out_dim", D)
                op_params = D * out_dim  # rough estimate
                effective_param_count += int(op_params * density)

    # Get dense param count from model
    dense_params = sum(p.numel() for p in model.parameters())
    if effective_param_count == 0:
        effective_param_count = dense_params

    result.dense_params = dense_params
    result.effective_params = effective_param_count
    result.compression_ratio = dense_params / max(effective_param_count, 1)
    result.sparse_ops = len(sparse_names)
    result.total_ops = total_op_count
    result.sparse_op_coverage = len(sparse_names) / max(total_op_count, 1)
    result.sparse_op_names = sorted(set(sparse_names))

    # --- 2. Theoretical sizes ---
    result.theoretical_size_fp16_mb = (dense_params * 2) / (1024 * 1024)
    result.theoretical_size_int8_mb = (dense_params * 1) / (1024 * 1024)
    result.theoretical_size_int4_mb = (dense_params * 0.5) / (1024 * 1024)

    # --- 3. Pruning curve (one-shot at 4 sparsity levels) ---
    try:
        from research.eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss

        dev = torch.device(device)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

        # Baseline loss
        baseline = estimate_lm_ce_loss(model, [input_ids], dev)
        if baseline is not None and baseline > 0:
            result.baseline_loss = baseline
            sparsity_levels = [0.25, 0.50, 0.75, 0.90]
            curve = []

            for sp in sparsity_levels:
                t0 = time.monotonic()
                try:
                    pruned = copy.deepcopy(model)
                    pruned.to(dev)
                    apply_one_shot_pruning(pruned, target_sparsity=sp)
                    loss = estimate_lm_ce_loss(pruned, [input_ids], dev)
                    del pruned
                    if loss is not None:
                        ratio = loss / max(baseline, 1e-8)
                        curve.append({"sparsity": sp, "loss": round(loss, 4), "loss_ratio": round(ratio, 4)})
                except Exception:
                    pass
                # Time-box: skip remaining levels if this one exceeded 5s
                if (time.monotonic() - t0) > 5.0:
                    break

            result.pruning_curve = curve

            # Pruning tolerance: inverse of average loss ratio degradation
            if curve:
                avg_ratio = sum(p["loss_ratio"] for p in curve) / len(curve)
                # tolerance=1 means no degradation, tolerance=0 means 2x+ degradation
                result.pruning_tolerance = max(0.0, min(1.0, 2.0 - avg_ratio))
    except Exception:
        pass

    # --- 4. Memory efficiency score (smaller models → higher) ---
    # Scale: 0 at 100M+ params, 1 at <1M params (log scale)
    import math
    if dense_params > 0:
        log_p = math.log10(max(dense_params, 1))
        result.memory_efficiency_score = max(0.0, min(1.0, (8.0 - log_p) / 2.0))

    try:
        from research.eval.triton_compatibility import check_triton_compatibility
        triton_res = check_triton_compatibility(graph)
        result.triton_compatibility_score = triton_res.coverage_score
    except Exception:
        result.triton_compatibility_score = 0.0

    # --- 5. Composite efficiency score ---
    result.efficiency_score = (
        0.25 * result.pruning_tolerance
        + 0.20 * min(result.compression_ratio / 4.0, 1.0)
        + 0.15 * result.sparse_op_coverage
        + 0.20 * result.triton_compatibility_score
        + 0.20 * getattr(result, "memory_efficiency_score", 0.5)
    )

    return result


from research.synthesis.result_schemas import BridgeResult, SandboxResult, FingerprintResult

# ── Main evaluation pipeline ─────────────────────────────────────────

def evaluate_workflow(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
    vocab_size: int = 32000,
    device: str = "cpu",
    run_fingerprint: bool = True,
    run_novelty: bool = True,
    batch_size: int = 2,
    seq_len: int = 128,
) -> BridgeResult:
    """Full pipeline: workflow → graph → compile → eval → fingerprint → novelty.

    Args:
        workflow_json: aria-designer workflow in workflow_graph.v1 format
        model_dim: feature dimension (D in (B, S, D))
        vocab_size: vocabulary size for the synthesized model
        device: "cpu" or "cuda"
        run_fingerprint: whether to compute behavioral fingerprint
        run_novelty: whether to compute novelty scores
        batch_size: batch size for evaluation
        seq_len: sequence length for evaluation

    Returns:
        BridgeResult with all evaluation metrics
    """
    result = BridgeResult(status="error")
    t0 = time.monotonic()

    # Step 1: Convert workflow to ComputationGraph
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
    except ValueError as e:
        result.error = str(e)
        result.error_stage = "conversion"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 2: Graph-level analysis (no GPU needed)
    result.graph_fingerprint = graph.fingerprint()
    result.n_ops = graph.n_ops()
    result.depth = graph.depth()
    result.n_params_estimate = graph.n_params_estimate()
    result.has_gradient_path = bool(graph.has_gradient_path())

    from research.eval.triton_compatibility import check_triton_compatibility
    triton_res = check_triton_compatibility(graph)
    result.triton_compatibility_score = triton_res.coverage_score

    if not result.has_gradient_path:
        result.status = "error"
        result.error = "No differentiable path from input to output"
        result.error_stage = "analysis"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 3: Compile to PyTorch module
    try:
        from research.synthesis.compiler import compile_model
        model = compile_model([graph], vocab_size=vocab_size)
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.error_stage = "compilation"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 4: Sandbox evaluation (Stage 0 / 0.5)
    try:
        from research.eval.sandbox import safe_eval
        sandbox_res = safe_eval(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
        )
        result.sandbox = sandbox_res

        if not sandbox_res.passed:
            result.status = "failed_sandbox"
            result.error = sandbox_res.error
            result.error_stage = sandbox_res.stage
            result.total_time_ms = (time.monotonic() - t0) * 1000
            return result
    except Exception as e:
        result.status = "error"
        result.error = f"Sandbox error: {e}"
        result.error_stage = "sandbox"
        result.total_time_ms = (time.monotonic() - t0) * 1000
        return result

    # Step 5: Behavioral fingerprint (Stage 1)
    if run_fingerprint:
        try:
            from research.eval.fingerprint import compute_fingerprint
            fp_res = compute_fingerprint(
                model,
                seq_len=min(seq_len, 64),
                model_dim=model_dim,
                vocab_size=vocab_size,
                device=device,
            )
            result.fingerprint.cka_vs_transformer = getattr(fp_res, "cka_vs_transformer", 0.0)
            result.fingerprint.cka_vs_ssm = getattr(fp_res, "cka_vs_ssm", 0.0)
            result.fingerprint.cka_vs_conv = getattr(fp_res, "cka_vs_conv", 0.0)
            result.fingerprint.interaction_locality = getattr(fp_res, "interaction_locality", 0.0)
            result.fingerprint.interaction_sparsity = getattr(fp_res, "interaction_sparsity", 0.0)
            result.fingerprint.intrinsic_dim = getattr(fp_res, "intrinsic_dim", 0.0)
            result.fingerprint.isotropy = getattr(fp_res, "isotropy", 0.0)
        except Exception:
            pass  # Fingerprint is optional; don't fail the whole eval

    # Step 6: Novelty scoring
    if run_novelty:
        try:
            from research.eval.metrics import novelty_score
            fp_obj = None
            if run_fingerprint:
                try:
                    from research.eval.fingerprint import compute_fingerprint
                    fp_obj = compute_fingerprint(
                        model, seq_len=min(seq_len, 64),
                        model_dim=model_dim, vocab_size=vocab_size,
                        device=device,
                    )
                except Exception:
                    pass
            metrics = novelty_score(graph, fingerprint=fp_obj)
            result.fingerprint.structural_novelty = metrics.structural_novelty
            result.fingerprint.behavioral_novelty = metrics.behavioral_novelty
            result.fingerprint.overall_novelty = metrics.overall_novelty
            result.fingerprint.most_similar_to = getattr(metrics, "most_similar_to", "")
        except Exception:
            pass  # Novelty is optional

    result.status = "success"
    result.total_time_ms = (time.monotonic() - t0) * 1000
    return result


# ── Utility functions for the API layer ──────────────────────────────

def list_available_primitives() -> List[Dict[str, Any]]:
    """List all primitives available for use in aria-designer workflows."""
    result = []
    for name, op in sorted(PRIMITIVE_REGISTRY.items()):
        result.append({
            "name": op.name,
            "category": op.category.value if hasattr(op.category, "value") else str(op.category),
            "n_inputs": op.n_inputs,
            "shape_rule": op.shape_rule,
            "has_params": op.has_params,
            "param_formula": op.param_formula,
            "config_keys": list(op.config_keys) if op.config_keys else [],
        })
    return result


def validate_workflow_graph(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
) -> Dict[str, Any]:
    """Validate that a workflow can be converted to a valid ComputationGraph.

    Returns {"valid": True, "graph_info": {...}} or {"valid": False, "error": "..."}.
    """
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
        return {
            "valid": True,
            "graph_info": {
                "fingerprint": graph.fingerprint(),
                "n_ops": graph.n_ops(),
                "depth": graph.depth(),
                "n_params_estimate": int(graph.n_params_estimate()),
                "has_gradient_path": bool(graph.has_gradient_path()),
                "model_dim": model_dim,
            },
        }
    except ValueError as e:
        return {"valid": False, "error": str(e)}


def estimate_performance(
    workflow_json: Dict[str, Any],
    model_dim: int = 256,
) -> Dict[str, Any]:
    """Quick performance estimate without running the model.

    Returns param count, estimated FLOPs, op breakdown.
    """
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    # Op histogram
    op_counts: Dict[str, int] = {}
    for node in graph.nodes.values():
        if not node.is_input:
            op_counts[node.op_name] = op_counts.get(node.op_name, 0) + 1

    # Category histogram
    cat_counts: Dict[str, int] = {}
    for node in graph.nodes.values():
        if not node.is_input and node.op_name in PRIMITIVE_REGISTRY:
            cat = PRIMITIVE_REGISTRY[node.op_name].category
            cat_name = cat.value if hasattr(cat, "value") else str(cat)
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

    # Rough FLOP estimate (per token)
    D = model_dim
    flops = 0
    for node in graph.nodes.values():
        if node.is_input:
            continue
        if node.op_name in PRIMITIVE_REGISTRY:
            op = PRIMITIVE_REGISTRY[node.op_name]
            if op.shape_rule == "linear":
                out_dim = node.config.get("out_dim", D)
                flops += 2 * D * out_dim  # matmul
            elif op.shape_rule == "matmul":
                flops += 2 * D * D
            elif op.shape_rule == "identity":
                flops += D  # elementwise
            elif op.shape_rule == "binary_broadcast":
                flops += D
            else:
                flops += D  # conservative estimate

    return {
        "valid": True,
        "n_params_estimate": graph.n_params_estimate(),
        "flops_per_token_estimate": flops,
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "op_counts": op_counts,
        "category_counts": cat_counts,
        "has_gradient_path": graph.has_gradient_path(),
    }
