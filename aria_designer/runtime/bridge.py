"""
Runtime Bridge: aria_designer workflow → research/ eval pipeline.

Converts aria_designer WorkflowGraphModel JSON into research ComputationGraph,
then drives compilation, sandbox evaluation, fingerprinting, and novelty scoring.

Usage:
    from runtime.bridge import evaluate_workflow, workflow_to_graph

    result = evaluate_workflow(workflow_json, device="cuda")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from research.defaults import MODEL_DIM, VOCAB_SIZE
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from research.mathspaces.registry import register_all_mathspaces
from research.synthesis.component_registry import registry
from research.synthesis.workflow_converter import workflow_to_computation_graph as _w2cg
from research.synthesis.result_schemas import BridgeResult

# Ensure mathspace primitives are available for resolution.
register_all_mathspaces()

# Stable I/O leaves that do not map to backend primitives.
_IO_COMPONENTS = {"graph_input", "graph_output", "input", "output", "output_head"}

# ── Backward-compatible primitive resolution ─────────────────────────


def _normalize_component_leaf(component_type: str) -> str:
    """Normalize any category/id component token into a lowercased leaf id."""
    token = str(component_type or "").strip().lower()
    if not token:
        return ""
    return token.split("/")[-1]


def _resolve_primitive(component_type: str) -> Optional[str]:
    """Resolve frontend component tokens to backend primitive names.

    Returns `None` for explicit I/O components and raises for unknown tokens.
    """
    leaf = _normalize_component_leaf(component_type)
    if not leaf:
        raise ValueError("Unknown component: empty component type")
    if leaf in _IO_COMPONENTS:
        return None

    if leaf in PRIMITIVE_REGISTRY:
        return leaf
    raise ValueError(f"Unknown component: {component_type}")


# ── Component Mapping Helpers (Delegating to Registry) ───────────────


def get_component_execution_capability(component_type: str) -> Dict[str, Any]:
    """Return bridge execution capability metadata for a component type."""
    parts = str(component_type or "").split("/")
    cid = parts[-1]
    category = parts[0] if len(parts) > 1 else None
    category_class = registry.category_execution_class.get(
        category or "", "primitive_candidate"
    )

    # Handle IO components
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

    # Delegate to centralized registry for non-primitive lowering kinds.
    if registry.is_source(component_type):
        kind, fidelity, warn = (
            "source",
            "approximate",
            "Source lowering to deterministic graph input.",
        )
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": kind,
            "execution_class": category_class,
            "semantic_fidelity": fidelity,
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": [warn],
            "reason": f"Supported via {kind} lowering.",
        }
    elif registry.is_passthrough(component_type):
        kind, fidelity, warn = (
            "passthrough",
            "approximate",
            "Passthrough lowering (wire-through identity).",
        )
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": kind,
            "execution_class": category_class,
            "semantic_fidelity": fidelity,
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": [warn],
            "reason": f"Supported via {kind} lowering.",
        }
    elif cid in registry.template_lowered_components:
        kind, fidelity, warn = (
            "template",
            "approximate",
            "Template lowering to primitive subgraph.",
        )
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": kind,
            "execution_class": "composite",
            "semantic_fidelity": fidelity,
            "bridge_supported": True,
            "primitive_name": None,
            "warnings": [warn],
            "reason": f"Supported via {kind} lowering.",
        }

    # Primitive/direct/alias path.
    try:
        primitive = _resolve_primitive(component_type)
    except ValueError:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "unsupported",
            "execution_class": category_class
            if category_class != "primitive_candidate"
            else "unsupported",
            "semantic_fidelity": "unsupported",
            "bridge_supported": False,
            "primitive_name": None,
            "warnings": [],
            "reason": "No primitive mapping registered.",
        }

    if primitive is not None:
        return {
            "component_type": component_type,
            "component_leaf": cid,
            "mapping_kind": "direct",
            "execution_class": "primitive",
            "semantic_fidelity": "exact",
            "bridge_supported": True,
            "primitive_name": primitive,
            "warnings": [],
            "reason": "Direct primitive mapping.",
        }

    return {
        "component_type": component_type,
        "component_leaf": cid,
        "mapping_kind": "unsupported",
        "execution_class": "unsupported",
        "semantic_fidelity": "unsupported",
        "bridge_supported": False,
        "primitive_name": None,
        "warnings": [],
        "reason": "No primitive mapping registered.",
    }


# ── Workflow → ComputationGraph conversion ───────────────────────────


def workflow_to_graph(
    workflow_json: Dict[str, Any],
    model_dim: int = MODEL_DIM,
    return_id_map: bool = False,
) -> Any:
    """Convert an aria_designer workflow JSON to a research ComputationGraph."""
    return _w2cg(workflow_json, model_dim, return_id_map)


# ── Compression / Efficiency Analysis ─────────────────────────────────


@dataclass(slots=True)
class CompressionResult:
    pruning_curve: List[Dict[str, float]] = field(default_factory=list)
    baseline_loss: float = 0.0
    dense_params: int = 0
    effective_params: int = 0
    compression_ratio: float = 1.0
    sparse_ops: int = 0
    total_ops: int = 0
    sparse_op_coverage: float = 0.0
    theoretical_size_fp16_mb: float = 0.0
    efficiency_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_compression(
    model,
    graph,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cpu",
    batch_size: int = 2,
    seq_len: int = 64,
) -> CompressionResult:
    """Analyze compression characteristics (minimal implementation for bridge).

    Accepts batch/sequence knobs for API compatibility; current baseline analysis
    only uses model and graph-level statistics.
    """
    res = CompressionResult()
    res.dense_params = sum(p.numel() for p in model.parameters())
    res.total_ops = graph.n_ops()
    res.theoretical_size_fp16_mb = (res.dense_params * 2) / (1024 * 1024)
    return res


def bridge_analyze_routing(model, graph) -> List[Dict[str, Any]]:
    """Extract per-op routing telemetry from a live model."""
    results = []
    for name, module in model.named_modules():
        rt = getattr(module, "routing_telemetry", None)
        if rt:
            try:
                node_id = int(name.split(".")[-1])
                results.append(
                    {
                        "node_id": node_id,
                        "op_name": getattr(module, "op_name", "unknown"),
                        "savings_ratio": rt.get("savings_ratio"),
                        "heatmap": rt.get("heatmap"),
                    }
                )
            except (ValueError, IndexError):
                continue
    return results


# ── Main evaluation pipeline ─────────────────────────────────────────


def evaluate_workflow(
    workflow_json: Dict[str, Any],
    model_dim: int = MODEL_DIM,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cpu",
    run_fingerprint: bool = True,
    run_novelty: bool = True,
    batch_size: int = 2,
    seq_len: int = 128,
) -> BridgeResult:
    """Full pipeline: workflow → graph → compile → eval → fingerprint → novelty."""
    result = BridgeResult(status="error")
    t0 = time.monotonic()
    fp = None

    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
        result.graph_fingerprint = graph.fingerprint()
        result.n_ops = graph.n_ops()
        result.depth = graph.depth()
        result.n_params_estimate = graph.n_params_estimate()
        result.has_gradient_path = bool(graph.has_gradient_path())

        from research.synthesis.compiler import compile_model

        model = compile_model([graph], vocab_size=vocab_size)
        model.to(device)

        from research.eval.sandbox import safe_eval

        sandbox_res = safe_eval(
            model,
            batch_size=max(1, int(batch_size)),
            seq_len=max(1, int(seq_len)),
            vocab_size=vocab_size,
            device=device,
            run_stability_probe=False,
        )
        result.sandbox = sandbox_res
        if not sandbox_res.passed:
            result.status = "failed_sandbox"
            result.error = sandbox_res.error
            return result

        if run_fingerprint:
            from research.eval.fingerprint import compute_fingerprint

            fp = compute_fingerprint(
                model,
                seq_len=min(max(1, int(seq_len)), 64),
                model_dim=model_dim,
                vocab_size=vocab_size,
                device=device,
            )
            result.fingerprint.cka_vs_transformer = getattr(
                fp, "cka_vs_transformer", 0.0
            )
            result.fingerprint.cka_vs_ssm = getattr(fp, "cka_vs_ssm", 0.0)
            result.fingerprint.cka_vs_conv = getattr(fp, "cka_vs_conv", 0.0)
            result.fingerprint.interaction_locality = getattr(
                fp, "interaction_locality", 0.0
            )
            result.fingerprint.interaction_sparsity = getattr(
                fp, "interaction_sparsity", 0.0
            )
            result.fingerprint.intrinsic_dim = getattr(fp, "intrinsic_dim", 0.0)
            result.fingerprint.isotropy = getattr(fp, "isotropy", 0.0)
            result.fingerprint.behavioral_novelty = getattr(fp, "novelty_score", 0.0)
            result.fingerprint.most_similar_to = max(
                {
                    "transformer": result.fingerprint.cka_vs_transformer,
                    "ssm": result.fingerprint.cka_vs_ssm,
                    "conv": result.fingerprint.cka_vs_conv,
                },
                key=lambda k: {
                    "transformer": result.fingerprint.cka_vs_transformer,
                    "ssm": result.fingerprint.cka_vs_ssm,
                    "conv": result.fingerprint.cka_vs_conv,
                }[k],
            )

        if run_novelty:
            from research.eval.metrics import novelty_score

            metrics = novelty_score(graph, fingerprint=fp)
            result.fingerprint.structural_novelty = metrics.structural_novelty
            result.fingerprint.behavioral_novelty = metrics.behavioral_novelty
            result.fingerprint.overall_novelty = metrics.overall_novelty
            result.fingerprint.most_similar_to = metrics.most_similar_to

        result.status = "success"
    except Exception as e:
        result.error = str(e)
        result.error_stage = "pipeline"

    result.total_time_ms = (time.monotonic() - t0) * 1000
    return result


# ── Utility functions for the API layer ──────────────────────────────


def list_available_primitives() -> List[Dict[str, Any]]:
    """List all primitives available for use in aria_designer workflows."""
    return [
        {
            "name": op.name,
            "category": str(op.category),
            "n_inputs": op.n_inputs,
            "config_keys": list(op.config_keys),
        }
        for name, op in sorted(PRIMITIVE_REGISTRY.items())
    ]


def validate_workflow_graph(
    workflow_json: Dict[str, Any], model_dim: int = MODEL_DIM
) -> Dict[str, Any]:
    """Validate that a workflow can be converted to a valid ComputationGraph."""
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)

        # Wiring constraint check — catch misconnected signal chains
        from research.synthesis.primitives import validate_wiring

        wiring_errors = validate_wiring(graph)

        if wiring_errors:
            return {
                "valid": False,
                "error": f"Wiring error: {wiring_errors[0]}",
                "wiring_errors": wiring_errors,
            }

        return {
            "valid": True,
            "graph_info": {
                "fingerprint": graph.fingerprint(),
                "n_params_estimate": int(graph.n_params_estimate()),
                "n_ops": graph.n_ops(),
                "depth": graph.depth(),
                "model_dim": model_dim,
                "has_gradient_path": bool(graph.has_gradient_path()),
            },
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def estimate_performance(
    workflow_json: Dict[str, Any], model_dim: int = MODEL_DIM
) -> Dict[str, Any]:
    """Quick performance estimate without running the model."""
    try:
        graph = workflow_to_graph(workflow_json, model_dim=model_dim)
        op_counts: Dict[str, int] = {}
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            if getattr(node, "is_input", False):
                continue
            op = str(getattr(node, "op_name", "unknown"))
            op_counts[op] = op_counts.get(op, 0) + 1
        n_params = int(graph.n_params_estimate())
        # Coarse estimate: roughly 2 FLOPs per parameter per token.
        flops_per_token = max(1, n_params * 2)
        return {
            "valid": True,
            "n_params_estimate": n_params,
            "flops_per_token_estimate": int(flops_per_token),
            "n_ops": graph.n_ops(),
            "depth": graph.depth(),
            "op_counts": op_counts,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
