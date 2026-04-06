"""
Runtime Bridge: aria_designer workflow → research/ eval pipeline.

Converts aria_designer WorkflowGraphModel JSON into research ComputationGraph,
then drives compilation, sandbox evaluation, fingerprinting, and novelty scoring.

Usage:
    from aria_designer.runtime.bridge import evaluate_workflow, workflow_to_graph

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


def _capability_result(
    component_type: str,
    cid: str,
    *,
    mapping_kind: str,
    execution_class: str,
    fidelity: str,
    supported: bool,
    primitive: str | None = None,
    warnings: list | None = None,
    reason: str,
) -> Dict[str, Any]:
    return {
        "component_type": component_type,
        "component_leaf": cid,
        "mapping_kind": mapping_kind,
        "execution_class": execution_class,
        "semantic_fidelity": fidelity,
        "bridge_supported": supported,
        "primitive_name": primitive,
        "warnings": warnings or [],
        "reason": reason,
    }


def get_component_execution_capability(component_type: str) -> Dict[str, Any]:
    """Return bridge execution capability metadata for a component type."""
    parts = str(component_type or "").split("/")
    cid = parts[-1]
    category = parts[0] if len(parts) > 1 else None
    category_class = registry.category_execution_class.get(
        category or "", "primitive_candidate"
    )

    if cid in _IO_COMPONENTS:
        return _capability_result(
            component_type,
            cid,
            mapping_kind="io",
            execution_class="io",
            fidelity="exact",
            supported=True,
            reason="IO passthrough node.",
        )

    # Non-primitive lowering kinds
    _LOWERING = {
        "source": ("Source lowering to deterministic graph input.", category_class),
        "passthrough": (
            "Passthrough lowering (wire-through identity).",
            category_class,
        ),
        "template": ("Template lowering to primitive subgraph.", "composite"),
    }
    if registry.is_source(component_type):
        kind = "source"
    elif registry.is_passthrough(component_type):
        kind = "passthrough"
    elif cid in registry.template_lowered_components:
        kind = "template"
    else:
        kind = None

    if kind is not None:
        warn_msg, exec_class = _LOWERING[kind]
        return _capability_result(
            component_type,
            cid,
            mapping_kind=kind,
            execution_class=exec_class,
            fidelity="approximate",
            supported=True,
            warnings=[warn_msg],
            reason=f"Supported via {kind} lowering.",
        )

    # Primitive / direct path
    try:
        primitive = _resolve_primitive(component_type)
    except ValueError:
        exec_class = (
            category_class if category_class != "primitive_candidate" else "unsupported"
        )
        return _capability_result(
            component_type,
            cid,
            mapping_kind="unsupported",
            execution_class=exec_class,
            fidelity="unsupported",
            supported=False,
            reason="No primitive mapping registered.",
        )

    if primitive is not None:
        return _capability_result(
            component_type,
            cid,
            mapping_kind="direct",
            execution_class="primitive",
            fidelity="exact",
            supported=True,
            primitive=primitive,
            reason="Direct primitive mapping.",
        )

    return _capability_result(
        component_type,
        cid,
        mapping_kind="unsupported",
        execution_class="unsupported",
        fidelity="unsupported",
        supported=False,
        reason="No primitive mapping registered.",
    )


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
                        "keep_drop_ratio": rt.get("keep_drop_ratio"),
                        "lane_histogram": rt.get("lane_histogram")
                        .detach()
                        .cpu()
                        .tolist()
                        if hasattr(rt.get("lane_histogram"), "detach")
                        else rt.get("lane_histogram"),
                        "route_confidence_mean": rt.get("route_confidence_mean"),
                        "span_type": rt.get("span_type"),
                        "gate_type": rt.get("gate_type"),
                        "trace_payload": rt.get("trace_payload"),
                    }
                )
            except (ValueError, IndexError):
                continue
    return results


def _validate_hybrid_routing_workflow(workflow_json: Dict[str, Any]) -> list[str]:
    nodes = workflow_json.get("nodes", [])
    node_types = {str(node.get("component_type", "")).split("/")[-1] for node in nodes}
    errors: list[str] = []
    has_hybrid_router = "hybrid_sparse_router" in node_types
    has_default_path = "default_path" in node_types
    if has_hybrid_router and not has_default_path:
        errors.append("Routed blocks must define a default_path node.")
    for node in nodes:
        leaf = str(node.get("component_type", "")).split("/")[-1]
        params = node.get("params", {})
        if leaf == "sparse_span_builder":
            if "span_width" not in params:
                errors.append(
                    f"sparse_span_builder '{node.get('id')}' must define span_width."
                )
            if "fallback_behavior" not in params:
                errors.append(
                    f"sparse_span_builder '{node.get('id')}' must define fallback_behavior."
                )
        if leaf == "hybrid_sparse_router":
            if int(params.get("lane_count", 0) or 0) <= 0:
                errors.append(
                    f"hybrid_sparse_router '{node.get('id')}' must define lane_count."
                )
            if (
                "lane_conditioned_block" not in node_types
                and "default_path" not in node_types
            ):
                errors.append(
                    "lane_router must define downstream lanes or lane-conditioned edges."
                )
        if (
            leaf in {"learned_token_gate", "confidence_token_gate", "hybrid_token_gate"}
            and not has_default_path
        ):
            errors.append(f"{leaf} is used without a cheap/default_path.")
    return list(dict.fromkeys(errors))


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
            _cka = {
                "transformer": result.fingerprint.cka_vs_transformer,
                "ssm": result.fingerprint.cka_vs_ssm,
                "conv": result.fingerprint.cka_vs_conv,
            }
            result.fingerprint.most_similar_to = max(_cka, key=_cka.get)

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
        routing_errors = _validate_hybrid_routing_workflow(workflow_json)
        if routing_errors:
            return {
                "valid": False,
                "error": routing_errors[0],
                "routing_errors": routing_errors,
                "design_suggestions": [
                    "recommend sparse routing when many cheap/default ops are present",
                    "recommend pair/triplet routing around local structural operators",
                ],
            }
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
            "design_suggestions": [
                msg
                for msg in (
                    "recommend sparse routing when graph contains many cheap/default ops"
                    if any(
                        str(node.get("component_type", "")).split("/")[-1]
                        in {
                            "default_path",
                            "hybrid_token_gate",
                            "learned_token_gate",
                            "confidence_token_gate",
                        }
                        for node in workflow_json.get("nodes", [])
                    )
                    else None,
                    "recommend pair/triplet routing when graph contains local structural operators"
                    if any(
                        str(node.get("component_type", "")).split("/")[-1]
                        in {
                            "conv_only",
                            "local_window_attn",
                            "adjacent_token_merge",
                            "sparse_span_builder",
                        }
                        for node in workflow_json.get("nodes", [])
                    )
                    else None,
                )
                if msg
            ],
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
