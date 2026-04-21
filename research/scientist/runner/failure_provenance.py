from __future__ import annotations

import json
from typing import Any

from ..native.core import _try_import_rust_scheduler
from ...synthesis.graph_validator import validate_dim_flow
from ...synthesis.serializer import graph_to_json
from ...synthesis.validator import validate_graph

_GENERIC_SINK_OPS = frozenset(
    {
        "add",
        "rmsnorm",
        "layernorm",
        "linear_proj",
        "linear_proj_up",
        "linear_proj_down",
        "identity",
    }
)

_ROUTING_CHAIN_OPS = (
    "hybrid_token_gate",
    "sparse_span_builder",
    "hybrid_sparse_router",
    "lane_conditioned_block",
    "calibrated_branch_merge",
)

_THREE_WAY_ROUTING_OPS = (
    "hetero_moe",
    "arch_router",
    "compute_budget_router",
    "difficulty_blend_3way",
    "adaptive_lane_mixer",
)

_VARIABLE_EXPERT_ROUTING_OPS = (
    "moe_2expert",
    "moe_topk",
    "relu_gated_moe",
)

_COUNT_CONTRACT_OPS = (
    "token_class_proj",
    "token_type_classifier",
    "dual_compression_blend",
    "compression_mixture_experts",
    "moe_2expert",
    "moe_topk",
    "relu_gated_moe",
    "hetero_moe",
    "arch_router",
    "compute_budget_router",
    "difficulty_blend_3way",
    "adaptive_lane_mixer",
)

_ROOT_SOURCE_PREFERENCES: dict[str, tuple[str, ...]] = {
    "selective_scan_contract": ("selective_scan", "conv1d_seq", "silu"),
    "token_merge_contract": ("token_merge", "adjacent_token_merge"),
    "hybrid_routing_assembly": _ROUTING_CHAIN_OPS,
    "routing_dead_path": _ROUTING_CHAIN_OPS,
    "routing_collapse": _ROUTING_CHAIN_OPS,
    "routing_telemetry_state_mismatch": _VARIABLE_EXPERT_ROUTING_OPS
    + _THREE_WAY_ROUTING_OPS
    + ("token_class_proj", "token_type_classifier"),
    "stale_routing_bias_state": _VARIABLE_EXPERT_ROUTING_OPS + _THREE_WAY_ROUTING_OPS,
    "causality_contract": (
        "token_merge",
        "adjacent_token_merge",
        "hybrid_sparse_router",
        "sparse_span_builder",
        "hybrid_token_gate",
        "selective_scan",
        "ternary_projection",
        "identity",
    ),
    "cuda_kernel_or_illegal_access": (
        "nm_sparse_linear",
        "block_sparse_linear",
        "token_class_proj",
        "token_type_classifier",
    ),
    "dtype_mismatch": (
        "linear_proj",
        "linear_proj_up",
        "linear_proj_down",
        "token_class_proj",
        "token_type_classifier",
        "block_sparse_linear",
        "latent_attention_compressor",
        "conv1d_seq",
    ),
    "shape_or_residual_mismatch": (
        "split2",
        "split3",
        "concat",
        "token_merge",
        "adjacent_token_merge",
        "linear_proj_down",
        "linear_proj_up",
        "token_class_proj",
        "token_type_classifier",
    ),
    "residual_dominant_no_learning": (
        "swiglu_mlp",
        "linear_proj_down",
        "linear_proj_up",
        "token_class_proj",
        "token_type_classifier",
        "identity",
    ),
    "generalization_failure": (
        "hybrid_sparse_router",
        "hybrid_token_gate",
        "selective_scan",
        "swiglu_mlp",
        "token_class_proj",
        "token_type_classifier",
    ),
}


def _normalize_error_type(error_type: str | None) -> str:
    normalized = str(error_type or "unknown").strip()
    if normalized.startswith("s1_"):
        normalized = normalized[3:]
    return normalized or "unknown"


def _graph_op_names(graph: Any) -> list[str]:
    names: list[str] = []
    for node in graph.nodes.values():
        if getattr(node, "is_input", False) or getattr(node, "is_output", False):
            continue
        op_name = getattr(node, "op_name", "")
        if op_name:
            names.append(str(op_name))
    return names


def _native_graph_provenance(
    graph: Any, failure_op: str | None
) -> tuple[list[str], str | None] | None:
    rust = _try_import_rust_scheduler()
    if rust is None:
        return None
    try:
        graph_json = graph_to_json(graph)
    except Exception:
        return None
    payload = None
    if hasattr(rust, "analyze_graph_provenance_native_py"):
        try:
            payload = rust.analyze_graph_provenance_native_py(
                graph_json,
                sorted(_GENERIC_SINK_OPS),
                failure_op,
            )
        except Exception:
            payload = None
    if payload is None:
        if not hasattr(rust, "analyze_graph_provenance_native"):
            return None
        try:
            raw = rust.analyze_graph_provenance_native(
                graph_json,
                sorted(_GENERIC_SINK_OPS),
                failure_op,
            )
        except Exception:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(payload, dict):
        return None
    raw_op_names = payload.get("op_names")
    if not isinstance(raw_op_names, list):
        return None
    op_names = [str(op) for op in raw_op_names if str(op)]
    source_op = payload.get("source_op")
    if source_op is not None:
        source_op = str(source_op) or None
    return op_names, source_op


def _graph_generic_sink_ratio(op_names: list[str]) -> float:
    if not op_names:
        return 0.0
    return sum(op in _GENERIC_SINK_OPS for op in op_names) / len(op_names)


def _pick_failure_op(op_names: list[str], preferred: tuple[str, ...]) -> str | None:
    for op_name in preferred:
        if op_name in op_names:
            return op_name
    for op_name in op_names:
        if op_name not in _GENERIC_SINK_OPS:
            return op_name
    return op_names[0] if op_names else None


def _classify_from_validation(
    validation_errors: list[str],
    dim_flow_errors: list[str],
    op_names: list[str],
) -> tuple[str, str | None]:
    joined = "\n".join((*validation_errors, *dim_flow_errors))
    if "selective_scan" in joined:
        return "selective_scan_contract", "selective_scan"
    if (
        "split2 branch restore mismatch" in joined
        or "split2 concat restore mismatch" in joined
    ):
        return "split_branch_restore_contract", _pick_failure_op(
            op_names, ("split2", "concat", "linear_proj")
        )
    if any(op in joined for op in _ROUTING_CHAIN_OPS):
        return "hybrid_routing_assembly", _pick_failure_op(op_names, _ROUTING_CHAIN_OPS)
    if "token_merge" in joined or "adjacent_token_merge" in joined:
        return "token_merge_contract", _pick_failure_op(
            op_names, ("token_merge", "adjacent_token_merge")
        )
    if any(
        phrase in joined
        for phrase in (
            "dim mismatch",
            "needs model_dim",
            "input[",
            "Output dim ",
            "Graph is skip-only",
            "Too few effective ops",
            "No parameterized ops on reachable path",
        )
    ):
        if any(
            phrase in joined
            for phrase in (
                "Graph is skip-only",
                "Too few effective ops",
                "No parameterized ops on reachable path",
            )
        ):
            return "residual_dominant_no_learning", _pick_failure_op(
                op_names, ("identity", "add", "linear_proj", "swiglu_mlp")
            )
        return "shape_or_residual_mismatch", _pick_failure_op(
            op_names,
            (
                "split2",
                "split3",
                "concat",
                "linear_proj_down",
                "linear_proj_up",
                "add",
            ),
        )
    return "", None


def _classify_from_error(
    error_type: str,
    error_message: str,
    op_names: list[str],
) -> tuple[str, str | None]:
    msg = (error_message or "").lower()
    if error_type in {"RuntimeError", "shape_mismatch"} and (
        "bfloat16" in msg or "expected scalar type" in msg
    ):
        return "dtype_mismatch", _pick_failure_op(
            op_names,
            (
                "linear_proj",
                "linear_proj_up",
                "linear_proj_down",
                "token_class_proj",
                "token_type_classifier",
                "block_sparse_linear",
                "latent_attention_compressor",
                "conv1d_seq",
            ),
        )
    if (
        error_type == "RuntimeError"
        and "size of tensor a (3)" in msg
        and ("size of tensor b (2)" in msg or "size of tensor b (8)" in msg)
        and any(op in op_names for op in _THREE_WAY_ROUTING_OPS)
        and any(op in op_names for op in _VARIABLE_EXPERT_ROUTING_OPS)
    ):
        return "stale_routing_bias_state", _pick_failure_op(
            op_names,
            _VARIABLE_EXPERT_ROUTING_OPS + _THREE_WAY_ROUTING_OPS,
        )
    if (
        error_type == "RuntimeError"
        and "size of tensor a (" in msg
        and "size of tensor b (" in msg
        and any(op in op_names for op in _COUNT_CONTRACT_OPS)
    ):
        return "routing_telemetry_state_mismatch", _pick_failure_op(
            op_names, _COUNT_CONTRACT_OPS
        )
    if error_type in {"RuntimeError", "shape_mismatch"} and "size of tensor a" in msg:
        return "shape_or_residual_mismatch", _pick_failure_op(
            op_names,
            (
                "split2",
                "split3",
                "concat",
                "linear_proj_down",
                "linear_proj_up",
                "add",
            ),
        )
    if error_type == "cuda_fatal":
        return "cuda_kernel_or_illegal_access", _pick_failure_op(
            op_names, _ROOT_SOURCE_PREFERENCES["cuda_kernel_or_illegal_access"]
        )
    if error_type in {"causality_violation", "forward_error"} and (
        "causality gate failed" in msg or "looks ahead at future tokens" in msg
    ):
        return "causality_contract", _pick_failure_op(
            op_names, _ROOT_SOURCE_PREFERENCES["causality_contract"]
        )
    if error_type == "rapid_screening_error":
        if "routing entropy" in msg or "entropy gate collapsed" in msg:
            return "routing_collapse", _pick_failure_op(op_names, _ROUTING_CHAIN_OPS)
        if "grad norm " in msg and " at step " in msg:
            return "grad_explosion", _pick_failure_op(
                op_names,
                (
                    "hybrid_token_gate",
                    "hybrid_sparse_router",
                    "gated_lane_blend",
                    "depth_weighted_proj",
                    "moe_topk",
                    "moe_2expert",
                    "relu_gated_moe",
                ),
            )
        if "loss is nan" in msg or "loss is inf" in msg or "gradient nan/inf" in msg:
            return "nonfinite_training_dynamics", _pick_failure_op(op_names, ())
        if "loss spiked from" in msg:
            return "loss_spike_instability", _pick_failure_op(
                op_names,
                (
                    "hybrid_sparse_router",
                    "hybrid_token_gate",
                    "selective_scan",
                    "token_merge",
                    "swiglu_mlp",
                ),
            )
        if "no learning after" in msg:
            if any(op in op_names for op in _ROUTING_CHAIN_OPS):
                return "routing_dead_path", _pick_failure_op(
                    op_names, _ROUTING_CHAIN_OPS
                )
            return "rapid_no_learning_signal", _pick_failure_op(
                op_names, ("swiglu_mlp", "linear_proj", "token_class_proj")
            )
        if "loss " in msg and (" at step 25" in msg or " at step 50" in msg):
            if any(op in op_names for op in _ROUTING_CHAIN_OPS):
                return "routing_dead_path", _pick_failure_op(
                    op_names, _ROUTING_CHAIN_OPS
                )
            return "rapid_loss_stall", _pick_failure_op(op_names, ())
    if error_type == "zero_grad":
        return "residual_dominant_no_learning", _pick_failure_op(
            op_names, ("identity", "add", "linear_proj", "swiglu_mlp")
        )
    if error_type == "activation_collapse":
        return "degenerate_activations", _pick_failure_op(
            op_names, ("swiglu_mlp", "selective_scan", "token_class_proj")
        )
    if error_type == "inflight_grad_explosion":
        return "grad_explosion", _pick_failure_op(
            op_names,
            (
                "hybrid_sparse_router",
                "hybrid_token_gate",
                "gated_lane_blend",
                "depth_weighted_proj",
                "moe_topk",
                "moe_2expert",
                "relu_gated_moe",
            ),
        )
    if error_type in {
        "inflight_loss_spike",
        "inflight_divergence",
        "inflight_oscillation",
    }:
        return "loss_spike_instability", _pick_failure_op(op_names, ())
    if error_type == "inflight_no_progress":
        if any(op in op_names for op in _ROUTING_CHAIN_OPS):
            return "routing_dead_path", _pick_failure_op(op_names, _ROUTING_CHAIN_OPS)
        return "rapid_no_learning_signal", _pick_failure_op(
            op_names, ("swiglu_mlp", "linear_proj", "token_class_proj")
        )
    if error_type == "unstable_dynamics":
        return "unstable_init_norm_scaling", _pick_failure_op(
            op_names,
            (
                "selective_scan",
                "swiglu_mlp",
                "kronecker_linear",
                "ternary_projection",
                "linear_proj_up",
                "linear_proj_down",
            ),
        )
    if error_type in {"insufficient_learning", "failed_convergence"}:
        if "validation loss ratio" in msg and "failed to generalize" in msg:
            if any(op in op_names for op in _ROUTING_CHAIN_OPS):
                return "routing_dead_path", _pick_failure_op(
                    op_names, _ROUTING_CHAIN_OPS
                )
            if _graph_generic_sink_ratio(op_names) >= 0.6:
                return "residual_dominant_no_learning", _pick_failure_op(
                    op_names, ("swiglu_mlp", "linear_proj", "token_class_proj")
                )
            return "generalization_failure", _pick_failure_op(
                op_names,
                (
                    "selective_scan",
                    "swiglu_mlp",
                    "token_class_proj",
                    "token_type_classifier",
                ),
            )
        if any(op in op_names for op in _ROUTING_CHAIN_OPS):
            return "routing_dead_path", _pick_failure_op(op_names, _ROUTING_CHAIN_OPS)
        return "residual_dominant_no_learning", _pick_failure_op(
            op_names, ("swiglu_mlp", "linear_proj", "token_class_proj", "identity")
        )
    return error_type, _pick_failure_op(op_names, tuple(op_names))


def _infer_source_op(
    graph: Any,
    *,
    root_cause_code: str,
    failure_op: str | None,
    op_names: list[str],
) -> str | None:
    preferred = _ROOT_SOURCE_PREFERENCES.get(root_cause_code, ())
    picked = _pick_failure_op(op_names, preferred)
    if picked is not None:
        return picked
    native = _native_graph_provenance(graph, failure_op)
    if native is not None:
        _native_op_names, source_op = native
        if source_op is not None:
            return source_op
        return failure_op
    return failure_op


def infer_graph_failure_provenance(
    graph: Any,
    *,
    error_type: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    normalized_error_type = _normalize_error_type(error_type)
    op_names = _graph_op_names(graph)
    validation = validate_graph(graph)
    dim_flow = validate_dim_flow(graph)
    root_cause_code, failure_op = _classify_from_validation(
        validation.errors,
        dim_flow.errors,
        op_names,
    )
    if not root_cause_code:
        root_cause_code, failure_op = _classify_from_error(
            normalized_error_type,
            error_message or "",
            op_names,
        )
    source_op = _infer_source_op(
        graph,
        root_cause_code=root_cause_code,
        failure_op=failure_op,
        op_names=op_names,
    )
    return {
        "failure_op": failure_op,
        "failure_details_json": json.dumps(
            {
                "error_type": normalized_error_type,
                "error_message": (error_message or "")[:240],
                "failure_op": failure_op,
                "source_op": source_op,
                "root_cause_code": root_cause_code,
                "validator_errors": validation.errors[:8],
                "dim_flow_errors": dim_flow.errors[:8],
            }
        ),
    }
