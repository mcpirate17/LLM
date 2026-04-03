"""Context rules — motif template allowlists and motif query helpers."""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional

from .motifs import MOTIFS_BY_CLASS, Motif

from ._context_types import (
    CONTEXT_CLASS_GENERAL,
    CONTEXT_CLASS_REHAB,
    CONTEXT_CLASS_RESTRICTED,
    CONTEXT_CLASS_STRUCTURAL,
)


_MOTIF_TEMPLATE_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "attn_causal_mask": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            # Attention templates that use attention motifs
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
        }
    ),
    "attn_local_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "local_attention_block",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
            "local_attn_ffn_block",
            "local_attn_swiglu",
            "local_attn_routing",
            "local_attn_moe",
            "local_attn_ssm_hybrid",
        }
    ),
    "attn_sliding_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "windowed_attention",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
        }
    ),
    "attn_graph": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "graph_attention_block",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
            "graph_attn_ffn_block",
            "graph_attn_moe",
            "graph_attn_sparse_ffn",
        }
    ),
    "ssm_state_space": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "state_space_block",
        }
    ),
    "mix_integral_kernel": frozenset(
        {
            "residual_block",
            "gated_residual",
            "integral_kernel_block",
        }
    ),
    "ffn_fused_gelu": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "fused_gelu_ffn",
        }
    ),
    "reduce_sum": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_mean": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_max": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_norm": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "route_identity": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "route_early_exit": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "spiking_lif_rate": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_threshold_stdp": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_rate_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_threshold_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "tropical_center_norm": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "tropical_center_block",
        }
    ),
    "hyperbolic_residual_bridge": frozenset(
        {"residual_block", "hyperbolic_bridge_block"}
    ),
    "poincare_add_bridge": frozenset({"residual_block", "hyperbolic_bridge_block"}),
    "conv_only_block": frozenset(
        {"residual_block", "gated_residual", "conv_residual_block"}
    ),
    "attn_gated_delta": frozenset(
        {
            "residual_block",
            "transformer_block",
            "recurrent_delta_block",
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "dual_attn_block",
            "cascaded_attn_ffn",
        }
    ),
    "mix_fixed_point": frozenset({"residual_block", "iterative_refinement"}),
    "n_way_routing": frozenset({"moe", "n_way_moe_block"}),
    "proj_bottleneck": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "proj_low_rank": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "route_adaptive_recursion": frozenset(
        {"recursive_depth_router", "difficulty_routed_block", "conditional_compute"}
    ),
}

_CONTEXT_CLASS_PRIORS = {
    CONTEXT_CLASS_GENERAL: 1.0,
    CONTEXT_CLASS_RESTRICTED: 0.55,
    CONTEXT_CLASS_STRUCTURAL: 0.30,
    CONTEXT_CLASS_REHAB: 0.15,
}

# Import here (not at top) to avoid circular dependency — _OP_CONTEXT_CLASS
# is defined in _context_registry which also imports from _context_types.
from ._context_registry import _OP_CONTEXT_CLASS


def get_op_context_class(op_name: str) -> str:
    return _OP_CONTEXT_CLASS.get(op_name, CONTEXT_CLASS_GENERAL)


def _motif_context_class(motif: Motif) -> str:
    classes = {get_op_context_class(step.op_name) for step in motif.steps}
    if CONTEXT_CLASS_REHAB in classes:
        return CONTEXT_CLASS_REHAB
    if CONTEXT_CLASS_STRUCTURAL in classes:
        return CONTEXT_CLASS_STRUCTURAL
    if CONTEXT_CLASS_RESTRICTED in classes:
        return CONTEXT_CLASS_RESTRICTED
    return CONTEXT_CLASS_GENERAL


def apply_context_rule_priors(
    motif_weights: Optional[Dict[str, float]],
    exploration_targets: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, float]]:
    weights = dict(motif_weights) if motif_weights else {}
    targeted_ops = set(exploration_targets or ())
    seen_motifs: set[str] = set()
    for motifs in MOTIFS_BY_CLASS.values():
        for motif in motifs:
            if motif.name in seen_motifs:
                continue
            seen_motifs.add(motif.name)
            motif_ops = {step.op_name for step in motif.steps}
            if motif_ops & targeted_ops:
                continue
            factor = _CONTEXT_CLASS_PRIORS[_motif_context_class(motif)]
            if motif.name == "attn_local_window":
                factor *= 0.7
            base = weights.get(motif.name, motif.lift)
            weights[motif.name] = max(0.05, base * factor)
    return weights or None


def motif_allowed_in_template(motif: Motif, template_name: Optional[str]) -> bool:
    if template_name is None:
        return True
    allowed_templates = _MOTIF_TEMPLATE_ALLOWLIST.get(motif.name)
    if allowed_templates is None:
        return True
    return template_name in allowed_templates
