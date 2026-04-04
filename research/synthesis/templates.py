"""Template Registry — maps names to template functions + weights.

Template implementations live in submodules:
  _templates_core.py     — workhorse templates (residual, transformer, etc.)
  _templates_routing.py  — routing-first templates (difficulty-gated, etc.)
  _templates_exotic.py   — binary-op safety, math-space, spiking templates
  _templates_attention.py — attention-heavy structural templates
  _templates_attention_tail.py — generated attention wrappers and tail templates
  _templates_research.py — 0% S1 fixes, zero-coverage ops, reference architectures
  _template_helpers.py   — shared helpers (motif picking, instantiation)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .graph import ComputationGraph

# Ops the grammar can emit through split template modules, alias substitution,
# or helper-inserted structural nodes even when the literal op name does not
# appear in this file's template registry body.
GRAMMAR_REACHABLE_OPS = frozenset(
    {
        "adaptive_lane_mixer",
        "adaptive_recursion",
        "add",
        "cascade",
        "causal_mask",
        "compression_mixture_experts",
        "concat",
        "cosine_similarity",
        "div_safe",
        "early_exit",
        "entropy_score",
        "gather_topk",
        "geometric_product",
        "hyp_distance",
        "matmul",
        "maximum",
        "minimum",
        "mixed_recursion_gate",
        "mod_topk",
        "mul",
        "n_way_sparse_router",
        "outer_product",
        "progressive_compression_gate",
        "relu_gate_routing",
        "route_lanes",
        "route_recursion",
        "route_topk",
        "routing_conditioned_compression",
        "softmax_last",
        "speculative",
        "split2",
        "split3",
        "sub",
        "token_merge",
        "token_type_classifier",
        "tropical_add",
    }
)

# Template families that intentionally do not require a 1:1 dedicated component
# graph. Many are variant wrappers around the same validated structure or
# reference architecture generators better covered by native hotpath and
# grammar-level tests.
COMPONENT_GRAPH_EXEMPT_TEMPLATE_PREFIXES = (
    "adaptive_",
    "attn_",
    "diff_attn_",
    "graph_attn_",
    "latent_attn_",
    "linear_attn_",
    "local_attn_",
)

COMPONENT_GRAPH_EXEMPT_TEMPLATES = frozenset(
    {
        "arch_router_block",
        "causal_mix_block",
        "chebyshev_block",
        "compute_budget_block",
        "conv_residual_block",
        "cross_dim_mixer",
        "cumulative_sequence",
        "depth_gated_block",
        "depth_token_mask_block",
        "diff_attention_block",
        "dual_attn_block",
        "dual_axis_block",
        "dual_routing_deep",
        "dual_routing_stack",
        "exp_gated_residual",
        "feature_sparse_block",
        "fused_gelu_ffn",
        "gpt2_reference",
        "hetero_moe_block",
        "hyperbolic_bridge_block",
        "integral_kernel_block",
        "iterative_refinement",
        "kronecker_block",
        "log_gated",
        "mamba_reference",
        "multi_head_mix_block",
        "n_way_moe_block",
        "poincare_add_bridge",
        "reciprocal_gated",
        "recurrent_delta_block",
        "reduce_attend",
        "rope_attention_block",
        "routing_conditioned_moe",
        "rwkv_block",
        "rwkv_double_norm",
        "rwkv_sparse_chain",
        "sign_ste_gated",
        "spiking_moe_block",
        "spiking_residual_block",
        "spiking_stdp_block",
        "sqrt_gated_ffn",
        "state_space_block",
        "token_merge_conv",
        "tropical_center_block",
        "ultrametric_attention_block",
        "windowed_attention",
    }
)


def is_component_graph_exempt_template(template_name: str) -> bool:
    return (
        template_name in COMPONENT_GRAPH_EXEMPT_TEMPLATES
        or template_name.startswith(COMPONENT_GRAPH_EXEMPT_TEMPLATE_PREFIXES)
    )


# Re-export public API used by grammar.py, tests, etc.
from ._template_helpers import (  # noqa: F401
    MotifWeights,
    TemplateFn,
    TemplateBuildError,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _fix_dim,
    _motif_is_compatible,
)

# ── Import all template functions from submodules ──────────────────

from ._templates_core import (  # noqa: F401
    tpl_residual_block,
    tpl_sequential,
    tpl_transformer_block,
    tpl_parallel_split,
    tpl_gated_maximum,
    tpl_three_way_split,
    tpl_bottleneck,
    tpl_moe,
    tpl_hybrid_parallel,
    tpl_gated_residual,
    tpl_dense_cascade,
    tpl_sparse_ffn,
    tpl_sparse_moe_block,
    tpl_routed_bottleneck,
    tpl_token_merge_block,
    tpl_token_merge_conv,
    tpl_conditional_compute,
)

from ._templates_routing import (  # noqa: F401
    ROUTING_TEMPLATES,
    tpl_difficulty_routed_block,
    tpl_three_lane_adaptive,
    tpl_cascaded_early_exit,
    tpl_recursive_depth_router,
    tpl_depth_token_mask_block,
    tpl_latent_compress_block,
    tpl_latent_compress_rwkv,
    tpl_signal_routed_compression,
    tpl_dual_routing_stack,
    tpl_dual_routing_deep,
    tpl_routing_conditioned_moe,
    tpl_mixed_recursion,
    tpl_topk_retrieval,
    tpl_depth_gated_block,
    tpl_feature_sparse_block,
    tpl_adaptive_sparse,
    tpl_adaptive_conv_ffn,
    tpl_adaptive_ssm_chain,
    tpl_adaptive_lane_recursion,
)

from ._templates_exotic import (  # noqa: F401
    tpl_normalized_matmul,
    tpl_gated_product,
    tpl_safe_division,
    tpl_cosine_scoring,
    tpl_decay_sequence,
    tpl_hyp_distance_scoring,
    tpl_tropical_residual,
    tpl_tropical_center_block,
    tpl_geometric_product_block,
    tpl_residual_difference,
    tpl_tropical_matmul_block,
    tpl_gated_minimum,
    tpl_spiking_residual_block,
    tpl_spiking_moe_block,
    tpl_hyperbolic_bridge_block,
    tpl_poincare_add_bridge,
    tpl_n_way_moe_block,
    tpl_conv_residual_block,
    tpl_causal_mix_block,
    tpl_iterative_refinement,
    tpl_recurrent_delta_block,
)

from ._templates_attention import (  # noqa: F401
    tpl_attn_residual_block,
    tpl_attn_gated_residual,
    tpl_attn_three_way_split,
    tpl_attn_dense_cascade,
    tpl_attn_conditional_compute,
    tpl_attn_cross_dim,
    tpl_attn_multi_head_mix,
    tpl_local_attn_ffn_block,
    tpl_local_attn_swiglu,
    tpl_diff_attn_gated_ffn,
    tpl_attn_ssm_hybrid,
    tpl_attn_conv_hybrid,
    tpl_attn_rwkv_hybrid,
    tpl_attn_bottleneck_hybrid,
    tpl_attn_routing_block,
    tpl_dual_attn_block,
    tpl_attn_state_space_hybrid,
    tpl_cascaded_attn_ffn,
    tpl_attn_exp_gated,
    tpl_attn_gated_product,
    tpl_diff_attn_routing,
    tpl_local_attn_routing,
    tpl_attn_chebyshev_hybrid,
    tpl_attn_sparse_moe,
)

from ._templates_attention_tail import (  # noqa: F401
    tpl_attn_decay_sequence,
    tpl_attn_dual_axis,
    tpl_attn_gated_maximum,
    tpl_attn_gated_minimum,
    tpl_attn_hyperbolic,
    tpl_attn_kronecker_hybrid,
    tpl_attn_log_gated,
    tpl_attn_moe_block,
    tpl_attn_normalized_matmul,
    tpl_attn_reciprocal_gated,
    tpl_attn_safe_division,
    tpl_attn_spiking_hybrid,
    tpl_attn_spectral_filter,
    tpl_diff_attn_conv_hybrid,
    tpl_diff_attn_ffn_block,
    tpl_diff_attn_moe,
    tpl_graph_attn_ffn_block,
    tpl_graph_attn_moe,
    tpl_graph_attn_sparse_ffn,
    tpl_latent_attn_conv_hybrid,
    tpl_latent_attn_ffn_block,
    tpl_latent_attn_moe,
    tpl_latent_attn_sparse_ffn,
    tpl_latent_attn_ssm_hybrid,
    tpl_linear_attn_ffn_block,
    tpl_linear_attn_sparse_ffn,
    tpl_local_attn_moe,
    tpl_local_attn_ssm_hybrid,
)

from ._templates_research import (  # noqa: F401
    tpl_state_space_block,
    tpl_cumulative_sequence,
    tpl_sqrt_gated_ffn,
    tpl_reduce_attend,
    tpl_fused_gelu_ffn,
    tpl_exp_gated_residual,
    tpl_integral_kernel_block,
    tpl_windowed_attention,
    tpl_local_attention_block,
    tpl_rwkv_block,
    tpl_rwkv_double_norm,
    tpl_rwkv_sparse_chain,
    tpl_reciprocal_gated,
    tpl_log_gated,
    tpl_sign_ste_gated,
    tpl_ultrametric_attention_block,
    tpl_diff_attention_block,
    tpl_graph_attention_block,
    tpl_chebyshev_block,
    tpl_kronecker_block,
    tpl_multi_head_mix_block,
    tpl_spiking_stdp_block,
    tpl_rope_attention_block,
    tpl_gpt2_reference,
    tpl_mamba_reference,
    tpl_hetero_moe_block,
    tpl_arch_router_block,
    tpl_compute_budget_block,
    tpl_cross_dim_mixer,
    tpl_dual_axis_block,
)


# ── Template Registry ───────────────────────────────────────────────

TEMPLATES: Dict[str, TemplateFn] = {
    "residual_block": tpl_residual_block,
    "sequential": tpl_sequential,
    "transformer_block": tpl_transformer_block,
    "parallel_split": tpl_parallel_split,
    "bottleneck": tpl_bottleneck,
    "moe": tpl_moe,
    "hybrid_parallel": tpl_hybrid_parallel,
    "gated_residual": tpl_gated_residual,
    "dense_cascade": tpl_dense_cascade,
    "sparse_ffn": tpl_sparse_ffn,
    "sparse_moe_block": tpl_sparse_moe_block,
    "routed_bottleneck": tpl_routed_bottleneck,
    "token_merge_block": tpl_token_merge_block,
    "token_merge_conv": tpl_token_merge_conv,
    "conditional_compute": tpl_conditional_compute,
    "difficulty_routed_block": tpl_difficulty_routed_block,
    "three_lane_adaptive": tpl_three_lane_adaptive,
    "cascaded_early_exit": tpl_cascaded_early_exit,
    "recursive_depth_router": tpl_recursive_depth_router,
    "depth_token_mask_block": tpl_depth_token_mask_block,
    "latent_compress_block": tpl_latent_compress_block,
    "latent_compress_rwkv": tpl_latent_compress_rwkv,
    "signal_routed_compression": tpl_signal_routed_compression,
    "dual_routing_stack": tpl_dual_routing_stack,
    "dual_routing_deep": tpl_dual_routing_deep,
    "routing_conditioned_moe": tpl_routing_conditioned_moe,
    "mixed_recursion": tpl_mixed_recursion,
    "topk_retrieval": tpl_topk_retrieval,
    "depth_gated_block": tpl_depth_gated_block,
    "feature_sparse_block": tpl_feature_sparse_block,
    "normalized_matmul": tpl_normalized_matmul,
    "gated_product": tpl_gated_product,
    "safe_division": tpl_safe_division,
    "cosine_scoring": tpl_cosine_scoring,
    "decay_sequence": tpl_decay_sequence,
    "residual_difference": tpl_residual_difference,
    "gated_minimum": tpl_gated_minimum,
    "hyp_distance_scoring": tpl_hyp_distance_scoring,
    "tropical_residual": tpl_tropical_residual,
    "tropical_matmul_block": tpl_tropical_matmul_block,
    "geometric_product_block": tpl_geometric_product_block,
    "gated_maximum": tpl_gated_maximum,
    "three_way_split": tpl_three_way_split,
    "cumulative_sequence": tpl_cumulative_sequence,
    "sqrt_gated_ffn": tpl_sqrt_gated_ffn,
    "reduce_attend": tpl_reduce_attend,
    "fused_gelu_ffn": tpl_fused_gelu_ffn,
    "exp_gated_residual": tpl_exp_gated_residual,
    "integral_kernel_block": tpl_integral_kernel_block,
    "windowed_attention": tpl_windowed_attention,
    "local_attention_block": tpl_local_attention_block,
    "state_space_block": tpl_state_space_block,
    "rwkv_block": tpl_rwkv_block,
    "rwkv_double_norm": tpl_rwkv_double_norm,
    "rwkv_sparse_chain": tpl_rwkv_sparse_chain,
    "reciprocal_gated": tpl_reciprocal_gated,
    "ultrametric_attention_block": tpl_ultrametric_attention_block,
    "diff_attention_block": tpl_diff_attention_block,
    "graph_attention_block": tpl_graph_attention_block,
    "spiking_residual_block": tpl_spiking_residual_block,
    "spiking_moe_block": tpl_spiking_moe_block,
    "hyperbolic_bridge_block": tpl_hyperbolic_bridge_block,
    "poincare_add_bridge": tpl_poincare_add_bridge,
    "n_way_moe_block": tpl_n_way_moe_block,
    "conv_residual_block": tpl_conv_residual_block,
    "causal_mix_block": tpl_causal_mix_block,
    "iterative_refinement": tpl_iterative_refinement,
    "recurrent_delta_block": tpl_recurrent_delta_block,
    "sign_ste_gated": tpl_sign_ste_gated,
    "log_gated": tpl_log_gated,
    "tropical_center_block": tpl_tropical_center_block,
    "multi_head_mix_block": tpl_multi_head_mix_block,
    "chebyshev_block": tpl_chebyshev_block,
    "kronecker_block": tpl_kronecker_block,
    "spiking_stdp_block": tpl_spiking_stdp_block,
    "rope_attention_block": tpl_rope_attention_block,
    "gpt2_reference": tpl_gpt2_reference,
    "mamba_reference": tpl_mamba_reference,
    "hetero_moe_block": tpl_hetero_moe_block,
    "arch_router_block": tpl_arch_router_block,
    "compute_budget_block": tpl_compute_budget_block,
    "adaptive_sparse": tpl_adaptive_sparse,
    "adaptive_conv_ffn": tpl_adaptive_conv_ffn,
    "adaptive_ssm_chain": tpl_adaptive_ssm_chain,
    "adaptive_lane_recursion": tpl_adaptive_lane_recursion,
    "cross_dim_mixer": tpl_cross_dim_mixer,
    "dual_axis_block": tpl_dual_axis_block,
    # ── Attention templates (Groups A-D) ──────────────────────────────
    "attn_residual_block": tpl_attn_residual_block,
    "attn_gated_residual": tpl_attn_gated_residual,
    "attn_three_way_split": tpl_attn_three_way_split,
    "attn_conditional_compute": tpl_attn_conditional_compute,
    "attn_cross_dim": tpl_attn_cross_dim,
    "attn_multi_head_mix": tpl_attn_multi_head_mix,
    "latent_attn_ffn_block": tpl_latent_attn_ffn_block,
    "local_attn_ffn_block": tpl_local_attn_ffn_block,
    "diff_attn_ffn_block": tpl_diff_attn_ffn_block,
    "linear_attn_ffn_block": tpl_linear_attn_ffn_block,
    "latent_attn_sparse_ffn": tpl_latent_attn_sparse_ffn,
    "local_attn_swiglu": tpl_local_attn_swiglu,
    "diff_attn_gated_ffn": tpl_diff_attn_gated_ffn,
    "graph_attn_ffn_block": tpl_graph_attn_ffn_block,
    "attn_ssm_hybrid": tpl_attn_ssm_hybrid,
    "attn_conv_hybrid": tpl_attn_conv_hybrid,
    "attn_rwkv_hybrid": tpl_attn_rwkv_hybrid,
    "attn_bottleneck_hybrid": tpl_attn_bottleneck_hybrid,
    "attn_routing_block": tpl_attn_routing_block,
    "dual_attn_block": tpl_dual_attn_block,
    "attn_state_space_hybrid": tpl_attn_state_space_hybrid,
    "cascaded_attn_ffn": tpl_cascaded_attn_ffn,
    "attn_exp_gated": tpl_attn_exp_gated,
    "attn_reciprocal_gated": tpl_attn_reciprocal_gated,
    "attn_decay_sequence": tpl_attn_decay_sequence,
    "attn_gated_product": tpl_attn_gated_product,
    "diff_attn_routing": tpl_diff_attn_routing,
    "local_attn_routing": tpl_local_attn_routing,
    "attn_chebyshev_hybrid": tpl_attn_chebyshev_hybrid,
    "attn_kronecker_hybrid": tpl_attn_kronecker_hybrid,
    "attn_sparse_moe": tpl_attn_sparse_moe,
    "attn_log_gated": tpl_attn_log_gated,
    # Group E: additional to reach 60%
    "attn_gated_maximum": tpl_attn_gated_maximum,
    "attn_hyperbolic": tpl_attn_hyperbolic,
    "attn_spectral_filter": tpl_attn_spectral_filter,
    "attn_normalized_matmul": tpl_attn_normalized_matmul,
    "latent_attn_conv_hybrid": tpl_latent_attn_conv_hybrid,
    "diff_attn_conv_hybrid": tpl_diff_attn_conv_hybrid,
    # Group F
    "attn_safe_division": tpl_attn_safe_division,
    "latent_attn_ssm_hybrid": tpl_latent_attn_ssm_hybrid,
    "local_attn_ssm_hybrid": tpl_local_attn_ssm_hybrid,
    "attn_spiking_hybrid": tpl_attn_spiking_hybrid,
    "latent_attn_moe": tpl_latent_attn_moe,
    "local_attn_moe": tpl_local_attn_moe,
    "diff_attn_moe": tpl_diff_attn_moe,
    "graph_attn_moe": tpl_graph_attn_moe,
    "linear_attn_sparse_ffn": tpl_linear_attn_sparse_ffn,
    "graph_attn_sparse_ffn": tpl_graph_attn_sparse_ffn,
}

DEFAULT_TEMPLATE_WEIGHTS: Dict[str, float] = {
    "residual_block": 3.0,
    "transformer_block": 3.0,
    "sequential": 2.0,
    "parallel_split": 1.5,
    "bottleneck": 1.5,
    "moe": 2.0,
    "hybrid_parallel": 1.0,
    "gated_residual": 1.5,
    "dense_cascade": 0.8,
    "sparse_ffn": 2.0,
    "sparse_moe_block": 4.0,
    "routed_bottleneck": 4.0,
    "token_merge_block": 7.0,
    "token_merge_conv": 6.0,
    "conditional_compute": 3.5,
    "difficulty_routed_block": 5.0,
    "three_lane_adaptive": 5.0,
    "cascaded_early_exit": 4.5,
    "recursive_depth_router": 6.0,
    "depth_token_mask_block": 4.5,
    # Reduced from 6.0: ops_mining_report shows mean_loss=0.804 inflated by
    # forced_exploration (48% of samples). Organic mean is 0.593 — decent but
    # not worth 12% of sampling budget. The op itself is sound (test_lac.py passes).
    "latent_compress_block": 0.5,
    "latent_compress_rwkv": 0.5,
    "signal_routed_compression": 5.0,
    "dual_routing_stack": 6.0,
    "dual_routing_deep": 5.0,
    "routing_conditioned_moe": 4.5,
    "mixed_recursion": 5.5,
    "topk_retrieval": 3.5,
    "normalized_matmul": 2.0,
    "gated_product": 2.0,
    "safe_division": 1.5,
    "cosine_scoring": 2.0,
    "decay_sequence": 3.0,
    "residual_difference": 2.5,
    "gated_minimum": 2.5,
    "hyp_distance_scoring": 1.5,
    "tropical_residual": 2.5,
    "tropical_matmul_block": 2.5,
    "geometric_product_block": 1.5,
    "gated_maximum": 1.5,
    "three_way_split": 2.5,
    "cumulative_sequence": 2.5,
    "sqrt_gated_ffn": 2.5,
    "reduce_attend": 3.0,
    "fused_gelu_ffn": 3.0,
    "exp_gated_residual": 2.5,
    "integral_kernel_block": 3.0,
    "windowed_attention": 3.0,
    "local_attention_block": 3.0,
    "state_space_block": 3.5,
    "rwkv_block": 1.0,
    "rwkv_double_norm": 5.0,
    "rwkv_sparse_chain": 4.0,
    "reciprocal_gated": 2.5,
    "sign_ste_gated": 2.5,
    "log_gated": 2.5,
    "tropical_center_block": 2.5,
    "ultrametric_attention_block": 3.5,
    "diff_attention_block": 3.5,
    "graph_attention_block": 3.5,
    "spiking_residual_block": 3.0,
    "spiking_moe_block": 4.0,
    "hyperbolic_bridge_block": 3.0,
    "poincare_add_bridge": 3.0,
    "n_way_moe_block": 3.5,
    "conv_residual_block": 3.0,
    "causal_mix_block": 2.5,
    "iterative_refinement": 2.5,
    "recurrent_delta_block": 3.5,
    "multi_head_mix_block": 4.0,
    "chebyshev_block": 3.0,
    "kronecker_block": 3.0,
    "spiking_stdp_block": 3.0,
    "rope_attention_block": 1.0,  # Demoted: 4.9% S1 rate
    "gpt2_reference": 3.0,
    "mamba_reference": 3.0,
    "hetero_moe_block": 4.0,
    "arch_router_block": 4.0,
    "compute_budget_block": 4.0,
    "depth_gated_block": 2.0,
    "feature_sparse_block": 2.0,
    "adaptive_sparse": 6.0,
    "adaptive_conv_ffn": 5.5,
    "adaptive_ssm_chain": 5.0,
    "adaptive_lane_recursion": 4.5,
    "cross_dim_mixer": 3.5,
    "dual_axis_block": 3.0,
    # ── Attention templates (Groups A-D) ──────────────────────────────
    # Group A: forced-attention variants of existing templates
    "attn_residual_block": 3.5,
    "attn_gated_residual": 3.5,
    "attn_three_way_split": 4.0,  # Parent is 86.4% S1
    "attn_conditional_compute": 3.5,
    "attn_cross_dim": 3.0,
    "attn_multi_head_mix": 3.5,
    # Group B: attention+FFN with specific ops
    "latent_attn_ffn_block": 4.0,  # Best attn op (30.2% S1)
    "local_attn_ffn_block": 4.0,  # 27.5% S1
    "diff_attn_ffn_block": 3.5,  # 21.2% S1
    "linear_attn_ffn_block": 2.5,  # 11.3% S1 — weaker
    "latent_attn_sparse_ffn": 4.0,
    "local_attn_swiglu": 4.0,
    "diff_attn_gated_ffn": 3.5,
    "graph_attn_ffn_block": 3.5,
    # Group C: hybrid attention+X
    "attn_ssm_hybrid": 3.5,
    "attn_conv_hybrid": 3.5,
    "attn_rwkv_hybrid": 3.5,
    "attn_bottleneck_hybrid": 3.0,
    "attn_routing_block": 3.0,
    "dual_attn_block": 3.0,
    "attn_state_space_hybrid": 3.5,
    "cascaded_attn_ffn": 3.0,
    # Group D: attention + exotic ops
    "attn_exp_gated": 3.0,
    "attn_reciprocal_gated": 3.0,
    "attn_decay_sequence": 3.0,
    "attn_gated_product": 3.0,
    "diff_attn_routing": 3.5,
    "local_attn_routing": 3.5,
    "attn_chebyshev_hybrid": 3.0,
    "attn_kronecker_hybrid": 3.0,
    "attn_sparse_moe": 3.5,
    "attn_log_gated": 3.0,
    # Group E: additional to reach 60%
    "attn_gated_maximum": 3.0,
    "attn_hyperbolic": 3.0,
    "attn_spectral_filter": 3.5,  # spectral_filter has best S1 rate (0.329)
    "attn_normalized_matmul": 3.0,
    "latent_attn_conv_hybrid": 4.0,  # Best attn + conv parallel
    "diff_attn_conv_hybrid": 3.5,
    # Group F: final templates for 60% threshold
    "attn_safe_division": 3.0,
    "latent_attn_ssm_hybrid": 4.0,
    "local_attn_ssm_hybrid": 3.5,
    "attn_spiking_hybrid": 3.0,
    "latent_attn_moe": 4.0,
    "local_attn_moe": 3.5,
    "diff_attn_moe": 3.5,
    "graph_attn_moe": 3.5,
    "linear_attn_sparse_ffn": 2.5,
    "graph_attn_sparse_ffn": 3.0,
}


def pick_template(
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, TemplateFn]:
    """Pick a template weighted by success priors."""
    names = list(TEMPLATES.keys())
    template_weights = [
        (weights or {}).get(n, DEFAULT_TEMPLATE_WEIGHTS.get(n, 1.0)) for n in names
    ]
    name = rng.choices(names, weights=template_weights, k=1)[0]
    return name, TEMPLATES[name]


def apply_template(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    template_name: Optional[str] = None,
    template_weights: Optional[Dict[str, float]] = None,
    motif_weights: MotifWeights = None,
    op_weights: Optional[Dict[str, float]] = None,
) -> int:
    """Apply a template to the graph. Main entry point for grammar."""
    prev_node_ids = set(graph.nodes.keys())
    prev_next_id = graph._next_id
    prev_output_id = graph._output_node_id
    prev_ir_version = graph._ir_version
    prev_metadata = dict(graph.metadata)

    if template_name and template_name in TEMPLATES:
        name = template_name
        fn = TEMPLATES[name]
    else:
        name, fn = pick_template(rng, template_weights)
    if op_weights:
        graph.metadata["_op_weights"] = op_weights
    graph.metadata.setdefault("templates_used", []).append(name)
    prev_template = graph.metadata.get("_active_template")
    prev_slot_counter = graph.metadata.get("_active_template_slot_counter")
    prev_template_instance = graph.metadata.get("_active_template_instance")
    graph.metadata["_active_template"] = name
    graph.metadata["_active_template_slot_counter"] = 0
    graph.metadata["_active_template_instance"] = (
        len(graph.metadata.get("templates_used", [])) - 1
    )
    try:
        return fn(graph, input_id, rng, motif_weights)
    except Exception:
        added_ids = set(graph.nodes.keys()) - prev_node_ids
        for nid in added_ids:
            del graph.nodes[nid]
        graph._next_id = prev_next_id
        graph._output_node_id = prev_output_id
        graph._ir_version = prev_ir_version
        graph.metadata = prev_metadata
        graph._cache.clear()
        raise
    finally:
        if prev_template is None:
            graph.metadata.pop("_active_template", None)
        else:
            graph.metadata["_active_template"] = prev_template
        if prev_slot_counter is None:
            graph.metadata.pop("_active_template_slot_counter", None)
        else:
            graph.metadata["_active_template_slot_counter"] = prev_slot_counter
        if prev_template_instance is None:
            graph.metadata.pop("_active_template_instance", None)
        else:
            graph.metadata["_active_template_instance"] = prev_template_instance
