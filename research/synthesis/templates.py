"""Template Registry — maps names to template functions + weights.

Template implementations live in submodules:
  _templates_core.py     — workhorse templates (residual, transformer, etc.)
  _templates_routing.py  — routing-first templates (difficulty-gated, etc.)
  _templates_exotic.py   — binary-op safety, math-space, spiking templates
  _templates_research.py — 0% S1 fixes, zero-coverage ops, reference architectures
  _template_helpers.py   — shared helpers (motif picking, instantiation)
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from .graph import ComputationGraph

# Re-export public API used by grammar.py, tests, etc.
from ._template_helpers import (  # noqa: F401
    MotifWeights,
    TemplateFn,
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
    tpl_n_way_moe_block,
    tpl_conv_residual_block,
    tpl_causal_mix_block,
    tpl_iterative_refinement,
    tpl_recurrent_delta_block,
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
    "n_way_moe_block": 3.5,
    "conv_residual_block": 3.0,
    "causal_mix_block": 2.5,
    "iterative_refinement": 2.5,
    "recurrent_delta_block": 3.5,
    "multi_head_mix_block": 4.0,
    "chebyshev_block": 3.0,
    "kronecker_block": 3.0,
    "spiking_stdp_block": 3.0,
    "rope_attention_block": 3.5,
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
    graph.metadata["_active_template_instance"] = len(
        graph.metadata.get("templates_used", [])
    ) - 1
    try:
        return fn(graph, input_id, rng, motif_weights)
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
