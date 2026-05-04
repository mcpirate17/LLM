"""Structured routing template manifest.

This is the native-ready registration surface for routing-family templates.
`templates.py` consumes these tables directly instead of repeating routing
template names across import lists, registry entries, and default-weight maps.
"""

from __future__ import annotations

from ._templates_routing import (
    tpl_adaptive_conv_ffn,
    tpl_adaptive_lane_recursion,
    tpl_adaptive_sparse,
    tpl_adaptive_ssm_chain,
    tpl_cascaded_early_exit,
    tpl_depth_gated_block,
    tpl_depth_gated_block_matmul_norm,
    tpl_depth_gated_block_matmul_stable,
    tpl_depth_token_mask_block,
    tpl_difficulty_routed_block,
    tpl_dual_routing_deep,
    tpl_dual_routing_stack,
    tpl_feature_sparse_block,
    tpl_gated_lane_blend_block,
    tpl_hybrid_sparse_triplet_router,
    tpl_intelligent_multilane_router,
    tpl_latent_compress_block,
    tpl_latent_compress_rwkv,
    tpl_mixed_recursion,
    tpl_multiscale_difficulty_router,
    tpl_multiscale_difficulty_router_adaptive_attn_ssm,
    tpl_multiscale_rich_lane_router,
    tpl_recursive_depth_router,
    tpl_routing_conditioned_moe,
    tpl_signal_routed_compression,
    tpl_three_lane_adaptive,
    tpl_topk_retrieval,
)


ROUTING_TEMPLATE_REGISTRY = {
    "difficulty_routed_block": tpl_difficulty_routed_block,
    "three_lane_adaptive": tpl_three_lane_adaptive,
    "cascaded_early_exit": tpl_cascaded_early_exit,
    "hybrid_sparse_triplet_router": tpl_hybrid_sparse_triplet_router,
    "intelligent_multilane_router": tpl_intelligent_multilane_router,
    "multiscale_difficulty_router": tpl_multiscale_difficulty_router,
    "multiscale_difficulty_router_adaptive_attn_ssm": tpl_multiscale_difficulty_router_adaptive_attn_ssm,
    "multiscale_rich_lane_router": tpl_multiscale_rich_lane_router,
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
    "depth_gated_block_matmul_norm": tpl_depth_gated_block_matmul_norm,
    "depth_gated_block_matmul_stable": tpl_depth_gated_block_matmul_stable,
    "gated_lane_blend_block": tpl_gated_lane_blend_block,
    "feature_sparse_block": tpl_feature_sparse_block,
    "adaptive_sparse": tpl_adaptive_sparse,
    "adaptive_conv_ffn": tpl_adaptive_conv_ffn,
    "adaptive_ssm_chain": tpl_adaptive_ssm_chain,
    "adaptive_lane_recursion": tpl_adaptive_lane_recursion,
}


ROUTING_TEMPLATE_DEFAULT_WEIGHTS = {
    "difficulty_routed_block": 0.5,
    "three_lane_adaptive": 0.5,
    "cascaded_early_exit": 4.5,
    "hybrid_sparse_triplet_router": 0.5,
    "intelligent_multilane_router": 5.5,
    "multiscale_difficulty_router": 5.5,
    "multiscale_difficulty_router_adaptive_attn_ssm": 0.25,
    "multiscale_rich_lane_router": 5.0,
    "recursive_depth_router": 6.0,
    "depth_token_mask_block": 4.5,
    # Reduced from 6.0: ops_mining_report shows mean_loss=0.804 inflated by
    # forced_exploration (48% of samples). Organic mean is 0.593 — decent but
    # not worth 12% of sampling budget. The op itself is sound.
    "latent_compress_block": 0.5,
    "latent_compress_rwkv": 0.5,
    "signal_routed_compression": 0.5,
    "dual_routing_stack": 6.0,
    "dual_routing_deep": 0.5,
    "routing_conditioned_moe": 4.5,
    "mixed_recursion": 5.5,
    "topk_retrieval": 3.5,
    "depth_gated_block": 2.0,
    "depth_gated_block_matmul_norm": 1.5,
    # Rewritten: added FFN sub-block after depth-gated matmul
    "depth_gated_block_matmul_stable": 3.0,
    "gated_lane_blend_block": 2.0,
    "feature_sparse_block": 2.0,
    "adaptive_sparse": 6.0,
    "adaptive_conv_ffn": 8.0,
    "adaptive_ssm_chain": 5.0,
    "adaptive_lane_recursion": 4.5,
}
