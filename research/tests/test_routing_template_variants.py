from research.tools.routing_template_variants import (
    build_observable_three_lane_router,
    build_hybrid_sparse_triplet_variant,
    build_intelligent_multilane_variant,
    build_locked_multiscale_variant,
    build_multiscale_difficulty_variant,
    build_recursive_depth_variant,
)


def test_locked_multiscale_variant_uses_calibrated_merge():
    graph = build_locked_multiscale_variant(model_dim=64, span_widths=(2, 3))
    ops = [node.op_name for node in graph.nodes.values()]
    assert ops.count("calibrated_branch_merge") == 4


def test_multiscale_difficulty_variant_has_expected_hard_path():
    graph = build_multiscale_difficulty_variant(model_dim=64)
    ops = [node.op_name for node in graph.nodes.values()]
    assert "moe_topk" in ops
    assert ops.count("hybrid_sparse_router") == 3


def test_intelligent_multilane_variant_accepts_locked_ops():
    graph = build_intelligent_multilane_variant(
        model_dim=64,
        easy_op="conv_only",
        medium_op="adaptive_lane_mixer",
        hard_op="moe_topk",
    )
    ops = [node.op_name for node in graph.nodes.values()]
    assert "conv_only" in ops
    assert "adaptive_lane_mixer" in ops
    assert "moe_topk" in ops


def test_observable_three_lane_router_is_simple_and_explicit():
    graph = build_observable_three_lane_router(model_dim=64)
    ops = [node.op_name for node in graph.nodes.values()]
    assert "rmsnorm" in ops
    assert "token_class_proj" in ops
    assert "cheap_verify_blend" in ops
    assert "block_sparse_linear" in ops
    assert "moe_topk" in ops
    assert ops.count("calibrated_branch_merge") == 2
    assert ops.count("signal_conditioned_compression") == 1


def test_observable_three_lane_router_accepts_simple_lane_overrides():
    graph = build_observable_three_lane_router(
        model_dim=64,
        medium_op="conv_only",
        hard_op="dual_compression_blend",
    )
    ops = [node.op_name for node in graph.nodes.values()]
    assert "conv_only" in ops
    assert "dual_compression_blend" in ops
    assert "signal_conditioned_compression" in ops
    assert ops.count("calibrated_branch_merge") == 2


def test_other_routing_variants_build():
    triplet = build_hybrid_sparse_triplet_variant(model_dim=64)
    depth = build_recursive_depth_variant(
        model_dim=64, max_depth=3, post_op="conv_only"
    )
    three_lane = build_observable_three_lane_router(model_dim=64)
    assert triplet._output_node_id is not None
    assert depth._output_node_id is not None
    assert three_lane._output_node_id is not None
