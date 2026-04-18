from __future__ import annotations

from typing import Any

from research.synthesis.graph import ComputationGraph
from research.tools.audit_multiscale_rich_lane_router import build_multiscale_variant


def build_locked_multiscale_variant(
    *,
    model_dim: int,
    span_widths: tuple[int, ...],
) -> ComputationGraph:
    return build_multiscale_variant(
        model_dim=model_dim,
        span_widths=span_widths,
        medium_op="conv_only",
        hard_op="mixed_recursion_gate",
        route_temperature=0.85,
        min_keep_fraction=0.125,
        confidence_threshold=0.55,
        enable_curriculum=False,
        use_calibrated_merge=True,
    )


def build_observable_three_lane_router(
    *,
    model_dim: int,
    medium_op: str = "block_sparse_linear",
    hard_op: str = "moe_topk",
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    stem = graph.add_op("rmsnorm", [inp], {})
    difficulty = graph.add_op("token_class_proj", [stem], {"n_classes": 3})
    routed_seed = graph.add_op(
        "signal_conditioned_compression",
        [stem, difficulty],
        {},
    )
    easy = graph.add_op("cheap_verify_blend", [stem], {})
    medium = graph.add_op(medium_op, [routed_seed], _three_lane_config(medium_op))
    if medium_op in {"conv_only", "block_sparse_linear"}:
        medium = graph.add_op("linear_proj", [medium], {"out_dim": model_dim})
    hard = graph.add_op(
        hard_op,
        _three_lane_hard_inputs(
            hard_op,
            routed_seed=routed_seed,
            difficulty=difficulty,
        ),
        _three_lane_config(hard_op),
    )
    if hard_op in {"moe_topk", "moe_2expert", "dual_compression_blend"}:
        hard = graph.add_op("linear_proj", [hard], {"out_dim": model_dim})
    merged = graph.add_op(
        "calibrated_branch_merge",
        [easy, medium],
        {
            "n_branches": 2,
            "normalize_inputs": True,
            "merge_temperature": 0.9,
            "primary_role": "easy",
            "secondary_role": "medium",
            "min_secondary_share": 0.12,
            "max_secondary_share": 0.36,
        },
    )
    merged = graph.add_op(
        "calibrated_branch_merge",
        [merged, hard],
        {
            "n_branches": 2,
            "normalize_inputs": True,
            "merge_temperature": 0.85,
            "primary_role": "easy_medium",
            "secondary_role": "hard",
            "min_secondary_share": 0.08,
            "max_secondary_share": 0.24,
        },
    )
    out = graph.add_op("add", [inp, merged], {})
    out = graph.add_op("rmsnorm", [out], {})
    graph.set_output(out)
    return graph


def build_hybrid_sparse_triplet_variant(*, model_dim: int) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    default_path = graph.add_op("default_path", [inp], {})
    gated = graph.add_op("hybrid_token_gate", [inp], {"threshold": 0.5})
    gated_skip = graph.add_op("add", [inp, gated], {})
    graph.add_op(
        "sparse_span_builder",
        [gated],
        {"span_width": 3, "fallback_behavior": "default_path"},
    )
    routed = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {
            "span_width": 3,
            "lane_count": 3,
            "confidence_threshold": 0.45,
            "min_keep_fraction": 0.125,
            "route_temperature": 0.85,
        },
    )
    lane = graph.add_op("lane_conditioned_block", [routed], {"lane_id": 1})
    merged = graph.add_op("add", [default_path, lane], {})
    fused = graph.add_op("add", [gated_skip, merged], {})
    out = graph.add_op("rmsnorm", [fused], {})
    graph.set_output(out)
    return graph


def build_multiscale_difficulty_variant(*, model_dim: int) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    default_path = graph.add_op("default_path", [inp], {})
    gated = graph.add_op("hybrid_token_gate", [inp], {"threshold": 0.5})
    gated_skip = graph.add_op("add", [inp, gated], {})

    for width in (2, 3, 4):
        graph.add_op(
            "sparse_span_builder",
            [gated],
            {"span_width": width, "fallback_behavior": "default_path"},
        )

    pair = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 2, "lane_count": 2, "confidence_threshold": 0.55},
    )
    pair = graph.add_op("lane_conditioned_block", [pair], {"lane_id": 0})
    triplet = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.55},
    )
    triplet = graph.add_op("lane_conditioned_block", [triplet], {"lane_id": 1})
    quartet = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 4, "lane_count": 4, "confidence_threshold": 0.55},
    )
    quartet = graph.add_op("lane_conditioned_block", [quartet], {"lane_id": 2})

    medium = graph.add_op("add", [pair, triplet], {})
    medium = graph.add_op("add", [medium, quartet], {})

    hard_signal = graph.add_op("token_class_proj", [gated], {"n_classes": 4})
    hard_seed = graph.add_op("signal_conditioned_compression", [gated, hard_signal], {})
    hard = graph.add_op("moe_topk", [hard_seed], {"num_experts": 4, "top_k": 1})
    hard = graph.add_op("linear_proj", [hard], {"out_dim": model_dim})

    out = graph.add_op("add", [default_path, medium], {})
    out = graph.add_op("add", [out, hard], {})
    out = graph.add_op("add", [gated_skip, out], {})
    out = graph.add_op("add", [inp, out], {})
    graph.set_output(out)
    return graph


def build_intelligent_multilane_variant(
    *,
    model_dim: int,
    easy_op: str,
    medium_op: str,
    hard_op: str,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    stem = inp
    gated = graph.add_op("hybrid_token_gate", [stem], {"threshold": 0.5})
    gated_skip = graph.add_op("add", [stem, gated], {})

    easy_input = stem
    if easy_op == "conv1d_seq":
        easy_input = graph.add_op("rmsnorm", [stem], {})
    easy_lane = graph.add_op(easy_op, [easy_input], {})
    if easy_op == "conv_only":
        easy_lane = graph.add_op("linear_proj", [easy_lane], {"out_dim": model_dim})

    pair = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 2, "lane_count": 2, "confidence_threshold": 0.55},
    )
    triplet = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.55},
    )
    quartet = graph.add_op(
        "hybrid_sparse_router",
        [gated],
        {"span_width": 4, "lane_count": 4, "confidence_threshold": 0.55},
    )
    routed = graph.add_op("add", [pair, triplet], {})
    routed = graph.add_op("add", [routed, quartet], {})

    medium_inputs = [routed, routed] if medium_op == "adaptive_lane_mixer" else [routed]
    medium = graph.add_op(medium_op, medium_inputs, _single_input_config(medium_op))

    hard_signal = graph.add_op("token_class_proj", [routed], {"n_classes": 4})
    hard_seed = graph.add_op(
        "signal_conditioned_compression",
        [routed, hard_signal],
        {},
    )
    hard_inputs = [hard_seed]
    if hard_op in {"compression_mixture_experts", "routing_conditioned_compression"}:
        hard_inputs = [routed, hard_signal]
    hard = graph.add_op(hard_op, hard_inputs, _single_input_config(hard_op))

    out = graph.add_op("add", [easy_lane, medium], {})
    out = graph.add_op("add", [out, hard], {})
    out = graph.add_op("add", [gated_skip, out], {})
    merge_norm = graph.add_op("rmsnorm", [out], {})
    merged_tokens = graph.add_op("linear_proj", [merge_norm], {"out_dim": model_dim})
    post = graph.add_op("rmsnorm", [merged_tokens], {})
    out = graph.add_op("add", [merge_norm, merged_tokens], {})
    out = graph.add_op("add", [out, post], {})
    out = graph.add_op("add", [inp, out], {})
    graph.set_output(out)
    return graph


def build_recursive_depth_variant(
    *,
    model_dim: int,
    max_depth: int,
    post_op: str,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    routed = graph.add_op("depth_weighted_proj", [inp], {"max_depth": max_depth})
    current = graph.add_op(post_op, [routed], _single_input_config(post_op))
    if post_op == "conv_only":
        current = graph.add_op("linear_proj", [current], {"out_dim": model_dim})
    out = graph.add_op("add", [inp, current], {})
    graph.set_output(out)
    return graph


def _single_input_config(op_name: str) -> dict[str, Any]:
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        return {"max_depth": 3}
    if op_name == "moe_topk":
        return {"num_experts": 4, "top_k": 1}
    if op_name == "moe_2expert":
        return {"num_experts": 2, "top_k": 1}
    if op_name == "adaptive_lane_mixer":
        return {"n_lanes": 3}
    return {}


def _three_lane_config(op_name: str) -> dict[str, Any]:
    if op_name == "moe_topk":
        return {"num_experts": 4, "top_k": 1}
    if op_name == "moe_2expert":
        return {"num_experts": 2, "top_k": 1}
    if op_name in {"mixed_recursion_gate", "adaptive_recursion", "route_recursion"}:
        return {"max_depth": 3}
    return {}


def _three_lane_hard_inputs(
    op_name: str,
    *,
    routed_seed: int,
    difficulty: int,
) -> list[int]:
    if op_name in {"mixed_recursion_gate", "dual_compression_blend"}:
        return [routed_seed, difficulty]
    return [routed_seed]
