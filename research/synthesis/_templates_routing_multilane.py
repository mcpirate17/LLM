"""Routing-first multilane + depth-recursion templates — private split."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_NORM,
    MotifWeights,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_routing import (
    _apply_optional_single_input_ops,
    _single_input_op_config,
    INTELLIGENT_EASY_OPTIONAL_OPS,
    INTELLIGENT_HARD_OPTIONAL_OPS,
    INTELLIGENT_MEDIUM_OPTIONAL_OPS,
    INTELLIGENT_POST_MERGE_OPTIONAL_OPS,
    INTELLIGENT_PRE_ROUTER_OPTIONAL_OPS,
)


def tpl_intelligent_multilane_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Real easy/medium/hard router with mandatory lane compute, bounded optionals, and token merge."""
    template_name = "intelligent_multilane_router"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    graph.metadata["_skip_global_decorators"] = True
    norm = _pick_compatible_motif(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    stem = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    optional_budget = 0
    pre_count = rng.randint(0, min(2, optional_budget))
    optional_budget -= pre_count
    stem, pre_selected = _apply_optional_single_input_ops(
        graph,
        stem,
        rng,
        INTELLIGENT_PRE_ROUTER_OPTIONAL_OPS,
        pre_count,
        context_prefix="intelligent_multilane_router.pre_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{template_name}[{template_instance}].pre_router",
        slot_classes=["stem"],
        selected_name=pre_selected[-1]
        if pre_selected
        else (norm.name if norm else "identity"),
        selected_class="component" if pre_selected else "norm_wrap",
        input_node_id=input_id,
    )

    gated = _add(
        graph,
        "hybrid_token_gate",
        [stem],
        {"threshold": 0.5},
        context="intelligent_multilane_router.token_gate",
    )
    gated_with_skip = _residual(
        graph,
        stem,
        gated,
        context="intelligent_multilane_router.token_gate_skip",
    )

    easy_op = rng.choice(
        (
            "cheap_verify_blend",
            "conv_only",
            "conv1d_seq",
            "linear_proj",
            "nm_sparse_linear",
            "default_path",
        )
    )
    easy_input = stem
    if easy_op == "conv1d_seq":
        easy_input = _add(
            graph,
            "rmsnorm",
            [stem],
            context="intelligent_multilane_router.easy_pre_norm",
        )
    easy_lane = _add(
        graph,
        easy_op,
        [easy_input],
        _single_input_op_config(easy_op, graph.model_dim, rng),
        context="intelligent_multilane_router.easy_mandatory",
    )
    if easy_op == "conv_only":
        easy_lane = _add(
            graph,
            "linear_proj",
            [easy_lane],
            {"out_dim": graph.model_dim},
            context="intelligent_multilane_router.easy_conv_follow",
        )
    easy_count = rng.randint(0, min(1, optional_budget))
    optional_budget -= easy_count
    easy_lane, easy_selected = _apply_optional_single_input_ops(
        graph,
        easy_lane,
        rng,
        INTELLIGENT_EASY_OPTIONAL_OPS,
        easy_count,
        context_prefix="intelligent_multilane_router.easy_lane",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{template_name}[{template_instance}].easy_router",
        slot_classes=["easy_router"],
        selected_name=easy_selected[-1] if easy_selected else easy_op,
        selected_class="component",
        input_node_id=stem,
    )

    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=3,
        slot_key=f"{template_name}[{template_instance}].pair_spans",
        slot_classes=["pair_path"],
        selected_name="sparse_span_builder",
        selected_class="component",
        input_node_id=gated,
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=4,
        slot_key=f"{template_name}[{template_instance}].triplet_spans",
        slot_classes=["triplet_path"],
        selected_name="sparse_span_builder",
        selected_class="component",
        input_node_id=gated,
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=5,
        slot_key=f"{template_name}[{template_instance}].quartet_spans",
        slot_classes=["quartet_path"],
        selected_name="sparse_span_builder",
        selected_class="component",
        input_node_id=gated,
    )

    pair_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        {"span_width": 2, "lane_count": 2, "confidence_threshold": 0.55},
        context="intelligent_multilane_router.pair_router",
    )
    triplet_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.55},
        context="intelligent_multilane_router.triplet_router",
    )
    routed_spans = _residual(
        graph,
        pair_routed,
        triplet_routed,
        context="intelligent_multilane_router.merge_pair_triplet",
    )

    medium_op = rng.choice(
        (
            "route_lanes",
            "adaptive_lane_mixer",
            "semi_structured_2_4_linear",
            "block_sparse_linear",
            "linear_proj",
            "nm_sparse_linear",
        )
    )
    medium_inputs = (
        [routed_spans, routed_spans]
        if medium_op == "adaptive_lane_mixer"
        else [routed_spans]
    )
    medium_lane = _add(
        graph,
        medium_op,
        medium_inputs,
        _single_input_op_config(medium_op, graph.model_dim, rng),
        context="intelligent_multilane_router.medium_mandatory",
    )
    medium_count = 0
    optional_budget -= medium_count
    medium_lane, medium_selected = _apply_optional_single_input_ops(
        graph,
        medium_lane,
        rng,
        INTELLIGENT_MEDIUM_OPTIONAL_OPS,
        medium_count,
        context_prefix="intelligent_multilane_router.medium_lane",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=6,
        slot_key=f"{template_name}[{template_instance}].medium_router",
        slot_classes=["medium_router"],
        selected_name=medium_selected[-1] if medium_selected else medium_op,
        selected_class="component",
        input_node_id=routed_spans,
    )

    hard_signal = _add(
        graph,
        "token_class_proj",
        [gated],
        {"n_classes": 4},
        context="intelligent_multilane_router.hard_signal",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=7,
        slot_key=f"{template_name}[{template_instance}].difficulty_signal",
        slot_classes=["difficulty_signal"],
        selected_name="token_class_proj",
        selected_class="component",
        input_node_id=gated,
    )
    hard_seed = _add(
        graph,
        "signal_conditioned_compression",
        [gated, hard_signal],
        context="intelligent_multilane_router.hard_seed",
    )
    hard_op = rng.choice(
        (
            "adaptive_recursion",
            "route_recursion",
            "moe_topk",
            "moe_2expert",
            "state_space",
            "linear_proj",
        )
    )
    hard_bridge = _add(
        graph,
        "linear_proj",
        [hard_seed],
        {"out_dim": graph.model_dim},
        context="intelligent_multilane_router.hard_bridge",
    )
    hard_inputs = [hard_bridge]
    hard_lane = _add(
        graph,
        hard_op,
        [
            _add(
                graph,
                "rmsnorm",
                hard_inputs,
                context="intelligent_multilane_router.hard_state_norm",
            )
        ]
        if hard_op == "state_space"
        else hard_inputs,
        _single_input_op_config(hard_op, graph.model_dim, rng),
        context="intelligent_multilane_router.hard_mandatory",
    )
    hard_count = 0
    optional_budget -= hard_count
    hard_lane, hard_selected = _apply_optional_single_input_ops(
        graph,
        hard_lane,
        rng,
        INTELLIGENT_HARD_OPTIONAL_OPS,
        hard_count,
        context_prefix="intelligent_multilane_router.hard_lane",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=8,
        slot_key=f"{template_name}[{template_instance}].hard_router",
        slot_classes=["hard_router"],
        selected_name=hard_selected[-1] if hard_selected else hard_op,
        selected_class="component",
        input_node_id=hard_seed,
    )

    merged = _residual(
        graph,
        easy_lane,
        medium_lane,
        context="intelligent_multilane_router.merge_easy_medium",
    )
    merged = _residual(
        graph,
        merged,
        hard_lane,
        context="intelligent_multilane_router.merge_hard",
    )
    merged = _residual(
        graph,
        gated_with_skip,
        merged,
        context="intelligent_multilane_router.merge_gate",
    )
    merged_tokens = _add(
        graph,
        "linear_proj",
        [merged],
        {"out_dim": graph.model_dim},
        context="intelligent_multilane_router.token_merge",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=9,
        slot_key=f"{template_name}[{template_instance}].token_merge",
        slot_classes=["token_merge"],
        selected_name="linear_proj",
        selected_class="component",
        input_node_id=merged,
    )

    post = _add(
        graph,
        "rmsnorm",
        [merged_tokens],
        context="intelligent_multilane_router.post_mandatory",
    )
    post_count = 0
    post, post_selected = _apply_optional_single_input_ops(
        graph,
        post,
        rng,
        INTELLIGENT_POST_MERGE_OPTIONAL_OPS,
        post_count,
        context_prefix="intelligent_multilane_router.post_merge",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=10,
        slot_key=f"{template_name}[{template_instance}].post_merge",
        slot_classes=["post_merge"],
        selected_name=post_selected[-1] if post_selected else "rmsnorm",
        selected_class="component",
        input_node_id=merged_tokens,
    )

    return _residual(
        graph,
        input_id,
        post,
        context="intelligent_multilane_router.output",
    )


def tpl_recursive_depth_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → adaptive_recursion(depth-conditional) → motif → residual.

    Depth-adaptive: tokens re-enter the block with different parameters
    each iteration. Depth is conditional on input difficulty. Easy tokens
    get 1 pass, hard tokens get up to max_depth passes.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Depth-adaptive routing
    max_depth = rng.choice([2, 3, 4])
    depth_routed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        {"max_depth": max_depth},
        context="recursive_depth_router.route",
    )

    # Post-routing motif (operates on depth-scaled tokens)
    core = _pick_compatible_motif_from_classes(
        graph,
        depth_routed,
        rng,
        list(_MIXER_CLASSES + _FFN_CLASSES),
        weights,
    )
    if core:
        processed = _instantiate_motif(graph, depth_routed, core, rng)
    else:
        processed = depth_routed
    processed = _fix_dim(graph, processed)

    return _residual(
        graph,
        input_id,
        processed,
        context="recursive_depth_router.output",
    )


def tpl_depth_token_mask_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → class signal → score_depth_blend → depth_token_mask → proj → FFN → residual.

    depth_token_mask is highly destructive on its own. Keep the routed branch,
    force the post-mask projection required by the routing rules, and only let
    the masked path act as a refinement inside a residual scaffold.
    """
    D = graph.model_dim
    normed = _add(graph, "rmsnorm", [input_id], context="depth_token_mask_block.norm")

    signal = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": 4},
        context="depth_token_mask_block.signal",
    )
    routed = _add(
        graph,
        "score_depth_blend",
        [normed, signal],
        {"max_depth": 3},
        context="depth_token_mask_block.score",
    )
    routed = _fix_dim(graph, routed)
    mask_input = _add(
        graph,
        "rmsnorm",
        [routed],
        context="depth_token_mask_block.mask_pre_norm",
    )
    masked = _add(
        graph,
        "depth_token_mask",
        [mask_input],
        {"capacity_factor": rng.choice([0.875, 0.9])},
        context="depth_token_mask_block.mask",
    )
    mask_bypass = _residual(
        graph,
        mask_input,
        masked,
        context="depth_token_mask_block.mask_bypass",
    )

    current = _add(
        graph,
        "linear_proj",
        [masked],
        {"out_dim": D},
        context="depth_token_mask_block.proj",
    )
    branch_refine = _pick_compatible_motif_from_classes(
        graph,
        current,
        rng,
        (MOTIF_CLASS_EFFICIENT_PROJ, MOTIF_CLASS_CONV),
        weights,
    )
    if branch_refine:
        current = _instantiate_motif(graph, current, branch_refine, rng)
        current = _fix_dim(graph, current)
    current = _add(
        graph,
        "layernorm",
        [current],
        context="depth_token_mask_block.post_norm",
    )
    post = _pick_compatible_motif_from_classes(
        graph, current, rng, list(_FFN_CLASSES), weights
    )
    if post:
        current = _instantiate_motif(graph, current, post, rng)
    else:
        current = _add(
            graph,
            "swiglu_mlp",
            [current],
            {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
            context="depth_token_mask_block.ffn",
        )

    current = _fix_dim(graph, current)
    current = _residual(
        graph,
        mask_bypass,
        current,
        context="depth_token_mask_block.branch_output",
    )
    return _residual(
        graph,
        input_id,
        current,
        context="depth_token_mask_block.output",
    )
