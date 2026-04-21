"""Routing-first templates — private split. Re-exported from _templates_routing."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
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
    _multiscale_gate_config,
    _multiscale_merge_config,
    _multiscale_sparse_router_config,
    _next_multiscale_hard_config,
    _next_multiscale_medium_config,
)


def tpl_difficulty_routed_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → entropy_score → {fast_path, slow_path} → gated_merge → residual.

    2-lane routing: token_type_classifier produces class logits, entropy_score
    measures their uncertainty as a (B,S,1) difficulty signal.
    Easy tokens (low entropy) get mostly the fast path (cheap linear).
    Hard tokens (high entropy) get fast + slow path (expensive motif).
    Uses mul broadcasting: (B,S,D) * (B,S,1) for differentiable gating.
    """
    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Classify → entropy: token_type_classifier (B,S,D)→(B,S,D) logits,
    # then entropy_score (B,S,D)→(B,S,1) difficulty signal.
    class_logits = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": 4},
        context="difficulty_routed_block.classify",
    )
    difficulty = _add(
        graph,
        "token_entropy",
        [class_logits],
        context="difficulty_routed_block.entropy",
    )

    # Fast path: cheap linear projection (always runs on all tokens)
    fast_out = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": graph.model_dim},
        context="difficulty_routed_block.fast_path",
    )

    # Slow path: expensive motif (attention/SSM/MoE + FFN)
    slow_motif = _pick_compatible_motif_from_classes(
        graph,
        normed,
        rng,
        list(_MIXER_CLASSES + _FFN_CLASSES),
        weights,
    )
    if slow_motif:
        slow_out = _instantiate_motif(graph, normed, slow_motif, rng)
    else:
        slow_out = normed
    slow_out = _fix_dim(graph, slow_out)

    # Gate slow path by difficulty: hard tokens get more slow-path signal
    slow_weighted = _add(
        graph,
        "mul",
        [slow_out, difficulty],
        context="difficulty_routed_block.slow_weighted",
    )

    # Merge: fast + difficulty-weighted slow
    merged = _residual(
        graph,
        fast_out,
        slow_weighted,
        context="difficulty_routed_block.merge",
    )

    # Residual
    return _residual(
        graph,
        input_id,
        merged,
        context="difficulty_routed_block.output",
    )


def tpl_three_lane_adaptive(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → adaptive_lane_mixer(3-way) → residual.

    Built-in 3-lane router: fast (identity), medium (low-rank), hard (MLP).
    The adaptive_lane_mixer op handles all lane logic internally with a
    learned gate that softly assigns tokens to difficulty lanes.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # adaptive_lane_mixer: self-contained 3-way routing
    routed = _add(
        graph,
        "difficulty_blend_3way",
        [normed, normed],
        context="three_lane_adaptive.route",
    )

    routed = _fix_dim(graph, routed)

    # Optional post-routing FFN for capacity
    ffn = _pick_compatible_motif_from_classes(graph, routed, rng, _FFN_CLASSES, weights)
    if ffn and rng.random() < 0.5:
        processed = _instantiate_motif(graph, routed, ffn, rng)
        processed = _fix_dim(graph, processed)
    else:
        processed = routed

    return _residual(
        graph,
        input_id,
        processed,
        context="three_lane_adaptive.output",
    )


def tpl_cascaded_early_exit(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → difficulty_scorer → mixer → early_exit → FFN → residual.

    True early exit: easy tokens are zeroed by the confidence gate so the
    heavy FFN does minimal useful work on them.  The outer residual add
    recovers their original representations.  During training, the early_exit
    op stores hidden states for auxiliary loss computation against the shared
    lm_head, giving the gate real gradient signal.

    Pattern:
    1. token_type_classifier → entropy_score produces per-token difficulty
    2. Mixer processes input with difficulty weighting
    3. early_exit zeros easy tokens (hard tokens continue)
    4. FFN processes remaining signal
    5. Outer residual add(input, processed) recovers easy tokens
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    classified = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": 4},
        context="cascaded_early_exit.classify",
    )
    difficulty = _add(
        graph,
        "token_entropy",
        [classified],
        context="cascaded_early_exit.entropy",
    )
    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="cascaded_early_exit.proj",
    )
    weighted = _add(
        graph,
        "mul",
        [proj, difficulty],
        context="cascaded_early_exit.weighted",
    )
    mixed = _add(
        graph,
        "linear_proj",
        [weighted],
        {"out_dim": D},
        context="cascaded_early_exit.mixed",
    )
    exited = _add(
        graph,
        "confidence_token_gate",
        [mixed],
        {"threshold": 0.5},
        context="cascaded_early_exit.exit",
    )

    # FFN processes the output — easy tokens are zero so FFN work is
    # wasted on them (future: skip zero tokens for compute savings)
    ffn = _pick_compatible_motif_from_classes(
        graph, exited, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, exited, ffn, rng) if ffn else exited
    processed = _fix_dim(graph, processed)

    # Outer residual: recovers easy tokens' original representations
    return _residual(
        graph,
        input_id,
        processed,
        context="cascaded_early_exit.output",
    )


def tpl_hybrid_sparse_triplet_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """default path + token gate + sparse triplet router + lane-conditioned block."""
    template_name = "hybrid_sparse_triplet_router"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    gate_threshold = rng.choice((0.4, 0.5, 0.6))
    confidence_threshold = rng.choice((0.4, 0.45, 0.5, 0.55))
    lane_id = rng.randrange(3)
    norm = _pick_compatible_motif(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    default_path = _add(
        graph,
        "default_path",
        [normed],
        context="hybrid_sparse_triplet_router.default_path",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{template_name}[{template_instance}].default_path",
        slot_classes=["fallback"],
        selected_name="default_path",
        selected_class="component",
        input_node_id=normed,
    )
    gated = _add(
        graph,
        "hybrid_token_gate",
        [normed],
        {"threshold": gate_threshold},
        context="hybrid_sparse_triplet_router.token_gate",
    )
    gated_with_skip = _residual(
        graph,
        normed,
        gated,
        context="hybrid_sparse_triplet_router.token_gate_skip",
    )
    spans = _add(
        graph,
        "sparse_span_builder",
        [gated],
        {"span_width": 3, "fallback_behavior": "default_path"},
        context="hybrid_sparse_triplet_router.sparse_span_builder",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{template_name}[{template_instance}].sparse_spans",
        slot_classes=["path"],
        selected_name="sparse_span_builder",
        selected_class="component",
        input_node_id=gated,
    )
    routed = _add(
        graph,
        "hybrid_sparse_router",
        [spans],
        {
            "span_width": 3,
            "lane_count": 3,
            "confidence_threshold": confidence_threshold,
        },
        context="hybrid_sparse_triplet_router.lane_router",
    )
    lane_block = _add(
        graph,
        "lane_conditioned_block",
        [routed],
        {"lane_id": lane_id},
        context="hybrid_sparse_triplet_router.lane_block",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=3,
        slot_key=f"{template_name}[{template_instance}].routed_lane",
        slot_classes=["lane"],
        selected_name="lane_conditioned_block",
        selected_class="component",
        input_node_id=routed,
    )
    merged = _residual(
        graph,
        default_path,
        lane_block,
        context="hybrid_sparse_triplet_router.merge",
    )
    fused = _residual(
        graph,
        gated_with_skip,
        merged,
        context="hybrid_sparse_triplet_router.output_fuse",
    )
    return _add(
        graph,
        "rmsnorm",
        [fused],
        context="hybrid_sparse_triplet_router.output",
    )


def tpl_multiscale_difficulty_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Easy tokens stay cheap, medium tokens route through multi-span lanes, hard tokens hit a heavy expert path."""
    graph.metadata["_skip_global_decorators"] = True
    template_name = "multiscale_difficulty_router"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    hard_classes = rng.choice((3, 4, 5))
    norm = _pick_compatible_motif(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    default_path = _add(
        graph,
        "default_path",
        [normed],
        context="multiscale_difficulty_router.default_path",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{template_name}[{template_instance}].default_path",
        slot_classes=["fallback"],
        selected_name="default_path",
        selected_class="component",
        input_node_id=normed,
    )

    gated = _add(
        graph,
        "hybrid_token_gate",
        [normed],
        _multiscale_gate_config(),
        context="multiscale_difficulty_router.token_gate",
    )
    gated_with_skip = _residual(
        graph,
        normed,
        gated,
        context="multiscale_difficulty_router.token_gate_skip",
    )

    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
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
        slot_index=3,
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
        slot_index=4,
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
        _multiscale_sparse_router_config(2),
        context="multiscale_difficulty_router.pair_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=5,
        slot_key=f"{template_name}[{template_instance}].pair_router",
        slot_classes=["pair_router"],
        selected_name="hybrid_sparse_router",
        selected_class="component",
        input_node_id=gated,
    )
    triplet_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        _multiscale_sparse_router_config(3),
        context="multiscale_difficulty_router.triplet_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=6,
        slot_key=f"{template_name}[{template_instance}].triplet_router",
        slot_classes=["triplet_router"],
        selected_name="hybrid_sparse_router",
        selected_class="component",
        input_node_id=gated,
    )
    quartet_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        _multiscale_sparse_router_config(4),
        context="multiscale_difficulty_router.quartet_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=7,
        slot_key=f"{template_name}[{template_instance}].quartet_router",
        slot_classes=["quartet_router"],
        selected_name="hybrid_sparse_router",
        selected_class="component",
        input_node_id=gated,
    )

    medium_pre = _residual(
        graph,
        pair_routed,
        triplet_routed,
        context="multiscale_difficulty_router.merge_pair_triplet",
    )
    medium_pre = _residual(
        graph,
        medium_pre,
        quartet_routed,
        context="multiscale_difficulty_router.merge_quartet",
    )
    medium_op = "adaptive_lane_mixer"
    medium_inputs = [medium_pre, medium_pre]
    medium_core = _add(
        graph,
        medium_op,
        medium_inputs,
        _next_multiscale_medium_config(medium_op, rng),
        context="multiscale_difficulty_router.medium_core",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=8,
        slot_key=f"{template_name}[{template_instance}].medium_router",
        slot_classes=["medium_router"],
        selected_name=medium_op,
        selected_class="component",
        input_node_id=medium_pre,
    )

    hard_signal = _add(
        graph,
        "token_class_proj",
        [gated],
        {"n_classes": hard_classes},
        context="multiscale_difficulty_router.hard_signal",
    )
    hard_op = "route_recursion"
    hard_seed = _add(
        graph,
        "signal_conditioned_compression",
        [gated, hard_signal],
        context="multiscale_difficulty_router.hard_seed",
    )
    hard_inputs = [
        _add(
            graph,
            "linear_proj",
            [hard_seed],
            {"out_dim": graph.model_dim},
            context="multiscale_difficulty_router.hard_bridge",
        )
    ]
    hard_core = _add(
        graph,
        hard_op,
        hard_inputs,
        _next_multiscale_hard_config(hard_op, rng),
        context="multiscale_difficulty_router.hard_core",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=9,
        slot_key=f"{template_name}[{template_instance}].hard_router",
        slot_classes=["hard_router"],
        selected_name=hard_op,
        selected_class="hybrid",
        input_node_id=hard_inputs[0],
    )

    merged = _add(
        graph,
        "calibrated_branch_merge",
        [default_path, medium_core],
        _multiscale_merge_config(
            primary_role="default",
            secondary_role="medium",
            min_secondary_share=0.16,
            max_secondary_share=0.4,
            min_secondary_start=0.24,
            min_secondary_mid=0.2,
            min_secondary_end=0.16,
        ),
        context="multiscale_difficulty_router.merge_medium",
    )
    merged = _add(
        graph,
        "calibrated_branch_merge",
        [merged, hard_core],
        _multiscale_merge_config(
            primary_role="routed",
            secondary_role="hard",
            min_secondary_share=0.08,
            max_secondary_share=0.2,
            min_secondary_start=0.04,
            min_secondary_mid=0.08,
            min_secondary_end=0.1,
            max_secondary_start=0.1,
            max_secondary_mid=0.16,
            max_secondary_end=0.22,
        ),
        context="multiscale_difficulty_router.merge_hard",
    )
    merged = _residual(
        graph,
        gated_with_skip,
        merged,
        context="multiscale_difficulty_router.merge_gate",
    )
    merged = _residual(
        graph,
        input_id,
        merged,
        context="multiscale_difficulty_router.output_residual",
    )
    return merged


def tpl_multiscale_difficulty_router_adaptive_attn_ssm(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Controlled router with fixed adaptive medium lane and attention-bearing hard lane."""
    graph.metadata["_skip_global_decorators"] = True
    template_name = "multiscale_difficulty_router_adaptive_attn_ssm"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    hard_classes = rng.choice((3, 4, 5))
    norm = _pick_compatible_motif(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    easy_ssm = _add(
        graph,
        "state_space",
        [normed],
        context="multiscale_difficulty_router_adaptive_attn_ssm.easy_ssm",
    )
    default_path = _add(
        graph,
        "default_path",
        [easy_ssm],
        context="multiscale_difficulty_router_adaptive_attn_ssm.default_path_wrap",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{template_name}[{template_instance}].default_path",
        slot_classes=["fallback"],
        selected_name="easy_attn_ssm",
        selected_class="hybrid",
        input_node_id=normed,
    )

    gated = _add(
        graph,
        "hybrid_token_gate",
        [normed],
        _multiscale_gate_config(),
        context="multiscale_difficulty_router_adaptive_attn_ssm.token_gate",
    )
    pair_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        _multiscale_sparse_router_config(2),
        context="multiscale_difficulty_router_adaptive_attn_ssm.pair_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{template_name}[{template_instance}].pair_router",
        slot_classes=["pair_router"],
        selected_name="hybrid_sparse_router",
        selected_class="component",
        input_node_id=gated,
    )
    triplet_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        _multiscale_sparse_router_config(3),
        context="multiscale_difficulty_router_adaptive_attn_ssm.triplet_router",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=3,
        slot_key=f"{template_name}[{template_instance}].triplet_router",
        slot_classes=["triplet_router"],
        selected_name="hybrid_sparse_router",
        selected_class="component",
        input_node_id=gated,
    )

    medium_pre = _residual(
        graph,
        pair_routed,
        triplet_routed,
        context="multiscale_difficulty_router_adaptive_attn_ssm.merge_pair_triplet",
    )
    medium_core = _add(
        graph,
        "adaptive_lane_mixer",
        [medium_pre, medium_pre],
        _next_multiscale_medium_config("adaptive_lane_mixer", rng),
        context="multiscale_difficulty_router_adaptive_attn_ssm.medium_core",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=4,
        slot_key=f"{template_name}[{template_instance}].medium_router",
        slot_classes=["medium_router"],
        selected_name="adaptive_lane_mixer",
        selected_class="component",
        input_node_id=medium_pre,
    )

    hard_signal = _add(
        graph,
        "token_class_proj",
        [gated],
        {"n_classes": hard_classes},
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_signal",
    )
    hard_seed = _add(
        graph,
        "signal_conditioned_compression",
        [gated, hard_signal],
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_seed",
    )
    hard_seed = _add(
        graph,
        "linear_proj",
        [hard_seed],
        {"out_dim": graph.model_dim},
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_seed_bridge",
    )
    hard_core = _add(
        graph,
        "route_recursion",
        [hard_seed],
        _next_multiscale_hard_config("route_recursion", rng),
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_core",
    )
    hard_attn_input = _add(
        graph,
        "rmsnorm",
        [hard_core],
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_attn_input",
    )
    hard_post = _add(
        graph,
        "graph_attention",
        [hard_attn_input],
        context="multiscale_difficulty_router_adaptive_attn_ssm.hard_post",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=5,
        slot_key=f"{template_name}[{template_instance}].hard_router",
        slot_classes=["hard_router"],
        selected_name="route_recursion+graph_attention",
        selected_class="hybrid",
        input_node_id=hard_seed,
    )

    merged = _add(
        graph,
        "calibrated_branch_merge",
        [default_path, medium_core],
        _multiscale_merge_config(
            primary_role="default",
            secondary_role="medium",
            min_secondary_share=0.16,
            max_secondary_share=0.4,
            min_secondary_start=0.24,
            min_secondary_mid=0.2,
            min_secondary_end=0.16,
        ),
        context="multiscale_difficulty_router_adaptive_attn_ssm.merge_medium",
    )
    merged = _add(
        graph,
        "calibrated_branch_merge",
        [merged, hard_post],
        _multiscale_merge_config(
            primary_role="routed",
            secondary_role="hard",
            min_secondary_share=0.08,
            max_secondary_share=0.2,
            min_secondary_start=0.04,
            min_secondary_mid=0.08,
            min_secondary_end=0.1,
            max_secondary_start=0.1,
            max_secondary_mid=0.16,
            max_secondary_end=0.22,
        ),
        context="multiscale_difficulty_router_adaptive_attn_ssm.merge_hard",
    )
    merged = _residual(
        graph,
        input_id,
        merged,
        context="multiscale_difficulty_router_adaptive_attn_ssm.output_residual",
    )
    return merged


def tpl_multiscale_rich_lane_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Three-tier router with richer medium/hard lane menus and bounded lane scaffolds."""
    graph.metadata["_skip_global_decorators"] = True
    template_name = "multiscale_rich_lane_router"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    norm = _pick_compatible_motif(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    default_path = _add(
        graph,
        "default_path",
        [normed],
        context="multiscale_rich_lane_router.default_path",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{template_name}[{template_instance}].default_path",
        slot_classes=["fallback"],
        selected_name="default_path",
        selected_class="component",
        input_node_id=normed,
    )

    gated = _add(
        graph,
        "hybrid_token_gate",
        [normed],
        _multiscale_gate_config(),
        context="multiscale_rich_lane_router.token_gate",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
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
        slot_index=3,
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
        slot_index=4,
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
        _multiscale_sparse_router_config(2),
        context="multiscale_rich_lane_router.pair_router",
    )
    triplet_routed = _add(
        graph,
        "hybrid_sparse_router",
        [gated],
        _multiscale_sparse_router_config(3),
        context="multiscale_rich_lane_router.triplet_router",
    )
    medium_pre = _residual(
        graph,
        pair_routed,
        triplet_routed,
        context="multiscale_rich_lane_router.merge_pair_triplet",
    )
    medium_op = "adaptive_lane_mixer"
    medium_inputs = [medium_pre, medium_pre]
    medium_core = _add(
        graph,
        medium_op,
        medium_inputs,
        _next_multiscale_medium_config(medium_op, rng),
        context="multiscale_rich_lane_router.medium_core",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=5,
        slot_key=f"{template_name}[{template_instance}].medium_router",
        slot_classes=["medium_router"],
        selected_name=medium_op,
        selected_class="component",
        input_node_id=medium_pre,
    )

    hard_signal = _add(
        graph,
        "token_class_proj",
        [gated],
        {"n_classes": 4},
        context="multiscale_rich_lane_router.hard_signal",
    )
    hard_seed = _add(
        graph,
        "signal_conditioned_compression",
        [gated, hard_signal],
        context="multiscale_rich_lane_router.hard_seed",
    )
    hard_op = "route_recursion"
    hard_inputs = [
        _add(
            graph,
            "linear_proj",
            [hard_seed],
            {"out_dim": graph.model_dim},
            context="multiscale_rich_lane_router.hard_bridge",
        )
    ]
    hard_core = _add(
        graph,
        hard_op,
        hard_inputs,
        _next_multiscale_hard_config(hard_op, rng),
        context="multiscale_rich_lane_router.hard_core",
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=6,
        slot_key=f"{template_name}[{template_instance}].hard_router",
        slot_classes=["hard_router"],
        selected_name=hard_op,
        selected_class="component",
        input_node_id=hard_seed,
    )

    merged = _add(
        graph,
        "calibrated_branch_merge",
        [default_path, medium_core],
        _multiscale_merge_config(
            primary_role="default",
            secondary_role="medium",
            min_secondary_share=0.18,
            max_secondary_share=0.42,
            min_secondary_start=0.26,
            min_secondary_mid=0.22,
            min_secondary_end=0.18,
        ),
        context="multiscale_rich_lane_router.merge_medium",
    )
    merged = _add(
        graph,
        "calibrated_branch_merge",
        [merged, hard_core],
        _multiscale_merge_config(
            primary_role="routed",
            secondary_role="hard",
            min_secondary_share=0.08,
            max_secondary_share=0.2,
            min_secondary_start=0.04,
            min_secondary_mid=0.08,
            min_secondary_end=0.1,
            max_secondary_start=0.1,
            max_secondary_mid=0.16,
            max_secondary_end=0.22,
        ),
        context="multiscale_rich_lane_router.merge_hard",
    )
    return _residual(
        graph, input_id, merged, context="multiscale_rich_lane_router.output_residual"
    )
