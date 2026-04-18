"""Routing-first templates — mandatory routing structure."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
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


# ── Routing-First Templates (Phase 2) ──────────────────────────────
#
# These templates MANDATE routing structure: every graph produced by
# these templates has a difficulty scorer and differential compute paths.
# The grammar fills motif slots, but the routing skeleton is fixed.

# Set of all routing-first template names for grammar filtering.
ROUTING_TEMPLATES: frozenset = frozenset(
    {
        "difficulty_routed_block",
        "three_lane_adaptive",
        "cascaded_early_exit",
        "hybrid_sparse_triplet_router",
        "multiscale_difficulty_router",
        "multiscale_difficulty_router_easy_attn_ssm",
        "multiscale_difficulty_router_adaptive_attn_ssm",
        "multiscale_difficulty_router_blocksparse_attn_ssm",
        "multiscale_rich_lane_router",
        "intelligent_multilane_router",
        "recursive_depth_router",
        "conditional_compute",
        "token_merge_block",
        "routed_bottleneck",
        "sparse_moe_block",
        # Attention templates that produce routing ops internally
        "attn_routing_block",
        "attn_moe_block",
        "attn_three_way_split",
        "attn_conditional_compute",
        "attn_sparse_moe",
        "diff_attn_routing",
        "local_attn_routing",
        "latent_attn_moe",
        "local_attn_moe",
        "diff_attn_moe",
        "graph_attn_moe",
        # Role-slot trunk+sidecar templates (capability-first family).
        # These enforce explicit binding_write → global_retrieval → binding_read
        # with a typed-entropy or sparse-router controller and gated merge.
        "typed_slot_memory_block",
        "sparse_relation_graph_block",
        "token_program_interpreter_block",
        # Role-slot v2: proven trunks + retrieval sidecar.
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    }
)


# Capability-first subset: templates that wire an explicit exact-retrieval
# sidecar merged into a compression trunk via a gated add. Used by the
# ``capability_first`` grammar preset to pressure the search toward graphs
# that can win ppl AND binding/induction/ar simultaneously.
CAPABILITY_FIRST_TEMPLATES: frozenset = frozenset(
    {
        "typed_slot_memory_block",
        "sparse_relation_graph_block",
        "token_program_interpreter_block",
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    }
)

# Curated candidate pools for the promoted multiscale rich router.
# The broader manifest-compatible menus remain documented in component metadata,
# but generation is intentionally narrower here because short-run audit data
# showed several medium/hard options were either underpowered, collapse-prone,
# or outright non-viable in this template context.
NEXT_MULTISCALE_MEDIUM_LANE_OPS: tuple[str, ...] = (
    "conv1d_seq",
    "semi_structured_2_4_linear",
    "block_sparse_linear",
    "adaptive_lane_mixer",
    "cheap_verify_blend",
)

DIFFICULTY_ROUTER_MEDIUM_LANE_OPS: tuple[str, ...] = (
    "semi_structured_2_4_linear",
    "block_sparse_linear",
    "adaptive_lane_mixer",
)

NEXT_MULTISCALE_HARD_LANE_OPS: tuple[str, ...] = (
    "mixed_recursion_gate",
    "dual_compression_blend",
    "adaptive_recursion",
    "route_recursion",
    "moe_topk",
)

DIFFICULTY_ROUTER_HARD_LANE_OPS: tuple[str, ...] = ("route_recursion",)

DIFFICULTY_ROUTER_HARD_ATTENTION_OPS: tuple[str, ...] = ("graph_attention",)

INTELLIGENT_PRE_ROUTER_OPTIONAL_OPS: tuple[str, ...] = (
    "cheap_verify_blend",
    "linear_proj",
)

INTELLIGENT_EASY_MANDATORY_OPS: tuple[str, ...] = (
    "cheap_verify_blend",
    "conv_only",
    "conv1d_seq",
)

INTELLIGENT_EASY_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
    "default_path",
)

INTELLIGENT_MEDIUM_MANDATORY_OPS: tuple[str, ...] = (
    "route_lanes",
    "adaptive_lane_mixer",
    "semi_structured_2_4_linear",
    "block_sparse_linear",
)

INTELLIGENT_MEDIUM_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
)

INTELLIGENT_HARD_MANDATORY_OPS: tuple[str, ...] = (
    "adaptive_recursion",
    "route_recursion",
    "moe_topk",
    "moe_2expert",
)

INTELLIGENT_HARD_OPTIONAL_OPS: tuple[str, ...] = (
    "state_space",
    "linear_proj",
)

INTELLIGENT_POST_MERGE_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
)


def _next_multiscale_medium_config(op_name: str, rng: random.Random) -> dict:
    if op_name == "route_lanes":
        return {"n_lanes": 3}
    if op_name == "adaptive_lane_mixer":
        return {"n_lanes": 3}
    return {}


def _next_multiscale_hard_config(op_name: str, rng: random.Random) -> dict:
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        max_depth = rng.choice([2, 3, 4])
        return {
            "max_depth": max_depth,
            "curriculum_enabled": True,
            "curriculum_warmup_frac": 0.25,
            "curriculum_mid_frac": 0.65,
            "active_depth_start": 1,
            "active_depth_mid": min(2, max_depth),
            "active_depth_end": max_depth,
        }
    if op_name == "moe_topk":
        return {"num_experts": rng.choice([2, 4]), "top_k": 1}
    if op_name == "moe_2expert":
        return {}
    if op_name == "state_space":
        return {}
    return {}


def _multiscale_gate_config() -> dict:
    return {
        "threshold": 0.5,
        "gate_temperature": 1.0,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "threshold_start": 0.34,
        "threshold_mid": 0.4,
        "threshold_end": 0.46,
        "gate_temperature_start": 1.35,
        "gate_temperature_mid": 1.1,
        "gate_temperature_end": 1.0,
    }


def _multiscale_sparse_router_config(span_width: int) -> dict:
    return {
        "span_width": span_width,
        "lane_count": span_width,
        "confidence_threshold": 0.55,
        "min_keep_fraction": 0.125,
        "route_temperature": 0.85,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "confidence_threshold_start": 0.3,
        "confidence_threshold_mid": 0.4,
        "confidence_threshold_end": 0.48,
        "min_keep_fraction_start": 0.28,
        "min_keep_fraction_mid": 0.2,
        "min_keep_fraction_end": 0.16,
        "route_temperature_start": 1.35,
        "route_temperature_mid": 1.05,
        "route_temperature_end": 0.9,
    }


def _multiscale_merge_config(
    *,
    primary_role: str,
    secondary_role: str,
    min_secondary_share: float,
    max_secondary_share: float,
    min_secondary_start: float | None = None,
    min_secondary_mid: float | None = None,
    min_secondary_end: float | None = None,
    max_secondary_start: float | None = None,
    max_secondary_mid: float | None = None,
    max_secondary_end: float | None = None,
) -> dict:
    return {
        "n_branches": 2,
        "primary_role": primary_role,
        "secondary_role": secondary_role,
        "normalize_inputs": True,
        "merge_temperature": 0.9,
        "min_secondary_share": min_secondary_share,
        "max_secondary_share": max_secondary_share,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "min_secondary_share_start": min_secondary_share
        if min_secondary_start is None
        else min_secondary_start,
        "min_secondary_share_mid": min_secondary_share
        if min_secondary_mid is None
        else min_secondary_mid,
        "min_secondary_share_end": min_secondary_share
        if min_secondary_end is None
        else min_secondary_end,
        "max_secondary_share_start": max_secondary_share
        if max_secondary_start is None
        else max_secondary_start,
        "max_secondary_share_mid": max_secondary_share
        if max_secondary_mid is None
        else max_secondary_mid,
        "max_secondary_share_end": max_secondary_share
        if max_secondary_end is None
        else max_secondary_end,
    }


def _single_input_op_config(op_name: str, model_dim: int, rng: random.Random) -> dict:
    if op_name == "linear_proj":
        return {"out_dim": model_dim}
    if op_name in {"route_lanes", "adaptive_lane_mixer"}:
        return {"n_lanes": 3}
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        return {"max_depth": rng.choice([2, 3, 4])}
    if op_name == "moe_topk":
        return {"num_experts": rng.choice([2, 4]), "top_k": 1}
    return {}


def _apply_optional_single_input_ops(
    graph: ComputationGraph,
    start_id: int,
    rng: random.Random,
    candidates: tuple[str, ...],
    count: int,
    *,
    context_prefix: str,
) -> tuple[int, list[str]]:
    node_id = start_id
    selected: list[str] = []
    for idx, op_name in enumerate(rng.sample(list(candidates), k=count), start=1):
        node_id = _add(
            graph,
            op_name,
            [node_id],
            _single_input_op_config(op_name, graph.model_dim, rng),
            context=f"{context_prefix}.optional_{idx}",
        )
        selected.append(op_name)
    return node_id, selected


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
    hard_post = _add(graph, "graph_attention", [hard_attn_input], context="multiscale_difficulty_router_adaptive_attn_ssm.hard_post")
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
    return _residual(graph, input_id, merged, context="multiscale_rich_lane_router.output_residual")


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
    masked = _add(
        graph,
        "depth_token_mask",
        [routed],
        {"capacity_factor": rng.choice([0.875, 0.9])},
        context="depth_token_mask_block.mask",
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
        routed,
        current,
        context="depth_token_mask_block.branch_output",
    )
    return _residual(
        graph,
        input_id,
        current,
        context="depth_token_mask_block.output",
    )


# ── Latent Compression Templates ──────────────────────────────────
#
# Dedicated template for latent_attention_compressor — the single best-
# performing op in the leaderboard (lr=0.0061) but severely underexplored
# because it has no template forcing its selection.


def tpl_latent_compress_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → linear_proj → latent_attention_compressor → add →
    sparse_linear → act → residual_add.

    Based on the best-ever architecture pattern (5bc26a03, lr=0.0061):
    linear_proj → latent_attention_compressor → add → nm_sparse_linear →
    progressive_compression_gate → rmsnorm → rwkv_channel → add
    """
    D = graph.model_dim
    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Projection → latent attention compressor
    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="latent_compress_block.proj",
    )
    compressed = _add(
        graph,
        "latent_attention_compressor",
        [proj],
        context="latent_compress_block.compress",
    )

    # Inner residual (normed + compressed)
    inner_res = _residual(
        graph,
        normed,
        compressed,
        context="latent_compress_block.inner_residual",
    )

    # Sparse linear (nm_sparse or semi_structured)
    sparse_op = rng.choice(["nm_sparse_linear", "semi_structured_2_4_linear"])
    sparse_config: dict = {"out_dim": D}
    if sparse_op == "nm_sparse_linear":
        sparse_config.update({"n": 2, "m": 4})
    sparse = _add(
        graph,
        sparse_op,
        [inner_res],
        sparse_config,
        context="latent_compress_block.sparse",
    )

    # Activation
    act_op = rng.choice(["silu", "gelu", "relu"])
    activated = _add(
        graph,
        act_op,
        [sparse],
        context="latent_compress_block.activation",
    )

    activated = _fix_dim(graph, activated)

    # Outer residual
    return _residual(
        graph,
        input_id,
        activated,
        context="latent_compress_block.output",
    )


def tpl_latent_compress_rwkv(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → linear_proj → latent_attention_compressor → add →
    sparse_linear → adaptive_rank_gate → norm → rwkv_channel → residual.

    Based on the best-ever graph pattern (5bc26a03, lr=0.0061) with
    randomized sparse op choice.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="latent_compress_rwkv.proj",
    )
    compressed = _add(
        graph,
        "latent_attention_compressor",
        [proj],
        context="latent_compress_rwkv.compress",
    )
    inner_res = _residual(
        graph,
        normed,
        compressed,
        context="latent_compress_rwkv.inner_residual",
    )

    sparse_op = rng.choice(
        ["nm_sparse_linear", "semi_structured_2_4_linear", "block_sparse_linear"]
    )
    sparse_cfg: dict = {"out_dim": D}
    if sparse_op == "nm_sparse_linear":
        sparse_cfg.update({"n": 2, "m": 4})
    elif sparse_op == "block_sparse_linear":
        sparse_cfg.update(
            {
                "block_size": rng.choice([8, 16, 32]),
                "block_density": rng.uniform(0.1, 0.4),
            }
        )
    sparse = _add(
        graph,
        sparse_op,
        [inner_res],
        sparse_cfg,
        context="latent_compress_rwkv.sparse",
    )
    sparse = _add(
        graph,
        "linear_proj",
        [sparse],
        {"out_dim": D},
        context="latent_compress_rwkv.sparse_bridge",
    )

    # Progressive compression gate (if available)
    gated = _add(
        graph,
        "adaptive_rank_gate",
        [sparse],
        context="latent_compress_rwkv.rank_gate",
    )

    # Post-norm + RWKV channel mixing
    norm2 = _pick_compatible_motif(graph, gated, rng, MOTIF_CLASS_NORM, weights)
    post_normed = _instantiate_motif(graph, gated, norm2, rng) if norm2 else gated

    mixed = _add(
        graph,
        "rwkv_channel",
        [post_normed],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="latent_compress_rwkv.mixed",
    )

    mixed = _fix_dim(graph, mixed)

    return _residual(
        graph,
        input_id,
        mixed,
        context="latent_compress_rwkv.output",
    )


# ── 2-Input Routing Templates ─────────────────────────────────────
#
# These templates wire routing ops that require a signal producer as
# input[1], matching OP_WIRING_RULES input_signals constraints.


def tpl_signal_routed_compression(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → {compression_mixture_experts | routing_conditioned_compression} → residual.

    2-input routing: token_type_classifier produces routing signal,
    which drives per-token compression method selection.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Produce routing signal
    signal = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": rng.choice([2, 3, 4])},
        context="signal_routed_compression.signal",
    )

    # 40% chance: attention on data before compression routing
    if rng.random() < 0.4:
        from ._template_helpers import MOTIF_CLASS_ATTENTION

        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            attended = _instantiate_motif(graph, normed, attn, rng)
            normed = _fix_dim(graph, attended)

    # Route through compression op (2-input: data + signal)
    comp_op = rng.choice(["dual_compression_blend", "signal_conditioned_compression"])
    compressed = _add(
        graph,
        comp_op,
        [normed, signal],
        context="signal_routed_compression.compress",
    )

    compressed = _fix_dim(graph, compressed)

    # Optional moe_topk after compression (60% chance) — data mining shows
    # dual_compression_blend + moe_topk is the top underexplored high-signal combo
    if rng.random() < 0.6:
        compressed = _add(
            graph,
            "moe_topk",
            [compressed],
            {
                "num_experts": rng.choice([2, 4]),
                "top_k": rng.choice([1, 2]),
            },
            context="signal_routed_compression.moe",
        )
        compressed = _fix_dim(graph, compressed)

    return _residual(
        graph,
        input_id,
        compressed,
        context="signal_routed_compression.output",
    )


def tpl_dual_routing_stack(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → token_class_proj → dual_compression_blend → moe_topk → [FFN] → residual.

    Stacks a compression router on top of an expert router. Data-mined pattern:
    dual_compression_blend + moe_topk is the most underexplored high-signal combo
    (loss_ratio=0.057, n=8). The 2-input wiring sends normed data as input[0] and
    classifier output as input[1] to dual_compression_blend.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Routing signal from classifier
    signal = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": rng.choice([2, 3, 4])},
        context="dual_routing_stack.signal",
    )

    # 2-input compression: data + routing signal
    compressed = _add(
        graph,
        "dual_compression_blend",
        [normed, signal],
        context="dual_routing_stack.compress",
    )

    # Expert routing
    routed = _add(
        graph,
        "moe_topk",
        [compressed],
        {
            "num_experts": rng.choice([2, 4]),
            "top_k": rng.choice([1, 2]),
        },
        context="dual_routing_stack.moe",
    )

    # Optional FFN motif (50% chance)
    if rng.random() < 0.5:
        ffn = _pick_compatible_motif_from_classes(
            graph, routed, rng, list(_FFN_CLASSES), weights
        )
        if ffn:
            routed = _instantiate_motif(graph, routed, ffn, rng)

    routed = _fix_dim(graph, routed)

    return _residual(
        graph,
        input_id,
        routed,
        context="dual_routing_stack.output",
    )


def tpl_dual_routing_deep(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → token_class_proj → dual_compression_blend → layernorm → moe_topk → [FFN] → residual.

    Deeper variant with a second norm between the two routing ops. Matches the
    4-combo pattern: dual_compression_blend + layernorm + moe_topk (loss_ratio=0.057).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    signal = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": 4},
        context="dual_routing_deep.signal",
    )
    compressed = _add(
        graph,
        "dual_compression_blend",
        [normed, signal],
        context="dual_routing_deep.compress",
    )
    mid_normed = _add(
        graph,
        "layernorm",
        [compressed],
        context="dual_routing_deep.mid_norm",
    )
    mid_normed = _add(
        graph,
        "linear_proj",
        [mid_normed],
        {"out_dim": graph.model_dim},
        context="dual_routing_deep.pre_moe_bridge",
    )
    routed = _add(
        graph,
        "moe_topk",
        [mid_normed],
        {"num_experts": 4, "top_k": 2},
        context="dual_routing_deep.moe",
    )
    routed = _add(
        graph,
        "linear_proj",
        [routed],
        {"out_dim": graph.model_dim},
        context="dual_routing_deep.post_moe_bridge",
    )

    # Optional FFN motif
    if rng.random() < 0.5:
        routed = _add(
            graph,
            "swiglu_mlp",
            [routed],
            {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
            context="dual_routing_deep.ffn",
        )

    routed = _fix_dim(graph, routed)

    return _residual(
        graph,
        input_id,
        routed,
        context="dual_routing_deep.output",
    )


def tpl_routing_conditioned_moe(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → token_class_proj → signal_conditioned_compression → moe_topk → residual.

    Variant using signal_conditioned_compression instead of dual_compression_blend.
    Same 2-input wiring pattern, targeting the adjacent high-signal combo.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    signal = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": rng.choice([2, 3, 4])},
        context="routing_conditioned_moe.signal",
    )
    compressed = _add(
        graph,
        "signal_conditioned_compression",
        [normed, signal],
        context="routing_conditioned_moe.compress",
    )
    routed = _add(
        graph,
        "moe_topk",
        [compressed],
        {"num_experts": rng.choice([2, 4]), "top_k": 1},
        context="routing_conditioned_moe.moe",
    )

    routed = _fix_dim(graph, routed)

    return _residual(
        graph,
        input_id,
        routed,
        context="routing_conditioned_moe.output",
    )


def tpl_mixed_recursion(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → mixed_recursion_gate(x, scores) → motif → residual.

    Depth-conditional: token_type_classifier produces depth scores,
    mixed_recursion_gate applies per-step transforms masked by depth.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Depth scores from classifier
    scores = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": rng.choice([3, 4, 5])},
        context="mixed_recursion.scores",
    )
    gated = _add(
        graph,
        "score_depth_blend",
        [normed, scores],
        {"max_depth": rng.choice([2, 3, 4])},
        context="mixed_recursion.gated",
    )

    # Post-routing motif: 40% chance of forced attention
    from ._template_helpers import MOTIF_CLASS_ATTENTION

    if rng.random() < 0.4:
        core = _pick_compatible_motif(graph, gated, rng, MOTIF_CLASS_ATTENTION, weights)
    else:
        core = _pick_compatible_motif_from_classes(
            graph,
            gated,
            rng,
            list(_MIXER_CLASSES + _FFN_CLASSES),
            weights,
        )
    if core:
        processed = _instantiate_motif(graph, gated, core, rng)
    else:
        processed = gated
    processed = _fix_dim(graph, processed)

    return _residual(
        graph,
        input_id,
        processed,
        context="mixed_recursion.output",
    )


def _tpl_depth_gated_block_impl(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    template_name: str,
    matmul_branch_prob: float,
    stabilize_matmul: bool = False,
    matmul_follower_classes: tuple[str, ...] | None = None,
) -> int:
    """norm → depth_gated_transform → controlled mixer branch → residual.

    Depth gating: tokens get variable-depth processing based on learned difficulty.
    """
    D = graph.model_dim
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    gated = _add(
        graph,
        "depth_gated_transform",
        [normed],
        {"out_dim": D},
        context="depth_gated_block.gated",
    )

    slot_index = int(graph.metadata.get("_active_template_slot_counter", 0) or 0)
    # Keep the shell stable, but give it a direct matmul branch often enough to
    # test whether the strong loss behavior comes from a better full-width
    # refinement core rather than the existing motif menu alone.
    use_matmul_branch = rng.random() < matmul_branch_prob
    if use_matmul_branch:
        proj_a = _add(
            graph,
            "linear_proj",
            [gated],
            {"out_dim": D},
            context="depth_gated_block.matmul_proj_a",
        )
        proj_b = _add(
            graph,
            "linear_proj",
            [gated],
            {"out_dim": D},
            context="depth_gated_block.matmul_proj_b",
        )
        refined = _add(
            graph,
            "matmul",
            [proj_a, proj_b],
            context="depth_gated_block.matmul",
        )
        mixed = _add(
            graph,
            "linear_proj",
            [refined],
            {"out_dim": D},
            context="depth_gated_block.matmul_out",
        )
        if stabilize_matmul:
            mixed = _add(
                graph,
                "rmsnorm",
                [mixed],
                context="depth_gated_block.matmul_stabilize",
            )
            if matmul_follower_classes:
                follower = _pick_compatible_motif_from_classes(
                    graph,
                    mixed,
                    rng,
                    matmul_follower_classes,
                    weights,
                )
                if follower:
                    mixed = _instantiate_motif(graph, mixed, follower, rng)
        record_template_slot_binding(
            graph,
            template_name=template_name,
            template_instance=template_instance,
            slot_index=slot_index,
            slot_key=f"{template_name}[{template_instance}].slot{slot_index}",
            slot_classes=[
                MOTIF_CLASS_ATTENTION,
                MOTIF_CLASS_SSM,
                MOTIF_CLASS_CONV,
                MOTIF_CLASS_CHANNEL,
                MOTIF_CLASS_MATH_SPACE,
            ],
            selected_name="matmul_refine",
            selected_class=MOTIF_CLASS_MATH_SPACE,
            input_node_id=gated,
        )
        graph.metadata["_active_template_slot_counter"] = slot_index + 1
    else:
        # Exclude MATH_SPACE motifs from the generic picker: many math-space
        # motifs in this catalog are themselves gating-heavy. The direct
        # matmul branch above is the controlled math-space experiment here.
        depth_gated_mixer = (
            MOTIF_CLASS_ATTENTION,
            MOTIF_CLASS_CHANNEL,
            MOTIF_CLASS_SSM,
            MOTIF_CLASS_CONV,
        )
        mixer = _pick_compatible_motif_from_classes(
            graph, gated, rng, depth_gated_mixer, weights
        )
        mixed = _instantiate_motif(graph, gated, mixer, rng) if mixer else gated
    mixed = _fix_dim(graph, mixed)

    return _residual(
        graph,
        input_id,
        mixed,
        context="depth_gated_block.output",
    )


def tpl_depth_gated_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    return _tpl_depth_gated_block_impl(
        graph,
        input_id,
        rng,
        weights,
        template_name="depth_gated_block",
        matmul_branch_prob=0.35,
    )


def tpl_depth_gated_block_matmul_stable(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → depth_gated_transform → {latent_attn || state_space} → merge → residual → FFN → residual.

    Combines depth-gating (variable processing depth per token) with the
    proven parallel attention+SSM mixing pattern. The depth_gated_transform
    produces gated features that are then processed by complementary
    attention (structure) and SSM (long-range) paths.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Depth gating: variable processing depth per token
    gated = _add(
        graph,
        "depth_gated_transform",
        [normed],
        {"out_dim": D},
        context="depth_gated_block_matmul_stable.gated",
    )

    # Path A: latent attention on gated features
    pa = _add(
        graph,
        "latent_attention_compressor",
        [gated],
        context="depth_gated_block_matmul_stable.latent_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="depth_gated_block_matmul_stable.attn_proj",
    )

    # Path B: state_space on gated features
    pb = _add(
        graph,
        "state_space",
        [gated],
        context="depth_gated_block_matmul_stable.ssm",
    )
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(
        graph, pa, pb, context="depth_gated_block_matmul_stable.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph,
        input_id,
        merged,
        context="depth_gated_block_matmul_stable.mid",
    )

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid,
        ffned,
        context="depth_gated_block_matmul_stable.output",
    )


def tpl_depth_gated_block_matmul_norm(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    return _tpl_depth_gated_block_impl(
        graph,
        input_id,
        rng,
        weights,
        template_name="depth_gated_block_matmul_norm",
        matmul_branch_prob=1.0,
        stabilize_matmul=True,
    )


def tpl_gated_lane_blend_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → gated_lane_blend → proj → FFN_motif → residual.

    Multi-lane routing: tokens are soft-routed to N parallel compute lanes
    based on learned difficulty scoring.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    n_lanes = rng.choice([2, 3, 4])
    routed = _add(
        graph,
        "gated_lane_blend",
        [normed],
        {"out_dim": D, "n_lanes": n_lanes},
        context="gated_lane_blend_block.routed",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, routed, rng, list(_FFN_CLASSES), weights
    )
    current = _instantiate_motif(graph, routed, ffn, rng) if ffn else routed
    current = _fix_dim(graph, current)

    return _residual(
        graph,
        input_id,
        current,
        context="gated_lane_blend_block.output",
    )


def tpl_feature_sparse_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → feature_sparsity → FFN_motif → proj → residual.

    Sparse feature selection before expensive computation.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    k = rng.choice([32, 64, 128])
    sparse = _add(
        graph,
        "feature_sparsity",
        [normed],
        {"k": k},
        context="feature_sparse_block.sparse",
    )
    sparse = _add(
        graph,
        "linear_proj",
        [sparse],
        {"out_dim": graph.model_dim},
        context="feature_sparse_block.sparse_bridge",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, sparse, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, sparse, ffn, rng) if ffn else sparse
    processed = _fix_dim(graph, processed)

    return _residual(
        graph,
        input_id,
        processed,
        context="feature_sparse_block.output",
    )


def tpl_topk_retrieval(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → cosine_similarity(proj, proj) → gather_topk → motif → residual.

    Retrieval-style: compute self-similarity scores, gather top-k
    vectors, process selected subset. Inspired by RAG reference arch.
    """
    D = graph.model_dim
    normed = _add(graph, "rmsnorm", [input_id], context="topk_retrieval.norm")

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="topk_retrieval.proj",
    )
    scores = _add(graph, "matmul", [proj, proj], context="topk_retrieval.scores")
    score_proj = _add(
        graph,
        "linear_proj",
        [scores],
        {"out_dim": D},
        context="topk_retrieval.score_proj",
    )
    gathered = _add(
        graph,
        "gather_topk",
        [normed, score_proj],
        {"k": rng.choice([4, 8, 16])},
        context="topk_retrieval.gathered",
    )

    # Process gathered subset
    processed = _add(
        graph,
        "swiglu_mlp",
        [gathered],
        {"mlp_ratio": rng.choice([2.0, 4.0])},
        context="topk_retrieval.ffn",
    )
    processed = _fix_dim(graph, processed)

    return _residual(
        graph,
        input_id,
        processed,
        context="topk_retrieval.output",
    )


# ── Adaptive Recursion Templates ────────────────────────────────────
# Data-mined from 11,447 programs: depth_weighted_proj (adaptive_recursion)
# drives 8 of the top 20 best 2-grams by mean_loss.


def tpl_adaptive_sparse(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → depth_weighted_proj → [sparse op] → gelu → residual add.

    Encodes the top 2-grams: adaptive_recursion → low_rank_proj (0.1344),
    adaptive_recursion → nm_sparse_linear (0.1909),
    adaptive_recursion → ternary_projection (0.2305).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    sparse_ops = [
        "nm_sparse_linear",
        "low_rank_proj",
        "block_sparse_linear",
        "ternary_projection",
    ]
    recursed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        context="adaptive_sparse.recursed",
    )
    sparse = _add(
        graph,
        rng.choice(sparse_ops),
        [recursed],
        context="adaptive_sparse.sparse",
    )
    activated = _add(
        graph,
        "gelu",
        [sparse],
        context="adaptive_sparse.activated",
    )

    processed = _fix_dim(graph, activated)
    return _residual(
        graph,
        input_id,
        processed,
        context="adaptive_sparse.output",
    )


def tpl_adaptive_conv_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → depth_weighted_proj → conv1d_seq → [FFN motif] → residual add.

    Encodes the highest-n confirmed 2-gram: adaptive_recursion → conv1d_seq
    (mean_loss=0.1934, n=72) and the strong 3-gram:
    adaptive_recursion → conv1d_seq → swiglu_mlp (0.1822).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    recursed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        context="adaptive_conv_ffn.recursed",
    )
    mid_norm = _add(
        graph,
        "rmsnorm",
        [recursed],
        context="adaptive_conv_ffn.mid_norm",
    )
    conved = _add(
        graph,
        "conv1d_seq",
        [mid_norm],
        context="adaptive_conv_ffn.conved",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, conved, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, conved, ffn, rng) if ffn else conved
    processed = _fix_dim(graph, processed)
    return _residual(
        graph,
        input_id,
        processed,
        context="adaptive_conv_ffn.output",
    )


def tpl_adaptive_ssm_chain(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → depth_weighted_proj → conv1d_seq → silu → selective_scan → gelu → residual add.

    The old scan → ternary path is now disallowed by the live context rules and
    showed up repeatedly in failure telemetry. Keep the adaptive-SSM intent, but
    force the proven conv → SiLU preconditioning before the scan and avoid the
    quantized projection immediately after it.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    recursed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        context="adaptive_ssm_chain.recursed",
    )
    conved = _add(
        graph,
        "conv1d_seq",
        [recursed],
        context="adaptive_ssm_chain.conv",
    )
    activated = _add(graph, "silu", [conved], context="adaptive_ssm_chain.silu")
    scanned = _add(
        graph,
        "selective_scan",
        [activated],
        context="adaptive_ssm_chain.scanned",
    )
    scanned = _add(
        graph,
        "gelu",
        [scanned],
        context="adaptive_ssm_chain.post_scan",
    )

    processed = _fix_dim(graph, scanned)
    return _residual(
        graph,
        input_id,
        processed,
        context="adaptive_ssm_chain.output",
    )


def tpl_adaptive_lane_recursion(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """difficulty_blend_3way(x,x) → add(input) → norm → depth_weighted_proj → [FFN] → residual add.

    Encodes the best 4-gram: adaptive_lane_mixer → add → rmsnorm →
    adaptive_recursion (0.1236). difficulty_blend_3way takes 2 inputs
    (routes tokens by difficulty across 3 lanes).
    """
    blended = _add(
        graph,
        "difficulty_blend_3way",
        [input_id, input_id],
        context="adaptive_lane_recursion.blended",
    )
    lane_merged = _residual(
        graph,
        input_id,
        blended,
        context="adaptive_lane_recursion.lane_merged",
    )

    norm = _pick_compatible_motif(graph, lane_merged, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, lane_merged, norm, rng) if norm else lane_merged

    recursed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        context="adaptive_lane_recursion.recursed",
    )
    recursed = _add(
        graph,
        "rmsnorm",
        [recursed],
        context="adaptive_lane_recursion.recursed_norm",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, recursed, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, recursed, ffn, rng) if ffn else recursed
    processed = _fix_dim(graph, processed)
    return _residual(
        graph,
        input_id,
        processed,
        context="adaptive_lane_recursion.output",
    )
