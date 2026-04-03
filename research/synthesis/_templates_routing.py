"""Routing-first templates — mandatory routing structure."""

from __future__ import annotations

import random

from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_NORM,
    MotifWeights,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
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
    }
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
    routed = _add(
        graph,
        "moe_topk",
        [mid_normed],
        {"num_experts": 4, "top_k": 2},
        context="dual_routing_deep.moe",
    )

    # Optional FFN motif
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


def tpl_depth_gated_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → depth_gated_transform → mixer_motif → proj → residual.

    Depth gating: tokens get variable-depth processing based on learned difficulty.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    gated = _add(
        graph,
        "depth_gated_transform",
        [normed],
        {"out_dim": D},
        context="depth_gated_block.gated",
    )

    # Exclude MATH_SPACE motifs: depth_gated_transform is a gating op, and many
    # math_space ops (tropical_gate, etc.) are also gating ops — forbidden as
    # successors of depth_gated_transform by context rules.
    _DEPTH_GATED_MIXER = tuple(c for c in _MIXER_CLASSES if c != MOTIF_CLASS_MATH_SPACE)
    mixer = _pick_compatible_motif_from_classes(
        graph, gated, rng, _DEPTH_GATED_MIXER, weights
    )
    mixed = _instantiate_motif(graph, gated, mixer, rng) if mixer else gated
    mixed = _fix_dim(graph, mixed)

    return _residual(
        graph,
        input_id,
        mixed,
        context="depth_gated_block.output",
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="topk_retrieval.proj",
    )
    scores = _add(
        graph,
        "cosine_similarity",
        [normed, proj],
        context="topk_retrieval.scores",
    )
    gathered = _add(
        graph,
        "gather_topk",
        [normed, scores],
        {"k": rng.choice([4, 8, 16])},
        context="topk_retrieval.gathered",
    )

    # Process gathered subset
    core = _pick_compatible_motif_from_classes(
        graph,
        gathered,
        rng,
        list(_FFN_CLASSES),
        weights,
    )
    if core:
        processed = _instantiate_motif(graph, gathered, core, rng)
    else:
        processed = gathered
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
    """norm → depth_weighted_proj → selective_scan → ternary_projection → gelu → residual add.

    Encodes the best 4-gram: adaptive_recursion → rmsnorm → selective_scan →
    swiglu_mlp (0.0571). The norm is placed before adaptive_recursion per the
    3-gram data (rmsnorm → adaptive_recursion consistently outperforms raw).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    recursed = _add(
        graph,
        "depth_weighted_proj",
        [normed],
        context="adaptive_ssm_chain.recursed",
    )
    mid_norm = _add(
        graph,
        "rmsnorm",
        [recursed],
        context="adaptive_ssm_chain.mid_norm",
    )
    scanned = _add(
        graph,
        "selective_scan",
        [mid_norm],
        context="adaptive_ssm_chain.scanned",
    )
    projected = _add(
        graph,
        "ternary_projection",
        [scanned],
        context="adaptive_ssm_chain.projected",
    )
    activated = _add(
        graph,
        "gelu",
        [projected],
        context="adaptive_ssm_chain.activated",
    )

    processed = _fix_dim(graph, activated)
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
