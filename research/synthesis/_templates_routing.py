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
)
from ._templates_core import tpl_residual_block


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
    try:
        class_logits = graph.add_op(
            "token_class_proj", [normed], config={"n_classes": 4}
        )
        difficulty = graph.add_op("token_entropy", [class_logits])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Fast path: cheap linear projection (always runs on all tokens)
    try:
        fast_out = graph.add_op(
            "linear_proj", [normed], config={"out_dim": graph.model_dim}
        )
    except ValueError:
        fast_out = normed

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
    try:
        slow_weighted = graph.add_op("mul", [slow_out, difficulty])
    except ValueError:
        slow_weighted = slow_out

    # Merge: fast + difficulty-weighted slow
    try:
        merged = graph.add_op("add", [fast_out, slow_weighted])
    except ValueError:
        merged = slow_weighted

    # Residual
    try:
        return graph.add_op("add", [input_id, merged])
    except ValueError:
        return merged


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
    try:
        routed = graph.add_op("difficulty_blend_3way", [normed, normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    routed = _fix_dim(graph, routed)

    # Optional post-routing FFN for capacity
    ffn = _pick_compatible_motif_from_classes(graph, routed, rng, _FFN_CLASSES, weights)
    if ffn and rng.random() < 0.5:
        processed = _instantiate_motif(graph, routed, ffn, rng)
        processed = _fix_dim(graph, processed)
    else:
        processed = routed

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        # Difficulty scoring: token_type_classifier → entropy_score
        classified = graph.add_op("token_class_proj", [normed], config={"n_classes": 4})
        difficulty = graph.add_op("token_entropy", [classified])

        # Mixer: process input with difficulty weighting
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        weighted = graph.add_op("mul", [proj, difficulty])
        mixed = graph.add_op("linear_proj", [weighted], config={"out_dim": D})

        # Early exit: zeros easy tokens, hard tokens pass through
        exited = graph.add_op(
            "confidence_token_gate", [mixed], config={"threshold": 0.5}
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # FFN processes the output — easy tokens are zero so FFN work is
    # wasted on them (future: skip zero tokens for compute savings)
    ffn = _pick_compatible_motif_from_classes(
        graph, exited, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, exited, ffn, rng) if ffn else exited
    processed = _fix_dim(graph, processed)

    # Outer residual: recovers easy tokens' original representations
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    try:
        depth_routed = graph.add_op(
            "depth_weighted_proj", [normed], config={"max_depth": max_depth}
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

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

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        compressed = graph.add_op("latent_attention_compressor", [proj])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Inner residual (normed + compressed)
    try:
        inner_res = graph.add_op("add", [normed, compressed])
    except ValueError:
        inner_res = compressed

    # Sparse linear (nm_sparse or semi_structured)
    sparse_op = rng.choice(["nm_sparse_linear", "semi_structured_2_4_linear"])
    sparse_config: dict = {"out_dim": D}
    if sparse_op == "nm_sparse_linear":
        sparse_config.update({"n": 2, "m": 4})
    try:
        sparse = graph.add_op(sparse_op, [inner_res], config=sparse_config)
    except (ValueError, KeyError):
        sparse = inner_res

    # Activation
    act_op = rng.choice(["silu", "gelu", "relu"])
    try:
        activated = graph.add_op(act_op, [sparse])
    except ValueError:
        activated = sparse

    activated = _fix_dim(graph, activated)

    # Outer residual
    try:
        return graph.add_op("add", [input_id, activated])
    except ValueError:
        return activated


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

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        compressed = graph.add_op("latent_attention_compressor", [proj])
        inner_res = graph.add_op("add", [normed, compressed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

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
    try:
        sparse = graph.add_op(sparse_op, [inner_res], config=sparse_cfg)
    except (ValueError, KeyError):
        sparse = inner_res

    # Progressive compression gate (if available)
    try:
        gated = graph.add_op("adaptive_rank_gate", [sparse])
    except (ValueError, KeyError):
        gated = sparse

    # Post-norm + RWKV channel mixing
    norm2 = _pick_compatible_motif(graph, gated, rng, MOTIF_CLASS_NORM, weights)
    post_normed = _instantiate_motif(graph, gated, norm2, rng) if norm2 else gated

    try:
        mixed = graph.add_op(
            "rwkv_channel",
            [post_normed],
            config={"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        )
    except (ValueError, KeyError):
        mixed = post_normed

    mixed = _fix_dim(graph, mixed)

    try:
        return graph.add_op("add", [input_id, mixed])
    except ValueError:
        return mixed


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
    try:
        signal = graph.add_op(
            "token_class_proj",
            [normed],
            config={"n_classes": rng.choice([2, 3, 4])},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Route through compression op (2-input: data + signal)
    comp_op = rng.choice(["dual_compression_blend", "signal_conditioned_compression"])
    try:
        compressed = graph.add_op(comp_op, [normed, signal])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    compressed = _fix_dim(graph, compressed)

    # Optional moe_topk after compression (60% chance) — data mining shows
    # dual_compression_blend + moe_topk is the top underexplored high-signal combo
    if rng.random() < 0.6:
        try:
            compressed = graph.add_op(
                "moe_topk",
                [compressed],
                config={
                    "num_experts": rng.choice([2, 4]),
                    "top_k": rng.choice([1, 2]),
                },
            )
            compressed = _fix_dim(graph, compressed)
        except (ValueError, KeyError):
            pass

    try:
        return graph.add_op("add", [input_id, compressed])
    except ValueError:
        return compressed


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
    try:
        signal = graph.add_op(
            "token_class_proj",
            [normed],
            config={"n_classes": rng.choice([2, 3, 4])},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # 2-input compression: data + routing signal
    try:
        compressed = graph.add_op("dual_compression_blend", [normed, signal])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Expert routing
    try:
        routed = graph.add_op(
            "moe_topk",
            [compressed],
            config={
                "num_experts": rng.choice([2, 4]),
                "top_k": rng.choice([1, 2]),
            },
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Optional FFN motif (50% chance)
    if rng.random() < 0.5:
        ffn = _pick_compatible_motif_from_classes(
            graph, routed, rng, list(_FFN_CLASSES), weights
        )
        if ffn:
            routed = _instantiate_motif(graph, routed, ffn, rng)

    routed = _fix_dim(graph, routed)

    try:
        return graph.add_op("add", [input_id, routed])
    except ValueError:
        return routed


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

    try:
        signal = graph.add_op("token_class_proj", [normed], config={"n_classes": 4})
        compressed = graph.add_op("dual_compression_blend", [normed, signal])
        # Second norm between routing ops — key pattern from data mining
        mid_normed = graph.add_op("layernorm", [compressed])
        routed = graph.add_op(
            "moe_topk",
            [mid_normed],
            config={"num_experts": 4, "top_k": 2},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Optional FFN motif
    if rng.random() < 0.5:
        ffn = _pick_compatible_motif_from_classes(
            graph, routed, rng, list(_FFN_CLASSES), weights
        )
        if ffn:
            routed = _instantiate_motif(graph, routed, ffn, rng)

    routed = _fix_dim(graph, routed)

    try:
        return graph.add_op("add", [input_id, routed])
    except ValueError:
        return routed


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

    try:
        signal = graph.add_op(
            "token_class_proj",
            [normed],
            config={"n_classes": rng.choice([2, 3, 4])},
        )
        compressed = graph.add_op("signal_conditioned_compression", [normed, signal])
        routed = graph.add_op(
            "moe_topk",
            [compressed],
            config={
                "num_experts": rng.choice([2, 4]),
                "top_k": 1,
            },
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    routed = _fix_dim(graph, routed)

    try:
        return graph.add_op("add", [input_id, routed])
    except ValueError:
        return routed


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
    try:
        scores = graph.add_op(
            "token_class_proj",
            [normed],
            config={"n_classes": rng.choice([3, 4, 5])},
        )
        gated = graph.add_op(
            "score_depth_blend",
            [normed, scores],
            config={"max_depth": rng.choice([2, 3, 4])},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Post-routing motif
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

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        gated = graph.add_op("depth_gated_transform", [normed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Exclude MATH_SPACE motifs: depth_gated_transform is a gating op, and many
    # math_space ops (tropical_gate, etc.) are also gating ops — forbidden as
    # successors of depth_gated_transform by context rules.
    _DEPTH_GATED_MIXER = tuple(c for c in _MIXER_CLASSES if c != MOTIF_CLASS_MATH_SPACE)
    mixer = _pick_compatible_motif_from_classes(
        graph, gated, rng, _DEPTH_GATED_MIXER, weights
    )
    mixed = _instantiate_motif(graph, gated, mixer, rng) if mixer else gated
    mixed = _fix_dim(graph, mixed)

    try:
        return graph.add_op("add", [input_id, mixed])
    except ValueError:
        return mixed


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

    try:
        k = rng.choice([32, 64, 128])
        sparse = graph.add_op("feature_sparsity", [normed], config={"k": k})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, sparse, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, sparse, ffn, rng) if ffn else sparse
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        scores = graph.add_op("cosine_similarity", [normed, proj])
        gathered = graph.add_op(
            "gather_topk", [normed, scores], config={"k": rng.choice([4, 8, 16])}
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

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

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    try:
        recursed = graph.add_op("depth_weighted_proj", [normed])
        sparse = graph.add_op(rng.choice(sparse_ops), [recursed])
        activated = graph.add_op("gelu", [sparse])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    processed = _fix_dim(graph, activated)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        recursed = graph.add_op("depth_weighted_proj", [normed])
        mid_norm = graph.add_op("rmsnorm", [recursed])
        conved = graph.add_op("conv1d_seq", [mid_norm])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, conved, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, conved, ffn, rng) if ffn else conved
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        recursed = graph.add_op("depth_weighted_proj", [normed])
        mid_norm = graph.add_op("rmsnorm", [recursed])
        scanned = graph.add_op("selective_scan", [mid_norm])
        projected = graph.add_op("ternary_projection", [scanned])
        activated = graph.add_op("gelu", [projected])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    processed = _fix_dim(graph, activated)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    try:
        blended = graph.add_op("difficulty_blend_3way", [input_id, input_id])
        lane_merged = graph.add_op("add", [input_id, blended])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    norm = _pick_compatible_motif(graph, lane_merged, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, lane_merged, norm, rng) if norm else lane_merged

    try:
        recursed = graph.add_op("depth_weighted_proj", [normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, recursed, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, recursed, ffn, rng) if ffn else recursed
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed
