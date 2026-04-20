"""Routing-first gated/compression/2-input/adaptive templates — private split."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
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

    # Bound compounded variance from classifier→compression→moe stack
    # so the residual contribution stays in the spectral band.
    compressed = _add(
        graph,
        "rmsnorm",
        [compressed],
        context="signal_routed_compression.output_norm",
    )

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
    # Bound compounded variance from classifier→compression→moe stack.
    routed = _add(graph, "rmsnorm", [routed], context="dual_routing_stack.output_norm")

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
    merged = _residual(graph, pa, pb, context="depth_gated_block_matmul_stable.merge")
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
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
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
