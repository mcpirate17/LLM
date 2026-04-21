"""Attention-first templates — 60% attention coverage target.

Groups:
  A: Forced-attention variants of existing high-performers
  B: Attention+FFN blocks with specific attention ops
  C: Hybrid attention+X parallel templates
  D: Attention paired with exotic/routing ops
"""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE,
    MotifWeights,
    TemplateBuildError,
    _FFN_CLASSES,
    _SPARSE_FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _tpl_attention_ffn_block,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._selection_utils import with_local_wildcard_probability


def _pick_with_local_wildcard(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_classes,
    weights: MotifWeights = None,
    *,
    wildcard_prob: float,
):
    return with_local_wildcard_probability(
        graph,
        lambda: _pick_compatible_motif(graph, node_id, rng, motif_classes, weights),
        wildcard_prob=wildcard_prob,
    )


# ═══════════════════════════════════════════════════════════════════════
# Group A: Forced-Attention Variants of Existing High-Performers
# ═══════════════════════════════════════════════════════════════════════


def tpl_attn_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention motif] → residual_add.

    Forced-attention variant of residual_block. Eliminates the 12-class
    lottery — guarantees an attention motif in the core slot.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn is None:
        raise TemplateBuildError(
            "attn_residual_block requires a compatible attention motif"
        )
    out = _instantiate_motif(graph, normed, attn, rng)
    out = _fix_dim(graph, out)
    return _residual(graph, input_id, out, context="attn_residual_block.output")


def tpl_attn_gated_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention motif] → gate → residual_add.

    Forced-attention variant of gated_residual. Attention + learned gate.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn is None:
        raise TemplateBuildError(
            "attn_gated_residual requires a compatible attention motif"
        )
    processed = _instantiate_motif(graph, normed, attn, rng)
    processed = _fix_dim(graph, processed)

    gate = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_GATE, weights)
    if gate:
        gated = _instantiate_motif(graph, processed, gate, rng)
        gated = _fix_dim(graph, gated)
    else:
        gated = processed

    return _residual(graph, input_id, gated, context="attn_gated_residual.output")


def tpl_attn_three_way_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → route(3) → split3 → {attention | FFN | gate} → concat → residual.

    Forced-attention variant of three_way_split with lane-local processing only.
    Avoids the old up/down lane scaffold that repeatedly produced illegal
    projection pairs inside attention and gate motifs.
    """
    normed = _add(graph, "rmsnorm", [input_id], context="attn_three_way_split.norm")

    routed = _add(
        graph,
        "gated_lane_blend",
        [normed],
        {"n_lanes": 3},
        context="attn_three_way_split.route",
    )
    shape = graph.nodes[routed].output_shape
    if shape.dim % 3 != 0:
        target_dim = max(24, (shape.dim // 3) * 3)
        routed = _add(
            graph,
            "linear_proj",
            [routed],
            {"out_dim": target_dim},
            context="attn_three_way_split.reproject",
        )

    part0 = _add(
        graph, "split3", [routed], {"part": 0}, context="attn_three_way_split.part0"
    )
    part1 = _add(
        graph, "split3", [routed], {"part": 1}, context="attn_three_way_split.part1"
    )
    part2 = _add(
        graph, "split3", [routed], {"part": 2}, context="attn_three_way_split.part2"
    )

    lane0 = _add(
        graph,
        "linear_proj",
        [part0],
        {"out_dim": graph.model_dim},
        context="attn_three_way_split.attn_up",
    )
    lane0 = _add(
        graph, "softmax_attention", [lane0], context="attn_three_way_split.attn_lane"
    )
    lane0 = _add(graph, "rmsnorm", [lane0], context="attn_three_way_split.attn_norm")
    p0 = _add(
        graph,
        "linear_proj",
        [lane0],
        {"out_dim": graph.nodes[part0].output_shape.dim},
        context="attn_three_way_split.attn_down",
    )

    p1 = _add(
        graph,
        "linear_proj",
        [part1],
        {"out_dim": graph.nodes[part1].output_shape.dim},
        context="attn_three_way_split.ffn_lane",
    )
    p1 = _add(graph, "gelu", [p1], context="attn_three_way_split.ffn_act")

    p2 = _add(
        graph,
        "linear_proj",
        [part2],
        {"out_dim": graph.nodes[part2].output_shape.dim},
        context="attn_three_way_split.gate_lane",
    )
    p2 = _add(graph, "sigmoid", [p2], context="attn_three_way_split.gate_act")

    combined01 = _residual(graph, p0, p1, context="attn_three_way_split.merge01")
    combined = _residual(graph, combined01, p2, context="attn_three_way_split.merge")
    combined = _add(
        graph,
        "linear_proj",
        [combined],
        {"out_dim": graph.model_dim},
        context="attn_three_way_split.project",
    )
    return _residual(graph, input_id, combined, context="attn_three_way_split.output")


def tpl_attn_dense_cascade(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → dense_add → norm → motif → dense_add → ... → residual.

    Forced-attention variant of dense_cascade. First stage is always attention,
    remaining stages pick from all classes.
    """
    from ._template_helpers import _ALL_CLASSES

    outputs = [input_id]

    for i in range(3):
        prev = outputs[-1]
        norm = _pick_compatible_motif(graph, prev, rng, MOTIF_CLASS_NORM, weights)
        normed = _instantiate_motif(graph, prev, norm, rng) if norm else prev

        if i == 0:
            # First stage: forced attention
            motif = _pick_compatible_motif(
                graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
            )
        else:
            motif = _pick_compatible_motif_from_classes(
                graph, normed, rng, _ALL_CLASSES, weights
            )

        if motif:
            processed = _instantiate_motif(graph, normed, motif, rng)
            processed = _fix_dim(graph, processed)
        else:
            processed = normed

        if i > 0 and processed != outputs[0]:
            processed = _residual(
                graph, outputs[0], processed, context="attn_dense_cascade.dense_add"
            )
        outputs.append(processed)

    result = outputs[-1]
    if result != input_id:
        return _residual(graph, input_id, result, context="attn_dense_cascade.output")
    return result


def tpl_attn_conditional_compute(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention motif] → gate → FFN → residual.

    Forced-attention variant of conditional_compute.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn is None:
        raise TemplateBuildError(
            "attn_conditional_compute requires a compatible attention motif"
        )
    mixed = _instantiate_motif(graph, normed, attn, rng)
    mixed = _fix_dim(graph, mixed)

    gate = _pick_compatible_motif(graph, mixed, rng, MOTIF_CLASS_GATE, weights)
    if gate:
        gated = _instantiate_motif(graph, mixed, gate, rng)
        gated = _fix_dim(graph, gated)
    else:
        gated = mixed

    ffn = _pick_compatible_motif_from_classes(graph, gated, rng, _FFN_CLASSES, weights)
    processed = _instantiate_motif(graph, gated, ffn, rng) if ffn else gated
    processed = _fix_dim(graph, processed)

    return _residual(
        graph, input_id, processed, context="attn_conditional_compute.output"
    )


def tpl_attn_cross_dim(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → transpose_sd → FFN → transpose_sd → residual.

    Forced-attention cross-dim mixer: attention on sequence dim,
    FFN on transposed (channel) dim.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_cross_dim.attn",
    )
    attended = _fix_dim(graph, attended)

    transposed = _add(
        graph, "transpose_sd", [attended], context="attn_cross_dim.transpose_in"
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, transposed, rng, _FFN_CLASSES, weights
    )
    processed = _instantiate_motif(graph, transposed, ffn, rng) if ffn else transposed
    processed = _fix_dim(graph, processed)

    processed = _add(
        graph, "transpose_sd", [processed], context="attn_cross_dim.transpose_out"
    )
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="attn_cross_dim.output")


def tpl_attn_multi_head_mix(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → multi_head_mix → attention → proj → residual.

    Multi-head mixing with forced attention after the head split.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    heads = _add(
        graph,
        "multi_head_mix",
        [normed],
        {"n_heads": rng.choice([2, 4, 8])},
        context="attn_multi_head_mix.heads",
    )

    attn = _pick_compatible_motif(graph, heads, rng, MOTIF_CLASS_ATTENTION, weights)
    processed = _instantiate_motif(graph, heads, attn, rng) if attn else heads
    processed = _fix_dim(graph, processed)

    return _residual(graph, input_id, processed, context="attn_multi_head_mix.output")


# ═══════════════════════════════════════════════════════════════════════
# Group B: Attention+FFN Block Variants with Specific Attention Ops
# ═══════════════════════════════════════════════════════════════════════


def tpl_local_attn_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → add → norm → FFN → add.

    Second-best attention op (27.5% S1) with standard FFN.
    """
    D = graph.model_dim
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    return _tpl_attention_ffn_block(
        graph,
        input_id,
        rng,
        weights,
        attn_op="local_window_attn",
        attn_config={"window_size": rng.choice(choices)},
        ffn_classes=(
            MOTIF_CLASS_SPARSE,
            MOTIF_CLASS_EFFICIENT_PROJ,
            MOTIF_CLASS_GUARDED_ACT,
        ),
    )


def tpl_local_attn_swiglu(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → add → norm → swiglu_mlp → add.

    Local attention + fixed SwiGLU FFN (proven combo from rwkv_double_norm pattern).
    """
    D = graph.model_dim
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attended = _add(
        graph,
        "local_window_attn",
        [normed1],
        {"window_size": rng.choice(choices)},
        context="local_attn_swiglu.attended",
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="local_attn_swiglu.project",
    )
    attended = _fix_dim(graph, attended)

    mid = _residual(graph, input_id, attended, context="local_attn_swiglu.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="local_attn_swiglu.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="local_attn_swiglu.output")


def tpl_diff_attn_gated_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → diff_attention → add → norm → gate/guarded_act FFN → add.

    Differential attention + gated FFN for noise suppression.
    """
    _GATED_FFN = (MOTIF_CLASS_GATE, MOTIF_CLASS_GUARDED_ACT)
    return _tpl_attention_ffn_block(
        graph, input_id, rng, weights, attn_op="diff_attention", ffn_classes=_GATED_FFN
    )


# ═══════════════════════════════════════════════════════════════════════
# Group C: Hybrid Attention + X Templates (parallel paths)
# ═══════════════════════════════════════════════════════════════════════


def tpl_attn_ssm_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention | SSM} → add → norm → FFN → add.

    Jamba-style hybrid with forced attention path + SSM path.
    Like hybrid_parallel but adds an FFN sub-block.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    path_attn = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_ssm_hybrid.attn",
    )
    path_attn = _fix_dim(graph, path_attn)

    path_ssm = _add(
        graph,
        "state_space",
        [normed],
        context="attn_ssm_hybrid.ssm",
    )
    path_ssm = _fix_dim(graph, path_ssm)

    merged = _residual(graph, path_attn, path_ssm, context="attn_ssm_hybrid.merge")
    merged = _fix_dim(graph, merged)

    mid = _residual(graph, input_id, merged, context="attn_ssm_hybrid.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="attn_ssm_hybrid.ffn",
    )
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="attn_ssm_hybrid.output")


def tpl_attn_conv_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention | conv} → add → norm → FFN → add.

    Global (attention) + local (conv) receptive fields in parallel.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    path_attn = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    path_attn = _fix_dim(graph, path_attn)

    conv = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_CONV, weights)
    path_conv = _instantiate_motif(graph, normed, conv, rng) if conv else normed
    path_conv = _fix_dim(graph, path_conv)

    merged = _residual(graph, path_attn, path_conv, context="attn_conv_hybrid.merge")
    merged = _fix_dim(graph, merged)

    mid = _residual(graph, input_id, merged, context="attn_conv_hybrid.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="attn_conv_hybrid.output")


def tpl_attn_rwkv_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """layernorm → {projected attention | rwkv_channel} → add → layernorm → swiglu_mlp → add.

    Keep the proven RWKV double-norm backbone and treat attention as a bounded
    refinement path instead of replacing the pre/post-norm scaffold.
    """
    D = graph.model_dim
    normed = _add(graph, "layernorm", [input_id], context="attn_rwkv_hybrid.norm")

    path_attn = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_rwkv_hybrid.attn",
    )
    path_attn = _fix_dim(graph, path_attn)
    path_attn = _add(
        graph,
        "rmsnorm",
        [path_attn],
        context="attn_rwkv_hybrid.attn_norm",
    )
    path_attn = _add(
        graph,
        "linear_proj",
        [path_attn],
        {"out_dim": D},
        context="attn_rwkv_hybrid.attn_proj",
    )
    path_attn = _fix_dim(graph, path_attn)

    path_rwkv = _add(
        graph,
        "rwkv_channel",
        [normed],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="attn_rwkv_hybrid.rwkv",
    )
    path_rwkv = _fix_dim(graph, path_rwkv)

    merged = _residual(graph, path_attn, path_rwkv, context="attn_rwkv_hybrid.merge")
    merged = _fix_dim(graph, merged)

    mid = _residual(graph, input_id, merged, context="attn_rwkv_hybrid.mid")
    normed2 = _add(graph, "layernorm", [mid], context="attn_rwkv_hybrid.norm2")
    if rng.random() < 0.5:
        normed2 = _add(
            graph,
            "conv1d_seq",
            [normed2],
            context="attn_rwkv_hybrid.channel_refine",
        )
        normed2 = _fix_dim(graph, normed2)

    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="attn_rwkv_hybrid.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="attn_rwkv_hybrid.output")


def tpl_attn_bottleneck_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel attention + SSM hybrid with full-width FFN. Replaces the
    original sparse bottleneck (D→D//2→sparse→D) which killed gradient
    flow through 50% of dimensions.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Path A: attention
    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    pa = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    pa = _fix_dim(graph, pa)

    # Path B: state_space (safe SSM op, no context constraints)
    pb = _add(
        graph,
        "state_space",
        [normed],
        context="attn_bottleneck_hybrid.ssm",
    )
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="attn_bottleneck_hybrid.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="attn_bottleneck_hybrid.mid")

    # Full-width FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 3.0])},
        context="attn_bottleneck_hybrid.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="attn_bottleneck_hybrid.output")


def tpl_attn_routing_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → softmax_attn → add → route/depth → swiglu → residual.

    Grammar-stable routing redesign: fixed attention scaffold into routing,
    then a dense FFN tail. This remains weak, but it is the strongest routing
    variant so far that still survives targeted generation.
    """
    D = graph.model_dim
    norm = _pick_with_local_wildcard(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_routing_block.attended",
    )
    attended = _add(
        graph,
        "rmsnorm",
        [attended],
        context="attn_routing_block.attn_norm",
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_routing_block.project",
    )
    attended = _fix_dim(graph, attended)
    mid = _residual(graph, input_id, attended, context="attn_routing_block.mid")

    blended = _add(
        graph,
        "difficulty_blend_3way",
        [mid, mid],
        context="attn_routing_block.blended",
    )
    lane_merged = _residual(
        graph,
        mid,
        blended,
        context="attn_routing_block.lane_merged",
    )

    norm2 = _pick_with_local_wildcard(
        graph,
        lane_merged,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed2 = (
        _instantiate_motif(graph, lane_merged, norm2, rng) if norm2 else lane_merged
    )
    routed = _add(
        graph,
        "depth_weighted_proj",
        [normed2],
        context="attn_routing_block.routed",
    )
    routed = _add(graph, "rmsnorm", [routed], context="attn_routing_block.route_norm")
    mid2 = _residual(graph, lane_merged, routed, context="attn_routing_block.mid2")

    processed = _add(
        graph,
        "swiglu_mlp",
        [mid2],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="attn_routing_block.ffn",
    )
    processed = _fix_dim(graph, processed)

    return _residual(
        graph,
        mid2,
        processed,
        context="attn_routing_block.output",
    )


def tpl_dual_attn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attn_A → add → norm → attn_B → add.

    Two different attention types stacked. Designed for diversity —
    the motif picker will likely select different attention variants
    for each slot.
    """
    # First attention sub-block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attn1 = _pick_compatible_motif(graph, normed1, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn1 is None:
        raise TemplateBuildError("dual_attn_block first stage requires attention")
    out1 = _instantiate_motif(graph, normed1, attn1, rng)
    out1 = _fix_dim(graph, out1)
    mid = _residual(graph, input_id, out1, context="dual_attn_block.mid")

    # Second attention sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    attn2 = _pick_compatible_motif(graph, normed2, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn2 is None:
        raise TemplateBuildError("dual_attn_block second stage requires attention")
    out2 = _instantiate_motif(graph, normed2, attn2, rng)
    out2 = _fix_dim(graph, out2)
    return _residual(graph, mid, out2, context="dual_attn_block.output")


def tpl_attn_state_space_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention | state_space} → add → norm → FFN → add.

    Attention + SSM in parallel paths with FFN sub-block.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    path_attn = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    path_attn = _fix_dim(graph, path_attn)

    path_ssm = _add(
        graph, "state_space", [normed], context="attn_state_space_hybrid.state_space"
    )
    path_ssm = _add(
        graph,
        "linear_proj",
        [path_ssm],
        {"out_dim": D},
        context="attn_state_space_hybrid.project",
    )
    merged = _residual(
        graph, path_attn, path_ssm, context="attn_state_space_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="attn_state_space_hybrid.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="attn_state_space_hybrid.output")


def tpl_cascaded_attn_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attn → add → norm → attn → add → norm → FFN → add.

    Deep attention: two attention layers + one FFN. For models that need
    more attention depth per block.
    """
    # First attention
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    out1 = _add(
        graph,
        "softmax_attention",
        [normed1],
        context="cascaded_attn_ffn.attn1",
    )
    out1 = _fix_dim(graph, out1)
    mid1 = _residual(graph, input_id, out1, context="cascaded_attn_ffn.mid1")

    # Second attention
    norm2 = _pick_compatible_motif(graph, mid1, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    out2 = _add(
        graph,
        "softmax_attention",
        [normed2],
        context="cascaded_attn_ffn.attn2",
    )
    out2 = _fix_dim(graph, out2)
    mid2 = _residual(graph, mid1, out2, context="cascaded_attn_ffn.mid2")

    # FFN
    norm3 = _pick_compatible_motif(graph, mid2, rng, MOTIF_CLASS_NORM, weights)
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffn = _pick_compatible_motif_from_classes(
        graph, normed3, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed3, ffn, rng) if ffn else normed3
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid2, ffned, context="cascaded_attn_ffn.output")


# ═══════════════════════════════════════════════════════════════════════
# Group D: Attention Paired with Exotic/Routing/Activation Ops
# ═══════════════════════════════════════════════════════════════════════


def tpl_attn_exp_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → exp → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)

    proj = _add(
        graph, "linear_proj", [attended], {"out_dim": D}, context="attn_exp_gated.proj"
    )
    bounded = _add(graph, "sigmoid", [proj], context="attn_exp_gated.bounded")
    gated = _add(graph, "exp", [bounded], context="attn_exp_gated.gated")
    gated = _add(graph, "rmsnorm", [gated], context="attn_exp_gated.stabilized")
    gated = _fix_dim(graph, gated)
    return _residual(graph, input_id, gated, context="attn_exp_gated.output")


def tpl_attn_gated_product(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → proj_a ⊗ sigmoid(proj_b) → residual.

    Attention + gated product for feature selection.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context="attn_gated_product.norm")
    proj_a = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_product.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_product.proj_b",
    )
    gate = _add(graph, "sigmoid", [proj_b], context="attn_gated_product.gate")
    gated = _add(graph, "mul", [proj_a, gate], context="attn_gated_product.gated")
    gated = _fix_dim(graph, gated)
    return _residual(graph, input_id, gated, context="attn_gated_product.output")


def tpl_diff_attn_routing(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → diff_attention → depth_weighted_proj → FFN → residual.

    Differential attention + depth-aware routing.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph, "diff_attention", [normed], context="diff_attn_routing.attended"
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="diff_attn_routing.project",
    )
    routed = _add(
        graph, "depth_weighted_proj", [attended], context="diff_attn_routing.routed"
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, routed, rng, _SPARSE_FFN_CLASSES, weights
    )
    processed = _instantiate_motif(graph, routed, ffn, rng) if ffn else routed
    processed = _fix_dim(graph, processed)

    return _residual(graph, input_id, processed, context="diff_attn_routing.output")


def tpl_local_attn_routing(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → depth_weighted_proj → FFN → residual.

    Local attention + depth-aware routing.
    """
    D = graph.model_dim
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph,
        "local_window_attn",
        [normed],
        {"window_size": rng.choice(choices)},
        context="local_attn_routing.attended",
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="local_attn_routing.project",
    )
    routed = _add(
        graph, "depth_weighted_proj", [attended], context="local_attn_routing.routed"
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, routed, rng, _SPARSE_FFN_CLASSES, weights
    )
    processed = _instantiate_motif(graph, routed, ffn, rng) if ffn else routed
    processed = _fix_dim(graph, processed)

    return _residual(graph, input_id, processed, context="local_attn_routing.output")


def tpl_attn_chebyshev_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention | chebyshev} → add → proj → residual.

    Attention + spectral (Chebyshev) paths in parallel.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    path_attn = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    path_attn = _fix_dim(graph, path_attn)

    path_cheb = _add(
        graph,
        "chebyshev_spectral_mix",
        [normed],
        context="attn_chebyshev_hybrid.cheb",
    )
    path_cheb = _add(
        graph,
        "linear_proj",
        [path_cheb],
        {"out_dim": D},
        context="attn_chebyshev_hybrid.project",
    )
    merged = _residual(
        graph, path_attn, path_cheb, context="attn_chebyshev_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    return _residual(graph, input_id, merged, context="attn_chebyshev_hybrid.output")


def tpl_attn_sparse_moe(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → add → norm → sparse → MoE → add.

    Attention sub-block + sparse→MoE sub-block.
    """
    from ._template_helpers import MOTIF_CLASS_SPARSE

    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attn = _pick_compatible_motif(graph, normed1, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed1, attn, rng) if attn else normed1
    attended = _fix_dim(graph, attended)

    mid = _residual(graph, input_id, attended, context="attn_sparse_moe.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    sparse = _pick_compatible_motif(graph, normed2, rng, MOTIF_CLASS_SPARSE, weights)
    processed = _instantiate_motif(graph, normed2, sparse, rng) if sparse else normed2
    processed = _fix_dim(graph, processed)

    moe = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_MOE, weights)
    if moe:
        processed = _instantiate_motif(graph, processed, moe, rng)
        processed = _fix_dim(graph, processed)

    return _residual(graph, mid, processed, context="attn_sparse_moe.output")


# ═══════════════════════════════════════════════════════════════════════
# Shared helper for Group D "attention → single_op → proj → residual"
# ═══════════════════════════════════════════════════════════════════════


# Group E/F and generated wrapper variants live in _templates_attention_tail.py.
