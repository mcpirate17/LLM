"""Core workhorse templates — the most commonly used structural patterns."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_SSM,
    MotifWeights,
    TemplateBuildError,
    _ALL_CLASSES,
    _BOTTLENECK_CLASSES,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _shuffle_wrap,
    template_add_op as _add,
    template_add_residual as _residual,
)


def tpl_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → residual_add.

    The workhorse template: pre-norm + any functional motif + skip.
    """
    # Pre-norm
    norm_motif = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    if norm_motif:
        normed = _instantiate_motif(graph, input_id, norm_motif, rng)
    else:
        normed = input_id

    # Core motif (mixer or FFN)
    core_classes = list(_MIXER_CLASSES + _FFN_CLASSES)
    core_motif = _pick_compatible_motif_from_classes(
        graph, normed, rng, core_classes, weights
    )
    if core_motif:
        processed = _instantiate_motif(graph, normed, core_motif, rng)
    else:
        processed = normed

    # Fix dim and add residual
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="residual_block.output")


def tpl_sequential(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → add → norm → motif → add → ... → residual_add.

    Stack 2-3 mini residual blocks in sequence with pre-norm.
    """
    n_motifs = 2
    current = input_id
    for _ in range(n_motifs):
        # Pre-norm before each motif
        norm = _pick_compatible_motif(graph, current, rng, MOTIF_CLASS_NORM, weights)
        normed = _instantiate_motif(graph, current, norm, rng) if norm else current

        motif = _pick_compatible_motif_from_classes(
            graph,
            normed,
            rng,
            (
                MOTIF_CLASS_ATTENTION,
                MOTIF_CLASS_CONV,
                MOTIF_CLASS_FFN,
                MOTIF_CLASS_EFFICIENT_PROJ,
            ),
            weights,
        )
        if motif:
            processed = _instantiate_motif(graph, normed, motif, rng)
            processed = _fix_dim(graph, processed)
            # Per-step residual
            current = _residual(
                graph, current, processed, context="sequential.step_residual"
            )
        # If no motif found, current stays unchanged

    # Outer residual back to original input
    if current != input_id:
        return _residual(graph, input_id, current, context="sequential.output")
    return current


def tpl_transformer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → mixer → add → norm → ffn → add.

    Classic pre-norm transformer block with any mixer + any FFN.
    """
    # Attention sub-block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # 50% chance of RoPE positional encoding before mixer
    if rng.random() < 0.5:
        normed1 = _add(
            graph, "rope_rotate", [normed1], context="transformer_block.rope"
        )

    mixer = _pick_compatible_motif_from_classes(
        graph, normed1, rng, _MIXER_CLASSES, weights
    )
    if mixer:
        mixed = _instantiate_motif(graph, normed1, mixer, rng)
    else:
        mixed = normed1
    mixed = _fix_dim(graph, mixed)

    mid = _residual(graph, input_id, mixed, context="transformer_block.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    if ffn:
        ffned = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        ffned = normed2
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="transformer_block.output")


def tpl_parallel_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → split → {norm → motif_a | norm → motif_b} → concat → project → residual.

    Width: parallel processing paths with different motifs, pre-normed.
    """
    shape = graph.nodes[input_id].output_shape
    if shape.dim < 16:
        raise TemplateBuildError("parallel_split requires input dim >= 16")

    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    split_a = _add(
        graph, "split2", [normed], {"part": 0}, context="parallel_split.split_a"
    )
    split_b = _add(
        graph, "split2", [normed], {"part": 1}, context="parallel_split.split_b"
    )

    # Per-lane norm + motif. split2 halves width, so both lanes must stay in
    # bottleneck-safe motif classes until concat. Letting a lane restore to
    # full width before the join is what created the asymmetric concat
    # scaffolds now rejected by validation.
    norm_a = _pick_compatible_motif(graph, split_a, rng, MOTIF_CLASS_NORM, weights)
    lane_a = _instantiate_motif(graph, split_a, norm_a, rng) if norm_a else split_a
    motif_a = _pick_compatible_motif_from_classes(
        graph, lane_a, rng, _BOTTLENECK_CLASSES, weights
    )
    path_a = _instantiate_motif(graph, lane_a, motif_a, rng) if motif_a else lane_a

    norm_b = _pick_compatible_motif(graph, split_b, rng, MOTIF_CLASS_NORM, weights)
    lane_b = _instantiate_motif(graph, split_b, norm_b, rng) if norm_b else split_b

    # 30% chance: channel-shuffle wrap around a bottleneck-safe path_b motif.
    path_b = _shuffle_wrap(graph, lane_b, rng, _BOTTLENECK_CLASSES, weights, prob=0.3)

    merged = _add(graph, "concat", [path_a, path_b], context="parallel_split.concat")

    merged = _fix_dim(graph, merged)
    return _residual(graph, input_id, merged, context="parallel_split.output")


def tpl_gated_maximum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """proj_a → proj_b → maximum(a,b) → linear_proj → residual.

    Element-wise maximum for winner-take-all feature selection.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj_a = _add(
        graph, "linear_proj", [normed], {"out_dim": D}, context="gated_maximum.proj_a"
    )
    proj_b = _add(
        graph, "linear_proj", [normed], {"out_dim": D}, context="gated_maximum.proj_b"
    )
    maxed = _add(graph, "maximum", [proj_a, proj_b], context="gated_maximum.maximum")
    out = _add(
        graph, "linear_proj", [maxed], {"out_dim": D}, context="gated_maximum.out"
    )
    return _residual(graph, input_id, out, context="gated_maximum.output")


def tpl_three_way_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → route_lanes(3) → split3 → {up → motif → down} per lane → concat → residual.

    Routed feature split with DISTINCT motif classes per lane.
    Each lane is up-projected to full D before its motif so attention/FFN/MoE
    operate at full width, then down-projected back to D/3 before concat.
    Lane 0: sequence mixing (attention/SSM/conv)
    Lane 1: channel mixing (FFN/gate/sparse)
    Lane 2: routing (MoE/gate/guarded activation)
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    routed = _add(
        graph,
        "gated_lane_blend",
        [normed],
        {"n_lanes": 3},
        context="three_way_split.route",
    )

    shape = graph.nodes[routed].output_shape
    if shape.dim % 3 != 0:
        target_dim = max(24, (shape.dim // 3) * 3)
        routed = _add(
            graph,
            "linear_proj",
            [routed],
            {"out_dim": target_dim},
            context="three_way_split.reproject",
        )

    part0 = _add(
        graph, "split3", [routed], {"part": 0}, context="three_way_split.part0"
    )
    part1 = _add(
        graph, "split3", [routed], {"part": 1}, context="three_way_split.part1"
    )
    part2 = _add(
        graph, "split3", [routed], {"part": 2}, context="three_way_split.part2"
    )

    lane0 = _add(
        graph,
        "linear_proj_up",
        [part0],
        {"out_dim": D},
        context="three_way_split.lane0",
    )
    lane1 = _add(
        graph,
        "linear_proj_up",
        [part1],
        {"out_dim": D},
        context="three_way_split.lane1",
    )
    lane2 = _add(
        graph,
        "linear_proj_up",
        [part2],
        {"out_dim": D},
        context="three_way_split.lane2",
    )

    m0 = _pick_compatible_motif_from_classes(graph, lane0, rng, _MIXER_CLASSES, weights)
    p0 = _instantiate_motif(graph, lane0, m0, rng) if m0 else lane0

    m1 = _pick_compatible_motif_from_classes(graph, lane1, rng, _FFN_CLASSES, weights)
    p1 = _instantiate_motif(graph, lane1, m1, rng) if m1 else lane1

    _GATE_CLASSES = (MOTIF_CLASS_MOE, MOTIF_CLASS_GATE, MOTIF_CLASS_GUARDED_ACT)
    m2 = _pick_compatible_motif_from_classes(graph, lane2, rng, _GATE_CLASSES, weights)
    p2 = _instantiate_motif(graph, lane2, m2, rng) if m2 else lane2

    lane_dim0 = D // 3
    lane_dim1 = D // 3
    lane_dim2 = D - lane_dim0 - lane_dim1
    p0 = _add(
        graph,
        "linear_proj_down",
        [p0],
        {"out_dim": lane_dim0},
        context="three_way_split.down0",
    )
    p1 = _add(
        graph,
        "linear_proj_down",
        [p1],
        {"out_dim": lane_dim1},
        context="three_way_split.down1",
    )
    p2 = _add(
        graph,
        "linear_proj_down",
        [p2],
        {"out_dim": lane_dim2},
        context="three_way_split.down2",
    )

    combined01 = _add(graph, "concat", [p0, p1], context="three_way_split.concat01")
    combined = _add(graph, "concat", [combined01, p2], context="three_way_split.concat")
    return _residual(graph, input_id, combined, context="three_way_split.output")


def tpl_bottleneck(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → project_down → motif → project_up → residual_add.

    Compression: information bottleneck with any core motif, pre-normed.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    down = _add(
        graph,
        "linear_proj_down",
        [normed],
        {"out_dim": D // 2},
        context="bottleneck.down",
    )

    core = _pick_compatible_motif_from_classes(
        graph, down, rng, _BOTTLENECK_CLASSES, weights
    )
    if core:
        processed = _instantiate_motif(graph, down, core, rng)
    else:
        processed = down

    up = _add(
        graph,
        "linear_proj_up",
        [processed],
        {"out_dim": D},
        context="bottleneck.up",
    )
    return _residual(graph, input_id, up, context="bottleneck.output")


def tpl_moe(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → moe_motif → residual_add.

    Sparsity: conditional computation via MoE.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    moe = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_MOE, weights)
    if moe is None:
        raise TemplateBuildError("moe template requires a compatible MoE motif")
    routed = _instantiate_motif(graph, normed, moe, rng)

    routed = _fix_dim(graph, routed)
    return _residual(graph, input_id, routed, context="moe.output")


def tpl_hybrid_parallel(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attn_path | ssm_path} → add(paths) → proj → residual.

    Hybrid: attention and SSM in parallel on full-width input (like Jamba).
    Both paths see full model_dim — no split2 dimension halving.
    """
    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Attention path (full dim)
    attn = _pick_compatible_motif_from_classes(
        graph, normed, rng, (MOTIF_CLASS_ATTENTION,), weights
    )
    path_attn = _instantiate_motif(graph, normed, attn, rng) if attn else normed

    # SSM/conv path (full dim), 30% chance with channel shuffle
    path_ssm = _shuffle_wrap(
        graph, normed, rng, (MOTIF_CLASS_SSM, MOTIF_CLASS_CONV), weights, prob=0.3
    )

    # Combine paths with learned gating
    merged = _residual(graph, path_attn, path_ssm, context="hybrid_parallel.merge")

    merged = _fix_dim(graph, merged)
    return _residual(graph, input_id, merged, context="hybrid_parallel.output")


def tpl_gated_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → gate → residual_add.

    Learned residual: adaptive skip weighting.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    core = _pick_compatible_motif_from_classes(
        graph, normed, rng, list(_MIXER_CLASSES + _FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed, core, rng) if core else normed
    processed = _fix_dim(graph, processed)

    # Gate
    gate = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_GATE, weights)
    if gate:
        gated = _instantiate_motif(graph, processed, gate, rng)
        gated = _fix_dim(graph, gated)
    else:
        gated = processed

    return _residual(graph, input_id, gated, context="gated_residual.output")


def tpl_dense_cascade(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → dense_add → norm → motif → dense_add → ... → residual.

    DenseNet-style: each motif receives all prior outputs, with pre-norm.
    """
    outputs = [input_id]

    for i in range(3):
        prev = outputs[-1]
        # Pre-norm before each motif
        norm = _pick_compatible_motif(graph, prev, rng, MOTIF_CLASS_NORM, weights)
        normed = _instantiate_motif(graph, prev, norm, rng) if norm else prev

        motif = _pick_compatible_motif_from_classes(
            graph, normed, rng, _ALL_CLASSES, weights
        )
        if motif:
            processed = _instantiate_motif(graph, normed, motif, rng)
            processed = _fix_dim(graph, processed)
        else:
            processed = normed

        # Dense skip: add to first available prior output
        if i > 0 and processed != outputs[0]:
            processed = _residual(
                graph, outputs[0], processed, context="dense_cascade.dense_add"
            )
        outputs.append(processed)

    # Outer residual
    result = outputs[-1]
    if result != input_id:
        return _residual(graph, input_id, result, context="dense_cascade.output")
    return result


def tpl_sparse_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [30% attention →] sparse_linear → activate → project → residual_add.

    Uses sparse linear ops (N:M, block, ternary) as the main projection.
    30% chance of prepending an attention sub-block for global context.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # 30% chance: attention sub-block before sparse
    current = normed
    if rng.random() < 0.3:
        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            attended = _instantiate_motif(graph, normed, attn, rng)
            attended = _fix_dim(graph, attended)
            current = _residual(
                graph, normed, attended, context="sparse_ffn.attn_residual"
            )

    sparse = _pick_compatible_motif(graph, current, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, current, sparse, rng)
    else:
        processed = current
    processed = _fix_dim(graph, processed)

    return _residual(graph, input_id, processed, context="sparse_ffn.output")


def tpl_sparse_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sparse_motif → moe_motif → residual_add.

    Compound efficiency: forces both sparse AND MoE ops structurally.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Sparse path
    sparse = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed

    # MoE routing
    moe = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_MOE, weights)
    if moe is None:
        raise TemplateBuildError("sparse_moe_block requires a compatible MoE motif")
    processed = _instantiate_motif(graph, processed, moe, rng)

    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="sparse_moe_block.output")


def tpl_routed_bottleneck(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → project_down(D/4) → route → sparse_core → project_up → residual_add.

    4x bottleneck + routing + sparse = compound efficiency, pre-normed.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    down = _add(
        graph,
        "linear_proj_down",
        [normed],
        {"out_dim": max(4, D // 4)},
        context="routed_bottleneck.down",
    )

    # Route op from gate class
    gate = _pick_compatible_motif(graph, down, rng, MOTIF_CLASS_GATE, weights)
    if gate:
        routed = _instantiate_motif(graph, down, gate, rng)
    else:
        routed = down

    # Sparse core
    sparse = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, routed, sparse, rng)
    else:
        processed = routed

    up = _add(
        graph,
        "linear_proj_up",
        [processed],
        {"out_dim": D},
        context="routed_bottleneck.up",
    )
    return _residual(graph, input_id, up, context="routed_bottleneck.output")


def tpl_token_merge_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → token_merge → rmsnorm → channel_op → act → [sparse] → add(normed).

    Data-mined scaffold: the top-3 token_merge graphs (loss_ratio < 0.05,
    all S1-passed) use token_merge → conv1d_seq/swiglu_mlp → gelu →
    nm_sparse_linear → add.

    Post-merge rmsnorm satisfies conv1d_seq.must_precede={rmsnorm,layernorm}.
    Residual skip from pre-merge normed satisfies REQUIRES_RESIDUAL_BYPASS.
    Activation before nm_sparse_linear avoids forbidden swiglu_mlp → nm_sparse
    context rule pair.  No SSM/attention/state_space post-merge.
    """
    D = graph.model_dim

    # Pre-merge norm
    normed = _add(graph, "rmsnorm", [input_id], context="token_merge_block.pre_norm")

    # Token merge (preserves seq_len via nearest-neighbor restore)
    merged = _add(
        graph,
        "adjacent_token_merge",
        [normed],
        context="token_merge_block.merge",
    )

    # Post-merge norm: satisfies conv1d_seq.must_precede and swiglu_mlp predecessor rules
    post_norm = _add(graph, "rmsnorm", [merged], context="token_merge_block.post_norm")

    # Channel op: conv1d_seq or swiglu_mlp (both confirmed high-perf)
    channel_op = rng.choice(["conv1d_seq", "swiglu_mlp"])
    if channel_op == "swiglu_mlp":
        processed = _add(
            graph,
            "swiglu_mlp",
            [post_norm],
            {"mlp_ratio": rng.choice([2.0, 3.0])},
            context="token_merge_block.channel",
        )
    else:
        processed = _add(
            graph, "conv1d_seq", [post_norm], context="token_merge_block.channel"
        )

    # Activation before sparse to break forbidden swiglu_mlp → nm_sparse_linear
    act_op = rng.choice(["gelu", "silu"])
    processed = _add(graph, act_op, [processed], context="token_merge_block.activation")

    # Optional sparse linear (50% chance — not all winning graphs have it)
    if rng.random() < 0.5:
        processed = _add(
            graph,
            "nm_sparse_linear",
            [processed],
            {"out_dim": D},
            context="token_merge_block.sparse",
        )

    processed = _fix_dim(graph, processed)
    # Residual from normed: satisfies REQUIRES_RESIDUAL_BYPASS for token_merge
    return _residual(graph, normed, processed, context="token_merge_block.output")


def tpl_token_merge_conv(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → token_merge → rmsnorm → conv1d_seq → swiglu_mlp → gelu → nm_sparse → add(normed).

    Most common confirmed high-performance variant from data mining.
    conv1d_seq then swiglu_mlp is the exact sequence found in the top-3
    graphs (loss_ratio 0.006–0.043, all S1-passed).

    Post-merge rmsnorm satisfies conv1d_seq.must_precede={rmsnorm,layernorm}.
    gelu before nm_sparse_linear avoids forbidden swiglu_mlp → nm_sparse pair.
    Residual from normed satisfies REQUIRES_RESIDUAL_BYPASS for token_merge.
    """
    D = graph.model_dim

    normed = _add(graph, "rmsnorm", [input_id], context="token_merge_conv.pre_norm")

    merged = _add(
        graph,
        "adjacent_token_merge",
        [normed],
        context="token_merge_conv.merge",
    )

    # Post-merge norm: satisfies conv1d_seq.must_precede
    post_norm = _add(graph, "rmsnorm", [merged], context="token_merge_conv.post_norm")

    # conv1d_seq → swiglu_mlp (exact winning sequence)
    conv = _add(graph, "conv1d_seq", [post_norm], context="token_merge_conv.conv")
    mlp = _add(
        graph,
        "swiglu_mlp",
        [conv],
        {"mlp_ratio": rng.choice([2.0, 3.0])},
        context="token_merge_conv.mlp",
    )

    # gelu BEFORE nm_sparse_linear: breaks forbidden swiglu_mlp → nm_sparse pair
    activated = _add(graph, "gelu", [mlp], context="token_merge_conv.activation")
    sparse = _add(
        graph,
        "nm_sparse_linear",
        [activated],
        {"out_dim": D},
        context="token_merge_conv.sparse",
    )

    sparse = _fix_dim(graph, sparse)
    # Residual from normed: satisfies REQUIRES_RESIDUAL_BYPASS for token_merge
    return _residual(graph, normed, sparse, context="token_merge_conv.output")


def tpl_conditional_compute(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → entropy_score → gate(sparse_core) → residual_add.

    token_type_classifier produces class logits, entropy_score measures their
    uncertainty as a (B,S,1) difficulty signal. Sparse core gated by entropy.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Classify → entropy: token_type_classifier (B,S,D)→(B,S,D) logits,
    # then entropy_score (B,S,D)→(B,S,1) difficulty signal.
    class_logits = _add(
        graph,
        "token_class_proj",
        [normed],
        {"n_classes": 4},
        context="conditional_compute.class_logits",
    )
    difficulty = _add(
        graph,
        "token_entropy",
        [class_logits],
        context="conditional_compute.difficulty",
    )

    # Sparse core: operates on full-dim normed input (NOT entropy output)
    sparse = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed
    processed = _fix_dim(graph, processed)

    # Gate by difficulty: mul broadcasts (B,S,D) * (B,S,1) → (B,S,D)
    gated = _add(
        graph, "mul", [processed, difficulty], context="conditional_compute.gated"
    )
    return _residual(graph, input_id, gated, context="conditional_compute.output")
