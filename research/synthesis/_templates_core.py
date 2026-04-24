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
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_SSM,
    MotifWeights,
    TemplateBuildError,
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
            if graph.nodes[processed].op_name == "linear_proj_down":
                processed = _add(
                    graph,
                    "rmsnorm",
                    [processed],
                    context="dense_cascade.post_down_norm",
                )
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

    # Pin mixer slot to attention only — the template name advertises a
    # transformer block, and broad _MIXER_CLASSES dilution (1/5 attention)
    # was producing graphs whose realized topology never reached
    # investigation tier despite passing s1.
    mixer = _pick_compatible_motif(graph, normed1, rng, MOTIF_CLASS_ATTENTION, weights)
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

    normed = _add(graph, "rmsnorm", [input_id], context="parallel_split.norm")
    split_a = _add(
        graph, "split2", [normed], {"part": 0}, context="parallel_split.split_a"
    )
    split_b = _add(
        graph, "split2", [normed], {"part": 1}, context="parallel_split.split_b"
    )
    path_a = _add(graph, "rmsnorm", [split_a], context="parallel_split.path_a_norm")
    path_a = _add(graph, "conv1d_seq", [path_a], context="parallel_split.path_a")
    path_b = _add(
        graph,
        "linear_proj",
        [split_b],
        {"out_dim": graph.nodes[split_b].output_shape.dim},
        context="parallel_split.path_b",
    )
    path_b = _add(graph, "gelu", [path_b], context="parallel_split.path_b_act")
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


# Per-lane safe op palettes for tpl_three_way_split. Curated to all work at
# D/3 dims (no must-divide-by-N attention head constraints, no must-be-power-of-2
# constraints). Picker rotates by rng to avoid the previous monoculture where
# every "three_way_split" graph realized to the same conv1d_seq/gelu/sigmoid arch.
_THREE_WAY_LANE0_OPS = (
    "conv1d_seq",
    "conv_only",
    "linear_attention",
    "gated_linear_attention",
    "rwkv_time_mixing",
)
_THREE_WAY_LANE1_OPS = (
    "fused_linear_gelu",
    "swiglu_mlp",
    "gated_linear",
    "linear_proj",  # falls back to linear+silu via the post-act below
)
_THREE_WAY_LANE2_OPS = (
    "learned_token_gate",
    "cheap_verify_blend",
    "depth_token_mask",
    "gated_delta",
    "rwkv_channel",
)


def tpl_three_way_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → route_lanes(3) → split3 → {distinct motif per lane} → merge → residual.

    Lane 0: sequence mixing (conv1d_seq / conv_only / linear_attention / gated_linear_attention / rwkv_time_mixing)
    Lane 1: channel mixing (fused_linear_gelu / swiglu_mlp / gated_linear / linear_proj+silu)
    Lane 2: gating / routing (learned_token_gate / cheap_verify_blend / depth_token_mask / gated_delta / rwkv_channel)

    Each lane operates at D/3 dims using ops verified compatible at sub-divisible widths.
    The 3-way merge is renormalized before injection into the residual stream so the
    Jacobian spectral norm stays in the investigation eligibility band.
    """
    D = graph.model_dim
    routed = _three_way_routed_input(graph, input_id)
    part0, part1, part2 = _three_way_parts(graph, routed)
    p0 = _three_way_lane0(graph, part0, rng)
    p1 = _three_way_lane1(graph, part1, rng)
    p2 = _three_way_lane2(graph, part2, rng)

    combined01 = _residual(graph, p0, p1, context="three_way_split.merge01")
    combined = _residual(graph, combined01, p2, context="three_way_split.merge")
    combined = _three_way_project_merge(graph, combined, D)
    return _residual(graph, input_id, combined, context="three_way_split.output")


def _three_way_routed_input(graph: ComputationGraph, input_id: int) -> int:
    normed = _add(graph, "rmsnorm", [input_id], context="three_way_split.norm")
    routed = _add(
        graph,
        "gated_lane_blend",
        [normed],
        {"n_lanes": 3},
        context="three_way_split.route",
    )

    if graph.nodes[routed].output_shape.dim % 3 != 0:
        routed = _add(
            graph,
            "linear_proj",
            [routed],
            {"out_dim": 255},
            context="three_way_split.reproject",
        )
    return routed


def _three_way_parts(graph: ComputationGraph, routed: int) -> tuple[int, int, int]:
    part0 = _add(
        graph, "split3", [routed], {"part": 0}, context="three_way_split.part0"
    )
    part1 = _add(
        graph, "split3", [routed], {"part": 1}, context="three_way_split.part1"
    )
    part2 = _add(
        graph, "split3", [routed], {"part": 2}, context="three_way_split.part2"
    )
    return part0, part1, part2


def _three_way_lane0(graph: ComputationGraph, part0: int, rng: random.Random) -> int:
    lane0_op = rng.choice(_THREE_WAY_LANE0_OPS)
    return _add(graph, lane0_op, [part0], context=f"three_way_split.lane0_{lane0_op}")


def _three_way_lane1(graph: ComputationGraph, part1: int, rng: random.Random) -> int:
    lane1_op = rng.choice(_THREE_WAY_LANE1_OPS)
    if lane1_op == "linear_proj":
        p1 = _add(
            graph,
            "linear_proj",
            [part1],
            {"out_dim": graph.nodes[part1].output_shape.dim},
            context="three_way_split.lane1_linear",
        )
        return _add(graph, "silu", [p1], context="three_way_split.lane1_act")
    return _add(graph, lane1_op, [part1], context=f"three_way_split.lane1_{lane1_op}")


def _three_way_lane2(graph: ComputationGraph, part2: int, rng: random.Random) -> int:
    lane2_op = rng.choice(_THREE_WAY_LANE2_OPS)
    if lane2_op == "depth_token_mask":
        return _three_way_depth_token_mask_lane(graph, part2)
    if lane2_op == "learned_token_gate":
        return _three_way_learned_token_gate_lane(graph, part2)
    return _add(graph, lane2_op, [part2], context=f"three_way_split.lane2_{lane2_op}")


def _three_way_depth_token_mask_lane(graph: ComputationGraph, part2: int) -> int:
    mask_input = _add(
        graph,
        "rmsnorm",
        [part2],
        context="three_way_split.lane2_depth_token_mask_pre_norm",
    )
    masked = _add(
        graph,
        "depth_token_mask",
        [mask_input],
        context="three_way_split.lane2_depth_token_mask",
    )
    projected = _add(
        graph,
        "linear_proj",
        [masked],
        {"out_dim": graph.nodes[part2].output_shape.dim},
        context="three_way_split.lane2_depth_token_mask_proj",
    )
    bypass = _residual(
        graph,
        mask_input,
        masked,
        context="three_way_split.lane2_depth_token_mask_bypass",
    )
    return _residual(
        graph,
        bypass,
        projected,
        context="three_way_split.lane2_depth_token_mask_merge",
    )


def _three_way_learned_token_gate_lane(graph: ComputationGraph, part2: int) -> int:
    gated = _add(
        graph,
        "learned_token_gate",
        [part2],
        context="three_way_split.lane2_learned_token_gate",
    )
    return _residual(
        graph,
        part2,
        gated,
        context="three_way_split.lane2_learned_token_gate_bypass",
    )


def _three_way_project_merge(graph: ComputationGraph, combined: int, dim: int) -> int:
    combined = _add(graph, "rmsnorm", [combined], context="three_way_split.merge_norm")
    return _add(
        graph,
        "linear_proj",
        [combined],
        {"out_dim": dim, "init_scale": 0.5},
        context="three_way_split.project",
    )


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

    bridge = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": graph.model_dim},
        context="moe.bridge",
    )
    moe_op = rng.choice(["moe_2expert", "moe_topk"])
    moe_config: dict = {}
    if moe_op == "moe_topk":
        moe_config = {"num_experts": rng.choice([2, 4]), "top_k": 1}
    routed = _add(graph, moe_op, [bridge], moe_config, context="moe.route")

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
    norm1 = _add(graph, "rmsnorm", [input_id], context="dense_cascade.norm1")
    stage1 = _add(graph, "conv1d_seq", [norm1], context="dense_cascade.stage1")
    mid1 = _residual(graph, input_id, stage1, context="dense_cascade.mid1")

    norm2 = _add(graph, "rmsnorm", [mid1], context="dense_cascade.norm2")
    stage2 = _add(
        graph,
        "swiglu_mlp",
        [norm2],
        {"mlp_ratio": rng.choice([2.0, 3.0])},
        context="dense_cascade.stage2",
    )
    stage2 = _fix_dim(graph, stage2)
    stage2 = _residual(graph, input_id, stage2, context="dense_cascade.dense_add")
    return _residual(graph, mid1, stage2, context="dense_cascade.output")


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

    # Insert a norm barrier between sparse/gate output and the up-projection
    # to break forbidden context rule (linear_proj_up → linear_proj_up).
    last_op = graph.nodes.get(processed)
    if last_op and last_op.op_name in ("linear_proj_up", "linear_proj"):
        processed = _add(
            graph, "rmsnorm", [processed], context="routed_bottleneck.pre_up_norm"
        )
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
    merge_bypass = _residual(
        graph,
        normed,
        merged,
        context="token_merge_block.merge_bypass",
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
    return _residual(graph, merge_bypass, processed, context="token_merge_block.output")


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
    merge_bypass = _residual(
        graph,
        normed,
        merged,
        context="token_merge_conv.merge_bypass",
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
    return _residual(graph, merge_bypass, sparse, context="token_merge_conv.output")


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


def tpl_recursive_attn_ssm_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → latent_attention_compressor → norm → state_space →
    depth_weighted_proj → [sparse] → activation → proj → residual_add.

    Combines the three strongest op families: latent attention (28.5% S1),
    state space (28.7% S1), and adaptive recursion (41.3% S1). Attention
    for pattern recognition, SSM for sequential state tracking, recursion
    for depth-adaptive refinement. Intermediate norms satisfy must_precede
    rules for LAC and state_space. Recursion is last because it cannot be
    followed by mixing ops (attention/SSM).
    """
    D = graph.model_dim
    norm_op = rng.choice(["rmsnorm", "layernorm"])

    # Pre-norm (satisfies LAC must_precede)
    normed = _add(
        graph,
        norm_op,
        [input_id],
        context="recursive_attn_ssm_hybrid.pre_norm",
    )

    # Latent attention compressor — best attention op
    compressed = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="recursive_attn_ssm_hybrid.lac",
    )

    # Norm before state_space (satisfies must_precede={rmsnorm, layernorm})
    mid_norm_op = rng.choice(["rmsnorm", "layernorm"])
    mid_norm = _add(
        graph,
        mid_norm_op,
        [compressed],
        context="recursive_attn_ssm_hybrid.mid_norm",
    )

    # State space — sequential state tracking
    ssm_out = _add(
        graph,
        "state_space",
        [mid_norm],
        context="recursive_attn_ssm_hybrid.ssm",
    )

    # Adaptive recursion (depth_weighted_proj) — depth-adaptive refinement
    # Must come after SSM; cannot be followed by mixing/attention ops
    max_depth = rng.choice([2, 3, 4])
    recursed = _add(
        graph,
        "depth_weighted_proj",
        [ssm_out],
        {"max_depth": max_depth},
        context="recursive_attn_ssm_hybrid.recursion",
    )

    # Optional sparse linear after recursion (50% chance)
    current = recursed
    if rng.random() < 0.5:
        sparse_op = rng.choice(
            [
                "nm_sparse_linear",
                "low_rank_proj",
                "ternary_projection",
            ]
        )
        current = _add(
            graph,
            sparse_op,
            [current],
            context="recursive_attn_ssm_hybrid.sparse",
        )

    # Activation for nonlinearity
    act_op = rng.choice(["gelu", "silu", "relu"])
    current = _add(
        graph,
        act_op,
        [current],
        context="recursive_attn_ssm_hybrid.activation",
    )

    # Final projection
    proj = _add(
        graph,
        "linear_proj",
        [current],
        {"out_dim": D},
        context="recursive_attn_ssm_hybrid.proj",
    )

    proj = _fix_dim(graph, proj)
    return _residual(graph, input_id, proj, context="recursive_attn_ssm_hybrid.output")


def tpl_induction_matmul_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → cumsum → matmul → norm → kronecker_linear → [activation] →
    proj → residual_add.

    Combines the ops with the strongest induction signal: cumsum (ind 0.024),
    matmul (ind 0.007), kronecker_linear (ind 0.006). Cumulative accumulation
    feeds into explicit matrix multiplication for proto-attention, then
    Kronecker-structured linear for efficient parameter sharing. Intermediate
    norms satisfy must_precede rules for kronecker_linear.
    """
    D = graph.model_dim
    norm_op = rng.choice(["rmsnorm", "layernorm"])

    # Pre-norm
    normed = _add(
        graph,
        norm_op,
        [input_id],
        context="induction_matmul_block.pre_norm",
    )

    # Cumulative sum — accumulation for positional/temporal signal
    accumulated = _add(
        graph,
        "cumsum",
        [normed],
        context="induction_matmul_block.cumsum",
    )
    stabilized = _add(
        graph,
        "rmsnorm",
        [accumulated],
        context="induction_matmul_block.post_cumsum_norm",
    )
    projected = _add(
        graph,
        "linear_proj",
        [stabilized],
        {"out_dim": D},
        context="induction_matmul_block.pre_matmul_proj",
    )

    # Matmul — explicit matrix ops for induction
    # matmul requires 2 inputs of same shape and prefers projection input.
    matmul_out = _add(
        graph,
        "matmul",
        [projected, projected],
        context="induction_matmul_block.matmul",
    )

    # Norm before kronecker_linear (satisfies must_precede={rmsnorm, layernorm})
    mid_norm_op = rng.choice(["rmsnorm", "layernorm"])
    mid_norm = _add(
        graph,
        mid_norm_op,
        [matmul_out],
        context="induction_matmul_block.mid_norm",
    )

    # Kronecker-structured linear — efficient structured projection
    kron = _add(
        graph,
        "kronecker_linear",
        [mid_norm],
        context="induction_matmul_block.kronecker",
    )

    # Activation for nonlinearity
    act_op = rng.choice(["gelu", "silu", "relu"])
    activated = _add(
        graph,
        act_op,
        [kron],
        context="induction_matmul_block.activation",
    )

    # Final projection
    proj = _add(
        graph,
        "linear_proj",
        [activated],
        {"out_dim": D},
        context="induction_matmul_block.proj",
    )

    proj = _fix_dim(graph, proj)
    return _residual(graph, input_id, proj, context="induction_matmul_block.output")


def tpl_recursive_moe_attn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → latent_attention_compressor → linear_proj → moe →
    depth_weighted_proj → [sparse] → activation → proj → residual_add.

    Combines MoE routing with recursion and attention: latent attention for
    pattern recognition, MoE for conditional specialization, adaptive
    recursion for depth. Linear proj between LAC and MoE satisfies MoE's
    forbidden_predecessors (no direct norm->MoE). Recursion last because it
    cannot precede mixing ops.
    """
    D = graph.model_dim
    norm_op = rng.choice(["rmsnorm", "layernorm"])

    # Pre-norm (satisfies LAC must_precede)
    normed = _add(
        graph,
        norm_op,
        [input_id],
        context="recursive_moe_attn.pre_norm",
    )

    # Latent attention compressor — pattern recognition
    compressed = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="recursive_moe_attn.lac",
    )

    # Linear proj — bridge between attention and MoE
    # moe_2expert/moe_topk forbid rmsnorm/layernorm as predecessor
    bridge = _add(
        graph,
        "linear_proj",
        [compressed],
        {"out_dim": D},
        context="recursive_moe_attn.bridge",
    )

    # MoE — conditional computation / specialization
    moe_op = rng.choice(["moe_2expert", "moe_topk"])
    moe_config: dict = {}
    if moe_op == "moe_topk":
        moe_config = {"num_experts": rng.choice([2, 4]), "top_k": 1}
    expert = _add(
        graph,
        moe_op,
        [bridge],
        moe_config,
        context="recursive_moe_attn.moe",
    )

    # Adaptive recursion (depth_weighted_proj) — depth refinement
    max_depth = rng.choice([2, 3, 4])
    recursed = _add(
        graph,
        "depth_weighted_proj",
        [expert],
        {"max_depth": max_depth},
        context="recursive_moe_attn.recursion",
    )

    # Optional sparse linear after recursion (50% chance)
    current = recursed
    if rng.random() < 0.5:
        sparse_op = rng.choice(
            [
                "nm_sparse_linear",
                "low_rank_proj",
                "ternary_projection",
            ]
        )
        current = _add(
            graph,
            sparse_op,
            [current],
            context="recursive_moe_attn.sparse",
        )

    # Activation for nonlinearity
    act_op = rng.choice(["gelu", "silu", "relu"])
    current = _add(
        graph,
        act_op,
        [current],
        context="recursive_moe_attn.activation",
    )

    # Final projection
    proj = _add(
        graph,
        "linear_proj",
        [current],
        {"out_dim": D},
        context="recursive_moe_attn.proj",
    )

    proj = _fix_dim(graph, proj)
    return _residual(graph, input_id, proj, context="recursive_moe_attn.output")
