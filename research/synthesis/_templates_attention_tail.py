"""Attention template tail: specialized op wrappers and generated variants."""

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
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MotifWeights,
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


def _tpl_attn_op_chain(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    post_op: str,
    post_config: dict | None = None,
) -> int:
    """norm → [attention motif] → rmsnorm → post_op → proj → residual_add."""
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context=f"{post_op}.attn_norm")
    processed = _add(
        graph,
        post_op,
        [attended],
        post_config or {},
        context=f"{post_op}.post_op",
    )
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context=f"{post_op}.output")


def tpl_attn_decay_sequence(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm -> attention -> gated decay branch -> value gating -> [FFN] -> residual.

    Unlike generic unary post-ops, ``cumprod_safe`` only behaves like a stable
    decay mask when it consumes bounded values. Preserve the attention front-end
    but rebuild the guarded ``sigmoid -> cumprod_safe -> mul(value, decay)``
    structure used by the base decay template.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(
        graph, "rmsnorm", [attended], context="attn_decay_sequence.attn_norm"
    )

    value = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_decay_sequence.value_proj",
    )
    decay_proj = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_decay_sequence.decay_proj",
    )
    decay_gate = _add(
        graph,
        "sigmoid",
        [decay_proj],
        context="attn_decay_sequence.decay_gate",
    )
    decay_weights = _add(
        graph,
        "cumprod_safe",
        [decay_gate],
        context="attn_decay_sequence.decay_weights",
    )
    weighted = _add(
        graph,
        "mul",
        [value, decay_weights],
        context="attn_decay_sequence.weighted_value",
    )
    projected = _add(
        graph,
        "linear_proj",
        [weighted],
        {"out_dim": D},
        context="attn_decay_sequence.post_proj",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="attn_decay_sequence.output")


def _pick_with_local_wildcard(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_classes,
    weights: MotifWeights = None,
    *,
    wildcard_prob: float,
):
    previous = graph.metadata.get("_wildcard_slot_prob", 0.0)
    graph.metadata["_wildcard_slot_prob"] = wildcard_prob
    try:
        return _pick_compatible_motif(graph, node_id, rng, motif_classes, weights)
    finally:
        graph.metadata["_wildcard_slot_prob"] = previous


def _add_explicit_norm(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    *,
    context: str,
    variants: tuple[str, ...] = ("rmsnorm", "layernorm"),
) -> int:
    """Add a bounded explicit norm choice without reopening motif-slot lottery."""
    op_name = variants[0] if len(variants) == 1 else rng.choice(variants)
    return _add(graph, op_name, [node_id], context=context)


def tpl_attn_spectral_filter(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → proj → rmsnorm → spectral_filter → proj → FFN → add.

    The bare attention → spectral_filter chain passes early screens but tends to
    stall before S1. Keep spectral_filter inside a densified residual branch and
    add an FFN stage so the block can recover useful gradients after the FFT.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_spectral_filter.pre_proj",
    )
    spectral_in = _add(
        graph,
        "rmsnorm",
        [attended],
        context="attn_spectral_filter.pre_norm",
    )
    filtered = _add(
        graph,
        "spectral_filter",
        [spectral_in],
        context="attn_spectral_filter.filter",
    )
    filtered = _add(
        graph,
        "linear_proj",
        [filtered],
        {"out_dim": D},
        context="attn_spectral_filter.post_proj",
    )
    filtered = _fix_dim(graph, filtered)

    spectral_mid = _residual(
        graph,
        attended,
        filtered,
        context="attn_spectral_filter.spectral_mid",
    )
    mid = _residual(graph, input_id, spectral_mid, context="attn_spectral_filter.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    bridge = _pick_compatible_motif_from_classes(
        graph,
        normed2,
        rng,
        (MOTIF_CLASS_CONV, MOTIF_CLASS_EFFICIENT_PROJ),
        weights,
    )
    if bridge:
        normed2 = _instantiate_motif(graph, normed2, bridge, rng)
        normed2 = _fix_dim(graph, normed2)
    post = _pick_with_local_wildcard(
        graph,
        normed2,
        rng,
        _FFN_CLASSES,
        weights,
        wildcard_prob=0.15,
    )
    if post:
        ffned = _instantiate_motif(graph, normed2, post, rng)
    else:
        ffned = _add(
            graph,
            "swiglu_mlp",
            [normed2],
            {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
            context="attn_spectral_filter.ffn",
        )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="attn_spectral_filter.output")


def tpl_attn_gated_minimum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → proj_a, proj_b → minimum → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context="attn_gated_minimum.norm")
    proj_a = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_minimum.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_minimum.proj_b",
    )
    out = _add(graph, "minimum", [proj_a, proj_b], context="attn_gated_minimum.out")
    out = _fix_dim(graph, out)
    return _residual(graph, input_id, out, context="attn_gated_minimum.output")


def tpl_attn_gated_maximum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → proj_a, proj_b → maximum → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context="attn_gated_maximum.norm")
    proj_a = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_maximum.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_gated_maximum.proj_b",
    )
    out = _add(graph, "maximum", [proj_a, proj_b], context="attn_gated_maximum.out")
    out = _fix_dim(graph, out)
    return _residual(graph, input_id, out, context="attn_gated_maximum.output")


def tpl_attn_hyperbolic(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → exp_map → hyp_distance path → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context="attn_hyperbolic.norm")
    proj_a = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_hyperbolic.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_hyperbolic.proj_b",
    )
    proj_a = _add(graph, "exp_map", [proj_a], context="attn_hyperbolic.exp_map_a")
    proj_b = _add(graph, "exp_map", [proj_b], context="attn_hyperbolic.exp_map_b")
    scored = _add(
        graph, "hyp_distance", [proj_a, proj_b], context="attn_hyperbolic.scored"
    )
    scored = _fix_dim(graph, scored)
    return _residual(graph, input_id, scored, context="attn_hyperbolic.output")


def tpl_attn_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → attention → add → rmsnorm → proj_a/proj_b → matmul → FFN → residual.

    The old single matmul branch often passed S0 then stalled. Keep the
    bilinear interaction, but place it inside a residualized attention block and
    add a recovery FFN after the matmul projection.
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

    attn = _pick_with_local_wildcard(
        graph,
        normed,
        rng,
        MOTIF_CLASS_ATTENTION,
        weights,
        wildcard_prob=0.0,
    )
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_normalized_matmul.project",
    )
    attended = _fix_dim(graph, attended)
    mid = _residual(graph, input_id, attended, context="attn_normalized_matmul.mid")

    attended = _add(graph, "rmsnorm", [mid], context="attn_normalized_matmul.norm")
    proj_a = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_normalized_matmul.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_normalized_matmul.proj_b",
    )
    out = _add(graph, "matmul", [proj_a, proj_b], context="attn_normalized_matmul.out")
    out = _add(
        graph,
        "linear_proj",
        [out],
        {"out_dim": D},
        context="attn_normalized_matmul.post_proj",
    )
    out = _fix_dim(graph, out)
    mid2 = _residual(graph, mid, out, context="attn_normalized_matmul.mid2")

    ffn = _pick_compatible_motif_from_classes(graph, mid2, rng, _FFN_CLASSES, weights)
    if ffn:
        ffned = _instantiate_motif(graph, mid2, ffn, rng)
    else:
        ffned = _add(
            graph,
            "swiglu_mlp",
            [mid2],
            {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
            context="attn_normalized_matmul.ffn",
        )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context="attn_normalized_matmul.output")


def _tpl_controlled_attn_matmul_ablation(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    name: str,
    attn_op: str,
    use_matmul_refine: bool,
    tail_kind: str,
) -> int:
    """Controlled ablation scaffold derived from attn_normalized_matmul."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attended = _add(graph, attn_op, [normed1], context=f"{name}.attn")
    if attn_op == "softmax_attention":
        attended = _add(graph, "rmsnorm", [attended], context=f"{name}.attn_norm")
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context=f"{name}.attn_proj",
    )
    attended = _fix_dim(graph, attended)
    mid = _residual(graph, input_id, attended, context=f"{name}.mid")

    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    refined_in = _add(graph, "rmsnorm", [normed2], context=f"{name}.refine_norm")

    if use_matmul_refine:
        proj_a = _add(
            graph,
            "linear_proj",
            [refined_in],
            {"out_dim": D},
            context=f"{name}.proj_a",
        )
        proj_b = _add(
            graph,
            "linear_proj",
            [refined_in],
            {"out_dim": D},
            context=f"{name}.proj_b",
        )
        refined = _add(graph, "matmul", [proj_a, proj_b], context=f"{name}.refined")
    else:
        refined = _add(
            graph,
            "swiglu_mlp",
            [refined_in],
            {"mlp_ratio": 2.0},
            context=f"{name}.refined",
        )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context=f"{name}.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(graph, mid, refined, context=f"{name}.mid2")
    tail_in = (
        mid2
        if tail_kind == "router_sidecar"
        else _add(
            graph,
            "rmsnorm",
            [mid2],
            context=f"{name}.tail_norm",
        )
    )

    if tail_kind == "dense":
        tail = _add(
            graph,
            "swiglu_mlp",
            [tail_in],
            {"mlp_ratio": 3.0},
            context=f"{name}.tail",
        )
    elif tail_kind == "sparse":
        tail = _add(graph, "nm_sparse_linear", [tail_in], context=f"{name}.tail")
        tail = _add(
            graph,
            "linear_proj",
            [tail],
            {"out_dim": D},
            context=f"{name}.tail_post_proj",
        )
    elif tail_kind == "router_sidecar":
        routed = _add(
            graph,
            "difficulty_blend_3way",
            [mid2, mid2],
            context=f"{name}.route_mix",
        )
        routed = _residual(graph, mid2, routed, context=f"{name}.route_mid")
        routed = _add(
            graph,
            "depth_weighted_proj",
            [routed],
            context=f"{name}.route_proj",
        )
        routed = _fix_dim(graph, routed)
        tail = _add(
            graph,
            "swiglu_mlp",
            [tail_in],
            {"mlp_ratio": 3.0},
            context=f"{name}.tail_ffn",
        )
        tail = _fix_dim(graph, tail)
        tail = _residual(graph, tail, routed, context=f"{name}.tail")
    else:
        raise ValueError(f"unknown tail_kind={tail_kind}")

    tail = _fix_dim(graph, tail)
    return _residual(graph, mid2, tail, context=f"{name}.output")


def tpl_attn_softmax_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Softmax attention with bilinear refinement and a dense recovery head."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "softmax_attention",
        [normed1],
        context="attn_softmax_normalized_matmul.attn1",
    )
    attended1 = _add(
        graph,
        "rmsnorm",
        [attended1],
        context="attn_softmax_normalized_matmul.attn1_norm",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_softmax_normalized_matmul.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_softmax_normalized_matmul.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="attn_softmax_normalized_matmul.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(
        graph, mid1, refined, context="attn_softmax_normalized_matmul.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_softmax_normalized_matmul.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid2, ffned, context="attn_softmax_normalized_matmul.output"
    )


def tpl_attn_linear_normalized_matmul_control(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention control anchored directly to the successful FFN scaffold."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_normalized_matmul_control.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_normalized_matmul_control.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_normalized_matmul_control.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="attn_linear_normalized_matmul_control.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(
        graph, mid1, refined, context="attn_linear_normalized_matmul_control.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_linear_normalized_matmul_control.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid2, ffned, context="attn_linear_normalized_matmul_control.output"
    )


def tpl_attn_linear_no_matmul_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention stack without matmul, using a softmax recovery pass."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_no_matmul_ffn.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_no_matmul_ffn.refine_norm",
    )
    attended2 = _add(
        graph,
        "softmax_attention",
        [refine_in],
        context="attn_linear_no_matmul_ffn.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(graph, mid1, attended2, context="attn_linear_no_matmul_ffn.mid2")

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_linear_no_matmul_ffn.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context="attn_linear_no_matmul_ffn.output")


def tpl_attn_linear_matmul_sparse_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear attention with matmul refinement, then a sparse output head."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_matmul_sparse_tail.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_matmul_sparse_tail.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_matmul_sparse_tail.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="attn_linear_matmul_sparse_tail.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(
        graph, mid1, refined, context="attn_linear_matmul_sparse_tail.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    sparse = _add(
        graph,
        "block_sparse_linear",
        [normed3],
        {"block_size": 16, "block_density": 0.25},
        context="attn_linear_matmul_sparse_tail.sparse_tail",
    )
    sparse = _fix_dim(graph, sparse)
    return _residual(
        graph, mid2, sparse, context="attn_linear_matmul_sparse_tail.output"
    )


def tpl_attn_linear_matmul_router_sidecar(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention control scaffold with routing as a side branch, not main path."""
    return _tpl_controlled_attn_matmul_ablation(
        graph,
        input_id,
        rng,
        weights,
        name="attn_linear_matmul_router_sidecar",
        attn_op="linear_attention",
        use_matmul_refine=True,
        tail_kind="router_sidecar",
    )


def _tpl_stabilized_attn_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    attn_op: str,
    final_ffn_classes: tuple[str, ...] | None,
    fixed_final_op: str | None = None,
    fixed_final_config: dict | None = None,
    name: str,
) -> int:
    """Residualized attention block with a dense recovery bridge before final FFN."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph,
        input_id,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attended = _add(graph, attn_op, [normed1], context=f"{name}.attended")
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context=f"{name}.project",
    )
    attended = _fix_dim(graph, attended)
    mid = _residual(graph, input_id, attended, context=f"{name}.mid")

    norm2 = _pick_with_local_wildcard(
        graph,
        mid,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    bridge_in = _add(graph, "rmsnorm", [normed2], context=f"{name}.bridge_norm")
    bridge = _add(
        graph,
        "swiglu_mlp",
        [bridge_in],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context=f"{name}.bridge_ffn",
    )
    bridge = _fix_dim(graph, bridge)
    mid2 = _residual(graph, mid, bridge, context=f"{name}.mid2")

    norm3 = _pick_with_local_wildcard(
        graph,
        mid2,
        rng,
        MOTIF_CLASS_NORM,
        weights,
        wildcard_prob=0.0,
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    if fixed_final_op is not None:
        ffned = _add(
            graph,
            fixed_final_op,
            [normed3],
            fixed_final_config or {},
            context=f"{name}.fixed_tail",
        )
    else:
        ffn = _pick_compatible_motif_from_classes(
            graph, normed3, rng, final_ffn_classes or _FFN_CLASSES, weights
        )
        if ffn:
            ffned = _instantiate_motif(graph, normed3, ffn, rng)
        else:
            ffned = _add(
                graph,
                "swiglu_mlp",
                [normed3],
                {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
                context=f"{name}.ffn_fallback",
            )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context=f"{name}.output")


def tpl_linear_attn_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear attention → matmul refinement → dense FFN.

    Redesign target: linear attention alone is too diffuse for induction.
    Follow it with a bilinear re-matching stage that can sharpen token-token
    retrieval before the dense FFN.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph, "linear_attention", [normed1], context="linear_attn_ffn_block.attn1"
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="linear_attn_ffn_block.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(graph, input_id, attended1, context="linear_attn_ffn_block.mid1")

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    attended2 = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="linear_attn_ffn_block.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="linear_attn_ffn_block.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="linear_attn_ffn_block.proj_b",
    )
    attended2 = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="linear_attn_ffn_block.refined",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="linear_attn_ffn_block.refined_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(graph, mid1, attended2, context="linear_attn_ffn_block.mid2")

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="linear_attn_ffn_block.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context="linear_attn_ffn_block.output")


def tpl_linear_attn_sparse_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear attention → matmul refinement → nm_sparse tail."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended = _add(
        graph, "linear_attention", [normed1], context="linear_attn_sparse_ffn.attn"
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="linear_attn_sparse_ffn.attn_proj",
    )
    attended = _fix_dim(graph, attended)
    mid1 = _residual(graph, input_id, attended, context="linear_attn_sparse_ffn.mid1")

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="linear_attn_sparse_ffn.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="linear_attn_sparse_ffn.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="linear_attn_sparse_ffn.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="linear_attn_sparse_ffn.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="linear_attn_sparse_ffn.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(graph, mid1, refined, context="linear_attn_sparse_ffn.mid2")

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "nm_sparse_linear",
        [normed3],
        context="linear_attn_sparse_ffn.sparse_tail",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context="linear_attn_sparse_ffn.output")


def tpl_graph_attn_sparse_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Graph attention → matmul refinement → block-sparse tail."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph, "graph_attention", [normed1], context="graph_attn_sparse_ffn.attn1"
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="graph_attn_sparse_ffn.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(graph, input_id, attended1, context="graph_attn_sparse_ffn.mid1")

    attended2 = _add(
        graph,
        "rmsnorm",
        [mid1],
        context="graph_attn_sparse_ffn.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="graph_attn_sparse_ffn.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="graph_attn_sparse_ffn.proj_b",
    )
    attended2 = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="graph_attn_sparse_ffn.refined",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="graph_attn_sparse_ffn.refined_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(graph, mid1, attended2, context="graph_attn_sparse_ffn.mid2")

    bridge = _add(
        graph,
        "rmsnorm",
        [mid2],
        context="graph_attn_sparse_ffn.bridge_norm",
    )
    bridge = _add(
        graph,
        "swiglu_mlp",
        [bridge],
        {"mlp_ratio": 2.0},
        context="graph_attn_sparse_ffn.bridge",
    )
    bridge = _fix_dim(graph, bridge)
    mid3 = _residual(graph, mid2, bridge, context="graph_attn_sparse_ffn.mid3")

    sparse = _add(
        graph,
        "block_sparse_linear",
        [mid3],
        {"block_size": 16, "block_density": 0.25},
        context="graph_attn_sparse_ffn.sparse_tail",
    )
    sparse = _fix_dim(graph, sparse)
    return _residual(graph, mid3, sparse, context="graph_attn_sparse_ffn.output")


def tpl_latent_attn_conv_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {latent_attention_compressor | conv} → add → FFN → add."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    path_attn = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="latent_attn_conv_hybrid.attn",
    )
    path_attn = _add(
        graph,
        "linear_proj",
        [path_attn],
        {"out_dim": D},
        context="latent_attn_conv_hybrid.project",
    )

    conv = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_CONV, weights)
    path_conv = _instantiate_motif(graph, normed, conv, rng) if conv else normed
    path_conv = _fix_dim(graph, path_conv)

    merged = _residual(
        graph, path_attn, path_conv, context="latent_attn_conv_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="latent_attn_conv_hybrid.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="latent_attn_conv_hybrid.output")


def tpl_diff_attn_conv_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {diff_attention | conv} → add → FFN → add."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    path_attn = _add(
        graph, "diff_attention", [normed], context="diff_attn_conv_hybrid.attn"
    )
    path_attn = _add(
        graph,
        "linear_proj",
        [path_attn],
        {"out_dim": D},
        context="diff_attn_conv_hybrid.project",
    )

    conv = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_CONV, weights)
    path_conv = _instantiate_motif(graph, normed, conv, rng) if conv else normed
    path_conv = _fix_dim(graph, path_conv)

    merged = _residual(
        graph, path_attn, path_conv, context="diff_attn_conv_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="diff_attn_conv_hybrid.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)

    return _residual(graph, mid, ffned, context="diff_attn_conv_hybrid.output")


def tpl_attn_safe_division(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm -> attention -> numerator / normalized denominator -> div_safe -> proj -> residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    pa = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_safe_division.pa",
    )
    pb = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_safe_division.pb",
    )
    denom = _add(
        graph,
        "softmax_last",
        [pb],
        context="attn_safe_division.denom",
    )
    out = _add(graph, "div_safe", [pa, denom], context="attn_safe_division.out")
    out = _add(
        graph,
        "linear_proj",
        [out],
        {"out_dim": D},
        context="attn_safe_division.post_proj",
    )
    out = _fix_dim(graph, out)
    return _residual(graph, input_id, out, context="attn_safe_division.output")


def tpl_latent_attn_ssm_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {latent_attention_compressor | SSM} → add → FFN → add."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    pa = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="latent_attn_ssm_hybrid.pa",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="latent_attn_ssm_hybrid.pa_proj",
    )
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    ps = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    ps = _fix_dim(graph, ps)
    merged = _residual(graph, pa, ps, context="latent_attn_ssm_hybrid.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="latent_attn_ssm_hybrid.mid")
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="latent_attn_ssm_hybrid.output")


def tpl_local_attn_ssm_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {local_window_attn | SSM} → add → FFN → add."""
    D = graph.model_dim
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    pa = _add(
        graph,
        "local_window_attn",
        [normed],
        {"window_size": rng.choice(choices)},
        context="local_attn_ssm_hybrid.pa",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="local_attn_ssm_hybrid.pa_proj",
    )
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    ps = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    ps = _fix_dim(graph, ps)
    merged = _residual(graph, pa, ps, context="local_attn_ssm_hybrid.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="local_attn_ssm_hybrid.mid")
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="local_attn_ssm_hybrid.output")


def tpl_attn_spiking_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention | lif_neuron → spike_rate_code} → add → residual."""
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    path_a = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    path_a = _fix_dim(graph, path_a)
    spiked = _add(graph, "lif_neuron", [normed], context="attn_spiking_hybrid.spiked")
    path_s = _add(
        graph, "spike_rate_code", [spiked], context="attn_spiking_hybrid.path_s"
    )
    path_s = _fix_dim(graph, path_s)
    merged = _residual(graph, path_a, path_s, context="attn_spiking_hybrid.merge")
    merged = _fix_dim(graph, merged)
    return _residual(graph, input_id, merged, context="attn_spiking_hybrid.output")


def tpl_local_attn_moe(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → add → norm → MoE → add."""
    choices = [8, 16] if graph.model_dim >= 256 else [8, 16, 32]
    return _tpl_attention_ffn_block(
        graph,
        input_id,
        rng,
        weights,
        attn_op="local_window_attn",
        attn_config={"window_size": rng.choice(choices)},
        ffn_classes=(MOTIF_CLASS_MOE, MOTIF_CLASS_GATE),
    )


def _make_attn_op_chain_template(post_op: str):
    """Factory for norm -> attention -> post_op -> residual templates."""

    def _template(graph, input_id, rng, weights=None):
        return _tpl_attn_op_chain(graph, input_id, rng, weights, post_op=post_op)

    _template.__name__ = f"tpl_attn_{post_op}"
    return _template


def _make_attn_ffn_template(attn_op=None, ffn_classes=None):
    """Factory for norm -> attention -> FFN -> residual templates."""
    kwargs = {}
    if attn_op is not None:
        kwargs["attn_op"] = attn_op
    if ffn_classes is not None:
        kwargs["ffn_classes"] = ffn_classes

    def _template(graph, input_id, rng, weights=None):
        return _tpl_attention_ffn_block(graph, input_id, rng, weights, **kwargs)

    name_parts = []
    if attn_op:
        name_parts.append(attn_op)
    if ffn_classes:
        name_parts.append("ffn")
    _template.__name__ = f"tpl_{'_'.join(name_parts) or 'attn_ffn'}"
    return _template


_ATTN_OP_CHAIN_TEMPLATES = {
    "attn_reciprocal_gated": "reciprocal",
    "attn_kronecker_hybrid": "kronecker_linear",
    "attn_log_gated": "log",
}

_MOE_CLASSES = (MOTIF_CLASS_MOE, MOTIF_CLASS_GATE)

_ATTN_FFN_TEMPLATES = {
    "attn_dual_axis": {},
    "latent_attn_ffn_block": {"attn_op": "latent_attention_compressor"},
    "diff_attn_ffn_block": {"attn_op": "diff_attention"},
    "latent_attn_sparse_ffn": {
        "attn_op": "latent_attention_compressor",
        "ffn_classes": _SPARSE_FFN_CLASSES,
    },
    "graph_attn_ffn_block": {"attn_op": "graph_attention"},
    "attn_moe_block": {"ffn_classes": _MOE_CLASSES},
    "latent_attn_moe": {
        "attn_op": "latent_attention_compressor",
        "ffn_classes": _MOE_CLASSES,
    },
    "diff_attn_moe": {"attn_op": "diff_attention", "ffn_classes": _MOE_CLASSES},
    "graph_attn_moe": {"attn_op": "graph_attention", "ffn_classes": _MOE_CLASSES},
}

for _name, _post_op in _ATTN_OP_CHAIN_TEMPLATES.items():
    globals()[f"tpl_{_name}"] = _make_attn_op_chain_template(_post_op)

for _name, _kwargs in _ATTN_FFN_TEMPLATES.items():
    globals()[f"tpl_{_name}"] = _make_attn_ffn_template(**_kwargs)
