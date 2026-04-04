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
    """norm → attention → proj_a, proj_b → matmul → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    attended = _instantiate_motif(graph, normed, attn, rng) if attn else normed
    attended = _fix_dim(graph, attended)
    attended = _add(graph, "rmsnorm", [attended], context="attn_normalized_matmul.norm")
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
    out = _fix_dim(graph, out)
    return _residual(graph, input_id, out, context="attn_normalized_matmul.output")


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
    """norm → attention → proj_a / proj_b → residual."""
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
    out = _add(graph, "div_safe", [pa, pb], context="attn_safe_division.out")
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
    "attn_decay_sequence": "cumprod_safe",
    "attn_kronecker_hybrid": "kronecker_linear",
    "attn_log_gated": "log",
}

_MOE_CLASSES = (MOTIF_CLASS_MOE, MOTIF_CLASS_GATE)

_ATTN_FFN_TEMPLATES = {
    "attn_dual_axis": {},
    "latent_attn_ffn_block": {"attn_op": "latent_attention_compressor"},
    "diff_attn_ffn_block": {"attn_op": "diff_attention"},
    "linear_attn_ffn_block": {"attn_op": "linear_attention"},
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
    "linear_attn_sparse_ffn": {
        "attn_op": "linear_attention",
        "ffn_classes": _SPARSE_FFN_CLASSES,
    },
    "graph_attn_sparse_ffn": {
        "attn_op": "graph_attention",
        "ffn_classes": _SPARSE_FFN_CLASSES,
    },
}

for _name, _post_op in _ATTN_OP_CHAIN_TEMPLATES.items():
    globals()[f"tpl_{_name}"] = _make_attn_op_chain_template(_post_op)

for _name, _kwargs in _ATTN_FFN_TEMPLATES.items():
    globals()[f"tpl_{_name}"] = _make_attn_ffn_template(**_kwargs)
