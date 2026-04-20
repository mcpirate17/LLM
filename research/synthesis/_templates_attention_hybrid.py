"""Attention template tail — private split. Re-exported from _templates_attention_tail."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MotifWeights,
    _FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _tpl_attention_ffn_block,
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_attention_tail import (
    _pick_with_local_wildcard,
)


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
    name = "graph_attn_sparse_ffn"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    normed = _add(graph, "rmsnorm", [input_id], context=f"{name}.norm")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=0,
        slot_key=f"{name}[{template_instance}].norm",
        slot_classes=("norm_wrap",),
        selected_name="rmsnorm",
        selected_class="norm_wrap",
        input_node_id=input_id,
    )
    attended = _add(graph, "graph_attention", [normed], context=f"{name}.attn1")
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context=f"{name}.attn1_proj",
    )
    mid = _residual(graph, input_id, attended, context=f"{name}.mid1")
    refine_in = _add(graph, "rmsnorm", [mid], context=f"{name}.refine_norm")
    proj_a = _add(
        graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_a"
    )
    proj_b = _add(
        graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_b"
    )
    refined = _add(graph, "matmul", [proj_a, proj_b], context=f"{name}.refined")
    refined = _add(graph, "rmsnorm", [refined], context=f"{name}.refined_norm")
    sparse = _add(
        graph,
        "block_sparse_linear",
        [refined],
        {"block_size": 16, "block_density": 0.25},
        context=f"{name}.sparse_tail",
    )
    sparse = _fix_dim(graph, sparse)
    return _residual(graph, mid, sparse, context=f"{name}.output")


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
        # 2 lane paths + skip = 3-way addition into the residual stream.
        {"out_dim": D, "init_scale": 0.577},
        context="latent_attn_conv_hybrid.project",
    )

    conv = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_CONV, weights)
    path_conv = _instantiate_motif(graph, normed, conv, rng) if conv else normed
    path_conv = _fix_dim(graph, path_conv)

    merged = _residual(
        graph, path_attn, path_conv, context="latent_attn_conv_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    # Bound multi-path variance before injecting back into the residual stream.
    merged = _add(
        graph, "rmsnorm", [merged], context="latent_attn_conv_hybrid.merge_norm"
    )
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
        # 2 lane paths + skip = 3-way addition into the residual stream.
        {"out_dim": D, "init_scale": 0.577},
        context="diff_attn_conv_hybrid.project",
    )

    conv = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_CONV, weights)
    path_conv = _instantiate_motif(graph, normed, conv, rng) if conv else normed
    path_conv = _fix_dim(graph, path_conv)

    merged = _residual(
        graph, path_attn, path_conv, context="diff_attn_conv_hybrid.merge"
    )
    merged = _fix_dim(graph, merged)
    # Bound multi-path variance before injecting back into the residual stream.
    merged = _add(
        graph, "rmsnorm", [merged], context="diff_attn_conv_hybrid.merge_norm"
    )
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
    attended = _add(
        graph,
        "rmsnorm",
        [attended],
        context="attn_safe_division.norm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context="attn_safe_division.pa",
    )
    numerator = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_safe_division.numerator_norm",
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
    out = _add(
        graph,
        "div_safe",
        [numerator, denom],
        context="attn_safe_division.out",
    )
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
        # 2 lane paths + skip = 3-way addition into the residual stream.
        {"out_dim": D, "init_scale": 0.577},
        context="latent_attn_ssm_hybrid.pa_proj",
    )
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    ps = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    ps = _fix_dim(graph, ps)
    merged = _residual(graph, pa, ps, context="latent_attn_ssm_hybrid.merge")
    merged = _fix_dim(graph, merged)
    # Bound multi-path variance before injecting back into the residual stream
    # — keeps Jacobian spectral norm in [0.5, 10] so investigation gate passes.
    merged = _add(
        graph, "rmsnorm", [merged], context="latent_attn_ssm_hybrid.merge_norm"
    )
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
        # 2 lane paths + skip = 3-way addition into the residual stream.
        {"out_dim": D, "init_scale": 0.577},
        context="local_attn_ssm_hybrid.pa_proj",
    )
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    ps = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    ps = _fix_dim(graph, ps)
    merged = _residual(graph, pa, ps, context="local_attn_ssm_hybrid.merge")
    merged = _fix_dim(graph, merged)
    # Bound multi-path variance before injecting back into the residual stream.
    merged = _add(
        graph, "rmsnorm", [merged], context="local_attn_ssm_hybrid.merge_norm"
    )
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


def tpl_attn_normalized_matmul_pinned(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {softmax_attn+proj+norm || padic_expand+proj} → merge → residual → rmsnorm → swiglu(4x) → residual.

    Fully pinned variant of attn_normalized_matmul. Every component is fixed
    to the empirically best choice — no motif lottery:
      - Norm: rmsnorm (fast, proven)
      - Attention: softmax_attention → rmsnorm → linear_proj (best induction 0.078)
      - Parallel: padic_expand → linear_proj (34% S1, hierarchical features)
      - FFN: swiglu_mlp ratio=4 (standard transformer FFN, full-width)
    """
    D = graph.model_dim

    # Pre-norm: rmsnorm (pinned)
    normed = _add(
        graph,
        "rmsnorm",
        [input_id],
        context="attn_normalized_matmul_pinned.norm1",
    )

    # Path A: softmax attention → post-attn norm → projection
    pa = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_normalized_matmul_pinned.softmax_attn",
    )
    pa = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_normalized_matmul_pinned.attn_postnorm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_normalized_matmul_pinned.attn_proj",
    )

    # Path B: padic_expand → projection
    pb = _add(
        graph,
        "padic_expand",
        [normed],
        context="attn_normalized_matmul_pinned.padic",
    )
    pb = _add(
        graph,
        "linear_proj",
        [pb],
        {"out_dim": D},
        context="attn_normalized_matmul_pinned.padic_proj",
    )
    pb = _fix_dim(graph, pb)

    # Merge parallel paths + skip connection
    merged = _residual(graph, pa, pb, context="attn_normalized_matmul_pinned.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_normalized_matmul_pinned.mid"
    )

    # FFN: rmsnorm → swiglu_mlp ratio=4 (pinned — standard transformer FFN)
    normed2 = _add(
        graph,
        "rmsnorm",
        [mid],
        context="attn_normalized_matmul_pinned.norm2",
    )
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 4.0},
        context="attn_normalized_matmul_pinned.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="attn_normalized_matmul_pinned.output")
