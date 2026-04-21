"""Attention template tail — private split. Re-exported from _templates_attention_tail."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MotifWeights,
    _FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
)


def _tpl_novel_mixing_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    primary_op: str,
    complement_op: str = "state_space",
    template_ctx: str,
) -> int:
    """Reusable pattern: rmsnorm → {primary_op || complement_op} → merge → residual → rmsnorm → swiglu(4x) → residual."""
    D = graph.model_dim
    normed = _add(graph, "rmsnorm", [input_id], context=f"{template_ctx}.norm1")
    pa = _add(graph, primary_op, [normed], context=f"{template_ctx}.primary")
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context=f"{template_ctx}.primary_proj",
    )
    pb = _add(graph, complement_op, [normed], context=f"{template_ctx}.complement")
    pb = _fix_dim(graph, pb)
    merged = _residual(graph, pa, pb, context=f"{template_ctx}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{template_ctx}.mid")
    normed2 = _add(graph, "rmsnorm", [mid], context=f"{template_ctx}.norm2")
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 4.0},
        context=f"{template_ctx}.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{template_ctx}.output")


def tpl_difficulty_routed_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {difficulty_routed_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Uses the dense difficulty-routed attention kernel currently implemented in
    this repo plus an SSM complement.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="difficulty_routed_attention",
        template_ctx="difficulty_routed_attention_block",
    )


def tpl_strided_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {strided_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Multi-head dilated attention: each head uses a different stride (1,2,4,8)
    over key/value positions for multi-scale coverage.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="strided_attention",
        template_ctx="strided_attention_block",
    )


def tpl_gated_progressive_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {gated_progressive_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Dense causal attention with a per-token output gate initialized OFF
    (bias=-2) that learns how much attention output to use.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="gated_progressive_attention",
        template_ctx="gated_progressive_attention_block",
    )


def tpl_gated_linear_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {gated_linear_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    GLA: linear attention with data-dependent decay gates. O(nd²) cost.
    Trains in parallel like transformer, infers like RNN. Used by Qwen3-Next.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="gated_linear_attention",
        template_ctx="gated_linear_attention_block",
    )


def tpl_long_conv_hyena_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {long_conv_hyena || gated_linear_attention} → merge → residual → rmsnorm → swiglu → residual.

    Hyena long conv paired with GLA. The FFT convolution handles broad mixing;
    the GLA side handles sharper retrieval.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="long_conv_hyena",
        complement_op="gated_linear_attention",
        template_ctx="long_conv_hyena_block",
    )


def tpl_associative_memory_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {associative_memory || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Modern Hopfield-style retrieval with a learnable temperature, implemented
    here as dense causal query-key-value retrieval.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="associative_memory",
        template_ctx="associative_memory_block",
    )


def tpl_mixture_of_recursions_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → {mixture_of_recursions || gated_delta} → merge → residual → rmsnorm → swiglu → residual.

    MoR: shared parameter block with a soft depth router over four recurrent
    refinement steps. Paired with gated delta rule for targeted state writes.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="mixture_of_recursions",
        complement_op="gated_delta",
        template_ctx="mixture_of_recursions_block",
    )


def tpl_codex_ssm_retention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {gated_linear_attention || SSM} → compress → residual → norm → FFN → residual.

    Codex variant inspired by RetNet/Kimi-style retention blocks: a fixed
    gated linear-attention path is paired with a picked SSM motif, then both
    paths are compressed before the FFN tail.
    """
    D = graph.model_dim
    name = "codex_ssm_retention_block"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    graph.metadata["_skip_global_decorators"] = True

    normed = _add(graph, "rmsnorm", [input_id], context=f"{name}.norm")

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{name}[{template_instance}].retention_core",
        slot_classes=("retention_core",),
        selected_name="gated_linear_attention",
        selected_class="component",
        input_node_id=normed,
    )
    pa = _add(graph, "gated_linear_attention", [normed], context=f"{name}.retention")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{name}[{template_instance}].retention_basis",
        slot_classes=("retention_basis",),
        selected_name="shared_basis_proj",
        selected_class="component",
        input_node_id=pa,
    )
    pa = _add(graph, "shared_basis_proj", [pa], context=f"{name}.retention_basis")
    pa = _add(
        graph, "linear_proj", [pa], {"out_dim": D}, context=f"{name}.retention_out"
    )

    pb = _add(graph, "state_space", [normed], context=f"{name}.memory")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=3,
        slot_key=f"{name}[{template_instance}].memory_bottleneck",
        slot_classes=("memory_bottleneck",),
        selected_name="bottleneck_proj",
        selected_class="component",
        input_node_id=pb,
    )
    pb = _add(graph, "bottleneck_proj", [pb], context=f"{name}.memory_bottleneck")

    merged = _residual(graph, pa, pb, context=f"{name}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{name}.mid")

    normed2 = _add(graph, "rmsnorm", [mid], context=f"{name}.tail_norm")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=4,
        slot_key=f"{name}[{template_instance}].tail_basis",
        slot_classes=("tail_basis",),
        selected_name="shared_basis_proj",
        selected_class="component",
        input_node_id=normed2,
    )
    tail = _add(
        graph,
        "linear_proj_down",
        [normed2],
        {"out_dim": max(64, D // 2)},
        context=f"{name}.tail_bottleneck",
    )
    tail = _add(graph, "rmsnorm", [tail], context=f"{name}.tail_bottleneck_norm")
    tail = _add(
        graph,
        "linear_proj_up",
        [tail],
        {"out_dim": D},
        context=f"{name}.tail_basis",
    )
    ffned = _add(graph, "swiglu_mlp", [tail], {"mlp_ratio": 2.0}, context=f"{name}.ffn")
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{name}.output")


def tpl_codex_ssm_delta_memory_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {gated_delta || linear_attention} → merge → residual → norm → FFN → residual.

    Codex variant inspired by delta-rule / fast-weight memory: a gated-delta
    write path is fused with a cheap linear-attention read path.
    """
    D = graph.model_dim
    name = "codex_ssm_delta_memory_block"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    graph.metadata["_skip_global_decorators"] = True

    normed = _add(graph, "rmsnorm", [input_id], context=f"{name}.norm")

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{name}[{template_instance}].delta_core",
        slot_classes=("delta_core",),
        selected_name="gated_delta",
        selected_class="component",
        input_node_id=normed,
    )
    pa = _add(graph, "gated_delta", [normed], context=f"{name}.delta")
    pa = _add(graph, "shared_basis_proj", [pa], context=f"{name}.delta_basis")

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{name}[{template_instance}].read_core",
        slot_classes=("read_core",),
        selected_name="linear_attention",
        selected_class="component",
        input_node_id=normed,
    )
    pb = _add(graph, "linear_attention", [normed], context=f"{name}.linear_read")
    pb = _add(graph, "linear_proj", [pb], {"out_dim": D}, context=f"{name}.linear_out")

    merged = _residual(graph, pa, pb, context=f"{name}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{name}.mid")

    normed2 = _add(graph, "rmsnorm", [mid], context=f"{name}.tail_norm")
    tail = _add(
        graph,
        "linear_proj_down",
        [normed2],
        {"out_dim": max(64, D // 2)},
        context=f"{name}.tail_bottleneck",
    )
    tail = _add(graph, "rmsnorm", [tail], context=f"{name}.tail_bottleneck_norm")
    tail = _add(
        graph,
        "linear_proj_up",
        [tail],
        {"out_dim": D},
        context=f"{name}.tail_restore",
    )
    ffned = _add(graph, "swiglu_mlp", [tail], {"mlp_ratio": 2.0}, context=f"{name}.ffn")
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{name}.output")


def tpl_codex_ssm_mla_gated_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {latent_attention_compressor || gated_linear_attention} → merge → residual → norm → FFN → residual.

    Codex variant that combines MLA-style compressed attention with a gated
    linear-attention memory path, then lets a picked efficient projection
    decide how aggressively to compress before the FFN.
    """
    D = graph.model_dim
    name = "codex_ssm_mla_gated_block"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)

    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{name}[{template_instance}].mla_core",
        slot_classes=("mla_core",),
        selected_name="latent_attention_compressor",
        selected_class="component",
        input_node_id=normed,
    )
    pa = _add(graph, "latent_attention_compressor", [normed], context=f"{name}.mla")
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context=f"{name}.mla_out")

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=2,
        slot_key=f"{name}[{template_instance}].retention_core",
        slot_classes=("retention_core",),
        selected_name="gated_linear_attention",
        selected_class="component",
        input_node_id=normed,
    )
    pb = _add(graph, "gated_linear_attention", [normed], context=f"{name}.retention")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=3,
        slot_key=f"{name}[{template_instance}].retention_compress",
        slot_classes=("retention_compress",),
        selected_name="efficient_proj",
        selected_class=MOTIF_CLASS_EFFICIENT_PROJ,
        input_node_id=pb,
    )
    proj = _pick_compatible_motif(graph, pb, rng, MOTIF_CLASS_EFFICIENT_PROJ, weights)
    pb = _instantiate_motif(graph, pb, proj, rng) if proj else pb
    pb = _fix_dim(graph, pb)

    merged = _residual(graph, pa, pb, context=f"{name}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{name}.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = (
        _instantiate_motif(graph, normed2, ffn, rng)
        if ffn
        else _add(
            graph, "swiglu_mlp", [normed2], {"mlp_ratio": 3.0}, context=f"{name}.ffn"
        )
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{name}.output")


def tpl_codex_ssm_local_recall_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {local_window_attn || SSM} → merge → residual → norm → FFN → residual.

    Codex variant inspired by dynamic sparse/local attention: dense local token
    interaction handles short-range induction while the SSM carries cheap
    long-range state.
    """
    D = graph.model_dim
    name = "codex_ssm_local_recall_block"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)

    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{name}[{template_instance}].local_core",
        slot_classes=("local_core",),
        selected_name="local_window_attn",
        selected_class="component",
        input_node_id=normed,
    )
    pa = _add(
        graph,
        "local_window_attn",
        [normed],
        {"window_size": 32},
        context=f"{name}.local_attn",
    )
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context=f"{name}.local_out")
    pa = _add(graph, "shared_basis_proj", [pa], context=f"{name}.local_basis")

    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _add(graph, "shared_basis_proj", [pb], context=f"{name}.memory_basis")
    pb = _fix_dim(graph, pb)

    merged = _residual(graph, pa, pb, context=f"{name}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{name}.mid")

    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = (
        _instantiate_motif(graph, normed2, ffn, rng)
        if ffn
        else _add(
            graph, "swiglu_mlp", [normed2], {"mlp_ratio": 4.0}, context=f"{name}.ffn"
        )
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{name}.output")


def tpl_recursive_attn_ssm_depth(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {latent_attn || SSM} → merge → adaptive_recursion → residual → norm → FFN → residual.

    Combines the top-performing parallel attention+SSM mixing with
    adaptive_recursion (41.3% S1 rate as an op) for depth-adaptive
    processing. The recursion gate adds variable-depth refinement
    after the initial hybrid mix.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Path A: latent attention compressor
    pa = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="recursive_attn_ssm_depth.latent_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="recursive_attn_ssm_depth.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge + adaptive recursion for depth-adaptive refinement
    merged = _residual(graph, pa, pb, context="recursive_attn_ssm_depth.merge")
    merged = _fix_dim(graph, merged)
    recursed = _add(
        graph,
        "adaptive_recursion",
        [merged],
        context="recursive_attn_ssm_depth.recursion",
    )
    recursed = _fix_dim(graph, recursed)
    mid = _residual(graph, input_id, recursed, context="recursive_attn_ssm_depth.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="recursive_attn_ssm_depth.output")


def tpl_latent_attn_padic_hybrid(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {latent_attn || padic_expand} → merge → residual → norm → FFN → residual.

    Parallel hybrid: latent attention compressor captures token interactions
    while padic_expand (34.1% S1, ind 0.004) provides multi-scale
    hierarchical feature expansion. Both are strong individual ops that
    complement each other — attention for local patterns, p-adic for
    hierarchical structure.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Path A: latent attention compressor
    pa = _add(
        graph,
        "latent_attention_compressor",
        [normed],
        context="latent_attn_padic_hybrid.latent_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="latent_attn_padic_hybrid.attn_proj",
    )

    # Path B: padic_expand
    pb = _add(
        graph,
        "padic_expand",
        [normed],
        context="latent_attn_padic_hybrid.padic",
    )
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="latent_attn_padic_hybrid.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="latent_attn_padic_hybrid.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="latent_attn_padic_hybrid.output")


def tpl_graph_attn_ssm_recursive(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {graph_attention || SSM} → merge → residual → norm → FFN → residual.

    Graph attention (40% S1 as template, 32% as block) + SSM hybrid.
    Graph attention captures structural/relational patterns while SSM
    provides sequential context. Targets the underexplored graph attention
    family identified in ops analysis.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Path A: graph attention
    pa = _add(
        graph,
        "graph_attention",
        [normed],
        context="graph_attn_ssm_recursive.graph_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="graph_attn_ssm_recursive.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="graph_attn_ssm_recursive.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="graph_attn_ssm_recursive.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="graph_attn_ssm_recursive.output")
