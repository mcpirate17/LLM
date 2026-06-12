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
    template_gated_lane_merge as _gated_merge,
)


def _codex_tail_mlp_ratio(rng: random.Random) -> float:
    return 2.0 if rng.random() < 0.5 else 3.0


def _codex_tail_ffn(
    graph: ComputationGraph, tail: int, rng: random.Random, *, context: str
) -> int:
    ffned = _add(
        graph,
        "swiglu_mlp",
        [tail],
        {"mlp_ratio": _codex_tail_mlp_ratio(rng)},
        context=context,
    )
    return _fix_dim(graph, ffned)


_NOVEL_MIXING_MLP_RATIOS: tuple[float, ...] = (2.0, 3.0, 4.0)
_NOVEL_MIXING_COMPLEMENT_PROJECTION_OPS: frozenset[str] = frozenset(
    {
        "conv_only",
        "local_window_attn",
    }
)


def _pick_norm_or_default(
    graph: ComputationGraph,
    src: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    fallback_context: str,
) -> int:
    motif = _pick_compatible_motif(graph, src, rng, MOTIF_CLASS_NORM, weights)
    if motif:
        return _instantiate_motif(graph, src, motif, rng)
    return _add(graph, "rmsnorm", [src], context=fallback_context)


def _record_fixed_core_slot(
    graph: ComputationGraph,
    *,
    template_ctx: str,
    template_instance: int,
    slot_index: int,
    role: str,
    op_name: str,
    input_node_id: int,
) -> None:
    record_template_slot_binding(
        graph,
        template_name=template_ctx,
        template_instance=template_instance,
        slot_index=slot_index,
        slot_key=f"{template_ctx}[{template_instance}].{role}",
        slot_classes=(role,),
        selected_name=op_name,
        selected_class="component",
        input_node_id=input_node_id,
    )


def _pick_ffn_or_swiglu(
    graph: ComputationGraph,
    src: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    fallback_context: str,
) -> int:
    motif = _pick_compatible_motif_from_classes(graph, src, rng, _FFN_CLASSES, weights)
    if motif:
        return _instantiate_motif(graph, src, motif, rng)
    return _add(
        graph,
        "swiglu_mlp",
        [src],
        {"mlp_ratio": rng.choice(_NOVEL_MIXING_MLP_RATIOS)},
        context=fallback_context,
    )


def _pick_ffn_or_default(
    graph: ComputationGraph,
    src: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    fallback_context: str,
) -> int:
    """Backward-compatible alias for external novel-template modules."""
    return _pick_ffn_or_swiglu(
        graph, src, rng, weights, fallback_context=fallback_context
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
    """Reusable pattern: norm → {primary_op || complement_op} → merge → residual → norm → FFN → residual.

    Head/tail norm and FFN are rng-picked from the standard motif classes so
    each invocation produces a distinct graph fingerprint. Primary/complement
    cores are recorded as fixed-component slot bindings for meta-analysis.
    """
    D = graph.model_dim
    instance = int(graph.metadata.get("_active_template_instance", 0) or 0)

    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    _record_fixed_core_slot(
        graph,
        template_ctx=template_ctx,
        template_instance=instance,
        slot_index=1,
        role="primary_core",
        op_name=primary_op,
        input_node_id=normed,
    )
    pa = _add(graph, primary_op, [normed], context=f"{template_ctx}.primary")
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context=f"{template_ctx}.primary_proj",
    )

    _record_fixed_core_slot(
        graph,
        template_ctx=template_ctx,
        template_instance=instance,
        slot_index=2,
        role="complement_core",
        op_name=complement_op,
        input_node_id=normed,
    )
    pb = _add(graph, complement_op, [normed], context=f"{template_ctx}.complement")
    if complement_op in _NOVEL_MIXING_COMPLEMENT_PROJECTION_OPS:
        pb = _add(
            graph,
            "linear_proj",
            [pb],
            {"out_dim": D},
            context=f"{template_ctx}.complement_proj",
        )
    else:
        pb = _fix_dim(graph, pb)

    # Gated merge: the complement lane (state_space) multiplicatively gates the
    # primary mixer. Both lanes are euclidean post-projection, so this introduces
    # no math-space adjacency violation.
    merged = _fix_dim(
        graph,
        _gated_merge(graph, pa, pb, context=f"{template_ctx}.merge", dim=D),
    )
    mid = _residual(graph, input_id, merged, context=f"{template_ctx}.mid")

    normed2 = _pick_norm_or_default(
        graph, mid, rng, weights, fallback_context=f"{template_ctx}.norm2"
    )
    ffned = _fix_dim(
        graph,
        _pick_ffn_or_swiglu(
            graph, normed2, rng, weights, fallback_context=f"{template_ctx}.ffn"
        ),
    )
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


def tpl_sparsemax_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {sparsemax_attention || state_space} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="sparsemax_attention",
        template_ctx="sparsemax_attention_block",
    )


def tpl_entmax_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {entmax_attention || state_space} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="entmax_attention",
        template_ctx="entmax_attention_block",
    )


def tpl_learnable_semiring_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {learnable_semiring_attention || state_space} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="learnable_semiring_attention",
        template_ctx="learnable_semiring_attention_block",
    )


def tpl_reciprocal_rank_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {reciprocal_rank_attention || state_space} → merge → residual → norm → FFN → residual.

    The primary mixer is full-range and content-addressed; the reciprocal
    boost explicitly prefers mutual query/key agreement for binding-style
    retrieval instead of changing only the output gate or value transform.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="reciprocal_rank_attention",
        template_ctx="reciprocal_rank_attention_block",
    )


def tpl_reciprocal_semiring_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {reciprocal_semiring_attention || state_space} → merge → residual → norm → FFN → residual.

    Composes the two leading novel attention-surface mixers: reciprocal-rank
    mutual-match addressing on the score, then learnable-semiring (mean↔max)
    pooling on the value aggregation. Both reduce to softmax attention at init.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="reciprocal_semiring_attention",
        template_ctx="reciprocal_semiring_attention_block",
    )


def tpl_phase_lock_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {phase_lock_attention || state_space} → merge → residual → norm → FFN → residual.

    Phase-lock attention keeps global content addressing but adds a
    synchrony score to the address itself, distinct from softmax/sparsemax
    normalizers and from semiring value aggregation.
    """
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="phase_lock_attention",
        template_ctx="phase_lock_attention_block",
    )


def tpl_stdp_reciprocal_memory_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → STDP temporal salience → reciprocal retrieval → sparse readout → FFN residual.

    This template uses STDP as a temporal salience preconditioner, then forces
    the headline novelty into the full-range content-addressed mixer slot via
    ``reciprocal_rank_attention``.  Sparsemax readout is a support lane, not the
    novel component, and gives the block a proven binding-friendly fallback.
    """
    D = graph.model_dim
    name = "stdp_reciprocal_memory_block"
    instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{name}.norm1"
    )

    spike_seed = _add(graph, "spike_rate_code", [normed], context=f"{name}.spike_seed")
    _record_fixed_core_slot(
        graph,
        template_ctx=name,
        template_instance=instance,
        slot_index=1,
        role="temporal_salience_core",
        op_name="stdp_attention",
        input_node_id=spike_seed,
    )
    salience = _add(graph, "stdp_attention", [spike_seed], context=f"{name}.stdp")
    salience = _add(graph, "rmsnorm", [salience], context=f"{name}.stdp_norm")

    _record_fixed_core_slot(
        graph,
        template_ctx=name,
        template_instance=instance,
        slot_index=2,
        role="reciprocal_retrieval_core",
        op_name="reciprocal_rank_attention",
        input_node_id=salience,
    )
    retrieved = _add(
        graph,
        "reciprocal_rank_attention",
        [salience],
        context=f"{name}.reciprocal",
    )
    retrieved = _add(
        graph,
        "linear_proj",
        [retrieved],
        {"out_dim": D},
        context=f"{name}.reciprocal_proj",
    )
    mid = _residual(graph, input_id, retrieved, context=f"{name}.mid")

    normed2 = _pick_norm_or_default(
        graph, mid, rng, weights, fallback_context=f"{name}.norm2"
    )
    _record_fixed_core_slot(
        graph,
        template_ctx=name,
        template_instance=instance,
        slot_index=3,
        role="sparse_readout_core",
        op_name="sparsemax_attention",
        input_node_id=normed2,
    )
    sparse = _add(graph, "sparsemax_attention", [normed2], context=f"{name}.sparse")
    sparse = _add(
        graph,
        "linear_proj",
        [sparse],
        {"out_dim": D},
        context=f"{name}.sparse_proj",
    )
    mid2 = _residual(graph, mid, sparse, context=f"{name}.sparse_merge")
    normed3 = _pick_norm_or_default(
        graph, mid2, rng, weights, fallback_context=f"{name}.norm3"
    )
    ffned = _fix_dim(
        graph,
        _pick_ffn_or_swiglu(
            graph, normed3, rng, weights, fallback_context=f"{name}.ffn"
        ),
    )
    return _residual(graph, mid2, ffned, context=f"{name}.output")


def tpl_dplr_gated_delta_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {dplr_gated_delta || retention_mix} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="dplr_gated_delta",
        complement_op="retention_mix",
        template_ctx="dplr_gated_delta_block",
    )


def tpl_token_hodge_mixer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {token_hodge_mixer || local_window_attn} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="token_hodge_mixer",
        complement_op="local_window_attn",
        template_ctx="token_hodge_mixer_block",
    )


def tpl_wavelet_packet_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {wavelet_packet_mix || conv_only} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="wavelet_packet_mix",
        complement_op="conv_only",
        template_ctx="wavelet_packet_mix_block",
    )


def tpl_retention_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {retention_mix || dplr_gated_delta} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="retention_mix",
        complement_op="dplr_gated_delta",
        template_ctx="retention_mix_block",
    )


def tpl_product_key_memory_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {product_key_memory || sparsemax_attention} → merge → residual → norm → FFN → residual."""
    return _tpl_novel_mixing_block(
        graph,
        input_id,
        rng,
        weights,
        primary_op="product_key_memory",
        complement_op="sparsemax_attention",
        template_ctx="product_key_memory_block",
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
    ffned = _codex_tail_ffn(graph, tail, rng, context=f"{name}.ffn")
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
    ffned = _codex_tail_ffn(graph, tail, rng, context=f"{name}.ffn")
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
    pb = _add(
        graph,
        "linear_proj",
        [pb],
        {"out_dim": D},
        context="latent_attn_padic_hybrid.padic_project",
    )

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
