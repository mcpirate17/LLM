"""Attention template tail: specialized op wrappers and generated variants."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MotifWeights,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _tpl_attention_ffn_block,
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
)
from .motifs import (
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_FFN,
)
from ._selection_utils import with_local_wildcard_probability


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
    if post_op == "log":
        attended = _add(
            graph,
            "sigmoid",
            [attended],
            context=f"{post_op}.bounded_input",
        )
    processed = _add(
        graph,
        post_op,
        [attended],
        post_config or {},
        context=f"{post_op}.post_op",
    )
    if post_op == "log":
        processed = _add(
            graph,
            "rmsnorm",
            [processed],
            context=f"{post_op}.stabilized",
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

    projected = _fix_dim(graph, projected)
    return _residual(graph, input_id, projected, context="attn_decay_sequence.output")


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
    normed = _add(graph, "rmsnorm", [input_id], context="attn_spectral_filter.norm")
    attended = _add(
        graph, "softmax_attention", [normed], context="attn_spectral_filter.attn"
    )
    attended = _add(
        graph, "rmsnorm", [attended], context="attn_spectral_filter.attn_norm"
    )
    filtered = _add(
        graph,
        "spectral_filter",
        [attended],
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
    return _residual(graph, input_id, filtered, context="attn_spectral_filter.output")


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
    scored = _add(
        graph,
        "linear_proj_up",
        [scored],
        {"out_dim": D},
        context="attn_hyperbolic.scored_proj",
    )
    return _residual(graph, input_id, scored, context="attn_hyperbolic.output")


def tpl_attn_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel attention+SSM hybrid with bilinear matmul used as the FFN
    refinement stage. The matmul provides a learned bilinear interaction
    in the FFN slot rather than acting as a bottleneck between residuals.
    """
    D = graph.model_dim
    name = "attn_normalized_matmul"
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
    pa = _add(graph, "softmax_attention", [normed], context=f"{name}.attn")
    record_template_slot_binding(
        graph,
        template_name=name,
        template_instance=template_instance,
        slot_index=1,
        slot_key=f"{name}[{template_instance}].attention",
        slot_classes=("attention_core",),
        selected_name="softmax_attention",
        selected_class="attention_core",
        input_node_id=normed,
    )
    pa = _add(graph, "rmsnorm", [pa], context=f"{name}.attn_norm")
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context=f"{name}.attn_proj")
    pb = _add(graph, "state_space", [normed], context="attn_normalized_matmul.ssm")
    merged = _residual(graph, pa, pb, context="attn_normalized_matmul.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{name}.mid")
    tail = _add(
        graph,
        "swiglu_mlp",
        [_add(graph, "rmsnorm", [mid], context=f"{name}.tail_norm")],
        {"mlp_ratio": 2.0},
        context=f"{name}.ffn",
    )
    tail = _fix_dim(graph, tail)
    return _residual(graph, mid, tail, context=f"{name}.output")


def _tpl_softmax_matmul_tail(
    graph: ComputationGraph,
    input_id: int,
    *,
    name: str,
    ffn_ratio: float,
) -> int:
    D = graph.model_dim
    normed = _add(graph, "rmsnorm", [input_id], context=f"{name}.norm")
    attended = _add(graph, "softmax_attention", [normed], context=f"{name}.attn")
    attended = _add(graph, "rmsnorm", [attended], context=f"{name}.attn_norm")
    attended = _add(
        graph, "linear_proj", [attended], {"out_dim": D}, context=f"{name}.attn_proj"
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
    refined = _add(
        graph, "linear_proj", [refined], {"out_dim": D}, context=f"{name}.refined_proj"
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(graph, mid, refined, context=f"{name}.mid2")
    ffned = _add(
        graph,
        "swiglu_mlp",
        [_add(graph, "rmsnorm", [mid2], context=f"{name}.tail_norm")],
        {"mlp_ratio": ffn_ratio},
        context=f"{name}.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context=f"{name}.output")


def _controlled_attn_block(
    graph: "ComputationGraph",
    input_id: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    name: str,
    attn_op: str,
    D: int,
) -> int:
    """norm → attn_op → optional post-norm → proj → fix_dim → residual.

    Returns mid node id (residual of input_id + attended projection).
    """
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended = _add(graph, attn_op, [normed1], context=f"{name}.attn")
    if attn_op == "softmax_attention":
        attended = _add(graph, "rmsnorm", [attended], context=f"{name}.attn_norm")
    attended = _add(
        graph, "linear_proj", [attended], {"out_dim": D}, context=f"{name}.attn_proj"
    )
    attended = _fix_dim(graph, attended)
    return _residual(graph, input_id, attended, context=f"{name}.mid")


def _controlled_refine_block(
    graph: "ComputationGraph",
    mid: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    name: str,
    use_matmul_refine: bool,
    D: int,
) -> int:
    """norm2 → (matmul-refine | swiglu) → proj → fix_dim → residual.

    Returns mid2 node id (residual of mid + refined projection).
    """
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    refined_in = _add(graph, "rmsnorm", [normed2], context=f"{name}.refine_norm")
    if use_matmul_refine:
        proj_a = _add(
            graph, "linear_proj", [refined_in], {"out_dim": D}, context=f"{name}.proj_a"
        )
        proj_b = _add(
            graph, "linear_proj", [refined_in], {"out_dim": D}, context=f"{name}.proj_b"
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
        graph, "linear_proj", [refined], {"out_dim": D}, context=f"{name}.refined_proj"
    )
    refined = _fix_dim(graph, refined)
    return _residual(graph, mid, refined, context=f"{name}.mid2")


def _tpl_controlled_attn_matmul_ablation(
    graph: "ComputationGraph",
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
    mid = _controlled_attn_block(
        graph, input_id, rng, weights, name=name, attn_op=attn_op, D=D
    )
    mid2 = _controlled_refine_block(
        graph, mid, rng, weights, name=name, use_matmul_refine=use_matmul_refine, D=D
    )
    tail_in = (
        mid2
        if tail_kind == "router_sidecar"
        else _add(graph, "rmsnorm", [mid2], context=f"{name}.tail_norm")
    )

    if tail_kind == "dense":
        tail = _add(
            graph, "swiglu_mlp", [tail_in], {"mlp_ratio": 3.0}, context=f"{name}.tail"
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
            graph, "difficulty_blend_3way", [mid2, mid2], context=f"{name}.route_mix"
        )
        routed = _residual(graph, mid2, routed, context=f"{name}.route_mid")
        routed = _add(
            graph, "depth_weighted_proj", [routed], context=f"{name}.route_proj"
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


# ── Split modules re-export ─────────────────────────────────────────
# Standalone variants live in private split modules to stay under the
# 1250-line file cap. Factory-generated templates remain in this module
# so `tpl_attn_dual_axis.__module__` etc. continue to point here.

from ._templates_attention_matmul import (  # noqa: E402,F401
    tpl_attn_softmax_normalized_matmul,
    tpl_attn_softmax_normalized_matmul_v2,
    tpl_attn_softmax_normalized_matmul_compact_ffn,
    tpl_attn_softmax_normalized_matmul_fixed_tail_norm,
    tpl_attn_linear_normalized_matmul_control,
    tpl_attn_linear_softmax_recovery_control,
    tpl_attn_linear_no_matmul_ffn,
    tpl_attn_linear_no_matmul_ffn_dense_tail,
    tpl_attn_linear_no_matmul_ffn_direct_recovery,
    tpl_attn_softmax_matmul_sparse_tail,
    tpl_attn_linear_matmul_sparse_tail,
    tpl_attn_linear_matmul_router_sidecar,
)
from ._templates_attention_hybrid import (  # noqa: E402,F401
    tpl_linear_attn_ffn_block,
    tpl_linear_attn_sparse_ffn,
    tpl_graph_attn_sparse_ffn,
    tpl_latent_attn_conv_hybrid,
    tpl_diff_attn_conv_hybrid,
    tpl_attn_safe_division,
    tpl_latent_attn_ssm_hybrid,
    tpl_local_attn_ssm_hybrid,
    tpl_attn_spiking_hybrid,
    tpl_local_attn_moe,
    tpl_attn_normalized_matmul_pinned,
)
from ._templates_attention_advanced import (  # noqa: E402,F401
    tpl_difficulty_routed_attention_block,
    tpl_strided_attention_block,
    tpl_gated_progressive_attention_block,
    tpl_gated_linear_attention_block,
    tpl_long_conv_hyena_block,
    tpl_associative_memory_block,
    tpl_dplr_gated_delta_block,
    tpl_entmax_attention_block,
    tpl_learnable_semiring_attention_block,
    tpl_mixture_of_recursions_block,
    tpl_phase_lock_attention_block,
    tpl_product_key_memory_block,
    tpl_reciprocal_rank_attention_block,
    tpl_reciprocal_semiring_attention_block,
    tpl_retention_mix_block,
    tpl_sparsemax_attention_block,
    tpl_stdp_reciprocal_memory_block,
    tpl_token_hodge_mixer_block,
    tpl_wavelet_packet_mix_block,
    tpl_codex_ssm_retention_block,
    tpl_codex_ssm_delta_memory_block,
    tpl_codex_ssm_mla_gated_block,
    tpl_codex_ssm_local_recall_block,
    tpl_recursive_attn_ssm_depth,
    tpl_latent_attn_padic_hybrid,
    tpl_graph_attn_ssm_recursive,
)


# ── Factory-generated templates (must live in this module) ─────────
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


_ATTN_OP_CHAIN_TEMPLATE_SPECS = {
    # Retired: no FFN, unstable reciprocal, 0% S1
    "attn_reciprocal_gated": {
        "factory": _make_attn_op_chain_template,
        "factory_arg": "reciprocal",
    },
    "attn_kronecker_hybrid": {
        "factory": _make_attn_op_chain_template,
        "factory_arg": "kronecker_linear",
    },
    "attn_log_gated": {
        "factory": _make_attn_op_chain_template,
        "factory_arg": "log",
    },
}

_MOE_CLASSES = (MOTIF_CLASS_MOE, MOTIF_CLASS_GATE)

# Phase 3.2 (2026-05-04) Bucket C rescue tightenings — fallbacks below.
# Phase 4.1 (2026-05-04) makes these data-driven via _slot_constraints_loader.
# At import time we ask the loader for the empirical pass-cohort qualifying
# motif_classes (n>=5, conditional pass_rate >= 0.60). If the meta DB is
# unreachable or the slot has no qualifying classes, the loader returns the
# fallback tuple verbatim — keeping the pre-Phase-4.1 hand-curated behavior.
from ._slot_constraints_loader import derive_slot_classes  # noqa: E402

_LATENT_ATTN_SPARSE_FFN_FFN_CLASSES_FALLBACK = (MOTIF_CLASS_CONV, MOTIF_CLASS_FFN)
_LATENT_ATTN_MOE_FFN_CLASSES_FALLBACK = (
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_MOE,
)
_LATENT_ATTN_SPARSE_FFN_FFN_CLASSES = derive_slot_classes(
    "latent_attn_sparse_ffn", 2, _LATENT_ATTN_SPARSE_FFN_FFN_CLASSES_FALLBACK
)
_LATENT_ATTN_MOE_FFN_CLASSES = derive_slot_classes(
    "latent_attn_moe", 2, _LATENT_ATTN_MOE_FFN_CLASSES_FALLBACK
)

_ATTN_FFN_TEMPLATE_SPECS = {
    "attn_dual_axis": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {},
    },
    "latent_attn_ffn_block": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {"attn_op": "latent_attention_compressor"},
    },
    "diff_attn_ffn_block": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {"attn_op": "diff_attention"},
    },
    "latent_attn_sparse_ffn": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {
            "attn_op": "latent_attention_compressor",
            # Phase 3.2 (2026-05-04) tightened from _SPARSE_FFN_CLASSES =
            # (SPARSE, EFFICIENT_PROJ, GATE). Empirical fail cohort fills
            # (kronecker_proj, codebook_proj, proj_tied, bottleneck_sparse,
            # ffn_bottleneck, proj_shared_basis) all came via SPARSE/
            # EFFICIENT_PROJ; pass cohort dominated by conv_swiglu (CONV).
            # See research/reports/slot_tightening_proposal.json.
            "ffn_classes": _LATENT_ATTN_SPARSE_FFN_FFN_CLASSES,
        },
    },
    "graph_attn_ffn_block": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {"attn_op": "graph_attention"},
    },
    "attn_moe_block": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {"ffn_classes": _MOE_CLASSES},
    },
    "latent_attn_moe": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {
            "attn_op": "latent_attention_compressor",
            # Phase 3.2 (2026-05-04) tightened from _MOE_CLASSES = (MOE, GATE).
            # Pass cohort uses channel_rwkv (CHANNEL) + conv_swiglu (CONV);
            # fail cohort uses act_log_*/kronecker_proj/codebook_proj.
            # See research/reports/slot_tightening_proposal.json.
            "ffn_classes": _LATENT_ATTN_MOE_FFN_CLASSES,
        },
    },
    "diff_attn_moe": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {
            "attn_op": "diff_attention",
            "ffn_classes": _MOE_CLASSES,
        },
    },
    "graph_attn_moe": {
        "factory": _make_attn_ffn_template,
        "factory_kwargs": {
            "attn_op": "graph_attention",
            "ffn_classes": _MOE_CLASSES,
        },
    },
}

_GENERATED_ATTN_TEMPLATE_SPECS = {
    **_ATTN_OP_CHAIN_TEMPLATE_SPECS,
    **_ATTN_FFN_TEMPLATE_SPECS,
}


def _build_generated_attn_templates() -> dict[str, callable]:
    generated: dict[str, callable] = {}
    for name, spec in _GENERATED_ATTN_TEMPLATE_SPECS.items():
        factory = spec["factory"]
        if "factory_arg" in spec:
            generated[name] = factory(spec["factory_arg"])
        else:
            generated[name] = factory(**dict(spec.get("factory_kwargs") or {}))
    return generated


GENERATED_ATTN_TEMPLATE_EXPORTS = _build_generated_attn_templates()

# The registry only consumes the live subset. Extra generated wrappers stay
# exported for compatibility but do not participate in template sampling.
GENERATED_ATTN_REGISTRY_TEMPLATES = {
    name: GENERATED_ATTN_TEMPLATE_EXPORTS[name]
    for name in (
        "attn_kronecker_hybrid",
        "attn_log_gated",
        "latent_attn_ffn_block",
        "diff_attn_ffn_block",
        "latent_attn_sparse_ffn",
        "graph_attn_ffn_block",
        "latent_attn_moe",
        "diff_attn_moe",
        "graph_attn_moe",
    )
}

GENERATED_ATTN_DEFAULT_WEIGHTS = {
    "latent_attn_ffn_block": 4.0,
    "diff_attn_ffn_block": 3.5,
    "latent_attn_sparse_ffn": 4.0,
    "graph_attn_ffn_block": 5.25,
    "attn_kronecker_hybrid": 3.0,
    "attn_log_gated": 3.0,
    "latent_attn_moe": 4.0,
    "diff_attn_moe": 3.5,
    "graph_attn_moe": 5.25,
}

globals().update(
    {f"tpl_{name}": fn for name, fn in GENERATED_ATTN_TEMPLATE_EXPORTS.items()}
)
