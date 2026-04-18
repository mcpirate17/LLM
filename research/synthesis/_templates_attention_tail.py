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
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
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
    tail = _add(graph, "swiglu_mlp", [_add(graph, "rmsnorm", [mid], context=f"{name}.tail_norm")], {"mlp_ratio": 2.0}, context=f"{name}.ffn")
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
    attended = _add(graph, "linear_proj", [attended], {"out_dim": D}, context=f"{name}.attn_proj")
    mid = _residual(graph, input_id, attended, context=f"{name}.mid1")
    refine_in = _add(graph, "rmsnorm", [mid], context=f"{name}.refine_norm")
    proj_a = _add(graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_a")
    proj_b = _add(graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_b")
    refined = _add(graph, "matmul", [proj_a, proj_b], context=f"{name}.refined")
    refined = _add(graph, "linear_proj", [refined], {"out_dim": D}, context=f"{name}.refined_proj")
    refined = _fix_dim(graph, refined)
    mid2 = _residual(graph, mid, refined, context=f"{name}.mid2")
    ffned = _add(graph, "swiglu_mlp", [_add(graph, "rmsnorm", [mid2], context=f"{name}.tail_norm")], {"mlp_ratio": ffn_ratio}, context=f"{name}.ffn")
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context=f"{name}.output")


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
    return _tpl_softmax_matmul_tail(
        graph, input_id, name="attn_softmax_normalized_matmul", ffn_ratio=3.0
    )


def tpl_attn_softmax_normalized_matmul_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {softmax_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel softmax attention + SSM hybrid. Replaces the original
    sequential matmul-bridge-FFN chain with the proven parallel mixing
    pattern that drives top S1 rates.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: softmax attention
    pa = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_softmax_normalized_matmul_v2.softmax_attn",
    )
    pa = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_softmax_normalized_matmul_v2.attn_norm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul_v2.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(
        graph, pa, pb, context="attn_softmax_normalized_matmul_v2.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_softmax_normalized_matmul_v2.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 3.0},
        context="attn_softmax_normalized_matmul_v2.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_softmax_normalized_matmul_v2.output"
    )


def tpl_attn_softmax_normalized_matmul_compact_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that changes only the final FFN width."""
    return _tpl_softmax_matmul_tail(
        graph, input_id, name="attn_softmax_normalized_matmul_compact_ffn", ffn_ratio=2.0
    )


def tpl_attn_softmax_normalized_matmul_fixed_tail_norm(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that fixes the tail norm placement."""
    return _tpl_softmax_matmul_tail(
        graph, input_id, name="attn_softmax_normalized_matmul_fixed_tail_norm", ffn_ratio=3.0
    )

    tail_in = _add(
        graph,
        "rmsnorm",
        [mid2],
        context="attn_softmax_normalized_matmul_fixed_tail_norm.tail_norm",
    )
    ffned = _add(
        graph,
        "swiglu_mlp",
        [tail_in],
        {"mlp_ratio": 3.0},
        context="attn_softmax_normalized_matmul_fixed_tail_norm.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid2,
        ffned,
        context="attn_softmax_normalized_matmul_fixed_tail_norm.output",
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


def tpl_attn_linear_softmax_recovery_control(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {linear_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel hybrid: linear attention captures token interactions while SSM
    provides complementary long-range decay. Follows the winning
    latent_attn_ssm_hybrid pattern.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: linear attention
    pa = _add(
        graph,
        "linear_attention",
        [normed],
        context="attn_linear_softmax_recovery_control.linear_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_linear_softmax_recovery_control.linear_attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(
        graph, pa, pb, context="attn_linear_softmax_recovery_control.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_linear_softmax_recovery_control.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 3.0},
        context="attn_linear_softmax_recovery_control.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_linear_softmax_recovery_control.output"
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


def tpl_attn_linear_no_matmul_ffn_dense_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that swaps only the tail MLP family."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn_dense_tail.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_no_matmul_ffn_dense_tail.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_no_matmul_ffn_dense_tail.refine_norm",
    )
    attended2 = _add(
        graph,
        "softmax_attention",
        [refine_in],
        context="attn_linear_no_matmul_ffn_dense_tail.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn_dense_tail.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(
        graph, mid1, attended2, context="attn_linear_no_matmul_ffn_dense_tail.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "fused_linear_gelu",
        [normed3],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid2,
        ffned,
        context="attn_linear_no_matmul_ffn_dense_tail.output",
    )


def tpl_attn_linear_no_matmul_ffn_direct_recovery(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that removes only the explicit recovery norm."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_direct_recovery.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph,
        input_id,
        attended1,
        context="attn_linear_no_matmul_ffn_direct_recovery.mid1",
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    attended2 = _add(
        graph,
        "softmax_attention",
        [normed2],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(
        graph,
        mid1,
        attended2,
        context="attn_linear_no_matmul_ffn_direct_recovery.mid2",
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
        context="attn_linear_no_matmul_ffn_direct_recovery.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid2,
        ffned,
        context="attn_linear_no_matmul_ffn_direct_recovery.output",
    )


def tpl_attn_softmax_matmul_sparse_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {softmax_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel softmax attention + SSM hybrid. Replaces the original
    matmul+sparse_tail dead-gradient path with the proven parallel mixing
    pattern followed by a full-width FFN.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: softmax attention + proj
    pa = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_softmax_matmul_sparse_tail.softmax_attn",
    )
    pa = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_softmax_matmul_sparse_tail.attn_norm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_softmax_matmul_sparse_tail.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(
        graph, pa, pb, context="attn_softmax_matmul_sparse_tail.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_softmax_matmul_sparse_tail.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    if ffn:
        ffned = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        ffned = _add(
            graph,
            "swiglu_mlp",
            [normed2],
            {"mlp_ratio": 3.0},
            context="attn_softmax_matmul_sparse_tail.ffn",
        )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_softmax_matmul_sparse_tail.output"
    )


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
    attended = _add(
        graph, "graph_attention", [normed], context=f"{name}.attn1"
    )
    attended = _add(
        graph,
        "linear_proj",
        [attended],
        {"out_dim": D},
        context=f"{name}.attn1_proj",
    )
    mid = _residual(graph, input_id, attended, context=f"{name}.mid1")
    refine_in = _add(graph, "rmsnorm", [mid], context=f"{name}.refine_norm")
    proj_a = _add(graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_a")
    proj_b = _add(graph, "linear_proj", [refine_in], {"out_dim": D}, context=f"{name}.proj_b")
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
        graph, "rmsnorm", [input_id],
        context="attn_normalized_matmul_pinned.norm1",
    )

    # Path A: softmax attention → post-attn norm → projection
    pa = _add(
        graph, "softmax_attention", [normed],
        context="attn_normalized_matmul_pinned.softmax_attn",
    )
    pa = _add(
        graph, "rmsnorm", [pa],
        context="attn_normalized_matmul_pinned.attn_postnorm",
    )
    pa = _add(
        graph, "linear_proj", [pa], {"out_dim": D},
        context="attn_normalized_matmul_pinned.attn_proj",
    )

    # Path B: padic_expand → projection
    pb = _add(
        graph, "padic_expand", [normed],
        context="attn_normalized_matmul_pinned.padic",
    )
    pb = _add(
        graph, "linear_proj", [pb], {"out_dim": D},
        context="attn_normalized_matmul_pinned.padic_proj",
    )
    pb = _fix_dim(graph, pb)

    # Merge parallel paths + skip connection
    merged = _residual(graph, pa, pb, context="attn_normalized_matmul_pinned.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="attn_normalized_matmul_pinned.mid")

    # FFN: rmsnorm → swiglu_mlp ratio=4 (pinned — standard transformer FFN)
    normed2 = _add(
        graph, "rmsnorm", [mid],
        context="attn_normalized_matmul_pinned.norm2",
    )
    ffned = _add(
        graph, "swiglu_mlp", [normed2], {"mlp_ratio": 4.0},
        context="attn_normalized_matmul_pinned.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="attn_normalized_matmul_pinned.output")


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
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context=f"{template_ctx}.primary_proj")
    pb = _add(graph, complement_op, [normed], context=f"{template_ctx}.complement")
    pb = _fix_dim(graph, pb)
    merged = _residual(graph, pa, pb, context=f"{template_ctx}.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context=f"{template_ctx}.mid")
    normed2 = _add(graph, "rmsnorm", [mid], context=f"{template_ctx}.norm2")
    ffned = _add(graph, "swiglu_mlp", [normed2], {"mlp_ratio": 4.0}, context=f"{template_ctx}.ffn")
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context=f"{template_ctx}.output")


def tpl_difficulty_routed_attention_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {difficulty_routed_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Uses the dense difficulty-routed attention kernel currently implemented in
    this repo plus an SSM complement.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="difficulty_routed_attention",
        template_ctx="difficulty_routed_attention_block",
    )


def tpl_strided_attention_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {strided_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Multi-head dilated attention: each head uses a different stride (1,2,4,8)
    over key/value positions for multi-scale coverage.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="strided_attention",
        template_ctx="strided_attention_block",
    )


def tpl_gated_progressive_attention_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {gated_progressive_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Dense causal attention with a per-token output gate initialized OFF
    (bias=-2) that learns how much attention output to use.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="gated_progressive_attention",
        template_ctx="gated_progressive_attention_block",
    )


def tpl_gated_linear_attention_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {gated_linear_attention || state_space} → merge → residual → rmsnorm → swiglu → residual.

    GLA: linear attention with data-dependent decay gates. O(nd²) cost.
    Trains in parallel like transformer, infers like RNN. Used by Qwen3-Next.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="gated_linear_attention",
        template_ctx="gated_linear_attention_block",
    )


def tpl_long_conv_hyena_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {long_conv_hyena || gated_linear_attention} → merge → residual → rmsnorm → swiglu → residual.

    Hyena long conv paired with GLA. The FFT convolution handles broad mixing;
    the GLA side handles sharper retrieval.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="long_conv_hyena",
        complement_op="gated_linear_attention",
        template_ctx="long_conv_hyena_block",
    )


def tpl_associative_memory_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {associative_memory || state_space} → merge → residual → rmsnorm → swiglu → residual.

    Modern Hopfield-style retrieval with a learnable temperature, implemented
    here as dense causal query-key-value retrieval.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
        primary_op="associative_memory",
        template_ctx="associative_memory_block",
    )


def tpl_mixture_of_recursions_block(
    graph: ComputationGraph, input_id: int, rng: random.Random, weights: MotifWeights = None,
) -> int:
    """rmsnorm → {mixture_of_recursions || gated_delta} → merge → residual → rmsnorm → swiglu → residual.

    MoR: shared parameter block with a soft depth router over four recurrent
    refinement steps. Paired with gated delta rule for targeted state writes.
    """
    return _tpl_novel_mixing_block(
        graph, input_id, rng, weights,
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
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context=f"{name}.retention_out")

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
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else _add(
        graph, "swiglu_mlp", [normed2], {"mlp_ratio": 3.0}, context=f"{name}.ffn"
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
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else _add(
        graph, "swiglu_mlp", [normed2], {"mlp_ratio": 4.0}, context=f"{name}.ffn"
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
    mid = _residual(
        graph, input_id, recursed, context="recursive_attn_ssm_depth.mid"
    )

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
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
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
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
    mid = _residual(
        graph, input_id, merged, context="graph_attn_ssm_recursive.mid"
    )

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="graph_attn_ssm_recursive.output")


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
    # Retired: no FFN, unstable reciprocal, 0% S1
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
