"""Novel Mixer Templates — identifying and creating novel, content-addressed mixing mechanisms."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MotifWeights,
    _fix_dim,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_attention_advanced import (
    _pick_norm_or_default,
    _pick_ffn_or_swiglu,
)


def tpl_clifford_geometric_mixer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {clifford_attention || geometric_product} → rotor_transform → residual → norm → FFN → residual.

    Novelty: Global mixing in Clifford space via Clifford attention (Cl(3,0)), followed by
    geometric interaction of multivector components. This achieves content-addressed
    retrieval with geometric inductive biases.
    """
    D = graph.model_dim
    template_ctx = "clifford_geometric_mixer_block"

    # 1. First block: Novel Mixing
    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    # Branch 1: Clifford Attention (Global content-based mixing)
    ca = _add(
        graph, "clifford_attention", [normed], context=f"{template_ctx}.clifford_attn"
    )

    # Branch 2: Geometric product of two multivector views. rotor_transform
    # consumes real input directly and emits multivectors, so no linear_proj
    # prefix is needed (kept lean to fit the generation op/depth budget). One
    # view comes from the normed input, the other from the clifford-attention
    # output, giving a content-dependent geometric interaction.
    ra = _add(graph, "rotor_transform", [normed], context=f"{template_ctx}.rotor_a")
    rb = _add(graph, "rotor_transform", [ca], context=f"{template_ctx}.rotor_b")
    gp = _add(graph, "geometric_product", [ra, rb], context=f"{template_ctx}.geom_prod")

    # Versor sandwich, then grade_select back to canonical (real) space.
    versor = _add(graph, "versor_apply", [gp, ra], context=f"{template_ctx}.versor")
    refined = _add(graph, "grade_select", [versor], context=f"{template_ctx}.grade_sel")
    # The multivector→real grade projection is unbounded and was collapsing the
    # loss (loss_ratio ~0.13, S1 fail). Normalise the geometric lane before merge.
    refined = _add(graph, "rmsnorm", [refined], context=f"{template_ctx}.grade_norm")

    # Merge clifford-attention lane with the geometric-product lane, project, residual.
    merged = _add(graph, "add", [ca, refined], context=f"{template_ctx}.merge")
    refined = _add(
        graph,
        "linear_proj",
        [merged],
        {"out_dim": D},
        context=f"{template_ctx}.refine",
    )
    mid = _residual(graph, input_id, refined, context=f"{template_ctx}.resid1")

    # 2. Second block: FFN
    normed2 = _pick_norm_or_default(
        graph, mid, rng, weights, fallback_context=f"{template_ctx}.norm2"
    )
    ffn = _pick_ffn_or_swiglu(
        graph, normed2, rng, weights, fallback_context=f"{template_ctx}.ffn"
    )
    ffned = _fix_dim(graph, ffn)
    return _residual(graph, mid, ffned, context=f"{template_ctx}.output")


def tpl_tropical_maxplus_mixer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {tropical_attention || associative_memory} → tropical_gate → residual → norm → FFN → residual.

    Novelty: Global mixing in the Max-plus semiring (Tropical geometry) paired with
    explicit associative retrieval. Supports robust content-addressed binding.
    """
    D = graph.model_dim
    template_ctx = "tropical_maxplus_mixer_block"

    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    # Branch 1: Tropical Attention (Global Max-Plus mixing)
    ta = _add(
        graph, "tropical_attention", [normed], context=f"{template_ctx}.tropical_attn"
    )
    ta = _add(graph, "tropical_gate", [ta], context=f"{template_ctx}.tropical_gate")
    ta = _add(graph, "tropical_center", [ta], context=f"{template_ctx}.tropical_center")
    # tropical_center must be consumed by linear_proj_down: it is the only op in
    # the math-space must_follow_with set that context rules also permit after
    # tropical_center (plain linear_proj / tropical_gate are blocked there).
    ta = _add(
        graph,
        "linear_proj_down",
        [ta],
        {"out_dim": D},
        context=f"{template_ctx}.tropical_proj",
    )

    # Branch 2: Associative Memory (Verified global mixer)
    am = _add(
        graph, "associative_memory", [normed], context=f"{template_ctx}.assoc_mem"
    )

    # Merge full-range tropical retrieval with associative retrieval.
    merged = _add(graph, "add", [ta, am], context=f"{template_ctx}.merge")

    refined = _add(
        graph, "linear_proj", [merged], {"out_dim": D}, context=f"{template_ctx}.refine"
    )
    mid = _residual(graph, input_id, refined, context=f"{template_ctx}.resid1")

    # 2. Second block: FFN
    normed2 = _pick_norm_or_default(
        graph, mid, rng, weights, fallback_context=f"{template_ctx}.norm2"
    )
    ffn = _pick_ffn_or_swiglu(
        graph, normed2, rng, weights, fallback_context=f"{template_ctx}.ffn"
    )
    ffned = _fix_dim(graph, ffn)
    return _residual(graph, mid, ffned, context=f"{template_ctx}.output")


def tpl_ultrametric_hierarchical_ensemble_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {ultrametric_attention || padic_expand || state_space} → residual → norm → FFN → residual.

    Novelty: Combines hierarchical p-adic distance-based global mixing with
    a separate p-adic feature expansion and SSM support lane.
    """
    D = graph.model_dim
    template_ctx = "ultrametric_hierarchical_ensemble_block"

    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    # Branch 1: Ultrametric (Hierarchical)
    ua = _add(
        graph,
        "ultrametric_attention",
        [normed],
        context=f"{template_ctx}.ultrametric_attn",
    )

    # Branch 2: p-adic feature expansion. Kept parallel to ultrametric_attention:
    # context rules reject padic_expand -> ultrametric_attention directly.
    expanded = _add(
        graph, "padic_expand", [normed], context=f"{template_ctx}.padic_expand"
    )
    # padic_expand must be consumed by linear_proj (math-space rule) before the
    # merge.
    expanded = _add(
        graph,
        "linear_proj",
        [expanded],
        {"out_dim": D},
        context=f"{template_ctx}.padic_proj",
    )

    # Branch 3: SSM support lane for long-context carrier state.
    gla = _add(graph, "state_space", [normed], context=f"{template_ctx}.ssm")

    # Merge
    merged = _add(graph, "add", [ua, expanded], context=f"{template_ctx}.padic_merge")
    merged = _add(graph, "add", [merged, gla], context=f"{template_ctx}.merge")

    refined = _add(
        graph, "linear_proj", [merged], {"out_dim": D}, context=f"{template_ctx}.refine"
    )
    mid = _residual(graph, input_id, refined, context=f"{template_ctx}.resid1")

    normed2 = _pick_norm_or_default(
        graph, mid, rng, weights, fallback_context=f"{template_ctx}.norm2"
    )
    ffn = _pick_ffn_or_swiglu(
        graph, normed2, rng, weights, fallback_context=f"{template_ctx}.ffn"
    )
    ffned = _fix_dim(graph, ffn)
    return _residual(graph, mid, ffned, context=f"{template_ctx}.output")
