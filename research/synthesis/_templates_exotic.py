"""Binary-op safety, math-space, and spiking templates."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MOTIFS_BY_CLASS,
    MotifWeights,
    _FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _motif_is_compatible,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _tpl_norm_dual_op_residual,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_core import tpl_residual_block


# ── Binary-Op Safety Templates ───────────────────────────────────
#
# These templates make binary UNSAFE ops reachable by providing the
# structural context that guarantees numerical safety.


def tpl_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → matmul → fix_dim → residual."""
    return _tpl_norm_dual_op_residual(graph, input_id, rng, weights, merge_op="matmul")


def tpl_gated_product(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """feature → sigmoid(gate) → outer_product(feature, gate) → norm → residual.

    One input bounded by sigmoid ⇒ product bounded.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        feature = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate_sig = graph.add_op("sigmoid", [gate])
        product = graph.add_op("outer_product", [feature, gate_sig])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    product = _fix_dim(graph, product)
    try:
        return graph.add_op("add", [input_id, product])
    except ValueError:
        return product


def tpl_safe_division(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """numerator → softmax(denom) → div_safe(num, denom) → proj → residual.

    Denominator from softmax ⇒ always > 0, sums to 1.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        numerator = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        denom_raw = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        denom = graph.add_op("softmax_last", [denom_raw])
        divided = graph.add_op("div_safe", [numerator, denom])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    divided = _fix_dim(graph, divided)
    try:
        return graph.add_op("add", [input_id, divided])
    except ValueError:
        return divided


def tpl_cosine_scoring(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → cosine_similarity → fix_dim → residual."""
    return _tpl_norm_dual_op_residual(
        graph, input_id, rng, weights, merge_op="cosine_similarity"
    )


def tpl_decay_sequence(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → value_proj → decay_weights(sigmoid→cumprod) → mul(value, decay) → proj → [FFN] → residual.

    Exponential decay weighting: cumprod(sigmoid(x)) produces monotonically
    decaying weights along the sequence. These weights are applied to projected
    values via element-wise multiply — similar to RWKV time-decay mixing.
    Previous pattern fed signal directly through cumprod, which decayed to zero.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        # Value branch: what to weight
        value = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        # Decay branch: how to weight (sigmoid bounds to (0,1), cumprod decays)
        decay_proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        decay_gate = graph.add_op("sigmoid", [decay_proj])
        decay_weights = graph.add_op("cumprod_safe", [decay_gate])
        # Apply decay weighting to values
        weighted = graph.add_op("mul", [value, decay_weights])
        projected = graph.add_op("linear_proj", [weighted], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_hyp_distance_scoring(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """exp_map(a) → exp_map(b) → hyp_distance(a,b) → linear_proj_up → residual.

    Hyperbolic distance scoring: two projections into Poincaré ball, distance
    reduces to dim=1, linear_proj_up restores to model_dim.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        map_a = graph.add_op("exp_map", [proj_a])
        map_b = graph.add_op("exp_map", [proj_b])
        dist = graph.add_op("hyp_distance", [map_a, map_b])
        out = graph.add_op("linear_proj_up", [dist], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_tropical_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → proj_a → proj_b → tropical_add(a,b) → linear_proj → residual.

    Tropical semiring addition (element-wise min) over two projections.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        tadd = graph.add_op("tropical_add", [proj_a, proj_b])
        out = graph.add_op("linear_proj", [tadd], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_tropical_center_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """layernorm → tropical_attention → tropical_gate → tropical_center → residual.

    Full tropical stack: the proven architecture from 11 S1-passing programs
    (best lr=0.079). tropical_attention provides min-plus sequence mixing,
    tropical_gate does shortest-path routing, tropical_center removes the
    running minimum baseline. All three are needed for learning.

    Satisfies MATH_SPACE_RULES:
      tropical_attention.must_precede = {rmsnorm, layernorm}  ✓
      tropical_attention.must_follow_with includes tropical_center  ✓
      tropical_gate.must_precede = {rmsnorm, layernorm}  ✓ (via tropical_attention)
      tropical_gate.must_follow_with includes tropical_center  ✓
      tropical_center.must_follow = {tropical_attention, tropical_gate}  ✓
      tropical_center.must_follow_with includes linear_proj  ✓
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph,
        "tropical_attention",
        [normed],
        context="tropical_center_block.attention",
    )
    gated = _add(
        graph,
        "tropical_gate",
        [attended],
        context="tropical_center_block.gate",
    )
    centered = _add(
        graph,
        "tropical_center",
        [gated],
        context="tropical_center_block.center",
    )
    out = _add(
        graph,
        "linear_proj_down",
        [centered],
        {"out_dim": D},
        context="tropical_center_block.project",
    )
    out = _add(graph, "rmsnorm", [out], context="tropical_center_block.post_norm")

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_geometric_product_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → rotor_a → rotor_b → geometric_product(a,b) → grade_select → proj → [FFN] → residual.

    Clifford geometric product: rotor_transform bridges euclidean→multivector.
    FFN motif provides capacity to interpret the multivector output.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj_a = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="geometric_product_block.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="geometric_product_block.proj_b",
    )
    rotor_a = _add(
        graph,
        "rotor_transform",
        [proj_a],
        context="geometric_product_block.rotor_a",
    )
    rotor_b = _add(
        graph,
        "rotor_transform",
        [proj_b],
        context="geometric_product_block.rotor_b",
    )
    product = _add(
        graph,
        "geometric_product",
        [rotor_a, rotor_b],
        context="geometric_product_block.product",
    )
    selected = _add(
        graph,
        "grade_select",
        [product],
        context="geometric_product_block.grade_select",
    )
    projected = _add(
        graph,
        "linear_proj",
        [selected],
        {"out_dim": D},
        context="geometric_product_block.project",
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_geometric_product_versor_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → versor_proj → rotor_transform → versor_apply(versor, mv) → grade_select → proj → residual.

    Phase 5 V2 Clifford companion (2026-05-04). Pairs the proven
    ``geometric_product`` (op_certification: 84.7% pass solo, n=150) with
    the new ``versor_apply`` op (sandwich product v · mv · v⁻¹) — the
    canonical Clifford rotation primitive. Differs from the existing
    ``geometric_product_block`` by passing a versor explicitly through
    the sandwich pipeline rather than two parallel rotors → product.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Versor source — independent linear projection bridges euclidean → multivector
    versor_proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="geometric_product_versor_block.versor_proj",
    )
    versor = _add(
        graph,
        "rotor_transform",
        [versor_proj],
        context="geometric_product_versor_block.versor",
    )
    # Sandwich application: rotates the input multivector by the versor
    sandwiched = _add(
        graph,
        "versor_apply",
        [versor, normed],
        context="geometric_product_versor_block.sandwich",
    )
    selected = _add(
        graph,
        "grade_select",
        [sandwiched],
        context="geometric_product_versor_block.grade_select",
    )
    projected = _add(
        graph,
        "linear_proj",
        [selected],
        {"out_dim": D},
        context="geometric_product_versor_block.project",
    )
    projected = _fix_dim(graph, projected)
    return _residual(
        graph, input_id, projected, context="geometric_product_versor_block.output"
    )


def tpl_residual_difference(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → sub(a,b) → proj → [FFN] → residual.

    Difference-based feature extraction (contrastive representation).
    FFN motif provides capacity to learn from the contrastive signal.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        diff = graph.add_op("sub", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [diff], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_tropical_matmul_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → tropical_matmul(a,b) → proj → [FFN] → residual.

    Tropical (min,+) matmul computes shortest-path distances.
    FFN motif re-densifies gradients after sparse min-plus operation.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        result = graph.add_op("tropical_matmul", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [result], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_tropical_softmax_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → score_proj → tropical_softmax → mul(input) → proj → [FFN] → residual.

    Uses the gradient-friendly softmin (tropical_softmax) as a per-token,
    per-feature gating distribution: low scores get high mass. Drop-in for
    sigmoid/softmax gating in tropical-flavored contexts.

    Per external_research_2026-05-10.md §3.5 — tropical_softmax replaces
    hard max with LogSumExp-temperature, preserving gradient flow.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        scores = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("tropical_softmax", [scores])
        gated = graph.add_op("mul", [normed, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_gated_minimum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → minimum(a,b) → proj → [FFN] → residual.

    Element-wise minimum for competitive feature selection.
    FFN motif provides downstream capacity to interpret min-selected features.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        minned = graph.add_op("minimum", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [minned], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── Phase 3: Component research templates ────────────────────────────
# Dedicated templates for 24 underperforming ops. Each template
# provides a guaranteed path for ops that otherwise never get selected.
# Built from debug harness profiling (2026-03-21): all ops verified
# compile+forward+backward in these contexts.

_SPIKING_OPS = frozenset(
    {"lif_neuron", "spike_rate_code", "sparse_threshold", "stdp_attention"}
)
_HYPERBOLIC_OPS = frozenset(
    {"exp_map", "hyp_linear", "hyp_tangent_nonlinear", "log_map"}
)


def tpl_spiking_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [spiking motif] → linear_proj → residual_add.

    Forces selection of spiking motifs (lif_neuron → spike_rate_code or
    lif_neuron → sparse_threshold → stdp_attention chains).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Filter to spiking motifs only
    spiking_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
        if any(s.op_name in _SPIKING_OPS for s in m.steps)
        and _motif_is_compatible(graph, normed, m)
    ]
    if spiking_motifs:
        motif = rng.choice(spiking_motifs)
        processed = _instantiate_motif(graph, normed, motif, rng)
    else:
        processed = normed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_spiking_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [spiking_op] → tropical_gate → linear_proj → residual_add.

    Proven pattern: spiking encoding → tropical routing → projection.
    spike_rate_code + tropical_moe achieved lr=0.007 (best spiking result).
    Pre-norm stabilizes activations for deep stacking.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Pick a spiking+tropical motif
    spiking_tropical_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
        if m.name
        in {
            "spiking_tropical_gate",
            "spiking_rate_tropical_gate",
            "spiking_threshold_tropical_gate",
        }
        and _motif_is_compatible(graph, normed, m)
    ]
    if spiking_tropical_motifs:
        motif = rng.choice(spiking_tropical_motifs)
        processed = _instantiate_motif(graph, normed, motif, rng)
    else:
        # Fallback: manual construction
        spiking_ops = ["lif_neuron", "spike_rate_code"]
        spike_op = rng.choice(spiking_ops)
        try:
            spiked = graph.add_op(spike_op, [normed])
            gated = graph.add_op("tropical_gate", [spiked])
            D = graph.model_dim
            processed = graph.add_op("linear_proj", [gated], config={"out_dim": D})
        except (ValueError, KeyError):
            return tpl_residual_block(graph, input_id, rng, weights)

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_hyperbolic_bridge_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → exp_map → hyp_linear → hyp_tangent_nonlinear → log_map → proj → residual.

    Forces the full Poincaré ball round-trip via hyperbolic_residual_bridge motif.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Filter to hyperbolic motifs
    hyp_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_SSM, [])
        if any(s.op_name in _HYPERBOLIC_OPS for s in m.steps)
        and _motif_is_compatible(graph, normed, m)
    ]
    if not hyp_motifs:
        # Also check math_space class
        hyp_motifs = [
            m
            for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
            if any(s.op_name in _HYPERBOLIC_OPS for s in m.steps)
            and _motif_is_compatible(graph, normed, m)
        ]
    if hyp_motifs:
        motif = rng.choice(hyp_motifs)
        processed = _instantiate_motif(graph, normed, motif, rng)
    else:
        processed = normed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_poincare_add_bridge(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj branches → exp_map → poincare_add → log_map → [FFN] → residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    branch_a = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="poincare_add_bridge.branch_a",
    )
    branch_b = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="poincare_add_bridge.branch_b",
    )
    if rng.random() < 0.5:
        branch_b = _add(
            graph,
            rng.choice(["silu", "gelu", "tanh"]),
            [branch_b],
            context="poincare_add_bridge.branch_b_act",
        )
    if rng.random() < 0.35:
        branch_b = _add(
            graph,
            "linear_proj",
            [branch_b],
            {"out_dim": D},
            context="poincare_add_bridge.branch_b_project",
        )

    mixed = _residual(graph, branch_a, branch_b, context="poincare_add_bridge.mix")
    tangent = _add(
        graph,
        "linear_proj",
        [mixed],
        {"out_dim": D},
        context="poincare_add_bridge.tangent_project",
    )
    mapped = _add(graph, "exp_map", [tangent], context="poincare_add_bridge.exp")
    bridged = _add(
        graph,
        "poincare_add",
        [mapped],
        context="poincare_add_bridge.poincare_add",
    )
    current = _add(graph, "log_map", [bridged], context="poincare_add_bridge.log")

    if rng.random() < 0.7:
        post = _pick_compatible_motif_from_classes(
            graph, current, rng, _FFN_CLASSES, weights
        )
        if post:
            current = _instantiate_motif(graph, current, post, rng)

    current = _fix_dim(graph, current)
    try:
        return graph.add_op("add", [input_id, current])
    except ValueError:
        return current


def tpl_n_way_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention →] n_way_sparse_router → norm → [FFN motif] → residual_add.

    N-way sparse routing with bottleneck experts. 40% chance of attention
    before the MoE routing for global context.
    """
    from ._template_helpers import MOTIF_CLASS_ATTENTION

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # 40% chance: attention before MoE routing
    current = normed
    if rng.random() < 0.4:
        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            attended = _instantiate_motif(graph, normed, attn, rng)
            current = _fix_dim(graph, attended)

    # Pick n_ways that divides D
    safe_n_ways = [n for n in [2, 4, 8] if D % n == 0]
    if not safe_n_ways:
        safe_n_ways = [2]
    n_ways = rng.choice(safe_n_ways)

    try:
        routed = graph.add_op(
            "sparse_bottleneck_moe",
            [current],
            config={"n_ways": n_ways, "top_k": min(2, n_ways)},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Post-routing norm + FFN
    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        processed = normed2

    processed = _fix_dim(graph, processed)
    # Bound MoE-routed expert outputs (which can spike when one expert
    # dominates) before injecting back into the residual stream.
    processed = _add(
        graph, "rmsnorm", [processed], context="n_way_moe_block.output_norm"
    )
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_conv_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv_only → [FFN motif] → residual_add.

    Local depthwise conv for short-range mixing + FFN for learning.
    Satisfies conv_only MATH_SPACE_RULES (must_precede norm, must_follow_with proj).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        conved = graph.add_op("conv_only", [normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, conved, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, conved, ffn, rng)
    else:
        processed = conved

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_causal_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → causal_mask → [attention motif] → proj → [FFN motif] → residual_add.

    Causal cumulative average as a lightweight O(S*D) causal mixer.
    50% chance of attention after the causal mask for richer mixing.
    """

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        mixed = graph.add_op("causal_mask", [normed])
        projected = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Keep the post-causal mixer inside a grammar-safe attention scaffold.
    if rng.random() < 0.5:
        attended = _add(
            graph,
            "softmax_attention",
            [projected],
            context="causal_mix_block.attn",
        )
        attended = _add(
            graph,
            "rmsnorm",
            [attended],
            context="causal_mix_block.attn_norm",
        )
        attended = _add(
            graph,
            "linear_proj",
            [attended],
            {"out_dim": D},
            context="causal_mix_block.attn_proj",
        )
        projected = _fix_dim(graph, attended)

    processed = _add(
        graph,
        "swiglu_mlp",
        [projected],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="causal_mix_block.ffn",
    )

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_iterative_refinement(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention →] fixed_point_iter → linear_proj → residual_add.

    Damped fixed-point iteration: z = (1-d)*z + d*tanh(z@W+b), repeated n_iters times.
    50% chance of attention before iteration for global context seeding.
    """
    from ._template_helpers import MOTIF_CLASS_ATTENTION

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # 50% chance: attention pre-seeds the iterative refinement
    current = normed
    if rng.random() < 0.5:
        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            attended = _instantiate_motif(graph, normed, attn, rng)
            current = _fix_dim(graph, attended)

    try:
        refined = graph.add_op(
            "fixed_point_iter",
            [current],
            config={
                "n_iters": rng.choice([2, 3]),
                "damping": round(rng.uniform(0.3, 0.7), 2),
            },
        )
        projected = graph.add_op("linear_proj", [refined], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_recurrent_delta_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [attention →] gated_delta → proj → [FFN motif] → residual_add.

    GatedDeltaNet: recurrent outer-product state with gated decay/update.
    40% chance of attention before delta for global context.
    """
    from ._template_helpers import MOTIF_CLASS_ATTENTION

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # 40% chance: attention before delta recurrence
    current = normed
    if rng.random() < 0.4:
        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            attended = _instantiate_motif(graph, normed, attn, rng)
            current = _fix_dim(graph, attended)

    try:
        recurrent = graph.add_op("gated_delta", [current])
        projected = graph.add_op("linear_proj", [recurrent], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Optional FFN after recurrence
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── Bucket D mines (Phase 3.1, 2026-05-04) ─────────────────────────
#
# Three new templates extracted from passing-cohort sub-patterns that pair an
# `exotic_functional` op (passes solo at >=60% with no other mixer in graph)
# with a certified `mixer` op. Empirical evidence in
# research/reports/op_mixer_certification_v2.csv +
# research/reports/new_templates_proposal.json.


def _tpl_exotic_after_mixer(
    graph: "ComputationGraph",
    input_id: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    exotic_op: str,
    name: str,
    add_post_proj: bool = True,
) -> int:
    """Shared body: norm -> conv1d_seq -> residual -> norm -> exotic -> [proj] -> residual."""
    D = graph.model_dim
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    mixed = _add(graph, "conv1d_seq", [normed1], context=f"{name}.mixer")
    mixed = _fix_dim(graph, mixed)
    mid = _residual(graph, input_id, mixed, context=f"{name}.mixer_resid")
    norm2 = _add(graph, "rmsnorm", [mid], context=f"{name}.norm2")
    refined = _add(graph, exotic_op, [norm2], context=f"{name}.exotic")
    if add_post_proj:
        refined = _add(
            graph,
            "linear_proj",
            [refined],
            {"out_dim": D},
            context=f"{name}.refine_proj",
        )
    refined = _fix_dim(graph, refined)
    return _residual(graph, mid, refined, context=f"{name}.exotic_resid")


def tpl_tropical_attn_conv1d_seq_block(
    graph: "ComputationGraph",
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv1d_seq → residual → norm → tropical_attention → linear_proj → residual.

    Bucket D mine: tropical_attention (exotic_functional, solo pass=0.67) +
    conv1d_seq (mixer, solo pass=0.84). Sub-pattern observed in passing graphs
    of `conditional_compute` and `mixed_recursion`.
    """
    return _tpl_exotic_after_mixer(
        graph,
        input_id,
        rng,
        weights,
        exotic_op="tropical_attention",
        name="tropical_attn_conv1d_seq_block",
    )


def tpl_rwkv_channel_conv1d_seq_block(
    graph: "ComputationGraph",
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv1d_seq → residual → norm → rwkv_channel → linear_proj → residual.

    Bucket D mine: rwkv_channel (exotic_functional, solo pass=0.81) + conv1d_seq.
    Sub-pattern from passing `recursive_depth_router` and `residual_block` graphs.
    """
    return _tpl_exotic_after_mixer(
        graph,
        input_id,
        rng,
        weights,
        exotic_op="rwkv_channel",
        name="rwkv_channel_conv1d_seq_block",
    )


def tpl_matmul_conv1d_seq_block(
    graph: "ComputationGraph",
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv1d_seq → residual → norm → proj_a × proj_b (matmul) → linear_proj → residual.

    Bucket D mine: matmul refinement (exotic_functional, solo pass=0.71) +
    conv1d_seq. Sub-pattern from passing `token_merge_block` graphs. Matmul is
    binary, so two parallel projections feed it.
    """
    D = graph.model_dim
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    mixed = _add(graph, "conv1d_seq", [normed1], context="matmul_conv1d.mixer")
    mixed = _fix_dim(graph, mixed)
    mid = _residual(graph, input_id, mixed, context="matmul_conv1d.mixer_resid")
    norm2 = _add(graph, "rmsnorm", [mid], context="matmul_conv1d.norm2")
    proj_a = _add(
        graph, "linear_proj", [norm2], {"out_dim": D}, context="matmul_conv1d.proj_a"
    )
    proj_b = _add(
        graph, "linear_proj", [norm2], {"out_dim": D}, context="matmul_conv1d.proj_b"
    )
    refined = _add(graph, "matmul", [proj_a, proj_b], context="matmul_conv1d.matmul")
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="matmul_conv1d.refine_proj",
    )
    refined = _fix_dim(graph, refined)
    return _residual(graph, mid, refined, context="matmul_conv1d.exotic_resid")
