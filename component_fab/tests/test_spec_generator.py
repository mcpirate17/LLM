"""Smoke + behavior tests for component_fab.proposer.spec_generator."""

from __future__ import annotations

from component_fab.proposer.property_miner import AxisLift, CandidateTuple
from component_fab.proposer.spec_generator import (
    CATEGORY_COMPRESSION,
    CATEGORY_LANE,
    CATEGORY_ROUTING,
    SYNTHESIS_KIND_SEMIRING_SWAP,
    SYNTHESIS_KIND_STATE_KERNEL_SWAP,
    axes_fingerprint,
    category_from_axes,
    dedupe_specs_by_axes,
    make_proposal_id,
    spec_from_candidate,
    spec_to_json,
    synthesis_kind_for_axes,
)


def _candidate(
    values: dict, witnesses: tuple[str, ...] = ("softmax_attention",)
) -> CandidateTuple:
    lifts = tuple(
        AxisLift(
            axis=k,
            value=v,
            n_ops=3,
            total_evals=100,
            total_s1_pass=40,
            pass_rate=0.4,
            representative_ops=(witnesses[0],),
        )
        for k, v in values.items()
    )
    return CandidateTuple(
        tuple_values=tuple(values.items()),
        predicted_lift=0.4,
        per_axis_lift=lifts,
        witness_ops=witnesses,
    )


def test_category_from_axes_routing_for_topk_global() -> None:
    axes = {
        "op_activation_sparsity_pattern": "top_k",
        "op_geometric_receptive_field": "global",
        "op_dynamical_has_state": 0,
    }
    assert category_from_axes(axes) == CATEGORY_ROUTING


def test_category_from_axes_compression_for_stateful_sparse() -> None:
    axes = {
        "op_dynamical_has_state": 1,
        "op_activation_sparsity_pattern": "learned_structured",
    }
    assert category_from_axes(axes) == CATEGORY_COMPRESSION


def test_category_from_axes_lane_default() -> None:
    axes = {
        "op_dynamical_has_state": 0,
        "op_activation_sparsity_pattern": "dense",
        "op_dynamical_memory_length_class": "O(L^2)",
        "op_geometric_receptive_field": "global",
    }
    assert category_from_axes(axes) == CATEGORY_LANE


def test_synthesis_kind_semiring_swap_on_novel_algebra() -> None:
    axes = {"op_algebraic_space": "tropical"}
    anchor = {"op_algebraic_space": "euclidean"}
    assert synthesis_kind_for_axes(axes, anchor) == SYNTHESIS_KIND_SEMIRING_SWAP


def test_synthesis_kind_state_kernel_swap_on_state_change() -> None:
    axes = {"op_dynamical_has_state": 1}
    anchor = {"op_dynamical_has_state": 0}
    assert synthesis_kind_for_axes(axes, anchor) == SYNTHESIS_KIND_STATE_KERNEL_SWAP


def test_spec_from_candidate_with_anchor_axes_labels_state_kernel_swap() -> None:
    """Adding state on top of a tropical anchor should label state_kernel_swap,
    not semiring_swap. Without anchor_axes on CandidateTuple, the algebra rule
    short-circuits and every novel-algebra spec collapses to semiring_swap."""
    axes_with_state = {
        "op_algebraic_space": "tropical",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_activation_sparsity_pattern": "dense",
    }
    cand = _candidate(axes_with_state, witnesses=("tropical_attention",))
    # Construct an anchor that matches host axes (tropical, stateless, dense)
    cand_with_host = type(cand)(
        tuple_values=cand.tuple_values,
        predicted_lift=cand.predicted_lift,
        per_axis_lift=cand.per_axis_lift,
        witness_ops=cand.witness_ops,
        anchor_axes=(
            ("op_algebraic_space", "tropical"),
            ("op_dynamical_has_state", 0),
            ("op_activation_sparsity_pattern", "dense"),
        ),
    )
    spec = spec_from_candidate(cand_with_host)
    assert spec.synthesis_kind == SYNTHESIS_KIND_STATE_KERNEL_SWAP


def test_spec_from_candidate_without_anchor_axes_falls_back_to_semiring() -> None:
    """Legacy behavior preserved when anchor_axes is empty (back-compat)."""
    axes = {"op_algebraic_space": "tropical", "op_dynamical_has_state": 1}
    cand = _candidate(axes, witnesses=("tropical_attention",))
    spec = spec_from_candidate(cand)
    assert spec.synthesis_kind == SYNTHESIS_KIND_SEMIRING_SWAP


def test_spec_from_candidate_well_formed() -> None:
    candidate = _candidate(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "top_k",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        witnesses=("tropical_attention", "selective_scan"),
    )
    spec = spec_from_candidate(candidate)
    assert spec.category in (CATEGORY_LANE, CATEGORY_ROUTING, CATEGORY_COMPRESSION)
    assert spec.anchor_witness_op == "tropical_attention"
    assert spec.synthesis_kind in (
        SYNTHESIS_KIND_SEMIRING_SWAP,
        SYNTHESIS_KIND_STATE_KERNEL_SWAP,
    )
    assert spec.math_axes["op_algebraic_space"] == "tropical"
    assert spec.declared_property_row["op_n_inputs"] == 1
    assert spec.declared_property_row["op_is_stateless"] == 0
    assert "tropical" in spec.name
    assert spec.proposal_id.startswith(spec.name + "_")


def test_spec_to_json_roundtrips_axis_keys() -> None:
    candidate = _candidate(
        {
            "op_algebraic_space": "clifford",
            "op_dynamical_has_state": 0,
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "global",
        }
    )
    spec = spec_from_candidate(candidate)
    blob = spec_to_json(spec)
    assert blob["math_axes"]["op_algebraic_space"] == "clifford"
    assert blob["category"] in (CATEGORY_LANE, CATEGORY_ROUTING, CATEGORY_COMPRESSION)
    assert blob["proposal_id"] == spec.proposal_id


def test_axes_fingerprint_invariant_to_key_order() -> None:
    a = {"op_algebraic_space": "tropical", "op_dynamical_has_state": 1}
    b = {"op_dynamical_has_state": 1, "op_algebraic_space": "tropical"}
    assert axes_fingerprint(a) == axes_fingerprint(b)


def test_axes_fingerprint_differs_when_a_value_differs() -> None:
    a = {"op_algebraic_space": "tropical", "op_dynamical_has_state": 1}
    b = {"op_algebraic_space": "tropical", "op_dynamical_has_state": 0}
    assert axes_fingerprint(a) != axes_fingerprint(b)


def test_make_proposal_id_matches_inline_digest_format() -> None:
    axes = {"op_algebraic_space": "clifford", "op_activation_sparsity_pattern": "dense"}
    pid = make_proposal_id("compose_test", axes)
    assert pid.startswith("compose_test_")
    assert pid.split("_")[-1] == axes_fingerprint(axes)


def test_dedupe_specs_by_axes_keeps_shorter_name_on_collision() -> None:
    """Two specs with identical math_axes -> dedupe keeps shorter-named one."""
    base_axes = {
        "op_algebraic_space": "tropical",
        "op_dynamical_has_state": 0,
        "op_activation_sparsity_pattern": "dense",
        "op_geometric_receptive_field": "global",
    }
    cand = _candidate(base_axes)
    short = spec_from_candidate(cand)
    # Build a longer-named spec with the same math_axes by replacing the name
    long_name = "cross_a_x_cross_b_x_cross_c_x_long_donor"
    long_spec = type(short)(
        proposal_id=make_proposal_id(long_name, short.math_axes),
        name=long_name,
        category=short.category,
        synthesis_kind=short.synthesis_kind,
        math_axes=short.math_axes,
        anchor_witness_op=short.anchor_witness_op,
        anchor_witnesses_all=short.anchor_witnesses_all,
        declared_property_row=short.declared_property_row,
        predicted_lift=short.predicted_lift,
        rationale=short.rationale,
        notes=short.notes,
    )
    # Long arrives first; short should still win on shorter-name tiebreak
    deduped = dedupe_specs_by_axes([long_spec, short])
    assert len(deduped) == 1
    assert deduped[0].name == short.name


def test_dedupe_specs_by_axes_preserves_order_of_distinct_fingerprints() -> None:
    """Distinct axes -> all kept, order preserved (first-seen)."""
    spec_a = spec_from_candidate(
        _candidate({"op_algebraic_space": "tropical", "op_dynamical_has_state": 0})
    )
    spec_b = spec_from_candidate(
        _candidate({"op_algebraic_space": "clifford", "op_dynamical_has_state": 0})
    )
    spec_c = spec_from_candidate(
        _candidate({"op_algebraic_space": "padic", "op_dynamical_has_state": 1})
    )
    deduped = dedupe_specs_by_axes([spec_a, spec_b, spec_c])
    assert [s.proposal_id for s in deduped] == [
        spec_a.proposal_id,
        spec_b.proposal_id,
        spec_c.proposal_id,
    ]
