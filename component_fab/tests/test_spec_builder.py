"""Spec-assembly consolidation guards.

``spec_generator.build_spec_from_axes`` replaced four near-copy
field-by-field ``ProposalSpec`` constructions (cross_anchor, axis_variants,
math_knob_catalog, dynamic). Each test below rebuilds the OLD-style spec
inline (byte-for-byte replica of the pre-consolidation code) and asserts
the consolidated path emits an identical spec — names, ids, axes, notes,
predicted_lift, rationale.
"""

from __future__ import annotations

from typing import Any

from component_fab.improver.axis_variants import (
    AnchorAxes,
    AxisVariant,
    spec_for_variant,
)
from component_fab.improver.cross_anchor import _INHERITED_AXES, _hybrid_spec
from component_fab.improver.math_knob_catalog import DEFAULT_MATH_KNOBS, _spec_for_knobs
from component_fab.proposer.dynamic import (
    DynamicEvidenceCase,
    _Repair,
    _spec_from_case_and_repair,
)
from component_fab.proposer.property_miner import AxisLift, CandidateTuple
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    make_proposal_id,
    spec_from_candidate,
)

_HOST = AnchorAxes(
    op_name="tropical_attention",
    axes={
        "op_algebraic_space": "tropical",
        "op_spectral_preferred_basis": "content",
        "op_dynamical_memory_length_class": "O(L^2)",
        "op_dynamical_has_state": 0,
        "op_activation_sparsity_pattern": "dense",
        "op_geometric_receptive_field": "global",
    },
    eval_count=300,
    pass_rate=0.18,
)

_DONOR = AnchorAxes(
    op_name="padic_gate",
    axes={
        "op_algebraic_space": "padic",
        "op_spectral_preferred_basis": "identity",
        "op_dynamical_memory_length_class": "O(L)",
        "op_dynamical_has_state": 1,
        "op_activation_sparsity_pattern": "learned_structured",
        "op_geometric_receptive_field": "local",
    },
    eval_count=120,
    pass_rate=0.05,
)


def _old_synthetic_lift(axis: str, value: Any, pass_rate: float = 0.5) -> AxisLift:
    return AxisLift(
        axis=axis,
        value=value,
        n_ops=1,
        total_evals=1,
        total_s1_pass=1 if pass_rate >= 0.5 else 0,
        pass_rate=pass_rate,
        representative_ops=(),
    )


def test_hybrid_spec_matches_old_cross_anchor_construction() -> None:
    merged: dict[str, Any] = dict(_HOST.axes)
    for axis in _INHERITED_AXES:
        if axis in _DONOR.axes:
            merged[axis] = _DONOR.axes[axis]
    name = f"hybrid_{_HOST.op_name}_plus_{_DONOR.op_name}"
    witness_ops = (_HOST.op_name, _DONOR.op_name)
    notes = (
        f"frontier_host={_HOST.op_name} (proven binder, global mixing kept)",
        f"donor={_DONOR.op_name} (gave state/sparsity axes)",
    )
    # Old cross_anchor._build_spec, replicated.
    tuple_values = tuple(merged.items())
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=tuple(_old_synthetic_lift(a, v) for a, v in tuple_values),
        witness_ops=witness_ops,
        anchor_axes=tuple(_HOST.axes.items()),
    )
    base_spec = spec_from_candidate(candidate)
    expected = ProposalSpec(
        proposal_id=make_proposal_id(name, merged),
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=witness_ops[0],
        anchor_witnesses_all=witness_ops,
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=base_spec.rationale,
        notes=notes,
    )

    assert _hybrid_spec(_HOST, _DONOR) == expected


def test_spec_for_knobs_matches_old_math_knob_construction() -> None:
    knobs = DEFAULT_MATH_KNOBS[:2]
    knob_ids = tuple(knob.knob_id for knob in knobs)
    axes = dict(_HOST.axes)
    axes["op_math_knobs"] = "+".join(knob_ids)
    axes["op_math_family"] = "composite"
    for knob in knobs:
        axes.update(knob.axes)
    # Old math_knob_catalog._spec_for_knobs, replicated.
    tuple_values = tuple(axes.items())
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=tuple(_old_synthetic_lift(a, v) for a, v in tuple_values),
        witness_ops=(_HOST.op_name,),
        anchor_axes=tuple(_HOST.axes.items()),
    )
    base = spec_from_candidate(candidate)
    name = f"compose_{_HOST.op_name}_{'__'.join(knob_ids)}"
    expected = ProposalSpec(
        proposal_id=make_proposal_id(name, axes),
        name=name,
        category=base.category,
        synthesis_kind=base.synthesis_kind,
        math_axes=axes,
        anchor_witness_op=_HOST.op_name,
        anchor_witnesses_all=(_HOST.op_name,),
        declared_property_row=base.declared_property_row,
        predicted_lift=base.predicted_lift,
        rationale=base.rationale,
        notes=tuple(
            [f"anchor={_HOST.op_name}", f"math_knobs={'+'.join(knob_ids)}"]
            + [knob.rationale for knob in knobs]
        ),
    )

    actual = _spec_for_knobs(_HOST, knobs)
    assert actual == expected
    # The math-knob path historically emits math_axes WITHOUT the mirrored
    # synthesis_kind — a drift build_spec_from_axes must preserve.
    assert "synthesis_kind" not in actual.math_axes


def test_spec_for_variant_matches_old_axis_variant_construction() -> None:
    variant = AxisVariant(
        delta_name="add_state_OL",
        delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
        },
        rationale="add SSM-style running state on the sequence dim",
    )
    # Old axis_variants spec_for_variant (+_candidate_for_variant), replicated.
    merged: dict[str, Any] = {**_HOST.axes, **variant.delta}
    tuple_values = tuple(merged.items())
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=tuple(_old_synthetic_lift(a, v) for a, v in tuple_values),
        witness_ops=(_HOST.op_name,),
        anchor_axes=tuple(_HOST.axes.items()),
    )
    base_spec = spec_from_candidate(candidate)
    name = f"improve_{_HOST.op_name}_{variant.delta_name}"
    expected = ProposalSpec(
        proposal_id=make_proposal_id(name, base_spec.math_axes),
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=_HOST.op_name,
        anchor_witnesses_all=(_HOST.op_name,),
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=base_spec.rationale,
        notes=(
            f"anchor={_HOST.op_name} "
            f"(pass_rate={_HOST.pass_rate:.2f} on {_HOST.eval_count} evals)",
            variant.rationale,
        ),
    )

    assert spec_for_variant(_HOST, variant) == expected


def test_spec_from_case_and_repair_matches_old_dynamic_construction() -> None:
    case = DynamicEvidenceCase(
        source_id="p1",
        name="tropical attention v2",
        base_axes=dict(_HOST.axes),
        anchor_axes=dict(_HOST.axes),
        score=0.31,
        weaknesses=("range_blind", "weak_nano_bind"),
        notes=("ledger_status=rejected",),
    )
    repair = _Repair(
        name="extend_receptive_state",
        delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        rationale="repair measured range/ERF weakness",
    )
    # Old dynamic._spec_from_case_and_repair, replicated.
    axes = {**case.base_axes, **repair.delta}
    axes.pop("synthesis_kind", None)
    tuple_values = tuple(axes.items())
    pass_rate = min(1.0, max(0.05, case.score))
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=max(0.1, min(1.0, case.score + 0.08)),
        per_axis_lift=tuple(
            _old_synthetic_lift(a, v, pass_rate) for a, v in tuple_values
        ),
        witness_ops=(case.name,),
        anchor_axes=tuple(case.anchor_axes.items()),
    )
    base_spec = spec_from_candidate(candidate)
    name = "dynamic_tropical_attention_v2_extend_receptive_state_range_blind_weak_nano_bind"
    expected = ProposalSpec(
        proposal_id=make_proposal_id(name, base_spec.math_axes),
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=case.name,
        anchor_witnesses_all=(case.name,),
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=(
            f"Dynamic proposal derived from ledger evidence for {case.source_id}. "
            f"Weaknesses={', '.join(case.weaknesses)}. {repair.rationale}."
        ),
        notes=(
            f"source_id={case.source_id}",
            f"source_score={case.score:.4f}",
            f"repair={repair.name}",
            *case.notes,
        ),
    )

    assert _spec_from_case_and_repair(case, repair) == expected
