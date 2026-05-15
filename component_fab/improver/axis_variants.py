"""Axis-variant generator for goal-(b) anchor ops.

For each anchor op (a "novel-looking but underperforming" op surfaced by
the scoper — tropical_attention, clifford_attention, padic_gate, etc.)
this module enumerates a small set of axis-delta variants and emits
``ProposalSpec`` objects ready for ``code_generator`` + ``solo`` validator.

A variant is an axis-tuple change relative to the anchor — e.g. "+state"
flips ``op_dynamical_has_state=1`` and ``op_dynamical_memory_length_class=O(L)``.
The idea: the anchor's math is novel but underperforms; one axis change
may unlock it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..proposer.property_miner import AxisLift, CandidateTuple
from ..proposer.spec_generator import (
    ProposalSpec,
    make_proposal_id,
    spec_from_candidate,
)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_META_DB = _REPO / "research" / "meta_analysis.db"

_AXES_OF_INTEREST: tuple[str, ...] = (
    "op_algebraic_space",
    "op_spectral_preferred_basis",
    "op_dynamical_memory_length_class",
    "op_dynamical_has_state",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
)


@dataclass(frozen=True, slots=True)
class AxisVariant:
    delta_name: str
    delta: dict[str, Any]
    rationale: str


@dataclass(frozen=True, slots=True)
class AnchorAxes:
    op_name: str
    axes: dict[str, Any]
    eval_count: int
    pass_rate: float


DEFAULT_AXIS_VARIANT_TEMPLATES: tuple[AxisVariant, ...] = (
    AxisVariant(
        delta_name="add_state_OL",
        delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
        },
        rationale="add SSM-style running state on the sequence dim",
    ),
    AxisVariant(
        delta_name="top_k_sparsity",
        delta={"op_activation_sparsity_pattern": "top_k"},
        rationale="replace dense activation with top-k sparsity",
    ),
    AxisVariant(
        delta_name="fourier_basis",
        delta={"op_spectral_preferred_basis": "frequency"},
        rationale="apply the op in the frequency basis along sequence",
    ),
    AxisVariant(
        delta_name="global_receptive",
        delta={"op_geometric_receptive_field": "global"},
        rationale="widen to global receptive field",
    ),
    AxisVariant(
        delta_name="calculus_finite_difference",
        delta={
            "op_math_family": "calculus",
            "op_calculus_operator": "causal_finite_difference_integral",
        },
        rationale="add causal finite-difference and running-integral features",
    ),
    AxisVariant(
        delta_name="linear_algebra_low_rank",
        delta={
            "op_math_family": "linear_algebra",
            "op_linear_algebra_structure": "low_rank_factorized",
        },
        rationale="replace dense feature mixing with low-rank factorized mixing",
    ),
    AxisVariant(
        delta_name="sparse_matrix_banded",
        delta={
            "op_math_family": "sparse_matrix",
            "op_sparse_matrix_pattern": "causal_banded",
        },
        rationale="apply a causal banded sparse sequence matrix",
    ),
)


def anchor_axes_for_op(
    op_name: str, db_path: Path | str = DEFAULT_META_DB
) -> AnchorAxes | None:
    """Look up the declared math axes for ``op_name`` from op_property_catalog."""
    path = Path(db_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM op_property_catalog WHERE op_name = ?", (op_name,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    axes = {key: row[key] for key in _AXES_OF_INTEREST if key in row.keys()}
    evals = int(row["eval_count"] or 0)
    s1 = int(row["s1_pass_count"] or 0)
    return AnchorAxes(
        op_name=op_name,
        axes=axes,
        eval_count=evals,
        pass_rate=(s1 / evals) if evals else 0.0,
    )


def _synthetic_axis_lift(axis: str, value: Any) -> AxisLift:
    return AxisLift(
        axis=axis,
        value=value,
        n_ops=1,
        total_evals=1,
        total_s1_pass=0,
        pass_rate=0.5,
        representative_ops=(),
    )


def _candidate_for_variant(anchor: AnchorAxes, variant: AxisVariant) -> CandidateTuple:
    merged: dict[str, Any] = {**anchor.axes, **variant.delta}
    tuple_values = tuple(merged.items())
    lifts = tuple(_synthetic_axis_lift(a, v) for a, v in tuple_values)
    return CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=(anchor.op_name,),
        anchor_axes=tuple(anchor.axes.items()),
    )


def spec_for_variant(anchor: AnchorAxes, variant: AxisVariant) -> ProposalSpec:
    base_spec = spec_from_candidate(_candidate_for_variant(anchor, variant))
    name = f"improve_{anchor.op_name}_{variant.delta_name}"
    proposal_id = make_proposal_id(name, base_spec.math_axes)
    notes = (
        f"anchor={anchor.op_name} "
        f"(pass_rate={anchor.pass_rate:.2f} on {anchor.eval_count} evals)",
        variant.rationale,
    )
    # Replace name + id; everything else inherits from the dispatched spec.
    return ProposalSpec(
        proposal_id=proposal_id,
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=anchor.op_name,
        anchor_witnesses_all=(anchor.op_name,),
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=base_spec.rationale,
        notes=notes,
    )


def enumerate_axis_variants(
    anchor_op_names: Sequence[str],
    *,
    variants: Sequence[AxisVariant] = DEFAULT_AXIS_VARIANT_TEMPLATES,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[ProposalSpec]:
    out: list[ProposalSpec] = []
    for name in anchor_op_names:
        anchor = anchor_axes_for_op(name, db_path=db_path)
        if anchor is None:
            continue
        for variant in variants:
            out.append(spec_for_variant(anchor, variant))
    return out
