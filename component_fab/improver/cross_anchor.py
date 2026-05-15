"""Cross-anchor variant generator — combine math axes from different anchors.

The single-axis improver (``axis_variants.py``) mutates one axis at a
time relative to one anchor. Cross-anchor variants are bolder: take
anchor A's algebra and combine with anchor B's state/sparsity profile.

E.g. anchor_a=``tropical_attention`` and anchor_b=``clifford_attention``
produces a proposal with ``tropical`` algebra but ``clifford``-shaped
state and sparsity axes — a hybrid that single-axis mutation can't
reach.

Cheap combinatoric explosion guard: only emit pairs where the *host*
algebra maps to a primitive that mixes across positions (tropical,
clifford). Per-position-only algebras (padic, spiking) are excluded as
hosts because their dispatched module can't satisfy a
``op_geometric_receptive_field=global`` declaration — ERF would
collapse to the 1/seq_len structural floor. They can still appear as
donors, contributing state/sparsity/receptive axes onto a mixing host.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from ..proposer.property_miner import AxisLift, CandidateTuple
from ..proposer.spec_generator import (
    ProposalSpec,
    make_proposal_id,
    spec_from_candidate,
)
from .axis_variants import (
    DEFAULT_META_DB,
    AnchorAxes,
    anchor_axes_for_op,
)

_HOSTING_ALGEBRAS = frozenset({"tropical", "clifford", "fab_promoted"})
_INHERITED_AXES: tuple[str, ...] = (
    "op_dynamical_has_state",
    "op_dynamical_memory_length_class",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
)


def is_hosting_anchor(anchor: AnchorAxes) -> bool:
    """Whether ``anchor`` can be the *host* of a cross-anchor hybrid.

    Hosts dispatch to a primitive that mixes across positions; donors
    only contribute inherited state/sparsity/receptive axes. Per-position
    algebras (padic, spiking) would collapse ERF to the 1/seq_len floor
    if they hosted, so they're donor-only. ``fab_promoted`` is the
    sentinel for already-validated fab components (they passed every gate,
    so by construction they mix).
    """
    algebra = str(anchor.axes.get("op_algebraic_space") or "")
    return algebra in _HOSTING_ALGEBRAS


def _synthetic_lift(axis: str, value: Any) -> AxisLift:
    return AxisLift(
        axis=axis,
        value=value,
        n_ops=1,
        total_evals=1,
        total_s1_pass=0,
        pass_rate=0.5,
        representative_ops=(),
    )


def _hybrid_spec(host: AnchorAxes, donor: AnchorAxes) -> ProposalSpec:
    merged: dict[str, Any] = dict(host.axes)
    for axis in _INHERITED_AXES:
        if axis in donor.axes:
            merged[axis] = donor.axes[axis]
    tuple_values = tuple(merged.items())
    lifts = tuple(_synthetic_lift(a, v) for a, v in tuple_values)
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=(host.op_name, donor.op_name),
        anchor_axes=tuple(host.axes.items()),
    )
    base_spec = spec_from_candidate(candidate)
    name = f"cross_{host.op_name}_x_{donor.op_name}"
    notes = (
        f"host={host.op_name} (algebra={host.axes.get('op_algebraic_space')})",
        f"donor={donor.op_name} (state/sparsity/receptive inherited)",
    )
    return ProposalSpec(
        proposal_id=make_proposal_id(name, merged),
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=host.op_name,
        anchor_witnesses_all=(host.op_name, donor.op_name),
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=base_spec.rationale,
        notes=notes,
    )


def enumerate_cross_anchor_variants(
    anchor_op_names: Sequence[str],
    *,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[ProposalSpec]:
    """Return cross-anchor specs for every (host, donor) pair where the
    host's algebra mixes across positions. Donors can be any algebra."""
    anchors: list[AnchorAxes] = []
    for name in anchor_op_names:
        anchor = anchor_axes_for_op(name, db_path=db_path)
        if anchor is None:
            continue
        anchors.append(anchor)

    out: list[ProposalSpec] = []
    for a, b in combinations(anchors, 2):
        if is_hosting_anchor(a):
            out.append(_hybrid_spec(a, b))
        if is_hosting_anchor(b):
            out.append(_hybrid_spec(b, a))
    return out
