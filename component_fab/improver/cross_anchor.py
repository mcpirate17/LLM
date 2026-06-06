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


def _build_spec(
    name: str,
    merged: dict[str, Any],
    *,
    witness_ops: tuple[str, ...],
    anchor_axes: dict[str, Any],
    notes: tuple[str, ...],
) -> ProposalSpec:
    """Assemble a ProposalSpec from already-merged axes.

    Shared by the generic cross-anchor and the frontier-hybrid generators so
    spec assembly (synthetic lifts, candidate, id, witnesses) lives in one place.
    """
    tuple_values = tuple(merged.items())
    lifts = tuple(_synthetic_lift(a, v) for a, v in tuple_values)
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=witness_ops,
        anchor_axes=tuple(anchor_axes.items()),
    )
    base_spec = spec_from_candidate(candidate)
    return ProposalSpec(
        proposal_id=make_proposal_id(name, merged),
        name=name,
        category=base_spec.category,
        synthesis_kind=base_spec.synthesis_kind,
        math_axes=base_spec.math_axes,
        anchor_witness_op=witness_ops[0] if witness_ops else name,
        anchor_witnesses_all=witness_ops,
        declared_property_row=base_spec.declared_property_row,
        predicted_lift=base_spec.predicted_lift,
        rationale=base_spec.rationale,
        notes=notes,
    )


def _hybrid_spec(host: AnchorAxes, donor: AnchorAxes) -> ProposalSpec:
    merged: dict[str, Any] = dict(host.axes)
    for axis in _INHERITED_AXES:
        if axis in donor.axes:
            merged[axis] = donor.axes[axis]
    return _build_spec(
        f"cross_{host.op_name}_x_{donor.op_name}",
        merged,
        witness_ops=(host.op_name, donor.op_name),
        anchor_axes=host.axes,
        notes=(
            f"host={host.op_name} (algebra={host.axes.get('op_algebraic_space')})",
            f"donor={donor.op_name} (state/sparsity/receptive inherited)",
        ),
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


# Frontier hybrids inherit ONLY the donor's novel mechanism (state + memory +
# sparsity). The host's op_geometric_receptive_field=global is preserved on
# purpose — that global mixing is the source of the core's binding strength, so
# unlike the generic cross-anchor path a donor is NOT allowed to downgrade it.
_FRONTIER_INHERITED_AXES: tuple[str, ...] = (
    "op_dynamical_has_state",
    "op_dynamical_memory_length_class",
    "op_activation_sparsity_pattern",
)


def _resolve_frontier_hosts(
    hosts: Sequence[AnchorAxes] | None,
) -> list[AnchorAxes]:
    if hosts is not None:
        return list(hosts)
    from ..proposer.frontier_cores import frontier_core_anchors

    return frontier_core_anchors()


def _frontier_hybrid_spec(host: AnchorAxes, donor: AnchorAxes) -> ProposalSpec:
    merged: dict[str, Any] = dict(host.axes)
    for axis in _FRONTIER_INHERITED_AXES:
        if axis in donor.axes:
            merged[axis] = donor.axes[axis]
    return _build_spec(
        f"frontier_{host.op_name}_plus_{donor.op_name}",
        merged,
        witness_ops=(host.op_name, donor.op_name),
        anchor_axes=host.axes,
        notes=(
            f"frontier_host={host.op_name} (proven binder, global mixing kept)",
            f"donor={donor.op_name} (state/memory/sparsity grafted on)",
        ),
    )


def enumerate_frontier_core_specs(
    hosts: Sequence[AnchorAxes] | None = None,
) -> list[ProposalSpec]:
    """The bare proven-binder cores as gradeable specs.

    Grades the frontier cores directly so they enter the ledger as high-quality
    reference points, not only as hybrid hosts.
    """
    return [
        _build_spec(
            host.op_name,
            dict(host.axes),
            witness_ops=(host.op_name,),
            anchor_axes=host.axes,
            notes=("frontier core (proven binder, graded standalone)",),
        )
        for host in _resolve_frontier_hosts(hosts)
    ]


def enumerate_frontier_hybrids(
    donor_op_names: Sequence[str],
    *,
    hosts: Sequence[AnchorAxes] | None = None,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[ProposalSpec]:
    """Graft each donor's novel mechanism onto each proven frontier core.

    This is the "frontier + delta" generator: the host supplies a binder that
    already reaches frontier-tied bAbI accuracy; the donor (typically an
    underperforming-novel op) supplies the candidate new mechanism. Donors that
    don't resolve in the meta DB are skipped.
    """
    donors: list[AnchorAxes] = []
    for name in donor_op_names:
        donor = anchor_axes_for_op(name, db_path=db_path)
        if donor is not None:
            donors.append(donor)
    out: list[ProposalSpec] = []
    for host in _resolve_frontier_hosts(hosts):
        for donor in donors:
            out.append(_frontier_hybrid_spec(host, donor))
    return out
