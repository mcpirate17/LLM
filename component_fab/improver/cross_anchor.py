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

import logging
from collections.abc import Mapping
from typing import Any, Sequence

from ..proposer.spec_generator import ProposalSpec, build_spec_from_axes
from .axis_variants import AnchorAxes, anchor_axes_for_op
from ..inventor.mechanism_catalog import enumerate_invention_specs

logger = logging.getLogger(__name__)

_HOSTING_ALGEBRAS = frozenset({"tropical", "clifford", "fab_promoted"})
_INHERITED_AXES: tuple[str, ...] = (
    "op_dynamical_has_state",
    "op_dynamical_memory_length_class",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
)
_VALIDATED_LOSS_PARTNER_OPS = frozenset(
    {
        "hyper_mor_b_145m",
        "hyper_mor_bilane",
        "slot_dplr",
        "native_semiring",
        "native_semiring_surprise_memory",
    }
)
_LOSS_PARTNER_METRICS: tuple[str, ...] = (
    "ar_held_pair",
    "ar_held",
    "induction_screening_auc",
    "induction_auc",
    "induction",
    "binding_interm",
    "binding_v2",
    "binding",
    "mqar",
    "recall",
    "blimp",
)
_LOSS_SLOT_BY_TEMPLATE = {
    "routed_bottleneck": "routed_bottleneck",
    "bottleneck": "low_rank",
    "moe": "routed_bottleneck",
    "parallel_split": "routed_bottleneck",
    "recursive_depth_router": "routed_bottleneck",
    "residual_block": "low_rank",
}


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


def _hybrid_spec(host: AnchorAxes, donor: AnchorAxes) -> ProposalSpec:
    merged: dict[str, Any] = dict(host.axes)
    for axis in _INHERITED_AXES:
        if axis in donor.axes:
            merged[axis] = donor.axes[axis]
    return build_spec_from_axes(
        f"hybrid_{host.op_name}_plus_{donor.op_name}",
        merged,
        witness_ops=(host.op_name, donor.op_name),
        anchor_axes=host.axes,
        notes=(
            f"frontier_host={host.op_name} (proven binder, global mixing kept)",
            f"donor={donor.op_name} (gave state/sparsity axes)",
        ),
    )


def _evidence_score(row: Mapping[str, Any]) -> float:
    values: list[float] = []
    for key in _LOSS_PARTNER_METRICS:
        raw = row.get(key)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _has_loss_partner_evidence(
    partner: AnchorAxes,
    evidence_by_op: Mapping[str, Mapping[str, Any]] | None,
    *,
    min_score: float = 0.55,
) -> bool:
    op = partner.op_name
    if evidence_by_op is not None:
        evidence = evidence_by_op.get(op)
        if evidence is None:
            return False
        if bool(evidence.get("long_range_evidence") or evidence.get("induction_evidence")):
            return True
        return _evidence_score(evidence) >= min_score
    return op in _VALIDATED_LOSS_PARTNER_OPS


def _partner_kind_for_anchor(partner: AnchorAxes) -> str:
    name = partner.op_name.lower()
    if "hyper_mor" in name:
        return "hyper_mor"
    if "slot" in name or "dplr" in name:
        return "slot_dplr"
    if "semiring" in name:
        return "native_semiring"
    return "anchor"


def _loss_slot_for_donor(donor: ProposalSpec) -> str:
    axes = donor.math_axes
    explicit = axes.get("op_block_slot_loss") or axes.get("op_loss_monster_slot")
    if explicit:
        return str(explicit)
    template = str(axes.get("op_block_template") or donor.name)
    return _LOSS_SLOT_BY_TEMPLATE.get(template, "routed_bottleneck")


def _loss_pair_spec(
    partner: AnchorAxes,
    donor: ProposalSpec,
    evidence: Mapping[str, Any] | None,
) -> ProposalSpec:
    merged: dict[str, Any] = dict(partner.axes)
    merged.update(
        {
            "op_block_template": "loss_monster_paired",
            "op_partner_kind": _partner_kind_for_anchor(partner),
            "op_block_slot_loss": _loss_slot_for_donor(donor),
            "op_partner_floor": 0.5,
            "op_candidate_role": "loss_specialist_pair",
            "op_loss_specialist_paired": 1,
            "op_loss_specialist_partner_op": partner.op_name,
            "op_loss_specialist_donor_id": donor.proposal_id,
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        }
    )
    score = _evidence_score(evidence or {})
    notes = (
        f"loss_partner={partner.op_name} (long-range evidence required)",
        f"loss_donor={donor.proposal_id}",
        f"loss_donor_slot={merged['op_block_slot_loss']}",
        f"partner_evidence_score={score:.3f}",
        "promotion requires paired capability delta over partner-alone baseline",
    )
    return build_spec_from_axes(
        f"loss_pair_{partner.op_name}_plus_{donor.name}",
        merged,
        witness_ops=(partner.op_name, donor.proposal_id),
        anchor_axes=partner.axes,
        notes=notes,
    )


def enumerate_loss_monster_pairs(
    loss_donors: Sequence[ProposalSpec],
    *,
    partners: Sequence[AnchorAxes] | None = None,
    evidence_by_op: Mapping[str, Mapping[str, Any]] | None = None,
    max_pairs: int | None = None,
) -> list[ProposalSpec]:
    """Pair local loss-specialist donors with proven long-range carriers.

    No partner evidence means no emission. Loss-only donors are never promoted
    through this path; every emitted spec carries role metadata and is intended
    to be graded against the partner-alone carrier baseline.
    """

    if not loss_donors:
        return []
    partner_anchors = _resolve_loss_partner_hosts(partners)
    out: list[ProposalSpec] = []
    for partner in partner_anchors:
        if not _has_loss_partner_evidence(partner, evidence_by_op):
            continue
        evidence = evidence_by_op.get(partner.op_name) if evidence_by_op else None
        for donor in loss_donors:
            out.append(_loss_pair_spec(partner, donor, evidence))
            if max_pairs is not None and len(out) >= max_pairs:
                return out
    return out


def enumerate_cross_anchor_variants(op_names: Sequence[str]) -> list[ProposalSpec]:
    """All pairwise hybrids between the provided anchors."""
    anchors = [a for n in op_names if (a := anchor_axes_for_op(n)) is not None]
    hosts = [a for a in anchors if is_hosting_anchor(a)]
    donors = anchors  # any anchor can be a donor
    if not hosts:
        return []
    out: list[ProposalSpec] = []
    for host in hosts:
        for donor in donors:
            if host.op_name == donor.op_name:
                continue
            out.append(_hybrid_spec(host, donor))
    return out


def _resolve_frontier_hosts(
    hosts: Sequence[AnchorAxes] | None = None,
) -> list[AnchorAxes]:
    if hosts is not None:
        return [h for h in hosts if h is not None]
    # Default: the three "canonical" frontier cores from the meta-DB
    # (these must exist in meta_db.db).
    out: list[AnchorAxes] = []
    for name in ("tropical_attention", "clifford_attention", "poincare_attention"):
        try:
            if (a := anchor_axes_for_op(name)) is not None:
                out.append(a)
        except Exception as exc:  # noqa: BLE001 - a broken meta-DB must not kill the run
            logger.warning(
                "frontier host lookup failed for %s (meta-DB unreadable?): %s",
                name,
                exc,
            )
    return out


def _resolve_loss_partner_hosts(
    partners: Sequence[AnchorAxes] | None = None,
) -> list[AnchorAxes]:
    if partners is not None:
        return [p for p in partners if p is not None]
    out: list[AnchorAxes] = []
    for name in _VALIDATED_LOSS_PARTNER_OPS:
        try:
            if (anchor := anchor_axes_for_op(name)) is not None:
                out.append(anchor)
        except Exception as exc:  # noqa: BLE001 - meta-DB lookup must not kill proposal loop
            logger.warning(
                "loss-partner lookup failed for %s (meta-DB unreadable?): %s",
                name,
                exc,
            )
    return out


def enumerate_frontier_core_specs(
    hosts: Sequence[AnchorAxes] | None = None,
) -> list[ProposalSpec]:
    """The bare proven-binder cores as gradeable specs.

    Grades the frontier cores directly so they enter the ledger as high-quality
    reference points, not only as hybrid hosts.
    """
    cores = [
        build_spec_from_axes(
            host.op_name,
            dict(host.axes),
            witness_ops=(host.op_name,),
            anchor_axes=host.axes,
            notes=("frontier core (proven binder, graded standalone)",),
        )
        for host in _resolve_frontier_hosts(hosts)
    ]
    # Inject invention-track specs (like data_dependent_decay) into the
    # autonomous loop via the 'frontier core' path to ensure they are
    # considered even when not anchored to a known failure.
    invention_specs = enumerate_invention_specs()
    return cores + invention_specs


def enumerate_frontier_hybrids(
    donor_op_names: Sequence[str],
    *,
    hosts: Sequence[AnchorAxes] | None = None,
) -> list[ProposalSpec]:
    """Combine proven-binder cores with novel donor axis-profiles."""
    host_anchors = _resolve_frontier_hosts(hosts)
    if not host_anchors:
        return []
    donor_anchors = [
        a for n in donor_op_names if (a := anchor_axes_for_op(n)) is not None
    ]
    out: list[ProposalSpec] = []
    for host in host_anchors:
        for donor in donor_anchors:
            out.append(_hybrid_spec(host, donor))
    return out
