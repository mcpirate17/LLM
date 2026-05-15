"""Tiered capability validator — chained gates ordered cheapest-first.

Gate stack (in order; each is a hard no-go unless noted):

  1. **S0.5 stability + causality** — forward-only, ~ms. Catches non-causal
     primitives and pathological numerics.
  2. **ERF density** — single forward+backward. Catches information-bottleneck
     primitives whose last-position output ignores upstream positions.
  3. **AR Gate / NB 0.5 binding** — short K-class binding training with
     persistent-at-baseline rejection. Catches frequency-mode-collapse
     degenerates.
  4. **AR easy / medium retrieval** — sprint-7 binding probes. Soft signal
     (boosts composite when ``can_bind``); not hard-rejection.

Each gate emits an elimination flag in the scorecard so the autonomous
loop can surface "N eliminated by gate X" per cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from torch import nn

from ..harness.capability_probes import (
    DEFAULT_CAPABILITY_PROBES,
    CapabilityProbe,
    causality_stability_gate,
    train_and_score,
)
from ..harness.erf_probe import measure_erf
from ..harness.nano_bind_probe import nano_bind_gate
from ..harness.standard_block import LaneTestBlock
from ..proposer.spec_generator import ProposalSpec


@dataclass(frozen=True, slots=True)
class CapabilityScorecard:
    proposal_id: str
    name: str
    s05_passed: bool
    s05_stability_passed: bool
    s05_causality_passed: bool
    s05_max_first_half_drift: float
    erf_passed: bool
    erf_density: float
    erf_density_entropy: float
    erf_decay_slope: float
    nb_passed: bool
    nb_max_accuracy: float
    nb_rejected_persistent_zero: bool
    can_bind: bool
    binds_per_probe: dict[str, bool]
    relative_recall_per_probe: dict[str, float]
    eliminated_by: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _empty_scorecard(
    spec: ProposalSpec,
    *,
    eliminated_by: str,
    notes: tuple[str, ...] = (),
    s05_passed: bool = False,
    s05_stability_passed: bool = False,
    s05_causality_passed: bool = False,
    s05_max_first_half_drift: float = 0.0,
    erf_passed: bool = False,
    erf_density: float = 0.0,
    erf_density_entropy: float = 0.0,
    erf_decay_slope: float = 0.0,
    nb_passed: bool = False,
    nb_max_accuracy: float = 0.0,
    nb_rejected_persistent_zero: bool = False,
) -> CapabilityScorecard:
    return CapabilityScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        s05_passed=s05_passed,
        s05_stability_passed=s05_stability_passed,
        s05_causality_passed=s05_causality_passed,
        s05_max_first_half_drift=s05_max_first_half_drift,
        erf_passed=erf_passed,
        erf_density=erf_density,
        erf_density_entropy=erf_density_entropy,
        erf_decay_slope=erf_decay_slope,
        nb_passed=nb_passed,
        nb_max_accuracy=nb_max_accuracy,
        nb_rejected_persistent_zero=nb_rejected_persistent_zero,
        can_bind=False,
        binds_per_probe={},
        relative_recall_per_probe={},
        eliminated_by=eliminated_by,
        notes=notes,
    )


def _run_ar_probes(
    lane: nn.Module,
    *,
    dim: int,
    seq_len: int,
    probes: Sequence[CapabilityProbe],
    binding_threshold: float,
) -> tuple[dict[str, bool], dict[str, float]]:
    binds: dict[str, bool] = {}
    recall: dict[str, float] = {}
    for index, probe in enumerate(probes):
        fresh_block = LaneTestBlock(lane, dim).train()
        result = train_and_score(
            fresh_block,
            probe,
            seq_len=seq_len,
            dim=dim,
            seed=index,
        )
        binds[probe.name] = bool(
            result.trained_successfully and result.relative_recall >= binding_threshold
        )
        recall[probe.name] = round(result.relative_recall, 3)
    return binds, recall


def validate_capabilities(
    spec: ProposalSpec,
    lane: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 32,
    probes: Sequence[CapabilityProbe] = DEFAULT_CAPABILITY_PROBES,
    causality_threshold: float = 0.05,
    binding_threshold: float = 0.3,
    erf_density_threshold: float = 0.05,
    nb_n_classes: int = 4,
) -> CapabilityScorecard:
    """Run the tiered gate stack on ``lane``."""
    block = LaneTestBlock(lane, dim).eval()

    s05 = causality_stability_gate(
        block,
        seq_len=seq_len,
        dim=dim,
        causality_threshold=causality_threshold,
    )
    if not s05.passed:
        return _empty_scorecard(
            spec,
            eliminated_by="s05_causality_stability",
            s05_stability_passed=s05.stability_passed,
            s05_causality_passed=s05.causality_passed,
            s05_max_first_half_drift=s05.max_first_half_drift,
            notes=s05.notes + ("skipped downstream gates",),
        )

    erf = measure_erf(
        LaneTestBlock(lane, dim),
        seq_len=seq_len,
        dim=dim,
        density_threshold=erf_density_threshold,
    )
    if not erf.passed:
        return _empty_scorecard(
            spec,
            eliminated_by="erf_density",
            s05_passed=True,
            s05_stability_passed=True,
            s05_causality_passed=True,
            s05_max_first_half_drift=s05.max_first_half_drift,
            erf_density=erf.density,
            erf_density_entropy=erf.density_entropy,
            erf_decay_slope=erf.decay_slope,
            notes=erf.notes,
        )

    nb_block = LaneTestBlock(lane, dim)
    nb = nano_bind_gate(nb_block, dim=dim, seq_len=seq_len, n_classes=nb_n_classes)
    if not nb.passed:
        return _empty_scorecard(
            spec,
            eliminated_by="nano_bind",
            s05_passed=True,
            s05_stability_passed=True,
            s05_causality_passed=True,
            s05_max_first_half_drift=s05.max_first_half_drift,
            erf_passed=True,
            erf_density=erf.density,
            erf_density_entropy=erf.density_entropy,
            erf_decay_slope=erf.decay_slope,
            nb_max_accuracy=nb.max_accuracy,
            nb_rejected_persistent_zero=nb.rejected_persistent_zero,
            notes=nb.notes,
        )

    binds, recall = _run_ar_probes(
        lane,
        dim=dim,
        seq_len=seq_len,
        probes=probes,
        binding_threshold=binding_threshold,
    )

    return CapabilityScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        s05_passed=True,
        s05_stability_passed=True,
        s05_causality_passed=True,
        s05_max_first_half_drift=s05.max_first_half_drift,
        erf_passed=True,
        erf_density=erf.density,
        erf_density_entropy=erf.density_entropy,
        erf_decay_slope=erf.decay_slope,
        nb_passed=True,
        nb_max_accuracy=nb.max_accuracy,
        nb_rejected_persistent_zero=nb.rejected_persistent_zero,
        can_bind=any(binds.values()),
        binds_per_probe=binds,
        relative_recall_per_probe={k: round(v, 3) for k, v in recall.items()},
        eliminated_by=None,
    )


def capability_scorecard_to_dict(card: CapabilityScorecard) -> dict[str, Any]:
    return {
        "proposal_id": card.proposal_id,
        "name": card.name,
        "s05_passed": card.s05_passed,
        "s05_stability_passed": card.s05_stability_passed,
        "s05_causality_passed": card.s05_causality_passed,
        "s05_max_first_half_drift": card.s05_max_first_half_drift,
        "erf_passed": card.erf_passed,
        "erf_density": card.erf_density,
        "erf_density_entropy": card.erf_density_entropy,
        "erf_decay_slope": card.erf_decay_slope,
        "nb_passed": card.nb_passed,
        "nb_max_accuracy": card.nb_max_accuracy,
        "nb_rejected_persistent_zero": card.nb_rejected_persistent_zero,
        "can_bind": card.can_bind,
        "binds_per_probe": dict(card.binds_per_probe),
        "relative_recall_per_probe": dict(card.relative_recall_per_probe),
        "eliminated_by": card.eliminated_by,
        "notes": list(card.notes),
    }
