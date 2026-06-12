"""Tiered capability validator — chained gates ordered cheapest-first.

Gate stack (cheapest first):

  1. **S0.5 stability + causality** — forward-only, ~ms. Hard reject for
     non-causal primitives and pathological numerics.
  2. **ERF density** — single forward+backward. Hard reject for
     information-bottleneck primitives.
  3. **AR Gate / NB 0.5 binding** — short K-class binding training. Hard
     reject for frequency-mode-collapse degenerates.
  4. **Nano induction (NI 0.5)** — 2-stacked-block induction probe. **Soft
     signal only** — records ``ind_max_accuracy`` / ``ind_above_baseline``
     so the autonomous loop can rank, but never sets ``eliminated_by``.
     Pure WTA architectures (TropicalAttention alone) are expected to
     score at-or-near baseline here; we want that recorded, not zeroed.
  5. **AR easy / medium retrieval** — sprint-7 binding probes. Soft signal
     (boosts composite when ``can_bind``); not hard-rejection.

Each gate emits an elimination flag (or ``None``) in the scorecard so
the autonomous loop can surface "N eliminated by gate X" per cycle.
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
from ..harness.nano_induction_probe import nano_induction_gate
from ..harness.range_binding_probe import DEFAULT_DISTANCES, range_binding_gate
from ..harness.standard_block import LaneTestBlock
from ..proposer.spec_generator import ProposalSpec
from ..state.gates import (
    GATE_ERF_DENSITY,
    GATE_NANO_BIND,
    GATE_S05_CAUSALITY_STABILITY,
)


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
    ind_ran: bool
    ind_max_accuracy: float
    ind_final_accuracy: float
    ind_above_baseline: bool
    can_bind: bool
    binds_per_probe: dict[str, bool]
    relative_recall_per_probe: dict[str, float]
    eliminated_by: str | None
    range_ran: bool = False
    range_aggregate_acc: float = 0.0
    range_effective_distance: int = 0
    range_max_accuracy: float = 0.0
    range_per_distance: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)


_DEFAULT_SIGNALS: dict[str, Any] = {
    "s05_passed": False,
    "s05_stability_passed": False,
    "s05_causality_passed": False,
    "s05_max_first_half_drift": 0.0,
    "erf_passed": False,
    "erf_density": 0.0,
    "erf_density_entropy": 0.0,
    "erf_decay_slope": 0.0,
    "nb_passed": False,
    "nb_max_accuracy": 0.0,
    "nb_rejected_persistent_zero": False,
    "ind_ran": False,
    "ind_max_accuracy": 0.0,
    "ind_final_accuracy": 0.0,
    "ind_above_baseline": False,
}


def _scorecard(
    spec: ProposalSpec,
    signals: dict[str, Any],
    *,
    eliminated_by: str | None,
    notes: tuple[str, ...] = (),
    binds: dict[str, bool] | None = None,
    recall: dict[str, float] | None = None,
) -> CapabilityScorecard:
    """Materialize a scorecard from the running signals dict."""
    merged = {**_DEFAULT_SIGNALS, **signals}
    binds_map = binds or {}
    return CapabilityScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        can_bind=any(binds_map.values()),
        binds_per_probe=binds_map,
        relative_recall_per_probe={k: round(v, 3) for k, v in (recall or {}).items()},
        eliminated_by=eliminated_by,
        notes=notes,
        **merged,
    )


def _s05_signals(s05: Any) -> dict[str, Any]:
    return {
        "s05_passed": bool(s05.passed),
        "s05_stability_passed": bool(s05.stability_passed),
        "s05_causality_passed": bool(s05.causality_passed),
        "s05_max_first_half_drift": float(s05.max_first_half_drift),
    }


def _erf_signals(erf: Any) -> dict[str, Any]:
    return {
        "erf_passed": bool(erf.passed),
        "erf_density": float(erf.density),
        "erf_density_entropy": float(erf.density_entropy),
        "erf_decay_slope": float(erf.decay_slope),
    }


def _nb_signals(nb: Any) -> dict[str, Any]:
    return {
        "nb_passed": bool(nb.passed),
        "nb_max_accuracy": float(nb.max_accuracy),
        "nb_rejected_persistent_zero": bool(nb.rejected_persistent_zero),
    }


def _ind_signals(ind: Any) -> dict[str, Any]:
    return {
        "ind_ran": True,
        "ind_max_accuracy": float(ind.max_accuracy),
        "ind_final_accuracy": float(ind.final_accuracy),
        "ind_above_baseline": bool(ind.above_baseline),
    }


def _stacked_induction_block(lane: nn.Module, dim: int) -> nn.Module:
    """Two ``LaneTestBlock`` wrappers sharing ``lane`` params (RNN-style).

    Induction is a 2-hop circuit; a 1-block wrapper cannot pass for any
    primitive, so we stack two before grading. Param-sharing keeps the
    test honest — passing means the lane composes with itself, not that
    a deeper independent stack memorized the task.
    """
    return nn.Sequential(LaneTestBlock(lane, dim), LaneTestBlock(lane, dim))


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


def _range_signals(
    lane: nn.Module,
    dim: int,
    distances: tuple[int, ...],
    n_train_steps: int,
) -> dict[str, Any]:
    """Distance-resolved binding signals from the opt-in range probe."""
    rng = range_binding_gate(
        lane, dim=dim, distances=distances, n_train_steps=n_train_steps
    )
    return {
        "range_ran": True,
        "range_aggregate_acc": float(rng.aggregate_accuracy),
        "range_effective_distance": int(rng.effective_distance),
        "range_max_accuracy": float(rng.max_accuracy),
        "range_per_distance": {
            str(k): round(float(v), 3) for k, v in rng.per_distance_accuracy.items()
        },
    }


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
    ind_n_classes: int = 8,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
    range_distances: tuple[int, ...] = DEFAULT_DISTANCES,
) -> CapabilityScorecard:
    """Run the tiered gate stack on ``lane``."""
    signals: dict[str, Any] = {}

    s05 = causality_stability_gate(
        LaneTestBlock(lane, dim).eval(),
        seq_len=seq_len,
        dim=dim,
        causality_threshold=causality_threshold,
    )
    signals.update(_s05_signals(s05))
    if not s05.passed:
        return _scorecard(
            spec,
            signals,
            eliminated_by=GATE_S05_CAUSALITY_STABILITY,
            notes=s05.notes + ("skipped downstream gates",),
        )

    erf = measure_erf(
        LaneTestBlock(lane, dim),
        seq_len=seq_len,
        dim=dim,
        density_threshold=erf_density_threshold,
    )
    signals.update(_erf_signals(erf))
    if not erf.passed:
        return _scorecard(
            spec, signals, eliminated_by=GATE_ERF_DENSITY, notes=erf.notes
        )

    nb = nano_bind_gate(
        LaneTestBlock(lane, dim),
        dim=dim,
        seq_len=seq_len,
        n_classes=nb_n_classes,
    )
    signals.update(_nb_signals(nb))
    if not nb.passed:
        return _scorecard(spec, signals, eliminated_by=GATE_NANO_BIND, notes=nb.notes)

    ind = nano_induction_gate(
        _stacked_induction_block(lane, dim),
        dim=dim,
        seq_len=max(seq_len, 24),
        n_classes=ind_n_classes,
    )
    signals.update(_ind_signals(ind))

    binds, recall = _run_ar_probes(
        lane,
        dim=dim,
        seq_len=seq_len,
        probes=probes,
        binding_threshold=binding_threshold,
    )

    # Soft signal: distance-resolved binding (sparse/long-range mixing). Opt-in
    # because it trains one model per call (cheap for parallel lanes, slow for
    # sequential-scan lanes). effective_distance is training-budget-limited —
    # the full per-distance curve is the honest signal.
    if run_range_probe:
        signals.update(_range_signals(lane, dim, range_distances, range_train_steps))

    return _scorecard(
        spec,
        signals,
        eliminated_by=None,
        notes=ind.notes,
        binds=binds,
        recall=recall,
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
        "ind_ran": card.ind_ran,
        "ind_max_accuracy": card.ind_max_accuracy,
        "ind_final_accuracy": card.ind_final_accuracy,
        "ind_above_baseline": card.ind_above_baseline,
        "can_bind": card.can_bind,
        "binds_per_probe": dict(card.binds_per_probe),
        "relative_recall_per_probe": dict(card.relative_recall_per_probe),
        "range_ran": card.range_ran,
        "range_aggregate_acc": card.range_aggregate_acc,
        "range_effective_distance": card.range_effective_distance,
        "range_max_accuracy": card.range_max_accuracy,
        "range_per_distance": dict(card.range_per_distance),
        "eliminated_by": card.eliminated_by,
        "notes": list(card.notes),
    }
