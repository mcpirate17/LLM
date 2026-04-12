"""Scoring helpers for investigation candidate analysis.

This module isolates per-candidate scoring and packaging so the investigation
thread can remain focused on orchestration, checkpointing, and I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..thresholds import (
    INVESTIGATION_BRITTLE_OVERRIDE_LR,
    INVESTIGATION_EARLY_PASS_LR,
)


INFRA_ERROR_MARKERS = (
    "cuda",
    "illegal memory",
    "device-side assert",
    "out of memory",
)


@dataclass(frozen=True)
class InvestigationProgramSummary:
    """Compact candidate-level summary derived from investigation program runs."""

    n_passed: int
    robustness: float
    best_tp: dict[str, Any] | None
    best_lr: float | None
    lr_multiplier: float | None
    brittle_risk: bool
    investigation_passed_early: bool
    training_errors: list[str]
    infra_failures: int
    real_failures: int


def classify_investigation_failures(
    tp_results: Sequence[Mapping[str, Any]],
    infra_markers: Sequence[str] = INFRA_ERROR_MARKERS,
) -> tuple[int, int]:
    """Classify infra-only versus real failures from training-program results."""

    infra_failures = sum(
        1
        for result in tp_results
        if not result.get("passed")
        and any(
            marker in str(result.get("error") or "").lower() for marker in infra_markers
        )
    )
    real_failures = (
        len(tp_results)
        - infra_failures
        - sum(1 for result in tp_results if result.get("passed"))
    )
    return infra_failures, real_failures


def summarize_investigation_program_runs(
    *,
    tp_results: Sequence[Mapping[str, Any]],
    screening_lr: float | None,
    investigation_max_loss_ratio_multiplier: float,
    loss_multiplier_fn,
) -> InvestigationProgramSummary:
    """Reduce investigation training-program runs to one scoring summary."""

    n_passed = sum(1 for result in tp_results if result.get("passed"))
    robustness = n_passed / max(len(tp_results), 1)
    best_tp = min(
        (result for result in tp_results if result.get("loss_ratio") is not None),
        key=lambda result: result["loss_ratio"],
        default=None,
    )
    best_lr = float(best_tp["loss_ratio"]) if best_tp is not None else None
    lr_multiplier = loss_multiplier_fn(screening_lr, best_lr)
    brittle_risk = lr_multiplier is not None and lr_multiplier > float(
        investigation_max_loss_ratio_multiplier
    )
    investigation_passed_early = (best_lr or 1.0) < INVESTIGATION_EARLY_PASS_LR and (
        not brittle_risk
        or (best_lr is not None and best_lr < INVESTIGATION_BRITTLE_OVERRIDE_LR)
    )
    infra_failures, real_failures = classify_investigation_failures(tp_results)
    return InvestigationProgramSummary(
        n_passed=n_passed,
        robustness=robustness,
        best_tp=best_tp,
        best_lr=best_lr,
        lr_multiplier=lr_multiplier,
        brittle_risk=brittle_risk,
        investigation_passed_early=investigation_passed_early,
        training_errors=[
            str(result["error"]) for result in tp_results if result.get("error")
        ],
        infra_failures=infra_failures,
        real_failures=real_failures,
    )


def build_investigation_entry(
    *,
    source_result_id: str,
    config,
    source: Mapping[str, Any],
    tp_sched: Mapping[str, Any],
    n_programs_tested: int,
    fingerprint_incomplete: bool,
    summary: InvestigationProgramSummary,
) -> dict[str, Any]:
    """Build the persisted investigation-results entry for one candidate."""

    return {
        "result_id": source_result_id,
        "data_mode": str(getattr(config, "data_mode", "random") or "random"),
        "data_source": str(
            getattr(config, "hf_dataset", None)
            or getattr(config, "corpus_path", None)
            or "random"
        ),
        "robustness": summary.robustness,
        "best_loss_ratio": summary.best_lr,
        "screening_loss_ratio": source.get("loss_ratio"),
        "baseline_loss_ratio": source.get("baseline_loss_ratio"),
        "novelty_confidence": source.get("novelty_confidence"),
        "loss_ratio_multiplier": summary.lr_multiplier,
        "brittle_risk": summary.brittle_risk,
        "investigation_passed": summary.investigation_passed_early,
        "fingerprint_incomplete": fingerprint_incomplete,
        "n_programs_passed": summary.n_passed,
        "n_programs_tested": n_programs_tested,
        "best_training_program": summary.best_tp.get("training_program")
        if summary.best_tp
        else None,
        "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
        "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
        "training_errors": summary.training_errors,
    }
