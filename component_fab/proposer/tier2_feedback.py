"""Tier-2 cohort feedback for dynamic fab proposal generation.

The heavy work happens in ``research.tools.run_tier2_binding_cohort``. This
module only reads its small JSON summaries and turns per-task deltas into
compact signatures that the proposer and ranker can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
_AUDIT_DIR = _REPO / "tasks" / "audit"

TASK_LONG_GAP = "long_gap_recall"
TASK_COMPOSITIONAL = "compositional_binding"
TASK_DISTRACTOR = "distractor_kv_recall"
TASK_BROAD_KV: frozenset[str] = frozenset(
    {
        "multi_query_kv_recall",
        "variable_layout_recall",
        "heldout_pair_recall",
    }
)
TASK_NICHE: frozenset[str] = frozenset({TASK_LONG_GAP, TASK_COMPOSITIONAL})

WEAK_NARROW_DISTRACTOR_ONLY = "tier2_narrow_distractor_only"
WEAK_FAIL_LONG_GAP = "tier2_fail_long_gap_recall"
WEAK_FAIL_COMPOSITIONAL = "tier2_fail_compositional_binding"
WEAK_FAIL_BROAD_KV = "tier2_fail_broad_kv_recall"
WEAK_NEAR_SURVIVOR = "tier2_near_survivor"
WEAK_REJECTED = "tier2_rejected"


@dataclass(frozen=True, slots=True)
class Tier2TaskResult:
    task: str
    candidate_eval_acc: float
    baseline_max: float
    delta: float
    beats: bool


@dataclass(frozen=True, slots=True)
class Tier2Feedback:
    proposal_id: str
    name: str
    pass_count: int
    n_tasks: int
    tier2_passed: bool
    tier2_passed_niche: bool
    mean_delta: float
    wins: tuple[str, ...]
    failures: tuple[str, ...]
    signatures: tuple[str, ...]
    task_results: tuple[Tier2TaskResult, ...]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _task_result(task: str, row: Mapping[str, Any]) -> Tier2TaskResult:
    return Tier2TaskResult(
        task=task,
        candidate_eval_acc=_as_float(row.get("candidate_eval_acc")),
        baseline_max=_as_float(row.get("baseline_max")),
        delta=_as_float(row.get("delta")),
        beats=bool(row.get("beats")),
    )


def signatures_for_tasks(task_results: Sequence[Tier2TaskResult]) -> tuple[str, ...]:
    wins = {row.task for row in task_results if row.beats}
    failures = {row.task for row in task_results if not row.beats}
    signatures: list[str] = []
    if wins == {TASK_DISTRACTOR}:
        signatures.append(WEAK_NARROW_DISTRACTOR_ONLY)
    if TASK_LONG_GAP in failures:
        signatures.append(WEAK_FAIL_LONG_GAP)
    if TASK_COMPOSITIONAL in failures:
        signatures.append(WEAK_FAIL_COMPOSITIONAL)
    if any(task in failures for task in TASK_BROAD_KV):
        signatures.append(WEAK_FAIL_BROAD_KV)
    if len(wins) >= 2 and bool(wins & TASK_NICHE):
        signatures.append(WEAK_NEAR_SURVIVOR)
    if len(wins) < 4:
        signatures.append(WEAK_REJECTED)
    return tuple(dict.fromkeys(signatures))


def feedback_from_result(
    proposal_id: str, row: Mapping[str, Any]
) -> Tier2Feedback | None:
    if row.get("status") != "ok":
        return None
    per_task = row.get("per_task") or {}
    task_results = tuple(
        _task_result(str(task), task_row)
        for task, task_row in sorted(per_task.items())
        if isinstance(task_row, Mapping)
    )
    if not task_results:
        return None
    wins = tuple(result.task for result in task_results if result.beats)
    failures = tuple(result.task for result in task_results if not result.beats)
    mean_delta = sum(result.delta for result in task_results) / len(task_results)
    return Tier2Feedback(
        proposal_id=proposal_id,
        name=str(row.get("name") or proposal_id),
        pass_count=int(row.get("pass_count") or len(wins)),
        n_tasks=int(row.get("n_tasks") or len(task_results)),
        tier2_passed=bool(row.get("tier2_passed")),
        tier2_passed_niche=bool(row.get("tier2_passed_niche")),
        mean_delta=mean_delta,
        wins=wins,
        failures=failures,
        signatures=signatures_for_tasks(task_results),
        task_results=task_results,
    )


def load_tier2_feedback(
    paths: Sequence[Path | str] | None = None,
) -> dict[str, Tier2Feedback]:
    """Load Tier-2 feedback from explicit paths or recent audit artifacts."""

    resolved = [Path(path) for path in paths] if paths else latest_tier2_artifacts()
    out: dict[str, Tier2Feedback] = {}
    for path in resolved:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results = payload.get("results") or {}
        if not isinstance(results, Mapping):
            continue
        for proposal_id, row in results.items():
            if not isinstance(row, Mapping):
                continue
            feedback = feedback_from_result(str(proposal_id), row)
            if feedback is not None:
                out[feedback.proposal_id] = feedback
    return out


def latest_tier2_artifacts(*, limit: int = 4) -> list[Path]:
    if not _AUDIT_DIR.exists():
        return []
    paths = sorted(
        _AUDIT_DIR.glob("fab_tier2*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    return paths[:limit]


def tier2_score_multiplier(feedback: Tier2Feedback | None) -> float:
    """Return an autonomous-score multiplier from downstream evidence."""

    if feedback is None:
        return 1.0
    if feedback.tier2_passed:
        return 1.05
    signatures = set(feedback.signatures)
    if WEAK_NARROW_DISTRACTOR_ONLY in signatures:
        return 0.55
    if WEAK_NEAR_SURVIVOR in signatures and feedback.mean_delta > 0.0:
        return 0.85
    if feedback.pass_count <= 1:
        return 0.65
    return 0.75
