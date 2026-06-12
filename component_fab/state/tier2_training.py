"""Append-only training table for a Tier-2 value predictor.

Every Tier-2 cohort run appends one row per evaluated candidate here, building the
labeled dataset a numerical predictor needs. As of 2026-06-03 a regressor trained
on the 34 labels we had generalized worse than predicting the mean (leave-arch-out
R² < 0) — the blocker is labels, not the model. This table accumulates them so the
predictor (``tools/train_tier2_predictor.py``) becomes trainable as cohorts run.

Rows store the rebuildable recipe (``math_axes``) + the measured outcome only;
features are computed at train time so feature engineering can evolve without
re-running cohorts. This is a permanent dataset (``research/data/``) — NOT a
rotated ledger; it must grow.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ledger import iter_jsonl_records, latest_by_key

_REPO = Path(__file__).resolve().parents[2]
TIER2_TABLE_PATH = _REPO / "research" / "data" / "tier2_predictor" / "labels.jsonl"


def arch_group(math_axes: Mapping[str, Any]) -> str:
    """Coarse architecture key for leave-architecture-out CV / dedup."""

    return "|".join(
        str(math_axes.get(k))
        for k in ("op_algebraic_space", "op_block_template", "op_routing_kind")
    )


@dataclass(frozen=True, slots=True)
class Tier2TaskResult:
    task: str
    candidate_eval_acc: float
    baseline_max: float
    delta: float
    beats: bool


@dataclass(frozen=True, slots=True)
class Tier2RowMetrics:
    """Parsed Tier-2 cohort-summary row — the ONE row parser's output.

    Shared by ``validator/trust.py`` (downstream evidence), this module
    (training labels) and ``proposer/tier2_feedback.py`` (proposal feedback)
    so the three consumers can never drift on the row schema.
    """

    status: str
    name: str | None
    math_axes: dict[str, Any]
    task_results: tuple[Tier2TaskResult, ...]  # row order preserved
    mean_delta: float
    min_delta: float
    wins: tuple[str, ...]
    failures: tuple[str, ...]
    pass_count: int
    n_tasks: int
    tier2_passed: bool
    tier2_passed_niche: bool
    seed_count: int

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_tier2_row(row: Mapping[str, Any]) -> Tier2RowMetrics:
    """Parse one cohort-summary result row into ``Tier2RowMetrics``."""
    per_task = row.get("per_task") or {}
    task_results = tuple(
        Tier2TaskResult(
            task=str(task),
            candidate_eval_acc=float((t or {}).get("candidate_eval_acc") or 0.0),
            baseline_max=float((t or {}).get("baseline_max") or 0.0),
            delta=float((t or {}).get("delta") or 0.0),
            beats=bool((t or {}).get("beats")),
        )
        for task, t in per_task.items()
    )
    deltas = [r.delta for r in task_results]
    name = row.get("name")
    return Tier2RowMetrics(
        status=str(row.get("status") or "unknown"),
        name=str(name) if name is not None else None,
        math_axes=dict(row.get("math_axes") or {}),
        task_results=task_results,
        mean_delta=sum(deltas) / len(deltas) if deltas else 0.0,
        min_delta=min(deltas) if deltas else 0.0,
        wins=tuple(r.task for r in task_results if r.beats),
        failures=tuple(r.task for r in task_results if not r.beats),
        pass_count=int(row.get("pass_count") or 0),
        n_tasks=int(row.get("n_tasks") or len(task_results)),
        tier2_passed=bool(row.get("tier2_passed")),
        tier2_passed_niche=bool(row.get("tier2_passed_niche")),
        seed_count=int(row.get("seed_count") or 0),
    )


def tier2_label_row(
    proposal_id: str,
    row: Mapping[str, Any],
    *,
    baseline_names: Sequence[str],
    dim: int,
    n_blocks: int,
    n_train_steps: int,
    seed_count: int,
    timestamp: str,
) -> dict[str, Any] | None:
    """Build one training row from a cohort result, or None if it did not run ok."""

    metrics = parse_tier2_row(row)
    if not metrics.ok:
        return None
    return {
        "proposal_id": proposal_id,
        "name": metrics.name,
        "math_axes": metrics.math_axes,
        "arch_group": arch_group(metrics.math_axes),
        # labels / targets
        "mean_delta": metrics.mean_delta,
        "min_delta": metrics.min_delta,
        "pass_count": metrics.pass_count,
        "n_tasks": metrics.n_tasks,
        "tier2_passed": metrics.tier2_passed,
        "per_task": {
            r.task: {
                "delta": r.delta,
                "beats": r.beats,
                "candidate_eval_acc": r.candidate_eval_acc,
                "baseline_max": r.baseline_max,
            }
            for r in metrics.task_results
        },
        # provenance — labels are only comparable within the same recipe
        "baseline_names": list(baseline_names),
        "dim": dim,
        "n_blocks": n_blocks,
        "n_train_steps": n_train_steps,
        "seed_count": seed_count,
        "timestamp": timestamp,
    }


def append_tier2_labels(
    results: Mapping[str, Mapping[str, Any]],
    *,
    baseline_names: Sequence[str],
    dim: int,
    n_blocks: int,
    n_train_steps: int,
    seed_count: int,
    table_path: Path = TIER2_TABLE_PATH,
) -> int:
    """Append all ok candidates from a cohort ``results`` dict. Returns rows written."""

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    rows = [
        tier2_label_row(
            pid,
            row,
            baseline_names=baseline_names,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            seed_count=seed_count,
            timestamp=timestamp,
        )
        for pid, row in results.items()
    ]
    rows = [r for r in rows if r is not None]
    if not rows:
        return 0
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return len(rows)


def load_tier2_labels(table_path: Path = TIER2_TABLE_PATH) -> list[dict[str, Any]]:
    """Load all training rows. Latest row per proposal_id wins (re-runs overwrite)."""

    return list(latest_by_key(iter_jsonl_records(table_path), "proposal_id").values())
