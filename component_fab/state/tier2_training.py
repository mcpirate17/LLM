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
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
TIER2_TABLE_PATH = _REPO / "research" / "data" / "tier2_predictor" / "labels.jsonl"


def arch_group(math_axes: Mapping[str, Any]) -> str:
    """Coarse architecture key for leave-architecture-out CV / dedup."""

    return "|".join(
        str(math_axes.get(k))
        for k in ("op_algebraic_space", "op_block_template", "op_routing_kind")
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

    if row.get("status") != "ok":
        return None
    math_axes = dict(row.get("math_axes") or {})
    per_task = row.get("per_task") or {}
    deltas = [float((t or {}).get("delta") or 0.0) for t in per_task.values()]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "proposal_id": proposal_id,
        "name": row.get("name"),
        "math_axes": math_axes,
        "arch_group": arch_group(math_axes),
        # labels / targets
        "mean_delta": mean_delta,
        "min_delta": min(deltas) if deltas else 0.0,
        "pass_count": int(row.get("pass_count") or 0),
        "n_tasks": int(row.get("n_tasks") or len(per_task)),
        "tier2_passed": bool(row.get("tier2_passed")),
        "per_task": {
            str(task): {
                "delta": float((t or {}).get("delta") or 0.0),
                "beats": bool((t or {}).get("beats")),
                "candidate_eval_acc": float((t or {}).get("candidate_eval_acc") or 0.0),
                "baseline_max": float((t or {}).get("baseline_max") or 0.0),
            }
            for task, t in per_task.items()
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

    if not table_path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for line in table_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = row.get("proposal_id")
        if pid:
            by_id[str(pid)] = row
    return list(by_id.values())
