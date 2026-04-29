"""Shared audit helpers for long-running backfill and trainer scripts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from research.scientist.json_utils import json_safe as _json_ready
from research.scientist.notebook import LabNotebook


def start_script_experiment(
    *,
    db_path: str | Path,
    experiment_type: str,
    config: Dict[str, Any],
    source_script: str,
    hypothesis: str,
    research_question: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[LabNotebook, str]:
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type=experiment_type,
        config=_json_ready(config),
        hypothesis=hypothesis,
        research_question=research_question,
        hypothesis_metadata={
            "source": source_script,
            **(_json_ready(metadata or {})),
        },
    )
    return nb, exp_id


def _result_count(results: Dict[str, Any]) -> int:
    for key in (
        "total",
        "evaluated",
        "updated",
        "components_trained",
        "cases",
        "entries",
        "selected",
    ):
        value = results.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(int(value), 0)
    return 0


def complete_script_experiment(
    nb: LabNotebook,
    experiment_id: str,
    *,
    results: Dict[str, Any],
    summary: str,
) -> None:
    now = time.time()
    started = nb.conn.execute(
        "SELECT started_at FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    duration = now - started["started_at"] if started else 0.0
    payload = _json_ready(results)
    nb.conn.execute(
        """UPDATE experiments SET
           status = 'completed',
           results_json = ?,
           n_programs_generated = ?,
           n_stage0_passed = ?,
           n_stage05_passed = ?,
           n_stage1_passed = ?,
           best_loss_ratio = ?,
           best_novelty_score = ?,
           aria_summary = ?,
           completed_at = ?,
           duration_seconds = ?
           WHERE experiment_id = ?""",
        (
            nb._compress(payload),
            _result_count(payload),
            int(payload.get("stage0_passed") or 0),
            int(payload.get("stage05_passed") or 0),
            int(payload.get("stage1_passed") or 0),
            float(payload["best_loss_ratio"])
            if payload.get("best_loss_ratio") is not None
            else None,
            float(payload["best_novelty_score"])
            if payload.get("best_novelty_score") is not None
            else None,
            summary,
            now,
            duration,
            experiment_id,
        ),
    )
    nb._maybe_commit()
    lifecycle_results = dict(payload)
    lifecycle_results.setdefault("total", _result_count(payload))
    nb._publish_lifecycle_event_safe(
        event_type="experiment_completed",
        run_id=experiment_id,
        payload={
            "completed_at": now,
            "results": lifecycle_results,
            "aria_summary": summary,
            "aria_mood": "contemplative",
            "insights": [],
            "llm_analysis": None,
        },
    )


def fail_script_experiment(
    nb: LabNotebook,
    experiment_id: str,
    *,
    error: str,
    results: Optional[Dict[str, Any]] = None,
) -> None:
    payload = _json_ready(results or {})
    now = time.time()
    started = nb.conn.execute(
        "SELECT started_at FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    duration = now - started["started_at"] if started else 0.0
    nb.conn.execute(
        """UPDATE experiments SET
           status = 'failed',
           completed_at = ?,
           duration_seconds = ?,
           aria_summary = ?,
           results_json = ?,
           n_programs_generated = ?
           WHERE experiment_id = ?""",
        (
            now,
            duration,
            f"FAILED: {error}",
            nb._compress(payload) if payload else None,
            _result_count(payload),
            experiment_id,
        ),
    )
    nb._maybe_commit()
    lifecycle_results = dict(payload)
    lifecycle_results.setdefault("total", _result_count(payload))
    nb._publish_lifecycle_event_safe(
        event_type="experiment_failed",
        run_id=experiment_id,
        payload={
            "completed_at": now,
            "error": error,
            "results": lifecycle_results,
        },
    )


def build_metric_backfill_context(
    *,
    kind: str,
    source_script: str,
    experiment_id: str,
    device: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "source_script": source_script,
        "experiment_id": experiment_id,
        "device": device,
        "updated_at": time.time(),
        **_json_ready(extra),
    }
