"""Shared helpers for programs API route submodules."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, Optional

from ...notebook.graph_artifacts import resolve_graph_json_value

logger = logging.getLogger(__name__)

_TRUST_LABEL_RANK = {
    "": 0,
    "candidate_screening": 1,
    "candidate_grade": 2,
    "reference": 3,
}

_COMPARABILITY_LABEL_RANK = {
    "": 0,
    "screening_only": 1,
    "candidate_comparable": 2,
    "reference_comparable": 3,
}


def _preserve_stronger_label(*values: Any, ranks: Dict[str, int], fallback: str) -> str:
    best = fallback
    best_rank = ranks.get(str(fallback).strip().lower(), 0)
    for value in values:
        normalized = str(value or "").strip().lower()
        if ranks.get(normalized, 0) > best_rank:
            best = normalized
            best_rank = ranks[normalized]
    return best


def _leaderboard_backed_program_detail(nb, result_id: str) -> Optional[Dict[str, Any]]:
    """Synthesize a program-detail payload from leaderboard/reference data."""
    lb = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if not lb:
        return None

    merged: Dict[str, Any] = dict(lb)
    pr = nb.conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if pr:
        merged.update(dict(pr))
        if "graph_json" in merged:
            merged["graph_json"] = resolve_graph_json_value(
                nb.conn,
                nb.db_path,
                merged["graph_json"],
            )
        merged = nb._parse_program_json_fields(merged)

    merged.setdefault("result_id", result_id)
    merged["is_reference"] = bool(merged.get("is_reference"))
    if merged["is_reference"]:
        merged["model_source"] = "reference"
    merged["loss_ratio"] = (
        merged.get("loss_ratio")
        if merged.get("loss_ratio") is not None
        else merged.get("screening_loss_ratio")
    )
    merged["novelty_score"] = (
        merged.get("novelty_score")
        if merged.get("novelty_score") is not None
        else merged.get("screening_novelty")
    )
    if not merged.get("graph_fingerprint"):
        ref = next(
            (row for row in nb.get_references() if row.get("result_id") == result_id),
            None,
        )
        if ref:
            merged["graph_fingerprint"] = ref.get("graph_fingerprint")
            merged["architecture_family"] = ref.get("architecture_family")
            merged["param_count"] = merged.get("param_count") or ref.get("param_count")

    if not merged.get("architecture_family"):
        merged["architecture_family"] = nb._classify_architecture_family(
            graph_json=merged.get("graph_json"),
            routing_mode=merged.get("routing_mode"),
        )
    if merged.get("architecture_family") == "Unknown":
        merged["architecture_family"] = nb._reference_family_fallback(
            merged.get("reference_name")
        )

    if merged.get("graph_json") and isinstance(merged.get("graph_json"), str):
        try:
            merged["graph_json_parsed"] = json.loads(merged["graph_json"])
        except json.JSONDecodeError as exc:
            logger.debug(
                "Failed to parse graph_json for result_id=%s in leaderboard-backed detail: %s",
                result_id,
                exc,
            )

    merged.setdefault("stage1_passed", 1 if merged.get("is_reference") else 0)
    merged.setdefault("has_training_curve", False)
    merged.setdefault("experiment_id", None)
    merged.setdefault("reference_like", bool(merged.get("is_reference")))
    merged.setdefault(
        "most_similar_to",
        merged.get("reference_name") or merged.get("architecture_family"),
    )
    return merged


def attach_candidate_confirmation_status(nb, program: Dict[str, Any]) -> None:
    """Attach UI state for backfill rows undergoing exact replay confirmation."""
    result_id = str(program.get("result_id") or "").strip()
    if not result_id:
        return

    status: Dict[str, Any] = {"status": "none"}
    try:
        task = nb.conn.execute(
            """
            SELECT task_id, status, stage, timestamp, started_timestamp
            FROM followup_tasks
            WHERE stage = 'replay'
              AND status IN ('running', 'queued')
              AND EXISTS (
                  SELECT 1
                  FROM json_each(followup_tasks.result_ids_json)
                  WHERE CAST(value AS TEXT) = ?
              )
            ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                     timestamp DESC
            LIMIT 1
            """,
            (result_id,),
        ).fetchone()
        if task is not None:
            raw_status = str(task["status"] or "").strip().lower()
            status = {
                "status": raw_status,
                "task_id": task["task_id"],
                "stage": "screening",
                "label": (
                    "candidate confirmation running"
                    if raw_status == "running"
                    else "candidate confirmation queued"
                ),
                "queued_at": task["timestamp"],
                "started_at": task["started_timestamp"],
            }
        else:
            exp = nb.conn.execute(
                """
                SELECT experiment_id, status, timestamp, completed_at
                FROM experiments
                WHERE experiment_type = 'exact_graph_replay'
                  AND EXISTS (
                      SELECT 1
                      FROM json_each(experiments.config_json, '$.source_result_ids')
                      WHERE CAST(value AS TEXT) = ?
                  )
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (result_id,),
            ).fetchone()
            if exp is not None:
                replay_result = nb.conn.execute(
                    """
                    SELECT result_id
                    FROM program_results
                    WHERE experiment_id = ?
                      AND COALESCE(model_source, '') = 'exact_graph_replay'
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (exp["experiment_id"],),
                ).fetchone()
                raw_status = str(exp["status"] or "").strip().lower()
                status = {
                    "status": raw_status,
                    "experiment_id": exp["experiment_id"],
                    "stage": "screening",
                    "label": (
                        "candidate confirmed"
                        if raw_status == "completed" and replay_result is not None
                        else f"candidate confirmation {raw_status or 'started'}"
                    ),
                    "queued_at": exp["timestamp"],
                    "completed_at": exp["completed_at"],
                    "confirmed_result_id": (
                        replay_result["result_id"]
                        if replay_result is not None
                        else None
                    ),
                }
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "Candidate confirmation lookup degraded for result_id=%s: %s",
            result_id,
            exc,
        )
        status = {
            "status": "none",
            "degraded": True,
            "error": str(exc),
        }

    program["candidate_confirmation_status"] = status
    if status.get("status") in {"queued", "running"}:
        program["display_result_cohort"] = f"confirmation_{status['status']}"
        program["display_trust_label"] = status["label"]


def _get_cached_program_explanation(nb, result_id: str) -> Optional[str]:
    row = nb.conn.execute(
        "SELECT llm_explanation FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if not row:
        return None
    explanation = row[0] if isinstance(row, (tuple, list)) else row["llm_explanation"]
    return explanation or None


def _generate_program_explanation(
    nb, result_id: str, program: Dict[str, Any]
) -> Optional[str]:
    from ..llm.context_experiment import build_program_context
    from ._helpers import get_aria_for_notebook

    aria = get_aria_for_notebook(str(nb.db_path))
    explanation = aria.explain_fingerprint(build_program_context(program))
    if not explanation:
        return None
    nb.conn.execute(
        "UPDATE program_results SET llm_explanation = ? WHERE result_id = ?",
        (explanation, result_id),
    )
    nb.conn.commit()
    return explanation
