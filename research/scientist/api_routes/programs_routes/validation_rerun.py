"""Validation rerun queue management: queue, pending, drain, cancel."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict
from flask import jsonify, request

from ...capability_ranker_metrics import enable_capability_rankers
from ...runner._types import RunConfig
from .._helpers import get_runner
from .._strategy_preflight import (
    has_backfill_provenance,
    resolve_confirmed_candidate_result_id,
)

from ._shared import (
    attach_candidate_confirmation_status,
    _leaderboard_backed_program_detail,
)

logger = logging.getLogger(__name__)

_STAGE_DEFAULT_STEPS = {
    "screening": 750,  # STAGE1_STEPS
    "investigation": 2500,  # INVESTIGATION_STEPS
    "capability_ranking": 2500,
    "validation": 10000,  # VALIDATION_STEPS
}
_STAGE_QUEUE_NAMES = {
    "screening": "replay",  # S1 reruns go through the exact_graph_replay path
    "investigation": "investigation",
    "capability_ranking": "capability_ranking",
    "validation": "validation",
}


def _json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _task_matches_result(row: Any, result_id: str) -> bool:
    """Return true when a queued task belongs on a program-detail screen.

    Backfill reruns may execute against a candidate-confirmed sibling while
    still being requested from the original backfill result.  The executable
    result id lives in ``result_ids_json``; the screen owner lives in
    ``metadata_json.requested_result_id``.
    """
    rid = str(result_id)
    ids = [str(x) for x in _json_list(row["result_ids_json"])]
    if rid in ids:
        return True
    metadata = _json_object(row["metadata_json"])
    if str(metadata.get("requested_result_id") or "") == rid:
        return True
    requested_ids = metadata.get("requested_result_ids")
    if isinstance(requested_ids, list) and rid in [str(x) for x in requested_ids]:
        return True
    return False


def _api_program_queue_validation_rerun(result_id, nb=None):
    """Queue N reruns at a chosen stage for a program.

    Each rerun is a row in ``followup_tasks``; the runner claims them
    sequentially and re-runs the stage's pipeline (S1 screening replay
    via exact_graph_replay, investigation via start_investigation, or
    validation via start_validation).  Each completed rerun produces a
    new ``program_results`` row; the leaderboard aggregator means the
    metrics across rows of the same fingerprint+tier.

    Body (optional):
        stage  str   "screening" | "investigation" | "capability_ranking" | "validation"
                     (default "validation").
        n      int   number of reruns (default 1, max 5).
        n_seeds int  seeds per rerun (default 1; only used at validation).
        n_steps int  step budget per rerun (default depends on stage).
        candidate_confirmation bool  screening reruns always use this path.
        reason str   free-text shown in evidence_pack.
    """
    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Program not found"}), 404

    body = request.get_json(silent=True) or {}
    stage_in = str(body.get("stage") or "validation").strip().lower()
    if stage_in not in _STAGE_DEFAULT_STEPS:
        return (
            jsonify({"error": f"stage must be one of {sorted(_STAGE_DEFAULT_STEPS)}"}),
            400,
        )
    try:
        n_req = int(body.get("n", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "n must be an integer"}), 400
    if n_req < 1 or n_req > 5:
        return jsonify({"error": "n must be in [1, 5]"}), 400
    reason = str(body.get("reason") or "").strip()[:500]
    try:
        n_seeds = max(1, min(5, int(body.get("n_seeds", 1))))
    except (TypeError, ValueError):
        n_seeds = 1
    try:
        n_steps = int(body.get("n_steps", _STAGE_DEFAULT_STEPS[stage_in]))
    except (TypeError, ValueError):
        n_steps = _STAGE_DEFAULT_STEPS[stage_in]
    n_steps = max(50, min(50000, n_steps))

    fp = (program.get("graph_fingerprint") or "").strip() or None
    queue_stage = _STAGE_QUEUE_NAMES[stage_in]
    queue_result_id = str(result_id)

    if queue_stage in {
        "investigation",
        "capability_ranking",
        "validation",
    } and has_backfill_provenance(program):
        confirmed_id = resolve_confirmed_candidate_result_id(
            nb,
            str(result_id),
            program=program,
            leaderboard_entry=nb.get_leaderboard_entry(str(result_id)),
        )
        if not confirmed_id:
            return (
                jsonify(
                    {
                        "error": (
                            f"{stage_in.title()} rerun requires a candidate-confirmed "
                            "Stage-1 sibling. Queue S1 confirmation first."
                        ),
                        "code": "candidate_confirmation_required",
                        "result_id": str(result_id),
                        "graph_fingerprint": fp,
                    }
                ),
                409,
            )
        queue_result_id = confirmed_id
        program = nb.get_program_detail(confirmed_id) or program

    if queue_stage == "replay":
        # S1 screening replay path: exact_graph_replay reads
        # ``repeat_per_source``, ``device``, ``fast`` from config_json.
        # CRITICAL: fast=False for stability reruns.  fast=True triggers
        # _apply_fast_replay_budget, which clamps stage1_steps to 80
        # regardless of the user's request — at that budget most archs
        # fail S0/S05 gates and produce zero new rows.  We want the full
        # user-specified budget so the rerun is comparable to the
        # original sample.
        config_payload = {
            "repeat_per_source": 1,
            "device": "cuda",
            "fast": False,
            "stage1_steps": n_steps,
            "candidate_confirmation": True,
        }
    else:
        config = RunConfig()
        config.gbm_prescreener_enabled = False
        config.allow_unproven_ml_influence = False
        if queue_stage == "investigation":
            config.investigation_steps = n_steps
        elif queue_stage == "capability_ranking":
            config.investigation_steps = n_steps
            enable_capability_rankers(config)
        else:  # validation
            config.validation_n_seeds = n_seeds
            config.validation_steps = n_steps
        config_payload = config.to_dict()

    task_ids: list[str] = []
    for i in range(n_req):
        tid = nb.enqueue_followup_task(
            stage=queue_stage,
            result_ids=[queue_result_id],
            hypothesis=(
                f"User-triggered {stage_in} rerun: add a sample at "
                f"{n_steps}-step budget for mean/CV aggregation within "
                f"the {stage_in} pool."
            ),
            config=config_payload,
            evidence_pack={
                "reason": reason,
                "rerun_index": i + 1,
                "rerun_total": n_req,
                "stage": stage_in,
                "n_steps": n_steps,
                "n_seeds": n_seeds,
                "fingerprint": fp,
            },
            source_context="program_detail_rerun",
            priority_score=float(program.get("composite_score") or 0.0),
            priority_reasons={
                "policy": "user_triggered_program_detail",
                "stage": stage_in,
                "n_steps": n_steps,
                "reason": reason or None,
                "fingerprint": fp,
            },
            metadata={
                "source": "ui_program_detail",
                "stage": stage_in,
                "requested_result_id": str(result_id),
                "n_steps": n_steps,
                "n_seeds": n_seeds,
                "rerun_index": i + 1,
                "rerun_total": n_req,
            },
            bypass_dedup=True,
        )
        if tid:
            task_ids.append(tid)
    return jsonify(
        {
            "status": "queued",
            "result_id": queue_result_id,
            "requested_result_id": str(result_id),
            "graph_fingerprint": fp,
            "stage": stage_in,
            "n_steps": n_steps,
            "n_seeds": n_seeds,
            "n_requested": n_req,
            "task_ids": task_ids,
            "queued_count": len(task_ids),
        }
    )


def _api_program_pending_reruns(result_id, nb=None):
    """List queued/running reruns for a program across all stages.

    Filters ``followup_tasks`` for stage in (replay, investigation,
    capability_ranking, validation) — i.e. the stages exposed by the rerun panel.
    Returns the most recent 50 with status, queued time, source
    context, and the stage label inferred from evidence_pack.
    """
    rows = nb.conn.execute("""SELECT task_id, stage, status, source_context,
                  result_ids_json, timestamp,
                  started_timestamp, completed_timestamp,
                  outcome, priority_score, evidence_pack_json,
                  metadata_json
           FROM followup_tasks
           WHERE stage IN ('replay', 'investigation', 'capability_ranking', 'validation')
             AND status IN ('queued','running')
           ORDER BY timestamp DESC
           LIMIT 300""").fetchall()
    rid = str(result_id)
    out: list[Dict[str, Any]] = []
    for r in rows:
        if not _task_matches_result(r, rid):
            continue
        ids = [str(x) for x in _json_list(r["result_ids_json"])]
        evidence = _json_object(r["evidence_pack_json"])
        metadata = _json_object(r["metadata_json"])
        # Map runner-stage back to user-facing label: 'replay' = S1
        # screening rerun.
        runner_stage = r["stage"]
        ui_stage = evidence.get("stage") or (
            "screening" if runner_stage == "replay" else runner_stage
        )
        out.append(
            {
                "task_id": r["task_id"],
                "status": r["status"],
                "stage": ui_stage,
                "runner_stage": runner_stage,
                "n_steps": evidence.get("n_steps"),
                "n_seeds": evidence.get("n_seeds"),
                "source_context": r["source_context"],
                "queued_at": r["timestamp"],
                "started_at": r["started_timestamp"],
                "completed_at": r["completed_timestamp"],
                "outcome": r["outcome"],
                "priority_score": r["priority_score"],
                "result_ids": ids,
                "requested_result_id": metadata.get("requested_result_id"),
                "rerun_index": evidence.get("rerun_index"),
                "rerun_total": evidence.get("rerun_total"),
                "reason": evidence.get("reason"),
            }
        )
        if len(out) >= 50:
            break
    program = {"result_id": rid}
    attach_candidate_confirmation_status(nb, program)
    return jsonify(
        {
            "result_id": rid,
            "tasks": out,
            "candidate_confirmation_status": program.get(
                "candidate_confirmation_status",
                {"status": "none"},
            ),
        }
    )


def _api_drain_pending_validation_rerun(notebook_path: str, nb=None):
    """Pop one queued rerun (any stage) and start it now.

    Stage priority: replay (S1) > investigation > capability ranking > validation.  Mirrors
    what continuous mode does on each cycle tick.  Refuses if an
    experiment is already running.

    Returns the runner-stage that was launched and the task_id, or
    ``idle`` if all queues are empty / ``no_op`` if the runner refused.
    """
    runner = get_runner(notebook_path)
    if runner.is_running:
        running_id = getattr(runner, "current_experiment_id", None)
        return (
            jsonify(
                {
                    "status": "busy",
                    "running_experiment_id": running_id,
                    "message": "An experiment is already running; queue will drain when it finishes.",
                }
            ),
            409,
        )

    body = request.get_json(silent=True) or {}
    target_result_id = str(body.get("result_id") or "").strip()

    drain_stages = (
        ("replay", runner._run_pending_replay),
        ("investigation", runner._run_pending_investigation),
        ("capability_ranking", runner._run_pending_capability_ranking),
        ("validation", runner._run_pending_validation),
    )
    if target_result_id:
        return _drain_targeted_rerun(nb, runner, target_result_id, drain_stages)

    return _drain_next_rerun(nb, runner, drain_stages)


def _drain_targeted_rerun(nb, runner, target_result_id: str, drain_stages):
    for stage_name, drain_fn in drain_stages:
        rows = nb.conn.execute(
            """
            SELECT task_id, result_ids_json, metadata_json
            FROM followup_tasks
            WHERE stage = ?
              AND status = 'queued'
            ORDER BY priority_score DESC, timestamp ASC
            """,
            (stage_name,),
        ).fetchall()
        row = next((r for r in rows if _task_matches_result(r, target_result_id)), None)
        if row is None:
            continue
        task_id = str(row["task_id"])
        try:
            drain_fn(task_id=task_id)
        except Exception as exc:
            logger.exception("Failed to drain pending %s rerun", stage_name)
            return jsonify({"error": f"drain failed: {exc}"}), 500
        refreshed = nb.conn.execute(
            "SELECT status FROM followup_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        status = refreshed["status"] if refreshed is not None else None
        if status != "queued":
            return jsonify(
                {
                    "status": "launched",
                    "stage": stage_name,
                    "task_ids": [task_id],
                    "result_id": target_result_id,
                    "running_experiment_id": getattr(
                        runner, "current_experiment_id", None
                    ),
                }
            )
    return jsonify(
        {
            "status": "idle",
            "result_id": target_result_id,
            "message": "no queued rerun task for this program",
        }
    )


def _queued_task_ids(nb, stage_name: str) -> set[str]:
    return {
        row["task_id"]
        for row in nb.conn.execute(
            "SELECT task_id FROM followup_tasks WHERE stage = ? AND status='queued'",
            (stage_name,),
        ).fetchall()
    }


def _drain_next_rerun(nb, runner, drain_stages):
    for stage_name, drain_fn in drain_stages:
        pre = _queued_task_ids(nb, stage_name)
        if not pre:
            continue
        try:
            drain_fn()
        except Exception as exc:
            logger.exception("Failed to drain pending %s rerun", stage_name)
            return jsonify({"error": f"drain failed: {exc}"}), 500
        launched = list(pre - _queued_task_ids(nb, stage_name))
        if launched:
            return jsonify(
                {
                    "status": "launched",
                    "stage": stage_name,
                    "task_ids": launched,
                    "running_experiment_id": getattr(
                        runner, "current_experiment_id", None
                    ),
                }
            )
    return jsonify({"status": "idle", "message": "no queued rerun tasks"})


def _api_program_cancel_rerun(result_id, task_id, nb=None):
    """Cancel a queued validation rerun task.

    Refuses to cancel if the task is already running — at that point
    the runner owns it.
    """
    row = nb.conn.execute(
        """SELECT status, result_ids_json, metadata_json
           FROM followup_tasks
           WHERE task_id = ?""",
        (task_id,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "task not found"}), 404
    if not _task_matches_result(row, str(result_id)):
        return jsonify({"error": "task does not belong to this program"}), 400
    if row["status"] != "queued":
        return (
            jsonify({"error": f"cannot cancel task in status {row['status']!r}"}),
            409,
        )
    nb.conn.execute(
        """UPDATE followup_tasks
              SET status = 'cancelled',
                  completed_timestamp = ?,
                  outcome = 'user_cancelled'
            WHERE task_id = ? AND status = 'queued'""",
        (time.time(), task_id),
    )
    nb._maybe_commit()
    return jsonify({"status": "cancelled", "task_id": task_id})
