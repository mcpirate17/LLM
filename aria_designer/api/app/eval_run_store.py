from __future__ import annotations

import threading
import time as _time_mod
from typing import Any, Dict, List, Optional

from . import database as db
from .models import utc_now_iso as _utc_now
from .type_utils import dig

_EVAL_RUNS: Dict[str, Dict[str, Any]] = {}
_EVAL_RUNS_LOCK = threading.Lock()
_EVAL_RUNS_MAX = 200
_EVAL_RUNS_TTL_S = 3600


def _persist_run_snapshot(run_id: str) -> None:
    run = _EVAL_RUNS.get(run_id)
    if not run:
        return
    workflow_id = run.get("workflow_id")
    if not workflow_id:
        return
    error_details = run.get("error_details")
    if error_details is None and (run.get("error") or run.get("error_stage")):
        error_details = {
            "stage": run.get("error_stage"),
            "error_type": run.get("status") if run.get("status") != "success" else None,
            "error_message": run.get("error"),
            "root_cause_code": run.get("error_stage") or "unknown",
        }
    db.save_workflow_run(
        workflow_id=str(workflow_id),
        run_id=run_id,
        status=str(run.get("status") or "unknown"),
        results=run.get("result"),
        perf=run.get("perf_contract"),
        stages=run.get("stages"),
        error=error_details,
        semantic_warnings=run.get("semantic_warnings"),
        started_at=run.get("created_at"),
        completed_at=run.get("completed_at"),
        updated_at=run.get("completed_at") or _utc_now(),
    )


def _evict_old_runs() -> None:
    cutoff = _time_mod.time() - _EVAL_RUNS_TTL_S
    expired = [k for k, v in _EVAL_RUNS.items() if v.get("_created_ts", 0) < cutoff]
    for k in expired:
        del _EVAL_RUNS[k]
    if len(_EVAL_RUNS) > _EVAL_RUNS_MAX:
        by_ts = sorted(_EVAL_RUNS.items(), key=lambda kv: kv[1].get("_created_ts", 0))
        for k, _ in by_ts[: len(_EVAL_RUNS) - _EVAL_RUNS_MAX]:
            del _EVAL_RUNS[k]


def _store_run(run_id: str, data: Dict[str, Any]) -> None:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        data["_created_ts"] = _time_mod.time()
        _EVAL_RUNS[run_id] = data
        _persist_run_snapshot(run_id)


def _update_run(run_id: str, updates: Dict[str, Any]) -> None:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        if run_id in _EVAL_RUNS:
            _EVAL_RUNS[run_id].update(updates)
            _EVAL_RUNS[run_id]["_updated_ts"] = _time_mod.time()
            _persist_run_snapshot(run_id)


def _get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        run = _EVAL_RUNS.get(run_id)
        if run is not None:
            return run
    persisted = db.get_workflow_run(run_id)
    if persisted is None:
        return None
    compact = {
        "run_id": persisted.get("run_id"),
        "workflow_id": persisted.get("workflow_id"),
        "status": persisted.get("status", "unknown"),
        "created_at": persisted.get("started_at"),
        "completed_at": persisted.get("completed_at"),
        "total_time_ms": dig(persisted, "result", "total_time_ms", default=None)
        or persisted.get("total_time_ms"),
        "stages": persisted.get("stages", {}),
        "result": persisted.get("result", {}),
        "perf_contract": persisted.get("perf_contract"),
        "semantic_warnings": persisted.get("semantic_warnings", []),
        "error_details": persisted.get("error_details"),
    }
    if persisted.get("error_details"):
        compact["error"] = dig(
            persisted, "error_details", "error_message", default=None
        )
        compact["error_stage"] = dig(persisted, "error_details", "stage", default=None)
    return compact


def _list_runs() -> List[Dict[str, Any]]:
    with _EVAL_RUNS_LOCK:
        _evict_old_runs()
        out: List[Dict[str, Any]] = []
        for run_id, data in sorted(
            _EVAL_RUNS.items(),
            key=lambda kv: kv[1].get("_created_ts", 0),
            reverse=True,
        ):
            out.append(
                {
                    "run_id": run_id,
                    "workflow_id": data.get("workflow_id"),
                    "status": data.get("status", "unknown"),
                    "created_at": data.get("created_at"),
                    "total_time_ms": data.get("total_time_ms"),
                    "stages_completed": len(data.get("stages", {})),
                }
            )
        if out:
            return out
    persisted = db.list_workflow_runs(limit=_EVAL_RUNS_MAX)
    return [
        {
            "run_id": data.get("run_id"),
            "workflow_id": data.get("workflow_id"),
            "status": data.get("status", "unknown"),
            "created_at": data.get("started_at"),
            "total_time_ms": dig(data, "result", "total_time_ms", default=None),
            "stages_completed": len(data.get("stages", {})),
        }
        for data in persisted
    ]
