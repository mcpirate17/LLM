"""Shared API helper functions and singleton state.

General-purpose utilities used across multiple blueprint modules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..runner._types import LiveProgress
from ..runtime_events import (
    get_runtime_event_services,
    start_runtime_event_projector,
)
from .deps import get_notebook
from ..native.telemetry import native_runner_capability_report
from ..persona import get_aria
from ._utils import is_malformed_db_error

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..notebook import LabNotebook
    from ..runner import ExperimentRunner

# ── Singleton runner ────────────────────────────────────────────────────
_runner: Optional[ExperimentRunner] = None
_EXTERNAL_RUN_ACTIVITY_TIMEOUT_SECONDS = 3600.0


def projected_runtime_status_enabled() -> bool:
    raw = str(os.environ.get("ARIA_ENABLE_PROJECTED_RUNTIME_STATUS", "0")).strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def get_runner(
    notebook_path: str,
    *,
    create_if_missing: bool = True,
    start_projector: bool = False,
) -> Optional[ExperimentRunner]:
    global _runner
    if _runner is None and not create_if_missing:
        return None
    if _runner is None:
        from ..runner import ExperimentRunner

        _runner = ExperimentRunner(notebook_path)
    if start_projector:
        try:
            start_runtime_event_projector(notebook_path)
        except Exception:
            logger.warning(
                "Runtime event projector bootstrap failed for %s",
                notebook_path,
                exc_info=True,
            )
    return _runner


def reset_runner_launch_state(
    runner: ExperimentRunner, *, error: Optional[str] = None
) -> None:
    """Clear phantom in-memory launch state after a failed start attempt."""
    with runner._lock:
        if runner._thread is not None and not runner._thread.is_alive():
            runner._thread = None
        runner._progress = LiveProgress(
            status="failed" if error else "idle",
            error=error,
            aria_message=error or "",
        )
    try:
        runner._set_aria_cycle_phase(
            "idle",
            continuous_active=False,
            cycle_index=0,
            selected_mode=None,
            note="Launch failed; runner state reset.",
            emit_event=False,
        )
    except Exception:
        logger.debug("Failed to reset cycle phase after launch error", exc_info=True)


def _json_dict_or_empty(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_external_running_experiment_snapshot(
    nb: LabNotebook,
    *,
    activity_timeout_seconds: float = _EXTERNAL_RUN_ACTIVITY_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Infer an active external experiment from notebook state.

    This is used when the dashboard/API process did not launch the runner itself,
    for example when a separate ``python -m research --mode=synthesize`` process
    is writing to the same notebook database.
    """
    row = nb.conn.execute(
        """
        SELECT
            e.experiment_id,
            e.timestamp,
            e.started_at,
            e.experiment_type,
            e.status,
            e.hypothesis,
            e.config_json,
            e.n_programs_generated,
            e.n_stage0_passed,
            e.n_stage05_passed,
            e.n_stage1_passed,
            COALESCE((
                SELECT MAX(pr.timestamp)
                FROM program_results pr
                WHERE pr.experiment_id = e.experiment_id
            ), 0) AS last_program_ts,
            COALESCE((
                SELECT MAX(ml.timestamp)
                FROM metrics_log ml
                WHERE ml.experiment_id = e.experiment_id
            ), 0) AS last_metric_ts,
            COALESCE((
                SELECT MAX(en.timestamp)
                FROM entries en
                WHERE en.experiment_id = e.experiment_id
                  AND en.entry_type != 'hypothesis'
            ), 0) AS last_entry_ts,
            COALESCE((
                SELECT COUNT(*)
                FROM program_results pr
                WHERE pr.experiment_id = e.experiment_id
            ), 0) AS program_results_count
        FROM experiments e
        WHERE e.status = 'running'
        ORDER BY e.timestamp DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None

    config = _json_dict_or_empty(row["config_json"])
    total_programs = (
        config.get("n_programs")
        or config.get("num_programs")
        or row["n_programs_generated"]
        or 0
    )
    try:
        total_programs = int(total_programs or 0)
    except (TypeError, ValueError):
        total_programs = 0

    current_program = max(
        int(row["program_results_count"] or 0),
        int(row["n_programs_generated"] or 0),
    )
    last_activity_ts = max(
        float(row["started_at"] or 0.0),
        float(row["timestamp"] or 0.0),
        float(row["last_program_ts"] or 0.0),
        float(row["last_metric_ts"] or 0.0),
        float(row["last_entry_ts"] or 0.0),
    )
    age_seconds = max(0.0, time.time() - last_activity_ts)
    if activity_timeout_seconds > 0 and age_seconds > activity_timeout_seconds:
        return None

    mode = str(
        config.get("mode")
        or config.get("run_mode")
        or config.get("experiment_mode")
        or row["experiment_type"]
        or "external"
    )
    hypothesis = str(row["hypothesis"] or "").strip()
    return {
        "experiment_id": str(row["experiment_id"] or ""),
        "status": "running",
        "mode": mode,
        "hypothesis": hypothesis,
        "current_program": current_program,
        "total_programs": max(total_programs, current_program),
        "stage0_passed": int(row["n_stage0_passed"] or 0),
        "stage05_passed": int(row["n_stage05_passed"] or 0),
        "stage1_passed": int(row["n_stage1_passed"] or 0),
        "started_at": float(row["started_at"] or row["timestamp"] or 0.0),
        "last_activity_ts": last_activity_ts,
        "elapsed_seconds": max(
            0.0,
            time.time() - float(row["started_at"] or row["timestamp"] or time.time()),
        ),
        "source": "external_notebook_process",
    }


def resolve_runner_status(
    nb: LabNotebook, runner: Optional[ExperimentRunner]
) -> Dict[str, Any]:
    """Return dashboard-safe running/progress state for local or external runs."""
    if runner is not None and runner.is_running:
        return {
            "is_running": True,
            "progress": with_native_runner_progress(runner.progress.to_dict()),
            "external_snapshot": None,
        }

    lifecycle_snapshot = get_registry_running_experiment_snapshot(nb)
    if lifecycle_snapshot is not None:
        return {
            "is_running": True,
            "progress": with_native_runner_progress(
                _build_registry_progress(lifecycle_snapshot)
            ),
            "external_snapshot": lifecycle_snapshot,
        }

    try:
        if runner is not None and projected_runtime_status_enabled():
            projected = get_projected_running_experiment_snapshot(nb)
            if projected is not None:
                return {
                    "is_running": True,
                    "progress": with_native_runner_progress(
                        _build_external_progress(
                            projected,
                            current_stage="runtime_projector",
                            default_message_prefix="Projected",
                        )
                    ),
                    "external_snapshot": projected,
                }

        external = get_external_running_experiment_snapshot(nb)
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        logger.warning(
            "Runner status falling back to in-memory progress due to malformed DB: %s",
            exc,
        )
        progress = with_native_runner_progress(
            runner.progress.to_dict() if runner is not None else LiveProgress().to_dict()
        )
        progress["database_status"] = {
            "healthy": False,
            "error_type": "malformed",
            "message": str(exc),
        }
        return {
            "is_running": False,
            "progress": progress,
            "external_snapshot": None,
        }
    if not external:
        return {
            "is_running": False,
            "progress": with_native_runner_progress(
                runner.progress.to_dict() if runner is not None else LiveProgress().to_dict()
            ),
            "external_snapshot": None,
        }

    progress = {
        "experiment_id": external["experiment_id"],
        "status": "running",
        "current_program": external["current_program"],
        "total_programs": external["total_programs"],
        "stage0_passed": external["stage0_passed"],
        "stage05_passed": external["stage05_passed"],
        "stage1_passed": external["stage1_passed"],
        "novel_count": 0,
        "current_stage": "external_cli",
        "current_fingerprint": "",
        "best_loss_ratio": None,
        "best_novelty": None,
        "elapsed_seconds": external["elapsed_seconds"],
        "aria_message": (
            f"External {external['mode']} experiment detected via notebook"
            if not external["hypothesis"]
            else f"External {external['mode']} experiment: {external['hypothesis'][:160]}"
        ),
        "error": None,
        "estimated_cost": 0.0,
        "total_tokens": 0,
        "current_generation": 0,
        "total_generations": 0,
        "best_fitness": None,
        "avg_fitness": None,
        "archive_size": 0,
        "hypothesis_critique": None,
        "run_source": external["source"],
        "last_activity_ts": external["last_activity_ts"],
        "external_process": True,
    }
    return {
        "is_running": True,
        "progress": with_native_runner_progress(progress),
        "external_snapshot": external,
    }


def get_registry_running_experiment_snapshot(
    nb: LabNotebook,
) -> Optional[Dict[str, Any]]:
    services = get_runtime_event_services(nb.db_path)
    registry = services.registry
    registry.spool_unhealthy = bool(services.bus_health().last_spool_error)
    registry.projector_unhealthy = bool(services.projector_health().degraded)
    run_id = registry.active_run_id()
    if not run_id:
        return None
    state = registry.get(run_id)
    if state is None or state.status != "running":
        return None
    notebook_row = nb.conn.execute(
        "SELECT status, started_at, completed_at FROM experiments WHERE experiment_id = ?",
        (run_id,),
    ).fetchone()
    if notebook_row is not None:
        notebook_status = str(notebook_row["status"] or "").strip().lower()
        if notebook_status in {"completed", "failed", "cancelled", "canceled"}:
            logger.debug(
                "Ignoring stale registry-active run %s because notebook status is %s",
                run_id,
                notebook_status,
            )
            return None
    payload = dict(state.last_event.payload or {})
    config = _json_dict_or_empty(payload.get("config"))
    mode = str(
        config.get("mode")
        or config.get("run_mode")
        or config.get("experiment_mode")
        or payload.get("experiment_type")
        or "runtime_event"
    )
    return {
        "experiment_id": run_id,
        "status": state.status,
        "mode": mode,
        "hypothesis": str(payload.get("hypothesis") or "").strip(),
        "current_program": 0,
        "total_programs": 0,
        "stage0_passed": 0,
        "stage05_passed": 0,
        "stage1_passed": 0,
        "started_at": float(payload.get("started_at", state.last_event.created_at)),
        "last_activity_ts": float(state.last_event.created_at),
        "elapsed_seconds": max(0.0, time.time() - float(state.last_event.created_at)),
        "source": "runtime_lifecycle_registry",
    }


def get_projected_running_experiment_snapshot(
    nb: LabNotebook,
) -> Optional[Dict[str, Any]]:
    start_runtime_event_projector(nb.db_path)
    row = nb.conn.execute(
        """
        SELECT
            e.experiment_id,
            e.timestamp,
            e.started_at,
            e.experiment_type,
            e.status,
            e.hypothesis,
            e.config_json,
            e.n_programs_generated,
            e.n_stage0_passed,
            e.n_stage05_passed,
            e.n_stage1_passed,
            MAX(arev.applied_at) AS last_projected_at
        FROM experiments e
        JOIN applied_runtime_events arev
          ON arev.run_id = e.experiment_id
        WHERE e.status = 'running'
        GROUP BY
            e.experiment_id,
            e.timestamp,
            e.started_at,
            e.experiment_type,
            e.status,
            e.hypothesis,
            e.config_json,
            e.n_programs_generated,
            e.n_stage0_passed,
            e.n_stage05_passed,
            e.n_stage1_passed
        ORDER BY COALESCE(e.started_at, e.timestamp) DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None

    config = _json_dict_or_empty(row["config_json"])
    total_programs = (
        config.get("n_programs")
        or config.get("num_programs")
        or row["n_programs_generated"]
        or 0
    )
    try:
        total_programs = int(total_programs or 0)
    except (TypeError, ValueError):
        total_programs = 0

    current_program = int(row["n_programs_generated"] or 0)
    started_at = float(row["started_at"] or row["timestamp"] or time.time())
    last_projected_at = float(row["last_projected_at"] or started_at)
    return {
        "experiment_id": str(row["experiment_id"] or ""),
        "status": "running",
        "mode": str(
            config.get("mode")
            or config.get("run_mode")
            or config.get("experiment_mode")
            or row["experiment_type"]
            or "projected"
        ),
        "hypothesis": str(row["hypothesis"] or "").strip(),
        "current_program": current_program,
        "total_programs": max(total_programs, current_program),
        "stage0_passed": int(row["n_stage0_passed"] or 0),
        "stage05_passed": int(row["n_stage05_passed"] or 0),
        "stage1_passed": int(row["n_stage1_passed"] or 0),
        "started_at": started_at,
        "last_activity_ts": last_projected_at,
        "elapsed_seconds": max(0.0, time.time() - started_at),
        "source": "projected_runtime_lifecycle",
    }


def _build_registry_progress(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = str(snapshot.get("hypothesis") or "").strip()
    mode = str(snapshot.get("mode") or "runtime_event")
    return {
        "experiment_id": snapshot["experiment_id"],
        "status": "running",
        "current_program": 0,
        "total_programs": 0,
        "stage0_passed": 0,
        "stage05_passed": 0,
        "stage1_passed": 0,
        "novel_count": 0,
        "current_stage": "runtime_events",
        "current_fingerprint": "",
        "best_loss_ratio": None,
        "best_novelty": None,
        "elapsed_seconds": snapshot["elapsed_seconds"],
        "aria_message": (
            f"Runtime lifecycle registry reports active {mode} experiment"
            if not hypothesis
            else f"Runtime lifecycle registry: {hypothesis[:160]}"
        ),
        "error": None,
        "estimated_cost": 0.0,
        "total_tokens": 0,
        "current_generation": 0,
        "total_generations": 0,
        "best_fitness": None,
        "avg_fitness": None,
        "archive_size": 0,
        "hypothesis_critique": None,
        "run_source": snapshot["source"],
        "last_activity_ts": snapshot["last_activity_ts"],
        "external_process": True,
    }


def _build_external_progress(
    snapshot: Dict[str, Any],
    *,
    current_stage: str,
    default_message_prefix: str,
) -> Dict[str, Any]:
    mode = str(snapshot.get("mode") or "external")
    hypothesis = str(snapshot.get("hypothesis") or "").strip()
    source = str(snapshot.get("source") or "external")
    return {
        "experiment_id": snapshot["experiment_id"],
        "status": "running",
        "current_program": int(snapshot.get("current_program") or 0),
        "total_programs": int(snapshot.get("total_programs") or 0),
        "stage0_passed": int(snapshot.get("stage0_passed") or 0),
        "stage05_passed": int(snapshot.get("stage05_passed") or 0),
        "stage1_passed": int(snapshot.get("stage1_passed") or 0),
        "novel_count": 0,
        "current_stage": current_stage,
        "current_fingerprint": "",
        "best_loss_ratio": None,
        "best_novelty": None,
        "elapsed_seconds": snapshot["elapsed_seconds"],
        "aria_message": (
            f"{default_message_prefix} {mode} experiment detected via notebook"
            if not hypothesis
            else f"{default_message_prefix} {mode} experiment: {hypothesis[:160]}"
        ),
        "error": None,
        "estimated_cost": 0.0,
        "total_tokens": 0,
        "current_generation": 0,
        "total_generations": 0,
        "best_fitness": None,
        "avg_fitness": None,
        "archive_size": 0,
        "hypothesis_critique": None,
        "run_source": source,
        "last_activity_ts": snapshot["last_activity_ts"],
        "external_process": True,
    }


# ── SSE timeout ─────────────────────────────────────────────────────────


def get_sse_timeout_seconds() -> float:
    """Get SSE stream polling timeout from env with safe fallback."""
    raw = os.environ.get("ARIA_SSE_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    if timeout <= 0:
        logger.warning("Non-positive ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    return timeout


# ── JSON safety ─────────────────────────────────────────────────────────


# ── Env helpers ─────────────────────────────────────────────────────────


def env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


# ── Native runner progress ──────────────────────────────────────────────


def with_native_runner_progress(
    progress_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(progress_payload or {})
    try:
        payload["native_runner"] = native_runner_capability_report(deep=False)
    except Exception as exc:
        payload["native_runner"] = {
            "enabled": False,
            "strict": False,
            "designer_runtime_available": False,
            "status": f"native_runner_report_error:{exc}",
            "fallback_metrics": {
                "total_compiles": 0,
                "native_enabled_compiles": 0,
                "fallback_compiles": 0,
                "probe_successes": 0,
                "probe_failures": 0,
                "fallback_rate": 0.0,
                "samples_considered": 0,
                "all_compile_calls": 0,
            },
            "semantic_warning_count": 0,
            "semantic_warnings": [],
            "selective_guardrail": {
                "consecutive_requested_not_candidate": 0,
                "threshold": 5,
                "triggered": False,
                "trigger_count": 0,
                "last_reason": None,
                "history": [],
            },
        }
    return payload


# ── Native runner canary ────────────────────────────────────────────────
_NATIVE_CANARY_LOCK = threading.Lock()
_NATIVE_CANARY_CACHE: Dict[str, Any] = {
    "updated_at": 0.0,
    "payload": None,
}


def native_runner_canary_status_payload(
    *, force_refresh: bool = False
) -> Dict[str, Any]:
    enabled = env_bool("NATIVE_RUNNER_CANARY_STATUS_ENABLED", False)
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    try:
        iterations = int(str(os.environ.get("NATIVE_RUNNER_CANARY_ITERATIONS", "8")))
    except (TypeError, ValueError):
        iterations = 8
    iterations = max(1, min(iterations, 50))

    try:
        seed = int(str(os.environ.get("NATIVE_RUNNER_CANARY_SEED", "1337")))
    except (TypeError, ValueError):
        seed = 1337

    try:
        ttl_seconds = float(str(os.environ.get("NATIVE_RUNNER_CANARY_TTL_S", "300")))
    except (TypeError, ValueError):
        ttl_seconds = 300.0
    ttl_seconds = max(0.0, min(ttl_seconds, 3600.0))

    now = time.time()
    with _NATIVE_CANARY_LOCK:
        cache_updated = float(_NATIVE_CANARY_CACHE.get("updated_at") or 0.0)
        cached_payload = _NATIVE_CANARY_CACHE.get("payload")
        if (
            (not force_refresh)
            and cached_payload is not None
            and (now - cache_updated) <= ttl_seconds
        ):
            out = dict(cached_payload)
            out["cached"] = True
            out["age_s"] = round(max(0.0, now - cache_updated), 3)
            return out

        try:
            from ..native_runner_canary import run_selective_canary_latency_benchmark

            result = run_selective_canary_latency_benchmark(
                iterations=iterations,
                seed=seed,
            )
            payload = {
                "enabled": True,
                "status": "ok",
                "cached": False,
                "age_s": 0.0,
                "generated_at": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "iterations": int(result.iterations),
                "seed": int(result.seed),
                "probe_avg_latency_ms": float(result.probe_avg_latency_ms),
                "selective_avg_latency_ms": float(result.selective_avg_latency_ms),
                "latency_delta_ms": float(result.latency_delta_ms),
                "latency_ratio": float(result.latency_ratio),
                "probe_execution_paths": dict(result.probe_execution_paths),
                "selective_execution_paths": dict(result.selective_execution_paths),
                "selective_applied_layers_avg": float(
                    result.selective_applied_layers_avg
                ),
            }
        except Exception as exc:
            payload = {
                "enabled": True,
                "status": "error",
                "error": str(exc),
                "cached": False,
                "age_s": 0.0,
            }

        _NATIVE_CANARY_CACHE["updated_at"] = now
        _NATIVE_CANARY_CACHE["payload"] = payload
        return payload


# ── Insight deduplication ───────────────────────────────────────────────


def _insight_dedup_key(content: str) -> str:
    """Normalize numeric values to create a stable dedup key for insights."""
    s = re.sub(r"\d+\.\d+%?", "#", content)
    s = re.sub(r"\b\d{2,}\b", "#", s)
    return s


def deduplicate_insights(insights: list) -> list:
    """Keep only the most recent insight per semantic dedup key."""
    seen: dict = {}
    for ins in insights:
        semantic_key = str(ins.get("semantic_key") or "").strip()
        key = semantic_key or _insight_dedup_key(ins.get("content", ""))
        if key not in seen:
            seen[key] = ins
    return list(seen.values())


# ── Result ID normalization ─────────────────────────────────────────────


def normalize_result_ids(raw_ids: Any) -> List[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: List[str] = []
    seen: set = set()
    for value in raw_ids:
        if value is None:
            continue
        result_id = str(value).strip()
        if not result_id or result_id in seen:
            continue
        seen.add(result_id)
        normalized.append(result_id)
    return normalized


# ── Run trigger tracking ────────────────────────────────────────────────
_RUN_TRIGGER_LOCK = threading.Lock()
_LAST_RUN_TRIGGER: Dict[str, Any] = {
    "experiment_id": None,
    "source": "unknown",
    "mode": None,
    "timestamp": None,
    "details": {},
}


def record_run_trigger(
    experiment_id: str,
    source: str,
    mode: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "experiment_id": str(experiment_id or "").strip() or None,
        "source": str(source or "unknown").strip() or "unknown",
        "mode": (str(mode).strip() if mode else None),
        "timestamp": time.time(),
        "details": details if isinstance(details, dict) else {},
    }
    payload["details"] = {
        key: value for key, value in payload["details"].items() if value is not None
    }
    with _RUN_TRIGGER_LOCK:
        _LAST_RUN_TRIGGER.update(payload)
        return dict(_LAST_RUN_TRIGGER)


def get_run_trigger_snapshot(
    active_experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    active_id = str(active_experiment_id or "").strip() or None
    with _RUN_TRIGGER_LOCK:
        snap = dict(_LAST_RUN_TRIGGER)
    if (
        active_id
        and snap.get("experiment_id")
        and snap.get("experiment_id") != active_id
    ):
        return {
            "experiment_id": active_id,
            "source": "unknown",
            "mode": None,
            "timestamp": None,
            "details": {},
            "matched": False,
        }
    snap["experiment_id"] = active_id or snap.get("experiment_id")
    snap["matched"] = (
        bool(active_id)
        and bool(snap.get("timestamp"))
        and snap.get("experiment_id") == active_id
    )
    return snap


# ── Code agent task tracking ───────────────────────────────────────────
_CODE_AGENT_TASKS: Dict[str, Dict[str, Any]] = {}
_CODE_AGENT_TASKS_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX: Dict[str, Dict[str, Any]] = {}
_WORKSPACE_FILE_INDEX_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX_BUILT_AT: float = 0.0

# ── Chat guardrails ────────────────────────────────────────────────────
_CHAT_GUARDRAIL_LOCK = threading.Lock()
_CHAT_GUARDRAIL_EVENTS = deque(maxlen=500)
_ALLOWED_CHAT_ACTION_TYPES = {
    "adjust_config",
    "adjust_grammar",
    "start_experiment",
    "edit_file",
    "spawn_agent",
    "maintain_database",
}

# ── Dismissed actions ──────────────────────────────────────────────────
_DISMISSED_ACTIONS: set = set()

# ── Batch rerun state ─────────────────────────────────────────────────
_BATCH_RERUN_STATE: Dict[str, Any] = {
    "active": False,
    "total": 0,
    "completed": 0,
    "current": None,
    "remaining": [],
    "results": [],
}

# ── Autonomy engine singletons ─────────────────────────────────────────
_aria_autonomy = None
_aria_action_store = None


def get_autonomy(notebook_path: str):
    """Get or create the singleton AriaAutonomy instance."""
    global _aria_autonomy, _aria_action_store
    if _aria_autonomy is None:
        from ..autonomy import AriaAutonomy
        from ..actions import ActionStore

        nb = get_notebook(notebook_path, read_only=False)
        _aria_autonomy = AriaAutonomy(notebook=nb)
        _aria_action_store = ActionStore(nb.conn)
    return _aria_autonomy, _aria_action_store


# ── LLM config persistence ─────────────────────────────────────────────


def llm_config_path(notebook_path: str) -> Path:
    """Path for persisted LLM configuration, next to the notebook DB."""
    return Path(notebook_path).parent / "llm_config.json"


def load_persisted_llm_config(notebook_path: str):
    """Auto-load LLM config from disk if present."""
    config_path = llm_config_path(notebook_path)
    if not config_path.exists():
        return
    try:
        import json as _json

        data = _json.loads(config_path.read_text())
        backend = str(data.get("backend", "")).strip()
        if not backend:
            return
        aria = get_aria()
        api_key_env = str(data.get("api_key_env", "")).strip()
        api_key = (
            os.environ.get(api_key_env, "")
            if api_key_env
            else str(data.get("api_key", "")).strip()
        )
        aria.configure_llm(
            backend_name=backend,
            api_key=api_key,
            model=str(data.get("model", "")).strip(),
            host=str(data.get("host", "")).strip(),
        )
        logger.info(f"Loaded persisted LLM config: {backend}")
    except Exception as e:
        logger.warning(f"Failed to load persisted LLM config: {e}")


def save_llm_config(notebook_path: str, config: Dict):
    """Persist LLM config to disk so it survives restarts."""
    config_path = llm_config_path(notebook_path)
    try:
        import json as _json

        config_path.write_text(_json.dumps(config, indent=2))
        logger.info(f"Saved LLM config to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save LLM config: {e}")
