"""Shared API helper functions and singleton state.

General-purpose utilities used across multiple blueprint modules.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..shared_utils import safe_float as _to_safe_float
from ..notebook import LabNotebook
from ..runner import ExperimentRunner, RunConfig
from ..native_runner import native_runner_capability_report
from ..persona import get_aria

logger = logging.getLogger(__name__)

# ── Singleton runner ────────────────────────────────────────────────────
_runner: Optional[ExperimentRunner] = None


def get_runner(notebook_path: str) -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(notebook_path)
    return _runner


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

def json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable primitives for API/SSE payloads."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # Torch-like tensors
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        try:
            tensor_like = value.detach()
            if hasattr(tensor_like, "cpu") and callable(getattr(tensor_like, "cpu")):
                tensor_like = tensor_like.cpu()
            if hasattr(tensor_like, "tolist") and callable(getattr(tensor_like, "tolist")):
                return json_safe(tensor_like.tolist())
            if hasattr(tensor_like, "item") and callable(getattr(tensor_like, "item")):
                return json_safe(tensor_like.item())
            return str(tensor_like)
        except Exception:
            return str(value)
    # NumPy-like arrays/scalars
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


# ── Env helpers ─────────────────────────────────────────────────────────

def env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


# ── Native runner progress ──────────────────────────────────────────────

def with_native_runner_progress(progress_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(progress_payload or {})
    try:
        payload["native_runner"] = native_runner_capability_report()
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


def native_runner_canary_status_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    enabled = env_bool("NATIVE_RUNNER_CANARY_STATUS_ENABLED", False)
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    try:
        iterations = int(str(os.environ.get("NATIVE_RUNNER_CANARY_ITERATIONS", "8")))
    except Exception:
        iterations = 8
    iterations = max(1, min(iterations, 50))

    try:
        seed = int(str(os.environ.get("NATIVE_RUNNER_CANARY_SEED", "1337")))
    except Exception:
        seed = 1337

    try:
        ttl_seconds = float(str(os.environ.get("NATIVE_RUNNER_CANARY_TTL_S", "300")))
    except Exception:
        ttl_seconds = 300.0
    ttl_seconds = max(0.0, min(ttl_seconds, 3600.0))

    now = time.time()
    with _NATIVE_CANARY_LOCK:
        cache_updated = float(_NATIVE_CANARY_CACHE.get("updated_at") or 0.0)
        cached_payload = _NATIVE_CANARY_CACHE.get("payload")
        if (not force_refresh) and cached_payload is not None and (now - cache_updated) <= ttl_seconds:
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
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "iterations": int(result.iterations),
                "seed": int(result.seed),
                "probe_avg_latency_ms": float(result.probe_avg_latency_ms),
                "selective_avg_latency_ms": float(result.selective_avg_latency_ms),
                "latency_delta_ms": float(result.latency_delta_ms),
                "latency_ratio": float(result.latency_ratio),
                "probe_execution_paths": dict(result.probe_execution_paths),
                "selective_execution_paths": dict(result.selective_execution_paths),
                "selective_applied_layers_avg": float(result.selective_applied_layers_avg),
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
    s = re.sub(r'\d+\.\d+%?', '#', content)
    s = re.sub(r'\b\d{2,}\b', '#', s)
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


def get_run_trigger_snapshot(active_experiment_id: Optional[str] = None) -> Dict[str, Any]:
    active_id = str(active_experiment_id or "").strip() or None
    with _RUN_TRIGGER_LOCK:
        snap = dict(_LAST_RUN_TRIGGER)
    if active_id and snap.get("experiment_id") and snap.get("experiment_id") != active_id:
        return {
            "experiment_id": active_id,
            "source": "unknown",
            "mode": None,
            "timestamp": None,
            "details": {},
            "matched": False,
        }
    snap["experiment_id"] = active_id or snap.get("experiment_id")
    snap["matched"] = bool(active_id) and bool(snap.get("timestamp")) and snap.get("experiment_id") == active_id
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
        nb = LabNotebook(notebook_path)
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
        aria.configure_llm(
            backend_name=backend,
            api_key=str(data.get("api_key", "")).strip(),
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
