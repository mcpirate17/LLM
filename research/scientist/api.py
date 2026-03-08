"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import ast
import json
import csv
import io
import importlib.util
from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
import math
import os
import re
import struct
import threading
import time
import traceback
import uuid
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from .shared_utils import safe_float as _to_safe_float

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from .notebook import LabNotebook
from .evidence import build_evidence_pack
from .persona import get_aria
from .runner import ExperimentRunner, RunConfig
from .native_runner import native_runner_capability_report
from .llm.context import build_program_context
from .designer_utils import compile_designer_graph, validate_designer_graph, run_designer_graph, get_designer_components, generate_python_module

import requests as _requests

logger = logging.getLogger(__name__)

# ── Designer proxy configuration ────────────────────────────────────────
# When set, /api/designer/* endpoints proxy to the aria_designer API first,
# falling back to local implementation if the proxy is unavailable.
_DESIGNER_PROXY_BASE = os.environ.get("ARIA_DESIGNER_PROXY_BASE", "http://127.0.0.1:8091")
_DESIGNER_PROXY_ENABLED = os.environ.get("ARIA_DESIGNER_PROXY_ENABLED", "1") != "0"
_DESIGNER_PROXY_TIMEOUT = float(os.environ.get("ARIA_DESIGNER_PROXY_TIMEOUT", "10"))

# ── Designer lifecycle orchestration ────────────────────────────────────
_ARIA_DESIGNER_ROOT = Path(
    os.environ.get(
        "ARIA_DESIGNER_ROOT",
        str(Path(__file__).resolve().parents[2] / "aria_designer"),
    )
)
_ARIA_DESIGNER_API_HEALTH = os.environ.get("ARIA_DESIGNER_API_HEALTH", "http://127.0.0.1:8091/health")
_ARIA_DESIGNER_UI_HEALTH = os.environ.get("ARIA_DESIGNER_UI_HEALTH", "http://127.0.0.1:5174")
_ARIA_DESIGNER_BOOT_TIMEOUT_S = float(os.environ.get("ARIA_DESIGNER_BOOT_TIMEOUT_S", "30"))
_ARIA_DESIGNER_IDLE_TIMEOUT_S = float(os.environ.get("ARIA_DESIGNER_IDLE_TIMEOUT_S", "900"))
_DESIGNER_LIFECYCLE_LOCK = threading.Lock()
_DESIGNER_ACTIVITY_LOCK = threading.Lock()
_DESIGNER_LAST_ACTIVITY_TS = time.time()
_DESIGNER_LAST_ACTIVITY_REASON = "startup"
_DESIGNER_LAST_AUTOSTOP_TS: float | None = None
_DESIGNER_IDLE_WATCHDOG_STARTED = False
_NATIVE_CANARY_LOCK = threading.Lock()
_NATIVE_CANARY_CACHE: Dict[str, Any] = {
    "updated_at": 0.0,
    "payload": None,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _native_runner_canary_status_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    enabled = _env_bool("NATIVE_RUNNER_CANARY_STATUS_ENABLED", False)
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
        }

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
            from .native_runner_canary import run_selective_canary_latency_benchmark

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


def _designer_service_status() -> Dict[str, Any]:
    """Probe aria_designer API/UI health."""
    api_up = False
    ui_up = False
    try:
        r = _requests.get(_ARIA_DESIGNER_API_HEALTH, timeout=1.0)
        api_up = r.status_code < 500
    except Exception:
        api_up = False
    try:
        r = _requests.get(_ARIA_DESIGNER_UI_HEALTH, timeout=1.0)
        ui_up = r.status_code < 500
    except Exception:
        ui_up = False
    return {
        "api_up": api_up,
        "ui_up": ui_up,
        "running": bool(api_up and ui_up),
        "api_health_url": _ARIA_DESIGNER_API_HEALTH,
        "ui_health_url": _ARIA_DESIGNER_UI_HEALTH,
    }


def _designer_touch_activity(reason: str = "activity") -> Dict[str, Any]:
    now = time.time()
    with _DESIGNER_ACTIVITY_LOCK:
        global _DESIGNER_LAST_ACTIVITY_TS, _DESIGNER_LAST_ACTIVITY_REASON
        _DESIGNER_LAST_ACTIVITY_TS = now
        _DESIGNER_LAST_ACTIVITY_REASON = str(reason or "activity")
        idle_timeout = max(0.0, _ARIA_DESIGNER_IDLE_TIMEOUT_S)
        return {
            "activity_at": now,
            "activity_reason": _DESIGNER_LAST_ACTIVITY_REASON,
            "idle_timeout_s": idle_timeout,
            "auto_stop_enabled": idle_timeout > 0.0,
        }


def _designer_idle_state() -> Dict[str, Any]:
    now = time.time()
    with _DESIGNER_ACTIVITY_LOCK:
        idle_s = max(0.0, now - _DESIGNER_LAST_ACTIVITY_TS)
        idle_timeout = max(0.0, _ARIA_DESIGNER_IDLE_TIMEOUT_S)
        remaining = max(0.0, idle_timeout - idle_s) if idle_timeout > 0.0 else None
        return {
            "activity_at": _DESIGNER_LAST_ACTIVITY_TS,
            "activity_reason": _DESIGNER_LAST_ACTIVITY_REASON,
            "idle_for_s": idle_s,
            "idle_timeout_s": idle_timeout,
            "auto_stop_enabled": idle_timeout > 0.0,
            "auto_stop_in_s": remaining,
            "last_auto_stop_at": _DESIGNER_LAST_AUTOSTOP_TS,
        }


def _ensure_designer_idle_watchdog() -> None:
    global _DESIGNER_IDLE_WATCHDOG_STARTED
    with _DESIGNER_ACTIVITY_LOCK:
        if _DESIGNER_IDLE_WATCHDOG_STARTED:
            return
        _DESIGNER_IDLE_WATCHDOG_STARTED = True

    def _loop() -> None:
        global _DESIGNER_LAST_AUTOSTOP_TS
        while True:
            try:
                idle_timeout = max(0.0, _ARIA_DESIGNER_IDLE_TIMEOUT_S)
                if idle_timeout <= 0.0:
                    time.sleep(15.0)
                    continue
                with _DESIGNER_ACTIVITY_LOCK:
                    idle_for = time.time() - _DESIGNER_LAST_ACTIVITY_TS
                if idle_for >= idle_timeout:
                    status = _designer_service_status()
                    if status.get("running"):
                        result = _stop_designer_services()
                        if result.get("ok"):
                            with _DESIGNER_ACTIVITY_LOCK:
                                _DESIGNER_LAST_AUTOSTOP_TS = time.time()
                time.sleep(5.0)
            except Exception:
                logger.exception("Designer idle auto-stop watchdog failed; retrying")
                time.sleep(5.0)

    thread = threading.Thread(
        target=_loop,
        name="designer-idle-watchdog",
        daemon=True,
    )
    thread.start()


def _designer_dev_up_path() -> Path:
    return _ARIA_DESIGNER_ROOT / "tools" / "dev_up.sh"


def _designer_dev_down_path() -> Path:
    return _ARIA_DESIGNER_ROOT / "tools" / "dev_down.sh"


def _start_designer_services(force_restart: bool = False) -> Dict[str, Any]:
    """Best-effort start of aria_designer FE/BE via shared scripts."""
    if not _ARIA_DESIGNER_ROOT.exists():
        return {
            "ok": False,
            "error": f"ARIA_DESIGNER_ROOT not found: {_ARIA_DESIGNER_ROOT}",
        }
    dev_up = _designer_dev_up_path()
    dev_down = _designer_dev_down_path()
    if not dev_up.exists() or not dev_down.exists():
        return {
            "ok": False,
            "error": f"Missing lifecycle scripts under {_ARIA_DESIGNER_ROOT / 'tools'}",
        }

    with _DESIGNER_LIFECYCLE_LOCK:
        status0 = _designer_service_status()
        if status0["running"] and not force_restart:
            return {"ok": True, "already_running": True, "status": status0}

        if force_restart or status0["api_up"] or status0["ui_up"]:
            try:
                subprocess.run(
                    [str(dev_down)],
                    cwd=str(_ARIA_DESIGNER_ROOT),
                    check=False,
                    timeout=20,
                )
            except Exception:
                pass

        log_path = _ARIA_DESIGNER_ROOT / ".run" / "research_designer_boot.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")
        proc = subprocess.Popen(
            [str(dev_up)],
            cwd=str(_ARIA_DESIGNER_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_fh.close()

        deadline = time.time() + max(5.0, _ARIA_DESIGNER_BOOT_TIMEOUT_S)
        latest = _designer_service_status()
        while time.time() < deadline:
            latest = _designer_service_status()
            if latest["running"]:
                return {
                    "ok": True,
                    "already_running": False,
                    "pid": proc.pid,
                    "status": latest,
                    "log_path": str(log_path),
                }
            time.sleep(0.5)

        return {
            "ok": False,
            "error": "Timed out while starting aria_designer services.",
            "pid": proc.pid,
            "status": latest,
            "log_path": str(log_path),
        }


def _stop_designer_services() -> Dict[str, Any]:
    """Best-effort stop of aria_designer FE/BE via shared scripts."""
    dev_down = _designer_dev_down_path()
    if not dev_down.exists():
        return {"ok": False, "error": f"Missing stop script: {dev_down}"}
    with _DESIGNER_LIFECYCLE_LOCK:
        status_before = _designer_service_status()
        try:
            subprocess.run(
                [str(dev_down)],
                cwd=str(_ARIA_DESIGNER_ROOT),
                check=False,
                timeout=25,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "status_before": status_before}
        deadline = time.time() + 10.0
        latest = _designer_service_status()
        while time.time() < deadline:
            latest = _designer_service_status()
            if not latest["api_up"] and not latest["ui_up"]:
                return {"ok": True, "status_before": status_before, "status_after": latest}
            time.sleep(0.3)
        return {"ok": True, "status_before": status_before, "status_after": latest}


def _designer_proxy(method: str, path: str, *, json_body=None, params=None,
                     timeout: float | None = None) -> _requests.Response | None:
    """Try to proxy a request to the aria_designer API.

    Returns the Response on success, or None if proxy is disabled/unavailable
    (caller should fall back to legacy implementation).
    """
    if not _DESIGNER_PROXY_ENABLED:
        return None
    url = f"{_DESIGNER_PROXY_BASE.rstrip('/')}{path}"
    _timeout = timeout or _DESIGNER_PROXY_TIMEOUT
    try:
        resp = _requests.request(
            method, url, json=json_body, params=params, timeout=_timeout,
        )
        return resp
    except _requests.ConnectionError:
        logger.debug("Designer proxy unavailable at %s", _DESIGNER_PROXY_BASE)
        return None
    except _requests.Timeout:
        logger.warning("Designer proxy timeout after %.1fs for %s %s", _timeout, method, path)
        return None
    except Exception:
        logger.exception("Designer proxy unexpected error for %s %s", method, path)
        return None


def _proxy_or_error(resp: _requests.Response | None):
    """Convert a proxy response to a Flask Response object, or return None
    if no proxy response is available (caller falls back to legacy)."""
    if resp is None:
        return None
    try:
        body = resp.json()
    except Exception:
        body = {"error": resp.text or "Proxy returned non-JSON response"}
    
    from flask import make_response
    return make_response(jsonify(body), resp.status_code)


def _proxy_stream(method: str, path: str, *, json_body=None, params=None):
    """Stream-proxy an SSE endpoint from the aria_designer backend.

    Unlike _designer_proxy which buffers the entire response, this streams
    chunks through to the browser so SSE events arrive incrementally.
    """
    if not _DESIGNER_PROXY_ENABLED:
        return jsonify({"error": "Designer proxy not enabled"}), 502
    url = f"{_DESIGNER_PROXY_BASE.rstrip('/')}{path}"
    try:
        upstream = _requests.request(
            method, url, json=json_body, params=params,
            stream=True, timeout=120,
        )

        def generate():
            try:
                for chunk in upstream.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        content_type = upstream.headers.get("content-type", "text/event-stream")
        return Response(generate(), status=upstream.status_code,
                        content_type=content_type)
    except _requests.ConnectionError:
        return jsonify({"error": "Designer backend unavailable"}), 502
    except _requests.Timeout:
        return jsonify({"error": "Designer backend timeout"}), 504
    except Exception as e:
        logger.exception("Stream proxy error for %s %s", method, path)
        return jsonify({"error": str(e)}), 502


# Singleton runner shared across requests
_runner: Optional[ExperimentRunner] = None
_CODE_AGENT_TASKS: Dict[str, Dict[str, Any]] = {}
_CODE_AGENT_TASKS_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX: Dict[str, Dict[str, Any]] = {}
_WORKSPACE_FILE_INDEX_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX_BUILT_AT: float = 0.0
_RUN_TRIGGER_LOCK = threading.Lock()
_CHAT_GUARDRAIL_LOCK = threading.Lock()
_CHAT_GUARDRAIL_EVENTS = deque(maxlen=500)
_LAST_RUN_TRIGGER: Dict[str, Any] = {
    "experiment_id": None,
    "source": "unknown",
    "mode": None,
    "timestamp": None,
    "details": {},
}

_ALLOWED_CHAT_ACTION_TYPES = {
    "adjust_config",
    "adjust_grammar",
    "start_experiment",
    "edit_file",
    "spawn_agent",
    "maintain_database",
}


def _record_run_trigger(
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


def _get_run_trigger_snapshot(active_experiment_id: Optional[str] = None) -> Dict[str, Any]:
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


def _insight_dedup_key(content: str) -> str:
    """Normalize numeric values to create a stable dedup key for insights.

    Replaces decimals/percentages and multi-digit integers so that
    'appears in 144 survivors' matches 'appears in 145 survivors'.
    Preserves single-digit suffixes in op names like 'split2'.
    """
    import re
    s = re.sub(r'\d+\.\d+%?', '#', content)   # decimals / pcts
    s = re.sub(r'\b\d{2,}\b', '#', s)           # multi-digit ints
    return s


def _deduplicate_insights(insights: list) -> list:
    """Keep only the most recent insight per semantic dedup key."""
    seen: dict = {}
    for ins in insights:
        semantic_key = str(ins.get("semantic_key") or "").strip()
        key = semantic_key or _insight_dedup_key(ins.get("content", ""))
        if key not in seen:
            seen[key] = ins
    return list(seen.values())


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable primitives for API/SSE payloads."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # Torch-like tensors
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        try:
            tensor_like = value.detach()
            if hasattr(tensor_like, "cpu") and callable(getattr(tensor_like, "cpu")):
                tensor_like = tensor_like.cpu()
            if hasattr(tensor_like, "tolist") and callable(getattr(tensor_like, "tolist")):
                return _json_safe(tensor_like.tolist())
            if hasattr(tensor_like, "item") and callable(getattr(tensor_like, "item")):
                return _json_safe(tensor_like.item())
            return str(tensor_like)
        except Exception:
            return str(value)

    # NumPy-like arrays/scalars
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    return str(value)


def _with_native_runner_progress(progress_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_report_date(value: Optional[str], end_of_day: bool = False) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.timestamp()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _report_program_matches_theme(program: Dict[str, Any], theme: str) -> bool:
    normalized = str(theme or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    graph_json = str(program.get("graph_json") or "").lower()
    arch_spec = str(program.get("arch_spec_json") or "").lower()
    pruning_method = str(program.get("pruning_method") or "").lower()
    if normalized == "sparsity":
        return (
            program.get("sparse_density_mean") is not None
            or "sparse" in graph_json
            or "sparse" in arch_spec
            or bool(pruning_method)
        )
    if normalized == "compression":
        compression_markers = (
            "low_rank", "shared_basis", "tied_proj", "grouped_linear", "bottleneck", "quant", "compressed"
        )
        return any(marker in graph_json or marker in arch_spec for marker in compression_markers)
    if normalized == "routing":
        return (
            program.get("routing_confidence_mean") is not None
            or "routing" in graph_json
            or "moe" in graph_json
            or "gate" in graph_json
        )
    if normalized == "mathspace":
        return (
            bool(program.get("graph_uses_math_spaces"))
            or "mathspace" in graph_json
            or "clifford" in graph_json
            or "hyperbolic" in graph_json
            or "padic" in graph_json
            or "tropical" in graph_json
        )
    if normalized == "failure_modes":
        return (program.get("stage1_passed") or 0) == 0 or bool(program.get("error_type"))
    return True


def _experiment_s1_rate(exp: Dict[str, Any]) -> Optional[float]:
    generated = exp.get("n_programs_generated")
    if generated is None:
        generated = exp.get("n_programs")
    passed = exp.get("n_stage1_passed")
    if passed is None:
        passed = exp.get("s1_passed")
    try:
        gen = float(generated or 0)
        s1 = float(passed or 0)
    except Exception:
        return None
    if gen <= 0:
        return None
    return s1 / gen


def _report_experiment_matches_trend(exp: Dict[str, Any], trend: str) -> bool:
    normalized = str(trend or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    rate = _experiment_s1_rate(exp)
    novelty = exp.get("best_novelty_score")
    if normalized == "high_novelty":
        return isinstance(novelty, (int, float)) and float(novelty) >= 0.5
    if rate is None:
        return False
    if normalized in {"improving", "high_survival"}:
        return rate >= 0.08
    if normalized == "declining":
        return rate < 0.03
    if normalized == "plateaued":
        return 0.03 <= rate < 0.08
    return True


def _build_filtered_report_summary(
    base_summary: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not experiments:
        return dict(base_summary or {})
    total_programs = 0
    total_survivors = 0
    for exp in experiments:
        total_programs += int(exp.get("n_programs_generated") or 0)
        total_survivors += int(exp.get("n_stage1_passed") or 0)
    out = dict(base_summary or {})
    out["total_experiments"] = len(experiments)
    out["total_programs_evaluated"] = total_programs
    out["stage1_survivors"] = total_survivors
    return out


def _build_report_snapshot_key(scope: str, query_payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        {"scope": scope, "query": query_payload or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _infer_tier_for_program(nb: LabNotebook, program: dict) -> str:
    """Infer tier for a raw program_results row by checking the leaderboard."""
    result_id = program.get("result_id")
    if not result_id:
        return "screening"
    row = nb.conn.execute(
        "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
    ).fetchone()
    return row["tier"] if row else "screening"


def _count_discovery_tiers(nb: LabNotebook) -> dict:
    """Count unique fingerprints per tier + total S1 survivors."""
    rows = nb.conn.execute(
        "SELECT tier, COUNT(*) AS cnt FROM leaderboard GROUP BY tier"
    ).fetchall()
    counts = {r["tier"]: r["cnt"] for r in rows}
    total_s1 = nb.conn.execute(
        "SELECT COUNT(*) AS cnt FROM program_results WHERE stage1_passed = 1"
    ).fetchone()
    counts["total_survivors"] = total_s1["cnt"] if total_s1 else 0
    return counts


def _compute_cross_run_stability(nb: LabNotebook, top_programs: list) -> dict:
    """Compute rank movement for top candidates across recent experiments.

    Uses graph fingerprint as the architecture key and tracks its rank
    among stage-1-passing programs for each completed experiment.
    """
    experiments = [
        exp for exp in nb.get_recent_experiments(40)
        if exp.get("status") == "completed"
    ]
    if not top_programs or not experiments:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": len(experiments),
        }

    fingerprint_ranks_by_experiment: dict[str, dict[str, int]] = {}
    for exp in experiments:
        experiment_id = exp.get("experiment_id")
        if not experiment_id:
            continue
        programs = nb.get_program_results(experiment_id)
        ranked = sorted(
            [
                p for p in programs
                if p.get("stage1_passed") and p.get("loss_ratio") is not None
            ],
            key=lambda p: p.get("loss_ratio", float("inf")),
        )
        ranks = {}
        for idx, program in enumerate(ranked, start=1):
            fp = program.get("graph_fingerprint")
            if fp and fp not in ranks:
                ranks[fp] = idx
        fingerprint_ranks_by_experiment[experiment_id] = ranks

    candidates = []
    summary = {"stable": 0, "up": 0, "down": 0, "new": 0}
    for index, program in enumerate(top_programs[:20], start=1):
        fp = program.get("graph_fingerprint")
        if not fp:
            continue

        history = []
        for exp in experiments:
            experiment_id = exp.get("experiment_id")
            if not experiment_id:
                continue
            rank = fingerprint_ranks_by_experiment.get(experiment_id, {}).get(fp)
            if rank is None:
                continue
            history.append({
                "experiment_id": experiment_id,
                "timestamp": exp.get("timestamp"),
                "rank": rank,
            })

        seen_runs = len(history)
        latest_rank = history[0]["rank"] if history else None
        previous_rank = history[1]["rank"] if len(history) > 1 else None
        delta = None
        if latest_rank is not None and previous_rank is not None:
            delta = latest_rank - previous_rank
        trend = _rank_label(delta, seen_runs)
        summary[trend] = summary.get(trend, 0) + 1

        candidates.append({
            "result_id": program.get("result_id"),
            "graph_fingerprint": fp,
            "current_overall_rank": index,
            "seen_runs": seen_runs,
            "latest_rank": latest_rank,
            "previous_rank": previous_rank,
            "rank_delta": delta,
            "trend": trend,
        })

    return {
        "summary": summary,
        "candidates": candidates,
        "window_size": len(experiments),
    }


def _get_sse_timeout_seconds() -> float:
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


def _get_runner(notebook_path: str) -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(notebook_path)
    return _runner


def _compute_recommendation(program: dict, leaderboard_entry: Optional[dict]) -> dict:
    """Deterministic next-action recommendation based on tier and pass/fail."""
    tier = (leaderboard_entry or {}).get("tier", "screening")
    s1 = program.get("stage1_passed", False)

    if not s1:
        return {
            "action": "archive",
            "rationale": "Program did not pass Stage 1 learning evaluation.",
            "confidence": "high",
        }

    if tier == "breakthrough":
        return {
            "action": "publish",
            "rationale": "Breakthrough-tier architecture with validated performance.",
            "confidence": "high",
        }

    if tier == "validation":
        passed = (leaderboard_entry or {}).get("validation_passed", False)
        if passed:
            return {
                "action": "scale up or publish",
                "rationale": "Validation passed with multi-seed stability confirmed.",
                "confidence": "high",
                "bias_check": "grammar_independence_verified",
            }
        return {
            "action": "re-validate",
            "rationale": "Validation tier but not yet passed; may need more seeds or longer training.",
            "confidence": "medium",
        }

    if tier == "investigation":
        passed = (leaderboard_entry or {}).get("investigation_passed", False)
        if passed:
            return {
                "action": "validate",
                "rationale": "Investigation passed; promote to validation for multi-seed confirmation.",
                "confidence": "high",
            }
        return {
            "action": "re-investigate or archive",
            "rationale": "Investigation tier but not yet passed; re-run or archive if stale.",
            "confidence": "medium",
        }

    # screening (default)
    return {
        "action": "investigate",
        "rationale": "Screening-tier candidate; needs deeper investigation to confirm potential.",
        "confidence": "medium",
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    
    # Handle stringified bytes if they look like "b'...'"
    if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass

    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 4:
                value = struct.unpack("<f", value)[0]
            elif len(value) == 8:
                value = struct.unpack("<d", value)[0]
            else:
                value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _promotion_evidence_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    seen_runs = int(((entry.get("cross_run_stability") or {}).get("seen_runs") or 0))
    baseline_ratio = _safe_float(entry.get("validation_baseline_ratio"))
    std = _safe_float(entry.get("validation_multi_seed_std"))

    checks = {
        "baselineEvidence": baseline_ratio is not None,
        "baselineBeat": baseline_ratio is not None and baseline_ratio < 1.0,
        "multiSeedStd": std is not None,
        "boundedStd": std is not None and std <= 0.12,
        "ckaArtifactBacked": entry.get("cka_source") == "artifact",
        "repeatObserved": seen_runs >= 3,
    }
    evidence_count = sum(1 for ok in checks.values() if ok)
    total_checks = len(checks)
    completeness = evidence_count / total_checks if total_checks else 0.0

    std_signal = 0.0
    if std is not None:
        if std <= 0.05:
            std_signal = 1.0
        elif std <= 0.12:
            std_signal = 0.65
        elif std <= 0.2:
            std_signal = 0.35
        else:
            std_signal = 0.1

    if seen_runs >= 5:
        repeat_signal = 1.0
    elif seen_runs >= 3:
        repeat_signal = 0.65
    elif seen_runs >= 2:
        repeat_signal = 0.4
    elif seen_runs >= 1:
        repeat_signal = 0.2
    else:
        repeat_signal = 0.0

    margin_signal = 0.0
    if baseline_ratio is not None:
        margin = 1.0 - baseline_ratio
        if margin >= 0.1:
            margin_signal = 1.0
        elif margin > 0:
            margin_signal = 0.7
        else:
            margin_signal = 0.15

    score = round((completeness * 0.5 + std_signal * 0.2 + repeat_signal * 0.2 + margin_signal * 0.1) * 100)
    missing = [name for name, ok in checks.items() if not ok]

    return {
        "score": score,
        "seen_runs": seen_runs,
        "std": std,
        "evidence_count": evidence_count,
        "total_checks": total_checks,
        "missing": missing,
    }


def _decision_gate_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    investigation_robustness = _safe_float(entry.get("investigation_robustness"))
    validation_baseline_ratio = _safe_float(entry.get("validation_baseline_ratio"))
    validation_multi_seed_std = _safe_float(entry.get("validation_multi_seed_std"))

    checks = {
        "screeningEvidence": entry.get("screening_loss_ratio") is not None and entry.get("screening_novelty") is not None,
        "investigationEvidence": entry.get("investigation_loss_ratio") is not None and entry.get("investigation_robustness") is not None,
        "robustnessFloor": investigation_robustness is not None and investigation_robustness >= 0.5,
        "validationEvidence": (
            entry.get("validation_loss_ratio") is not None
            and entry.get("validation_baseline_ratio") is not None
            and entry.get("validation_multi_seed_std") is not None
        ),
        "baselineBeatsReference": validation_baseline_ratio is not None and validation_baseline_ratio < 1.0,
        "consistencyBounded": validation_multi_seed_std is not None and validation_multi_seed_std <= 0.12,
    }
    decision_ready = all(checks.values())
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "decision_ready": decision_ready,
        "missing": missing,
    }


def _build_scale_up_templates_for_result(result_id: Optional[str]) -> List[Dict[str, Any]]:
    normalized = str(result_id or "").strip()
    if not normalized:
        return []

    return [
        {
            "template_id": "multi_seed_stress",
            "title": "Multi-seed stress validation",
            "description": "Run deeper multi-seed validation to confirm consistency and variance bounds.",
            "start_payload": {
                "mode": "validation",
                "result_ids": [normalized],
                "validation_steps": 12000,
                "validation_n_seeds": 7,
                "validation_batch_size": 8,
                "validation_seq_len": 512,
            },
        },
        {
            "template_id": "robustness_recheck",
            "title": "Robustness re-check",
            "description": "Re-run investigation-level robustness checks before heavier scale-up spend.",
            "start_payload": {
                "mode": "investigation",
                "result_ids": [normalized],
                "investigation_steps": 3500,
                "investigation_batch_size": 4,
                "n_training_programs": 4,
            },
        },
        {
            "template_id": "efficiency_scale_up",
            "title": "Scale-up + efficiency profile",
            "description": "Run scale-up training with one-shot pruning baseline to profile efficiency/quality trade-offs.",
            "start_payload": {
                "mode": "scale_up",
                "result_ids": [normalized],
                "scale_up_steps": 8000,
                "scale_up_batch_size": 8,
                "scale_up_seq_len": 512,
                "one_shot_pruning_baseline": True,
                "one_shot_pruning_method": "wanda",
                "one_shot_pruning_sparsity": 0.5,
            },
        },
    ]


def _build_reproducibility_workflow(
    repro_packet: Dict[str, Any],
    scale_up_templates: List[Dict[str, Any]],
    result_id: Optional[str] = None,
    graph_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    packet = repro_packet or {}
    ready_count = int(packet.get("ready_count") or 0)
    total_checks = int(packet.get("total_checks") or 6)
    missing = {str(item) for item in (packet.get("missing") or []) if item}

    template_by_id = {
        str(template.get("template_id")): template
        for template in (scale_up_templates or [])
        if isinstance(template, dict) and template.get("template_id")
    }

    def _payload(template_id: str) -> Optional[Dict[str, Any]]:
        template = template_by_id.get(template_id)
        if not template:
            return None
        payload = template.get("start_payload")
        return dict(payload) if isinstance(payload, dict) else None

    checks = [
        ("result_id", "Result identifier captured", None, "Program result must be persisted before repro closure."),
        ("graph_fingerprint", "Fingerprint captured", None, "Graph fingerprint is required for cross-run traceability."),
        ("arch_spec", "Architecture spec recorded", "robustness_recheck", "Re-run robustness check if architecture metadata is incomplete."),
        ("baseline_ratio", "Baseline ratio measured", "multi_seed_stress", "Run multi-seed validation to compute baseline ratio."),
        ("multi_seed_std", "Multi-seed variance measured", "multi_seed_stress", "Run multi-seed validation to measure stability variance."),
        ("cka_artifact", "Artifact-backed CKA recorded", "efficiency_scale_up", "After run completion, stamp CKA artifact integrity in artifact references."),
    ]

    steps: List[Dict[str, Any]] = []
    for check_id, label, template_id, guidance in checks:
        is_complete = check_id not in missing
        step: Dict[str, Any] = {
            "check_id": check_id,
            "label": label,
            "status": "complete" if is_complete else "missing",
            "guidance": guidance,
        }
        if result_id:
            step["result_id"] = result_id
        if graph_fingerprint:
            step["graph_fingerprint"] = graph_fingerprint
        if not is_complete and template_id:
            payload = _payload(template_id)
            if payload:
                step["action_label"] = "Run template"
                step["start_payload"] = payload
        steps.append(step)

    next_actions = [
        {
            "check_id": step.get("check_id"),
            "label": step.get("label"),
            "action_label": step.get("action_label"),
            "start_payload": step.get("start_payload"),
            "guidance": step.get("guidance"),
        }
        for step in steps
        if step.get("status") == "missing"
    ][:3]

    return {
        "status": "ready" if ready_count >= total_checks else "in_progress",
        "ready_count": ready_count,
        "total_checks": total_checks,
        "progress_label": f"{ready_count}/{total_checks}",
        "remaining": max(0, total_checks - ready_count),
        "steps": steps,
        "next_actions": next_actions,
        "result_id": result_id,
        "graph_fingerprint": graph_fingerprint,
    }


def _compute_breakthrough_production_readiness(nb: LabNotebook, analytics: Any) -> Dict[str, Any]:
    breakthroughs = nb.get_leaderboard(
        tier="breakthrough", limit=20, sort_by="composite_score", include_references=False
    )
    if not breakthroughs:
        return {
            "breakthrough_count": 0,
            "decision_ready_count": 0,
            "high_confidence_count": 0,
            "full_repro_packet_count": 0,
            "artifact_cka_count": 0,
            "epic_switch_recommendation": {
                "action": "stay_current_epic",
                "reason": "No breakthrough-tier candidates are available yet.",
            },
            "top_candidates": [],
            "scale_up_templates": [],
            "reproducibility_workflow": None,
        }

    stability = _compute_cross_run_stability(nb, nb.get_top_programs(20, sort_by="loss_ratio"))
    stability_by_result = {
        c.get("result_id"): c
        for c in stability.get("candidates", [])
        if c.get("result_id")
    }

    evaluated: List[Dict[str, Any]] = []
    for entry in leaderboard_entries:
        row = dict(entry)
        row["cross_run_stability"] = stability_by_result.get(
            row.get("result_id"),
            {
                "trend": "unknown",
                "seen_runs": 0,
                "latest_rank": None,
                "previous_rank": None,
                "rank_delta": None,
            },
        )
        row["reproducibility_packet"] = analytics.reproducibility_packet_status(row)
        promotion = _promotion_evidence_for_entry(row)
        gate = _decision_gate_for_entry(row)
        scale_up_templates = _build_scale_up_templates_for_result(row.get("result_id"))
        reproducibility_workflow = _build_reproducibility_workflow(
            row["reproducibility_packet"],
            scale_up_templates,
            result_id=row.get("result_id"),
            graph_fingerprint=row.get("graph_fingerprint"),
        )
        evaluated.append({
            "result_id": row.get("result_id"),
            "architecture_family": row.get("architecture_family"),
            "composite_score": _to_safe_float(row.get("composite_score"), 0.0),
            "promotion_confidence_score": promotion["score"],
            "seen_runs": promotion["seen_runs"],
            "decision_ready": gate["decision_ready"],
            "decision_missing": gate["missing"],
            "repro_packet": row["reproducibility_packet"],
            "cka_source": row.get("cka_source"),
            "scale_up_templates": scale_up_templates,
            "reproducibility_workflow": reproducibility_workflow,
        })

    breakthrough_count = len(evaluated)
    decision_ready_count = sum(1 for row in evaluated if row.get("decision_ready"))
    high_confidence_count = sum(1 for row in evaluated if int(row.get("promotion_confidence_score") or 0) >= 75)
    full_repro_packet_count = sum(1 for row in evaluated if (row.get("repro_packet") or {}).get("status") == "ready")
    artifact_cka_count = sum(1 for row in evaluated if row.get("cka_source") == "artifact")

    switch_ready = any(
        row.get("decision_ready")
        and int(row.get("promotion_confidence_score") or 0) >= 75
        and (row.get("repro_packet") or {}).get("status") == "ready"
        and row.get("cka_source") == "artifact"
        for row in evaluated
    )

    if switch_ready:
        recommendation = {
            "action": "switch_to_scale_up_epic",
            "reason": "At least one breakthrough candidate meets decision, confidence, repro, and artifact-backed CKA gates.",
        }
    else:
        recommendation = {
            "action": "stay_current_epic",
            "reason": "Breakthrough evidence is still incomplete; continue hardening reproducibility and validation before switching epics.",
        }

    top_candidates = sorted(
        evaluated,
        key=lambda row: (
            int(bool(row.get("decision_ready"))),
            int(row.get("promotion_confidence_score") or 0),
            _to_safe_float(row.get("composite_score"), 0.0),
        ),
        reverse=True,
    )[:3]
    scale_up_templates = top_candidates[0].get("scale_up_templates", []) if top_candidates else []

    return {
        "breakthrough_count": breakthrough_count,
        "decision_ready_count": decision_ready_count,
        "high_confidence_count": high_confidence_count,
        "full_repro_packet_count": full_repro_packet_count,
        "artifact_cka_count": artifact_cka_count,
        "epic_switch_recommendation": recommendation,
        "top_candidates": top_candidates,
        "scale_up_templates": scale_up_templates,
        "reproducibility_workflow": (
            top_candidates[0].get("reproducibility_workflow")
            if top_candidates
            else None
        ),
    }


def _annotate_qkv_usage(programs: list, analytics) -> None:
    for program in programs:
        if not isinstance(program, dict):
            continue
        qkv_usage = analytics.qkv_usage_enum(program)
        program["qkv_usage"] = qkv_usage
        program["uses_qkv"] = qkv_usage != "qkv_free"
        program["compression_metrics"] = analytics.canonical_compression_metrics(program)
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)


def _normalize_result_ids(raw_ids: Any) -> List[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        if value is None:
            continue
        result_id = str(value).strip()
        if not result_id or result_id in seen:
            continue
        seen.add(result_id)
        normalized.append(result_id)
    return normalized


def _resolve_scale_up_result_ids(
    nb: LabNotebook,
    result_ids: List[str],
    graph_fingerprints: List[str],
) -> Dict[str, Any]:
    """Resolve explicit result IDs and/or fingerprint prefixes for scale-up."""
    merged_result_ids: List[str] = []
    seen: set[str] = set()
    for result_id in result_ids:
        if result_id in seen:
            continue
        seen.add(result_id)
        merged_result_ids.append(result_id)

    resolved: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for fingerprint in graph_fingerprints:
        rows = nb.conn.execute(
            """
            SELECT result_id, graph_fingerprint, experiment_id, stage1_passed,
                   loss_ratio, timestamp
            FROM program_results
            WHERE graph_fingerprint LIKE ?
            ORDER BY stage1_passed DESC,
                     (loss_ratio IS NULL) ASC,
                     loss_ratio ASC,
                     timestamp DESC
            LIMIT 5
            """,
            (f"{fingerprint}%",),
        ).fetchall()

        if not rows:
            unresolved.append(fingerprint)
            continue

        chosen = dict(rows[0])
        chosen_result_id = str(chosen.get("result_id") or "")
        if chosen_result_id and chosen_result_id not in seen:
            seen.add(chosen_result_id)
            merged_result_ids.append(chosen_result_id)

        candidates = [
            {
                "result_id": row["result_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "experiment_id": row["experiment_id"],
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
            }
            for row in rows
        ]
        resolved.append({
            "requested_fingerprint": fingerprint,
            "selected_result_id": chosen.get("result_id"),
            "selected_graph_fingerprint": chosen.get("graph_fingerprint"),
            "selected_experiment_id": chosen.get("experiment_id"),
            "candidate_count": len(rows),
            "candidates": candidates,
        })

    return {
        "result_ids": merged_result_ids,
        "resolved_fingerprints": resolved,
        "unresolved_fingerprints": unresolved,
    }


def _build_start_mode_eligibility(
    nb: LabNotebook,
    mode: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """Validate candidate progression eligibility for start modes.

    Returns a structured payload containing per-candidate reasons.
    """
    payload: Dict[str, Any] = {
        "mode": mode,
        "requested_result_ids": list(result_ids),
        "eligible_result_ids": [],
        "ineligible": [],
        "all_eligible": False,
    }
    if not result_ids:
        return payload

    placeholders = ",".join("?" for _ in result_ids)
    leaderboard_rows = nb.conn.execute(
        f"""
        SELECT result_id, tier, investigation_passed, validation_passed,
               investigation_loss_ratio, validation_loss_ratio
        FROM leaderboard
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()
    program_rows = nb.conn.execute(
        f"""
        SELECT result_id, stage1_passed
        FROM program_results
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()

    leaderboard_by_id = {row["result_id"]: dict(row) for row in leaderboard_rows}
    program_by_id = {row["result_id"]: dict(row) for row in program_rows}

    for result_id in result_ids:
        lb = leaderboard_by_id.get(result_id)
        program = program_by_id.get(result_id)

        if lb is None:
            if program is None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "result_not_found",
                    "detail": "Result ID was not found in program results.",
                })
            elif not bool(program.get("stage1_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_stage1_survivor",
                    "detail": "Result exists but is not a Stage-1 survivor.",
                })
            else:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_in_leaderboard",
                    "detail": "Result exists but has no leaderboard progression record.",
                })
            continue

        tier = str(lb.get("tier") or "").lower()

        if mode == "investigation":
            if tier != "screening":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_screening_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; only screening tier can be investigated.",
                    "tier": tier or None,
                })
                continue
            # Note: We removed the check for existing investigation evidence here
            # to allow retries/overwrites of failed investigations.
            payload["eligible_result_ids"].append(result_id)
            continue

        if mode == "validation":
            if tier != "investigation":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; validation requires investigation tier.",
                    "tier": tier or None,
                })
                continue
            if not bool(lb.get("investigation_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_passed",
                    "detail": "Investigation evidence did not pass robustness gate.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        payload["ineligible"].append({
            "result_id": result_id,
            "reason": "unsupported_mode",
            "detail": f"Eligibility checks are not implemented for mode '{mode}'.",
        })

    payload["all_eligible"] = len(payload["ineligible"]) == 0 and len(payload["eligible_result_ids"]) > 0
    payload["summary"] = {
        "requested": len(result_ids),
        "eligible": len(payload["eligible_result_ids"]),
        "ineligible": len(payload["ineligible"]),
    }
    return payload


def _build_report_action_eligibility(
    nb: LabNotebook,
    result_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-result report action eligibility aligned with start guardrails."""
    normalized_ids = _normalize_result_ids(result_ids)
    if not normalized_ids:
        return {}

    inv = _build_start_mode_eligibility(nb, "investigation", normalized_ids)
    val = _build_start_mode_eligibility(nb, "validation", normalized_ids)

    inv_eligible = set(inv.get("eligible_result_ids") or [])
    val_eligible = set(val.get("eligible_result_ids") or [])
    inv_reason = {
        row.get("result_id"): row.get("reason")
        for row in (inv.get("ineligible") or [])
        if row.get("result_id")
    }
    val_reason = {
        row.get("result_id"): row.get("reason")
        for row in (val.get("ineligible") or [])
        if row.get("result_id")
    }

    eligibility_by_id: Dict[str, Dict[str, Any]] = {}
    for result_id in normalized_ids:
        investigation_eligible = result_id in inv_eligible
        validation_eligible = result_id in val_eligible
        queue_eligible = investigation_eligible or validation_eligible
        queue_reason = None
        if not queue_eligible:
            queue_reason = inv_reason.get(result_id) or val_reason.get(result_id) or "not_progression_eligible"

        eligibility_by_id[result_id] = {
            "investigationEligible": investigation_eligible,
            "validationEligible": validation_eligible,
            "queueEligible": queue_eligible,
            "queueReason": queue_reason,
            "investigationReason": inv_reason.get(result_id),
            "validationReason": val_reason.get(result_id),
        }

    return eligibility_by_id


def _llm_config_path(notebook_path: str) -> Path:
    """Path for persisted LLM configuration, next to the notebook DB."""
    return Path(notebook_path).parent / "llm_config.json"


def _load_persisted_llm_config(notebook_path: str):
    """Auto-load LLM config from disk if present."""
    config_path = _llm_config_path(notebook_path)
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


def _save_llm_config(notebook_path: str, config: Dict):
    """Persist LLM config to disk so it survives restarts."""
    config_path = _llm_config_path(notebook_path)
    try:
        import json as _json
        config_path.write_text(_json.dumps(config, indent=2))
        logger.info(f"Saved LLM config to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save LLM config: {e}")


_DISMISSED_ACTIONS: set = set()

# Singleton autonomy engine (created lazily on first API call)
_aria_autonomy = None
_aria_action_store = None


def _get_autonomy(notebook_path: str):
    """Get or create the singleton AriaAutonomy instance."""
    global _aria_autonomy, _aria_action_store
    if _aria_autonomy is None:
        from .autonomy import AriaAutonomy
        from .actions import ActionStore
        nb = LabNotebook(notebook_path)
        _aria_autonomy = AriaAutonomy(notebook=nb)
        _aria_action_store = ActionStore(nb.conn)
    return _aria_autonomy, _aria_action_store


def _compute_action_queue(nb, analytics=None) -> List[Dict[str, Any]]:
    """Aggregate prioritized actions from existing data sources."""
    actions: List[Dict[str, Any]] = []

    def _is_reference_like(entry: Dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False
        rid = str(entry.get("result_id") or "").strip().lower()
        model_source = str(entry.get("model_source") or "").strip().lower()
        reference_name = str(entry.get("reference_name") or "").strip()
        return (
            bool(entry.get("is_reference"))
            or bool(reference_name)
            or model_source == "reference"
            or rid.startswith("ref_")
        )

    # 1. Breakthrough candidates from leaderboard
    try:
        breakthroughs = nb.get_leaderboard(
            tier="breakthrough", limit=5, sort_by="composite_score", include_references=False
        )
        for entry in breakthroughs:
            # Defensive guard: references should never generate "breakthrough" actions,
            # even if tier flags drift during concurrent DB updates.
            if _is_reference_like(entry):
                continue
            rid = entry.get("result_id", "")
            actions.append({
                "id": f"breakthrough_{rid[:12]}",
                "type": "breakthrough",
                "priority": 1,
                "icon": "trophy",
                "title": f"Architecture {rid[:8]} — Breakthrough",
                "summary": f"Composite score {_to_safe_float(entry.get('composite_score'), 0.0):.3f}. Tier: breakthrough.",
                "detail": {
                    "result_id": rid,
                    "composite_score": _to_safe_float(entry.get("composite_score"), 0.0),
                    "screening_loss_ratio": entry.get("screening_loss_ratio"),
                    "tier": "breakthrough",
                },
                "actions": [
                    {"label": "View Details", "action": "navigate", "payload": {"tab": "discoveries", "result_id": rid}},
                ],
                "dismissable": True,
                "source": "leaderboard",
            })
    except Exception:
        pass

    # 2. Stalled run warning — last 3+ completed experiments with 0 S1 survivors
    try:
        recent = nb.get_recent_experiments(5)
        completed = [e for e in recent if e.get("status") == "completed"]
        if len(completed) >= 3 and all(
            (e.get("n_stage1_passed") or 0) == 0 for e in completed[:3]
        ):
            actions.append({
                "id": "warning_stalled_runs",
                "type": "warning",
                "priority": 2,
                "icon": "warning",
                "title": "Pipeline stalled — zero S1 survivors",
                "summary": f"Last {len(completed[:3])} completed runs produced no Stage 1 survivors.",
                "detail": {
                    "recent_experiments": [
                        {"id": e.get("experiment_id", "")[:12], "s1": e.get("n_stage1_passed", 0)}
                        for e in completed[:3]
                    ],
                },
                "actions": [
                    {"label": "Run Novelty Search", "action": "start", "payload": {"mode": "novelty"}},
                ],
                "dismissable": True,
                "source": "experiments",
            })
    except Exception:
        pass

    # 3. Healer fixes from recent tasks
    try:
        healer_tasks = nb.get_recent_healer_tasks(limit=5)
        active = [t for t in healer_tasks if t.get("state") not in ("completed", "failed")]
        for task in active[:2]:
            tid = task.get("task_id", "")
            actions.append({
                "id": f"healer_{tid[:12]}",
                "type": "healer",
                "priority": 4,
                "icon": "wrench",
                "title": f"Code healer: {task.get('trigger_type', 'repair')}",
                "summary": f"Task {tid[:12]} — {task.get('state', 'active')}. {task.get('scope', '')[:80]}",
                "detail": {
                    "task_id": tid,
                    "state": task.get("state"),
                    "trigger_type": task.get("trigger_type"),
                    "experiment_id": task.get("experiment_id"),
                },
                "actions": [],
                "dismissable": True,
                "source": "healer",
            })
    except Exception:
        pass

    # 4. Diagnosis issues from analytics
    try:
        if analytics:
            analytics_data = analytics.get_analytics_data() if hasattr(analytics, "get_analytics_data") else {}
        else:
            from .analytics import ExperimentAnalytics
            analytics_obj = ExperimentAnalytics(nb)
            analytics_data = analytics_obj.get_analytics_data() if hasattr(analytics_obj, "get_analytics_data") else {}
        issues = _diagnose_research_issues(analytics_data, nb)
        for i, issue in enumerate(issues[:3]):
            actions.append({
                "id": f"diagnosis_{i}",
                "type": "diagnosis",
                "priority": 3,
                "icon": "stethoscope",
                "title": "Diagnosis",
                "summary": issue.get("issue", "Unknown issue"),
                "detail": {"config_fix": issue.get("config_fix")},
                "actions": (
                    [{"label": "Apply Fix", "action": "config_fix", "payload": issue.get("config_fix", {})}]
                    if issue.get("action_type") in ("config_fix", "grammar_fix")
                    else []
                ),
                "dismissable": True,
                "source": "diagnostics",
            })
    except Exception:
        pass

    # 5. Strategy suggestion (lightweight — just presence indicator)
    try:
        summary = nb.get_dashboard_summary()
        total_exp = summary.get("total_experiments", 0)
        if total_exp == 0:
            actions.append({
                "id": "strategy_first_run",
                "type": "strategy",
                "priority": 5,
                "icon": "lightbulb",
                "title": "Get started",
                "summary": "No experiments yet. Start your first continuous run to begin exploring architectures.",
                "detail": {},
                "actions": [
                    {"label": "Start Continuous", "action": "start", "payload": {"mode": "continuous"}},
                ],
                "dismissable": False,
                "source": "strategy",
            })
    except Exception:
        pass

    # Filter out dismissed actions
    actions = [a for a in actions if a["id"] not in _DISMISSED_ACTIONS]

    # Sort by priority
    actions.sort(key=lambda a: a.get("priority", 10))

    return actions[:8]


def create_app(
    notebook_path: str = "research/lab_notebook.db",
    static_folder: Optional[str] = None,
) -> Flask:
    """Create the Flask API app."""

    if static_folder is None:
        static_folder = str(Path(__file__).parent.parent / "dashboard" / "build")

    app = Flask(__name__, static_folder=static_folder, static_url_path="")

    # Custom JSON encoder to handle bytes/numpy types leaking from SQLite
    import json as _json

    class _SafeEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, bytes):
                try:
                    return o.decode("utf-8")
                except UnicodeDecodeError:
                    return None
            if isinstance(o, (memoryview, bytearray)):
                return None
            # numpy scalar types
            type_name = type(o).__name__
            if type_name in ("bool_", "int64", "int32", "float64", "float32", "float16"):
                return o.item()
            return super().default(o)

    app.json.default = _SafeEncoder().default

    CORS(app)
    _ensure_designer_idle_watchdog()

    def _dashboard_index_path() -> Optional[Path]:
        if not app.static_folder:
            return None
        candidate = Path(app.static_folder) / "index.html"
        return candidate if candidate.is_file() else None

    def _dashboard_missing_response():
        expected = str((Path(__file__).parent.parent / "dashboard" / "build" / "index.html"))
        body = (
            "<html><body><h2>Dashboard frontend build is missing.</h2>"
            f"<p>Expected index file at: {expected}</p>"
            "<p>Build dashboard assets (dashboard/build) and retry.</p>"
            "</body></html>"
        )
        return body, 503, {"Content-Type": "text/html; charset=utf-8"}

    def _is_asset_path(path: str) -> bool:
        name = Path(path or "").name
        return "." in name

    # Auto-load persisted LLM config
    _load_persisted_llm_config(notebook_path)

    # ── Global error handlers ──

    @app.errorhandler(404)
    def not_found(e):
        # Only return JSON for API routes; let static files 404 naturally
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        index_path = _dashboard_index_path()
        if index_path and not _is_asset_path(request.path):
            return send_from_directory(app.static_folder, "index.html")
        if _is_asset_path(request.path):
            return "Not found", 404
        return _dashboard_missing_response()

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"500 error on {request.method} {request.path}: {e}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.error(f"Unhandled exception on {request.method} {request.path}: "
                     f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    @app.after_request
    def log_response(response):
        if request.path.startswith("/api/") and response.status_code >= 400:
            logger.warning(f"{request.method} {request.path} -> {response.status_code}")
        return response

    @app.before_request
    def designer_activity_hook():
        if not request.path.startswith("/api/designer"):
            return None
        if request.path in {"/api/designer/lifecycle", "/api/designer/stop"}:
            return None
        if request.method == "OPTIONS":
            return None
        _designer_touch_activity(f"{request.method} {request.path}")
        return None

    # ── Dashboard routes ──

    # (Moved to end of create_app to prevent shadowing API routes)

    # ── Read-only API routes ──


    # ── Register Split API Routes ──
    from .api_routes.deps import ApiRouteContext
    
    context = ApiRouteContext(
        notebook_path=notebook_path,
        dashboard_index_path=_dashboard_index_path,
        dashboard_missing_response=_dashboard_missing_response,
        is_asset_path=_is_asset_path,
        symbols=locals()  # Pass all local helpers/state
    )

    from .api_routes.analytics_bp import register_analytics_routes
    from .api_routes.experiments_bp import register_experiments_routes
    from .api_routes.programs_bp import register_programs_routes
    from .api_routes.leaderboard_bp import register_leaderboard_routes
    from .api_routes.native_bp import register_native_routes
    from .api_routes.campaigns_bp import register_campaigns_routes
    from .api_routes.knowledge_bp import register_knowledge_routes
    from .api_routes.actions_bp import register_actions_routes
    from .api_routes.diagnostics_bp import register_diagnostics_routes
    from .api_routes.config_bp import register_config_routes
    from .api_routes.events_bp import register_events_routes
    from .api_routes.misc_bp import register_misc_routes

    register_analytics_routes(app, context)
    register_experiments_routes(app, context)
    register_programs_routes(app, context)
    register_leaderboard_routes(app, context)
    register_native_routes(app, context)
    register_campaigns_routes(app, context)
    register_knowledge_routes(app, context)
    register_actions_routes(app, context)
    register_diagnostics_routes(app, context)
    register_config_routes(app, context)
    register_events_routes(app, context)
    # misc LAST, since it contains the catch-all /<path:path> fallback
    register_misc_routes(app, context)

    return app


class _PollEndpointFilter(logging.Filter):
    """Suppress noisy werkzeug logs for frequently-polled endpoints."""

    _SUPPRESSED = (
        # SSE / streaming
        "GET /api/events",
        # Dashboard polling (every 3-10s)
        "GET /api/aria/cycle-status",
        "GET /api/aria/autonomy",
        "GET /api/dashboard",
        "GET /api/actions",
        "GET /api/healer/tasks",
        "GET /api/diagnostics/fingerprint",
        # Analytics polling
        "GET /api/analytics/learning-trajectory",
        "GET /api/analytics/math-family-coverage",
        "GET /api/analytics/regression-vs-baseline",
        "GET /api/leaderboard",
        "GET /api/trends/context",
        "GET /api/insights",
        # Static / designer assets
        "GET /static/",
        "GET /designer-proxy/assets/",
        # Designer keepalive
        "POST /api/designer/touch",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)


def _setup_logging(log_dir: Optional[str] = None):
    """Configure logging with console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Suppress polling endpoint spam from werkzeug
    logging.getLogger("werkzeug").addFilter(_PollEndpointFilter())

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent)
    log_path = Path(log_dir) / "aria_dashboard.log"
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=1,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")


def run_server(
    notebook_path: str = "research/lab_notebook.db",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    _setup_logging()
    app = create_app(notebook_path)
    logger.info(f"Starting Aria's Dashboard API on http://{host}:{port}")
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
