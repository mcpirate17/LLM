"""Designer lifecycle orchestration and proxy utilities.

Manages the aria_designer backend/frontend lifecycle: start, stop, idle
watchdog, health probing, and HTTP proxy for designer API routes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests as _requests
from flask import jsonify, Response

from research.defaults import (
    DESIGNER_API_BASE,
    DESIGNER_UI_BASE,
    DESIGNER_API_HEALTH,
    DESIGNER_PROXY_TIMEOUT,
    DESIGNER_BOOT_TIMEOUT,
    DESIGNER_IDLE_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ── Designer proxy configuration ────────────────────────────────────────
_DESIGNER_PROXY_BASE = os.environ.get("ARIA_DESIGNER_PROXY_BASE", DESIGNER_API_BASE)
_DESIGNER_PROXY_ENABLED = os.environ.get("ARIA_DESIGNER_PROXY_ENABLED", "1") != "0"
_DESIGNER_PROXY_TIMEOUT = float(
    os.environ.get("ARIA_DESIGNER_PROXY_TIMEOUT", str(DESIGNER_PROXY_TIMEOUT))
)

# ── Designer lifecycle orchestration ────────────────────────────────────
_ARIA_DESIGNER_ROOT = Path(
    os.environ.get(
        "ARIA_DESIGNER_ROOT",
        str(Path(__file__).resolve().parents[3] / "aria_designer"),
    )
)
_ARIA_DESIGNER_API_HEALTH = os.environ.get(
    "ARIA_DESIGNER_API_HEALTH", DESIGNER_API_HEALTH
)
_ARIA_DESIGNER_UI_HEALTH = os.environ.get("ARIA_DESIGNER_UI_HEALTH", DESIGNER_UI_BASE)
_ARIA_DESIGNER_BOOT_TIMEOUT_S = float(
    os.environ.get("ARIA_DESIGNER_BOOT_TIMEOUT_S", str(DESIGNER_BOOT_TIMEOUT))
)
_ARIA_DESIGNER_IDLE_TIMEOUT_S = float(
    os.environ.get("ARIA_DESIGNER_IDLE_TIMEOUT_S", str(DESIGNER_IDLE_TIMEOUT))
)
_DESIGNER_LIFECYCLE_LOCK = threading.Lock()
_DESIGNER_ACTIVITY_LOCK = threading.Lock()
_DESIGNER_LAST_ACTIVITY_TS = time.time()
_DESIGNER_LAST_ACTIVITY_REASON = "startup"
_DESIGNER_LAST_AUTOSTOP_TS: Optional[float] = None
_DESIGNER_IDLE_WATCHDOG_STARTED = False


def designer_service_status() -> Dict[str, Any]:
    """Probe aria_designer API/UI health."""
    api_up = False
    ui_up = False
    try:
        r = _requests.get(_ARIA_DESIGNER_API_HEALTH, timeout=1.0)
        api_up = r.status_code < 500
    except (OSError, ValueError):
        api_up = False
    try:
        r = _requests.get(_ARIA_DESIGNER_UI_HEALTH, timeout=1.0)
        ui_up = r.status_code < 500
    except (OSError, ValueError):
        ui_up = False
    return {
        "api_up": api_up,
        "ui_up": ui_up,
        "running": bool(api_up and ui_up),
        "api_health_url": _ARIA_DESIGNER_API_HEALTH,
        "ui_health_url": _ARIA_DESIGNER_UI_HEALTH,
    }


def designer_touch_activity(reason: str = "activity") -> Dict[str, Any]:
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


def designer_idle_state() -> Dict[str, Any]:
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


def ensure_designer_idle_watchdog() -> None:
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
                    status = designer_service_status()
                    if status.get("running"):
                        result = stop_designer_services()
                        if result.get("ok"):
                            with _DESIGNER_ACTIVITY_LOCK:
                                _DESIGNER_LAST_AUTOSTOP_TS = time.time()
                time.sleep(5.0)
            except Exception:  # noqa: BLE001 — daemon watchdog must not crash
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


def start_designer_services(force_restart: bool = False) -> Dict[str, Any]:
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
        status0 = designer_service_status()
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
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)

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
        latest = designer_service_status()
        while time.time() < deadline:
            latest = designer_service_status()
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


def stop_designer_services() -> Dict[str, Any]:
    """Best-effort stop of aria_designer FE/BE via shared scripts."""
    dev_down = _designer_dev_down_path()
    if not dev_down.exists():
        return {"ok": False, "error": f"Missing stop script: {dev_down}"}
    with _DESIGNER_LIFECYCLE_LOCK:
        status_before = designer_service_status()
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
        latest = designer_service_status()
        while time.time() < deadline:
            latest = designer_service_status()
            if not latest["api_up"] and not latest["ui_up"]:
                return {
                    "ok": True,
                    "status_before": status_before,
                    "status_after": latest,
                }
            time.sleep(0.3)
        return {"ok": True, "status_before": status_before, "status_after": latest}


def designer_proxy(
    method: str,
    path: str,
    *,
    json_body=None,
    params=None,
    timeout: Optional[float] = None,
) -> Optional[_requests.Response]:
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
            method,
            url,
            json=json_body,
            params=params,
            timeout=_timeout,
        )
        return resp
    except _requests.ConnectionError:
        logger.debug("Designer proxy unavailable at %s", _DESIGNER_PROXY_BASE)
        return None
    except _requests.Timeout:
        logger.warning(
            "Designer proxy timeout after %.1fs for %s %s", _timeout, method, path
        )
        return None
    except Exception:  # noqa: BLE001 — catch-all after specific request exceptions
        logger.exception("Designer proxy unexpected error for %s %s", method, path)
        return None


def proxy_or_error(resp: Optional[_requests.Response]):
    """Convert a proxy response to a Flask Response object, or return None
    if no proxy response is available (caller falls back to legacy)."""
    if resp is None:
        return None
    try:
        body = resp.json()
    except (ValueError, TypeError):
        body = {"error": resp.text or "Proxy returned non-JSON response"}

    return jsonify(body), resp.status_code


def proxy_stream(method: str, path: str, *, json_body=None, params=None):
    """Stream-proxy an SSE endpoint from the aria_designer backend."""
    if not _DESIGNER_PROXY_ENABLED:
        return jsonify({"error": "Designer proxy not enabled"}), 502
    url = f"{_DESIGNER_PROXY_BASE.rstrip('/')}{path}"
    try:
        upstream = _requests.request(
            method,
            url,
            json=json_body,
            params=params,
            stream=True,
            timeout=120,
        )

        def generate():
            try:
                for chunk in upstream.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        content_type = upstream.headers.get("content-type", "text/event-stream")
        return Response(
            generate(), status=upstream.status_code, content_type=content_type
        )
    except _requests.ConnectionError:
        return jsonify({"error": "Designer backend unavailable"}), 502
    except _requests.Timeout:
        return jsonify({"error": "Designer backend timeout"}), 504
    except Exception as e:
        logger.exception("Stream proxy error for %s %s", method, path)
        return jsonify({"error": str(e)}), 502
