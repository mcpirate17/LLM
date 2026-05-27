"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import signal
import shutil
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from research.defaults import RUNS_DB
from .notebook import LabNotebook
from .api_routes import _designer as _designer_mod
from .api_routes._api_health import API_HEALTH_COUNTERS, API_HEALTH_LOCK

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
_DEFAULT_DASHBOARD_BUILD_DIR = _DASHBOARD_DIR / "build"
_FAULT_LOG_STREAM = None
_PROCESS_LIFECYCLE_REGISTERED = False

# Test hooks (monkeypatched by test_designer_proxy_live.py)
_DESIGNER_PROXY_ENABLED = _designer_mod._DESIGNER_PROXY_ENABLED
_DESIGNER_PROXY_BASE = _designer_mod._DESIGNER_PROXY_BASE


def create_app(
    notebook_path: str = RUNS_DB,
    static_folder: Optional[str] = None,
) -> Flask:
    """Create the Flask API app."""

    if static_folder is None:
        static_folder = str(_DEFAULT_DASHBOARD_BUILD_DIR)

    _ensure_default_dashboard_build(static_folder)

    app = Flask(__name__, static_folder=static_folder, static_url_path="")

    # Custom JSON encoder to handle bytes/numpy types leaking from SQLite
    from .json_utils import SafeJSONEncoder

    app.json.default = SafeJSONEncoder().default

    CORS(app)

    # Start designer idle watchdog
    from .api_routes._designer import (
        ensure_designer_idle_watchdog,
        designer_touch_activity,
    )

    ensure_designer_idle_watchdog()

    # Hourly notebook snapshot rotation (redundancy baseline).
    from .snapshot_rotator import ensure_snapshot_rotator

    ensure_snapshot_rotator(notebook_path)

    def _dashboard_index_path() -> Optional[Path]:
        if not app.static_folder:
            return None
        candidate = Path(app.static_folder) / "index.html"
        return candidate if candidate.is_file() else None

    def _dashboard_missing_response():
        expected = str(
            (Path(__file__).parent.parent / "dashboard" / "build" / "index.html")
        )
        body = (
            "<html><body><h2>Dashboard frontend build is missing.</h2>"
            f"<p>Expected index file at: {expected}</p>"
            "<p>Build dashboard assets (dashboard/build) and retry.</p>"
            "</body></html>"
        )
        return body, 503, {"Content-Type": "text/html; charset=utf-8"}

    def _is_asset_path(path: str) -> bool:
        normalized = (path or "").lstrip("/")
        if normalized == "@react-refresh" or normalized.startswith("@vite/"):
            return True
        name = Path(normalized).name
        return "." in name

    # ── Global error handlers ──

    @app.errorhandler(404)
    def not_found(e):
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
        logger.error(
            f"Unhandled exception on {request.method} {request.path}: "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    @app.after_request
    def log_response(response):
        if request.path.startswith("/api/"):
            code = response.status_code
            if code < 400:
                bucket = "2xx"
            elif code < 500:
                bucket = "4xx"
            else:
                bucket = "5xx"
            key = f"{request.path}:{bucket}"
            with API_HEALTH_LOCK:
                API_HEALTH_COUNTERS[key] += 1
            if code >= 400:
                logger.warning(f"{request.method} {request.path} -> {code}")
        return response

    @app.before_request
    def designer_activity_hook():
        if not request.path.startswith("/api/designer"):
            return None
        if request.path in {"/api/designer/lifecycle", "/api/designer/stop"}:
            return None
        if request.method == "OPTIONS":
            return None
        designer_touch_activity(f"{request.method} {request.path}")
        return None

    # ── Register Split API Routes ──
    from .api_routes.deps import ApiRouteContext

    context = ApiRouteContext(
        notebook_path=notebook_path,
        dashboard_index_path=_dashboard_index_path,
        dashboard_missing_response=_dashboard_missing_response,
        is_asset_path=_is_asset_path,
    )

    from .api_routes.analytics_bp import register_analytics_routes
    from .api_routes.experiments_bp import register_experiments_routes
    from .api_routes.programs_bp import register_programs_routes
    from .api_routes.reporting_bp import register_reporting_routes
    from .api_routes.strategy_bp import register_strategy_bp_routes
    from .api_routes.general_bp import register_general_routes
    from .api_routes.chat_bp import register_chat_routes
    from .api_routes.leaderboard_bp import register_leaderboard_routes
    from .api_routes.native_bp import register_native_routes
    from .api_routes.campaigns_bp import register_campaigns_routes
    from .api_routes.knowledge_bp import register_knowledge_routes
    from .api_routes.actions_bp import register_actions_routes
    from .api_routes.diagnostics_bp import register_diagnostics_routes
    from .api_routes.config_bp import register_config_routes
    from .api_routes.events_bp import register_events_routes
    from .api_routes.system_bp import register_system_routes
    from .api_routes.designer_bp import register_designer_routes
    from .api_routes.observability_bp import register_observability_routes
    from .api_routes.misc_bp import register_misc_routes

    register_analytics_routes(app, context)
    register_experiments_routes(app, context)
    register_programs_routes(app, context)
    register_reporting_routes(app, context)
    register_strategy_bp_routes(app, context)
    register_general_routes(app, context)
    register_chat_routes(app, context)
    register_leaderboard_routes(app, context)
    register_native_routes(app, context)
    register_campaigns_routes(app, context)
    register_knowledge_routes(app, context)
    register_actions_routes(app, context)
    register_diagnostics_routes(app, context)
    register_config_routes(app, context)
    register_events_routes(app, context)
    register_system_routes(app, context)
    register_designer_routes(app, context)
    register_observability_routes(app, context)
    # misc LAST, since it contains the catch-all /<path:path> fallback
    register_misc_routes(app, context)

    return app


def _ensure_default_dashboard_build(static_folder: Optional[str]) -> None:
    """Best-effort build of the bundled dashboard when its production assets are absent."""

    if not static_folder:
        return
    if os.environ.get("ARIA_AUTO_BUILD_DASHBOARD", "1") in {"0", "false", "False"}:
        return

    try:
        static_path = Path(static_folder).resolve()
        default_build_path = _DEFAULT_DASHBOARD_BUILD_DIR.resolve()
    except OSError:
        return

    if static_path != default_build_path:
        return
    if (default_build_path / "index.html").is_file():
        return

    npm_path = shutil.which("npm")
    package_json = _DASHBOARD_DIR / "package.json"
    node_modules = _DASHBOARD_DIR / "node_modules"

    if not package_json.is_file():
        logger.warning(
            "Dashboard build missing at %s and %s is unavailable",
            default_build_path,
            package_json,
        )
        return
    if not node_modules.is_dir():
        logger.warning(
            "Dashboard build missing at %s and %s is unavailable; skipping auto-build",
            default_build_path,
            node_modules,
        )
        return
    if not npm_path:
        logger.warning(
            "Dashboard build missing at %s and npm is not installed; skipping auto-build",
            default_build_path,
        )
        return

    logger.info("Dashboard build missing at %s; running npm build", default_build_path)
    try:
        subprocess.run(
            [npm_path, "run", "build"],
            cwd=str(_DASHBOARD_DIR),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or str(exc)).strip()
        logger.warning("Dashboard auto-build failed: %s", error_output)
        return

    if (default_build_path / "index.html").is_file():
        logger.info("Dashboard auto-build completed successfully")
    else:
        logger.warning(
            "Dashboard auto-build finished without producing %s",
            default_build_path / "index.html",
        )


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
    werkzeug_logger = logging.getLogger("werkzeug")
    if not any(isinstance(f, _PollEndpointFilter) for f in werkzeug_logger.filters):
        werkzeug_logger.addFilter(_PollEndpointFilter())

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    console._aria_dashboard_handler = True
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent)
    log_path = Path(log_dir) / "aria_dashboard.log"
    try:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=1,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        file_handler._aria_dashboard_handler = True
        root.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")

    return log_path


def _register_process_lifecycle_logging(
    notebook_path: str,
    log_path: Path,
) -> None:
    """Log dashboard process exits/signals and dump tracebacks on fatal paths."""
    global _FAULT_LOG_STREAM, _PROCESS_LIFECYCLE_REGISTERED

    if _PROCESS_LIFECYCLE_REGISTERED:
        return

    try:
        _FAULT_LOG_STREAM = open(log_path, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(_FAULT_LOG_STREAM, all_threads=True)
    except Exception as exc:
        logger.warning("Failed to enable faulthandler at %s: %s", log_path, exc)
        _FAULT_LOG_STREAM = None

    def _log_exit(reason: str) -> None:
        logger.error(
            "Dashboard process exit path: %s | pid=%d ppid=%d thread=%s notebook=%s",
            reason,
            os.getpid(),
            os.getppid(),
            threading.current_thread().name,
            notebook_path,
        )
        if _FAULT_LOG_STREAM is not None:
            faulthandler.dump_traceback(file=_FAULT_LOG_STREAM, all_threads=True)
            _FAULT_LOG_STREAM.flush()
        for handler in logging.getLogger().handlers:
            handler.flush()

    atexit.register(lambda: _log_exit("atexit"))

    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        prev_handler = signal.getsignal(sig)

        def _make_handler(_sig_name, _prev_handler):
            def _handler(signum, frame):
                _log_exit(f"signal={_sig_name}")
                if callable(_prev_handler) and _prev_handler not in (
                    signal.SIG_DFL,
                    signal.SIG_IGN,
                ):
                    return _prev_handler(signum, frame)
                if _prev_handler == signal.SIG_IGN:
                    return None
                raise SystemExit(128 + signum)

            return _handler

        signal.signal(sig, _make_handler(sig_name, prev_handler))

    _PROCESS_LIFECYCLE_REGISTERED = True


def _recover_orphaned_running_experiments(notebook_path: str) -> int:
    """At API startup, mark inherited 'running' rows as failed.

    The dashboard executes experiments in-process. If the API process is starting,
    any pre-existing DB rows still marked 'running' are orphaned from a dead process.

    Peeks via the read-only manager first so a healthy second writer never causes
    a noisy lock-conflict traceback. If a peer aria-db writer is already alive on
    this database, the 'running' rows belong to that writer (not orphaned), so we
    skip recovery silently.
    """
    # Peek with a short-lived read-only SQLite handle: no writer flock, no
    # process-wide read-only singleton snapshot.
    running_ids: list[str] = []
    nb_ro = None
    try:
        nb_ro = LabNotebook(notebook_path, read_only=True, use_native=False)
        running_rows = nb_ro.conn.execute(
            "SELECT experiment_id FROM experiments WHERE status = 'running'"
        ).fetchall()
        running_ids = [
            str(row["experiment_id"]) for row in running_rows if row["experiment_id"]
        ]
    except Exception as exc:
        logger.warning(
            "Startup orphan-recovery readonly probe failed: %s", exc, exc_info=True
        )
        return 0
    finally:
        if nb_ro is not None:
            nb_ro.close()

    if not running_ids:
        return 0

    # Need writer to mark them failed. If another writer holds the lock, those
    # rows belong to that live process — not orphaned.
    nb = None
    try:
        nb = LabNotebook(notebook_path)
    except Exception as exc:
        if "another process already holds the writer lock" in str(exc):
            logger.info(
                "Skipping startup orphan recovery: a peer aria-db writer is "
                "active on %s; %d 'running' row(s) belong to that writer.",
                notebook_path,
                len(running_ids),
            )
            return 0
        logger.warning(
            "Startup orphan recovery could not acquire writer: %s",
            exc,
            exc_info=True,
        )
        return 0
    try:
        cleaned = nb.cleanup_stale_experiments(
            timeout_minutes=0, startup_failure_minutes=0
        )
        if cleaned:
            logger.warning(
                "Recovered %d orphaned running experiment(s) at dashboard startup: %s",
                cleaned,
                ", ".join(running_ids[:10]),
            )
        else:
            logger.info(
                "Startup orphan recovery found %d pre-existing running row(s) "
                "but no rows required cleanup: %s",
                len(running_ids),
                ", ".join(running_ids[:10]),
            )
        return cleaned
    except Exception as exc:
        logger.warning("Startup orphaned-run recovery failed: %s", exc, exc_info=True)
        return 0
    finally:
        nb.close()


def run_server(
    notebook_path: str = RUNS_DB,
    host: str = "0.0.0.0",  # nosec B104
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    log_path = _setup_logging()
    _register_process_lifecycle_logging(notebook_path, Path(log_path))
    recovered = _recover_orphaned_running_experiments(notebook_path)
    app = create_app(notebook_path)
    if recovered:
        logger.warning(
            "Dashboard startup recovered %d orphaned experiment(s) before serving API",
            recovered,
        )
    logger.info(f"Starting Aria's Dashboard API on http://{host}:{port}")
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
