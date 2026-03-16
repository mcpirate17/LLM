"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from research.defaults import LAB_NOTEBOOK_DB
from .api_routes import _designer as _designer_mod
from .designer_utils import (
    validate_designer_graph,
    compile_designer_graph,
    run_designer_graph,
    get_designer_components,
    generate_python_module,
)

logger = logging.getLogger(__name__)

_requests = _designer_mod._requests
_DESIGNER_PROXY_ENABLED = _designer_mod._DESIGNER_PROXY_ENABLED
_DESIGNER_PROXY_BASE = _designer_mod._DESIGNER_PROXY_BASE
_DESIGNER_PROXY_TIMEOUT = _designer_mod._DESIGNER_PROXY_TIMEOUT


def _designer_proxy(method: str, path: str, *, json_body=None, params=None, timeout=None):
    _designer_mod._requests = _requests
    _designer_mod._DESIGNER_PROXY_ENABLED = _DESIGNER_PROXY_ENABLED
    return _designer_mod.designer_proxy(
        method,
        path,
        json_body=json_body,
        params=params,
        timeout=timeout,
    )


def _proxy_or_error(resp):
    return _designer_mod.proxy_or_error(resp)


def create_app(
    notebook_path: str = LAB_NOTEBOOK_DB,
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

    # Start designer idle watchdog
    from .api_routes._designer import ensure_designer_idle_watchdog, designer_touch_activity
    ensure_designer_idle_watchdog()

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
    from .api_routes._helpers import load_persisted_llm_config
    load_persisted_llm_config(notebook_path)

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
