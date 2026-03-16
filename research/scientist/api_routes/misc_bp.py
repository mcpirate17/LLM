"""Static dashboard asset route registration."""
from __future__ import annotations

import logging
from pathlib import Path

from flask import jsonify, request, send_from_directory
from ..notebook import LabNotebook
from ..persona import get_aria
from ..code_agent import _spawn_code_agent_task
from ._chat import code_agent_task_snapshot, run_local_chat_agent
from ._designer import (
    designer_idle_state,
    designer_proxy,
    designer_service_status,
    designer_touch_activity,
    proxy_or_error,
    proxy_stream,
    start_designer_services,
    stop_designer_services,
)
from ..designer_utils import (
    compile_designer_graph,
    get_designer_components,
    run_designer_graph,
    validate_designer_graph,
)
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_misc_routes(app, context: ApiRouteContext):
    _dashboard_index_path = context.dashboard_index_path
    _dashboard_missing_response = context.dashboard_missing_response
    _is_asset_path = context.is_asset_path

    designer_dist = str(Path(__file__).resolve().parents[3] / "aria_designer" / "ui" / "dist")

    @app.route("/designer-proxy/")
    def designer_index():
        """Serve the built aria_designer index for the embedded iframe."""
        return send_from_directory(designer_dist, "index.html")

    @app.route("/designer-proxy/<path:subpath>")
    def designer_assets(subpath):
        """Serve aria_designer static assets."""
        return send_from_directory(designer_dist, subpath)

    @app.route("/")
    def index():
        if not _dashboard_index_path():
            return _dashboard_missing_response()
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/favicon.ico")
    def favicon():
        if app.static_folder:
            icon = Path(app.static_folder) / "favicon.ico"
            if icon.is_file():
                return send_from_directory(app.static_folder, "favicon.ico")
        return "", 204

    @app.route("/<path:path>")
    def static_files(path):
        if app.static_folder:
            static_path = Path(app.static_folder) / path
            if static_path.is_file():
                return send_from_directory(app.static_folder, path)
        index_path = _dashboard_index_path()
        if index_path and not _is_asset_path(path):
            return send_from_directory(app.static_folder, "index.html")
        return "Not found", 404
