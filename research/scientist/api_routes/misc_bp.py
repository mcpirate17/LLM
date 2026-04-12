"""Static dashboard asset route registration."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from flask import Response, request, send_from_directory

from research.defaults import DESIGNER_UI_BASE
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)

_PROXY_TIMEOUT_S = 10.0
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _proxy_designer_ui(subpath: str = ""):
    """Proxy embedded designer UI requests to the live Vite dev server."""
    upstream = f"{DESIGNER_UI_BASE.rstrip('/')}/{subpath.lstrip('/')}"
    if not subpath:
        upstream = f"{DESIGNER_UI_BASE.rstrip('/')}/"
    try:
        upstream_response = requests.get(
            upstream,
            params=request.args,
            timeout=_PROXY_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        logger.warning("Designer UI proxy failed for %s: %s", upstream, exc)
        return Response("Designer UI unavailable", status=502)

    body = upstream_response.content
    content_type = upstream_response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = upstream_response.text
        html = html.replace(
            'from "/@react-refresh"', 'from "/designer-proxy/@react-refresh"'
        )
        html = html.replace('src="/@vite/client"', 'src="/designer-proxy/@vite/client"')
        html = html.replace('src="/src/main.jsx"', 'src="/designer-proxy/src/main.jsx"')
        body = html.encode(upstream_response.encoding or "utf-8")

    headers = [
        (key, value)
        for key, value in upstream_response.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    ]
    return Response(
        body,
        status=upstream_response.status_code,
        headers=headers,
    )


def register_misc_routes(app, context: ApiRouteContext):
    _dashboard_index_path = context.dashboard_index_path
    _dashboard_missing_response = context.dashboard_missing_response
    _is_asset_path = context.is_asset_path

    designer_dist = str(
        Path(__file__).resolve().parents[3] / "aria_designer" / "ui" / "dist"
    )
    designer_dist_path = Path(designer_dist)

    @app.route("/designer-proxy/")
    def designer_index():
        """Serve the built aria_designer index for the embedded iframe."""
        if (designer_dist_path / "index.html").is_file():
            return send_from_directory(designer_dist, "index.html")
        return _proxy_designer_ui()

    @app.route("/designer-proxy/<path:subpath>")
    def designer_assets(subpath):
        """Serve aria_designer static assets."""
        asset_path = designer_dist_path / subpath
        if asset_path.is_file():
            return send_from_directory(designer_dist, subpath)
        return _proxy_designer_ui(subpath)

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
