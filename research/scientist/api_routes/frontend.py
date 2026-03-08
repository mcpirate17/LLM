"""Frontend/static route registration for dashboard and embedded designer assets."""
from __future__ import annotations

from .deps import ApiRouteContext, install_legacy_symbols

def register_frontend_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)
    _designer_dist = str(
        Path(__file__).parent.parent.parent / "aria_designer" / "ui" / "dist"
    )

    @app.route("/designer-proxy/")
    def designer_index():
        """Serve the built aria_designer index.html for the embedded iframe.

        Serving the designer from the same origin as the dashboard avoids
        cross-origin iframe restrictions in Brave and other browsers.
        """
        return send_from_directory(_designer_dist, "index.html")

    @app.route("/designer-proxy/<path:subpath>")
    def designer_assets(subpath):
        """Serve aria_designer static assets (JS, CSS, etc.)."""
        return send_from_directory(_designer_dist, subpath)

    # ── Dashboard routes ──

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
