"""diagnostics API route registration."""
from __future__ import annotations

import logging
import os
from flask import jsonify, request
from ..notebook import LabNotebook
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_diagnostics_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/diagnostics/fingerprint")
    def api_fingerprint_diagnostics():
        """Expose lightweight runtime diagnostics for fingerprint analysis."""
        reset = str(request.args.get("reset", "0")).strip().lower() in {"1", "true", "yes"}
        try:
            from research.eval.fingerprint import get_sensitivity_skip_stats

            stats = get_sensitivity_skip_stats(reset=reset)
            return jsonify({
                "sensitivity_skips": stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/fingerprint: {e}")
            return jsonify({
                "sensitivity_skips": {
                    "total": 0,
                    "by_reason": {},
                },
                "error": str(e),
            }), 500

    @app.route("/api/diagnostics/report-cache")
    def api_report_cache_diagnostics():
        """Expose report snapshot cache usage and retention diagnostics."""
        nb = LabNotebook(notebook_path)
        try:
            cleanup = str(request.args.get("cleanup", "0")).strip().lower() in {"1", "true", "yes"}
            try:
                ttl_seconds = int(os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)))
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400"))
            except Exception:
                max_rows_per_scope = 400

            cleanup_stats = None
            if cleanup:
                cleanup_stats = nb.cleanup_report_snapshots(
                    ttl_seconds=max(60, ttl_seconds),
                    max_rows_per_scope=max(20, max_rows_per_scope),
                )

            snapshot_stats = nb.get_report_snapshot_stats()
            return jsonify({
                "snapshot_cache": snapshot_stats,
                "retention": {
                    "ttl_seconds": max(60, int(ttl_seconds or 0)),
                    "max_rows_per_scope": max(20, int(max_rows_per_scope or 0)),
                },
                "cleanup_triggered": bool(cleanup),
                "cleanup": cleanup_stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/report-cache: {e}")
            return jsonify({
                "snapshot_cache": {
                    "total_snapshots": 0,
                    "n_scopes": 0,
                    "oldest_age_seconds": None,
                    "newest_age_seconds": None,
                    "scopes": [],
                },
                "error": str(e),
            }), 500
        finally:
            nb.close()
