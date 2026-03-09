"""knowledge API route registration."""
from __future__ import annotations

import logging
from flask import jsonify, request
from ..notebook import LabNotebook
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_knowledge_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/knowledge")
    def api_knowledge():
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_knowledge(category=category)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in /api/knowledge: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/search")
    def api_knowledge_search():
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.search_knowledge(q)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/backfill", methods=["POST"])
    def api_knowledge_backfill():
        """Backfill missing knowledge categories from measured experiment data."""
        nb = LabNotebook(notebook_path)
        try:
            # _backfill_knowledge_from_real_data was never extracted from the
            # monolith — stub until the actual implementation is located.
            return jsonify({"status": "not_implemented", "detail": "Backfill helper not yet migrated"}), 501
        except Exception as e:
            logger.error(f"Error in /api/knowledge/backfill: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()
