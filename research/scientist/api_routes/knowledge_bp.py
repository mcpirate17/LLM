"""knowledge API route registration."""
from __future__ import annotations

import functools
import time
import datetime
from flask import jsonify, request, Response
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from .deps import ApiRouteContext, install_legacy_symbols

def register_knowledge_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)

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
            result = _backfill_knowledge_from_real_data(nb)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in /api/knowledge/backfill: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


