"""knowledge API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_knowledge_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/knowledge")
    @wnb
    def api_knowledge(nb=None):
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        entries = nb.get_knowledge(category=category)
        return jsonify(entries)

    @app.route("/api/knowledge/search")
    @wnb
    def api_knowledge_search(nb=None):
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        entries = nb.search_knowledge(q)
        return jsonify(entries)
