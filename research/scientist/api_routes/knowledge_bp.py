"""knowledge API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from ._utils import register_notebook_routes, with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_knowledge_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    def api_knowledge(nb=None):
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        entries = nb.get_knowledge(category=category)
        return jsonify(entries)

    def api_knowledge_search(nb=None):
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        entries = nb.search_knowledge(q)
        return jsonify(entries)

    register_notebook_routes(
        app,
        wnb,
        (
            ("/api/knowledge", "api_knowledge", api_knowledge),
            ("/api/knowledge/search", "api_knowledge_search", api_knowledge_search),
        ),
    )
