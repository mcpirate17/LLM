"""Shared route utilities — decorators and helpers for API blueprints."""

from __future__ import annotations

import functools
import logging

from flask import jsonify, request

from .deps import get_notebook

logger = logging.getLogger(__name__)


def with_notebook_context(notebook_path: str):
    """Decorator factory: injects a request-scoped ``nb`` kwarg and catches
    unhandled exceptions with a standard 500 JSON response.

    Usage::

        wnb = with_notebook_context(notebook_path)

        @app.route("/api/foo")
        @wnb
        def api_foo(nb=None):
            return jsonify(nb.get_something())
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nb = get_notebook(notebook_path)
            try:
                return fn(*args, nb=nb, **kwargs)
            except Exception as e:
                logger.error("Error in %s: %s", request.path, e, exc_info=True)
                return jsonify({"error": str(e)}), 500

        return wrapper

    return decorator
