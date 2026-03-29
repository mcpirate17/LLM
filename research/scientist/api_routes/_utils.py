"""Shared route utilities — decorators and helpers for API blueprints."""

from __future__ import annotations

import functools
import logging
import sqlite3

from flask import jsonify, request

from .deps import get_notebook

logger = logging.getLogger(__name__)


def with_notebook_context(notebook_path: str):
    """Decorator factory: injects a shared ``nb`` kwarg and catches
    unhandled exceptions with a standard JSON error response.

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
            try:
                nb = get_notebook(notebook_path)
            except sqlite3.OperationalError as e:
                logger.warning(
                    "%s %s -> 503 (db locked during init): %s",
                    request.method,
                    request.path,
                    e,
                )
                return jsonify(
                    {"error": "Database temporarily busy, retry shortly"}
                ), 503
            try:
                return fn(*args, nb=nb, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    logger.warning(
                        "%s %s -> 503 (db locked): %s",
                        request.method,
                        request.path,
                        e,
                    )
                    return jsonify(
                        {"error": "Database temporarily busy, retry shortly"}
                    ), 503
                logger.error(
                    "Unhandled db error on %s %s: %s",
                    request.method,
                    request.path,
                    e,
                    exc_info=True,
                )
                return jsonify({"error": str(e)}), 500
            except Exception as e:
                logger.error("Error in %s: %s", request.path, e, exc_info=True)
                return jsonify({"error": str(e)}), 500

        return wrapper

    return decorator
