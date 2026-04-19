"""Shared route utilities — decorators and helpers for API blueprints."""

from __future__ import annotations

import functools
import logging
import sqlite3
from typing import Any, Dict

from flask import jsonify, request

from .deps import get_notebook

logger = logging.getLogger(__name__)


def bind_view(handler, *bound_args):
    return functools.partial(handler, *bound_args) if bound_args else handler


def bind_notebook_view(wnb, handler, *bound_args):
    return wnb(bind_view(handler, *bound_args))


def is_malformed_db_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "database disk image is malformed",
            "malformed",
            "not a database",
            "database corrupt",
        )
    )


def malformed_db_response_payload(exc: Exception) -> Dict[str, Any]:
    return {
        "error": "Notebook database is corrupted; returning degraded response",
        "details": str(exc),
        "degraded": True,
        "database_status": {
            "healthy": False,
            "error_type": "malformed",
            "message": str(exc),
        },
    }


def with_notebook_context(notebook_path: str, *, read_only: bool | None = None):
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
            method = str(getattr(request, "method", "GET") or "GET").upper()
            effective_read_only = (
                read_only
                if read_only is not None
                else method in {"GET", "HEAD", "OPTIONS"}
            )
            nb = None
            try:
                nb = get_notebook(notebook_path, read_only=effective_read_only)
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
            except sqlite3.DatabaseError as e:
                if is_malformed_db_error(e):
                    logger.warning(
                        "%s %s -> 503 (db malformed during init): %s",
                        request.method,
                        request.path,
                        e,
                    )
                    return jsonify(malformed_db_response_payload(e)), 503
                logger.error(
                    "Unhandled db init error on %s %s: %s",
                    request.method,
                    request.path,
                    e,
                    exc_info=True,
                )
                return jsonify({"error": str(e)}), 500
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
            except sqlite3.DatabaseError as e:
                if is_malformed_db_error(e):
                    logger.warning(
                        "%s %s -> 503 (db malformed): %s",
                        request.method,
                        request.path,
                        e,
                    )
                    return jsonify(malformed_db_response_payload(e)), 503
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
            finally:
                if nb is not None:
                    try:
                        nb.close()
                    except Exception:
                        logger.debug(
                            "Failed closing notebook for %s %s",
                            request.method,
                            request.path,
                            exc_info=True,
                        )

        return wrapper

    return decorator
