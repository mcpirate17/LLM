"""Shared route utilities — decorators and helpers for API blueprints."""

from __future__ import annotations

import functools
import logging
import sqlite3
from contextlib import suppress
from typing import Any, Dict, Iterable, Sequence

from flask import jsonify, request

from .deps import get_notebook

logger = logging.getLogger(__name__)


def bind_view(handler, *bound_args):
    return functools.partial(handler, *bound_args) if bound_args else handler


def bind_notebook_view(wnb, handler, *bound_args):
    return wnb(bind_view(handler, *bound_args))


def register_routes(
    app,
    routes: Iterable[Sequence[Any]],
):
    """Register a table of pre-bound routes.

    Each route entry is ``(rule, endpoint, view[, methods])``.
    """
    for route in routes:
        if len(route) < 3 or len(route) > 4:
            raise ValueError("Each route must be (rule, endpoint, view[, methods])")
        rule, endpoint, view = route[:3]
        methods = route[3] if len(route) >= 4 else None
        kwargs = {"methods": list(methods)} if methods is not None else {}
        app.add_url_rule(rule, endpoint, view, **kwargs)


def register_notebook_routes(
    app,
    wnb,
    routes: Iterable[Sequence[Any]],
):
    """Register a table of notebook-backed routes.

    Each route entry is ``(rule, endpoint, handler[, methods][, bound_args])`` where
    ``methods`` is an iterable of HTTP verbs and ``bound_args`` is a tuple of extra
    positional arguments bound into the handler.
    """
    for route in routes:
        if len(route) < 3 or len(route) > 5:
            raise ValueError(
                "Each route must be (rule, endpoint, handler[, methods][, bound_args])"
            )
        rule, endpoint, handler = route[:3]
        methods = route[3] if len(route) >= 4 else None
        bound_args = route[4] if len(route) >= 5 else ()
        kwargs = {"methods": list(methods)} if methods is not None else {}
        app.add_url_rule(
            rule,
            endpoint,
            bind_notebook_view(wnb, handler, *tuple(bound_args or ())),
            **kwargs,
        )


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


def _db_error_response(exc: sqlite3.DatabaseError, phase: str):
    if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
        logger.warning(
            "%s %s -> 503 (db locked%s): %s",
            request.method,
            request.path,
            phase,
            exc,
        )
        return jsonify({"error": "Database temporarily busy, retry shortly"}), 503
    if is_malformed_db_error(exc):
        logger.warning(
            "%s %s -> 503 (db malformed%s): %s",
            request.method,
            request.path,
            phase,
            exc,
        )
        return jsonify(malformed_db_response_payload(exc)), 503
    logger.error(
        "Unhandled db error%s on %s %s: %s",
        phase,
        request.method,
        request.path,
        exc,
        exc_info=True,
    )
    return jsonify({"error": str(exc)}), 500


def with_notebook_context(notebook_path: str, *, read_only: bool | None = None):
    """Inject a shared ``nb`` kwarg and return standard JSON errors."""

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
            except sqlite3.DatabaseError as e:
                return _db_error_response(e, " during init")
            try:
                return fn(*args, nb=nb, **kwargs)
            except sqlite3.DatabaseError as e:
                return _db_error_response(e, "")
            except Exception as e:
                logger.error("Error in %s: %s", request.path, e, exc_info=True)
                return jsonify({"error": str(e)}), 500
            finally:
                if nb is not None:
                    with suppress(Exception):
                        nb.close()

        return wrapper

    return decorator
