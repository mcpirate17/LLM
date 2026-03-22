"""Shared dependency context for split scientist API route modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from flask import g

from ..notebook import LabNotebook

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApiRouteContext:
    """Explicit dependency contract passed into each route registrar."""

    notebook_path: str
    dashboard_index_path: Callable[[], Optional[Path]]
    dashboard_missing_response: Callable[[], Any]
    is_asset_path: Callable[[str], bool]


def get_notebook(notebook_path: str) -> LabNotebook:
    """Return a request-scoped LabNotebook singleton.

    Uses Flask's ``g`` object so only one LabNotebook (and one writer
    thread / SQLite connection) exists per request, regardless of how
    many route helpers call this function.  Teardown is handled by
    ``register_notebook_teardown``.
    """
    nb = getattr(g, "_notebook", None)
    if nb is None:
        nb = LabNotebook(notebook_path)
        g._notebook = nb
    return nb


def register_notebook_teardown(app) -> None:
    """Register a teardown hook that closes the request-scoped notebook."""

    @app.teardown_appcontext
    def _close_notebook(exc=None):
        nb = getattr(g, "_notebook", None)
        if nb is not None:
            try:
                nb.close()
            except Exception:
                logger.debug("Error closing request-scoped notebook", exc_info=True)
            g._notebook = None
