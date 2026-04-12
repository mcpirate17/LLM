"""Shared dependency context for split scientist API route modules."""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


from ..notebook import LabNotebook

logger = logging.getLogger(__name__)

# Process-wide shared notebook — avoids re-running _migrate() on every
# request and eliminates the "database is locked" errors that occur when
# per-request notebooks contend with the runner's writer thread.
_shared_notebooks: dict[str, LabNotebook] = {}
_shared_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class ApiRouteContext:
    """Explicit dependency contract passed into each route registrar."""

    notebook_path: str
    dashboard_index_path: Callable[[], Optional[Path]]
    dashboard_missing_response: Callable[[], Any]
    is_asset_path: Callable[[str], bool]


def get_notebook(notebook_path: str) -> LabNotebook:
    """Return a process-wide shared LabNotebook.

    A single LabNotebook (and its writer thread / SQLite connection) is
    reused across all requests for a given ``notebook_path``.  This avoids:
    - Running ``_migrate()`` on every request (DDL contention)
    - Opening/closing SQLite connections per request
    - "database is locked" errors from concurrent per-request migrations

    The notebook is never closed during the process lifetime — it stays
    alive for the Flask server's duration.
    """
    nb = _shared_notebooks.get(notebook_path)
    if nb is not None:
        return nb
    with _shared_lock:
        # Double-check after acquiring lock
        nb = _shared_notebooks.get(notebook_path)
        if nb is not None:
            return nb
        try:
            nb = LabNotebook(notebook_path, check_same_thread=False)
        except sqlite3.OperationalError:
            # Migration may hit a lock from the runner — retry once
            import time

            logger.warning("LabNotebook init hit db lock, retrying in 2s...")
            time.sleep(2)
            nb = LabNotebook(notebook_path, check_same_thread=False)
        _shared_notebooks[notebook_path] = nb
        return nb
