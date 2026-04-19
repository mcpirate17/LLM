"""Shared dependency context for split scientist API route modules."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..notebook import LabNotebook


@dataclass(frozen=True, slots=True)
class ApiRouteContext:
    """Explicit dependency contract passed into each route registrar."""

    notebook_path: str
    dashboard_index_path: Callable[[], Optional[Path]]
    dashboard_missing_response: Callable[[], Any]
    is_asset_path: Callable[[str], bool]


def get_notebook(notebook_path: str, *, read_only: bool) -> LabNotebook:
    """Return a notebook handle for the current request.

    Every request gets its own short-lived notebook so the dashboard does
    not pin database handles longer than necessary. Read-only callers use
    a short-lived read-only sqlite connection; writable callers use a
    short-lived writable sqlite connection.
    """
    from ..notebook import LabNotebook

    try:
        return LabNotebook(
            notebook_path,
            check_same_thread=False,
            read_only=read_only,
            use_native=False,
        )
    except sqlite3.OperationalError:
        if read_only:
            raise
        import time

        logger.warning("Writable LabNotebook init hit db lock, retrying in 2s...")
        time.sleep(2)
        return LabNotebook(
            notebook_path,
            check_same_thread=False,
            read_only=False,
            use_native=False,
        )
