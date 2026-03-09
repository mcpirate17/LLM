"""Shared dependency context for split scientist API route modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True, slots=True)
class ApiRouteContext:
    """Explicit dependency contract passed into each route registrar."""

    notebook_path: str
    dashboard_index_path: Callable[[], Optional[Path]]
    dashboard_missing_response: Callable[[], Any]
    is_asset_path: Callable[[str], bool]
