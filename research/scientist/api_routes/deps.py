"""Shared dependency context for split scientist API route modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True, slots=True)
class ApiRouteContext:
    """Explicit dependency contract passed into each route registrar."""

    notebook_path: str
    dashboard_index_path: Callable[[], Optional[Path]]
    dashboard_missing_response: Callable[[], Any]
    is_asset_path: Callable[[str], bool]
    symbols: Mapping[str, Any]


def install_legacy_symbols(module_globals: dict, context: ApiRouteContext) -> None:
    """Install legacy symbols without blind overwrite of module globals.

    Route modules still rely on the monolith's helper functions/constants.
    This provides a controlled bridge while we continue de-monolithization.
    """

    module_globals["notebook_path"] = context.notebook_path
    module_globals["_dashboard_index_path"] = context.dashboard_index_path
    module_globals["_dashboard_missing_response"] = context.dashboard_missing_response
    module_globals["_is_asset_path"] = context.is_asset_path

    protected = {
        "notebook_path",
        "_dashboard_index_path",
        "_dashboard_missing_response",
        "_is_asset_path",
    }

    for name, value in context.symbols.items():
        if not name or name.startswith("__") or name in protected:
            continue
        if name not in module_globals:
            module_globals[name] = value
