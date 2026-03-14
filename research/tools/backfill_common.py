"""Shared CLI and DB helpers for backfill tools."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from research.defaults import LAB_NOTEBOOK_DB, SQLITE_BUSY_TIMEOUT_MS


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _PROJECT_ROOT / LAB_NOTEBOOK_DB


def project_root() -> Path:
    return _PROJECT_ROOT


def default_lab_notebook_path() -> str:
    return str(Path(os.environ.get("LAB_NOTEBOOK_DB", str(_DEFAULT_DB))).expanduser())


def add_common_backfill_args(
    parser: argparse.ArgumentParser,
    *,
    include_db: bool = True,
    db_flag: str = "--db",
    db_dest: str = "db",
    include_device: bool = False,
    device_default: str = "cpu",
    include_dry_run: bool = False,
) -> None:
    """Attach shared backfill CLI arguments with consistent defaults."""
    if include_db and not _has_dest(parser, db_dest):
        parser.add_argument(db_flag, dest=db_dest, default=default_lab_notebook_path(), help="Path to lab_notebook.db")
    if include_device and not _has_dest(parser, "device"):
        parser.add_argument("--device", default=device_default, help=f"torch device (default: {device_default})")
    if include_dry_run and not _has_dest(parser, "dry_run"):
        parser.add_argument("--dry-run", action="store_true", help="Preview without writing")


def ensure_db_exists(db_path: str) -> str:
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    return str(path)


def open_sqlite(db_path: str, *, busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if busy_timeout_ms > 0:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


def _has_dest(parser: argparse.ArgumentParser, dest: str) -> bool:
    return any(action.dest == dest for action in parser._actions)
