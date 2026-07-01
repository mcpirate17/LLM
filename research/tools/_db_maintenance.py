from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from research.defaults import RUNS_DB

DEFAULT_WRITER_LOCK = Path(f"{RUNS_DB}.writer-lock")


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def connect_writer(db_path: Path) -> sqlite3.Connection:
    check_writer_lock(Path(f"{db_path}.writer-lock"))
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL, matching every other writer (and the aria-db native crate, which is
    # built around a single persistent WAL connection). journal_mode persists
    # per-DB, so a DELETE here would flip the whole DB back to writer<->reader
    # exclusion for the dashboard/API until the next WAL writer connects.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def check_writer_lock(lock_file: str | Path) -> None:
    path = Path(lock_file)
    if not path.exists():
        return
    try:
        pid_raw = path.read_text().strip()
        pid = int(pid_raw) if pid_raw.isdigit() else None
    except OSError:
        return
    if pid is None:
        return
    if os.path.exists(f"/proc/{pid}") and pid != os.getpid():
        raise RuntimeError(
            f"writer lock held by PID {pid}. Stop that writer or wait for "
            f"release before applying maintenance changes."
        )


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")]


def ensure_backup_table(
    conn: sqlite3.Connection,
    *,
    backup_table: str,
    source_table: str,
    extra_columns: Sequence[tuple[str, str]] = (),
    indexes: Sequence[tuple[str, Sequence[str]]] = (),
) -> list[str]:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {backup_table} AS "
        f"SELECT * FROM {source_table} WHERE 1=0"
    )
    existing = set(table_columns(conn, backup_table))
    for name, decl in extra_columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {backup_table} ADD COLUMN {name} {decl}")
            existing.add(name)
    for index_name, index_columns in indexes:
        column_sql = ", ".join(index_columns)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {backup_table}({column_sql})"
        )
    return list(existing)


def quoted_columns(columns: Iterable[str]) -> str:
    return ",".join(f'"{column}"' for column in columns)


def table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0] or 0) if row else 0
