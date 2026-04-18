"""Native SQLite connection wrapper using aria_db (Rust/PyO3).

Drop-in replacement for ``_ThreadSafeConnectionWrapper`` that delegates
all database access to the Rust ``aria_db.ConnectionManager``.  This
guarantees:

1. A single WAL-mode connection that never closes (prevents SHM teardown).
2. Thread-safe reads and writes (Rust mutex, no Python threading bugs).
3. Serialized async writes via the Rust writer thread.

Usage::

    from .native_conn import NativeConnectionWrapper
    conn = NativeConnectionWrapper(db_path)
    # Same interface as _ThreadSafeConnectionWrapper / sqlite3.Connection:
    row = conn.execute("SELECT * FROM t WHERE id = ?", (1,)).fetchone()
    rows = conn.execute("SELECT * FROM t").fetchall()
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    conn.executescript("CREATE TABLE ...")
    conn.commit()
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional, Sequence

import aria_db

logger = logging.getLogger("research.scientist.notebook")


def _translate_error(exc: RuntimeError) -> sqlite3.OperationalError:
    """Convert Rust RuntimeError to sqlite3.OperationalError.

    The existing notebook code catches ``sqlite3.OperationalError`` in dozens
    of places (migrations, schema probes, etc.).  The Rust module raises
    ``RuntimeError`` for all SQLite errors.  This function re-wraps them so
    existing exception handlers work unchanged.
    """
    return sqlite3.OperationalError(str(exc))


class _NativeRow:
    """sqlite3.Row-compatible wrapper: supports both dict-style and index access.

    Matches sqlite3.Row behavior:
    - ``row["col"]`` → value by column name
    - ``row[0]`` → value by index
    - ``dict(row)`` → dict via keys() (Mapping protocol)
    - ``for val in row`` → iterate values (like sqlite3.Row)
    - ``row.keys()`` → column names
    """

    __slots__ = ("_data", "_keys")

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._keys = list(data.keys())

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._data[self._keys[key]]
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        # sqlite3.Row iterates over values, not keys.
        return iter(self._data.values())

    def keys(self):
        return self._keys

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __repr__(self) -> str:
        return f"_NativeRow({self._data})"


class _CursorResult:
    """Mimics a sqlite3.Cursor enough for chained .fetchone()/.fetchall() calls.

    The notebook code does:
        row = self.conn.execute("SELECT ...", params).fetchone()
        rows = self.conn.execute("SELECT ...", params).fetchall()
        for row in self.conn.execute("PRAGMA table_info(...)").fetchall(): ...

    This class holds the query results and serves them via fetchone/fetchall.
    """

    __slots__ = ("_rows", "_pos", "_description", "lastrowid", "rowcount")

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        lastrowid: int = 0,
        rowcount: int = -1,
    ):
        self._rows = rows or []
        self._pos = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchone(self) -> Optional[_NativeRow]:
        if self._pos >= len(self._rows):
            return None
        row = _NativeRow(self._rows[self._pos])
        self._pos += 1
        return row

    def fetchall(self) -> list[_NativeRow]:
        remaining = self._rows[self._pos :]
        self._pos = len(self._rows)
        return [_NativeRow(r) for r in remaining]

    def fetchmany(self, size: int = 1) -> list[_NativeRow]:
        end = min(self._pos + size, len(self._rows))
        batch = self._rows[self._pos : end]
        self._pos = end
        return [_NativeRow(r) for r in batch]

    def __iter__(self):
        for row in self._rows[self._pos :]:
            self._pos += 1
            yield _NativeRow(row)

    @property
    def description(self):
        if not self._rows:
            return None
        return tuple((k, None, None, None, None, None, None) for k in self._rows[0])


class NativeConnectionWrapper:
    """Drop-in replacement for _ThreadSafeConnectionWrapper.

    Uses aria_db.ConnectionManager (Rust/PyO3) for all database access.
    The native connection is managed as a process-wide singleton per db_path.

    When ``read_only=True``, the wrapper routes through
    ``aria_db.get_manager_readonly()`` which does **not** take the writer
    flock. Use this from tests, backfill audits, and admin tools that
    only query — it prevents the close-time WAL teardown that used to
    strand writer data on the long-running dashboard process.
    """

    __slots__ = ("_mgr", "_db_path", "_read_only")

    def __init__(self, db_path: str, *, read_only: bool = False):
        self._db_path = db_path
        self._read_only = bool(read_only)
        if self._read_only:
            self._mgr = aria_db.get_manager_readonly(db_path)
        else:
            self._mgr = aria_db.get_manager(db_path)

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> _CursorResult:
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()

        try:
            # For SELECT / PRAGMA queries, fetch results
            if sql_upper.startswith(("SELECT", "PRAGMA", "WITH", "EXPLAIN")):
                rows = self._mgr.fetchall(sql, tuple(parameters))
                return _CursorResult(rows)

            # For DML / DDL, just execute
            changed = self._mgr.execute(sql, tuple(parameters))
            return _CursorResult(rowcount=changed)
        except RuntimeError as exc:
            raise _translate_error(exc) from exc

    def executemany(
        self, sql: str, seq_of_parameters: Sequence[Sequence[Any]]
    ) -> _CursorResult:
        params_list = [tuple(p) for p in seq_of_parameters]
        try:
            changed = self._mgr.executemany(sql, params_list)
        except RuntimeError as exc:
            raise _translate_error(exc) from exc
        return _CursorResult(rowcount=changed)

    def executescript(self, sql_script: str) -> _CursorResult:
        try:
            self._mgr.executescript(sql_script)
        except RuntimeError as exc:
            raise _translate_error(exc) from exc
        return _CursorResult()

    def commit(self) -> None:
        self._mgr.commit()

    def rollback(self) -> None:
        # Rust connection uses autocommit — rollback is a no-op.
        pass

    def close(self) -> None:
        # Intentionally a no-op — the connection must never close.
        # The Rust singleton will live for the process lifetime.
        pass

    def cursor(self) -> "NativeConnectionWrapper":
        # Return self — all execute methods are on this object.
        return self

    @property
    def row_factory(self):
        # Compatibility stub — our rows already act like sqlite3.Row.
        return None

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        # Accept but ignore — we always return _NativeRow which supports
        # both dict-style and index access.
        pass

    @property
    def total_changes(self) -> int:
        row = self._mgr.fetchone("SELECT total_changes() AS tc", ())
        return row["tc"] if row else 0

    def __enter__(self) -> "NativeConnectionWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
