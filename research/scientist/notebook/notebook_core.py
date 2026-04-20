from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import os
import queue
import sqlite3
from sqlite3 import connect as _original_sqlite3_connect
import subprocess
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .graph_features import build_graph_feature_rows
from .native_conn import NativeConnectionWrapper
from ._shared import (
    LOGGER,
    NOTEBOOK_SCHEMA,
    _PROGRAM_RESULTS_NEW_COLUMNS,
    infer_insight_identity,
)


class _SqliteConnectionAdapter:
    """Small patchable adapter over ``sqlite3.Connection``.

    Native notebook connections already provide thread-safe process-wide
    access. For the remaining legacy/in-memory sqlite paths we keep a minimal
    Python adapter so tests can patch ``execute`` and callers share one
    compatibility surface.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql, parameters=()):
        return self._conn.execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        return self._conn.executemany(sql, seq_of_parameters)

    def executescript(self, sql_script):
        return self._conn.executescript(sql_script)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def cursor(self):
        return self._conn.cursor()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn.row_factory = value

    @property
    def total_changes(self):
        return self._conn.total_changes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._conn.__exit__(*args)


_WAL_AUTOCHECKPOINT_PATCHED = False


def _install_wal_autocheckpoint_guard() -> None:
    """Monkey-patch sqlite3.connect to set wal_autocheckpoint=0 on all connections.

    When ANY connection to a WAL-mode database closes as the last reader, SQLite
    auto-checkpoints and deletes the -shm file.  With 30+ call sites across
    intelligence/, analytics/, and runner/ opening raw connections, we can't
    patch them all individually.  This global guard ensures every connection
    disables auto-checkpoint, preventing SHM teardown.
    """
    global _WAL_AUTOCHECKPOINT_PATCHED
    if _WAL_AUTOCHECKPOINT_PATCHED:
        return
    _WAL_AUTOCHECKPOINT_PATCHED = True

    def _guarded_connect(*args, **kwargs):
        conn = _original_sqlite3_connect(*args, **kwargs)
        try:
            conn.execute("PRAGMA wal_autocheckpoint=0")
        except Exception:
            pass  # non-WAL databases, :memory:, etc.
        return conn

    sqlite3.connect = _guarded_connect


class _NotebookCore:
    """Core operations for the Lab Notebook."""

    """Electronic lab notebook for the AI scientist."""
    __slots__ = ()

    _cached_code_version: Optional[str] = None
    _last_report_snapshot_cleanup_at: float = 0.0
    _schema_bootstrapped_paths: set[str] = set()

    @staticmethod
    def _configure_sqlite_connection(
        conn: sqlite3.Connection,
        *,
        db_path: Path,
        role: str,
    ) -> None:
        """Apply connection pragmas.

        WAL mode is deliberately not used.  With 30+ code paths opening raw
        ``sqlite3.connect()`` calls (intelligence, analytics, runner helpers),
        any connection closing as the last WAL reader triggers an auto-checkpoint
        that deletes the ``-shm`` file.  This corrupts the process-wide SQLite
        mmap state, causing ``SQLITE_IOERR_SHORT_READ`` (ext 522) on every
        subsequent connection for the rest of the process lifetime.  DELETE
        journal mode avoids SHM/mmap entirely.
        """

        def _run_pragma(sql: str, *, desc: str, required: bool = False) -> None:
            last_error: Optional[sqlite3.OperationalError] = None
            for delay_s in (0.0, 0.2, 0.5):
                if delay_s:
                    time.sleep(delay_s)
                try:
                    conn.execute(sql)
                    last_error = None
                    return
                except sqlite3.OperationalError as exc:
                    last_error = exc
                    ext_code = getattr(exc, "sqlite_errorcode", None)
                    LOGGER.warning(
                        "Pragma %s failed: %s (extended_errcode=%s, db=%s, role=%s)",
                        desc,
                        exc,
                        ext_code,
                        db_path,
                        role,
                    )
                    if "disk i/o error" not in str(exc).lower():
                        raise
            if last_error is None:
                return
            if required:
                raise last_error
            LOGGER.warning(
                "Failed to apply %s for %s connection to %s; continuing: %s",
                desc,
                role,
                db_path,
                last_error,
            )

        _run_pragma("PRAGMA foreign_keys=ON", desc="foreign_keys")
        _run_pragma("PRAGMA synchronous=NORMAL", desc="synchronous mode")
        _run_pragma("PRAGMA busy_timeout=15000", desc="busy timeout")

    @staticmethod
    def validate_db_path_arg(db_path: str | PathLike[str]) -> str | PathLike[str]:
        """Reject obviously bogus notebook paths before any filesystem side effects.

        Test doubles and sentinel strings should fail fast instead of creating
        files like ``<MagicMock name='mock.db_path' ...>`` or ``:memory:`` under
        the repo root.
        """
        if not isinstance(db_path, (str, os.PathLike)):
            raise TypeError(
                f"db_path must be str or os.PathLike, got {type(db_path).__name__}"
            )

        raw = os.fspath(db_path).strip()
        if not raw:
            raise ValueError("db_path must not be empty")
        if raw == ":memory:":
            return raw
        if raw.startswith("<MagicMock ") or "MagicMock name='mock.db_path'" in raw:
            raise TypeError(f"db_path must be a real filesystem path, got {raw!r}")
        return db_path

    @staticmethod
    def resolve_db_path(db_path: str | PathLike[str]) -> Path:
        """Resolve a database path to its absolute path, handling nested research/ cases.

        Ensures that if we are currently inside the research/ directory,
        a path like 'research/lab_notebook.db' refers to the one in the parent.
        """
        db_path = _NotebookCore.validate_db_path_arg(db_path)
        if isinstance(db_path, str) and db_path == ":memory:":
            return Path(":memory:")
        path = Path(db_path)
        if not path.is_absolute():
            # If we are in /some/path/LLM/research and db_path is 'research/lab_notebook.db'
            # then path.resolve() would be /some/path/LLM/research/research/lab_notebook.db.
            # We want /some/path/LLM/research/lab_notebook.db.
            cwd = Path.cwd()
            if cwd.name == "research" and path.parts and path.parts[0] == "research":
                # db_path starts with 'research/' and we are already in research/
                # assume the user meant the parent's research/ directory
                return (cwd.parent / db_path).absolute()
        return path.resolve()

    # Track whether migration has already run for a given DB path this process.
    # Once the schema is migrated once, subsequent connections can skip it.
    _migrated_paths: set[str] = set()

    def __init__(
        self,
        db_path: str | PathLike[str] = "research/lab_notebook.db",
        *,
        skip_migrate: bool = False,
        check_same_thread: bool = False,
        read_only: bool = False,
        use_native: bool = True,
    ):
        """Open the notebook.

        ``read_only=True`` opens via the aria-db read-only manager. Use
        this from tests, audit scripts, and admin tools that only query —
        it skips the writer flock and cannot trigger the close-time WAL
        teardown that previously stranded writer data on the long-running
        dashboard process. Write attempts from a read-only notebook will
        raise at the aria-db layer instead of silently dropping.

        ``use_native=False`` forces the legacy Python sqlite3 wrapper. This
        is useful for short-lived writable admin/API requests that must
        release the writer lock on close instead of pinning the process-wide
        aria-db writer manager for the lifetime of the process.
        """
        _install_wal_autocheckpoint_guard()
        self.db_path = self.resolve_db_path(db_path)
        self._is_memory = ":memory:" in str(self.db_path)
        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = bool(read_only)
        self._use_native = bool(use_native) and not self._is_memory

        if self._is_memory:
            # In-memory DBs can't use the native manager (no file path).
            # Fall back to the old Python wrapper for tests.
            self.conn = self._open_sqlite_connection(
                db_path=self.db_path,
                read_only=False,
                role="memory",
                check_same_thread=False,
                connect_target=":memory:",
            )
            self._use_native = False
        elif self._use_native:
            self.conn = NativeConnectionWrapper(
                str(self.db_path), read_only=self._read_only
            )
        else:
            self.conn = self._open_sqlite_connection(
                db_path=self.db_path,
                read_only=self._read_only,
                role="readonly" if self._read_only else "api-write",
                check_same_thread=check_same_thread,
            )

        self._batch_depth = 0
        self._program_results_columns: Optional[set[str]] = None
        self._leaderboard_columns: Optional[set[str]] = None
        self._dashboard_summary_cache: Dict[tuple[bool, bool], Dict[str, Any]] = {}
        self._dashboard_summary_cache_expires_at: float = 0.0
        self._template_observability_cache: Dict[int, Dict[str, Any]] = {}
        self._template_observability_cache_expires_at: float = 0.0
        self._writer_thread_started = False
        db_key = str(self.db_path)
        cls = type(self)
        if not self._read_only:
            self._ensure_schema_bootstrap(db_key=db_key)
        if (
            not skip_migrate
            and not self._read_only
            and db_key not in cls._migrated_paths
        ):
            self._migrate()
            cls._migrated_paths.add(db_key)

        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None

    def _has_core_schema(self) -> bool:
        try:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'experiments'"
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError as exc:
            LOGGER.warning("Core schema probe failed for %s: %s", self.db_path, exc)
            return False

    def _ensure_schema_bootstrap(self, *, db_key: str) -> None:
        cls = type(self)
        if self._is_memory:
            self.conn.executescript(NOTEBOOK_SCHEMA)
            self._maybe_commit()
            cls._schema_bootstrapped_paths.add(db_key)
            return
        if db_key in cls._schema_bootstrapped_paths:
            return
        if self._has_core_schema():
            cls._schema_bootstrapped_paths.add(db_key)
            return
        try:
            self.conn.executescript(NOTEBOOK_SCHEMA)
            self._maybe_commit()
            cls._schema_bootstrapped_paths.add(db_key)
        except sqlite3.OperationalError as exc:
            if self._has_core_schema():
                LOGGER.warning(
                    "Schema bootstrap failed for %s but core schema exists; continuing: %s",
                    self.db_path,
                    exc,
                )
                cls._schema_bootstrapped_paths.add(db_key)
                return
            raise

    @classmethod
    def _open_sqlite_connection(
        cls,
        *,
        db_path: Path,
        read_only: bool,
        role: str,
        check_same_thread: bool,
        connect_target: str | None = None,
    ) -> _SqliteConnectionAdapter:
        target = connect_target or str(db_path)
        connect_kwargs: Dict[str, Any] = {
            "timeout": 10.0,
            "check_same_thread": check_same_thread,
        }
        if read_only and connect_target is None:
            target = f"file:{db_path}?mode=ro"
            connect_kwargs["uri"] = True
        raw_conn = sqlite3.connect(target, **connect_kwargs)
        if target == ":memory:":
            raw_conn.execute("PRAGMA foreign_keys=ON")
        else:
            cls._configure_sqlite_connection(
                raw_conn,
                db_path=db_path,
                role=role,
            )
        raw_conn.row_factory = sqlite3.Row
        return _SqliteConnectionAdapter(raw_conn)

    def _writer_loop(self):
        """Background thread that handles all database writes."""
        # Use a separate connection for the writer thread
        writer_conn = self._open_sqlite_connection(
            db_path=self.db_path,
            read_only=False,
            role="writer",
            check_same_thread=False,
        )

        batch = []

        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                item = self._write_queue.get(timeout=0.1)
                if item is None:  # Sentinel
                    break

                sql, params = item
                if sql == "__flush__":
                    # Flush request: commit pending batch and signal caller
                    if batch:
                        writer_conn.commit()
                        batch = []
                    params.set()  # params is a threading.Event
                    continue
                if (
                    isinstance(params, list)
                    and params
                    and isinstance(params[0], (list, tuple))
                ):
                    writer_conn.executemany(sql, params)
                else:
                    writer_conn.execute(sql, params)
                batch.append(item)

                # Commit after every write to avoid holding the exclusive lock.
                # In DELETE journal mode, an uncommitted transaction blocks all
                # readers on other connections, causing deadlocks with the
                # screening thread's concurrent notebook queries.
                writer_conn.commit()
                batch = []

            except queue.Empty:
                if batch:
                    writer_conn.commit()
                    batch = []
                continue
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # Retry with exponential backoff — the main thread may be
                    # holding a write lock (WAL allows only one writer).
                    _recovered = False
                    for _retry in range(5):
                        time.sleep(0.05 * (2**_retry))  # 50ms → 800ms
                        try:
                            # Re-execute the statement that failed
                            if item not in batch:
                                if (
                                    isinstance(params, list)
                                    and params
                                    and isinstance(params[0], (list, tuple))
                                ):
                                    writer_conn.executemany(sql, params)
                                else:
                                    writer_conn.execute(sql, params)
                                batch.append(item)
                            # Commit the pending batch
                            writer_conn.commit()
                            batch = []
                            _recovered = True
                            break
                        except sqlite3.OperationalError:
                            continue
                    if not _recovered:
                        LOGGER.warning(
                            "LabNotebook async writer: db locked after 5 retries, "
                            "re-queuing %s",
                            sql[:60],
                        )
                        self._write_queue.put(item)
                else:
                    LOGGER.error("LabNotebook async writer error: %s", e)
            except Exception as e:
                LOGGER.error("LabNotebook async writer error: %s", e)

        if batch:
            writer_conn.commit()
        writer_conn.close()

    def _ensure_writer_thread(self) -> None:
        if self._writer_thread_started:
            return
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._writer_thread_started = True

    def _submit_write(self, sql: str, params: Any):
        """Submit a write task to the background queue."""
        self._invalidate_dashboard_summary_cache()
        if getattr(self, "_use_native", False):
            # Delegate to the Rust writer thread.
            mgr = self.conn._mgr
            if (
                isinstance(params, list)
                and params
                and isinstance(params[0], (list, tuple))
            ):
                mgr.submit_write_many(sql, params)
            else:
                mgr.submit_write(sql, tuple(params) if params else ())
            return
        self._ensure_writer_thread()
        self._write_queue.put((sql, params))

    def _store_graph_features_async(
        self,
        *,
        result_id: str,
        graph_fingerprint: str,
        graph_json: str,
    ) -> None:
        if not result_id or not isinstance(graph_json, str) or not graph_json.strip():
            return
        rows = build_graph_feature_rows(
            result_id=result_id,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
        )
        self._submit_write(
            "DELETE FROM program_graph_pairs WHERE result_id = ?",
            (result_id,),
        )
        self._submit_write(
            "DELETE FROM program_graph_ops WHERE result_id = ?",
            (result_id,),
        )
        self._submit_write(
            """INSERT OR REPLACE INTO program_graph_features
               (result_id, graph_fingerprint, template_name, templates_json, motifs_json,
                slot_usage_json, op_count, pair_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows["feature_row"],
        )
        if rows["op_rows"]:
            self._submit_write(
                """INSERT OR REPLACE INTO program_graph_ops
                   (result_id, graph_fingerprint, op_name)
                   VALUES (?, ?, ?)""",
                rows["op_rows"],
            )
        if rows["pair_rows"]:
            self._submit_write(
                """INSERT OR REPLACE INTO program_graph_pairs
                   (result_id, graph_fingerprint, signature)
                   VALUES (?, ?, ?)""",
                rows["pair_rows"],
            )

    def _backfill_missing_graph_features(
        self,
        *,
        result_ids: Optional[Iterable[str]] = None,
        limit: int = 2000,
    ) -> int:
        where_clause = (
            "pr.graph_json IS NOT NULL AND TRIM(CAST(pr.graph_json AS TEXT)) <> ''"
        )
        params: list[Any] = []
        if result_ids is not None:
            requested = [str(item).strip() for item in result_ids if str(item).strip()]
            if not requested:
                return 0
            placeholders = ",".join("?" for _ in requested)
            where_clause += f" AND pr.result_id IN ({placeholders})"
            params.extend(requested)
        query = f"""
            SELECT pr.result_id, COALESCE(pr.graph_fingerprint, '') AS graph_fingerprint, pr.graph_json
            FROM program_results pr
            LEFT JOIN program_graph_features gf ON gf.result_id = pr.result_id
            WHERE gf.result_id IS NULL
              AND {where_clause}
            LIMIT ?
        """
        params.append(int(limit))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        if not rows:
            return 0

        feature_rows = []
        op_rows = []
        pair_rows = []
        result_keys = []
        for row in rows:
            result_id = str(row["result_id"] or "").strip()
            if not result_id:
                continue
            built = build_graph_feature_rows(
                result_id=result_id,
                graph_fingerprint=str(row["graph_fingerprint"] or ""),
                graph_json=str(row["graph_json"] or ""),
            )
            result_keys.append((result_id,))
            feature_rows.append(built["feature_row"])
            op_rows.extend(built["op_rows"])
            pair_rows.extend(built["pair_rows"])
        if not feature_rows:
            return 0

        with self.batch():
            self.conn.executemany(
                "DELETE FROM program_graph_pairs WHERE result_id = ?",
                result_keys,
            )
            self.conn.executemany(
                "DELETE FROM program_graph_ops WHERE result_id = ?",
                result_keys,
            )
            self.conn.executemany(
                """INSERT OR REPLACE INTO program_graph_features
                   (result_id, graph_fingerprint, template_name, templates_json, motifs_json,
                    slot_usage_json, op_count, pair_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                feature_rows,
            )
            if op_rows:
                self.conn.executemany(
                    """INSERT OR REPLACE INTO program_graph_ops
                       (result_id, graph_fingerprint, op_name)
                       VALUES (?, ?, ?)""",
                    op_rows,
                )
            if pair_rows:
                self.conn.executemany(
                    """INSERT OR REPLACE INTO program_graph_pairs
                       (result_id, graph_fingerprint, signature)
                       VALUES (?, ?, ?)""",
                    pair_rows,
                )
        return len(feature_rows)

    def _ensure_graph_features(
        self,
        *,
        result_ids: Optional[Iterable[str]] = None,
        batch_limit: int = 2000,
    ) -> None:
        while self._backfill_missing_graph_features(
            result_ids=result_ids,
            limit=batch_limit,
        ):
            if result_ids is not None:
                break

    def flush_writes(self, timeout: float = 5.0):
        """Block until the async write queue is drained and committed.

        Useful in tests and any code that writes via ``_submit_write`` then
        immediately reads back via the main ``self.conn``.
        """
        if self._read_only:
            return
        if getattr(self, "_use_native", False):
            self.conn._mgr.flush_writes(timeout)
            return
        if not self._writer_thread_started:
            return
        # Put a sentinel-like marker and wait for drain
        flush_event = threading.Event()
        self._write_queue.put(("__flush__", flush_event))
        flush_event.wait(timeout=timeout)
        # Refresh the reader connection so subsequent reads observe the
        # writer thread's committed WAL snapshot immediately.
        self.conn.commit()

    def _migrate(self):
        """Add any missing columns to existing databases.

        Retries up to 3 times on ``OperationalError: database is locked``
        since migration runs DDL that requires a write lock, and the runner's
        writer thread may be holding it.
        """
        for attempt in range(3):
            try:
                self._migrate_impl()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < 2:
                    LOGGER.warning(
                        "Migration attempt %d hit db lock, retrying in %ds...",
                        attempt + 1,
                        2 * (attempt + 1),
                    )
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

    # -- Per-table migration helpers --

    def _migrate_experiments_table(self) -> None:
        try:
            self.conn.execute("SELECT llm_analysis FROM experiments LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN llm_analysis TEXT")
            self._maybe_commit()
        exp_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "campaign_id" not in exp_cols:
            self.conn.execute("ALTER TABLE experiments ADD COLUMN campaign_id TEXT")
        if "preregistration_id" not in exp_cols:
            self.conn.execute(
                "ALTER TABLE experiments ADD COLUMN preregistration_id TEXT"
            )

    def _migrate_program_results_table(self) -> None:
        existing = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(program_results)"
            ).fetchall()
        }
        for col_name, col_type in _PROGRAM_RESULTS_NEW_COLUMNS.items():
            if col_name not in existing:
                try:
                    self.conn.execute(
                        f"ALTER TABLE program_results ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass
        if "arch_spec_json" not in existing:
            self.conn.execute(
                "ALTER TABLE program_results ADD COLUMN arch_spec_json TEXT"
            )
        if "model_source" not in existing:
            self.conn.execute(
                "ALTER TABLE program_results ADD COLUMN model_source TEXT"
            )

    def _migrate_leaderboard_create(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                entry_id TEXT PRIMARY KEY,
                result_id TEXT REFERENCES program_results(result_id),
                timestamp REAL NOT NULL,
                model_source TEXT NOT NULL,
                architecture_desc TEXT,
                screening_loss_ratio REAL,
                screening_novelty REAL,
                screening_passed INTEGER DEFAULT 0,
                investigation_loss_ratio REAL,
                investigation_robustness REAL,
                investigation_best_training TEXT,
                investigation_passed INTEGER DEFAULT 0,
                validation_loss_ratio REAL,
                validation_baseline_ratio REAL,
                validation_multi_seed_std REAL,
                validation_passed INTEGER DEFAULT 0,
                composite_score REAL,
                tier TEXT DEFAULT 'screening',
                tags TEXT,
                notes TEXT,
                is_reference INTEGER DEFAULT 0,
                reference_name TEXT DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_leaderboard_tier ON leaderboard(tier);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(composite_score);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_result ON leaderboard(result_id);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_model_source ON leaderboard(model_source);
        """)
        self.conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_programs_stage1_passed ON program_results(stage1_passed);
            CREATE INDEX IF NOT EXISTS idx_programs_graph_fingerprint ON program_results(graph_fingerprint);
            CREATE INDEX IF NOT EXISTS idx_programs_routing_mode ON program_results(routing_mode);
            CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON program_results(timestamp);
            CREATE INDEX IF NOT EXISTS idx_programs_exp_ts ON program_results(experiment_id, timestamp);
        """)

    def _migrate_decisions_table(self) -> None:
        try:
            decision_cols = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(decisions)").fetchall()
            }
        except sqlite3.OperationalError:
            decision_cols = set()
        if "evidence_pack_json" not in decision_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE decisions ADD COLUMN evidence_pack_json TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def _migrate_op_success_rates_table(self) -> None:
        osr_cols = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(op_success_rates)"
            ).fetchall()
        }
        if "avg_novelty_confidence" not in osr_cols:
            self.conn.execute(
                "ALTER TABLE op_success_rates ADD COLUMN avg_novelty_confidence REAL"
            )

    def _migrate_orphan_preregistration_repair(self) -> None:
        try:
            orphan_prereg_rows = self.conn.execute("""
                SELECT preregistration_id
                FROM hypothesis_preregistrations hp
                WHERE hp.experiment_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM experiments e WHERE e.experiment_id = hp.experiment_id
                  )
            """).fetchall()
        except sqlite3.OperationalError:
            orphan_prereg_rows = []
        if orphan_prereg_rows:
            orphan_ids = [str(row[0]) for row in orphan_prereg_rows]
            ph = ",".join("?" for _ in orphan_ids)
            self.conn.execute(
                f"""
                UPDATE hypothesis_preregistrations
                SET experiment_id = NULL,
                    status = CASE WHEN status = 'linked' THEN 'registered' ELSE status END,
                    notes = TRIM(COALESCE(notes || '\n', '') || ?)
                WHERE preregistration_id IN ({ph})
                """,
                (
                    "Auto-repaired orphaned experiment link during notebook startup.",
                    *orphan_ids,
                ),
            )
            LOGGER.warning(
                "Repaired %d orphaned hypothesis_preregistrations rows", len(orphan_ids)
            )

    def _migrate_training_curves_fk(self) -> None:
        tc_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'training_curves'"
        ).fetchone()
        tc_sql = str(tc_row[0] or "") if tc_row else ""
        if "REFERENCES program_results" not in tc_sql:
            self.conn.execute(
                "DELETE FROM training_curves "
                "WHERE NOT EXISTS (SELECT 1 FROM program_results pr WHERE pr.result_id = training_curves.result_id)"
            )
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_curves_new (
                    result_id TEXT NOT NULL REFERENCES program_results(result_id) ON DELETE CASCADE,
                    step INTEGER NOT NULL, loss REAL, grad_norm REAL, step_time_ms REAL,
                    PRIMARY KEY (result_id, step)
                )
            """)
            self.conn.execute("""
                INSERT OR REPLACE INTO training_curves_new (result_id, step, loss, grad_norm, step_time_ms)
                SELECT tc.result_id, tc.step, tc.loss, tc.grad_norm, tc.step_time_ms
                FROM training_curves tc JOIN program_results pr ON pr.result_id = tc.result_id
            """)
            self.conn.execute("DROP TABLE training_curves")
            self.conn.execute(
                "ALTER TABLE training_curves_new RENAME TO training_curves"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_curves_result ON training_curves(result_id)"
            )

    def _migrate_hypotheses_table(self) -> None:
        hyp_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(hypotheses)").fetchall()
        }
        if "metadata_json" not in hyp_cols:
            self.conn.execute("ALTER TABLE hypotheses ADD COLUMN metadata_json TEXT")

    def _migrate_campaigns_table(self) -> None:
        camp_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(campaigns)").fetchall()
        }
        if "completion_reason" not in camp_cols:
            self.conn.execute("ALTER TABLE campaigns ADD COLUMN completion_reason TEXT")
        if "successor_campaign_id" not in camp_cols:
            self.conn.execute(
                "ALTER TABLE campaigns ADD COLUMN successor_campaign_id TEXT"
            )

    def _migrate_insights_semantic(self) -> None:
        insight_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(insights)").fetchall()
        }
        if "insight_type" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN insight_type TEXT")
        if "subject_key" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN subject_key TEXT")
        if "semantic_key" not in insight_cols:
            self.conn.execute("ALTER TABLE insights ADD COLUMN semantic_key TEXT")

        rows = self.conn.execute(
            """SELECT insight_id, category, content, insight_type, subject_key, semantic_key
               FROM insights"""
        ).fetchall()
        for row in rows:
            existing_type = (
                str(row["insight_type"] or "").strip()
                if isinstance(row, sqlite3.Row)
                else str(row[3] or "").strip()
            )
            existing_subject = (
                str(row["subject_key"] or "").strip()
                if isinstance(row, sqlite3.Row)
                else str(row[4] or "").strip()
            )
            existing_semantic = (
                str(row["semantic_key"] or "").strip()
                if isinstance(row, sqlite3.Row)
                else str(row[5] or "").strip()
            )
            if existing_type and existing_subject and existing_semantic:
                continue
            category = row["category"] if isinstance(row, sqlite3.Row) else row[1]
            content = row["content"] if isinstance(row, sqlite3.Row) else row[2]
            inferred_type, inferred_subject, inferred_semantic = infer_insight_identity(
                str(category or ""),
                str(content or ""),
            )
            self.conn.execute(
                """UPDATE insights
                   SET insight_type = COALESCE(NULLIF(insight_type, ''), ?),
                       subject_key = COALESCE(NULLIF(subject_key, ''), ?),
                       semantic_key = COALESCE(NULLIF(semantic_key, ''), ?)
                   WHERE insight_id = ?""",
                (
                    inferred_type,
                    inferred_subject,
                    inferred_semantic,
                    row["insight_id"] if isinstance(row, sqlite3.Row) else row[0],
                ),
            )

        self._supersede_active_semantic_duplicates()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_semantic_key ON insights(semantic_key)"
        )
        try:
            self.conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_insights_active_semantic_unique
                   ON insights(semantic_key)
                   WHERE status = 'active' AND semantic_key IS NOT NULL AND semantic_key != ''"""
            )
        except sqlite3.IntegrityError:
            self._supersede_active_semantic_duplicates()
            self.conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_insights_active_semantic_unique
                   ON insights(semantic_key)
                   WHERE status = 'active' AND semantic_key IS NOT NULL AND semantic_key != ''"""
            )

    def _supersede_active_semantic_duplicates(self) -> None:
        active_rows = self.conn.execute(
            """SELECT insight_id, semantic_key
               FROM insights
               WHERE status = 'active' AND semantic_key IS NOT NULL AND semantic_key != ''
               ORDER BY confidence DESC, timestamp DESC"""
        ).fetchall()
        seen_semantic: set[str] = set()
        for row in active_rows:
            sem = str(row["semantic_key"] if isinstance(row, sqlite3.Row) else row[1])
            insight_id = row["insight_id"] if isinstance(row, sqlite3.Row) else row[0]
            if sem in seen_semantic:
                self.conn.execute(
                    "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
                    (insight_id,),
                )
                continue
            seen_semantic.add(sem)

    def _migrate_insights_bayesian(self) -> None:
        insight_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(insights)").fetchall()
        }
        for col_name, col_def in (
            ("alpha", "REAL DEFAULT 1.0"),
            ("beta_", "REAL DEFAULT 1.0"),
            ("display_only", "INTEGER DEFAULT 0"),
            ("insight_level", "TEXT DEFAULT 'op'"),
            ("n_predictions", "INTEGER DEFAULT 0"),
            ("n_correct", "INTEGER DEFAULT 0"),
            ("evidence_json", "TEXT"),
        ):
            if col_name not in insight_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE insights ADD COLUMN {col_name} {col_def}"
                    )
                except sqlite3.OperationalError:
                    pass
        self.conn.execute(
            "UPDATE insights SET display_only = 1 WHERE category = 'failure_mode' AND display_only = 0"
        )

    def _migrate_leaderboard_columns(self) -> None:
        lb_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
        }
        for col in (
            "normalized_baseline_ratio REAL",
            "param_efficiency REAL",
            "quant_int8_retention REAL",
            "quant_quality_per_byte REAL",
            "robustness_long_ctx_score REAL",
            "robustness_noise_score REAL",
            "init_sensitivity_std REAL",
            "fp_jacobian_spectral_norm REAL",
            "scaling_param_efficiency REAL",
            "scaling_flop_efficiency REAL",
            "scaling_gate_passed INTEGER",
            "scaling_best_family TEXT",
            "scaling_d512_param_efficiency REAL",
            "scaling_confidence TEXT",
            "campaign_id TEXT",
            "is_pinned INTEGER DEFAULT 0",
            "routing_savings_ratio REAL",
            "compression_ratio REAL",
            "activation_sparsity_score REAL",
            "dead_neuron_ratio REAL",
            "routing_collapse_score REAL",
            "wikitext_perplexity REAL",
            "wikitext_score REAL",
            "wikitext_pre_perplexity REAL",
            "wikitext_ppl_improvement REAL",
            "screening_wikitext_status TEXT",
            "screening_wikitext_metric_version TEXT",
            "screening_wikitext_variant TEXT",
            "screening_wikitext_elapsed_ms REAL",
            "screening_wikitext_budget_json TEXT",
            "tinystories_perplexity REAL",
            "tinystories_score REAL",
            "cross_task_score REAL",
            "efficiency_wall_score REAL",
            "max_viable_seq_len INTEGER",
            "scaling_regime TEXT",
            "discovery_loss_ratio REAL",
            "pre_inv_score REAL",
            "ncd_score REAL",
            "robustness_long_ctx_scaling_score REAL",
            "robustness_long_ctx_assoc_score REAL",
            "robustness_long_ctx_multi_hop_score REAL",
            "robustness_long_ctx_passkey_score REAL",
            "robustness_long_ctx_retrieval_aggregate REAL",
            "robustness_long_ctx_combined_score REAL",
            "depth_savings_ratio REAL",
            "recursion_savings_ratio REAL",
            "activation_sparsity_score REAL",
            "routing_expert_count INTEGER",
            "routing_confidence_mean REAL",
            "routing_drop_rate REAL",
            "efficiency_multiple REAL",
            "robustness_grade TEXT",
            "evaluation_stage TEXT",
            "eval_budget_steps INTEGER",
            "capability_tier TEXT",
            "wikitext_ppl_improvement_ratio REAL",
            "peak_ppl REAL",
            "peak_step INTEGER",
            "steps_to_divergence INTEGER",
            "ppl_500 REAL",
            "reinvestigation_count INTEGER DEFAULT 0",
            "n_routing_ops INTEGER",
            "n_sparse_ops INTEGER",
            "n_moe_ops INTEGER",
            "replication_n INTEGER",
            "replication_loss_mean REAL",
            "replication_loss_std REAL",
            "replication_best_vs_mean_gap REAL",
            "hellaswag_acc REAL",
            "ar_auc REAL",
            "induction_auc REAL",
            "binding_auc REAL",
            "binding_composite REAL",
            "induction_v2_investigation_auc REAL",
            "induction_v2_investigation_max_gap_acc REAL",
            "induction_v2_investigation_protocol_version TEXT",
            "binding_v2_investigation_auc REAL",
            "binding_v2_investigation_max_distance_acc REAL",
            "binding_v2_investigation_protocol_version TEXT",
            "local_only INTEGER DEFAULT 0",
            "result_cohort TEXT",
            "trust_label TEXT",
            "comparability_label TEXT",
            "evaluation_protocol_version TEXT",
            "scoring_version TEXT",
        ):
            col_name = col.split()[0]
            if col_name not in lb_cols:
                try:
                    self.conn.execute(f"ALTER TABLE leaderboard ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
        if "is_reference" not in lb_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE leaderboard ADD COLUMN is_reference INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
        if "reference_name" not in lb_cols:
            try:
                self.conn.execute(
                    "ALTER TABLE leaderboard ADD COLUMN reference_name TEXT DEFAULT NULL"
                )
            except sqlite3.OperationalError:
                pass

    def _migrate_impl(self):
        """Internal migration logic — delegates to per-table helpers."""
        self._migrate_experiments_table()
        self._migrate_program_results_table()
        self._migrate_leaderboard_create()
        self._migrate_decisions_table()
        self._migrate_op_success_rates_table()
        self._migrate_orphan_preregistration_repair()
        self._migrate_training_curves_fk()
        self._migrate_hypotheses_table()
        self._migrate_campaigns_table()
        self._migrate_insights_semantic()
        self._migrate_insights_bayesian()
        self._migrate_leaderboard_columns()
        self._migrate_program_results_dedup_index()
        self._migrate_program_results_intentional_rerun_column()
        self._migrate_program_results_dedup_trigger()
        self._migrate_leaderboard_fp_dedup()
        self._program_results_columns = None
        self._leaderboard_columns = None
        self._maybe_commit()

    def _migrate_program_results_intentional_rerun_column(self) -> None:
        """Slice 4: track callers that intentionally re-evaluate a known graph."""
        cols = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(program_results)"
            ).fetchall()
        }
        if "intentional_rerun_reason" not in cols:
            try:
                self.conn.execute(
                    "ALTER TABLE program_results "
                    "ADD COLUMN intentional_rerun_reason TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def _migrate_program_results_dedup_trigger(self) -> None:
        """Slice 4 schema backstop for the cross-experiment dedup gate.

        Mirrors the application-level check in ``record_program_result``: any
        INSERT of a row whose ``graph_fingerprint`` already exists must set
        ``intentional_rerun_reason`` (replay, validation_promotion,
        reference_registration, etc.). The trigger only fires on INSERT, so
        the historical 1.5k cross-experiment duplicates are grandfathered
        until a separate cleanup runs.
        """
        try:
            existing = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name=?",
                ("reject_dup_fingerprint_no_reason",),
            ).fetchone()
            if existing:
                return
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS reject_dup_fingerprint_no_reason
                BEFORE INSERT ON program_results
                WHEN NEW.graph_fingerprint IS NOT NULL
                 AND TRIM(NEW.graph_fingerprint) <> ''
                 AND NEW.intentional_rerun_reason IS NULL
                 AND EXISTS (
                       SELECT 1 FROM program_results
                       WHERE graph_fingerprint = NEW.graph_fingerprint
                         AND result_id <> NEW.result_id
                     )
                BEGIN
                  SELECT RAISE(
                    ABORT,
                    'duplicate graph_fingerprint without intentional_rerun_reason'
                  );
                END
                """
            )
            LOGGER.info(
                "Installed reject_dup_fingerprint_no_reason trigger — cross-experiment fingerprint dedup is now enforced at the schema level."
            )
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "reject_dup_fingerprint_no_reason migration skipped: %s", exc
            )

    def _migrate_program_results_dedup_index(self) -> None:
        """Slice 3a: enforce per-experiment fingerprint uniqueness.

        The index forbids two rows in ``program_results`` from sharing the
        same ``(graph_fingerprint, experiment_id)`` pair. Cross-experiment
        re-runs (replay, validation, backfill, references) are still allowed
        because they have different ``experiment_id``.

        We probe for existing duplicates first. If any are found we log a
        loud warning and skip the index — the dashboard must keep booting,
        but the duplicates need to be cleaned up via
        ``research/tools/dedup_within_experiment.py --apply`` before the
        guarantee can be enforced.
        """
        try:
            existing = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                ("idx_pr_fp_per_experiment",),
            ).fetchone()
            if existing:
                return
            dup = self.conn.execute(
                """
                SELECT graph_fingerprint, experiment_id, COUNT(*) AS n
                FROM program_results
                WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                GROUP BY graph_fingerprint, experiment_id
                HAVING n > 1
                LIMIT 1
                """
            ).fetchone()
            if dup is not None:
                LOGGER.warning(
                    "Skipping idx_pr_fp_per_experiment install: "
                    "program_results contains within-experiment duplicates. "
                    "Run `python -m research.tools.dedup_within_experiment --apply` "
                    "first, then restart to install the dedup guarantee."
                )
                return
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_pr_fp_per_experiment "
                "ON program_results(graph_fingerprint, experiment_id) "
                "WHERE graph_fingerprint IS NOT NULL "
                "AND graph_fingerprint <> ''"
            )
            LOGGER.info(
                "Installed idx_pr_fp_per_experiment — within-experiment fingerprint dedup is now enforced at the schema level."
            )
        except sqlite3.OperationalError as exc:
            LOGGER.warning("idx_pr_fp_per_experiment migration skipped: %s", exc)

    def _migrate_leaderboard_fp_dedup(self) -> None:
        """Denormalize graph_fingerprint onto leaderboard and install a
        UNIQUE partial index so the schema blocks duplicate leaderboard
        entries for the same fingerprint.

        Three steps, all idempotent:

        1. Add ``graph_fingerprint TEXT`` column if missing.
        2. Backfill it from ``program_results`` for existing leaderboard rows.
        3. Install ``idx_leaderboard_fp`` UNIQUE partial index (skipped with a
           warning if duplicate fingerprints still exist — run
           ``research/tools/leaderboard_dedup_by_fingerprint.py --apply``
           first).

        Complements the Python-layer ``DuplicateLeaderboardFingerprintError``
        gate on ``upsert_leaderboard``. Defense-in-depth, matching the slice-3
        pattern on ``program_results``.
        """
        try:
            # 1. Add column
            cols = {
                row[1] for row in self.conn.execute("PRAGMA table_info(leaderboard)")
            }
            if "graph_fingerprint" not in cols:
                self.conn.execute(
                    "ALTER TABLE leaderboard ADD COLUMN graph_fingerprint TEXT"
                )
                LOGGER.info(
                    "Added leaderboard.graph_fingerprint column (denormalized "
                    "from program_results for dedup enforcement)."
                )

            # 2. Backfill NULLs — one UPDATE..FROM query would race with
            #    concurrent readers; do an in-place UPDATE via correlated
            #    subquery instead (SQLite-safe).
            self.conn.execute(
                "UPDATE leaderboard "
                "SET graph_fingerprint = ("
                "  SELECT pr.graph_fingerprint FROM program_results pr "
                "  WHERE pr.result_id = leaderboard.result_id "
                ") "
                "WHERE graph_fingerprint IS NULL"
            )

            # 3. Install UNIQUE partial index (skip if non-reference dups remain).
            # References are a separate namespace — GPT-2/Mamba/RWKV registered
            # as baselines can legitimately share a fingerprint with a
            # synthesized discovery. Exclude is_reference=1 from the index.
            existing_idx = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                ("idx_leaderboard_fp",),
            ).fetchone()
            if existing_idx:
                return
            dup = self.conn.execute(
                """
                SELECT graph_fingerprint, COUNT(*) AS n
                FROM leaderboard
                WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                  AND COALESCE(is_reference, 0) = 0
                GROUP BY graph_fingerprint
                HAVING n > 1
                LIMIT 1
                """
            ).fetchone()
            if dup is not None:
                LOGGER.warning(
                    "Skipping idx_leaderboard_fp install: leaderboard contains "
                    "duplicate graph_fingerprints. Run `python -m "
                    "research.tools.leaderboard_dedup_by_fingerprint --apply` "
                    "first, then restart to install the dedup guarantee."
                )
                return
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_leaderboard_fp "
                "ON leaderboard(graph_fingerprint) "
                "WHERE graph_fingerprint IS NOT NULL "
                "AND graph_fingerprint <> '' "
                "AND COALESCE(is_reference, 0) = 0"
            )
            LOGGER.info(
                "Installed idx_leaderboard_fp — leaderboard fingerprint dedup "
                "is now enforced at the schema level (excluding references)."
            )
        except sqlite3.OperationalError as exc:
            LOGGER.warning("idx_leaderboard_fp migration skipped: %s", exc)

    def _get_program_results_columns(self) -> set[str]:
        """Return current program_results columns for defensive inserts."""
        if self._program_results_columns is None:
            rows = self.conn.execute("PRAGMA table_info(program_results)").fetchall()
            self._program_results_columns = {str(row[1]) for row in rows}
        return self._program_results_columns

    def _get_leaderboard_columns(self) -> set[str]:
        """Return current leaderboard columns for defensive updates."""
        if self._leaderboard_columns is None:
            rows = self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
            self._leaderboard_columns = {str(row[1]) for row in rows}
        return self._leaderboard_columns

    @classmethod
    def _detect_code_version(cls) -> str:
        """Detect code version for experiment traceability."""
        if cls._cached_code_version:
            return cls._cached_code_version

        env_version = os.environ.get("RESEARCH_CODE_VERSION")
        if env_version:
            cls._cached_code_version = env_version
            return cls._cached_code_version

        repo_root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                text=True,
            ).strip()
            if commit:
                cls._cached_code_version = commit
                return cls._cached_code_version
        except (OSError, subprocess.SubprocessError):
            pass

        cls._cached_code_version = "unknown"
        return cls._cached_code_version

    # ── Knowledge Digests ──

    def store_digest(self, digest_dict: Dict) -> str:
        """Store a knowledge digest and return its ID."""
        digest_id = str(uuid.uuid4())
        ts = digest_dict.get("timestamp", time.time())
        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge_digests
               (digest_id, timestamp, cycle_number, digest_json,
                narrative_summary, n_experiments_analyzed, n_curves_analyzed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                digest_id,
                ts,
                digest_dict.get("cycle_number"),
                json.dumps(digest_dict),
                digest_dict.get("narrative", "")[:2000],
                digest_dict.get("n_experiments_analyzed"),
                digest_dict.get("n_curves_analyzed"),
            ),
        )
        self._maybe_commit()
        return digest_id

    def get_latest_digest(self) -> Optional[Dict]:
        """Return the most recent knowledge digest, or None."""
        try:
            row = self.conn.execute(
                "SELECT digest_json FROM knowledge_digests ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            sqlite3.OperationalError,
        ) as e:
            LOGGER.debug("Failed to load latest digest: %s", e)
        return None

    def close(self):
        if getattr(self, "_use_native", False):
            # Native mode: stop the Rust writer thread but do NOT close
            # the connection — it must stay alive to prevent SHM teardown.
            try:
                self.conn._mgr.stop_writer()
            except Exception:
                pass
            return
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_write_queue") and self._writer_thread_started:
            self._write_queue.put(None)  # Sentinel
        if (
            hasattr(self, "_writer_thread")
            and self._writer_thread is not None
            and self._writer_thread.is_alive()
        ):
            self._writer_thread.join(timeout=2.0)
        self.conn.close()

    def _compress(self, data: Any) -> bytes:
        """JSON-encode and zlib-compress data."""
        from ..json_utils import json_safe

        return zlib.compress(json.dumps(json_safe(data)).encode("utf-8"))

    def _decompress(self, blob: Any) -> Any:
        """Decompress zlib blob and JSON-decode with fallback for raw strings."""
        if not blob:
            return None
        if not isinstance(blob, bytes):
            # Already a string (old data)
            try:
                return json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                return blob
        try:
            return json.loads(zlib.decompress(blob).decode("utf-8"))
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
            # Fallback for old uncompressed bytes data if any
            return json.loads(blob.decode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def batch(self):
        """Context manager to batch multiple writes into a single commit."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._maybe_commit()

    def _maybe_commit(self):
        """Commit unless inside a batch() context."""
        if self._batch_depth == 0:
            self._invalidate_dashboard_summary_cache()
            self.conn.commit()

    def _invalidate_dashboard_summary_cache(self) -> None:
        """Clear the short-lived dashboard summary cache after writes."""
        self._dashboard_summary_cache = {}
        self._dashboard_summary_cache_expires_at = 0.0
        self._template_observability_cache = {}
        self._template_observability_cache_expires_at = 0.0
