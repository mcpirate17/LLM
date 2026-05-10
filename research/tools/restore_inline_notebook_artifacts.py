from __future__ import annotations

"""Restore selected notebook artifact pointer columns back to inline payloads."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from research.defaults import RUNS_DB
from research.scientist.notebook.artifact_store import (
    NotebookArtifactStore,
    parse_artifact_pointer,
)
from research.tools._db_maintenance import check_writer_lock, table_columns
from research.tools.db_health import assert_sqlite_health


DEFAULT_DB = Path(RUNS_DB)
DEFAULT_TARGETS = (
    ("experiments", "experiment_id", "config_json"),
    ("experiments", "experiment_id", "results_json"),
    ("graph_runs", "result_id", "data_provenance_json"),
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _metadata_for_pointer(
    conn: sqlite3.Connection, pointer_value: Any
) -> dict[str, Any] | None:
    pointer = parse_artifact_pointer(pointer_value)
    if not pointer:
        return None
    artifact_id = str(pointer.get("_notebook_artifact") or "")
    if not artifact_id:
        return None
    row = conn.execute(
        "SELECT * FROM notebook_artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    if row is not None:
        return dict(row)
    if pointer.get("path"):
        return {
            "artifact_id": artifact_id,
            "path": pointer["path"],
            "compression": pointer.get("compression") or "zstd",
        }
    return None


def _restore_column(
    conn: sqlite3.Connection,
    store: NotebookArtifactStore,
    *,
    table: str,
    pk: str,
    column: str,
    apply: bool,
    limit: int | None,
) -> dict[str, int]:
    existing = set(table_columns(conn, table))
    if pk not in existing or column not in existing:
        return {"candidates": 0, "rows": 0, "bytes": 0}

    rows = conn.execute(
        f"""
        SELECT {pk} AS row_pk, {column} AS payload
        FROM {table}
        WHERE {column} IS NOT NULL
          AND {column} LIKE '{{"_notebook_artifact"%'
        """
    ).fetchall()
    candidates = len(rows)
    restored = 0
    raw_bytes = 0
    for row in rows:
        if limit is not None and restored >= limit:
            break
        metadata = _metadata_for_pointer(conn, row["payload"])
        if metadata is None:
            continue
        raw = store.read_bytes(metadata)
        try:
            restored_value: str | bytes = raw.decode("utf-8")
        except UnicodeDecodeError:
            restored_value = raw
        raw_bytes += len(raw)
        if apply:
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {pk} = ?",
                (restored_value, row["row_pk"]),
            )
        restored += 1
    if apply:
        conn.commit()
    return {"candidates": candidates, "rows": restored, "bytes": raw_bytes}


def _parse_target(value: str) -> tuple[str, str, str]:
    parts = value.split(".")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("target must be table.pk.column")
    return parts[0], parts[1], parts[2]


def run(
    *,
    db_path: Path,
    targets: Iterable[tuple[str, str, str]] = DEFAULT_TARGETS,
    apply: bool,
    limit: int | None,
    vacuum: bool,
) -> dict[str, Any]:
    if apply:
        check_writer_lock(Path(f"{db_path.resolve()}.writer-lock"))
    assert_sqlite_health(db_path, label="pre-inline-restore")
    conn = _connect(db_path)
    try:
        store = NotebookArtifactStore(db_path)
        restored = []
        for table, pk, column in targets:
            item = _restore_column(
                conn,
                store,
                table=table,
                pk=pk,
                column=column,
                apply=apply,
                limit=limit,
            )
            restored.append({"table": table, "pk": pk, "column": column, **item})
    finally:
        conn.close()
    if apply:
        assert_sqlite_health(db_path, label="post-inline-restore")
        if vacuum:
            with sqlite3.connect(str(db_path), timeout=30.0) as vacuum_conn:
                vacuum_conn.execute("VACUUM")
            assert_sqlite_health(db_path, label="post-vacuum")
    return {"dry_run": not apply, "restored": restored}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        help="Column to restore as table.pk.column. Defaults to raw-SQL-sensitive columns.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        db_path=args.db,
        targets=args.target or DEFAULT_TARGETS,
        apply=args.apply,
        limit=args.limit,
        vacuum=args.vacuum,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
