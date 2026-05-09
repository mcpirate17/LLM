from __future__ import annotations

"""Create the Phase 3 runs database from the current lab notebook DB.

The migration uses SQLite's backup API to take a consistent source snapshot,
then drops legacy/event-bookkeeping tables from the candidate copy. The source
database is opened read-only for validation and is never modified.
"""

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from research.defaults import LAB_NOTEBOOK_DB, RUNS_DB
from research.tools.db_health import assert_sqlite_health


DROP_FROM_RUNS_DB = frozenset(
    {
        "applied_runtime_events",
        "runtime_projector_checkpoints",
        "repair_log",
        "leaderboard_dedup_backup",
        "leaderboard_reparent_archive",
        "program_results_cross_exp_merge_backup",
        "program_results_dedup_backup",
        "program_results_orphan_fingerprint_cleanup_backup",
    }
)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]


def _drop_tables_for_runs_db(
    conn: sqlite3.Connection, tables: Iterable[str]
) -> list[str]:
    dropped: list[str] = []
    for table in sorted(tables):
        if table.startswith("orphan_backup_") or table in DROP_FROM_RUNS_DB:
            conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(table)}")
            dropped.append(table)
    return dropped


def _row_counts(conn: sqlite3.Connection, tables: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in sorted(tables):
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()
        except sqlite3.OperationalError:
            continue
        counts[table] = int(row[0] or 0)
    return counts


def create_runs_db(
    *,
    source_db: str | Path = LAB_NOTEBOOK_DB,
    runs_db: str | Path = RUNS_DB,
    replace: bool = False,
    vacuum: bool = True,
) -> Dict[str, Any]:
    source = Path(source_db).resolve()
    dest = Path(runs_db).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() and not replace:
        raise FileExistsError(f"{dest} already exists; pass replace=True to rebuild it")

    assert_sqlite_health(source, label="pre-runs-db source")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{int(time.time())}")
    if tmp.exists():
        tmp.unlink()

    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
            src.execute("PRAGMA query_only=ON")
            source_tables = _user_tables(src)
            source_counts = _row_counts(src, source_tables)
            with sqlite3.connect(str(tmp)) as dst:
                src.backup(dst)
                dst.execute("PRAGMA foreign_keys=OFF")
                dropped = _drop_tables_for_runs_db(dst, source_tables)
                if vacuum:
                    dst.execute("VACUUM")
                dst.execute("PRAGMA optimize")
                dst.commit()

        if dest.exists():
            dest.unlink()
        tmp.replace(dest)
        assert_sqlite_health(dest, label="runs-db candidate")
        with sqlite3.connect(str(dest)) as conn:
            dest_tables = _user_tables(conn)
            dest_counts = _row_counts(conn, dest_tables)
        kept_tables = sorted(set(source_tables) - set(dropped))
        mismatches = {
            table: {
                "source": source_counts.get(table, 0),
                "runs": dest_counts.get(table, 0),
            }
            for table in kept_tables
            if source_counts.get(table, 0) != dest_counts.get(table, 0)
        }
        if mismatches:
            raise RuntimeError(f"runs DB row-count mismatches: {mismatches}")
        return {
            "source_db": str(source),
            "runs_db": str(dest),
            "dropped_tables": dropped,
            "kept_tables": kept_tables,
            "row_counts": {table: dest_counts.get(table, 0) for table in kept_tables},
        }
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=Path(LAB_NOTEBOOK_DB))
    parser.add_argument("--runs-db", type=Path, default=Path(RUNS_DB))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-vacuum", action="store_true")
    args = parser.parse_args()

    report = create_runs_db(
        source_db=args.source_db,
        runs_db=args.runs_db,
        replace=args.replace,
        vacuum=not args.no_vacuum,
    )
    print(f"runs_db: {report['runs_db']}")
    print(f"kept_tables: {len(report['kept_tables'])}")
    print(f"dropped_tables: {len(report['dropped_tables'])}")
    for table in report["dropped_tables"]:
        print(f"dropped: {table}")


if __name__ == "__main__":
    main()
