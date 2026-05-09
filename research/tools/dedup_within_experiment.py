#!/usr/bin/env python3
"""Collapse within-experiment duplicate (graph_fingerprint, experiment_id) rows.

Within each duplicate group: keep the latest row by `timestamp`, but FIRST merge
any NULL fields on the keeper from the most-recent non-NULL value across the
siblings. Backup deleted rows to `program_results_dedup_backup` before deletion.

Default mode (no flags) is a dry-run that prints a diff report. Use `--apply`
to execute against the writer; the writer flock must be free.

This is a prerequisite for installing
    CREATE UNIQUE INDEX idx_pr_fp_per_experiment
        ON program_results(graph_fingerprint, experiment_id)
        WHERE graph_fingerprint IS NOT NULL AND graph_fingerprint <> ''
which slice 3a of the dedup-governance plan adds.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from research.tools._db_maintenance import (
    DEFAULT_WRITER_LOCK,
    check_writer_lock,
    connect_readonly,
    connect_writer,
    ensure_backup_table,
    table_row_count,
)

DEFAULT_DB = Path("research/runs.db")
BACKUP_TABLE = "program_results_dedup_backup"
IDENTITY_COLUMNS = {"result_id", "experiment_id", "graph_fingerprint", "timestamp"}


def _list_columns(conn: sqlite3.Connection) -> List[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(program_results)")]


def _list_duplicate_groups(
    conn: sqlite3.Connection,
) -> List[Tuple[str, str, int]]:
    rows = conn.execute(
        """
        SELECT graph_fingerprint, experiment_id, COUNT(*) AS n
        FROM program_results
        WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
        GROUP BY graph_fingerprint, experiment_id
        HAVING n > 1
        ORDER BY n DESC, graph_fingerprint, experiment_id
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _fetch_group(
    conn: sqlite3.Connection, fp: str, experiment_id: str
) -> List[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM program_results
            WHERE graph_fingerprint = ? AND experiment_id = ?
            ORDER BY timestamp DESC
            """,
            (fp, experiment_id),
        )
    )


def _plan_merge(
    rows: List[sqlite3.Row], mergeable_columns: Iterable[str]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Return (column_updates, kept_result_id, deleted_result_ids).

    rows[0] is the keeper (sorted by timestamp DESC). For each mergeable column
    where keeper has NULL, take the first non-NULL value from the siblings.
    """
    keeper = dict(rows[0])
    sibling_dicts = [dict(r) for r in rows[1:]]
    updates: Dict[str, Any] = {}
    for col in mergeable_columns:
        if keeper.get(col) is not None:
            continue
        for sib in sibling_dicts:
            value = sib.get(col)
            if value is not None:
                updates[col] = value
                break
    return (
        updates,
        str(keeper["result_id"]),
        [str(s["result_id"]) for s in sibling_dicts],
    )


def _ensure_backup_table(conn: sqlite3.Connection) -> None:
    ensure_backup_table(
        conn,
        backup_table=BACKUP_TABLE,
        source_table="program_results",
        indexes=((f"idx_{BACKUP_TABLE}_result_id", ("result_id",)),),
    )


def run(db_path: Path, *, apply: bool, limit_groups: int | None) -> int:
    read_conn = connect_readonly(db_path)
    columns = _list_columns(read_conn)
    mergeable = [c for c in columns if c not in IDENTITY_COLUMNS]
    groups = _list_duplicate_groups(read_conn)
    if limit_groups is not None:
        groups = groups[:limit_groups]

    total_groups = len(groups)
    total_rows_to_delete = 0
    total_fields_merged = 0
    field_merge_counter: Counter[str] = Counter()
    plans: List[Dict[str, Any]] = []

    for fp, experiment_id, n in groups:
        rows = _fetch_group(read_conn, fp, experiment_id)
        updates, kept_id, deleted_ids = _plan_merge(rows, mergeable)
        total_rows_to_delete += len(deleted_ids)
        total_fields_merged += len(updates)
        field_merge_counter.update(updates.keys())
        plans.append(
            {
                "graph_fingerprint": fp,
                "experiment_id": experiment_id,
                "row_count": n,
                "kept_result_id": kept_id,
                "deleted_result_ids": deleted_ids,
                "merge_updates": updates,
            }
        )

    print(
        f"Plan: {total_groups} duplicate groups, {total_rows_to_delete} rows to delete, "
        f"{total_fields_merged} field-values to merge into keepers."
    )
    if field_merge_counter:
        print("Top merged fields:")
        for col, count in field_merge_counter.most_common(20):
            print(f"  {col}: {count}")

    print()
    print("Top 10 groups by row count:")
    for plan in plans[:10]:
        merged_keys = sorted(plan["merge_updates"].keys())[:6]
        merge_summary = ", ".join(merged_keys) + (
            "..." if len(plan["merge_updates"]) > 6 else ""
        )
        print(
            f"  fp={plan['graph_fingerprint'][:16]} "
            f"exp={plan['experiment_id'][:12]} n={plan['row_count']:3d} "
            f"keep={plan['kept_result_id'][:12]} "
            f"merge={len(plan['merge_updates'])} ({merge_summary or '-'})"
        )

    read_conn.close()

    if not apply:
        print()
        print("Dry-run only. Re-run with --apply to execute against the writer.")
        return 0

    print()
    print(f"Applying changes to {db_path}...")
    check_writer_lock(DEFAULT_WRITER_LOCK)
    write_conn = connect_writer(db_path)
    try:
        _ensure_backup_table(write_conn)
        # Use a single transaction so a failure rolls back everything.
        with write_conn:
            for plan in plans:
                kept_id = plan["kept_result_id"]
                deleted_ids = plan["deleted_result_ids"]
                if not deleted_ids:
                    continue
                placeholders = ",".join("?" for _ in deleted_ids)
                # Backup first
                write_conn.execute(
                    f"INSERT INTO {BACKUP_TABLE} "
                    f"SELECT * FROM program_results "
                    f"WHERE result_id IN ({placeholders})",
                    tuple(deleted_ids),
                )
                # Merge non-null fields onto the keeper
                if plan["merge_updates"]:
                    set_parts = ", ".join(f"{c} = ?" for c in plan["merge_updates"])
                    values = list(plan["merge_updates"].values()) + [kept_id]
                    write_conn.execute(
                        f"UPDATE program_results SET {set_parts} WHERE result_id = ?",
                        values,
                    )
                # Delete siblings
                write_conn.execute(
                    f"DELETE FROM program_results WHERE result_id IN ({placeholders})",
                    tuple(deleted_ids),
                )
        backup_count = table_row_count(write_conn, BACKUP_TABLE)
        print(
            f"Done. Deleted {total_rows_to_delete} rows; "
            f"merged {total_fields_merged} field-values; "
            f"backup table now has {backup_count} rows."
        )
    finally:
        write_conn.close()
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to runs.db (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the cleanup against the writer (default: dry-run only).",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Process only the first N duplicate groups (debugging).",
    )
    args = parser.parse_args(argv)
    return run(args.db, apply=args.apply, limit_groups=args.limit_groups)


if __name__ == "__main__":
    sys.exit(main())
