#!/usr/bin/env python
"""Physically rename legacy probe metric columns to cascade names.

The tool is explicit on purpose: normal notebook startup should not perform a
large schema rewrite without an operator-visible backup check.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from research.scientist.probe_metric_names import TABLE_RENAMES
from research.tools.check_backup_freshness import main as check_backup_freshness_main
from research.tools.db_health import assert_sqlite_health, backup_sqlite_db


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/runs.db"


@dataclass(frozen=True, slots=True)
class RenameAction:
    table: str
    old: str
    new: str


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    }


def plan_renames(conn: sqlite3.Connection) -> list[RenameAction]:
    actions: list[RenameAction] = []
    errors: list[str] = []
    for table, renames in TABLE_RENAMES.items():
        existing = _table_columns(conn, table)
        if not existing:
            continue
        for old, new in renames.items():
            old_exists = old in existing
            new_exists = new in existing
            if old_exists and new_exists:
                errors.append(f"{table}: both {old} and {new} exist")
            elif old_exists:
                actions.append(RenameAction(table=table, old=old, new=new))
    if errors:
        detail = "\n".join(f"  - {err}" for err in errors)
        raise RuntimeError(f"rename collisions detected:\n{detail}")
    return actions


def apply_renames(conn: sqlite3.Connection, actions: list[RenameAction]) -> None:
    with conn:
        for action in actions:
            conn.execute(
                "ALTER TABLE "
                f"{_quote_identifier(action.table)} "
                "RENAME COLUMN "
                f"{_quote_identifier(action.old)} "
                "TO "
                f"{_quote_identifier(action.new)}"
            )


def verify_no_pending_renames(conn: sqlite3.Connection) -> None:
    pending = plan_renames(conn)
    if pending:
        lines = "\n".join(
            f"  - {action.table}.{action.old} -> {action.new}" for action in pending
        )
        raise RuntimeError(f"pending probe metric renames remain:\n{lines}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--write", action="store_true", help="Apply the physical renames."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a fresh backup; require an existing fresh backup instead.",
    )
    parser.add_argument(
        "--max-backup-age-hours",
        type=float,
        default=24.0,
        help="Freshness window used when --no-backup is supplied.",
    )
    args = parser.parse_args(argv)

    db_path = args.db
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    assert_sqlite_health(db_path, label="pre-rename")
    conn = sqlite3.connect(str(db_path))
    try:
        actions = plan_renames(conn)
        if not actions:
            print("No legacy probe metric columns found.")
            return 0

        print("Planned probe metric column renames:")
        for action in actions:
            print(f"  {action.table}.{action.old} -> {action.new}")

        if not args.write:
            print("\nDry run only. Re-run with --write to apply.")
            return 0

        if args.no_backup:
            rc = check_backup_freshness_main(
                ["--max-age-hours", str(args.max_backup_age_hours)]
            )
            if rc != 0:
                return rc
        else:
            backup_path = backup_sqlite_db(db_path, suffix="pre_probe_metric_rename")
            print(f"Backup created: {backup_path}")

        apply_renames(conn, actions)
        verify_no_pending_renames(conn)
    finally:
        conn.close()

    assert_sqlite_health(db_path, label="post-rename")
    print(f"Applied {len(actions)} probe metric column rename(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
