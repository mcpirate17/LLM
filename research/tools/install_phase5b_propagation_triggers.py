"""Install Phase 5b UPDATE + DELETE propagation triggers on the live runs.db.

The notebook bootstrap path adds these triggers automatically on the next
init. This one-shot tool applies them immediately to a running database
without waiting for a notebook restart.

Triggers installed (idempotent — exits early if already present):

- ``_gn_sync_pr_update_to_runs`` — AFTER UPDATE ON program_results.
  Upserts the NEW row into graphs (for graph_json / arch_spec_json) and
  graph_runs (for every other column). Closes the writer-mirror gap that
  ``LabNotebook._submit_write`` only covered for callers going through the
  notebook write queue.
- ``_gn_sync_pr_delete_to_runs`` — AFTER DELETE ON program_results.
  Deletes the matching graph_runs row.

Together with the existing ``_gn_sync_pr_to_runs`` (AFTER INSERT), these
make raw ``conn.execute("UPDATE/DELETE/INSERT … program_results …")``
safe regardless of which writer issued the SQL. This is the precondition
for Phase 5b Stage 3 (drop legacy table).

Usage:
    python -m research.tools.install_phase5b_propagation_triggers          # dry-run
    python -m research.tools.install_phase5b_propagation_triggers --apply  # commit
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "research/runs.db"

_UPDATE_TRIGGER_NAME = "_gn_sync_pr_update_to_runs"
_DELETE_TRIGGER_NAME = "_gn_sync_pr_delete_to_runs"


def _trigger_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _build_update_trigger_sql(conn: sqlite3.Connection) -> str:
    pr_cols = [r[1] for r in conn.execute("PRAGMA table_info(program_results)")]
    if not pr_cols:
        raise RuntimeError("program_results has no columns — schema not initialized")
    arch_set = {"graph_json", "arch_spec_json"}
    run_set_cols = [c for c in pr_cols if c not in arch_set and c != "result_id"]
    set_clause = ", ".join(f'"{c}" = NEW."{c}"' for c in run_set_cols)
    return (
        f"CREATE TRIGGER {_UPDATE_TRIGGER_NAME} "
        "AFTER UPDATE ON program_results "
        "WHEN NEW.graph_fingerprint IS NOT NULL "
        "  AND TRIM(NEW.graph_fingerprint) <> '' "
        "BEGIN "
        "  INSERT INTO graphs "
        "    (graph_fingerprint, graph_json, arch_spec_json, "
        "     first_seen_ts, last_seen_ts, graph_json_is_placeholder) "
        "  VALUES (NEW.graph_fingerprint, "
        "          COALESCE(NEW.graph_json, '{}'), "
        "          NEW.arch_spec_json, "
        "          NEW.timestamp, NEW.timestamp, "
        "          CASE WHEN NEW.graph_json IS NULL "
        "                 OR NEW.graph_json IN ('', '{}') "
        "          THEN 1 ELSE 0 END) "
        "  ON CONFLICT(graph_fingerprint) DO UPDATE SET "
        "    graph_json = excluded.graph_json, "
        "    arch_spec_json = COALESCE(excluded.arch_spec_json, "
        "                              graphs.arch_spec_json), "
        "    last_seen_ts = MAX(graphs.last_seen_ts, excluded.last_seen_ts), "
        "    graph_json_is_placeholder = excluded.graph_json_is_placeholder; "
        f"  UPDATE graph_runs SET {set_clause} WHERE result_id = NEW.result_id; "
        "END"
    )


def _build_delete_trigger_sql() -> str:
    return (
        f"CREATE TRIGGER {_DELETE_TRIGGER_NAME} "
        "AFTER DELETE ON program_results "
        "WHEN OLD.graph_fingerprint IS NOT NULL "
        "  AND TRIM(OLD.graph_fingerprint) <> '' "
        "BEGIN "
        "  DELETE FROM graph_runs WHERE result_id = OLD.result_id; "
        "END"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="commit triggers (default: dry-run)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"sqlite db (default: {DEFAULT_DB})",
    )
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(args.db))
    try:
        have_update = _trigger_exists(conn, _UPDATE_TRIGGER_NAME)
        have_delete = _trigger_exists(conn, _DELETE_TRIGGER_NAME)

        print("Phase 5b propagation triggers:")
        print(f"  {_UPDATE_TRIGGER_NAME}: {'present' if have_update else 'MISSING'}")
        print(f"  {_DELETE_TRIGGER_NAME}: {'present' if have_delete else 'MISSING'}")

        if have_update and have_delete:
            print("\nNothing to do.")
            return 0

        update_sql = None if have_update else _build_update_trigger_sql(conn)
        delete_sql = None if have_delete else _build_delete_trigger_sql()

        if not args.apply:
            print("\nPlanned installs:")
            if update_sql:
                print(f"  + {_UPDATE_TRIGGER_NAME} ({len(update_sql)} chars)")
            if delete_sql:
                print(f"  + {_DELETE_TRIGGER_NAME} ({len(delete_sql)} chars)")
            print("\n(dry-run; rerun with --apply to commit)")
            return 0

        if update_sql:
            conn.execute(update_sql)
            print(f"installed: {_UPDATE_TRIGGER_NAME}")
        if delete_sql:
            conn.execute(delete_sql)
            print(f"installed: {_DELETE_TRIGGER_NAME}")
        conn.commit()
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
