"""Phase 5b Stage 3 — drop the legacy ``program_results`` table.

Prerequisites enforced by the tool:

1. **Lock-door whitelist sum == 0.** Every production write to
   ``program_results`` must have been retargeted to ``graph_runs``. The
   tool reads
   ``research/tests/test_phase5b_no_new_program_results_writes.py``'s
   whitelist and refuses to run while any entry remains.
2. **All ``graph_runs`` rows accounted for.** The compat view's row count
   must equal ``program_results``' non-null-fingerprint row count
   (sanity check that the backfill stayed in sync).
3. **A backup exists.** The tool refuses to apply unless a backup with
   today's date is present in the repo root.

Once the prerequisites hold, the tool:

- Repoints each child table's ``REFERENCES program_results`` FK to
  ``REFERENCES graph_runs`` using SQLite's table-rebuild pattern (rename
  old, create new with corrected FK, copy data, drop old). Affected
  tables (as of 2026-05-10):
    - ``leaderboard`` (NO ACTION)
    - ``program_graph_features`` (CASCADE)
    - ``program_graph_ops`` (CASCADE)
    - ``program_graph_pairs`` (CASCADE)
    - ``training_curves`` (CASCADE)
- Drops the four propagation triggers attached to ``program_results``:
  ``_gn_sync_pr_to_runs``, ``_gn_sync_pr_update_to_runs``,
  ``_gn_sync_pr_delete_to_runs``, ``reject_dup_fingerprint_no_reason``.
- Drops the ``program_results`` table itself.
- Re-enables FKs and runs ``PRAGMA foreign_key_check`` for verification.

Usage:
    python -m research.tools.drop_legacy_program_results              # readiness report
    python -m research.tools.drop_legacy_program_results --apply      # commit
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "research/runs.db"

_CHILD_TABLES = (
    "leaderboard",
    "program_graph_features",
    "program_graph_ops",
    "program_graph_pairs",
    "training_curves",
)
_TRIGGERS_TO_DROP = (
    "_gn_sync_pr_to_runs",
    "_gn_sync_pr_update_to_runs",
    "_gn_sync_pr_delete_to_runs",
    "reject_dup_fingerprint_no_reason",
)


def _import_whitelist() -> dict[str, int]:
    """Read the lock-door test's whitelist without importing pytest."""
    import importlib.util

    test_path = REPO / "research/tests/test_phase5b_no_new_program_results_writes.py"
    spec = importlib.util.spec_from_file_location(
        "_phase5b_lockdoor_whitelist", test_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load whitelist module from {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._LEGACY_WRITE_COUNTS  # type: ignore[attr-defined]


def _check_readiness(conn: sqlite3.Connection) -> list[str]:
    """Return a list of unmet prerequisites (empty = ready)."""
    problems: list[str] = []

    whitelist = _import_whitelist()
    legacy_writes = sum(whitelist.values())
    if legacy_writes > 0:
        offending = sorted(f"{p}={n}" for p, n in whitelist.items() if n)
        problems.append(
            f"{legacy_writes} legacy program_results writes remain across "
            f"{len(whitelist)} files. Retarget to graph_runs before drop. "
            f"Offenders: {offending}"
        )

    pr_total, pr_null_fp = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN graph_fingerprint IS NULL THEN 1 ELSE 0 END) "
        "FROM program_results"
    ).fetchone()
    gr_total = conn.execute("SELECT COUNT(*) FROM graph_runs").fetchone()[0]
    expected_gr = pr_total - (pr_null_fp or 0)
    if gr_total < expected_gr:
        problems.append(
            f"graph_runs={gr_total}, expected {expected_gr} (program_results minus "
            f"{pr_null_fp or 0} null-fingerprint rows). Re-run "
            "backfill_graph_runs_from_program_results before drop."
        )

    today = datetime.now().strftime("%Y%m%d")
    backups = list(REPO.glob(f"research/runs.db.pre-drop-{today}-*.bak"))
    if not backups:
        problems.append(
            "No same-day pre-drop backup found. Create one with:\n"
            f"  cp research/runs.db research/runs.db.pre-drop-{today}-"
            f"$(date +%H%M%S).bak"
        )

    return problems


def _get_create_sql(conn: sqlite3.Connection, table_name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"table not found: {table_name}")
    return row[0]


def _rewrite_fk(create_sql: str) -> str:
    """Substitute REFERENCES program_results → REFERENCES graph_runs."""
    return create_sql.replace(
        "REFERENCES program_results", "REFERENCES graph_runs"
    ).replace('REFERENCES "program_results"', 'REFERENCES "graph_runs"')


def _rebuild_child(conn: sqlite3.Connection, table_name: str) -> None:
    """Rename, recreate with new FK, copy data, drop old."""
    create_sql = _get_create_sql(conn, table_name)
    new_create_sql = _rewrite_fk(create_sql)
    if new_create_sql == create_sql:
        return  # no FK to rewrite — leave the table alone
    new_create_sql = new_create_sql.replace(
        f"CREATE TABLE {table_name}", f"CREATE TABLE {table_name}__new", 1
    ).replace(f'CREATE TABLE "{table_name}"', f'CREATE TABLE "{table_name}__new"', 1)
    conn.execute(new_create_sql)
    conn.execute(f"INSERT INTO {table_name}__new SELECT * FROM {table_name}")
    conn.execute(f"DROP TABLE {table_name}")
    conn.execute(f"ALTER TABLE {table_name}__new RENAME TO {table_name}")


def _print_problems(problems: list[str]) -> None:
    print("\nReadiness problems:")
    for p in problems:
        for i, line in enumerate(p.splitlines()):
            prefix = "  - " if i == 0 else "    "
            print(f"{prefix}{line}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="execute the drop (default: report readiness only)",
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
        problems = _check_readiness(conn)
        if problems:
            _print_problems(problems)
            if args.apply:
                print(
                    "\nRefusing to apply with unmet prerequisites.",
                    file=sys.stderr,
                )
                return 1
            print("\nNot ready for Stage 3. Resolve the above and retry.")
            return 0

        print("All readiness checks passed.")
        if not args.apply:
            print("(rerun with --apply to drop program_results)")
            return 0

        print("Beginning Stage 3 drop in a transaction…")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            for child in _CHILD_TABLES:
                _rebuild_child(conn, child)
                print(f"  repointed FK: {child} → graph_runs")
            for trig in _TRIGGERS_TO_DROP:
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
                print(f"  dropped trigger: {trig}")
            conn.execute("DROP TABLE program_results")
            print("  dropped table: program_results")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

        fk_problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if fk_problems:
            print(
                f"\nWARNING: {len(fk_problems)} FK violations after drop. "
                "Restore from backup and investigate.",
                file=sys.stderr,
            )
            return 1

        print("\nStage 3 complete. Run the test sweep to verify.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
