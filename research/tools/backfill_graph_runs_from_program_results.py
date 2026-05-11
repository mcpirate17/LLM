"""Backfill `graphs` + `graph_runs` from the legacy `program_results` table.

Phase 5b (commit ``1efff48``) cut new writes over to the graph-centric
storage (`graphs` + `graph_runs`) and made the `program_results_compat`
view read exclusively from the new tables. The cut-over assumed the
dual-write era had already mirrored every row; in practice the new tables
were empty (`graphs` = 0, `graph_runs` = 0) while the legacy table held
21680 rows. Every analytics consumer that reads through the compat view
(`mine_template_subpatterns`, `mine_template_subpatterns_v2`,
`cross_exp_probe_merge`, `rescore_champion_tiny_model`,
`export_bpe_backfill_inventory`) therefore sees 0 rows and silently
produces empty outputs.

This tool restores the migration's intended state by replaying every
legacy `program_results` row into the new tables. It is:

- **Idempotent.** `INSERT OR IGNORE` on `graphs` (PK = `graph_fingerprint`)
  and `graph_runs` (PK = `result_id`). Re-running is safe.
- **Additive only.** Never deletes or updates existing rows. The legacy
  `program_results` table is untouched; Phase 5b Stage 3 (drop legacy)
  remains out of scope here.
- **Dry-run by default.** Pass ``--apply`` to commit.

After a successful apply:
    SELECT COUNT(*) FROM program_results_compat
should match
    SELECT COUNT(*) FROM program_results
(modulo the NULL-fingerprint rows the compat view's NOT NULL FK rejects).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "research/runs.db"


def _graph_runs_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM pragma_table_info('graph_runs')").fetchall()
    return [r[0] for r in rows]


def _snapshot_state(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM program_results) AS pr_total,
            (SELECT COUNT(*) FROM program_results WHERE graph_fingerprint IS NULL) AS pr_null_fp,
            (SELECT COUNT(DISTINCT graph_fingerprint) FROM program_results
                WHERE graph_fingerprint IS NOT NULL) AS pr_distinct_fp,
            (SELECT COUNT(*) FROM program_results
                WHERE graph_fingerprint IS NOT NULL AND graph_json IS NULL) AS pr_null_gj,
            (SELECT COUNT(*) FROM graphs) AS graphs_count,
            (SELECT COUNT(*) FROM graph_runs) AS graph_runs_count,
            (SELECT COUNT(*) FROM program_results_compat) AS compat_count
        """
    ).fetchone()
    keys = (
        "program_results",
        "program_results_null_fingerprint",
        "program_results_distinct_fingerprints",
        "program_results_null_graph_json",
        "graphs",
        "graph_runs",
        "program_results_compat",
    )
    return dict(zip(keys, rows))


def _plan_graphs_inserts(conn: sqlite3.Connection) -> int:
    return conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT graph_fingerprint
            FROM program_results
            WHERE graph_fingerprint IS NOT NULL
              AND graph_fingerprint NOT IN (SELECT graph_fingerprint FROM graphs)
            GROUP BY graph_fingerprint
        )
        """
    ).fetchone()[0]


def _plan_graph_runs_inserts(conn: sqlite3.Connection) -> int:
    return conn.execute(
        """
        SELECT COUNT(*) FROM program_results
        WHERE graph_fingerprint IS NOT NULL
          AND result_id NOT IN (SELECT result_id FROM graph_runs)
        """
    ).fetchone()[0]


def _insert_graphs(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO graphs
            (graph_fingerprint, graph_json, arch_spec_json,
             first_seen_ts, last_seen_ts, graph_json_is_placeholder)
        SELECT
            graph_fingerprint,
            COALESCE(MAX(graph_json), '{}') AS graph_json,
            MAX(arch_spec_json) AS arch_spec_json,
            MIN(timestamp) AS first_seen_ts,
            MAX(timestamp) AS last_seen_ts,
            CASE WHEN MAX(graph_json) IS NULL THEN 1 ELSE 0 END
                AS graph_json_is_placeholder
        FROM program_results
        WHERE graph_fingerprint IS NOT NULL
        GROUP BY graph_fingerprint
        """
    )
    return cur.rowcount


def _insert_graph_runs(conn: sqlite3.Connection) -> int:
    cols = _graph_runs_columns(conn)
    col_list = ", ".join(f'"{c}"' for c in cols)
    cur = conn.execute(
        f"""
        INSERT OR IGNORE INTO graph_runs ({col_list})
        SELECT {col_list}
        FROM program_results
        WHERE graph_fingerprint IS NOT NULL
        """
    )
    return cur.rowcount


def _print_state(label: str, state: dict[str, int]) -> None:
    print(f"\n{label}:")
    for key, value in state.items():
        print(f"  {key:<42} {value:>10}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="commit changes (default: dry-run, no writes)",
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
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        before = _snapshot_state(conn)
        _print_state("State before", before)

        if before["program_results"] == 0:
            print("\nlegacy table empty — nothing to backfill")
            return 0

        graphs_plan = _plan_graphs_inserts(conn)
        runs_plan = _plan_graph_runs_inserts(conn)
        print("\nPlanned inserts:")
        print(f"  graphs:     {graphs_plan:>10}")
        print(f"  graph_runs: {runs_plan:>10}")

        if not args.apply:
            print("\n(dry-run; rerun with --apply to commit)")
            return 0

        t0 = time.monotonic()
        graphs_inserted = _insert_graphs(conn)
        runs_inserted = _insert_graph_runs(conn)
        conn.commit()
        dt = time.monotonic() - t0
        print(f"\nApplied in {dt:.1f}s:")
        print(f"  graphs inserted:     {graphs_inserted}")
        print(f"  graph_runs inserted: {runs_inserted}")

        after = _snapshot_state(conn)
        _print_state("State after", after)

        expected_compat = (
            before["program_results"] - before["program_results_null_fingerprint"]
        )
        if after["program_results_compat"] != expected_compat:
            print(
                f"\nWARNING: program_results_compat = {after['program_results_compat']}, "
                f"expected {expected_compat}",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nVerified: program_results_compat row count matches "
            f"non-null-fingerprint program_results rows ({expected_compat})"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
