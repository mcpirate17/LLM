#!/usr/bin/env python3
"""One-shot SQL backfill: for every leaderboard-bound program_results row,
fill NULL "sticky" probe columns by copying the best-populated sibling value
from program_results rows sharing the same graph_fingerprint.

Sticky = once observed, the column should never go back to NULL when the
leaderboard rebinds to a fresher result_id whose run-path didn't compute
that probe (e.g. a screening replay doesn't run language_control probes,
so a rebind to such a row would lose NB05/NB10/NBINV).

Aggregation policy:
  - For numeric scores (auc, *_score, *_acc): take MAX over siblings
    (highest observed signal). NULL values are excluded.
  - For status / version / metadata strings: take any non-NULL sibling
    value (latest by rowid for determinism).

Behaviour:
  - Only writes to the bound row.
  - Only fills currently-NULL columns; never overwrites existing data.
  - Idempotent: re-running has no effect once columns are populated.
  - Prints a summary of which fps and columns were updated.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

DB_PATH = Path("/home/tim/Projects/LLM/research/runs.db")

NUMERIC_STICKY_COLUMNS = (
    "language_control_s05_sentence_assoc_score",
    "language_control_s05_binding_order_acc",
    "language_control_s05_binding_score",
    "language_control_s10_sentence_assoc_score",
    "language_control_s10_binding_order_acc",
    "language_control_s10_binding_score",
    "language_control_investigation_sentence_assoc_score",
    "language_control_investigation_binding_order_acc",
    "language_control_investigation_binding_score",
    "ar_validation_rank_score",
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_steps_to_floor",
    "ar_curriculum_auc_pair_final",
    "ar_gate_score",
    "ar_gate_in_dist_pair_acc",
    "ar_gate_in_dist_class_acc",
    "ar_gate_held_pair_acc",
    "ar_gate_held_class_acc",
)

LATEST_NONNULL_STICKY_COLUMNS = (
    "language_control_metric_version",
    "ar_validation_metric_version",
    "ar_validation_status",
    "ar_validation_learning_curve_json",
    "ar_gate_status",
    "ar_gate_metric_version",
)


def fetch_bound_targets(conn: sqlite3.Connection):
    """Yield (entry_id, result_id, fingerprint) for each leaderboard entry."""
    rows = conn.execute(
        "SELECT entry_id, result_id, graph_fingerprint FROM leaderboard "
        "WHERE result_id IS NOT NULL AND graph_fingerprint IS NOT NULL"
    ).fetchall()
    for r in rows:
        yield r[0], r[1], r[2]


def best_numeric_sibling(conn: sqlite3.Connection, fp: str, col: str) -> float | None:
    row = conn.execute(
        f"SELECT MAX({col}) FROM program_results_compat "
        f"WHERE graph_fingerprint = ? AND {col} IS NOT NULL",
        (fp,),
    ).fetchone()
    return row[0] if row else None


def latest_nonnull_string_sibling(
    conn: sqlite3.Connection, fp: str, col: str
) -> str | None:
    row = conn.execute(
        f"SELECT {col} FROM program_results_compat "
        f"WHERE graph_fingerprint = ? AND {col} IS NOT NULL "
        f"ORDER BY rowid DESC LIMIT 1",
        (fp,),
    ).fetchone()
    return row[0] if row else None


def fill_sticky_for_bound_row(
    conn: sqlite3.Connection,
    bound_result_id: str,
    fingerprint: str,
    columns_present: set[str],
    dry_run: bool,
) -> Dict[str, object]:
    """Returns dict: column -> value written (only for cols actually written)."""
    bound_row = conn.execute(
        f"SELECT {', '.join(c for c in columns_present)} "
        "FROM program_results_compat WHERE result_id = ?",
        (bound_result_id,),
    ).fetchone()
    if bound_row is None:
        return {}
    bound_dict = dict(zip([c for c in columns_present], bound_row))
    updates: Dict[str, object] = {}
    for col in NUMERIC_STICKY_COLUMNS:
        if col not in columns_present:
            continue
        if bound_dict.get(col) is not None:
            continue
        v = best_numeric_sibling(conn, fingerprint, col)
        if v is not None:
            updates[col] = v
    for col in LATEST_NONNULL_STICKY_COLUMNS:
        if col not in columns_present:
            continue
        if bound_dict.get(col) is not None:
            continue
        v = latest_nonnull_string_sibling(conn, fingerprint, col)
        if v is not None:
            updates[col] = v
    if updates and not dry_run:
        set_clause = ", ".join(f"{c} = ?" for c in updates)
        params: List[object] = list(updates.values())
        params.append(bound_result_id)
        conn.execute(
            f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?",
            params,
        )
    return updates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--limit-fp",
        nargs="*",
        default=None,
        help="Optional list of graph_fingerprints to limit to (for testing)",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    pr_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(program_results)").fetchall()
    }
    all_sticky = NUMERIC_STICKY_COLUMNS + LATEST_NONNULL_STICKY_COLUMNS
    columns_present = pr_cols.intersection(all_sticky)
    missing = sorted(set(all_sticky) - columns_present)
    if missing:
        print(
            f"# WARN: {len(missing)} sticky columns not present in program_results, skipping:"
        )
        for c in missing:
            print(f"#   {c}")
    print(f"# Backfilling {len(columns_present)} sticky columns")

    n_total = 0
    n_updated = 0
    columns_updated_count: Dict[str, int] = {}
    fps_updated: List[Tuple[str, Dict[str, object]]] = []
    for entry_id, bound_rid, fp in fetch_bound_targets(conn):
        if args.limit_fp and fp not in args.limit_fp:
            continue
        n_total += 1
        u = fill_sticky_for_bound_row(
            conn, bound_rid, fp, columns_present, args.dry_run
        )
        if u:
            n_updated += 1
            fps_updated.append((fp, u))
            for c in u:
                columns_updated_count[c] = columns_updated_count.get(c, 0) + 1
    if not args.dry_run:
        conn.commit()

    print(f"# fps inspected: {n_total}")
    print(f"# fps with at least one sticky column written: {n_updated}")
    print()
    print("# updates per column (number of fps where column was filled):")
    for c, n in sorted(columns_updated_count.items(), key=lambda kv: -kv[1]):
        print(f"#   {c}: {n}")
    print()
    print("# detail (first 30):")
    for fp, u in fps_updated[:30]:
        cols = ", ".join(
            f"{c}={v}" if not isinstance(v, str) else f"{c}=<str>" for c, v in u.items()
        )
        print(f"#   {fp[:16]}: {cols}")


if __name__ == "__main__":
    main()
