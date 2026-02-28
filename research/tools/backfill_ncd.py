#!/usr/bin/env python3
"""
Backfill NCD scores for existing program_results that have graph_json and training curves.

Usage:
    python -m research.tools.backfill_ncd [--db path/to/lab_notebook.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def backfill_ncd(db_path: str, dry_run: bool = False) -> None:
    from research.eval.ncd import compute_graph_ncd

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find results with graph_json that don't have ncd_score yet
    # and have training curves available
    try:
        conn.execute("SELECT ncd_score FROM program_results LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE program_results ADD COLUMN ncd_score REAL")
        conn.execute("ALTER TABLE program_results ADD COLUMN ncd_description_length INTEGER")
        conn.execute("ALTER TABLE program_results ADD COLUMN ncd_description_length_per_param REAL")
        conn.commit()

    rows = conn.execute("""
        SELECT pr.result_id, pr.graph_json, pr.graph_n_params_estimate,
               tc.curve_json
        FROM program_results pr
        JOIN training_curves tc ON tc.result_id = pr.result_id
        WHERE pr.graph_json IS NOT NULL
          AND pr.ncd_score IS NULL
          AND tc.curve_json IS NOT NULL
    """).fetchall()

    print(f"Found {len(rows)} results to backfill")

    updated = 0
    errors = 0
    for row in rows:
        try:
            import json
            curve = json.loads(row["curve_json"])
            if not curve:
                continue

            result = compute_graph_ncd(
                row["graph_json"],
                curve,
                n_params=row["graph_n_params_estimate"],
            )

            if not dry_run:
                conn.execute(
                    """UPDATE program_results
                       SET ncd_score=?, ncd_description_length=?, ncd_description_length_per_param=?
                       WHERE result_id=?""",
                    (result["ncd_score"], result["description_length"],
                     result.get("description_length_per_param"), row["result_id"]),
                )

                # Also update leaderboard if entry exists
                conn.execute(
                    """UPDATE leaderboard SET ncd_score=? WHERE result_id=?""",
                    (result["ncd_score"], row["result_id"]),
                )

            updated += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {row['result_id']}: {e}")

    if not dry_run:
        conn.commit()

    print(f"{'Would update' if dry_run else 'Updated'} {updated} results ({errors} errors)")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill NCD scores")
    parser.add_argument("--db", default="research/lab_notebook.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    backfill_ncd(args.db, args.dry_run)


if __name__ == "__main__":
    main()
