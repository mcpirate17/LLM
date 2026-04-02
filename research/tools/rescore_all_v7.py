#!/usr/bin/env python3
"""Rescore all leaderboard entries using composite v7."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.scientist.notebook import LabNotebook
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)

DB_PATH = "research/lab_notebook.db"


def main():
    nb = LabNotebook(DB_PATH)
    cur = nb.conn.cursor()

    rows = cur.execute(
        "SELECT entry_id, result_id, tier, model_source, architecture_desc, "
        "is_reference, reference_name, tags, notes, composite_score "
        "FROM leaderboard ORDER BY composite_score DESC"
    ).fetchall()

    total = len(rows)
    print(f"Rescoring {total} leaderboard entries with v7...")

    # Batch-fetch all program_results in one query instead of N individual ones
    all_result_ids = [r["result_id"] for r in rows]
    pr_cache = prefetch_program_results(nb.conn, all_result_ids)
    print(f"  Pre-fetched {len(pr_cache)} program_results rows")

    updated = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        entry_id = row["entry_id"]
        result_id = row["result_id"]
        is_ref = bool(row["is_reference"])
        old_score = row["composite_score"]

        try:
            # Fetch existing leaderboard data for this entry
            existing = cur.execute(
                "SELECT * FROM leaderboard WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            if not existing:
                skipped += 1
                continue

            d = dict(existing)
            pr_dict = pr_cache.get(result_id, {})
            score_kwargs = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
            new_score = compute_composite(**score_kwargs)

            if new_score != old_score:
                cur.execute(
                    "UPDATE leaderboard SET composite_score = ?, "
                    "rescore_status = 'rescored_v7', "
                    "rescore_timestamp = ?, "
                    "old_composite_score = ?, "
                    "rescore_reason = 'bulk_v7_rescore' "
                    "WHERE entry_id = ?",
                    (new_score, time.time(), old_score, entry_id),
                )
                updated += 1
            else:
                # Score unchanged, still mark as rescored
                cur.execute(
                    "UPDATE leaderboard SET rescore_status = 'rescored_v7', "
                    "rescore_timestamp = ? WHERE entry_id = ?",
                    (time.time(), entry_id),
                )

        except Exception as e:
            errors += 1
            print(f"  ERROR [{entry_id}] {result_id}: {e}")

        if (i + 1) % 50 == 0:
            nb.conn.commit()
            elapsed = time.time() - t0
            print(
                f"  {i + 1}/{total} ({elapsed:.1f}s) — {updated} changed, {errors} errors"
            )

    nb.conn.commit()
    elapsed = time.time() - t0

    # Print summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total:   {total}")
    print(f"  Changed: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")

    # Show top 10
    top = cur.execute(
        "SELECT entry_id, result_id, tier, composite_score, is_reference, "
        "reference_name, model_source "
        "FROM leaderboard ORDER BY composite_score DESC LIMIT 15"
    ).fetchall()
    print("\nTop 15 after rescore:")
    print(f"{'Score':>8}  {'Tier':<14}  {'Source':<20}  {'Ref?':<6}  {'ID'}")
    for r in top:
        ref = r["reference_name"] or ""
        src = (r["model_source"] or "")[:20]
        print(
            f"{r['composite_score']:>8.1f}  {r['tier']:<14}  {src:<20}  {ref:<6}  {r['entry_id']}"
        )

    nb.conn.close()


if __name__ == "__main__":
    main()
