#!/usr/bin/env python3
"""Backfill v8 composite scores and calibrate thresholds.

Recomputes composite scores using compute_composite_v8 for all leaderboard
entries. Generates a v7 vs v8 comparison report showing rank changes.
Prints calibrated v8 threshold recommendations.

Usage:
    python -m research.tools.backfill_v8_scores [--top N] [--report] [--dry-run]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite_v7,
    compute_composite_v8,
    prefetch_program_results,
)
from research.scientist.notebook import LabNotebook
from research.tools._backfill_shared import DB_PATH


def _fetch_leaderboard(nb, top: int = 0):
    """Fetch leaderboard entries ordered by composite_score desc."""
    q = "SELECT * FROM leaderboard ORDER BY composite_score DESC"
    if top > 0:
        q += f" LIMIT {top}"
    return nb.conn.execute(q).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Backfill v8 composite scores")
    parser.add_argument(
        "--top", type=int, default=0, help="Limit to top N entries (0=all)"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print v7 vs v8 comparison report"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument(
        "--calibrate", action="store_true", help="Print threshold calibration"
    )
    args = parser.parse_args()

    nb = LabNotebook(DB_PATH)
    rows = _fetch_leaderboard(nb, args.top)
    if not rows:
        print("No leaderboard entries found.")
        return

    print(f"Processing {len(rows)} leaderboard entries...")

    # Batch-fetch program_results
    result_ids = [r["result_id"] for r in rows if r.get("result_id")]
    pr_data = prefetch_program_results(nb.conn, result_ids)

    results = []
    for row in rows:
        d = dict(row)
        rid = d.get("result_id", "")
        pr_dict = pr_data.get(rid, {})
        is_ref = bool(d.get("is_reference"))

        # Build kwargs (works for both v7 and v8 — v7 ignores extra kwargs)
        kw = build_score_kwargs_from_prefetch(dict(pr_dict), d, is_ref)

        v7_result = compute_composite_v7(decompose=True, **kw)
        v8_result = compute_composite_v8(decompose=True, **kw)

        v7_score = v7_result["composite_score"]
        v8_score = v8_result["composite_score"]

        results.append(
            {
                "result_id": rid[:12],
                "tier": d.get("tier", "?"),
                "is_reference": is_ref,
                "v7_score": v7_score,
                "v8_score": v8_score,
                "delta": v8_score - v7_score,
                "v7_breakdown": v7_result["breakdown"],
                "v8_breakdown": v8_result["breakdown"],
            }
        )

        # Update DB with v8 score
        if not args.dry_run and rid:
            nb.conn.execute(
                "UPDATE leaderboard SET composite_score = ? WHERE result_id = ?",
                (v8_score, rid),
            )

    if not args.dry_run:
        nb.conn.commit()
        print(f"Updated {len(results)} entries with v8 scores.")

    # Sort by v8 score desc for report
    results.sort(key=lambda x: x["v8_score"], reverse=True)

    if args.report:
        print("\n" + "=" * 90)
        print("V7 vs V8 SCORING COMPARISON")
        print("=" * 90)
        print(
            f"{'Rank':>4} {'ID':>12} {'Tier':<15} {'V7':>8} {'V8':>8} {'Delta':>8}  New Components"
        )
        print("-" * 90)
        for i, r in enumerate(results, 1):
            bd = r["v8_breakdown"]
            new_components = []
            for k in (
                "tinystories",
                "cross_task",
                "diagnostic",
                "hellaswag",
                "hierarchy",
            ):
                v = bd.get(k, 0)
                if v > 0.5:
                    new_components.append(f"{k}={v:.1f}")
            new_str = ", ".join(new_components) if new_components else "-"
            ref_marker = " [REF]" if r["is_reference"] else ""
            print(
                f"{i:>4} {r['result_id']:>12} {r['tier']:<15} "
                f"{r['v7_score']:>8.1f} {r['v8_score']:>8.1f} {r['delta']:>+8.1f}  {new_str}{ref_marker}"
            )

        # Summary stats
        v7_scores = [r["v7_score"] for r in results]
        v8_scores = [r["v8_score"] for r in results]
        deltas = [r["delta"] for r in results]
        print("-" * 90)
        print(
            f"{'Mean':>34} {sum(v7_scores) / len(v7_scores):>8.1f} {sum(v8_scores) / len(v8_scores):>8.1f} {sum(deltas) / len(deltas):>+8.1f}"
        )
        print(
            f"{'Max':>34} {max(v7_scores):>8.1f} {max(v8_scores):>8.1f} {max(deltas):>+8.1f}"
        )
        print(
            f"{'Min':>34} {min(v7_scores):>8.1f} {min(v8_scores):>8.1f} {min(deltas):>+8.1f}"
        )

    if args.calibrate:
        print("\n" + "=" * 60)
        print("THRESHOLD CALIBRATION")
        print("=" * 60)

        # Reference entries
        refs = [r for r in results if r["is_reference"]]
        if refs:
            ref_screening = [r["v8_score"] for r in refs if r["tier"] == "screening"]
            ref_inv = [
                r["v8_score"]
                for r in refs
                if r["tier"] in ("investigation", "validation", "breakthrough")
            ]

            if ref_screening:
                avg = sum(ref_screening) / len(ref_screening)
                print(
                    f"Reference screening avg: {avg:.1f} → V8_SCREENING_THRESHOLD = {avg * 0.90:.1f}"
                )
            if ref_inv:
                avg = sum(ref_inv) / len(ref_inv)
                print(
                    f"Reference investigation avg: {avg:.1f} → V8_INVESTIGATION_THRESHOLD = {avg * 0.90:.1f}"
                )
        else:
            # Use top-10 non-reference entries as proxy
            non_ref = [r for r in results if not r["is_reference"]]
            screening = [r["v8_score"] for r in non_ref if r["tier"] == "screening"][
                :10
            ]
            inv = [
                r["v8_score"]
                for r in non_ref
                if r["tier"] in ("investigation", "validation")
            ][:10]
            if screening:
                avg = sum(screening) / len(screening)
                print(
                    f"Top screening avg: {avg:.1f} → V8_SCREENING_THRESHOLD = {avg * 0.90:.1f}"
                )
            if inv:
                avg = sum(inv) / len(inv)
                print(
                    f"Top investigation avg: {avg:.1f} → V8_INVESTIGATION_THRESHOLD = {avg * 0.90:.1f}"
                )

        # Understanding metrics coverage
        print("\nUnderstanding metrics coverage:")
        total = len(results)
        for metric in (
            "tinystories",
            "cross_task",
            "diagnostic",
            "hellaswag",
            "hierarchy",
        ):
            has_data = sum(1 for r in results if r["v8_breakdown"].get(metric, 0) > 0.5)
            print(
                f"  {metric}: {has_data}/{total} entries have data ({100 * has_data / total:.0f}%)"
            )


if __name__ == "__main__":
    main()
