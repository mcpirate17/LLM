#!/usr/bin/env python3
"""Rank-diff report for the v2 probe backfill (2026-04-18).

Compares leaderboard rankings before and after v2 probe scoring takes effect.
Reads old_composite_score (set during backfill) and current composite_score.

Usage:
    python -m research.tools.probe_v2_rank_diff
    python -m research.tools.probe_v2_rank_diff --tier investigation,validation
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from research.tools._db_maintenance import connect_readonly

DB_PATH = "research/runs.db"


def _rank_by_score(rows: List[object], score_col: str) -> Dict[str, int]:
    ordered = sorted(rows, key=lambda r: float(r[score_col] or 0.0), reverse=True)
    return {r["entry_id"]: i + 1 for i, r in enumerate(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default="investigation,validation",
        help="Comma-separated tiers to analyze",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show top-N largest rank shifts",
    )
    args = parser.parse_args()

    tiers = tuple(t.strip() for t in args.tier.split(","))
    conn = connect_readonly(Path(DB_PATH).resolve())

    ph = ",".join("?" for _ in tiers)
    rows = conn.execute(
        f"""
        SELECT l.entry_id, l.result_id, l.tier,
               l.composite_score, l.old_composite_score,
               pr.graph_fingerprint,
               pr.induction_screening_auc, pr.binding_screening_auc, pr.ar_legacy_auc,
               pr.induction_intermediate_auc,
               pr.binding_intermediate_auc
        FROM leaderboard l
        LEFT JOIN program_results pr ON l.result_id = pr.result_id
        WHERE l.tier IN ({ph})
        """,
        tiers,
    ).fetchall()

    backfilled = [
        r
        for r in rows
        if r["induction_intermediate_auc"] is not None
        or r["binding_intermediate_auc"] is not None
    ]
    missing = [r for r in rows if r not in backfilled]

    print(f"Rows in tiers {tiers}: {len(rows)}")
    print(f"  With v2 values: {len(backfilled)}")
    print(f"  Still None:     {len(missing)}")
    print()

    if backfilled:
        before_ranks = _rank_by_score(rows, "old_composite_score")
        after_ranks = _rank_by_score(rows, "composite_score")
        shifts: List[Tuple[int, object]] = []
        for r in backfilled:
            eid = r["entry_id"]
            if eid in before_ranks and eid in after_ranks:
                delta = after_ranks[eid] - before_ranks[eid]
                shifts.append((delta, r))
        shifts.sort(key=lambda x: abs(x[0]), reverse=True)

        print(f"=== Top {min(args.top, len(shifts))} rank shifts ===")
        print(
            f"{'FP':14s} {'tier':14s} {'before':>7s} {'after':>6s} {'Δ':>5s} "
            f"{'score_old':>10s} {'score_new':>10s} "
            f"{'v1_ind':>7s} {'v2_ind':>7s} {'v1_bin':>7s} {'v2_bin':>7s}"
        )
        for delta, r in shifts[: args.top]:
            fp = (r["graph_fingerprint"] or "")[:12]
            arrow = "↓" if delta > 0 else "↑" if delta < 0 else "="
            print(
                f"{fp:14s} {r['tier']:14s} "
                f"{before_ranks[r['entry_id']]:>7d} "
                f"{after_ranks[r['entry_id']]:>6d} "
                f"{arrow}{abs(delta):>3d} "
                f"{float(r['old_composite_score'] or 0):>10.2f} "
                f"{float(r['composite_score'] or 0):>10.2f} "
                f"{_fmt(r['induction_screening_auc'])} "
                f"{_fmt(r['induction_intermediate_auc'])} "
                f"{_fmt(r['binding_screening_auc'])} "
                f"{_fmt(r['binding_intermediate_auc'])}"
            )
        print()

        large = [s for s in shifts if abs(s[0]) >= 5]
        print(
            f"Entries shifting ≥5 ranks: {len(large)} "
            f"({100 * len(large) / max(len(shifts), 1):.1f}%)"
        )
        print(
            "Gate (plan): pause rollout if >10% of rows shift ≥5 ranks. "
            f"Observed: {100 * len(large) / max(len(shifts), 1):.1f}%"
        )

        # Magnitude summary
        score_deltas = [
            float(r["composite_score"] or 0) - float(r["old_composite_score"] or 0)
            for _, r in shifts
        ]
        if score_deltas:
            print()
            print("=== Score delta (new - old) ===")
            print(f"  min:    {min(score_deltas):.2f}")
            print(f"  max:    {max(score_deltas):.2f}")
            print(f"  mean:   {sum(score_deltas) / len(score_deltas):.2f}")
            median_val = sorted(score_deltas)[len(score_deltas) // 2]
            print(f"  median: {median_val:.2f}")

    conn.close()


def _fmt(v) -> str:
    if v is None:
        return "   —   "
    return f"{float(v):>7.3f}"


if __name__ == "__main__":
    main()
