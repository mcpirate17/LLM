"""Diff a leaderboard snapshot CSV against the current leaderboard.

Used to quantify the scoring impact of a backfill (e.g. BPE eval rewrite,
v10 scoring rollout). The snapshot CSV is produced ahead of the change;
this tool reads it back, joins on ``entry_id``, and reports composite-score
deltas with a tier-level breakdown plus the largest movers.

Usage::

    python -m research.tools.diff_leaderboard_snapshot \\
        --snapshot research/perf_artifacts/leaderboard_snapshot_pre_bpe_<ts>.csv \\
        --tier-filter validation,breakthrough \\
        --top-n 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path


def _f(x, default=None):
    if x in (None, "", "None"):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument(
        "--tier-filter", default=None,
        help="Comma-separated tiers to restrict the diff to (default: all).",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.snapshot.exists():
        print(f"snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(2)

    tier_filter = None
    if args.tier_filter:
        tier_filter = {t.strip() for t in args.tier_filter.split(",") if t.strip()}

    pre = {}
    with args.snapshot.open() as f:
        for row in csv.DictReader(f):
            pre[row["entry_id"]] = row

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    # Read metrics from program_results (the truth) not leaderboard's
    # stale mirror columns — backfills update program_results without
    # propagating to leaderboard.{wikitext_perplexity,blimp,...}.
    cur_rows = conn.execute(
        """
        SELECT lb.entry_id, lb.result_id, lb.tier, lb.composite_score,
               pr.wikitext_perplexity, pr.blimp_overall_accuracy,
               pr.hellaswag_acc, pr.tinystories_perplexity,
               lb.scoring_version
        FROM leaderboard lb
        LEFT JOIN program_results pr ON pr.result_id = lb.result_id
        """
    ).fetchall()

    deltas = []
    metric_changes = []
    for row in cur_rows:
        eid = str(row["entry_id"])
        if eid not in pre:
            continue
        if tier_filter and row["tier"] not in tier_filter:
            continue
        pre_row = pre[eid]
        pre_score = _f(pre_row["composite_score"], 0.0)
        post_score = _f(row["composite_score"], 0.0)
        delta = post_score - pre_score
        deltas.append({
            "entry_id": eid,
            "result_id": str(row["result_id"]),
            "tier": row["tier"],
            "pre_score": pre_score,
            "post_score": post_score,
            "delta": delta,
            "pre_wt_ppl": _f(pre_row["wikitext_perplexity"]),
            "post_wt_ppl": _f(row["wikitext_perplexity"]),
            "pre_blimp": _f(pre_row["blimp_overall_accuracy"]),
            "post_blimp": _f(row["blimp_overall_accuracy"]),
            "pre_hella": _f(pre_row["hellaswag_acc"]),
            "post_hella": _f(row["hellaswag_acc"]),
            "pre_ts": _f(pre_row["tinystories_perplexity"]),
            "post_ts": _f(row["tinystories_perplexity"]),
        })
        if any(
            _f(pre_row[k]) != _f(row[k])
            for k in ("wikitext_perplexity", "blimp_overall_accuracy",
                     "hellaswag_acc", "tinystories_perplexity")
            if pre_row[k] is not None or row[k] is not None
        ):
            metric_changes.append(eid)
    conn.close()

    if not deltas:
        print("No matching rows.")
        return

    score_deltas = [d["delta"] for d in deltas]
    abs_deltas = [abs(d) for d in score_deltas]
    n_changed = sum(1 for d in score_deltas if abs(d) > 1e-6)

    by_tier = {}
    for d in deltas:
        by_tier.setdefault(d["tier"], []).append(d["delta"])

    # Ranking shift on validation tier (reorder by post_score, see how many positions each row moved)
    val_rows = [d for d in deltas if d["tier"] in ("validation", "breakthrough")]
    val_rows_pre_rank = sorted(val_rows, key=lambda r: -r["pre_score"])
    val_rows_post_rank = sorted(val_rows, key=lambda r: -r["post_score"])
    pre_rank = {r["entry_id"]: i for i, r in enumerate(val_rows_pre_rank, 1)}
    post_rank = {r["entry_id"]: i for i, r in enumerate(val_rows_post_rank, 1)}
    rank_shifts = [
        (eid, pre_rank[eid], post_rank[eid], post_rank[eid] - pre_rank[eid])
        for eid in pre_rank
    ]
    biggest_climbers = sorted(rank_shifts, key=lambda x: x[3])[:args.top_n]
    biggest_droppers = sorted(rank_shifts, key=lambda x: -x[3])[:args.top_n]

    sorted_by_delta = sorted(deltas, key=lambda d: -abs(d["delta"]))
    biggest_movers = sorted_by_delta[:args.top_n]

    report = {
        "snapshot": str(args.snapshot),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tier_filter": sorted(tier_filter) if tier_filter else "all",
        "total_compared": len(deltas),
        "n_metric_changed": len(metric_changes),
        "n_score_changed": n_changed,
        "score_delta": {
            "mean": statistics.mean(score_deltas),
            "median": statistics.median(score_deltas),
            "stdev": statistics.stdev(score_deltas) if len(score_deltas) > 1 else 0.0,
            "min": min(score_deltas),
            "max": max(score_deltas),
            "abs_mean": statistics.mean(abs_deltas),
            "abs_median": statistics.median(abs_deltas),
            "abs_p90": statistics.quantiles(abs_deltas, n=10)[-1] if len(abs_deltas) >= 10 else max(abs_deltas),
        },
        "by_tier": {
            t: {
                "n": len(vs),
                "mean": statistics.mean(vs),
                "median": statistics.median(vs),
                "min": min(vs),
                "max": max(vs),
            }
            for t, vs in by_tier.items()
        },
        "biggest_score_movers": [
            {
                "entry_id": d["entry_id"], "tier": d["tier"],
                "pre": round(d["pre_score"], 1), "post": round(d["post_score"], 1),
                "delta": round(d["delta"], 1),
                "wt_ppl": [d["pre_wt_ppl"], d["post_wt_ppl"]],
                "blimp": [d["pre_blimp"], d["post_blimp"]],
                "hella": [d["pre_hella"], d["post_hella"]],
                "ts_ppl": [d["pre_ts"], d["post_ts"]],
            }
            for d in biggest_movers
        ],
        "validation_tier_rank_climbers": [
            {"entry_id": e, "pre_rank": p, "post_rank": q, "moved": -m}
            for e, p, q, m in biggest_climbers
        ],
        "validation_tier_rank_droppers": [
            {"entry_id": e, "pre_rank": p, "post_rank": q, "moved": m}
            for e, p, q, m in biggest_droppers
        ],
    }

    print(json.dumps(report, indent=2, default=str))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[wrote] {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
