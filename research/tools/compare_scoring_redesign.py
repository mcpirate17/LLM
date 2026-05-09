#!/usr/bin/env python3
"""Dry-run report comparing persisted leaderboard scores to active scoring.

This is read-only. It treats ``leaderboard.composite_score`` as the old score
and recomputes the active formula from current DB metrics, so it can be run
before deciding whether to rescore rows in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    get_scoring_version,
    prefetch_program_results,
)
from research.tools._db_maintenance import connect_readonly

DEFAULT_DB = Path("research/runs.db")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _query_rows(conn: Any, *, limit: int, tier: str) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ["l.composite_score IS NOT NULL"]
    if tier:
        tiers = [t.strip() for t in tier.split(",") if t.strip()]
        if tiers:
            where.append(f"l.tier IN ({','.join('?' for _ in tiers)})")
            params.extend(tiers)
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT l.*
        FROM leaderboard l
        WHERE {" AND ".join(where)}
        ORDER BY l.timestamp DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--tier",
        default="validation,breakthrough",
        help="Comma-separated tier filter; empty string means all tiers.",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to print.")
    args = parser.parse_args()

    conn = connect_readonly(args.db)
    try:
        rows = _query_rows(conn, limit=args.limit, tier=args.tier)
        pr_cache = prefetch_program_results(conn, [str(r["result_id"]) for r in rows])
        report = []
        for row in rows:
            rid = str(row["result_id"])
            pr = pr_cache.get(rid)
            if not pr:
                continue
            old_score = float(row.get("composite_score") or 0.0)
            new_score = float(
                compute_composite(
                    **build_score_kwargs_from_prefetch(
                        pr,
                        row,
                        bool(row.get("is_reference")),
                    )
                )
                or 0.0
            )
            report.append(
                {
                    "result_id": rid,
                    "tier": row.get("tier"),
                    "old": old_score,
                    "new": new_score,
                    "delta": new_score - old_score,
                    "ar_validation": pr.get("ar_validation_rank_score"),
                    "ar_gate": pr.get("ar_gate_score"),
                    "ppl": pr.get("wikitext_perplexity"),
                    "validation_loss_ratio": pr.get("validation_loss_ratio"),
                }
            )
    finally:
        conn.close()

    report.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    print(f"Active scoring config: {get_scoring_version()}")
    print(f"Rows compared: {len(report)}")
    if report:
        deltas = [float(r["delta"]) for r in report]
        missing_small = sum(1 for r in report if r["ar_validation"] is None)
        print(
            f"Delta min/mean/max: {min(deltas):.2f} / {sum(deltas) / len(deltas):.2f} / {max(deltas):.2f}"
        )
        print(f"Rows missing AR Validation: {missing_small}/{len(report)}")
        print()
        print(
            "result_id    tier          old      new    delta  ar_validation ar_gate  ppl    val_lr"
        )
        for item in report[: max(0, int(args.top))]:
            print(
                f"{item['result_id'][:12]:12s} "
                f"{str(item['tier'] or '')[:12]:12s} "
                f"{_fmt(item['old']):>8s} "
                f"{_fmt(item['new']):>8s} "
                f"{_fmt(item['delta']):>8s} "
                f"{_fmt(item['ar_validation']):>8s} "
                f"{_fmt(item['ar_gate']):>7s} "
                f"{_fmt(item['ppl'], 1):>6s} "
                f"{_fmt(item['validation_loss_ratio'], 3):>7s}"
            )


if __name__ == "__main__":
    main()
