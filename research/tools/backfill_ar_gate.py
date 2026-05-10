#!/usr/bin/env python
"""Backfill ``ar_gate_*`` columns on a fingerprint cohort.

Cohort (default): 1 breakthrough + 43 investigation + 62 investigation_failed
(no error_op) + top-N validation (default 51) by composite_score = ~157 rows.

For each row, runs ``ar_gate(graph_json=..., from_s1=False)`` with the
locked V4 config (wikitext warmup 2500 + finetune 400). On cuda this is
~20s/arch — full cohort completes in roughly an hour.

Writes results directly to ``program_results`` via UPDATE (no new rows
created). Skips rows that already have ``ar_gate_score IS NOT NULL``
unless ``--force`` is passed.

Prints a running per-arch summary plus a final tier breakdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

DB_PATH = REPO / "research/runs.db"
REPORTS_DIR = REPO / "research/reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--validation-top-n",
        type=int,
        default=51,
        help="How many top-by-composite validation rows to include.",
    )
    p.add_argument(
        "--include-investigation-failed",
        action="store_true",
        default=True,
        help="Include investigation_failed rows that have no failure_op (default True).",
    )
    p.add_argument(
        "--include-screened-out",
        action="store_true",
        default=False,
        help="Include screened_out rows (mostly nano_bind degenerates; predictable).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows that already have ar_gate_score persisted.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cohort + counts without running the probe.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap total cohort size.")
    return p.parse_args()


def fetch_cohort(args: argparse.Namespace) -> list[dict]:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows: list[dict] = []
    seen: set[str] = set()

    def _consume(query: str, params: tuple = ()) -> None:
        for row in conn.execute(query, params).fetchall():
            rid = row[0]
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(
                {
                    "result_id": rid,
                    "tier": row[1],
                    "composite_score": float(row[2] or 0),
                    "wikitext_perplexity": float(row[3] or 0),
                    "graph_json": resolve_graph_json_value(conn, args.db, row[4]),
                    "existing_score": row[5],
                }
            )

    base_select = (
        "SELECT pr.result_id, lb.tier, COALESCE(lb.composite_score, 0), "
        "       COALESCE(pr.wikitext_perplexity, 0), pr.graph_json, "
        "       pr.ar_gate_score "
        "FROM program_results_compat pr JOIN leaderboard lb ON lb.result_id = pr.result_id "
        "WHERE pr.graph_json IS NOT NULL"
    )

    # 1. breakthrough + investigation
    _consume(
        base_select
        + " AND lb.tier IN ('breakthrough', 'investigation')"
        + " ORDER BY lb.composite_score DESC"
    )

    # 2. investigation_failed (no failure_op / error_type)
    if args.include_investigation_failed:
        _consume(
            base_select
            + " AND lb.tier = 'investigation_failed'"
            + " AND (pr.failure_op IS NULL OR pr.failure_op = '')"
            + " AND (pr.error_type IS NULL OR pr.error_type = '')"
        )

    # 3. top-N validation by composite
    if args.validation_top_n > 0:
        _consume(
            base_select
            + " AND lb.tier = 'validation'"
            + " ORDER BY lb.composite_score DESC LIMIT ?",
            (int(args.validation_top_n),),
        )

    # 4. screened_out (optional; mostly nb_fail degenerates)
    if args.include_screened_out:
        _consume(
            base_select
            + " AND lb.tier = 'screened_out'"
            + " ORDER BY lb.composite_score DESC"
        )

    conn.close()

    if not args.force:
        rows = [r for r in rows if r["existing_score"] is None]

    if args.limit is not None:
        rows = rows[: int(args.limit)]
    return rows


_UPDATE_SQL = """
UPDATE program_results SET
    ar_gate_metric_version = ?,
    ar_gate_in_dist_pair_acc = ?,
    ar_gate_in_dist_class_acc = ?,
    ar_gate_held_pair_acc = ?,
    ar_gate_held_class_acc = ?,
    ar_gate_score = ?,
    ar_gate_status = ?,
    ar_gate_elapsed_ms = ?,
    ar_gate_train_steps_done = ?
WHERE result_id = ?
"""


def persist_result(conn: sqlite3.Connection, result_id: str, r) -> None:
    score = round(0.6 * r.in_dist_pair_acc + 0.4 * r.held_class_acc, 4)
    conn.execute(
        _UPDATE_SQL,
        (
            r.metric_version,
            r.in_dist_pair_acc,
            r.in_dist_class_acc,
            r.held_pair_acc,
            r.held_class_acc,
            score,
            r.status,
            r.elapsed_ms,
            r.finetune_steps_done,
            result_id,
        ),
    )
    conn.commit()


def main() -> None:
    args = parse_args()
    cohort = fetch_cohort(args)

    by_tier: dict[str, int] = {}
    for r in cohort:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    logger.info("Cohort size: %d", len(cohort))
    for tier, n in sorted(by_tier.items()):
        logger.info("  %s: %d", tier, n)

    if args.dry_run:
        logger.info("Dry run — exiting without running probe or writes.")
        return

    if not cohort:
        logger.info("Cohort empty — nothing to do.")
        return

    # Open writeable connection (separate from the readonly one used in fetch).
    write_conn = sqlite3.connect(args.db, timeout=30.0)
    write_conn.execute("PRAGMA journal_mode=WAL")

    from research.eval.ar_gate import ARGateConfig, ar_gate

    cfg = ARGateConfig(
        seed=int(args.seed),
        wikitext_warmup_steps=2500,
        finetune_steps=400,
        n_pairs_per_noun=1,
        reps=10,
        n_distractors=480,
        n_adjectives=20,
        n_objects=25,
        timeout_s=600.0,
        from_s1=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_log: list[dict] = []
    t0 = time.perf_counter()
    for i, arch in enumerate(cohort, start=1):
        rid = arch["result_id"]
        rid_short = rid[:12]
        try:
            r = ar_gate(graph_json=arch["graph_json"], device=args.device, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%d/%d] %s probe error: %s", i, len(cohort), rid_short, exc)
            continue
        score = round(0.6 * r.in_dist_pair_acc + 0.4 * r.held_class_acc, 4)
        out_log.append(
            {
                "result_id": rid,
                "tier": arch["tier"],
                "composite_score": arch["composite_score"],
                "wikitext_perplexity": arch["wikitext_perplexity"],
                "in_pair": r.in_dist_pair_acc,
                "in_class": r.in_dist_class_acc,
                "held_pair": r.held_pair_acc,
                "held_class": r.held_class_acc,
                "ar_gate_score": score,
                "status": r.status,
                "elapsed_ms": r.elapsed_ms,
            }
        )
        try:
            persist_result(write_conn, rid, r)
        except sqlite3.Error as exc:
            logger.error("DB write failed for %s: %s", rid_short, exc)
            continue
        logger.info(
            "[%d/%d] %s tier=%s comp=%.0f → score=%.2f (in_pair=%.2f held_class=%.2f) status=%s",
            i,
            len(cohort),
            rid_short,
            arch["tier"],
            arch["composite_score"],
            score,
            r.in_dist_pair_acc,
            r.held_class_acc,
            r.status,
        )

    elapsed = round(time.perf_counter() - t0, 1)
    write_conn.close()

    # Tier-level summary
    summary: dict[str, dict[str, float]] = {}
    for entry in out_log:
        tier = entry["tier"]
        bucket = summary.setdefault(
            tier,
            {"n": 0, "score_sum": 0.0, "in_pair_sum": 0.0, "held_class_sum": 0.0},
        )
        bucket["n"] += 1
        bucket["score_sum"] += entry["ar_gate_score"]
        bucket["in_pair_sum"] += entry["in_pair"]
        bucket["held_class_sum"] += entry["held_class"]

    logger.info("=== Backfill complete in %.1fs ===", elapsed)
    for tier, bucket in sorted(summary.items()):
        n = bucket["n"]
        if n == 0:
            continue
        logger.info(
            "  %s: n=%d score_avg=%.2f in_pair_avg=%.2f held_class_avg=%.2f",
            tier,
            int(n),
            bucket["score_sum"] / n,
            bucket["in_pair_sum"] / n,
            bucket["held_class_sum"] / n,
        )

    out_path = REPORTS_DIR / f"ar_gate_backfill_{int(time.time())}.json"
    out_path.write_text(
        json.dumps({"elapsed_s": elapsed, "rows": out_log}, indent=2, default=str)
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
