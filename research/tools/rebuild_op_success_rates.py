#!/usr/bin/env python3
"""Rebuild op_success_rates from full program_results history.

Problem: The op_success_rates table was polluted by 974 S1 failures that
had good learning (raw loss_ratio < 0.18) but were killed by the absolute
baseline gate (final_loss > 10.94) due to high initial loss from unscaled
projection chains. The grammar feedback loop treated these as evidence
that their ops are bad, when the ops are fine — the initialization was
the problem.

Fix: Rebuild the table from scratch, excluding programs whose S1 failure
was caused by initialization (high initial loss), not by bad ops.

Exclusion criteria for "init-poisoned" S1 failures:
  - stage0_passed = 1 AND stage1_passed = 0  (passed compilation, failed S1)
  - initial_loss > 50  (the S0.75 threshold — these are init-hostile)
  - final_loss / initial_loss < 0.50  (the model WAS learning)

These programs provide no signal about op quality because their failure
was deterministic given the initialization, regardless of which ops were used.

Usage:
    python -m research.tools.rebuild_op_success_rates [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "research/lab_notebook.db"

# Programs with initial_loss above this AND raw ratio below LEARNING_RATIO_CAP
# are excluded — their S1 failure was caused by initialization, not bad ops.
INIT_LOSS_THRESHOLD = 50.0
LEARNING_RATIO_CAP = 0.50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing"
    )
    parser.add_argument(
        "--db", default=DB_PATH, help=f"Database path (default: {DB_PATH})"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row

    # ── Phase 0: Fix mismarked stage0_passed in program_results ────────
    # Bug: results_analysis.py set stage0_passed = (fitness > 0) for
    # evolution/novelty search results. Programs that compiled and trained
    # (stage_at_death='stage1', stability_score > 0) but failed S1 got
    # fitness=0 → stage0_passed=0. Fix: if it reached S1, it passed S0.
    mismarked = conn.execute("""
        SELECT COUNT(*) FROM program_results
        WHERE stage0_passed = 0
          AND stability_score IS NOT NULL AND stability_score > 0
          AND stage_at_death = 'stage1'
    """).fetchone()[0]
    logger.info(f"Mismarked records (s0=0 but reached S1): {mismarked}")

    if mismarked > 0 and not args.dry_run:
        conn.execute("""
            UPDATE program_results
            SET stage0_passed = 1, stage05_passed = 1
            WHERE stage0_passed = 0
              AND stability_score IS NOT NULL AND stability_score > 0
              AND stage_at_death = 'stage1'
        """)
        conn.commit()
        logger.info(f"Fixed {mismarked} mismarked records: stage0_passed → 1")
    elif mismarked > 0:
        logger.info(f"[DRY RUN] Would fix {mismarked} mismarked records")

    # ── Phase 1: Count what we're excluding ────────────────────────────
    total = conn.execute(
        "SELECT COUNT(*) FROM program_results WHERE graph_json IS NOT NULL"
    ).fetchone()[0]
    init_poisoned = conn.execute(
        """
        SELECT COUNT(*) FROM program_results
        WHERE stage0_passed = 1 AND stage1_passed = 0
          AND initial_loss IS NOT NULL AND initial_loss > ?
          AND final_loss IS NOT NULL AND initial_loss > 0
          AND (final_loss / initial_loss) < ?
          AND graph_json IS NOT NULL
    """,
        (INIT_LOSS_THRESHOLD, LEARNING_RATIO_CAP),
    ).fetchone()[0]

    logger.info("=== Rebuild op_success_rates ===")
    logger.info(f"Total program_results with graph_json: {total}")
    logger.info(f"Init-poisoned S1 failures to exclude:  {init_poisoned}")
    logger.info(f"Programs to include:                    {total - init_poisoned}")
    logger.info(
        f"Exclusion criteria: initial_loss > {INIT_LOSS_THRESHOLD} "
        f"AND raw_ratio < {LEARNING_RATIO_CAP}"
    )

    # ── Phase 2: Rebuild from scratch ─────────────────────────────────
    rows = conn.execute("""
        SELECT graph_json, stage0_passed, stage05_passed, stage1_passed,
               loss_ratio, novelty_score, novelty_confidence,
               initial_loss, final_loss
        FROM program_results
        WHERE graph_json IS NOT NULL
    """).fetchall()

    op_stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "n_used": 0,
            "n_s0": 0,
            "n_s05": 0,
            "n_s1": 0,
            "lr_sum": 0.0,
            "lr_n": 0,
            "nov_sum": 0.0,
            "nov_n": 0,
            "nov_conf_sum": 0.0,
            "nov_conf_n": 0,
        }
    )

    included = 0
    excluded = 0

    for r in rows:
        graph_json = r["graph_json"]
        s0 = r["stage0_passed"]
        s05 = r["stage05_passed"]
        s1 = r["stage1_passed"]
        lr = r["loss_ratio"]
        nov = r["novelty_score"]
        nov_conf = r["novelty_confidence"]
        init_loss = r["initial_loss"]
        final_loss = r["final_loss"]

        # Exclude init-poisoned S1 failures
        if (
            s0
            and not s1
            and init_loss is not None
            and init_loss > INIT_LOSS_THRESHOLD
            and final_loss is not None
            and init_loss > 0
            and (final_loss / init_loss) < LEARNING_RATIO_CAP
        ):
            excluded += 1
            continue

        try:
            graph_data = json.loads(graph_json)
            nodes = graph_data.get("nodes", {})
        except (json.JSONDecodeError, TypeError):
            continue

        ops_in_graph = set()
        for node_data in nodes.values():
            op_name = node_data.get("op_name", "")
            if op_name and op_name != "input":
                ops_in_graph.add(op_name)

        for op_name in ops_in_graph:
            stats = op_stats[op_name]
            stats["n_used"] += 1
            if s0:
                stats["n_s0"] += 1
            if s05:
                stats["n_s05"] += 1
            if s1:
                stats["n_s1"] += 1
            if lr is not None:
                stats["lr_sum"] += lr
                stats["lr_n"] += 1
            if nov is not None:
                stats["nov_sum"] += nov
                stats["nov_n"] += 1
            if nov_conf is not None:
                stats["nov_conf_sum"] += nov_conf
                stats["nov_conf_n"] += 1

        included += 1

    logger.info(f"\nProcessed: {included} included, {excluded} excluded")
    logger.info(f"Ops found: {len(op_stats)}")

    # ── Phase 3: Show before/after comparison ─────────────────────────
    old_rows = conn.execute("""
        SELECT op_name, n_used, n_stage1_passed,
               CAST(n_stage1_passed AS FLOAT) / NULLIF(n_used, 0) AS s1_rate
        FROM op_success_rates
        ORDER BY n_used DESC
    """).fetchall()
    old_rates = {r["op_name"]: r for r in old_rows}

    total_new_used = sum(s["n_used"] for s in op_stats.values())
    total_new_s1 = sum(s["n_s1"] for s in op_stats.values())
    total_old_used = sum(r["n_used"] for r in old_rows)
    total_old_s1 = sum(r["n_stage1_passed"] for r in old_rows)

    logger.info(f"\n{'':30s} {'OLD':>15s}  {'NEW':>15s}")
    logger.info(f"{'Total usage':30s} {total_old_used:>15d}  {total_new_used:>15d}")
    logger.info(f"{'Total S1 passes':30s} {total_old_s1:>15d}  {total_new_s1:>15d}")
    logger.info(
        f"{'Global S1 rate':30s} "
        f"{total_old_s1 / max(total_old_used, 1) * 100:>14.1f}%  "
        f"{total_new_s1 / max(total_new_used, 1) * 100:>14.1f}%"
    )

    logger.info(
        f"\n{'op_name':30s} {'old_used':>8s} {'old_s1%':>7s}  {'new_used':>8s} {'new_s1%':>7s}"
    )
    logger.info("-" * 70)
    for op_name in sorted(
        op_stats.keys(), key=lambda k: op_stats[k]["n_used"], reverse=True
    )[:25]:
        stats = op_stats[op_name]
        new_rate = stats["n_s1"] / max(stats["n_used"], 1) * 100
        old = old_rates.get(op_name)
        if old:
            old_rate = (old["s1_rate"] or 0) * 100
            logger.info(
                f"{op_name:30s} {old['n_used']:8d} {old_rate:6.1f}%  "
                f"{stats['n_used']:8d} {new_rate:6.1f}%"
            )
        else:
            logger.info(
                f"{op_name:30s} {'—':>8s} {'—':>7s}  {stats['n_used']:8d} {new_rate:6.1f}%"
            )

    if args.dry_run:
        logger.info("\n[DRY RUN] No changes written. Remove --dry-run to apply.")
        conn.close()
        return

    # ── Phase 4: Truncate and rewrite ─────────────────────────────────
    conn.execute("DELETE FROM op_success_rates")
    now = time.time()
    for op_name, stats in op_stats.items():
        avg_lr = stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
        avg_nov = stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
        avg_nov_conf = (
            stats["nov_conf_sum"] / stats["nov_conf_n"] if stats["nov_conf_n"] else None
        )
        conn.execute(
            """INSERT INTO op_success_rates
               (op_name, n_used, n_stage0_passed, n_stage05_passed,
                n_stage1_passed, avg_loss_ratio, avg_novelty,
                avg_novelty_confidence, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op_name,
                stats["n_used"],
                stats["n_s0"],
                stats["n_s05"],
                stats["n_s1"],
                avg_lr,
                avg_nov,
                avg_nov_conf,
                now,
            ),
        )
    conn.commit()
    conn.close()
    logger.info(f"\nDone. Wrote {len(op_stats)} ops to op_success_rates.")


if __name__ == "__main__":
    main()
