"""Backfill robustness_grade and evaluation_stage for existing leaderboard rows.

This is a DERIVED-ONLY backfill — no re-running evals, no copying metrics
across tiers.  Computes robustness_grade (A/B/C) from investigation_robustness
and evaluation_stage (SCREENED/PROBED/ESCALATED/VALIDATED) from tier + existing
eval data.

Usage:
    # Dry run (mandatory first step)
    python -m research.tools.backfill_eval_stages --dry-run

    # Apply
    python -m research.tools.backfill_eval_stages

    # Limit to specific tier
    python -m research.tools.backfill_eval_stages --tier investigation --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.scientist.notebook import LabNotebook

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


def compute_robustness_grade(inv_robustness: float | None) -> str | None:
    """A: >=2/3, B: 1/3-2/3, C: <1/3, None: untested."""
    if inv_robustness is None:
        return None
    try:
        r = float(inv_robustness)
    except (TypeError, ValueError):
        return None
    if r >= 2 / 3:
        return "A"
    if r >= 1 / 3:
        return "B"
    return "C"


def compute_evaluation_stage(row: dict) -> str:
    """Derive evaluation_stage from tier and existing eval evidence.

    SCREENED  — screening tier, no real-token eval
    PROBED    — has wikitext metrics from screening
    ESCALATED — investigation tier with real-token data
    VALIDATED — validation tier
    """
    tier = row.get("tier", "screening")
    has_wikitext = row.get("wikitext_score") is not None or row.get("wikitext_perplexity") is not None

    if tier == "validation":
        return "VALIDATED"
    if tier == "investigation":
        return "ESCALATED" if has_wikitext else "PROBED"
    if tier == "screened_out":
        return "ESCALATED" if has_wikitext else "PROBED"
    # screening tier
    return "PROBED" if has_wikitext else "SCREENED"


def backfill(dry_run: bool = True, tier_filter: str | None = None) -> None:
    nb = LabNotebook()

    query = "SELECT * FROM leaderboard"
    params: list = []
    if tier_filter:
        query += " WHERE tier = ?"
        params.append(tier_filter)

    rows = nb.conn.execute(query, params).fetchall()
    log.info("Found %d leaderboard rows%s", len(rows),
             f" (tier={tier_filter})" if tier_filter else "")

    grade_updates = 0
    stage_updates = 0
    unchanged = 0

    for row in rows:
        d = dict(row)
        entry_id = d["entry_id"]
        sets: list[str] = []
        vals: list = []

        # Robustness grade
        current_grade = d.get("robustness_grade")
        new_grade = compute_robustness_grade(d.get("investigation_robustness"))
        if new_grade and current_grade != new_grade:
            sets.append("robustness_grade = ?")
            vals.append(new_grade)
            grade_updates += 1

        # Evaluation stage
        current_stage = d.get("evaluation_stage")
        new_stage = compute_evaluation_stage(d)
        if current_stage != new_stage:
            sets.append("evaluation_stage = ?")
            vals.append(new_stage)
            stage_updates += 1

        if not sets:
            unchanged += 1
            continue

        if dry_run:
            log.info(
                "  [DRY] %s: grade %s→%s, stage %s→%s",
                entry_id[:12],
                current_grade, new_grade or current_grade,
                current_stage, new_stage if current_stage != new_stage else current_stage,
            )
        else:
            vals.append(entry_id)
            nb.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                vals,
            )

    if not dry_run:
        nb.conn.commit()

    log.info(
        "Summary: %d grade updates, %d stage updates, %d unchanged",
        grade_updates, stage_updates, unchanged,
    )
    if dry_run:
        log.info("DRY RUN — no changes written. Run without --dry-run to apply.")

    nb.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill robustness_grade and evaluation_stage"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing")
    parser.add_argument("--tier", default=None,
                        help="Only process rows with this tier")
    args = parser.parse_args()

    backfill(dry_run=args.dry_run, tier_filter=args.tier)


if __name__ == "__main__":
    main()
