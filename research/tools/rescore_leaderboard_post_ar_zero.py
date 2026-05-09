#!/usr/bin/env python
"""Rescore the entire leaderboard after ar_legacy_auc was zeroed and ar_gate_score
was added to the scoring formula.

Calls ``research.scientist.leaderboard_rescore.rescore_leaderboard()`` which
recomputes ``composite_score`` for every leaderboard row using the current
scoring backend (compute_composite_v10 in this repo at the time of writing).
The new path threads ``ar_gate_score`` through ``_pr_dict_to_score_kwargs``
into the binding-composite + capability-tier calculations.

Rows without ar_gate backfill data simply skip the AR component (same
fallback the legacy code took when ar_legacy_auc was None) — composite_score becomes
``0.3 * induction_screening_auc + 0.3 * binding_screening_auc`` for the binding tier instead of
the previous ``0.4 * ar_legacy_auc + 0.3 * induction + 0.3 * binding``. Since the
prior ar_legacy_auc contribution was uniformly ~0.005 (V4 evidence), removing it is
a near-no-op for the un-backfilled rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "research/runs.db"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--only-stale",
        action="store_true",
        help="Skip rows already at the current scoring_config_hash.",
    )
    p.add_argument(
        "--reason",
        default="ar_legacy_auc_zeroed_ar_gate_added",
        help="Stamped into rescore_reason for audit.",
    )
    args = p.parse_args()

    from research.scientist.leaderboard_rescore import rescore_leaderboard
    from research.scientist.notebook import LabNotebook

    nb = LabNotebook(args.db)
    try:
        n_total, n_changed = rescore_leaderboard(
            nb,
            only_stale=bool(args.only_stale),
            reason=str(args.reason),
        )
        logger.info("Rescored %d rows; %d changed.", n_total, n_changed)
    finally:
        nb.close()


if __name__ == "__main__":
    main()
