#!/usr/bin/env python
"""Flag (do NOT demote) AR gate-INV no-go archs (offline scan).

Diagnostic-only flag — row keeps its tier so the dashboard does not hide
existing metric data via the screened_out promotion-path filter (see
``_entry_has_promotion_path`` in ``leaderboard_bp.py``). The composite_score
already penalizes the row via ``cap_ar=0`` from ``nai=0``; that handles
ranking without removing the row from view.

  - ``hard``: both ``in_dist_pair_acc < 0.10`` AND
    ``held_class_acc < 0.10`` ⇒ frequency-collapse degenerate ⇒
    set ``ar_gate_no_go=1``. NOT demoted. NOT failure_op-stamped.

  - ``soft``: ``ar_gate_score < 0.50`` ⇒ mark ``ar_gate_no_go=0``
    (gate ran but didn't trip).

Distribution-derived thresholds (V4 evidence + 170-arch backfill data,
2026-05-05):

  | nai_band | n  | avg_inv_nb | avg_bind_v2 | avg_ind_v2 |
  |----------|----|------------|-------------|------------|
  | 0.00     | 7  | 0.92       | 0.117       | 0.048      |
  | 0.10-0.30| 0  | empty band — natural distribution gap                |
  | 0.30-0.50| 83 | 0.66       | 0.009       | 0.011      | ← frame-only
  | 0.50-0.65| 13 | 0.78       | 0.434       | 0.286      | ← partial retrievers
  | 0.65-0.80| 16 | 0.80       | 0.205       | 0.213      |
  | 0.80-0.95| 12 | 0.83       | 0.279       | 0.319      |
  | ≥ 0.95   | 39 | 0.90       | 0.072       | 0.067      |

Hard threshold (0.10 / 0.10 conjunction) targets cluster 1 only — the
unambiguous frequency-collapse archs. Soft threshold (0.50) marks the
frame-only cluster (avg_bind_v2 ≈ 0 ⇒ no v2 capability either) but
defers demotion to the tier promotion logic via composite_score.

Usage::

    python -m research.tools.ar_gate_no_go_flag --dry-run
    python -m research.tools.ar_gate_no_go_flag             # apply
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

from research.tools.check_backup_freshness import main as check_backup_freshness_main
from research.tools.db_health import backup_sqlite_db

DB_PATH = REPO / "research/runs.db"
REPORTS_DIR = REPO / "research/reports/ar_gate_no_go"

HARD_PAIR_FLOOR = 0.10
HARD_HELD_CLASS_FLOOR = 0.10
SOFT_SCORE_FLOOR = 0.50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; do not write.",
    )
    p.add_argument(
        "--include-already-screened",
        action="store_true",
        help="Re-evaluate rows already at tier='screened_out'/'retired' (default skip).",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a SQLite backup, but require a fresh existing backup.",
    )
    return p.parse_args()


def fetch_candidates(args: argparse.Namespace) -> list[dict]:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    where = "pr.ar_gate_score IS NOT NULL"
    if not args.include_already_screened:
        where += " AND lb.tier NOT IN ('screened_out', 'retired')"
    rows = conn.execute(
        f"""
        SELECT pr.result_id, lb.tier,
               pr.ar_gate_score, pr.ar_gate_in_dist_pair_acc,
               pr.ar_gate_held_class_acc, pr.ar_gate_status,
               pr.failure_op
        FROM program_results_compat pr
        JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE {where}
        ORDER BY pr.ar_gate_score ASC
        """
    ).fetchall()
    conn.close()
    return [
        {
            "result_id": r[0],
            "tier": r[1],
            "score": float(r[2]),
            "pair": float(r[3]) if r[3] is not None else None,
            "held_class": float(r[4]) if r[4] is not None else None,
            "status": r[5],
            "failure_op": r[6],
        }
        for r in rows
    ]


def classify(row: dict) -> str:
    """Return 'hard_no_go', 'soft_below', or 'pass'."""
    if row["status"] != "ok":
        return "pass"  # don't punish transient failures
    if (
        row["pair"] is not None
        and row["held_class"] is not None
        and row["pair"] < HARD_PAIR_FLOOR
        and row["held_class"] < HARD_HELD_CLASS_FLOOR
    ):
        return "hard_no_go"
    if row["score"] < SOFT_SCORE_FLOOR:
        return "soft_below"
    return "pass"


def apply_updates(args: argparse.Namespace, rows: list[dict]) -> dict:
    counts = {
        "n_total": len(rows),
        "n_hard_no_go": 0,
        "n_soft_below": 0,
        "n_pass": 0,
        "no_go_flag_set": 0,
    }
    if not rows:
        return counts

    write_conn = None if args.dry_run else sqlite3.connect(args.db, timeout=30.0)
    if write_conn is not None:
        write_conn.execute("PRAGMA journal_mode=WAL")

    updates: list[tuple[int, str]] = []
    for row in rows:
        verdict = classify(row)
        counts[f"n_{verdict}"] += 1
        flag_value = 1 if verdict == "hard_no_go" else 0
        if flag_value == 1:
            counts["no_go_flag_set"] += 1
        if write_conn is not None:
            updates.append((flag_value, row["result_id"]))

    if write_conn is not None:
        write_conn.executemany(
            "UPDATE program_results SET ar_gate_no_go = ? WHERE result_id = ?",
            updates,
        )
        write_conn.commit()
        write_conn.close()
    return counts


def main() -> None:
    args = parse_args()
    rows = fetch_candidates(args)
    logger.info(
        "Scanning %d archs with ar_gate_score (hard floor: pair<%.2f AND "
        "held_class<%.2f; soft floor: score<%.2f)",
        len(rows),
        HARD_PAIR_FLOOR,
        HARD_HELD_CLASS_FLOOR,
        SOFT_SCORE_FLOOR,
    )
    if not args.dry_run and not args.no_backup:
        backup_path = backup_sqlite_db(args.db, suffix="pre_ar_gate_no_go")
        logger.info("backup=%s", backup_path)
    elif not args.dry_run:
        rc = check_backup_freshness_main([])
        if rc != 0:
            raise SystemExit(rc)
    counts = apply_updates(args, rows)
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    logger.info("=== %s ===", mode)
    logger.info("  total scanned:    %d", counts["n_total"])
    logger.info("  hard no-go flagged (no demote): %d", counts["n_hard_no_go"])
    logger.info("  soft below (composite handles): %d", counts["n_soft_below"])
    logger.info("  pass:             %d", counts["n_pass"])
    if not args.dry_run:
        logger.info("  no_go=1 flag set: %d", counts["no_go_flag_set"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"ar_gate_no_go_{int(time.time())}.json"
    out.write_text(
        json.dumps(
            {
                "mode": mode,
                "timestamp": time.time(),
                "thresholds": {
                    "hard_pair_floor": HARD_PAIR_FLOOR,
                    "hard_held_class_floor": HARD_HELD_CLASS_FLOOR,
                    "soft_score_floor": SOFT_SCORE_FLOOR,
                },
                "counts": counts,
                "rows": [
                    {
                        "result_id": r["result_id"],
                        "tier": r["tier"],
                        "score": r["score"],
                        "pair": r["pair"],
                        "held_class": r["held_class"],
                        "verdict": classify(r),
                    }
                    for r in rows
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
