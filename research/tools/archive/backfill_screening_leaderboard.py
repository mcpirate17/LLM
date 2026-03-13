#!/usr/bin/env python3
"""Backfill leaderboard entries for S1-passing programs that were never upserted.

Usage:
    python -m research.tools.backfill_screening_leaderboard --dry-run
    python -m research.tools.backfill_screening_leaderboard
    python -m research.tools.backfill_screening_leaderboard --limit 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.scientist.notebook import LabNotebook


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill screening leaderboard entries")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to backfill (0=all)")
    parser.add_argument("--db", default="research/lab_notebook.db", help="Database path")
    args = parser.parse_args()

    nb = LabNotebook(args.db)

    # Find S1 survivors with no leaderboard entry
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    orphans = nb.conn.execute(f"""
        SELECT pr.result_id, pr.graph_fingerprint, pr.loss_ratio, pr.model_source,
               pr.novelty_score, pr.novelty_confidence,
               pr.fp_jacobian_spectral_norm, pr.routing_savings_ratio,
               pr.activation_sparsity_score, pr.depth_savings_ratio,
               pr.compression_ratio
        FROM program_results pr
        WHERE pr.stage1_passed = 1
          AND pr.graph_fingerprint IS NOT NULL
          AND pr.result_id NOT IN (SELECT result_id FROM leaderboard)
        ORDER BY pr.loss_ratio ASC NULLS LAST
        {limit_clause}
    """).fetchall()

    print(f"Found {len(orphans)} S1 survivors without leaderboard entries")

    if args.dry_run:
        for i, row in enumerate(orphans[:20]):
            print(f"  [{i+1}] fp={row[1][:14]} lr={row[2] or 0:.4f} src={row[3]}")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")
        return

    created = 0
    for row in orphans:
        try:
            nb.upsert_leaderboard(
                result_id=row[0],
                model_source=row[3] or "graph_synthesis",
                architecture_desc=(row[1] or "")[:40],
                screening_loss_ratio=row[2],
                screening_novelty=row[4],
                screening_passed=True,
                tier="screening",
                novelty_confidence=row[5],
                fp_jacobian_spectral_norm=row[6],
                routing_savings_ratio=row[7],
                activation_sparsity_score=row[8],
                depth_savings_ratio=row[9],
                compression_ratio=row[10],
            )
            created += 1
        except Exception as e:
            print(f"  Failed for {row[0][:12]}: {e}")

    nb.conn.commit()
    nb.close()
    print(f"Created {created} leaderboard entries")


if __name__ == "__main__":
    main()
