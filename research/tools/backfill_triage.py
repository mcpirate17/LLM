"""Backfill triage evals for existing S1-passing leaderboard entries.

Recompiles each graph, runs triage analysis, and fills in NULL leaderboard
columns for routing census, compression ratio, activation sparsity, and
param efficiency.

Does NOT retrain or modify loss metrics. Only writes to leaderboard columns
that are currently NULL.

Usage:
    python -m research.tools.backfill_triage                 # all S1 passers
    python -m research.tools.backfill_triage --limit 50      # first 50
    python -m research.tools.backfill_triage --dry-run       # preview only
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill triage evals for leaderboard entries"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max entries to process (0=all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if fields already populated"
    )
    args = parser.parse_args()

    from research.scientist.notebook import LabNotebook
    from research.synthesis.serializer import graph_from_json
    from research.synthesis.compiler import compile_model
    from research.scientist.runner.execution_triage import run_triage

    nb = LabNotebook("research/lab_notebook.db")

    # Find S1-passing entries that need triage
    where_clause = "WHERE l.screening_passed = 1"
    if not args.force:
        # Only entries missing triage fields
        where_clause += (
            " AND (l.compression_ratio IS NULL OR l.activation_sparsity_score IS NULL)"
        )

    limit_clause = f" LIMIT {args.limit}" if args.limit > 0 else ""

    rows = nb.conn.execute(
        f"SELECT l.result_id, l.composite_score, pr.graph_json, pr.loss_ratio, "
        f"pr.initial_loss, pr.final_loss "
        f"FROM leaderboard l "
        f"JOIN program_results pr ON l.result_id = pr.result_id "
        f"{where_clause} "
        f"ORDER BY l.composite_score DESC{limit_clause}"
    ).fetchall()

    logger.info(
        "Found %d entries to backfill%s",
        len(rows),
        " (dry run)" if args.dry_run else "",
    )

    updated = 0
    failed = 0
    skipped = 0

    for i, row in enumerate(rows):
        result_id = row["result_id"]
        graph_json_str = row["graph_json"]
        loss_ratio = row["loss_ratio"]

        if not graph_json_str:
            skipped += 1
            continue

        try:
            # Parse graph
            json.loads(graph_json_str)
            graph = graph_from_json(graph_json_str)

            # Detect model_dim from graph
            model_dim = getattr(graph, "model_dim", 64)

            # Compile model (no training, just for param analysis)
            model = compile_model([graph], vocab_size=256, max_seq_len=128)

            # Build result dict for triage
            result = {
                "loss_ratio": loss_ratio,
                "initial_loss": row["initial_loss"],
                "final_loss": row["final_loss"],
                "stage1_passed": True,
            }

            # Run triage
            triage = run_triage(model, graph, result, model_dim=model_dim)

            if not triage:
                skipped += 1
                continue

            if args.dry_run:
                logger.info(
                    "  [%d/%d] %s: would write %d fields: %s",
                    i + 1,
                    len(rows),
                    result_id[:12],
                    len(triage),
                    {k: v for k, v in triage.items() if k != "scaling_confidence"},
                )
                updated += 1
                continue

            # Write to leaderboard (only non-NULL fields)
            # Fetch existing tier to avoid tier-downgrade warnings
            existing = nb.conn.execute(
                "SELECT tier, model_source FROM leaderboard WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            existing_tier = existing["tier"] if existing else "screening"
            existing_source = (
                existing["model_source"] if existing else "backfill_triage"
            )
            nb.upsert_leaderboard(
                result_id=result_id,
                model_source=existing_source,
                tier=existing_tier,
                **triage,
            )
            updated += 1

            if (i + 1) % 25 == 0:
                logger.info(
                    "  Progress: %d/%d (updated=%d, failed=%d)",
                    i + 1,
                    len(rows),
                    updated,
                    failed,
                )

            # Clean up GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("  [%d] %s failed: %s", i + 1, result_id[:12], e)

    logger.info(
        "Backfill complete: %d updated, %d failed, %d skipped out of %d total",
        updated,
        failed,
        skipped,
        len(rows),
    )


if __name__ == "__main__":
    main()
