"""Perplexity sweep for frontier leaderboard models.

Loads top leaderboard models, compiles them, runs wikitext perplexity
evaluation at multiple training step checkpoints (500, 750, 1000, 1500).
Stores results back into program_results.

Usage:
    python -m research.tools.ppl_sweep [--top N] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Perplexity sweep for frontier models")
    parser.add_argument(
        "--top", type=int, default=6, help="Number of top models to sweep"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--db", type=str, default="research/lab_notebook.db")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="500,750,1000,1500",
        help="Comma-separated step counts",
    )
    args = parser.parse_args()

    checkpoints = tuple(int(s) for s in args.checkpoints.split(","))
    device = args.device

    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model
    from research.eval.wikitext_eval import evaluate_wikitext_trajectory
    from research.defaults import VOCAB_SIZE

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.tier, l.composite_score,
               p.graph_fingerprint, p.graph_json, p.param_count, p.graph_n_ops
        FROM leaderboard l
        JOIN program_results p ON l.result_id = p.result_id
        WHERE COALESCE(l.is_reference, 0) = 0
          AND p.graph_json IS NOT NULL
        ORDER BY l.composite_score DESC
        LIMIT ?
    """,
        (args.top,),
    ).fetchall()

    logger.info("Sweeping %d frontier models at checkpoints %s", len(rows), checkpoints)

    for i, r in enumerate(rows):
        fp = r["graph_fingerprint"][:10]
        rid = r["result_id"]
        tier = r["tier"]
        score = r["composite_score"]

        logger.info(
            "[%d/%d] %s tier=%s score=%.1f ops=%s params=%s",
            i + 1,
            len(rows),
            fp,
            tier,
            score,
            r["graph_n_ops"],
            r["param_count"],
        )

        # Load and compile
        try:
            graph_dict = json.loads(r["graph_json"])
            graph = ComputationGraph.from_dict(graph_dict)
            n_layers = 4  # Old models used 4 layers
            layer_graphs = [graph] * n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=VOCAB_SIZE,
                max_seq_len=256,
            )
            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("  Compiled: %s params", f"{n_params:,}")
        except Exception as e:
            logger.error("  Failed to compile %s: %s", fp, e)
            continue

        # Run trajectory sweep
        try:
            t0 = time.time()
            result = evaluate_wikitext_trajectory(
                model=model,
                vocab_size=VOCAB_SIZE,
                device=device,
                checkpoints=checkpoints,
                seq_len=128,
                n_eval_batches=8,
                train_batch_size=4,
                eval_batch_size=4,
                lr=3e-4,
            )
            elapsed = time.time() - t0

            trajectory = result.get("trajectory", {})
            logger.info("  Completed in %.1fs", elapsed)
            for step, data in sorted(trajectory.items(), key=lambda x: int(x[0])):
                ppl = data.get("val_ppl", data.get("ppl"))
                logger.info("    step=%s ppl=%.2f", step, ppl if ppl else -1)

            # Store the best perplexity (highest step count) into program_results
            best_step = (
                max(trajectory.keys(), key=lambda k: int(k)) if trajectory else None
            )
            if best_step:
                best_ppl = trajectory[best_step].get(
                    "val_ppl", trajectory[best_step].get("ppl")
                )
                if best_ppl and best_ppl > 0:
                    conn.execute(
                        "UPDATE program_results SET wikitext_perplexity = ? WHERE result_id = ?",
                        (best_ppl, rid),
                    )
                    conn.commit()
                    logger.info(
                        "  Stored ppl=%.2f for %s (step %s)", best_ppl, fp, best_step
                    )

            # Also store full trajectory as JSON in a metadata column
            traj_json = json.dumps(
                {
                    "checkpoints": {str(k): v for k, v in trajectory.items()},
                    "elapsed_s": elapsed,
                    "sweep_ts": time.time(),
                }
            )
            conn.execute(
                """UPDATE program_results
                   SET wikitext_score = ?
                   WHERE result_id = ?""",
                (traj_json, rid),
            )
            conn.commit()

        except Exception as e:
            logger.error("  Sweep failed for %s: %s", fp, e)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
