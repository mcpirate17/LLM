#!/usr/bin/env python3
"""Run wiki103 ppl at investigation (2500) and validation (10000) step counts.

Usage:
    python -m research.tools.ppl_sweep_investigation --fingerprint c9c7075e741a8790
    python -m research.tools.ppl_sweep_investigation --all-investigated
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


def sweep_fingerprint(
    conn: sqlite3.Connection,
    fingerprint: str,
    checkpoints: tuple[int, ...],
    device: str,
    n_layers: int = 6,
) -> dict | None:
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model
    from research.eval.wikitext_eval import evaluate_wikitext_trajectory
    from research.defaults import VOCAB_SIZE

    row = conn.execute(
        "SELECT result_id, graph_json, param_count FROM program_results "
        "WHERE graph_fingerprint = ? LIMIT 1",
        (fingerprint,),
    ).fetchone()
    if not row:
        logger.error("Fingerprint %s not found", fingerprint)
        return None

    result_id = row["result_id"]
    graph = ComputationGraph.from_dict(json.loads(row["graph_json"]))
    model = compile_model([graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=256)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Compiled %s: %s params, %d layers", fingerprint[:10], f"{n_params:,}", n_layers
    )

    t0 = time.time()
    result = evaluate_wikitext_trajectory(
        model=model,
        vocab_size=VOCAB_SIZE,
        device=device,
        checkpoints=checkpoints,
        seq_len=128,
        n_eval_batches=16,
        train_batch_size=4,
        eval_batch_size=4,
        lr=3e-4,
    )
    elapsed = time.time() - t0
    trajectory = result.get("trajectory", {})

    logger.info("Completed in %.1fs", elapsed)
    for step in sorted(trajectory.keys(), key=lambda k: int(k)):
        data = trajectory[step]
        ppl = data.get("val_ppl", data.get("ppl"))
        logger.info("  step=%s ppl=%.2f", step, ppl if ppl else -1)

    # Store results
    if not row["param_count"] or row["param_count"] == 0:
        conn.execute(
            "UPDATE program_results SET param_count = ? WHERE result_id = ?",
            (n_params, result_id),
        )

    # Store ppl at specific checkpoints
    for step_key, data in trajectory.items():
        step_num = int(step_key)
        ppl = data.get("val_ppl", data.get("ppl"))
        if not ppl or ppl <= 0:
            continue
        if step_num == 500:
            conn.execute(
                "UPDATE program_results SET wikitext_ppl_500 = ? WHERE result_id = ?",
                (ppl, result_id),
            )
        elif step_num == 200:
            conn.execute(
                "UPDATE program_results SET wikitext_ppl_200 = ? WHERE result_id = ?",
                (ppl, result_id),
            )

    # Store best (highest step) as wikitext_perplexity
    best_step = max(trajectory.keys(), key=lambda k: int(k)) if trajectory else None
    if best_step:
        best_ppl = trajectory[best_step].get(
            "val_ppl", trajectory[best_step].get("ppl")
        )
        if best_ppl and best_ppl > 0:
            conn.execute(
                "UPDATE program_results SET wikitext_perplexity = ?, "
                "wikitext_eval_steps = ? WHERE result_id = ?",
                (best_ppl, int(best_step), result_id),
            )

    conn.commit()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "result_id": result_id,
        "trajectory": trajectory,
        "n_params": n_params,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", type=str, help="Single fingerprint to sweep")
    parser.add_argument(
        "--all-investigated",
        action="store_true",
        help="Sweep all investigation+ entries",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--db", type=str, default="research/lab_notebook.db")
    parser.add_argument("--checkpoints", type=str, default="100,500,1000,2500,10000")
    parser.add_argument("--n-layers", type=int, default=6)
    args = parser.parse_args()

    checkpoints = tuple(int(s) for s in args.checkpoints.split(","))

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    if args.fingerprint:
        result = sweep_fingerprint(
            conn, args.fingerprint, checkpoints, args.device, args.n_layers
        )
        if result:
            logger.info("Done: %s", result["result_id"])
    elif args.all_investigated:
        rows = conn.execute("""
            SELECT p.graph_fingerprint, l.tier, l.composite_score, l.entry_id,
                   p.wikitext_eval_steps
            FROM leaderboard l
            JOIN program_results p ON l.result_id = p.result_id
            WHERE l.tier IN ('investigation', 'validation', 'breakthrough')
              AND COALESCE(l.is_reference, 0) = 0
              AND p.graph_json IS NOT NULL
            ORDER BY l.composite_score DESC
        """).fetchall()
        logger.info("Found %d investigation+ entries to sweep", len(rows))
        for i, r in enumerate(rows):
            fp = r["graph_fingerprint"]
            existing_steps = r["wikitext_eval_steps"] or 0
            max_needed = max(checkpoints)
            if existing_steps >= max_needed:
                logger.info(
                    "[%d/%d] %s already has %d steps, skipping",
                    i + 1,
                    len(rows),
                    fp[:10],
                    existing_steps,
                )
                continue
            logger.info(
                "[%d/%d] %s tier=%s score=%.1f",
                i + 1,
                len(rows),
                fp[:10],
                r["tier"],
                r["composite_score"],
            )
            try:
                sweep_fingerprint(conn, fp, checkpoints, args.device, args.n_layers)
            except Exception as e:
                logger.error("Failed %s: %s", fp[:10], e)

    conn.close()


if __name__ == "__main__":
    main()
