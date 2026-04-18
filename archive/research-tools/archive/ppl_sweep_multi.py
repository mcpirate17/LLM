"""Perplexity sweep across multiple dataset sizes.

For each frontier model, trains on three corpus sizes and measures
val perplexity at 500/750/1000/1500 steps:
  - wikitext-2 (11MB train)
  - wikitext-103 capped at 20MB train ("medium")
  - wikitext-103 full (~546MB train)

All use the same validation set (wikitext-103 val, 1.1MB).

Usage:
    python -m research.tools.ppl_sweep_multi [--top N] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sqlite3
import time

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINTS = (500, 750, 1000, 1500)

CORPUS_CONFIGS = [
    {
        "name": "wikitext-2",
        "variant": "wikitext-2-raw-v1",
        "max_chars_train": 11_000_000,
    },
    {
        "name": "wiki103-20MB",
        "variant": "wikitext-103-raw-v1",
        "max_chars_train": 20_000_000,
    },
    {
        "name": "wiki103-full",
        "variant": "wikitext-103-raw-v1",
        "max_chars_train": 200_000_000,
    },
]


def run_trajectory(model_template, vocab_size, device, variant, max_chars_train):
    """Clone model, train, measure ppl at checkpoints. Returns trajectory dict."""
    from research.eval.wikitext_eval import evaluate_wikitext_trajectory

    # Deep copy so each corpus config starts from the same init
    model = copy.deepcopy(model_template).to(device)

    result = evaluate_wikitext_trajectory(
        model=model,
        vocab_size=vocab_size,
        device=device,
        checkpoints=CHECKPOINTS,
        variant=variant,
        seq_len=128,
        n_eval_batches=16,
        train_batch_size=4,
        eval_batch_size=4,
        lr=3e-4,
        max_chars_train=max_chars_train,
        max_chars_val=200_000,  # consistent val size
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--db", type=str, default="research/lab_notebook.db")
    args = parser.parse_args()

    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model
    from research.defaults import VOCAB_SIZE

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.tier, l.composite_score,
               p.graph_fingerprint, p.graph_json, p.graph_n_ops
        FROM leaderboard l
        JOIN program_results p ON l.result_id = p.result_id
        WHERE COALESCE(l.is_reference, 0) = 0
          AND p.graph_json IS NOT NULL
        ORDER BY l.composite_score DESC
        LIMIT ?
    """,
        (args.top,),
    ).fetchall()

    conn.close()

    logger.info(
        "Sweeping %d models x %d corpora x %d checkpoints",
        len(rows),
        len(CORPUS_CONFIGS),
        len(CHECKPOINTS),
    )

    all_results = []

    for i, r in enumerate(rows):
        fp = r["graph_fingerprint"][:10]
        logger.info(
            "[%d/%d] %s (tier=%s score=%.1f ops=%s)",
            i + 1,
            len(rows),
            fp,
            r["tier"],
            r["composite_score"],
            r["graph_n_ops"],
        )

        try:
            graph_dict = json.loads(r["graph_json"])
            graph = ComputationGraph.from_dict(graph_dict)
            model_template = compile_model(
                [graph] * 4, vocab_size=VOCAB_SIZE, max_seq_len=256
            )
            n_params = sum(p.numel() for p in model_template.parameters())
            logger.info("  Compiled: %s params", f"{n_params:,}")
        except Exception as e:
            logger.error("  Failed to compile: %s", e)
            continue

        model_results = {
            "fingerprint": fp,
            "ops": r["graph_n_ops"],
            "tier": r["tier"],
            "score": r["composite_score"],
            "params": n_params,
            "corpora": {},
        }

        for cfg in CORPUS_CONFIGS:
            logger.info("  Corpus: %s", cfg["name"])
            t0 = time.time()
            try:
                result = run_trajectory(
                    model_template,
                    VOCAB_SIZE,
                    args.device,
                    cfg["variant"],
                    cfg["max_chars_train"],
                )
                trajectory = result.get("checkpoints", {})
                elapsed = time.time() - t0

                ppl_line = []
                for step in CHECKPOINTS:
                    data = trajectory.get(step, {})
                    ppl = data.get("ppl")
                    ppl_line.append(f"{step}={ppl:.1f}" if ppl else f"{step}=N/A")
                logger.info("    %s (%.0fs)", "  ".join(ppl_line), elapsed)

                model_results["corpora"][cfg["name"]] = {
                    str(step): trajectory.get(step, {}).get("ppl")
                    for step in CHECKPOINTS
                }
            except Exception as e:
                logger.error("    Failed: %s", e)
                model_results["corpora"][cfg["name"]] = {"error": str(e)}

        all_results.append(model_results)

        del model_template
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':12s} {'Ops':>4s} {'Params':>12s} | ", end="")
    for cfg in CORPUS_CONFIGS:
        for step in CHECKPOINTS:
            print(f"{cfg['name'][:6]}@{step:>4d} ", end="")
        print("| ", end="")
    print()
    print("-" * 100)

    for mr in all_results:
        print(
            f"{mr['fingerprint']:12s} {mr['ops'] or 0:>4d} {mr['params']:>12,} | ",
            end="",
        )
        for cfg in CORPUS_CONFIGS:
            corpora_data = mr["corpora"].get(cfg["name"], {})
            if isinstance(corpora_data, dict) and "error" not in corpora_data:
                for step in CHECKPOINTS:
                    ppl = corpora_data.get(str(step))
                    print(f"{ppl:>11.1f} " if ppl else f"{'N/A':>11s} ", end="")
            else:
                for _ in CHECKPOINTS:
                    print(f"{'ERR':>11s} ", end="")
            print("| ", end="")
        print()


if __name__ == "__main__":
    main()
