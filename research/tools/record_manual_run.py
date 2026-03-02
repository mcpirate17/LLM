#!/usr/bin/env python3
"""Record manual training run results and trigger leaderboard rescoring.

Usage examples:
    # Record TinyStories benchmark result
    python -m research.tools.record_manual_run \
      --result-id b7a3eda6cdca769d \
      --benchmark tinystories --loss 2.3 --steps 1000 --seq-len 1024

    # Record raw operational metrics
    python -m research.tools.record_manual_run \
      --result-id b7a3eda6cdca769d \
      --throughput-tok-s 50000 --peak-memory-mb 200

    # Just rescore from existing data
    python -m research.tools.record_manual_run \
      --result-id b7a3eda6cdca769d --rescore
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.scientist.notebook import LabNotebook

DB_PATH = "research/lab_notebook.db"


def _compute_score_from_loss(loss: float, vocab_size: int) -> float:
    """1/(1+log(ppl/vocab_size)), clamped to [0,1]."""
    ppl = math.exp(loss)
    ratio = ppl / vocab_size
    score = 1.0 / (1.0 + math.log(max(ratio, 1e-6)))
    return round(max(0.0, min(1.0, score)), 4)


def record_benchmark(nb: LabNotebook, result_id: str, benchmark: str,
                     loss: float, steps: int, seq_len: int,
                     vocab_size: int) -> dict:
    """Store benchmark result in program_results and external_benchmarks_json."""
    ppl = round(math.exp(loss), 2)
    score = _compute_score_from_loss(loss, vocab_size)

    key_prefix = benchmark  # "tinystories" or "wikitext"
    ppl_col = f"{key_prefix}_perplexity"
    score_col = f"{key_prefix}_score"

    # Update program_results columns directly
    nb.conn.execute(
        f"UPDATE program_results SET {ppl_col} = ?, {score_col} = ? WHERE result_id = ?",
        (ppl, score, result_id),
    )
    nb._maybe_commit()

    # Also store in external_benchmarks_json with metadata
    payload = {
        benchmark: {
            "loss": loss,
            "perplexity": ppl,
            "score": score,
            "steps": steps,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "source": "manual",
        }
    }
    nb.set_external_benchmarks(result_id, payload)

    return {"perplexity": ppl, "score": score, ppl_col: ppl, score_col: score}


def record_operational_metrics(nb: LabNotebook, result_id: str,
                               throughput_tok_s: float | None,
                               peak_memory_mb: float | None) -> dict:
    """Update operational metrics on program_results."""
    updates = {}
    if throughput_tok_s is not None:
        updates["throughput_tok_s"] = throughput_tok_s
    if peak_memory_mb is not None:
        updates["peak_memory_mb"] = peak_memory_mb

    if not updates:
        return {}

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [result_id]
    nb.conn.execute(
        f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
        vals,
    )
    nb._maybe_commit()
    return updates


def rescore(nb: LabNotebook, result_id: str) -> dict | None:
    """Re-run upsert_leaderboard to recompute composite_score + efficiency_multiple."""
    entry = nb.get_leaderboard_entry(result_id)
    if entry is None:
        print(f"  No leaderboard entry for {result_id} — cannot rescore")
        return None

    before_composite = entry.get("composite_score")
    before_eff = entry.get("efficiency_multiple")

    # Re-upsert with existing values to trigger score recomputation
    nb.upsert_leaderboard(
        result_id=result_id,
        model_source=entry.get("model_source", "unknown"),
        architecture_desc=entry.get("architecture_desc", ""),
        tier=entry.get("tier", "screening"),
        tags=entry.get("tags"),
        notes=entry.get("notes"),
        is_reference=bool(entry.get("is_reference")),
        reference_name=entry.get("reference_name"),
    )

    after = nb.get_leaderboard_entry(result_id)
    after_composite = after.get("composite_score") if after else None
    after_eff = after.get("efficiency_multiple") if after else None

    return {
        "composite_score": {"before": before_composite, "after": after_composite},
        "efficiency_multiple": {"before": before_eff, "after": after_eff},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Record manual training run results and trigger leaderboard rescoring.",
    )
    parser.add_argument("--result-id", required=True, help="Program result ID")
    parser.add_argument("--db", default=DB_PATH, help="Path to lab_notebook.db")

    # Benchmark recording
    parser.add_argument("--benchmark", choices=["tinystories", "wikitext"],
                        help="Benchmark name")
    parser.add_argument("--loss", type=float, help="Training loss")
    parser.add_argument("--steps", type=int, help="Training steps completed")
    parser.add_argument("--seq-len", type=int, help="Sequence length used")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")

    # Operational metrics
    parser.add_argument("--throughput-tok-s", type=float,
                        help="Throughput in tokens/second")
    parser.add_argument("--peak-memory-mb", type=float,
                        help="Peak memory usage in MB")

    # Rescore-only
    parser.add_argument("--rescore", action="store_true",
                        help="Just rescore from existing data")

    args = parser.parse_args()

    # Validate: need at least one action
    has_benchmark = args.benchmark is not None
    has_ops = args.throughput_tok_s is not None or args.peak_memory_mb is not None
    if not has_benchmark and not has_ops and not args.rescore:
        parser.error("Provide --benchmark, --throughput-tok-s/--peak-memory-mb, or --rescore")

    if has_benchmark and args.loss is None:
        parser.error("--benchmark requires --loss")

    nb = LabNotebook(args.db)

    # Verify result exists
    row = nb.conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = ?",
        (args.result_id,),
    ).fetchone()
    if not row:
        print(f"ERROR: result_id {args.result_id!r} not found in program_results")
        sys.exit(1)

    print(f"Recording for result_id={args.result_id}")

    # 1. Benchmark
    if has_benchmark:
        bm = record_benchmark(
            nb, args.result_id, args.benchmark,
            loss=args.loss,
            steps=args.steps or 0,
            seq_len=args.seq_len or 128,
            vocab_size=args.vocab_size,
        )
        print(f"  {args.benchmark}: perplexity={bm['perplexity']}, score={bm['score']}")

    # 2. Operational metrics
    if has_ops:
        ops = record_operational_metrics(
            nb, args.result_id,
            throughput_tok_s=args.throughput_tok_s,
            peak_memory_mb=args.peak_memory_mb,
        )
        for k, v in ops.items():
            print(f"  {k} = {v}")

    # 3. Rescore (always do it if data was written, or if --rescore)
    if has_benchmark or has_ops or args.rescore:
        print("  Rescoring leaderboard entry...")
        result = rescore(nb, args.result_id)
        if result:
            for metric, vals in result.items():
                before = vals["before"]
                after = vals["after"]
                delta = ""
                if before is not None and after is not None:
                    diff = after - before
                    delta = f" ({'+' if diff >= 0 else ''}{diff:.4f})"
                print(f"  {metric}: {before} -> {after}{delta}")

    print("Done.")


if __name__ == "__main__":
    main()
