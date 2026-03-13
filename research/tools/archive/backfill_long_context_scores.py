"""Backfill leaderboard long-context robustness scores.

Recomputes `robustness_long_ctx_score` for leaderboard entries by rebuilding
models from stored `graph_json` and running:
  - long-context scaling sweep
  - passkey retrieval evaluation

Score matches runner logic:
  combined = 0.5 * scaling_score + 0.5 * retrieval_score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.eval.long_context import run_long_context_sweep
from research.eval.passkey import evaluate_long_context_retrieval
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.notebook import LabNotebook
from research.synthesis.serializer import graph_from_json


def _build_model(graph_json: str, vocab_size: int, max_seq_len: int, n_layers: int):
    graph = graph_from_json(graph_json)
    layer_graphs = [graph] * max(1, int(n_layers))
    return compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=max_seq_len)


def _select_rows(conn, tier: str, limit: int, min_composite: float):
    where_tier = "" if tier == "all" else "AND lb.tier = ?"
    params = []
    if tier != "all":
        params.append(tier)
    params.extend([min_composite, limit])

    query = f"""
        SELECT
            lb.entry_id,
            lb.result_id,
            lb.tier,
            lb.composite_score,
            lb.robustness_long_ctx_score,
            pr.graph_json,
            pr.final_loss,
            pr.external_benchmarks_json,
            e.config_json
        FROM leaderboard lb
        JOIN program_results pr ON lb.result_id = pr.result_id
        JOIN experiments e ON pr.experiment_id = e.experiment_id
        WHERE lb.is_reference = 0
          AND pr.graph_json IS NOT NULL
          AND (lb.robustness_long_ctx_score IS NULL OR lb.robustness_long_ctx_score = 0.0)
          {where_tier}
          AND COALESCE(lb.composite_score, 0.0) >= ?
        ORDER BY COALESCE(lb.composite_score, 0.0) DESC
        LIMIT ?
    """
    return conn.execute(query, params).fetchall()


def backfill(args):
    db = LabNotebook(args.db)
    db.conn.execute("PRAGMA busy_timeout = 30000")
    dev = torch.device(args.device)

    rows = _select_rows(db.conn, args.tier, args.limit, args.min_composite)
    print(f"Found {len(rows)} entries for long-context backfill (tier={args.tier}, min_composite={args.min_composite}).")
    if not rows:
        return

    ok = 0
    failed = 0
    t0 = time.time()
    for idx, row in enumerate(rows, start=1):
        entry_id, result_id, tier, comp, old_score, graph_json, final_loss, existing_benchmarks_json, config_json = row
        print(f"[{idx}/{len(rows)}] {result_id[:12]} tier={tier} composite={comp:.4f} old={old_score}")
        try:
            cfg = json.loads(config_json) if config_json else {}
            vocab_size = int(cfg.get("vocab_size", 32000))
            n_layers = int(cfg.get("n_layers", 2))
            base_loss = float(final_loss or 0.0)

            def _make():
                return _build_model(
                    graph_json=graph_json,
                    vocab_size=vocab_size,
                    max_seq_len=1024,
                    n_layers=n_layers,
                )

            lc = run_long_context_sweep(
                _make,
                vocab_size=vocab_size,
                device=dev,
                base_loss=base_loss,
                seq_lens=tuple(args.seq_lens),
                n_steps=args.n_steps,
                batch_size=args.batch_size,
            )
            scaling_score = float(lc.get("long_context_score", 0.0) or 0.0)

            retr_model = _make().to(dev)
            retr = evaluate_long_context_retrieval(
                retr_model,
                vocab_size=vocab_size,
                device=dev,
                lengths=list(args.retrieval_lengths),
            )
            del retr_model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            retrieval_score = float(
                retr.get("retrieval_aggregate_score", retr.get("retrieval_score", 0.0)) or 0.0
            )
            assoc_score = float(retr.get("assoc_retrieval_score", retr.get("retrieval_score", 0.0)) or 0.0)
            multi_hop_score = float(retr.get("multi_hop_score", 0.0) or 0.0)
            passkey_score = float(retr.get("passkey_score", 0.0) or 0.0)
            combined = (scaling_score * 0.5) + (retrieval_score * 0.5)

            print(
                f"  scaling={scaling_score:.4f} assoc={assoc_score:.4f} multi_hop={multi_hop_score:.4f} passkey={passkey_score:.4f} "
                f"retrieval={retrieval_score:.4f} combined={combined:.4f} "
                f"max_viable={lc.get('max_viable_len', 0)}"
            )

            if not args.dry_run:
                db.conn.execute(
                    "UPDATE leaderboard SET robustness_long_ctx_score=? WHERE entry_id=?",
                    (combined, entry_id),
                )
                existing_payload = {}
                if existing_benchmarks_json:
                    try:
                        parsed = json.loads(existing_benchmarks_json)
                        if isinstance(parsed, dict):
                            existing_payload = parsed
                    except Exception:
                        existing_payload = {}
                existing_payload["long_context"] = {
                    "scaling": lc,
                    "retrieval": retr,
                    "scaling_score": scaling_score,
                    "assoc_retrieval_score": assoc_score,
                    "multi_hop_score": multi_hop_score,
                    "passkey_score": passkey_score,
                    "retrieval_aggregate_score": retrieval_score,
                    "combined_score": combined,
                    "benchmark_version": "v3_assoc_multihop_passkey",
                }
                db.conn.execute(
                    "UPDATE program_results SET external_benchmarks_json=? WHERE result_id=?",
                    (json.dumps(existing_payload), result_id),
                )
                db.conn.commit()
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"Done. updated={ok} failed={failed} elapsed={elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Backfill long-context robustness scores")
    parser.add_argument("--db", default=os.path.join(os.path.dirname(__file__), "..", "lab_notebook.db"))
    parser.add_argument("--tier", default="validation", choices=["screening", "investigation", "validation", "all"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-composite", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--retrieval-lengths", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(args)


if __name__ == "__main__":
    main()
