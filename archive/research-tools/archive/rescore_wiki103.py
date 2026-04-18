#!/usr/bin/env python3
"""Rescore leaderboard entries: re-evaluate with WikiText-103 and recompute scores.

Targets entries whose screening_metric_version is NOT 'wiki103_v7'.
For each entry:
  1. Reconstruct model from graph_json
  2. Run screening_wikitext_eval with wikitext-103-raw-v1
  3. Update leaderboard ppl + screening_metric_version
  4. Recompute composite score via build_score_kwargs -> compute_composite_v7

Usage:
    python -m research.tools.rescore_wiki103 [--dry-run] [--limit N] [--device cuda]
"""

import argparse
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.defaults import VOCAB_SIZE
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from research.scientist.notebook import LabNotebook
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json

DB_PATH = "research/lab_notebook.db"


def _reconstruct_model(graph_json_str: str, device: str):
    """Reconstruct a SynthesizedModel from stored graph_json."""
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph], vocab_size=VOCAB_SIZE)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Rescore leaderboard with WikiText-103"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max entries to process (0=all)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip re-evaluation, only recompute scores from existing data",
    )
    args = parser.parse_args()

    nb = LabNotebook(DB_PATH)
    cur = nb.conn.cursor()

    # Find entries needing rescore
    rows = cur.execute(
        "SELECT l.entry_id, l.result_id, l.tier, l.composite_score, l.model_source, "
        "l.is_reference, l.screening_metric_version, l.wikitext_perplexity, "
        "l.screening_loss_ratio, "
        "pr.graph_json, pr.graph_fingerprint "
        "FROM leaderboard l "
        "LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        "WHERE l.screening_metric_version IS NULL "
        "   OR l.screening_metric_version <> 'wiki103_v7' "
        "ORDER BY l.composite_score DESC"
    ).fetchall()

    if args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    print(
        f"Entries to rescore: {total}  (device={args.device}, dry_run={args.dry_run})"
    )
    print(f"{'skip_eval' if args.skip_eval else 'full re-evaluation with wiki103'}")
    print()

    # Batch-fetch all program_results up front (eliminates N+1 per-result queries)
    all_result_ids = [r["result_id"] for r in rows]
    pr_cache = prefetch_program_results(nb.conn, all_result_ids)
    print(f"Pre-fetched {len(pr_cache)} program_results rows")

    updated = 0
    eval_ok = 0
    eval_fail = 0
    no_graph = 0
    score_changes = []
    t0 = time.time()

    for i, row in enumerate(rows):
        entry_id = row["entry_id"]
        result_id = row["result_id"]
        tier = row["tier"] or "screening"
        old_score = row["composite_score"] or 0.0
        graph_json = row["graph_json"]
        fp = (row["graph_fingerprint"] or "")[:12]
        is_ref = bool(row["is_reference"])

        new_ppl = row["wikitext_perplexity"]  # keep existing if eval fails/skipped

        # Step 1: Re-evaluate with wiki103 (unless --skip-eval)
        if not args.skip_eval and graph_json and graph_json != "{}":
            try:
                model = _reconstruct_model(graph_json, args.device)

                from research.eval.wikitext_eval import screening_wikitext_eval

                result = screening_wikitext_eval(
                    model,
                    VOCAB_SIZE,
                    args.device,
                    variant="wikitext-103-raw-v1",
                )

                if result.get("screening_wikitext_status") == "ok":
                    new_ppl = result["wikitext_perplexity"]
                    eval_ok += 1
                else:
                    eval_fail += 1
                    status = result.get("screening_wikitext_status", "unknown")
                    print(f"  [{fp}] eval status={status}: {result.get('error', '')}")

                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                eval_fail += 1
                print(f"  [{fp}] eval error: {e}")
                if "CUDA" in str(e):
                    torch.cuda.empty_cache()
        elif not graph_json or graph_json == "{}":
            no_graph += 1

        # Step 2: Recompute composite score
        try:
            existing = cur.execute(
                "SELECT * FROM leaderboard WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            if not existing:
                continue

            d = dict(existing)
            # Inject updated ppl if we got new data
            if new_ppl is not None:
                d["wikitext_perplexity"] = new_ppl

            pr_dict = pr_cache.get(result_id, {})
            score_kwargs = build_score_kwargs_from_prefetch(pr_dict, d, is_ref)
            new_score = compute_composite(**score_kwargs)

            delta = new_score - old_score
            if abs(delta) > 0.01 or new_ppl != row["wikitext_perplexity"]:
                score_changes.append((fp, tier, old_score, new_score, delta))

                if not args.dry_run:
                    update_fields = {
                        "composite_score": new_score,
                        "screening_metric_version": "wiki103_v7",
                        "rescore_status": "rescored_wiki103",
                        "rescore_timestamp": time.time(),
                        "old_composite_score": old_score,
                        "rescore_reason": "wiki103_rescore",
                    }
                    if new_ppl is not None:
                        update_fields["wikitext_perplexity"] = new_ppl

                    set_clause = ", ".join(f"{k} = ?" for k in update_fields)
                    vals = list(update_fields.values()) + [entry_id]
                    cur.execute(
                        f"UPDATE leaderboard SET {set_clause} WHERE entry_id = ?",
                        vals,
                    )
                updated += 1
            else:
                # Score unchanged, just mark as rescored
                if not args.dry_run:
                    cur.execute(
                        "UPDATE leaderboard SET screening_metric_version = 'wiki103_v7', "
                        "rescore_status = 'rescored_wiki103', rescore_timestamp = ? "
                        "WHERE entry_id = ?",
                        (time.time(), entry_id),
                    )

        except Exception as e:
            print(f"  [{fp}] score error: {e}")
            traceback.print_exc()

        if (i + 1) % 25 == 0:
            if not args.dry_run:
                nb.conn.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s) — "
                f"{updated} changed, {eval_ok} eval ok, {eval_fail} eval fail"
            )

    if not args.dry_run:
        nb.conn.commit()

    elapsed = time.time() - t0
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Done in {elapsed:.1f}s")
    print(f"  Total:      {total}")
    print(f"  Changed:    {updated}")
    print(f"  Eval OK:    {eval_ok}")
    print(f"  Eval fail:  {eval_fail}")
    print(f"  No graph:   {no_graph}")

    if score_changes:
        score_changes.sort(key=lambda x: x[4])
        print("\nBiggest score changes:")
        print(f"  {'FP':<14} {'Tier':<20} {'Old':>8} {'New':>8} {'Delta':>8}")
        for fp, tier, old, new, delta in score_changes[:10]:
            print(f"  {fp:<14} {tier:<20} {old:>8.1f} {new:>8.1f} {delta:>+8.1f}")
        print("  ...")
        for fp, tier, old, new, delta in score_changes[-10:]:
            print(f"  {fp:<14} {tier:<20} {old:>8.1f} {new:>8.1f} {delta:>+8.1f}")

    # Show top 15
    top = cur.execute(
        "SELECT entry_id, result_id, tier, composite_score, is_reference, "
        "reference_name, wikitext_perplexity "
        "FROM leaderboard ORDER BY composite_score DESC LIMIT 15"
    ).fetchall()
    print("\nTop 15 after rescore:")
    print(f"  {'Score':>8} {'PPL':>8} {'Tier':<20} {'Ref':<8} {'ID'}")
    for r in top:
        ref = r["reference_name"] or ""
        ppl = r["wikitext_perplexity"]
        ppl_s = f"{ppl:.2f}" if ppl else "—"
        print(
            f"  {r['composite_score']:>8.1f} {ppl_s:>8} {r['tier']:<20} {ref:<8} {r['entry_id'][:16]}"
        )

    nb.conn.close()


if __name__ == "__main__":
    main()
