#!/usr/bin/env python3
"""Run full probe suite at a given training checkpoint.

Trains a model from scratch to the target step count, then evaluates all probes:
  HellaSwag, BLiMP, Induction, Binding, AR, WikiText PPL.

Usage:
    python -m research.tools.probe_sweep --fingerprint c9c7075e741a8790 --steps 10000
    python -m research.tools.probe_sweep --fingerprint c9c7075e741a8790 --steps 2500,5000,10000
    python -m research.tools.probe_sweep --all-investigated --steps 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def _format_metric(value: Any, digits: int = 4, missing: str = "--") -> str:
    """Render numeric metrics safely for CLI output."""
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(numeric):
        return missing
    return f"{numeric:.{digits}f}"


def _train_to_step(
    model: nn.Module,
    n_steps: int,
    device: str,
    vocab_size: int = 100277,
) -> Dict[str, Any]:
    """Train model for n_steps on WikiText-103, return trajectory with PPL."""
    from research.eval.wikitext_eval import evaluate_wikitext_trajectory

    result = evaluate_wikitext_trajectory(
        model=model,
        vocab_size=vocab_size,
        device=device,
        checkpoints=(n_steps,),
        seq_len=128,
        n_eval_batches=16,
        train_batch_size=4,
        eval_batch_size=4,
        lr=3e-4,
    )
    trajectory = result.get("trajectory", {})
    final_data = trajectory.get(str(n_steps), {})
    return {
        "final_loss": final_data.get("val_loss"),
        "wikitext_ppl": final_data.get("val_ppl", final_data.get("ppl")),
        "trajectory": trajectory,
    }


def _run_probes(
    model: nn.Module,
    device: str,
    vocab_size: int = 100277,
) -> Dict[str, Any]:
    """Run all probes on a trained model. Model is NOT modified."""
    results: Dict[str, Any] = {}

    # HellaSwag
    try:
        from research.eval.hellaswag_eval import evaluate_hellaswag

        hs = evaluate_hellaswag(model, vocab_size, device, n_examples=200)
        results["hellaswag_acc"] = hs.get("hellaswag_acc")
        results["hellaswag_correct"] = hs.get("hellaswag_correct")
        results["hellaswag_total"] = hs.get("hellaswag_total")
        results["hellaswag_status"] = hs.get("hellaswag_status")
        acc_text = _format_metric(results["hellaswag_acc"], digits=3)
        status = results.get("hellaswag_status") or "unknown"
        logger.info(
            "  HellaSwag: %s (%d/%d, status=%s)",
            acc_text,
            results.get("hellaswag_correct") or 0,
            results.get("hellaswag_total") or 0,
            status,
        )
    except Exception as e:
        logger.warning("  HellaSwag failed: %s", e)
        results["hellaswag_acc"] = None
        results["hellaswag_correct"] = 0
        results["hellaswag_total"] = 0
        results["hellaswag_status"] = "failed"
        results["hellaswag_error"] = str(e)

    # BLiMP
    try:
        from research.eval.blimp_eval import evaluate_blimp

        blimp = evaluate_blimp(model, vocab_size=vocab_size, device=device)
        results["blimp_overall_accuracy"] = blimp.overall_accuracy
        results["blimp_n_subtasks"] = blimp.n_subtasks
        results["blimp_status"] = blimp.status
        logger.info(
            "  BLiMP: %.3f (%d subtasks, %s)",
            blimp.overall_accuracy or 0,
            blimp.n_subtasks,
            blimp.status,
        )
    except Exception as e:
        logger.warning("  BLiMP failed: %s", e)
        results["blimp_error"] = str(e)

    # Induction
    try:
        from research.eval.native_induction import (
            induction_result_metadata,
            induction_score_gold,
        )

        ind = induction_score_gold(model, device=device, seed=42)
        ind_meta = induction_result_metadata(ind)
        results["induction_auc"] = ind_meta.get("induction_auc")
        results.update(
            {k: v for k, v in ind_meta.items() if k.startswith("induction_")}
        )
        logger.info("  Induction AUC: %.4f", results.get("induction_auc") or 0)
    except Exception as e:
        logger.warning("  Induction failed: %s", e)
        results["induction_error"] = str(e)

    # Binding (zero-shot range profile)
    try:
        from research.eval.binding_range import binding_range_profile

        br = binding_range_profile(
            model,
            distances=[4, 8, 16, 32, 64],
            n_eval=200,
            device=device,
            seed=42,
        )
        results["binding_auc"] = br.get("auc")
        results["binding_curve"] = br.get("curve")
        logger.info("  Binding AUC: %.4f", results.get("binding_auc") or 0)
    except Exception as e:
        logger.warning("  Binding failed: %s", e)
        results["binding_error"] = str(e)

    # Associative Recall
    try:
        from research.eval.associative_recall import associative_recall_score

        ar = associative_recall_score(
            model,
            n_pairs=20,
            n_train_steps=500,
            n_eval=200,
            device=device,
        )
        results["ar_auc"] = ar.auc
        results["ar_final_acc"] = ar.final_acc
        results["ar_above_chance"] = ar.above_chance
        logger.info(
            "  AR AUC: %.4f  final_acc: %.4f",
            ar.auc or 0,
            ar.final_acc or 0,
        )
    except Exception as e:
        logger.warning("  AR failed: %s", e)
        results["ar_error"] = str(e)

    # WikiText PPL (final snapshot)
    try:
        from research.eval.wikitext_eval import evaluate_wikitext_ppl

        ppl = evaluate_wikitext_ppl(
            model,
            vocab_size=vocab_size,
            device=device,
            seq_len=128,
            n_batches=16,
        )
        results["wikitext_ppl"] = ppl
        logger.info("  WikiText PPL: %.2f", ppl or 0)
    except Exception as e:
        logger.warning("  WikiText PPL failed: %s", e)
        results["wikitext_ppl_error"] = str(e)

    return results


def sweep_fingerprint(
    conn: sqlite3.Connection,
    fingerprint: str,
    step_counts: List[int],
    device: str,
    n_layers: int = 6,
) -> Dict[str, Any] | None:
    """Train model to each step count and run full probe suite."""
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model
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
    graph_json = json.loads(row["graph_json"])
    graph = ComputationGraph.from_dict(graph_json)

    all_results: Dict[str, Any] = {
        "fingerprint": fingerprint,
        "result_id": result_id,
        "checkpoints": {},
    }

    for steps in sorted(step_counts):
        logger.info("=== %s @ %d steps ===", fingerprint[:12], steps)

        # Fresh model each time (clean slate)
        model = compile_model(
            [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=256
        )
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        all_results["n_params"] = n_params

        if steps == step_counts[0]:
            logger.info(
                "Compiled %s: %s params, %d layers",
                fingerprint[:12],
                f"{n_params:,}",
                n_layers,
            )

        # Train on WikiText-103 (mutates model in-place)
        t0 = time.time()
        try:
            train_result = _train_to_step(model, steps, device, vocab_size=VOCAB_SIZE)
            ppl = train_result.get("wikitext_ppl")
            logger.info("  Training: %d steps, PPL=%.2f", steps, ppl or 0)
        except Exception as e:
            logger.error("  Training failed at %d steps: %s", steps, e)
            all_results["checkpoints"][steps] = {"error": f"training_failed: {e}"}
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Run probes on the trained model
        model.eval()
        probe_results = _run_probes(model, device, vocab_size=VOCAB_SIZE)

        elapsed = time.time() - t0
        probe_results["wikitext_ppl"] = ppl
        probe_results["final_loss"] = train_result.get("final_loss")
        probe_results["elapsed_s"] = round(elapsed, 1)
        all_results["checkpoints"][steps] = probe_results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary table
    print()
    print(
        f"{'Steps':>7} {'PPL':>7} {'Hella':>7} {'Ind':>7} {'AR':>7} {'Bind':>7} {'BLiMP':>7} {'Loss':>7} {'Time':>6}"
    )
    print("-" * 70)
    for steps in sorted(all_results["checkpoints"].keys()):
        r = all_results["checkpoints"][steps]
        if "error" in r:
            print(f"{steps:>7} {'ERROR':>7}")
            continue

        print(
            f"{steps:>7} {_format_metric(r.get('wikitext_ppl'), 2):>7} "
            f"{_format_metric(r.get('hellaswag_acc'), 3):>7} "
            f"{_format_metric(r.get('induction_auc'), 4):>7} "
            f"{_format_metric(r.get('ar_auc'), 4):>7} "
            f"{_format_metric(r.get('binding_auc'), 4):>7} "
            f"{_format_metric(r.get('blimp_overall_accuracy'), 3):>7} "
            f"{_format_metric(r.get('final_loss'), 4):>7} {r.get('elapsed_s', '--'):>6}"
        )
    print()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train to checkpoint and run full probe suite"
    )
    parser.add_argument("--fingerprint", type=str, help="Fingerprint to evaluate")
    parser.add_argument(
        "--all-investigated",
        action="store_true",
        help="Run on all investigation+ entries",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="10000",
        help="Comma-separated step counts (default: 10000)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--db", type=str, default="research/lab_notebook.db")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    step_counts = [int(s.strip()) for s in args.steps.split(",")]

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    all_outputs = []

    if args.fingerprint:
        result = sweep_fingerprint(
            conn, args.fingerprint, step_counts, args.device, args.n_layers
        )
        if result:
            all_outputs.append(result)
    elif args.all_investigated:
        rows = conn.execute(
            """
            SELECT p.graph_fingerprint, l.tier, l.composite_score
            FROM leaderboard l
            JOIN program_results p ON l.result_id = p.result_id
            WHERE l.tier IN ('investigation', 'validation', 'breakthrough')
              AND COALESCE(l.is_reference, 0) = 0
              AND p.graph_json IS NOT NULL
            ORDER BY l.composite_score DESC
            """
        ).fetchall()
        logger.info("Found %d entries to probe", len(rows))
        for i, r in enumerate(rows):
            fp = r["graph_fingerprint"]
            logger.info(
                "[%d/%d] %s tier=%s score=%.1f",
                i + 1,
                len(rows),
                fp[:12],
                r["tier"],
                r["composite_score"],
            )
            try:
                result = sweep_fingerprint(
                    conn, fp, step_counts, args.device, args.n_layers
                )
                if result:
                    all_outputs.append(result)
            except Exception as e:
                logger.error("Failed %s: %s", fp[:12], e)
    else:
        parser.print_help()
        conn.close()
        return

    if args.save_json and all_outputs:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

    conn.close()


if __name__ == "__main__":
    main()
