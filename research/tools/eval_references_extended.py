#!/usr/bin/env python3
"""Run extended evaluations on reference architectures.

Fills in the metrics that register_references.py doesn't compute:
  wikitext, tinystories, cross_task, efficiency_wall, activation_sparsity,
  routing_heatmap, and fixes is_pinned.

Usage:
    python -m research.tools.eval_references_extended --device cpu
    python -m research.tools.eval_references_extended --arch gpt2 --device cpu
    python -m research.tools.eval_references_extended --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_extended_evals(
    arch_filter: str | None = None,
    device: str = "cpu",
    dry_run: bool = False,
    n_train_steps: int = 200,
):
    from research.synthesis.reference_architectures import REFERENCE_ARCHITECTURES, build_reference
    from research.synthesis.compiler import compile_model
    from research.scientist.notebook import LabNotebook
    from research.eval.wikitext_eval import evaluate_wikitext_perplexity
    from research.eval.tinystories_eval import evaluate_tinystories
    from research.eval.cross_task_eval import evaluate_cross_task_robustness
    from research.eval.efficiency_wall import evaluate_efficiency_wall
    from research.eval.sparsity import evaluate_activation_sparsity
    from research.eval.routing_heatmap import evaluate_routing_heatmap

    nb = LabNotebook()
    dev = torch.device(device)

    # Get reference entries from leaderboard
    refs = nb.get_references()
    if not refs:
        log.error("No reference entries found in leaderboard. Run register_references first.")
        return

    for ref in refs:
        ref_name = ref.get("reference_name", "?")
        entry_id = ref["entry_id"]
        result_id = ref.get("result_id", "")

        if arch_filter and arch_filter not in ref_name.lower() and arch_filter not in result_id:
            continue

        log.info("=== Extended evals for %s (entry=%s) ===", ref_name, entry_id)

        # Find the arch key from the reference name
        arch_key = None
        for k, v in REFERENCE_ARCHITECTURES.items():
            if v["name"] == ref_name:
                arch_key = k
                break

        if arch_key is None:
            log.warning("  Could not find architecture key for '%s', skipping", ref_name)
            continue

        # Rebuild the model
        d_model = 256
        n_layers = 6
        vocab_size = 32000
        seq_len = 128

        log.info("  Building model...")
        layer_graphs = [build_reference(arch_key, d_model) for _ in range(n_layers)]
        model = compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=seq_len)
        model = model.to(dev)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        log.info("  Params: %s", f"{total_params:,}")

        updates = {}

        # --- Efficiency Wall (no training needed) ---
        log.info("  Running efficiency_wall eval...")
        try:
            ew = evaluate_efficiency_wall(model, vocab_size, dev)
            updates["efficiency_wall_score"] = ew.get("efficiency_wall_score")
            updates["max_viable_seq_len"] = ew.get("max_viable_seq_len")
            updates["scaling_regime"] = ew.get("scaling_regime")
            log.info("    score=%.4f, max_seq=%s, regime=%s",
                     ew.get("efficiency_wall_score", 0),
                     ew.get("max_viable_seq_len"),
                     ew.get("scaling_regime"))
        except Exception as e:
            log.warning("    efficiency_wall failed: %s", e)

        # --- Activation Sparsity (no training needed) ---
        log.info("  Running activation_sparsity eval...")
        try:
            input_batches = [
                torch.randint(0, vocab_size, (4, seq_len), device=dev)
                for _ in range(4)
            ]
            sp = evaluate_activation_sparsity(model, input_batches, dev)
            updates["activation_sparsity_score"] = sp.get("activation_sparsity_score")
            updates["dead_neuron_ratio"] = sp.get("dead_neuron_ratio")
            log.info("    sparsity_score=%.4f, dead_neuron=%.4f",
                     sp.get("activation_sparsity_score", 0),
                     sp.get("dead_neuron_ratio", 0))
        except Exception as e:
            log.warning("    activation_sparsity failed: %s", e)

        # --- Routing Heatmap (no training needed) ---
        log.info("  Running routing_heatmap eval...")
        try:
            rh = evaluate_routing_heatmap(model, input_batches, dev)
            updates["routing_collapse_score"] = rh.get("routing_collapse_score")
            log.info("    routing_collapse=%.4f, has_routing=%s, n_modules=%d",
                     rh.get("routing_collapse_score") or 0,
                     rh.get("has_routing"),
                     rh.get("n_routing_modules", 0))
        except Exception as e:
            log.warning("    routing_heatmap failed: %s", e)

        # --- WikiText (needs micro-training) ---
        log.info("  Running wikitext eval (%d steps)...", n_train_steps)
        try:
            wt = evaluate_wikitext_perplexity(
                model, vocab_size, dev, n_train_steps=n_train_steps)
            updates["wikitext_perplexity"] = wt.get("wikitext_perplexity")
            updates["wikitext_score"] = wt.get("wikitext_score")
            log.info("    ppl=%.2f, score=%.4f",
                     wt.get("wikitext_perplexity") or 0,
                     wt.get("wikitext_score") or 0)
        except Exception as e:
            log.warning("    wikitext eval failed: %s", e)

        # --- Need fresh model for tinystories (training corrupts weights) ---
        log.info("  Rebuilding model for tinystories...")
        model_ts = compile_model(
            [build_reference(arch_key, d_model) for _ in range(n_layers)],
            vocab_size=vocab_size, max_seq_len=seq_len,
        ).to(dev)

        log.info("  Running tinystories eval (%d steps)...", n_train_steps)
        try:
            ts = evaluate_tinystories(
                model_ts, vocab_size, dev, n_train_steps=n_train_steps)
            updates["tinystories_perplexity"] = ts.get("tinystories_perplexity")
            updates["tinystories_score"] = ts.get("tinystories_score")
            log.info("    ppl=%.2f, score=%.4f",
                     ts.get("tinystories_perplexity") or 0,
                     ts.get("tinystories_score") or 0)
        except Exception as e:
            log.warning("    tinystories eval failed: %s", e)

        # --- Cross-task robustness (needs 2x fresh models) ---
        log.info("  Running cross_task eval (%d steps x2)...", n_train_steps)
        try:
            def make_model_fn():
                m = compile_model(
                    [build_reference(arch_key, d_model) for _ in range(n_layers)],
                    vocab_size=vocab_size, max_seq_len=seq_len,
                )
                return m.to(dev)

            ct = evaluate_cross_task_robustness(
                make_model_fn, vocab_size, dev, n_train_steps=n_train_steps)
            updates["cross_task_score"] = ct.get("cross_task_score")
            log.info("    cross_task_score=%.4f, code_ppl=%.2f, nl_ppl=%.2f",
                     ct.get("cross_task_score") or 0,
                     ct.get("code_perplexity") or 0,
                     ct.get("nl_perplexity") or 0)
        except Exception as e:
            log.warning("    cross_task eval failed: %s", e)

        # --- Architecture telemetry (routing/compression from compiled model) ---
        log.info("  Extracting architecture telemetry...")
        try:
            from research.scientist.runner import ExperimentRunner
            runner = ExperimentRunner.__new__(ExperimentRunner)
            telemetry = runner._extract_architecture_telemetry(model)
            if telemetry.get("routing_savings_ratio") is not None:
                updates["routing_savings_ratio"] = telemetry["routing_savings_ratio"]
            if telemetry.get("compression_ratio") is not None:
                updates["compression_ratio"] = telemetry["compression_ratio"]
            if telemetry.get("routing_mode"):
                log.info("    routing_mode=%s, savings=%.4f",
                         telemetry.get("routing_mode"),
                         telemetry.get("routing_savings_ratio", 0))
        except Exception as e:
            log.warning("    telemetry extraction failed: %s", e)

        # --- Apply updates ---
        # Filter out None values
        updates = {k: v for k, v in updates.items() if v is not None}

        log.info("  Metrics to update: %s", list(updates.keys()))

        if dry_run:
            log.info("  [DRY RUN] Would update %d metrics for %s", len(updates), ref_name)
            continue

        if updates:
            nb.promote_to_tier(entry_id, "validation", **updates)
            log.info("  Updated %d metrics via promote_to_tier", len(updates))

        # Fix is_pinned
        nb.set_leaderboard_pin(entry_id, True)
        nb.flush_writes()
        log.info("  Set is_pinned=1")

    # Final verification
    log.info("\n=== Verification ===")
    refs = nb.get_references()
    null_cols = [
        "wikitext_perplexity", "wikitext_score", "tinystories_perplexity",
        "tinystories_score", "cross_task_score", "efficiency_wall_score",
        "max_viable_seq_len", "activation_sparsity_score", "dead_neuron_ratio",
        "routing_collapse_score",
    ]
    for ref in refs:
        name = ref.get("reference_name", "?")
        missing = [c for c in null_cols if ref.get(c) is None]
        pinned = ref.get("is_pinned", 0)
        log.info("  %s: pinned=%s, missing=%d cols %s",
                 name, pinned, len(missing), missing if missing else "")


def main():
    parser = argparse.ArgumentParser(description="Extended evals for reference architectures")
    parser.add_argument("--arch", default=None, help="Filter to specific arch (gpt2, mamba, etc)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=200, help="Training steps for micro-train evals")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_extended_evals(
        arch_filter=args.arch,
        device=args.device,
        dry_run=args.dry_run,
        n_train_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
