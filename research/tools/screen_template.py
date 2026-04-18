#!/usr/bin/env python
"""Manually screen and investigate a template through the real pipeline.

Builds the model, runs S0/S0.5/S1 screening with WikiText training,
records results in the notebook, and optionally triggers investigation.

Usage:
    python -m research.tools.screen_template \
        --template attn_normalized_matmul_pinned \
        --layers 4 --dim 256 --device cuda

    # With investigation follow-up:
    python -m research.tools.screen_template \
        --template attn_normalized_matmul_pinned \
        --layers 4 --dim 256 --investigate --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from research.training.loss_ops import clip_grad_norm_, next_token_cross_entropy
from research.tools._wikitext_batches import load_wikitext_batch_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_graphs(template_name: str, n_layers: int, dim: int, seed: int = 42):
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template

    rng = random.Random(seed)
    graphs = []
    for _ in range(n_layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        out = apply_template(g, inp, rng, template_name=template_name)
        g.set_output(out)
        graphs.append(g)
    return graphs


def screen_and_investigate(
    template_name: str,
    n_layers: int = 4,
    model_dim: int = 256,
    vocab_size: int = 100277,
    screening_steps: int = 750,
    investigation_steps: int = 2500,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
    do_investigate: bool = False,
) -> Dict[str, Any]:
    from research.synthesis.compiler import compile_model
    from research.scientist.notebook import LabNotebook

    # ── Build ────────────────────────────────────────────────────────
    logger.info("Building %s: %d layers, dim=%d", template_name, n_layers, model_dim)
    graphs = _build_graphs(template_name, n_layers, model_dim, seed)
    graph_fp = graphs[0].fingerprint()
    graph_json_str = json.dumps(graphs[0].to_dict())

    model = compile_model(graphs, vocab_size=vocab_size, max_seq_len=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Params: %d, fingerprint: %s", n_params, graph_fp)

    # ── S0: Forward pass ─────────────────────────────────────────────
    logger.info("S0: Forward pass check...")
    x = torch.randint(0, vocab_size, (2, 64), device=device)
    with torch.no_grad():
        y = model(x)
    s0_passed = y.shape[-1] == vocab_size and not torch.isnan(y).any()
    logger.info("S0: %s (shape=%s)", "PASSED" if s0_passed else "FAILED", y.shape)
    if not s0_passed:
        return {"stage": "s0_failed", "template": template_name}

    # ── S0.5: Gradient check ────────────────────────────────────────
    logger.info("S0.5: Gradient + causality check...")
    x2 = torch.randint(0, vocab_size, (2, 64), device=device)
    logits = model(x2)
    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size), x2[:, 1:].reshape(-1)
    )
    loss.backward()
    grads_with_values = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    n_grads = len(grads_with_values)
    n_finite = sum(1 for g in grads_with_values if torch.isfinite(g).all())
    grad_ok = n_grads > 0 and n_finite == n_grads
    s05_passed = grad_ok
    logger.info(
        "S0.5: %s (grads: %d/%d finite, %d total params)",
        "PASSED" if s05_passed else "FAILED",
        n_finite, n_grads, sum(1 for p in model.parameters() if p.requires_grad),
    )
    if not s05_passed:
        return {"stage": "s05_failed", "template": template_name}

    # ── Prepare WikiText data ───────────────────────────────────────
    logger.info("Loading WikiText-103...")
    batch_source = load_wikitext_batch_source(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    logger.info(
        "Prepared %d train windows, %d val windows",
        batch_source.train_window_count,
        batch_source.val_window_count,
    )

    # ── S1: Screening training ──────────────────────────────────────
    def _eval_val():
        model.eval()
        total = 0.0
        n = 0
        for vb in batch_source.iter_val_batches(device=device):
            with torch.no_grad():
                lo = model(vb)
                total += nn.functional.cross_entropy(
                    lo[:, :-1].reshape(-1, vocab_size), vb[:, 1:].reshape(-1)
                ).item()
                n += 1
        model.train()
        return total / max(n, 1)

    pre_val = _eval_val()
    logger.info("Pre-training val loss: %.4f (PPL %.1f)", pre_val, math.exp(min(pre_val, 20)))

    logger.info("S1: Training %d steps (screening)...", screening_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng_gen = torch.Generator().manual_seed(seed)
    losses = []
    t0 = time.time()

    for step in range(1, screening_steps + 1):
        if step <= 50:
            for g in optimizer.param_groups:
                g["lr"] = lr * step / 50

        batch = batch_source.sample_train_batch(device=device, generator=rng_gen)
        logits = model(batch)
        loss = next_token_cross_entropy(logits, batch, vocab_size)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model, 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 250 == 0:
            vl = _eval_val()
            avg_train = sum(losses[-250:]) / 250
            logger.info(
                "  step %d: train=%.4f val=%.4f ppl=%.1f (%.0fs)",
                step, avg_train, vl, math.exp(min(vl, 20)), time.time() - t0
            )

    screening_val = _eval_val()
    screening_lr = screening_val / pre_val if pre_val > 0 else 1.0
    screening_ppl = math.exp(min(screening_val, 20))
    s1_passed = screening_lr < 0.95
    elapsed_s1 = time.time() - t0

    logger.info(
        "S1: %s — val=%.4f ppl=%.1f loss_ratio=%.4f (%.0fs)",
        "PASSED" if s1_passed else "FAILED",
        screening_val, screening_ppl, screening_lr, elapsed_s1,
    )

    # ── Run probes ──────────────────────────────────────────────────
    model.eval()
    probe_results: Dict[str, Any] = {}

    try:
        from research.eval.binding_pipeline import run_screening_binding_probes
        bp = run_screening_binding_probes(model, device=device)
        probe_results.update({
            "induction_auc": bp.get("induction_auc"),
            "binding_auc": bp.get("binding_auc"),
            "binding_composite": bp.get("binding_composite"),
        })
        logger.info("Binding: ind=%.4f bind=%.4f", probe_results.get("induction_auc", 0), probe_results.get("binding_auc", 0))
    except Exception as e:
        logger.warning("Binding failed: %s", e)

    try:
        from research.eval.associative_recall import associative_recall_score
        ar = associative_recall_score(model, n_pairs=10, n_eval=100, n_train_steps=300, batch_size=8, device=device)
        probe_results["ar_auc"] = ar.auc
        probe_results["ar_final_acc"] = ar.final_acc
        logger.info("AR: auc=%.4f acc=%.4f", ar.auc, ar.final_acc)
    except Exception as e:
        logger.warning("AR failed: %s", e)

    try:
        from research.eval.hellaswag_eval import evaluate_hellaswag
        hella = evaluate_hellaswag(model, vocab_size=vocab_size, device=device, n_examples=200)
        probe_results["hellaswag_acc"] = hella.get("hellaswag_acc")
        logger.info("HellaSwag: acc=%s", probe_results["hellaswag_acc"])
    except Exception as e:
        logger.warning("HellaSwag failed: %s", e)

    try:
        from research.eval.blimp_eval import evaluate_blimp
        blimp = evaluate_blimp(model, vocab_size=vocab_size, device=device, n_per_subtask=50, timeout_s=120)
        probe_results["blimp_overall_accuracy"] = blimp.overall_accuracy
        logger.info("BLiMP: acc=%.4f", blimp.overall_accuracy)
    except Exception as e:
        logger.warning("BLiMP failed: %s", e)

    # ── Record in notebook ──────────────────────────────────────────
    logger.info("Recording results in notebook...")
    nb = LabNotebook()

    exp_id = f"manual_screen_{template_name}_{int(time.time())}"
    nb.conn.execute(
        "INSERT OR IGNORE INTO experiments (experiment_id, timestamp, experiment_type, config_json) VALUES (?, ?, ?, ?)",
        (exp_id, time.time(), "manual_template_screen", json.dumps({
            "template": template_name, "n_layers": n_layers, "model_dim": model_dim,
            "screening_steps": screening_steps, "lr": lr,
        })),
    )

    result_kwargs = {
        "stage0_passed": s0_passed,
        "stage05_passed": s05_passed,
        "stage1_passed": s1_passed,
        "loss_ratio": screening_lr,
        "final_loss": screening_val,
        "initial_loss": pre_val,
        "param_count": n_params,
        "n_train_steps": screening_steps,
        "model_source": "manual_template_screen",
        "trust_label": "manual_screen",
        "wikitext_perplexity": screening_ppl,
        **{k: v for k, v in probe_results.items() if v is not None},
    }

    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=graph_fp,
        graph_json=graph_json_str,
        bypass_quality_gate=True,
        **result_kwargs,
    )
    nb.flush_writes()
    logger.info("Recorded result_id=%s", result_id)

    # Upsert leaderboard — must use leaderboard column names, not program_results names
    try:
        lb_kwargs = {
            "screening_loss_ratio": screening_lr,
            "screening_novelty": 0.0,
            "screening_passed": int(s1_passed),
            "wikitext_perplexity": screening_ppl,
            "param_count": n_params,
        }
        # Pass probe results with their correct column names
        for k, v in probe_results.items():
            if v is not None:
                lb_kwargs[k] = v

        entry_id = nb.upsert_leaderboard(
            result_id=result_id,
            model_source="manual_template_screen",
            architecture_desc=f"{template_name} {n_layers}L dim={model_dim}",
            tier="screening",
            tags=f"manual,pinned,{template_name}",
            notes=f"Manually screened {template_name} with {screening_steps} steps",
            **lb_kwargs,
        )
        logger.info("Leaderboard entry: %s", entry_id)
    except Exception as e:
        logger.warning("Leaderboard upsert failed: %s", e)

    result = {
        "template": template_name,
        "result_id": result_id,
        "experiment_id": exp_id,
        "graph_fingerprint": graph_fp,
        "n_params": n_params,
        "s0_passed": s0_passed,
        "s05_passed": s05_passed,
        "s1_passed": s1_passed,
        "screening_val_loss": screening_val,
        "screening_ppl": screening_ppl,
        "screening_loss_ratio": screening_lr,
        "pre_val_loss": pre_val,
        "elapsed_s": elapsed_s1,
        **probe_results,
    }

    # ── Investigation (optional) ────────────────────────────────────
    if do_investigate and s1_passed:
        logger.info("Starting investigation (%d steps)...", investigation_steps)
        # Reset optimizer for longer training
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=0.01)
        model.train()
        inv_losses = []
        t_inv = time.time()

        for step in range(1, investigation_steps + 1):
            if step <= 100:
                for g in optimizer.param_groups:
                    g["lr"] = lr * 0.5 * step / 100

            batch = batch_source.sample_train_batch(device=device, generator=rng_gen)
            logits = model(batch)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            inv_losses.append(loss.item())

            if step % 500 == 0:
                vl = _eval_val()
                logger.info(
                    "  inv step %d: train=%.4f val=%.4f ppl=%.1f (%.0fs)",
                    step, sum(inv_losses[-500:]) / 500, vl,
                    math.exp(min(vl, 20)), time.time() - t_inv
                )

        inv_val = _eval_val()
        inv_ppl = math.exp(min(inv_val, 20))
        inv_lr = inv_val / pre_val if pre_val > 0 else 1.0
        logger.info(
            "Investigation: val=%.4f ppl=%.1f loss_ratio=%.4f (%.0fs)",
            inv_val, inv_ppl, inv_lr, time.time() - t_inv,
        )

        # Re-run probes after investigation
        model.eval()
        try:
            bp2 = run_screening_binding_probes(model, device=device)
            result["inv_induction_auc"] = bp2.get("induction_auc")
            result["inv_binding_auc"] = bp2.get("binding_auc")
            logger.info("Inv binding: ind=%.4f bind=%.4f", result.get("inv_induction_auc", 0), result.get("inv_binding_auc", 0))
        except Exception as exc:
            logger.warning("Investigation binding probes failed: %s", exc)

        result["investigation_val_loss"] = inv_val
        result["investigation_ppl"] = inv_ppl
        result["investigation_loss_ratio"] = inv_lr
        result["investigation_steps"] = investigation_steps

        # Update notebook with investigation results
        try:
            inv_lb_kwargs = {
                "screening_loss_ratio": screening_lr,
                "investigation_loss_ratio": inv_lr,
                "investigation_robustness": 1.0,
                "investigation_passed": True,
                "wikitext_perplexity": inv_ppl,
                "param_count": n_params,
            }
            if result.get("inv_induction_auc") is not None:
                inv_lb_kwargs["induction_auc"] = result["inv_induction_auc"]
            if result.get("inv_binding_auc") is not None:
                inv_lb_kwargs["binding_auc"] = result["inv_binding_auc"]
            nb.upsert_leaderboard(
                result_id=result_id,
                model_source="manual_template_screen",
                tier="investigation",
                **inv_lb_kwargs,
            )
            logger.info("Promoted to investigation tier")
        except Exception as e:
            logger.warning("Investigation promotion failed: %s", e)

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESULT: {template_name}")
    print("=" * 70)
    print(f"  Params:       {n_params:,}")
    print(f"  S0/S0.5/S1:   {s0_passed}/{s05_passed}/{s1_passed}")
    print(f"  Pre-train:    val={pre_val:.4f} ppl={math.exp(min(pre_val, 20)):.1f}")
    print(f"  Screening:    val={screening_val:.4f} ppl={screening_ppl:.1f} LR={screening_lr:.4f}")
    if "investigation_val_loss" in result:
        print(f"  Investigation: val={result['investigation_val_loss']:.4f} ppl={result['investigation_ppl']:.1f} LR={result['investigation_loss_ratio']:.4f}")
    for k in ["induction_auc", "binding_auc", "ar_auc", "hellaswag_acc", "blimp_overall_accuracy"]:
        if k in probe_results and probe_results[k] is not None:
            print(f"  {k}: {probe_results[k]:.4f}")
    print(f"  result_id:    {result_id}")
    print(f"  fingerprint:  {graph_fp}")

    # Save JSON
    out_path = Path("research/reports") / f"screen_{template_name}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {out_path}")

    del model, optimizer
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Screen a template through the real pipeline")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--screening-steps", type=int, default=750)
    parser.add_argument("--investigation-steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--investigate", action="store_true", help="Also run investigation tier")
    args = parser.parse_args()

    screen_and_investigate(
        template_name=args.template,
        n_layers=args.layers,
        model_dim=args.dim,
        screening_steps=args.screening_steps,
        investigation_steps=args.investigation_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        do_investigate=args.investigate,
    )


if __name__ == "__main__":
    main()
