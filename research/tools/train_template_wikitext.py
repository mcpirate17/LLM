#!/usr/bin/env python
"""Train template models on WikiText-103 with Muon optimizer.

Produces actual language-modeling loss (not random-token entropy).
Checkpoints loss every --checkpoint-every steps and runs probes at end.

Usage:
    # Top 2 candidates, 20K steps, Muon, checkpoint every 1K
    python -m research.tools.train_template_wikitext \
        --templates latent_attn_padic_hybrid,attn_normalized_matmul \
        --steps 20000 --checkpoint-every 1000 --optimizer muon \
        --device cuda

    # Compare against GPT-2 baseline
    python -m research.tools.train_template_wikitext \
        --templates latent_attn_padic_hybrid,gpt2_reference \
        --steps 20000 --checkpoint-every 1000 --optimizer muon
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

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


def _build_model(template_name: str, n_layers: int, dim: int, vocab: int, seed: int):
    from research.synthesis.compiler import compile_model
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
    return compile_model(graphs, vocab_size=vocab, max_seq_len=512)


def _make_optimizer(model, name: str, lr: float, wd: float):
    params = list(model.parameters())
    if name == "muon":
        from research.training._optimizer_muon import MuonOptimizer

        # Muon for 2D+ params, AdamW for 1D (embeddings, norms)
        muon_params = [p for p in params if p.ndim >= 2]
        adam_params = [p for p in params if p.ndim < 2]
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "optimizer": "muon"})
        if adam_params:
            groups.append({"params": adam_params, "optimizer": "adamw"})

        # Combined: Muon for bulk, AdamW for scalars
        if muon_params:
            muon_opt = MuonOptimizer(muon_params, lr=lr, weight_decay=wd, momentum=0.95)
        else:
            muon_opt = None
        adam_opt = (
            torch.optim.AdamW(adam_params, lr=lr * 0.1, weight_decay=wd)
            if adam_params
            else None
        )
        return muon_opt, adam_opt
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        return opt, None


def _eval_loss(model, batch_source, vocab_size, device: str):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in batch_source.iter_val_batches(device=device):
            logits = model(batch)
            loss = next_token_cross_entropy(logits, batch, vocab_size)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def train_template(
    template_name: str,
    n_steps: int = 20000,
    n_layers: int = 4,
    model_dim: int = 256,
    vocab_size: int = 100277,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 0.02,
    wd: float = 0.01,
    optimizer_name: str = "muon",
    checkpoint_every: int = 1000,
    warmup_steps: int = 200,
    device: str = "cuda",
    seed: int = 42,
    run_probes: bool = True,
) -> Dict[str, Any]:
    """Train a template on WikiText-103 and report checkpointed losses."""
    logger.info(
        "Building model: %s (%d layers, dim=%d)", template_name, n_layers, model_dim
    )
    model = _build_model(template_name, n_layers, model_dim, vocab_size, seed).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %d", n_params)

    # Build optimizer
    opt_main, opt_aux = _make_optimizer(model, optimizer_name, lr, wd)

    logger.info("Loading WikiText-103 batches...")
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

    # Training loop
    checkpoints: List[Dict[str, Any]] = []
    losses = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Warmup
        if step <= warmup_steps:
            factor = step / warmup_steps
            if opt_main is not None:
                for g in opt_main.param_groups:
                    g["lr"] = lr * factor
        if opt_aux is not None:
            for g in opt_aux.param_groups:
                g["lr"] = lr * 0.1 * factor

        # Forward
        batch = batch_source.sample_train_batch(device=device)
        logits = model(batch)
        loss = next_token_cross_entropy(logits, batch, vocab_size)

        # Backward
        if opt_main is not None:
            opt_main.zero_grad(set_to_none=True)
        if opt_aux is not None:
            opt_aux.zero_grad(set_to_none=True)

        loss.backward()
        all_params = list(model.parameters())
        clip_grad_norm_(all_params, 1.0)

        if opt_main is not None:
            opt_main.step()
        if opt_aux is not None:
            opt_aux.step()

        train_loss = loss.item()
        losses.append(train_loss)

        if not math.isfinite(train_loss):
            logger.error("DIVERGED at step %d", step)
            break

        # Checkpoint
        if step % checkpoint_every == 0 or step == n_steps:
            val_loss = _eval_loss(model, batch_source, vocab_size, device)
            avg_train = sum(losses[-checkpoint_every:]) / min(
                checkpoint_every, len(losses)
            )
            elapsed = time.time() - t0
            ppl = math.exp(min(val_loss, 20))
            cp = {
                "step": step,
                "train_loss": round(avg_train, 4),
                "val_loss": round(val_loss, 4),
                "ppl": round(ppl, 1),
                "elapsed_s": round(elapsed, 1),
            }
            checkpoints.append(cp)
            print(
                f"  [{template_name}] step={step:>6d}  "
                f"train={avg_train:.4f}  val={val_loss:.4f}  "
                f"ppl={ppl:.1f}  elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    result: Dict[str, Any] = {
        "template": template_name,
        "n_params": n_params,
        "n_steps": len(losses),
        "optimizer": optimizer_name,
        "lr": lr,
        "n_layers": n_layers,
        "model_dim": model_dim,
        "elapsed_s": elapsed,
        "checkpoints": checkpoints,
    }

    # Run probes
    if run_probes and checkpoints:
        model.eval()
        logger.info("Running probes for %s...", template_name)
        try:
            from research.eval.binding_pipeline import run_screening_binding_probes

            bp = run_screening_binding_probes(model, device=device)
            result["induction_auc"] = bp.get("induction_auc")
            result["binding_auc"] = bp.get("binding_auc")
            result["binding_composite"] = bp.get("binding_composite")
            logger.info(
                "  Binding: ind=%.4f bind=%.4f",
                result["induction_auc"] or 0,
                result["binding_auc"] or 0,
            )
        except Exception as e:
            logger.warning("Binding probes failed: %s", e)

        try:
            from research.eval.associative_recall import associative_recall_score

            ar = associative_recall_score(
                model,
                n_pairs=10,
                n_eval=100,
                n_train_steps=300,
                batch_size=8,
                device=device,
            )
            result["ar_auc"] = ar.auc
            result["ar_final_acc"] = ar.final_acc
            logger.info("  AR: auc=%.4f acc=%.4f", ar.auc, ar.final_acc)
        except Exception as e:
            logger.warning("AR failed: %s", e)

        try:
            from research.eval.hellaswag_eval import evaluate_hellaswag

            hella = evaluate_hellaswag(
                model, vocab_size=vocab_size, device=device, n_examples=200
            )
            result["hellaswag_acc"] = hella.get("hellaswag_acc")
            logger.info("  HellaSwag: acc=%s", result["hellaswag_acc"])
        except Exception as e:
            logger.warning("HellaSwag failed: %s", e)

        try:
            from research.eval.blimp_eval import evaluate_blimp

            blimp = evaluate_blimp(
                model,
                vocab_size=vocab_size,
                device=device,
                n_per_subtask=50,
                timeout_s=120,
            )
            result["blimp_accuracy"] = blimp.overall_accuracy
            logger.info("  BLiMP: acc=%.4f", blimp.overall_accuracy)
        except Exception as e:
            logger.warning("BLiMP failed: %s", e)

    del model
    if opt_main is not None:
        del opt_main
    if opt_aux is not None:
        del opt_aux
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train templates on WikiText-103 with loss checkpoints"
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Comma-separated template names",
    )
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000, help="Steps between checkpoints"
    )
    parser.add_argument(
        "--optimizer", type=str, default="muon", choices=["muon", "adamw"]
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 0.02 muon, 3e-4 adamw)",
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-probes", action="store_true", help="Skip eval probes")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.005 if args.optimizer == "muon" else 3e-4

    templates = [t.strip() for t in args.templates.split(",")]

    print(f"WikiText-103 training: {args.steps} steps, {args.optimizer} lr={args.lr}")
    print(
        f"Model: {args.layers} layers, dim={args.dim}, batch={args.batch_size}x{args.seq_len}"
    )
    print(f"Templates: {', '.join(templates)}")
    print(f"Checkpoints every {args.checkpoint_every} steps")
    print("=" * 80)

    results = []
    for name in templates:
        r = train_template(
            template_name=name,
            n_steps=args.steps,
            n_layers=args.layers,
            model_dim=args.dim,
            vocab_size=100277,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            wd=args.wd,
            optimizer_name=args.optimizer,
            checkpoint_every=args.checkpoint_every,
            warmup_steps=args.warmup,
            device=args.device,
            seed=args.seed,
            run_probes=not args.no_probes,
        )
        results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    for r in results:
        last_cp = r["checkpoints"][-1] if r["checkpoints"] else {}
        probes = " ".join(
            f"{k}={r[k]:.4f}"
            for k in [
                "induction_auc",
                "binding_auc",
                "ar_auc",
                "hellaswag_acc",
                "blimp_accuracy",
            ]
            if k in r and r[k] is not None
        )
        print(
            f"  {r['template']:40s}  val_loss={last_cp.get('val_loss', '?'):>7}  "
            f"ppl={last_cp.get('ppl', '?'):>8}  {probes}"
        )

    output = args.output or "research/reports/wikitext_train_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
