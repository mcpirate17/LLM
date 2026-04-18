#!/usr/bin/env python
"""Extended template evaluation: trains models, runs probes, reports results.

Usage:
    python -m research.tools.eval_templates [--steps 1000] [--templates t1,t2,...]

Evaluates each template by:
1. Building a 2-layer model (dim=128, vocab=100277)
2. Training for N steps on WikiText-103 with checkpoints
3. Running binding probes (induction, AR, binding_auc) at final
4. Running wikitext perplexity eval
5. Comparing to GPT-2 reference baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch

from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import apply_template, DEFAULT_TEMPLATE_WEIGHTS
from research.training.loss_ops import clip_grad_norm_, next_token_cross_entropy

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Templates under evaluation
DEFAULT_TEMPLATES = [
    # New high-performance templates
    "recursive_attn_ssm_depth",
    "latent_attn_padic_hybrid",
    "graph_attn_ssm_recursive",
    # Rewritten templates
    "attn_softmax_normalized_matmul_v2",
    "attn_linear_softmax_recovery_control",
    "attn_softmax_matmul_sparse_tail",
    "attn_normalized_matmul",
    "attn_bottleneck_hybrid",
    "depth_gated_block_matmul_stable",
    # Baselines
    "latent_attn_ssm_hybrid",
    "gpt2_reference",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(
    template_name: str,
    n_layers: int = 2,
    model_dim: int = 128,
    vocab_size: int = 100277,
    seed: int = 42,
):
    """Build a model from a template."""
    rng = random.Random(seed)
    graphs = []
    for _ in range(n_layers):
        g = ComputationGraph(model_dim=model_dim)
        inp = g.add_input()
        out = apply_template(g, inp, rng, template_name=template_name)
        g.set_output(out)
        graphs.append(g)

    model = compile_model(graphs, vocab_size=vocab_size, max_seq_len=256)
    return model


def train_and_eval(
    model,
    n_steps: int = 1000,
    batch_size: int = 8,
    seq_len: int = 128,
    lr: float = 3e-4,
    checkpoint_steps: tuple[int, ...] = (200, 500, 750, 1000),
    device: str = DEVICE,
):
    """Train model and collect loss trajectory + checkpoints."""
    model = model.to(device)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # Warmup schedule
    warmup_steps = min(50, n_steps // 10)

    losses = {}
    checkpoint_losses = {}
    vocab_size = model.vocab_size if hasattr(model, "vocab_size") else 100277

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Warmup
        if step <= warmup_steps:
            factor = step / warmup_steps
            for group in opt.param_groups:
                group["lr"] = lr * factor

        # Random token batch
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(tokens)

        # Next-token prediction loss
        loss = next_token_cross_entropy(logits, tokens, vocab_size)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, 1.0)
        opt.step()

        loss_val = loss.item()
        losses[step] = loss_val

        if step in checkpoint_steps:
            checkpoint_losses[step] = loss_val

        if not math.isfinite(loss_val):
            print(f"    DIVERGED at step {step}")
            break

    elapsed = time.time() - t0

    # Compute summary stats
    init_loss = sum(list(losses.values())[:10]) / min(10, len(losses))
    final_loss = sum(list(losses.values())[-10:]) / min(10, len(losses))
    improvement = (init_loss - final_loss) / init_loss if init_loss > 0 else 0
    loss_ratio = final_loss / init_loss if init_loss > 0 else 1.0

    return {
        "n_params": n_params,
        "n_steps": len(losses),
        "elapsed_s": elapsed,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "improvement": improvement,
        "loss_ratio": loss_ratio,
        "checkpoints": checkpoint_losses,
        "perplexity": math.exp(min(final_loss, 20)),
    }


def run_binding_probes(model, device: str = DEVICE):
    """Run binding probes if available."""
    try:
        from research.eval.binding_pipeline import run_screening_binding_probes

        result = run_screening_binding_probes(model, device=device)
        return {
            "binding_auc": getattr(result, "binding_auc", None),
            "induction_auc": getattr(result, "induction_auc", None),
            "ar_auc": getattr(result, "ar_auc", None),
            "ar_final_acc": getattr(result, "ar_final_acc", None),
            "ar_timed_out": getattr(result, "ar_timed_out", None),
        }
    except Exception as e:
        logger.warning("Binding probes failed: %s", e)
        return {"binding_auc": None, "induction_auc": None, "ar_auc": None}


def run_wikitext_eval(model, device: str = DEVICE):
    """Run wikitext perplexity evaluation if available."""
    try:
        from research.eval.wikitext_eval import screening_wikitext_eval

        result = screening_wikitext_eval(model, device=device)
        return {
            "wikitext_ppl": result.get("wikitext_perplexity"),
            "wikitext_score": result.get("wikitext_score"),
        }
    except Exception as e:
        logger.warning("WikiText eval failed: %s", e)
        return {"wikitext_ppl": None, "wikitext_score": None}


def evaluate_template(
    template_name: str,
    n_steps: int = 1000,
    run_probes: bool = True,
    device: str = DEVICE,
):
    """Full evaluation of a single template."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {template_name}")
    print(f"{'='*60}")

    # Build
    t0 = time.time()
    try:
        model = build_model(template_name)
    except Exception as e:
        print(f"  BUILD FAILED: {e}")
        return None
    build_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  Build: {build_time:.1f}s")

    # Train
    checkpoints = tuple(s for s in (200, 500, 750, 1000, 2000, 5000, 10000) if s <= n_steps)
    result = train_and_eval(
        model,
        n_steps=n_steps,
        checkpoint_steps=checkpoints,
        device=device,
    )

    print(f"  Training: {result['n_steps']} steps in {result['elapsed_s']:.1f}s")
    print(f"  Loss: {result['init_loss']:.4f} → {result['final_loss']:.4f} "
          f"(Δ={result['improvement']:+.1%}, ratio={result['loss_ratio']:.3f})")
    print(f"  PPL: {result['perplexity']:.1f}")

    for step, loss in sorted(result["checkpoints"].items()):
        print(f"    step {step:>5d}: loss={loss:.4f} ppl={math.exp(min(loss, 20)):.1f}")

    # Probes
    probe_results = {}
    if run_probes:
        print("  Running binding probes...")
        probe_results = run_binding_probes(model, device)
        for k, v in probe_results.items():
            if v is not None:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        print("  Running wikitext eval...")
        wiki_results = run_wikitext_eval(model, device)
        probe_results.update(wiki_results)
        for k, v in wiki_results.items():
            if v is not None:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    result.update(probe_results)
    result["template"] = template_name
    result["weight"] = DEFAULT_TEMPLATE_WEIGHTS.get(template_name, 0)

    # Cleanup GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Extended template evaluation")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--templates", type=str, default=None, help="Comma-separated template names")
    parser.add_argument("--no-probes", action="store_true", help="Skip binding/wikitext probes")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    templates = args.templates.split(",") if args.templates else DEFAULT_TEMPLATES

    print(f"Evaluating {len(templates)} templates for {args.steps} steps on {args.device}")
    print(f"Templates: {', '.join(templates)}")

    results = []
    for name in templates:
        result = evaluate_template(
            name,
            n_steps=args.steps,
            run_probes=not args.no_probes,
            device=args.device,
        )
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Template':45s} {'Loss':>7s} {'Ratio':>6s} {'PPL':>8s} {'Imp%':>6s} {'Ind':>6s} {'AR':>6s} {'Bind':>6s}")
    print("-" * 80)

    gpt2_loss = None
    for r in results:
        if r["template"] == "gpt2_reference":
            gpt2_loss = r["final_loss"]

    for r in results:
        ind = r.get("induction_auc")
        ar = r.get("ar_auc")
        bind = r.get("binding_auc")
        ind_s = f"{ind:.4f}" if ind is not None else "—"
        ar_s = f"{ar:.4f}" if ar is not None else "—"
        bind_s = f"{bind:.4f}" if bind is not None else "—"

        marker = ""
        if gpt2_loss and r["final_loss"] < gpt2_loss:
            margin = (gpt2_loss - r["final_loss"]) / gpt2_loss * 100
            marker = f" ✓ ({margin:.0f}% better)"

        print(f"{r['template']:45s} {r['final_loss']:>7.4f} {r['loss_ratio']:>6.3f} "
              f"{r['perplexity']:>8.1f} {r['improvement']:>+5.1%} "
              f"{ind_s:>6s} {ar_s:>6s} {bind_s:>6s}{marker}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
