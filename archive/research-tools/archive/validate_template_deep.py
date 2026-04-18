#!/usr/bin/env python
"""Deep validation of a template: overfitting check, multi-seed, probe trajectory.

Designed to run autonomously overnight. Produces a comprehensive report.

Usage:
    python -m research.tools.validate_template_deep \
        --template gated_linear_attention_block \
        --steps 20000 --checkpoint-every 2500 --seeds 3 --device cuda
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_model(template_name, n_layers, dim, vocab, seed):
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


def _load_wikitext(batch_size, seq_len, device):
    from research.eval.wikitext_eval import (
        _download_wikitext,
        _DEFAULT_MAX_CHARS_TRAIN,
        _DEFAULT_MAX_CHARS_VAL,
    )
    import tiktoken

    train_path, val_path = _download_wikitext(
        max_chars_train=_DEFAULT_MAX_CHARS_TRAIN,
        max_chars_val=_DEFAULT_MAX_CHARS_VAL,
    )
    enc = tiktoken.get_encoding("gpt2")
    train_tok = torch.tensor(
        enc.encode(
            train_path.read_text(encoding="utf-8", errors="replace"),
            allowed_special=set(),
        ),
        dtype=torch.long,
    )
    val_tok = torch.tensor(
        enc.encode(
            val_path.read_text(encoding="utf-8", errors="replace"),
            allowed_special=set(),
        ),
        dtype=torch.long,
    )
    stride = batch_size * seq_len
    train_w = [
        train_tok[i * stride : (i + 1) * stride].reshape(batch_size, seq_len).to(device)
        for i in range(len(train_tok) // stride)
    ]
    val_w = [
        val_tok[i * stride : (i + 1) * stride].reshape(batch_size, seq_len).to(device)
        for i in range(min(64, len(val_tok) // stride))
    ]
    return train_w, val_w


def _eval_loss(model, batches, vocab_size):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for b in batches:
            logits = model(b)
            total += nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                b[:, 1:].reshape(-1),
            ).item()
    model.train()
    return total / max(len(batches), 1)


def _run_probes(model, vocab_size, device):
    model.eval()
    probes = {}

    try:
        from research.eval.binding_pipeline import run_screening_binding_probes

        bp = run_screening_binding_probes(model, device=device)
        probes["induction_auc"] = bp.get("induction_auc", 0)
        probes["binding_auc"] = bp.get("binding_auc", 0)
        probes["binding_composite"] = bp.get("binding_composite", 0)
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
        probes["ar_auc"] = ar.auc
        probes["ar_final_acc"] = ar.final_acc
    except Exception as e:
        logger.warning("AR failed: %s", e)

    try:
        from research.eval.hellaswag_eval import evaluate_hellaswag

        h = evaluate_hellaswag(
            model, vocab_size=vocab_size, device=device, n_examples=200
        )
        probes["hellaswag_acc"] = h.get("hellaswag_acc")
    except Exception as e:
        logger.warning("HellaSwag failed: %s", e)

    try:
        from research.eval.blimp_eval import evaluate_blimp

        bl = evaluate_blimp(
            model, vocab_size=vocab_size, device=device, n_per_subtask=50, timeout_s=120
        )
        probes["blimp_accuracy"] = bl.overall_accuracy
    except Exception as e:
        logger.warning("BLiMP failed: %s", e)

    model.train()
    return probes


def validate_template(
    template_name: str,
    n_steps: int = 20000,
    checkpoint_every: int = 2500,
    n_layers: int = 4,
    model_dim: int = 256,
    vocab_size: int = 100277,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
    seeds: List[int] = None,
    device: str = "cuda",
    run_probes_at_checkpoints: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = [42]

    logger.info("Loading WikiText-103...")
    train_w, val_w = _load_wikitext(batch_size, seq_len, device)
    logger.info("Train: %d windows, Val: %d windows", len(train_w), len(val_w))

    all_seed_results = []

    for seed in seeds:
        logger.info("=" * 70)
        logger.info(
            "  %s  seed=%d  %dL dim=%d  %d steps",
            template_name,
            seed,
            n_layers,
            model_dim,
            n_steps,
        )
        logger.info("=" * 70)

        model = _build_model(template_name, n_layers, model_dim, vocab_size, seed).to(
            device
        )
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        rg = torch.Generator().manual_seed(seed)

        pre_val = _eval_loss(model, val_w, vocab_size)
        logger.info("Pre-train val: %.4f  Params: %d", pre_val, n_params)

        checkpoints = []
        train_losses = []
        t0 = time.time()

        for step in range(1, n_steps + 1):
            if step <= 100:
                for g in opt.param_groups:
                    g["lr"] = lr * step / 100

            batch = train_w[torch.randint(len(train_w), (1,), generator=rg).item()]
            logits = model(batch)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

            if not math.isfinite(loss.item()):
                logger.error("DIVERGED at step %d", step)
                break

            if step % checkpoint_every == 0 or step == n_steps:
                val_loss = _eval_loss(model, val_w, vocab_size)
                avg_train = sum(train_losses[-checkpoint_every:]) / min(
                    checkpoint_every, len(train_losses)
                )
                gen_gap = val_loss - avg_train
                ppl = math.exp(min(val_loss, 20))
                elapsed = time.time() - t0

                cp = {
                    "step": step,
                    "train_loss": round(avg_train, 4),
                    "val_loss": round(val_loss, 4),
                    "ppl": round(ppl, 1),
                    "gen_gap": round(gen_gap, 4),
                    "loss_ratio": round(val_loss / pre_val, 4),
                    "elapsed_s": round(elapsed, 1),
                }

                # Run probes at checkpoints if requested
                if run_probes_at_checkpoints and (
                    step == n_steps or step % (checkpoint_every * 2) == 0
                ):
                    logger.info("  Running probes at step %d...", step)
                    probes = _run_probes(model, vocab_size, device)
                    cp["probes"] = probes
                    probe_str = " ".join(
                        f"{k}={v:.4f}"
                        for k, v in probes.items()
                        if v is not None and isinstance(v, (int, float))
                    )
                    logger.info(
                        "  step %d: train=%.4f val=%.4f ppl=%.1f gap=%.4f | %s",
                        step,
                        avg_train,
                        val_loss,
                        ppl,
                        gen_gap,
                        probe_str,
                    )
                else:
                    logger.info(
                        "  step %d: train=%.4f val=%.4f ppl=%.1f gap=%.4f (%.0fs)",
                        step,
                        avg_train,
                        val_loss,
                        ppl,
                        gen_gap,
                        elapsed,
                    )

                checkpoints.append(cp)

        # Overfitting analysis
        if len(checkpoints) >= 2:
            first_gap = checkpoints[0]["gen_gap"]
            last_gap = checkpoints[-1]["gen_gap"]
            gap_trend = last_gap - first_gap
            val_improving = checkpoints[-1]["val_loss"] < checkpoints[-2]["val_loss"]
            train_val_ratio = checkpoints[-1]["train_loss"] / max(
                checkpoints[-1]["val_loss"], 1e-6
            )
        else:
            gap_trend = 0
            val_improving = False
            train_val_ratio = 1.0

        seed_result = {
            "seed": seed,
            "n_params": n_params,
            "pre_val": pre_val,
            "checkpoints": checkpoints,
            "final_val": checkpoints[-1]["val_loss"] if checkpoints else None,
            "final_ppl": checkpoints[-1]["ppl"] if checkpoints else None,
            "final_train": checkpoints[-1]["train_loss"] if checkpoints else None,
            "gen_gap_trend": round(gap_trend, 4),
            "val_still_improving": val_improving,
            "train_val_ratio": round(train_val_ratio, 4),
            "overfitting_risk": "HIGH"
            if gap_trend > 1.0
            else "MODERATE"
            if gap_trend > 0.3
            else "LOW",
        }
        all_seed_results.append(seed_result)

        logger.info(
            "  SEED %d DONE: val=%.4f ppl=%.1f gap_trend=%.4f overfit=%s",
            seed,
            seed_result["final_val"] or 0,
            seed_result["final_ppl"] or 0,
            gap_trend,
            seed_result["overfitting_risk"],
        )

        del model, opt
        torch.cuda.empty_cache()

    # Cross-seed analysis
    final_vals = [
        r["final_val"] for r in all_seed_results if r["final_val"] is not None
    ]
    final_ppls = [
        r["final_ppl"] for r in all_seed_results if r["final_ppl"] is not None
    ]

    report = {
        "template": template_name,
        "config": {
            "n_steps": n_steps,
            "n_layers": n_layers,
            "model_dim": model_dim,
            "seeds": seeds,
            "lr": lr,
        },
        "seed_results": all_seed_results,
        "summary": {
            "mean_val": round(sum(final_vals) / len(final_vals), 4)
            if final_vals
            else None,
            "std_val": round(
                (
                    sum(
                        (v - sum(final_vals) / len(final_vals)) ** 2 for v in final_vals
                    )
                    / len(final_vals)
                )
                ** 0.5,
                4,
            )
            if len(final_vals) > 1
            else 0,
            "mean_ppl": round(sum(final_ppls) / len(final_ppls), 1)
            if final_ppls
            else None,
            "consistent": all(
                r["overfitting_risk"] != "HIGH" for r in all_seed_results
            ),
            "all_seeds_improving": all(
                r["val_still_improving"] for r in all_seed_results
            ),
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print(f"DEEP VALIDATION REPORT: {template_name}")
    print("=" * 70)
    for r in all_seed_results:
        print(
            f"  seed={r['seed']:>3d}: val={r['final_val']:.4f} ppl={r['final_ppl']:.1f} "
            f"train={r['final_train']:.4f} gap_trend={r['gen_gap_trend']:+.4f} "
            f"overfit={r['overfitting_risk']}"
        )
        # Print final probes if available
        last_cp = r["checkpoints"][-1] if r["checkpoints"] else {}
        if "probes" in last_cp:
            probe_str = " ".join(
                f"{k}={v:.4f}"
                for k, v in last_cp["probes"].items()
                if v is not None and isinstance(v, (int, float))
            )
            print(f"         probes: {probe_str}")

    s = report["summary"]
    print(
        f"\n  Mean val: {s['mean_val']}  Std: {s['std_val']}  Mean PPL: {s['mean_ppl']}"
    )
    print(f"  Consistent: {s['consistent']}  All improving: {s['all_seeds_improving']}")

    # Verdict
    if s["consistent"] and s["mean_ppl"] and s["mean_ppl"] < 50:
        print("\n  VERDICT: STRONG — low PPL, not overfitting, consistent across seeds")
    elif s["consistent"] and s["mean_ppl"] and s["mean_ppl"] < 200:
        print("\n  VERDICT: PROMISING — decent PPL, needs more steps")
    elif not s["consistent"]:
        print("\n  VERDICT: UNSTABLE — overfitting or inconsistent across seeds")
    else:
        print("\n  VERDICT: INCONCLUSIVE — needs longer training")

    # Save
    out = (
        Path("research/reports")
        / f"deep_validate_{template_name}_{int(time.time())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved: {out}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Deep template validation")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-probe-checkpoints",
        action="store_true",
        help="Only run probes at final step",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    validate_template(
        template_name=args.template,
        n_steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        n_layers=args.layers,
        model_dim=args.dim,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seeds=seeds,
        device=args.device,
        run_probes_at_checkpoints=not args.no_probe_checkpoints,
    )


if __name__ == "__main__":
    main()
