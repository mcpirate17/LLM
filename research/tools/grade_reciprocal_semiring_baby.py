#!/usr/bin/env python
"""Baby-scale grade of reciprocal_semiring_attention vs parents vs softmax.

Trains each attention op on the SAME real data mix the frontier/MoR runs use
(``codex_ffw60_chat30_pleias10_local`` — 60% FineFineWeb-local + 30% small-chat
+ 10% Pleias, cl100k) via the HYDRA universal loader, then grades by held-out
perplexity on the same mix + BLiMP. Fixed seed/data order per op so the only
variable is the mixer.

Scale is set by --dim/--layers; dim288×6 ≈ 20.0M active (non-embedding) params.

Usage:
    python -m research.tools.grade_reciprocal_semiring_baby \
        --dim 288 --layers 6 --steps 1500 --seq-len 256 --batch 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

import research.tools.native_adaptive_hydra_train as T
from research.eval.blimp_eval import evaluate_blimp
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import apply_template
from research.training.loss_ops import clip_grad_norm_, next_token_cross_entropy

VOCAB = 100277

OPS = {
    "reciprocal_semiring (composed)": "reciprocal_semiring_attention_block",
    "reciprocal_rank (parent A)": "reciprocal_rank_attention_block",
    "learnable_semiring (parent B)": "learnable_semiring_attention_block",
    "softmax (gpt2_reference)": "gpt2_reference",
}


def _build(template: str, dim: int, layers: int, seq_len: int, seed: int):
    rng = random.Random(seed)
    graphs = []
    for _ in range(layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        g.set_output(apply_template(g, inp, rng, template_name=template))
        graphs.append(g)
    return compile_model(graphs, vocab_size=VOCAB, max_seq_len=seq_len)


def _active_params(model) -> int:
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embed" not in n.lower() and n not in ("norm.weight", "norm.bias")
    )


def _loader_ns(args, dataset_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        hydra_root=Path("HYDRA"),
        tokenizer="gpt2",
        vocab_size=VOCAB,
        batch=args.batch,
        seq_len=args.seq_len,
        num_workers=0,
        prefetch_factor=2,
        steps=dataset_steps,
        require_sources=True,
    )


@torch.no_grad()
def _val_ppl(model, val_loader, n_batches: int, device: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        ids, labels = T._prepare_batch(val_loader.get_batch(), vocab_size=VOCAB)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        tot += next_token_cross_entropy(logits, labels, VOCAB).item()
        n += 1
    model.train()
    return math.exp(tot / max(n, 1))


def _grade(label: str, template: str, args, device: str) -> dict:
    torch.manual_seed(args.seed)
    model = _build(template, args.dim, args.layers, args.seq_len, args.seed).to(device)
    active = _active_params(model)
    total = sum(p.numel() for p in model.parameters())

    train_loader = T._make_loader(
        _loader_ns(args, args.steps), dataset=T.LOCAL_MIX_NAME, seed=args.seed
    )
    val_loader = T._make_loader(
        _loader_ns(args, args.val_batches + 1),
        dataset=T.LOCAL_MIX_NAME,
        seed=args.seed + 1009,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, args.steps // 20)

    losses = []
    for step in range(1, args.steps + 1):
        if step <= warmup:
            frac = step / warmup
        else:
            prog = (step - warmup) / max(1, args.steps - warmup)
            frac = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        for grp in opt.param_groups:
            grp["lr"] = args.lr * frac

        ids, labels = T._prepare_batch(train_loader.get_batch(), vocab_size=VOCAB)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss = next_token_cross_entropy(logits, labels, VOCAB)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0:
            print(
                f"    [{label}] step {step}/{args.steps} "
                f"loss {sum(losses[-50:]) / len(losses[-50:]):.3f}",
                flush=True,
            )

    val_ppl = _val_ppl(model, val_loader, args.val_batches, device)
    blimp = None
    if not args.no_blimp:
        try:
            br = evaluate_blimp(
                model, vocab_size=VOCAB, device=device, n_per_subtask=args.blimp_n
            )
            blimp = round(br.overall_accuracy, 4)
        except Exception as e:  # noqa: BLE001 - report, do not swallow silently
            blimp = f"ERR:{e}"
    train_loader.close()
    val_loader.close()
    return {
        "label": label,
        "template": template,
        "active_m": round(active / 1e6, 2),
        "total_m": round(total / 1e6, 2),
        "final_train_loss": round(sum(losses[-50:]) / len(losses[-50:]), 4),
        "val_ppl": round(val_ppl, 2),
        "blimp": blimp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=288)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--blimp-n", type=int, default=200)
    ap.add_argument("--no-blimp", action="store_true")
    ap.add_argument("--ops", default=None, help="comma-separated subset of labels")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/reciprocal_semiring_baby20m_grade.json"),
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chosen = OPS
    if args.ops:
        keys = [k.strip() for k in args.ops.split(",")]
        chosen = {k: v for k, v in OPS.items() if any(s in k for s in keys)}

    rows = []
    for label, tpl in chosen.items():
        print(f"=== {label} ({tpl}) ===", flush=True)
        t_op = time.perf_counter()
        r = _grade(label, tpl, args, device)
        r["elapsed_s"] = round(time.perf_counter() - t_op, 1)
        rows.append(r)
        print(
            f"  {label:32s} active={r['active_m']}M val_ppl={r['val_ppl']} "
            f"blimp={r['blimp']} ({r['elapsed_s']}s)",
            flush=True,
        )

    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(
        f"\ndim{args.dim}x{args.layers} steps={args.steps} seq={args.seq_len} "
        f"batch={args.batch} seed={args.seed} device={device}"
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
