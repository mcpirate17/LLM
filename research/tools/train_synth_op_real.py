#!/usr/bin/env python
"""Real-budget trainer for a synthesis attention op on the codex local mix.

Trains a compiled SynthesizedModel (any attention template) on the SAME data the
frontier/MoR runs use (``codex_ffw60_chat30_pleias10_local``, cl100k) with
warmup→cosine AdamW, grad clipping, a non-finite-step guard, periodic checkpoints
and milestone BLiMP — so the result drops straight into the seq-512 BLiMP-per-
active-param table next to softmax/reciprocal/semiring-attn.

Usage:
    python -m research.tools.train_synth_op_real \
        --template reciprocal_semiring_attention_block \
        --dim 512 --layers 4 --seq-len 512 --batch 16 --steps 12000 \
        --run-label recip_semiring_dim512x4_100M
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


def _build(template: str, dim: int, layers: int, seq_len: int, seed: int):
    rng = random.Random(seed)
    graphs = []
    for _ in range(layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        g.set_output(apply_template(g, inp, rng, template_name=template))
        graphs.append(g)
    return compile_model(graphs, vocab_size=VOCAB, max_seq_len=seq_len)


def _active(model) -> int:
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embed" not in n.lower() and n not in ("norm.weight", "norm.bias")
    )


def _loader_ns(args, steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        hydra_root=Path("HYDRA"),
        tokenizer="gpt2",
        vocab_size=VOCAB,
        batch=args.batch,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        prefetch_factor=2,
        steps=steps,
        require_sources=True,
    )


@torch.no_grad()
def _val_ppl(model, val_loader, n_batches: int, device: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        ids, labels = T._prepare_batch(val_loader.get_batch(), vocab_size=VOCAB)
        logits = model(ids.to(device))
        tot += next_token_cross_entropy(logits, labels.to(device), VOCAB).item()
        n += 1
    model.train()
    return math.exp(tot / max(n, 1))


def _blimp(model, device: str, n: int):
    try:
        r = evaluate_blimp(model, vocab_size=VOCAB, device=device, n_per_subtask=n)
        return r.overall_accuracy
    except Exception as e:  # noqa: BLE001 - log, never silently swallow
        return f"ERR:{e}"


def _lr_at(step: int, args) -> float:
    if step <= args.warmup:
        return args.lr * step / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (
        args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    )


def _save_ckpt(model, args, step: int, active: int):
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = args.ckpt_dir / f"{args.run_label}_step{step:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": step,
            "template": args.template,
            "dim": args.dim,
            "n_blocks": args.layers,
            "seq_len": args.seq_len,
            "vocab_size": VOCAB,
            "active_params": active,
        },
        path,
    )
    return path


def _log(args, row: dict):
    with args.jsonl.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _parse_and_init(
    args: "argparse.Namespace",
    device: str,
) -> tuple[object, int, int, int]:
    """Seed RNG, build model, print/log start banner.

    Returns (model, active_params, total_params, tokens_total).
    """
    torch.manual_seed(args.seed)
    model = _build(args.template, args.dim, args.layers, args.seq_len, args.seed).to(
        device
    )
    active = _active(model)
    total = sum(p.numel() for p in model.parameters())
    tokens_total = args.steps * args.batch * args.seq_len
    print(
        f"{args.run_label}: {args.template} dim{args.dim}x{args.layers} "
        f"active={active / 1e6:.1f}M total={total / 1e6:.1f}M | "
        f"{args.steps} steps x {args.batch} x {args.seq_len} = {tokens_total / 1e6:.0f}M tok "
        f"({tokens_total / active:.1f} tok/actP) | device={device}",
        flush=True,
    )
    _log(
        args,
        {
            "event": "start",
            "run_label": args.run_label,
            "template": args.template,
            "dim": args.dim,
            "layers": args.layers,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "steps": args.steps,
            "active_params": active,
            "total_params": total,
            "tokens_total": tokens_total,
            "lr": args.lr,
        },
    )
    return model, active, total, tokens_total


def _run_loop(
    model: object,
    opt: torch.optim.Optimizer,
    train_loader: object,
    val_loader: object,
    args: "argparse.Namespace",
    device: str,
    active: int,
    started: float,
) -> tuple[list[float], int, float, object]:
    """Run the training for-loop; return (losses, skipped, last_ppl, last_bl)."""
    losses: list[float] = []
    skipped = 0
    last_ppl, last_bl = float("nan"), None
    for step in range(1, args.steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = _lr_at(step, args)
        ids, labels = T._prepare_batch(train_loader.get_batch(), vocab_size=VOCAB)
        logits = model(ids.to(device))
        loss = next_token_cross_entropy(logits, labels.to(device), VOCAB)
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(float(gnorm))):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            recent = sum(losses[-100:]) / max(1, len(losses[-100:]))
            tps = step * args.batch * args.seq_len / (time.perf_counter() - started)
            print(
                f"  step {step}/{args.steps} loss {recent:.3f} lr {_lr_at(step, args):.2e} "
                f"{tps:.0f} tok/s skip={skipped}",
                flush=True,
            )
            _log(
                args,
                {
                    "event": "step",
                    "step": step,
                    "loss": round(recent, 4),
                    "lr": _lr_at(step, args),
                    "tok_per_s": round(tps),
                    "skipped": skipped,
                },
            )
        if step % args.eval_every == 0 or step == args.steps:
            last_ppl = _val_ppl(model, val_loader, args.val_batches, device)
            n = args.blimp_n_final if step == args.steps else args.blimp_n
            last_bl = _blimp(model, device, n)
            print(
                f"  [eval @ {step}] val_ppl {last_ppl:.2f} BLiMP(n{n}) {last_bl}",
                flush=True,
            )
            _log(
                args,
                {
                    "event": "eval",
                    "step": step,
                    "val_ppl": round(last_ppl, 3),
                    "blimp": last_bl,
                    "blimp_n": n,
                },
            )
        if step % args.ckpt_every == 0 or step == args.steps:
            p = _save_ckpt(model, args, step, active)
            print(f"  saved {p}", flush=True)
    return losses, skipped, last_ppl, last_bl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="reciprocal_semiring_attention_block")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--ckpt-every", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--blimp-n", type=int, default=200)
    ap.add_argument("--blimp-n-final", type=int, default=1000)
    ap.add_argument("--run-label", default="recip_semiring_dim512x4_100M")
    ap.add_argument(
        "--ckpt-dir", type=Path, default=Path("research/reports/synth_op_ckpts")
    )
    ap.add_argument("--jsonl", type=Path, default=None)
    args = ap.parse_args()
    if args.jsonl is None:
        args.jsonl = Path(f"research/reports/{args.run_label}.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, active, _total, _tokens_total = _parse_and_init(args, device)
    train_loader = T._make_loader(
        _loader_ns(args, args.steps), dataset=T.LOCAL_MIX_NAME, seed=args.seed
    )
    val_loader = T._make_loader(
        _loader_ns(args, args.val_batches + 1),
        dataset=T.LOCAL_MIX_NAME,
        seed=args.seed + 1009,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.perf_counter()
    _losses, skipped, last_ppl, last_bl = _run_loop(
        model, opt, train_loader, val_loader, args, device, active, started
    )
    elapsed = round(time.perf_counter() - started, 1)
    train_loader.close()
    val_loader.close()
    _log(
        args,
        {
            "event": "done",
            "run_label": args.run_label,
            "steps": args.steps,
            "elapsed_sec": elapsed,
            "skipped": skipped,
            "final_val_ppl": round(last_ppl, 3),
            "final_blimp": last_bl,
            "active_params": active,
        },
    )
    print(
        f"\nDONE {args.run_label}: active={active / 1e6:.1f}M final_val_ppl={last_ppl:.2f} "
        f"final_BLiMP={last_bl} skipped={skipped} elapsed={elapsed}s"
    )


if __name__ == "__main__":
    main()
