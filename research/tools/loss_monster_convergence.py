"""Are loss monsters actually DECENT next-token predictors, or just trigrams?

The n-gram check showed the best monster (top-1 0.153) barely beats a trigram (0.139) at a
2-3k-step budget. That could mean (a) they're fundamentally trigram-equivalent, or (b) they
were undertrained. This trains one monster to a real budget with periodic eval and live
logging, tracking next-token top-1 / perplexity AND the +2/+3 free-rollout horizon vs the
trigram bar (0.139). If top-1 plateaus near 0.139 -> trigram-equivalent (weak). If it climbs
well past it and +2/+3 lift -> genuinely uses context (decent), and earlier horizon numbers
were a budget artifact.

Reuses W0 helpers; train-mode throughout. Logs every --eval-every steps (visibility).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE, MAX_SEQ_LEN, N_LAYERS
from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _CORPUS_TRAIN,
    _CORPUS_VAL,
    _OUT_DIR,
    _RUNS_DB,
    _sample_batch,
    evaluate,
    select_family_champions,
)
from research.tools.loss_monster_horizon import rollout_horizon

_TRIGRAM_TOP1 = 0.139  # from loss_monster_ngram_baseline (40M train tokens)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="recursive_depth_router")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_convergence.json"))
    return ap


def _eval_row(model, val, step, args, t0):
    ev = evaluate(
        model,
        val,
        batch=args.batch,
        seq=args.seq,
        n_batches=args.eval_batches,
        device=args.device,
    )
    hor = rollout_horizon(
        model,
        val,
        ctx_len=args.seq,
        horizon=3,
        n_contexts=128,
        batch=args.batch,
        device=args.device,
    )
    print(
        f"{step:6d} {ev['val_ppl']:8.1f} {ev['top1_acc']:7.4f} "
        f"{ev['top1_acc'] - _TRIGRAM_TOP1:+7.4f} {hor[1]:6.3f} {hor[2]:6.3f} "
        f"{time.time() - t0:7.0f}s",
        flush=True,
    )
    return {
        "step": step,
        "val_loss": ev["val_loss"],
        "ppl": ev["val_ppl"],
        "top1": ev["top1_acc"],
        "h2": hor[1],
        "h3": hor[2],
    }


def train_with_curve(model, train, val, args) -> list[dict]:
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(1234)
    t0 = time.time()
    curve = []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            curve.append(_eval_row(model, val, step, args, t0))
        if step == args.steps:
            break
        x, y = _sample_batch(train, args.batch, args.seq, gen, args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return curve


def main() -> int:
    args = _build_argparser().parse_args()

    champ = next(c for c in select_family_champions(_RUNS_DB, families=(args.family,)))
    graph = graph_from_json(champ.graph_json)
    model = compile_model_native_first(
        [graph] * args.n_layers, vocab_size=VOCAB_SIZE, max_seq_len=args.seq
    ).to(args.device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")

    print(
        f"family={args.family}  params={n_params / 1e6:.1f}M  steps={args.steps}  "
        f"trigram bar top-1={_TRIGRAM_TOP1}",
        flush=True,
    )
    print(
        f"{'step':>6} {'ppl':>8} {'top1':>7} {'vs_tri':>7} {'+2':>6} {'+3':>6} {'elapsed':>8}",
        flush=True,
    )

    curve = train_with_curve(model, train, val, args)
    final = curve[-1]
    verdict = (
        "DECENT (uses context, well past trigram)"
        if final["top1"] > _TRIGRAM_TOP1 + 0.05
        else "TRIGRAM-EQUIVALENT (plateaus near trigram)"
        if final["top1"] < _TRIGRAM_TOP1 + 0.02
        else "MARGINAL (slightly beats trigram)"
    )
    print(
        f"\nVERDICT: {verdict}  | final top-1 {final['top1']:.4f} vs trigram {_TRIGRAM_TOP1} "
        f"| best ppl {min(r['ppl'] for r in curve):.1f}"
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "params_m": round(n_params / 1e6, 2),
                "trigram_top1": _TRIGRAM_TOP1,
                "verdict": verdict,
                "curve": curve,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}  (log: {Path(args.out).with_suffix('.log')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
