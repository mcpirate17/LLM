"""Recursive-depth residual block: residual_block looped R times (tied weights), on nano.

residual_block was the best multi-token learner on the structured nano corpus (predicts
above floor to +7). recursive_depth_router's "recursion" is actually a per-token soft mixture
over 4 parallel depth-projections (depth_weighted_proj, max_depth=4) — NOT a literal loop.
This tests LITERAL weight-tied recursion (R passes of the residual_block stack, R up to 7 as
in the hyper_mor runs): does looping the best block deepen multi-token look-ahead?

Trains each R to 20K steps, evaluates multi-token horizon (+1..+7) at the 10K checkpoint AND
at 20K, on nano_corpus_v4. Reports which recursion depth predicts furthest. Reuses the tied
RecursedModel wrapper + nano loader; train-mode throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _OUT_DIR,
    _RUNS_DB,
    _sample_batch,
    evaluate,
    select_family_champions,
)
from research.tools.loss_monster_horizon import rollout_horizon
from research.tools.loss_monster_recursion_sweep import RecursedModel
from research.tools.nano_multitoken import load_nano_stream, unigram_top1


def _eval(model, val, args) -> dict:
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
        ctx_len=args.ctx_len,
        horizon=7,
        n_contexts=args.n_contexts,
        batch=args.batch,
        device=args.device,
    )
    return {
        "top1": round(ev["top1_acc"], 3),
        "ppl": round(ev["val_ppl"], 2),
        "horizon": [round(h, 3) for h in hor],
    }


def run_recursion(graph_json: str, r: int, train, val, args) -> dict:
    graph = graph_from_json(graph_json)
    base = compile_model_native_first(
        [graph] * args.n_layers, vocab_size=args.vocab, max_seq_len=args.seq
    )
    model = RecursedModel(base, r).to(args.device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(0)
    checkpoints = {}
    for step in range(1, args.steps + 1):
        x, y = _sample_batch(train, args.batch, args.seq, gen, args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in args.checkpoints:
            checkpoints[str(step)] = _eval(model, val, args)
            c = checkpoints[str(step)]
            print(
                f"  R={r} @ {step:5d}: top1={c['top1']:.3f}  +1..+7 "
                + " ".join(f"{h:.2f}" for h in c["horizon"]),
                flush=True,
            )
    return {"recursions": r, "checkpoints": checkpoints}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="residual_block")
    ap.add_argument("--recursions", type=int, nargs="*", default=[1, 2, 4, 7])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[10000, 20000])
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ctx-len", type=int, default=32)
    ap.add_argument("--n-contexts", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "recursive_residual_nano.json"))
    args = ap.parse_args()

    train, val, vocab = load_nano_stream()
    args.vocab = vocab
    floor = unigram_top1(np.concatenate([train, val]), vocab)
    print(
        f"{args.family} + tied recursion R={args.recursions} on nano (vocab={vocab}, floor={floor:.3f})"
    )
    print(f"checkpoints={args.checkpoints} (R=1 = the plain residual_block baseline)\n")

    champ = next(c for c in select_family_champions(_RUNS_DB, families=(args.family,)))
    results = [
        run_recursion(champ.graph_json, r, train, val, args) for r in args.recursions
    ]

    Path(args.out).write_text(
        json.dumps(
            {
                "config": {k: v for k, v in vars(args).items() if k != "vocab"},
                "vocab": vocab,
                "unigram_floor": floor,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nfloor={floor:.3f} | Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
