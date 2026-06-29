"""Can a next-tier model (good loss + real induction) predict 2-5 tokens out?

The pure loss monsters floored at +2/+3 because they're local (no induction). This probes the
NEXT tier — runs.db models with decent loss_ratio AND genuine induction (intermediate AUC
0.4-0.98, mostly conv/SSM hybrids) — to see whether their multi-token heads (+2..+5) rise
above the floor where the monsters' collapsed. Goal: find a quick learner for multi-token-out.

Rebuilds each candidate fresh from graph_json (checkpoints are gone), trains Medusa-style
multi-token heads (detached -> +1 loss ratio preserved), and reports DIRECT top-1 per offset.
Compare to the pure monster baseline (recursive_depth_router MTP: +1 .126 +2 .070 +3 .052
+4 .046 +5 .042). Reuses loss_monster_multitoken machinery; train-mode throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.defaults import VOCAB_SIZE, MAX_SEQ_LEN, N_LAYERS
from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import _CORPUS_TRAIN, _CORPUS_VAL, _OUT_DIR
from research.tools.loss_monster_horizon import _graph_json_for
from research.tools.loss_monster_multitoken import MultiTokenModel, _batch, _eval, _loss

_MONSTER_BASELINE = {"1": 0.126, "2": 0.070, "3": 0.052, "4": 0.046, "5": 0.042}


def _build(
    rid: str, n_layers: int, offsets: tuple[int, ...], seq: int, device: str, seed: int
) -> MultiTokenModel:
    torch.manual_seed(seed)
    graph = graph_from_json(_graph_json_for(rid))
    base = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    )
    return (
        MultiTokenModel(base, offsets, recursions=1, detach_aux=True).to(device).train()
    )


def probe(
    rid: str, label: str, train: np.ndarray, val: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    offsets = tuple(args.offsets)
    max_off = max(offsets)
    model = _build(rid, args.n_layers, offsets, args.seq, args.device, args.seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    gen = np.random.default_rng(1234)
    for _ in range(args.steps):
        chunk = _batch(train, args.batch, args.seq, max_off, gen, args.device)
        total, _ = _loss(model, chunk, args.seq)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
    acc = _eval(
        model,
        val,
        seq=args.seq,
        batch=args.batch,
        n_batches=args.eval_batches,
        max_off=max_off,
        device=args.device,
    )
    rec = {
        "label": label,
        "rid": rid,
        "direct_topk_acc": {str(o): round(acc[o], 4) for o in offsets},
    }
    a = rec["direct_topk_acc"]
    print(
        f"  {label:30s} " + " ".join(f"+{o}={a[str(o)]:.3f}" for o in offsets),
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # default: highest-induction + best-loss next-tier candidates (rid prefix : label)
    ap.add_argument(
        "--candidates",
        nargs="*",
        default=[
            "2001d241:adaptive_conv_ffn(ind.975)",
            "26970faf:local_attn_ssm_hybrid(ind.967)",
            "510f6060:local_attn_ssm_hybrid(loss.498)",
            "3018540c:latent_attn_ssm_hybrid(ind.561)",
        ],
    )
    ap.add_argument("--offsets", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "multitoken_carrier_probe.json"))
    args = ap.parse_args()

    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    b = _MONSTER_BASELINE
    print(
        "MONSTER baseline (pure local): "
        + " ".join(f"+{o}={b[o]}" for o in ("1", "2", "3", "4", "5"))
    )
    print(f"next-tier candidates (detached MTP heads, {args.steps} steps):", flush=True)

    results = []
    for spec in args.candidates:
        rid, _, label = spec.partition(":")
        try:
            results.append(probe(rid, label or rid, train, val, args))
        except Exception as exc:  # loud per-candidate
            print(f"  {label or rid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append({"label": label or rid, "rid": rid, "error": str(exc)})

    Path(args.out).write_text(
        json.dumps(
            {"config": vars(args), "monster_baseline": b, "results": results}, indent=2
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
