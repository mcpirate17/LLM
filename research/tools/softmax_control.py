"""Softmax-attention POSITIVE CONTROL: is multi-token look-ahead even reachable at this scale?

Every non-softmax model tested (local monsters, induction-probe-positive conv/graph, RWKV+conv
606d) floors at +2..+5 on real text. This is the diagnostic: a standard causal softmax
transformer — which provably CAN form induction heads — at the SAME dim/budget/measurement.

- If softmax predicts +2..+5 well above floor -> the floor is a MODEL limitation (only
  attention escapes it here); non-softmax mechanisms genuinely lack the look-ahead mechanism.
- If softmax ALSO floors -> it's a SCALE/BUDGET artifact; nothing predicts multi-token at
  dim-256/nano, and the question only opens up at real scale.

Mission note: this is a tiny positive-control PROBE (allowed), not a baseline-to-beat.
Reuses W0 eval helpers; rollout horizon measured at ctx_len 128 (< pos table).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE, MAX_SEQ_LEN
from research.tools.loss_monster_screen import (
    _CORPUS_TRAIN,
    _CORPUS_VAL,
    _OUT_DIR,
    _sample_batch,
    evaluate,
)
from research.tools.loss_monster_horizon import rollout_horizon

_TRIGRAM_TOP1 = 0.139


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(
        self, vocab: int, dim: int, n_layers: int, heads: int, max_seq: int
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ln = ids.shape[1]
        pos = torch.arange(ln, device=ids.device)  # pyright: ignore[reportPrivateImportUsage]
        x = self.tok(ids) + self.pos(pos)[None]
        mask = torch.triu(  # pyright: ignore[reportPrivateImportUsage]
            torch.ones(ln, ln, device=ids.device, dtype=torch.bool),  # pyright: ignore[reportPrivateImportUsage]
            diagonal=1,
        )
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lnf(x) @ self.tok.weight.t()


def _train_with_curve(model, opt, train, val, gen, args) -> list[dict]:
    t0 = time.time()
    curve = []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
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
                ctx_len=128,
                horizon=5,
                n_contexts=256,
                batch=args.batch,
                device=args.device,
            )
            curve.append(
                {
                    "step": step,
                    "ppl": ev["val_ppl"],
                    "top1": ev["top1_acc"],
                    "horizon": hor,
                }
            )
            print(
                f"{step:6d} {ev['val_ppl']:8.1f} {ev['top1_acc']:7.4f} "
                f"{hor[1]:6.3f} {hor[2]:6.3f} {hor[3]:6.3f} {hor[4]:6.3f} {time.time() - t0:7.0f}s",
                flush=True,
            )
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "softmax_control.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TinyGPT(VOCAB_SIZE, args.dim, args.layers, args.heads, args.seq).to(
        args.device
    )
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(1234)

    print(
        f"SOFTMAX control: dim{args.dim} L{args.layers} H{args.heads}  params={n_params / 1e6:.1f}M  "
        f"trigram +1 bar={_TRIGRAM_TOP1}  (non-softmax floored at +2~.06 +3~.05)",
        flush=True,
    )
    print(
        f"{'step':>6} {'ppl':>8} {'top1':>7} {'+2':>6} {'+3':>6} {'+4':>6} {'+5':>6} {'elapsed':>8}",
        flush=True,
    )

    curve = _train_with_curve(model, opt, train, val, gen, args)
    f = curve[-1]
    print(
        f"\nfinal: top1 {f['top1']:.4f}  horizon +2={f['horizon'][1]:.3f} +3={f['horizon'][2]:.3f} "
        f"+4={f['horizon'][3]:.3f} +5={f['horizon'][4]:.3f}  (floor ~0.035)",
        flush=True,
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "params_m": round(n_params / 1e6, 2),
                "curve": curve,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
