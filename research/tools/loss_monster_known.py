"""Build a loss monster from KNOWN math: a neural n-gram LM (Bengio 2003).

Instead of trusting the rebuilt runs.db monsters, construct a provably-local strong predictor
from well-understood, non-attention math: embed the last k tokens, concat, MLP, predict next
(tied unembed). This is a loss monster BY CONSTRUCTION — it can only see a fixed k-token
window, so it should (a) be a strong +1 predictor (beating the count-based trigram via neural
smoothing + longer context) and (b) collapse past +1 on rollout (n-grams can't look ahead).

Use it to (1) calibrate whether the rebuilt monsters are decent or junk, and (2) serve as a
clean, trustworthy local scaffold if the rebuilds disappoint. Reuses W0 eval helpers; trains
on real FineFineWeb; logs every --eval-every steps.
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


class NeuralNgramLM(nn.Module):
    """Bengio-style feedforward n-gram: concat last k token embeds -> MLP -> tied unembed."""

    def __init__(
        self, vocab: int, dim: int, context_k: int, hidden: int, layers: int
    ) -> None:
        super().__init__()
        self.k = context_k
        self.embed = nn.Embedding(vocab, dim)
        nn.init.normal_(self.embed.weight, std=dim**-0.5)
        mlp: list[nn.Module] = [nn.Linear(context_k * dim, hidden), nn.GELU()]
        for _ in range(layers - 1):
            mlp += [nn.Linear(hidden, hidden), nn.GELU()]
        mlp += [nn.Linear(hidden, dim)]
        self.mlp = nn.Sequential(*mlp)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)  # [B, L, D]
        b, ln, d = emb.shape
        padded = F.pad(emb, (0, 0, self.k - 1, 0))  # left-pad seq by k-1
        windows = torch.stack(
            [padded[:, j : j + ln, :] for j in range(self.k)], dim=2
        )  # [B,L,k,D]
        feat = windows.reshape(b, ln, self.k * d)
        h = self.norm(self.mlp(feat))
        return h @ self.embed.weight.t()  # tied unembed -> [B, L, vocab]


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--context-k", type=int, default=8, help="n-gram order (tokens of context)"
    )
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_known.json"))
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
        "ppl": ev["val_ppl"],
        "top1": ev["top1_acc"],
        "h2": hor[1],
        "h3": hor[2],
    }


def main() -> int:
    args = _build_argparser().parse_args()
    torch.manual_seed(0)
    model = NeuralNgramLM(
        VOCAB_SIZE, args.dim, args.context_k, args.hidden, args.layers
    ).to(args.device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(1234)

    print(
        f"KNOWN-MATH neural {args.context_k}-gram  params={n_params / 1e6:.1f}M  "
        f"trigram bar top-1={_TRIGRAM_TOP1}",
        flush=True,
    )
    print(
        f"{'step':>6} {'ppl':>8} {'top1':>7} {'vs_tri':>7} {'+2':>6} {'+3':>6} {'elapsed':>8}",
        flush=True,
    )

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

    final = curve[-1]
    print(
        f"\nfinal: top-1 {final['top1']:.4f} (trigram {_TRIGRAM_TOP1}, "
        f"rebuilt-monster ~0.153)  best ppl {min(r['ppl'] for r in curve):.1f}  "
        f"+2 {final['h2']:.3f} +3 {final['h3']:.3f}",
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
