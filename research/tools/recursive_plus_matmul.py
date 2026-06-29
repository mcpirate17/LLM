"""Add a token-mixing matmul to recursive_depth_router — does it unlock multi-token look-ahead?

recursive_depth_router is the best loss monster but floors at +2/+3: it's all channel-local
ops with NO token-mixing matmul, so information can't move across positions (no copy = no
look-ahead). This inserts ONE causal content matmul (non-softmax linear-attention-style
mixer: scores = QK^T masked + mean-normalized, out = scores @ V) after the local layers and
asks: does +2..+5 rise above the floor while +1/loss stay good?

Non-softmax by design (no exp/softmax on scores) so it's a raw matmul mixer, not a softmax
twin. Compares: pure recursion (top1 ~0.175, +2/+3 at floor) vs +matmul. Reuses W0 eval.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class CausalMatmulMixer(nn.Module):
    """Non-softmax causal content mixer: scores=QK^T (masked), out = (scores @ V)/count. One matmul-pair."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(
            self.o.weight
        )  # start as identity (residual) so +1/loss isn't disturbed at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, ln, d = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # [B,L,L]
        mask = torch.tril(  # pyright: ignore[reportPrivateImportUsage]
            torch.ones(ln, ln, device=x.device, dtype=torch.bool)  # pyright: ignore[reportPrivateImportUsage]
        )
        scores = scores.masked_fill(~mask, 0.0)  # linear (NO softmax)
        counts = torch.arange(  # pyright: ignore[reportPrivateImportUsage]
            1, ln + 1, device=x.device, dtype=x.dtype
        ).view(1, ln, 1)
        out = (scores @ v) / counts  # causal content-weighted mean
        return x + self.o(out)


class RecursionPlusMatmul(nn.Module):
    def __init__(self, base: nn.Module, recursions: int, with_matmul: bool) -> None:
        super().__init__()
        self.base = base
        self.recursions = recursions
        emb = base.embed
        assert isinstance(emb, nn.Embedding)
        self.mixer = CausalMatmulMixer(emb.weight.shape[1]) if with_matmul else None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = b.embed(ids)
        for _ in range(self.recursions):
            for i, layer in enumerate(b.layers):
                if b.layer_needs_residual[i]:
                    out = layer(x)
                    x = x + out if out.shape == x.shape else out
                else:
                    x = layer(x)
        if self.mixer is not None:
            x = self.mixer(x)
        return b.lm_head(b.norm(x))


def _build(
    graph_json: str, n_layers: int, with_matmul: bool, seq: int, device: str, seed: int
):
    torch.manual_seed(seed)
    graph = graph_from_json(graph_json)
    base = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    )
    return RecursionPlusMatmul(base, 1, with_matmul).to(device).train()


def run_arm(graph_json, label, with_matmul, train, val, args) -> dict:
    model = _build(
        graph_json, args.n_layers, with_matmul, args.seq, args.device, args.seed
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(1234)
    t0 = time.time()
    for _ in range(args.steps):
        x, y = _sample_batch(train, args.batch, args.seq, gen, args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
        horizon=5,
        n_contexts=256,
        batch=args.batch,
        device=args.device,
    )
    rec = {
        "arm": label,
        "top1": round(ev["top1_acc"], 4),
        "ppl": round(ev["val_ppl"], 1),
        "horizon": [round(h, 4) for h in hor],
        "elapsed_s": round(time.time() - t0),
    }
    print(
        f"  {label:18s} top1={ev['top1_acc']:.4f} ppl={ev['val_ppl']:6.1f}  "
        f"horizon +1..+5 {['%.3f' % h for h in hor]}",
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="recursive_depth_router")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "recursive_plus_matmul.json"))
    args = ap.parse_args()

    champ = next(c for c in select_family_champions(_RUNS_DB, families=(args.family,)))
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    print(
        f"{args.family}: baseline (no matmul) vs +causal-matmul mixer, {args.steps} steps\n"
    )

    results = [
        run_arm(champ.graph_json, "baseline", False, train, val, args),
        run_arm(champ.graph_json, "+matmul", True, train, val, args),
    ]
    base, mm = results[0]["horizon"], results[1]["horizon"]
    print(
        "\nhorizon delta (+matmul - baseline): "
        + " ".join(f"+{i + 1}={mm[i] - base[i]:+.3f}" for i in range(5))
    )
    Path(args.out).write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2)
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
