"""Optimize how many tokens *out* a loss monster predicts (+1..+5), keeping the loss ratio.

Goal (user): push direct prediction of the 2nd/3rd/4th/5th token ahead WITHOUT hurting the
+1 next-token loss ratio. Induction is explicitly NOT a goal here.

Approach = multi-token-prediction heads (Medusa / Gloeckle-style), which is also a known
*inference speedup* (speculative decoding), so it fits "don't lose speed":

- The monster's original +1 path (body -> norm -> tied lm_head) is kept EXACTLY and trained
  only by the +1 loss => the loss ratio is preserved by construction.
- Cheap auxiliary heads for offsets 2..K read the body's representation (DETACHED by default,
  so they cannot perturb the body / +1 loss) through a small per-offset dim->dim transform,
  then the SHARED tied unembed. ~65K params/head — negligible vs the 28M model.
- We measure DIRECT (teacher-forced, non-compounding) top-1 at each offset: "given the true
  context, can a head name the token k steps ahead?" This is the speculative-prediction
  capability and the basis for the speedup — distinct from compounding free rollout.

Optional ``--recursions R`` runs the body tied R times first (richer representation for the
heads to read further from). Reuses W0 machinery; train-mode (eval-mode breaks halt graphs).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
    select_family_champions,
    unigram_floor,
)


class MultiTokenModel(nn.Module):
    """Monster body + tied +1 head (untouched) + cheap detached heads for offsets 2..K."""

    def __init__(
        self,
        base: nn.Module,
        offsets: tuple[int, ...],
        recursions: int,
        detach_aux: bool,
    ) -> None:
        super().__init__()
        self.base = base
        self.offsets = offsets
        self.recursions = int(recursions)
        self.detach_aux = detach_aux
        emb = base.embed
        assert isinstance(emb, nn.Embedding), f"base.embed is {type(emb)}"
        dim = emb.weight.shape[1]
        # one dim->dim transform per aux offset (offset 1 reuses the original path)
        self.aux = nn.ModuleList([nn.Linear(dim, dim) for _ in offsets if _ != 1])
        for lin in (
            self.aux
        ):  # near-identity init: aux heads start ~ the +1 head, then specialize
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _hidden(self, ids: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = b.embed(ids)
        for _ in range(self.recursions):
            for i, layer in enumerate(b.layers):
                if b.layer_needs_residual[i]:
                    out = layer(x)
                    x = x + out if out.shape == x.shape else out
                else:
                    x = layer(x)
        return x

    def forward(self, ids: torch.Tensor) -> dict[int, torch.Tensor]:
        b = self.base
        h = self._hidden(ids)
        logits: dict[int, torch.Tensor] = {1: b.lm_head(b.norm(h))}
        h_aux = h.detach() if self.detach_aux else h
        ai = 0
        for off in self.offsets:
            if off == 1:
                continue
            logits[off] = b.lm_head(b.norm(self.aux[ai](h_aux)))
            ai += 1
        return logits


def _build(
    graph_json: str,
    n_layers: int,
    offsets: tuple[int, ...],
    recursions: int,
    detach_aux: bool,
    seq: int,
    device: str,
    seed: int,
) -> MultiTokenModel:
    torch.manual_seed(seed)
    graph = graph_from_json(graph_json)
    base = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    )
    return MultiTokenModel(base, offsets, recursions, detach_aux).to(device).train()


def _batch(
    tokens: np.ndarray,
    batch: int,
    seq: int,
    max_off: int,
    gen: np.random.Generator,
    device: str,
):
    hi = tokens.shape[0] - seq - max_off - 1
    starts = gen.integers(0, hi, size=batch)
    win = starts[:, None] + np.arange(seq + max_off)[None, :]
    chunk = torch.as_tensor(  # pyright: ignore[reportPrivateImportUsage]
        np.ascontiguousarray(tokens[win]),
        dtype=torch.int64,  # pyright: ignore[reportPrivateImportUsage]
        device=device,
    )
    return chunk  # [B, seq+max_off]; input=[:, :seq], target@off = [:, off:off+seq]


def _loss(
    model: MultiTokenModel, chunk: torch.Tensor, seq: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x = chunk[:, :seq]
    logits = model(x)
    ces: dict[int, torch.Tensor] = {}
    for off, lg in logits.items():
        tgt = chunk[:, off : off + seq]
        ces[off] = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
    total = torch.stack(list(ces.values())).sum()
    return total, ces[1].detach()


@torch.no_grad()
def _eval(
    model: MultiTokenModel,
    val: np.ndarray,
    *,
    seq: int,
    batch: int,
    n_batches: int,
    max_off: int,
    device: str,
) -> dict[int, float]:
    gen = np.random.default_rng(0)
    hits = {off: 0.0 for off in model.offsets}
    tot = 0.0
    for _ in range(n_batches):
        chunk = _batch(val, batch, seq, max_off, gen, device)
        x = chunk[:, :seq]
        logits = model(x)
        for off, lg in logits.items():
            tgt = chunk[:, off : off + seq]
            hits[off] += float((lg.argmax(-1) == tgt).sum())
        tot += x.numel()
    return {off: hits[off] / tot for off in model.offsets}


def sweep(
    graph_json: str,
    family: str,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    offsets = tuple(args.offsets)
    max_off = max(offsets)
    model = _build(
        graph_json,
        args.n_layers,
        offsets,
        args.recursions,
        not args.joint,
        args.seq,
        args.device,
        args.seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    gen = np.random.default_rng(1234)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    last_primary = 0.0
    for _ in range(args.steps):
        chunk = _batch(train, args.batch, args.seq, max_off, gen, args.device)
        total, primary = _loss(model, chunk, args.seq)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        last_primary = float(primary)
    if args.device == "cuda":
        torch.cuda.synchronize()
    tok_s = (args.batch * args.seq * args.steps) / max(time.time() - t0, 1e-6)
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
        "family": family,
        "recursions": args.recursions,
        "joint": args.joint,
        "params_m": round(n_params / 1e6, 2),
        "tok_per_s": round(tok_s, 0),
        "primary_plus1_loss": round(last_primary, 4),
        "direct_topk_acc": {str(o): round(acc[o], 4) for o in offsets},
    }
    accs = " ".join(f"+{o}={acc[o]:.3f}" for o in offsets)
    print(
        f"  {family:24s} R={args.recursions} {tok_s:7.0f}tok/s +1loss={last_primary:.3f}  {accs}",
        flush=True,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--offsets", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--recursions", type=int, default=1)
    ap.add_argument(
        "--joint",
        action="store_true",
        help="let aux heads train the body too (trades loss ratio); default = detached",
    )
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-families", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_multitoken.json"))
    args = ap.parse_args()

    champs = select_family_champions(
        _RUNS_DB, families=tuple(args.families) if args.families else None
    )
    if args.max_families > 0:
        champs = champs[: args.max_families]
    if not champs:
        print("No family champions matched.")
        return 1

    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    floor = unigram_floor(val[:2_000_000].astype(np.int64), VOCAB_SIZE)["unigram_top1"]
    print(
        f"unigram floor = {floor:.4f}  | offsets={args.offsets} recursions={args.recursions} "
        f"mode={'joint' if args.joint else 'detached(loss-ratio-safe)'}\n"
    )

    results = []
    for champ in champs:
        try:
            results.append(sweep(champ.graph_json, champ.family, train, val, args))
        except Exception as exc:  # loud per-family
            print(f"  {champ.family} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"family": champ.family, "error": str(exc)})

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"config": vars(args), "unigram_floor": floor, "results": results}, indent=2
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
