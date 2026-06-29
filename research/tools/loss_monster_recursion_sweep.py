"""Weight-tied recursion sweep on loss monsters: R = 1,2,3,4 recursive passes.

Question (user): "what happens if they all do 1,2,3,4 recursions?" — an optimization that
hopefully keeps speed + loss ratio while buying something (longer prediction horizon?).

Each model's layer stack is re-run R times with the SAME weights (tied recursion: no extra
params, R× the layer compute). For each (family, R) we train fresh on FineFineWeb and report:
- speed:      training throughput (tokens/sec) — the cost of recursion
- loss ratio: final val loss + next-token top-1 (quality)
- horizon:    free-rollout exact-match at +1..+4 (does recursion predict further out?)

R=1 reproduces the original monster. Train-mode throughout (eval-mode breaks halt graphs).
Reuses W0 + horizon machinery; no harness edits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
    _sample_batch,
    evaluate,
    select_family_champions,
    _RUNS_DB,
)
from research.tools.loss_monster_horizon import rollout_horizon


class RecursedModel(torch.nn.Module):
    """Wrap a SynthesizedModel and apply its layer stack R times with tied weights."""

    def __init__(self, base: torch.nn.Module, recursions: int) -> None:
        super().__init__()
        self.base = base
        self.recursions = int(recursions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = b.embed(input_ids)
        for _ in range(self.recursions):
            for i, layer in enumerate(b.layers):
                if b.layer_needs_residual[i]:
                    out = layer(x)
                    x = x + out if out.shape == x.shape else out
                else:
                    x = layer(x)
        return b.lm_head(b.norm(x))


def _build(
    graph_json: str, n_layers: int, recursions: int, seq: int, device: str, seed: int
) -> torch.nn.Module:
    torch.manual_seed(seed)
    graph = graph_from_json(graph_json)
    base = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    )
    model = RecursedModel(base, recursions).to(device)
    model.train()
    return model


def _train_timed(
    model: torch.nn.Module,
    train: np.ndarray,
    *,
    seq: int,
    batch: int,
    steps: int,
    lr: float,
    device: str,
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = np.random.default_rng(1234)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        x, y = _sample_batch(train, batch, seq, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return (batch * seq * steps) / max(elapsed, 1e-6)  # tokens/sec


def sweep_family(
    family: str,
    graph_json: str,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in args.recursions:
        try:
            model = _build(
                graph_json, args.n_layers, r, args.seq, args.device, args.seed
            )
            tok_s = _train_timed(
                model,
                train,
                seq=args.seq,
                batch=args.batch,
                steps=args.steps,
                lr=args.lr,
                device=args.device,
            )
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
                horizon=5,
                n_contexts=args.horizon_contexts,
                batch=args.batch,
                device=args.device,
            )
        except Exception as exc:  # loud per-cell
            print(f"  R={r} FAILED: {type(exc).__name__}: {exc}", flush=True)
            out.append({"family": family, "R": r, "error": str(exc)})
            continue
        rec = {
            "family": family,
            "R": r,
            "tok_per_s": round(tok_s, 0),
            "val_loss": round(ev["val_loss"], 4),
            "top1": round(ev["top1_acc"], 4),
            "horizon": [round(h, 4) for h in hor],
        }
        out.append(rec)
        print(
            f"  R={r}: {tok_s:7.0f} tok/s  val_loss={ev['val_loss']:.4f} "
            f"top1={ev['top1_acc']:.4f}  horizon={['%.3f' % h for h in hor]}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--recursions", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--horizon-contexts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-families", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out", default=str(_OUT_DIR / "loss_monster_recursion_sweep.json")
    )
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
    print(
        f"Recursion sweep R={args.recursions} on {len(champs)} families, "
        f"{args.steps} steps each, device={args.device}\n"
    )

    results: list[dict[str, Any]] = []
    for champ in champs:
        print(
            f"=== {champ.family} (loss_ratio={champ.screening_loss_ratio:.3f}) ===",
            flush=True,
        )
        results.extend(sweep_family(champ.family, champ.graph_json, train, val, args))

    out = Path(args.out)
    out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
