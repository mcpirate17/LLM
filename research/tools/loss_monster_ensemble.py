"""Combine loss-monster mechanisms by ensembling their next-token distributions.

Zero training. For each val position, average the softmax next-token distributions of all
saved monster checkpoints and measure top-1/top-2 vs each individual model and vs the best
single. Question: did different architecture families learn DIFFERENT local structure
(complementary -> ensemble beats best single) or the SAME (redundant -> no gain)?

Reloads W0 checkpoints (rebuild from graph_json + load weights), train-mode (eval-mode
breaks halt graphs). Probabilities accumulated one model at a time to bound memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE
from research.tools.loss_monster_screen import _CORPUS_VAL, _OUT_DIR, unigram_floor
from research.tools.loss_monster_horizon import _reload


@torch.no_grad()
def evaluate_ensemble(
    models: dict[str, torch.nn.Module],
    val: np.ndarray,
    *,
    seq: int,
    batch: int,
    n_batches: int,
    device: str,
) -> dict[str, Any]:
    gen = np.random.default_rng(0)
    indiv = {name: 0.0 for name in models}
    ens_top1 = ens_top2 = total = 0.0
    for _ in range(n_batches):
        hi = val.shape[0] - seq - 2
        starts = gen.integers(0, hi, size=batch)
        idx = starts[:, None] + np.arange(seq + 1)[None, :]
        chunk = torch.as_tensor(  # pyright: ignore[reportPrivateImportUsage]
            np.ascontiguousarray(val[idx]),
            dtype=torch.int64,  # pyright: ignore[reportPrivateImportUsage]
            device=device,
        )
        x, y = chunk[:, :-1], chunk[:, 1:]
        probsum = torch.zeros(  # pyright: ignore[reportPrivateImportUsage]
            x.shape[0], x.shape[1], VOCAB_SIZE, device=device
        )
        for name, m in models.items():
            logits = m(x)
            probs = F.softmax(logits.float(), dim=-1)
            probsum += probs
            indiv[name] += float((logits.argmax(-1) == y).sum())
            del logits, probs
        top = probsum.topk(2, dim=-1).indices
        c1 = top[..., 0] == y
        ens_top1 += float(c1.sum())
        ens_top2 += float((c1 | (top[..., 1] == y)).sum())
        total += float(y.numel())
        del probsum
    return {
        "ensemble_top1": ens_top1 / total,
        "ensemble_top2": ens_top2 / total,
        "individual_top1": {n: indiv[n] / total for n in models},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_ensemble.json"))
    args = ap.parse_args()

    ckpts = sorted(_OUT_DIR.glob("*.pt"))
    if args.families:
        ckpts = [p for p in ckpts if p.stem in set(args.families)]
    if not ckpts:
        print("No checkpoints in", _OUT_DIR)
        return 1

    val = np.load(_CORPUS_VAL, mmap_mode="r")
    floor = unigram_floor(val[:2_000_000].astype(np.int64), VOCAB_SIZE)["unigram_top1"]
    print(f"Ensembling {len(ckpts)} monsters | floor={floor:.4f}\n")

    models: dict[str, torch.nn.Module] = {}
    for p in ckpts:
        try:
            models[p.stem] = _reload(p, args.seq + 1, args.device)
        except Exception as exc:  # loud skip; one bad ckpt shouldn't kill the ensemble
            print(f"  skip {p.stem}: {type(exc).__name__}: {exc}")
    if not models:
        print("No models loaded.")
        return 1
    res = evaluate_ensemble(
        models,
        val,
        seq=args.seq,
        batch=args.batch,
        n_batches=args.n_batches,
        device=args.device,
    )

    indiv = res["individual_top1"]
    best = max(indiv, key=lambda k: indiv[k])
    print("individual +1 top-1:")
    for n in sorted(indiv, key=lambda k: -indiv[k]):
        print(f"  {n:24s} {indiv[n]:.4f}")
    print(f"\nbest single ({best})   +1 top-1 = {indiv[best]:.4f}")
    print(
        f"ENSEMBLE (all {len(models)})  +1 top-1 = {res['ensemble_top1']:.4f}  "
        f"top-2 = {res['ensemble_top2']:.4f}"
    )
    print(f"ensemble lift over best single = {res['ensemble_top1'] - indiv[best]:+.4f}")

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"config": vars(args), "unigram_floor": floor, "best_single": best, **res},
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
