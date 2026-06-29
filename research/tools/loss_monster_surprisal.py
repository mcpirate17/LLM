"""Per-token surprisal scorer: use a loss monster as a cheap difficulty/routing signal.

A loss monster is a smoothed neural trigram (see loss_monster_families note). Its per-token
surprisal (-log p(true_token)) therefore measures how predictable a token is from LOCAL
context. Tokens the monster CAN'T predict (high surprisal) are exactly the ones that need
long-range structure — i.e. where a carrier (induction/recall) must do the work. This makes
a monster a near-free router/curriculum signal for the data pipeline (Workstream D):

- low surprisal  -> locally predictable -> cheap monster lane can handle it
- high surprisal -> needs the long-range carrier (route / up-weight)

Outputs: surprisal distribution (bits/token, percentiles), a routing table (route the top
X% most-surprising tokens to the carrier -> what fraction of total loss that covers),
doc-boundary surprisal (sanity: monster is blind across <|endoftext|>), and decoded
high- vs low-surprisal examples. Pure inference, no GPU training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.tools.loss_monster_screen import _CORPUS_VAL, _OUT_DIR
from research.tools.loss_monster_horizon import _reload

_EOT = 100257  # cl100k_base <|endoftext|>


@torch.no_grad()
def score_tokens(
    model: torch.nn.Module,
    val: np.ndarray,
    *,
    seq: int,
    batch: int,
    n_tokens: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (surprisal_nats, target_ids, prev_is_eot) over ~n_tokens scored positions."""
    gen = np.random.default_rng(0)
    sur, ids, doc0 = [], [], []
    scored = 0
    while scored < n_tokens:
        hi = val.shape[0] - seq - 2
        starts = gen.integers(0, hi, size=batch)
        idx = starts[:, None] + np.arange(seq + 1)[None, :]
        chunk = torch.as_tensor(  # pyright: ignore[reportPrivateImportUsage]
            np.ascontiguousarray(val[idx]),
            dtype=torch.int64,  # pyright: ignore[reportPrivateImportUsage]
            device=device,
        )
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits = model(x)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        )
        sur.append(ce.float().cpu().numpy())
        ids.append(y.reshape(-1).cpu().numpy())
        doc0.append(
            (x.reshape(-1) == _EOT).cpu().numpy()
        )  # target whose PREV token is EOT
        scored += y.numel()
    return np.concatenate(sur), np.concatenate(ids), np.concatenate(doc0)


def routing_table(sur: np.ndarray, fracs: tuple[float, ...]) -> list[dict[str, float]]:
    """If we route the top-X% most-surprising tokens to the carrier, what % of total loss?"""
    order = np.sort(sur)[::-1]
    total = float(order.sum())
    out = []
    for f in fracs:
        k = max(1, int(len(order) * f))
        out.append(
            {
                "route_top_frac_tokens": f,
                "covers_frac_of_total_loss": round(float(order[:k].sum()) / total, 4),
                "surprisal_threshold_bits": round(float(order[k - 1]) / math.log(2), 3),
            }
        )
    return out


def decoded_examples(
    model: torch.nn.Module,
    val: np.ndarray,
    *,
    device: str,
    start: int,
    length: int,
    k: int,
) -> dict[str, Any]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return {"note": "tiktoken unavailable; skipped decoded examples"}
    ctx = val[start : start + length + 1].astype(np.int64)
    with torch.no_grad():
        x = torch.as_tensor(ctx[:-1][None], dtype=torch.int64, device=device)  # pyright: ignore[reportPrivateImportUsage]
        logits = model(x)
        ce = (
            F.cross_entropy(
                logits[0],
                torch.as_tensor(ctx[1:], device=device),  # pyright: ignore[reportPrivateImportUsage]
                reduction="none",
            )
            .float()
            .cpu()
            .numpy()
        )

    def show(pos: int) -> dict[str, Any]:
        lo = max(0, pos - 6)
        return {
            "bits": round(float(ce[pos]) / math.log(2), 2),
            "context": enc.decode([int(t) for t in ctx[lo : pos + 1]]),
            "true_next": enc.decode([int(ctx[pos + 1])]),
        }

    top = np.argsort(ce)[::-1][:k]
    bot = np.argsort(ce)[:k]
    return {
        "highest_surprisal": [show(int(p)) for p in top],
        "lowest_surprisal": [show(int(p)) for p in bot],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="recursive_depth_router")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-tokens", type=int, default=400_000)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_surprisal.json"))
    args = ap.parse_args()

    ckpt = _OUT_DIR / f"{args.family}.pt"
    if not ckpt.exists():
        print(f"Missing {ckpt}")
        return 1
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    model = _reload(ckpt, args.seq + 1, args.device)

    sur, _ids, doc0 = score_tokens(
        model,
        val,
        seq=args.seq,
        batch=args.batch,
        n_tokens=args.n_tokens,
        device=args.device,
    )
    bits = sur / math.log(2)
    pcts = {
        f"p{p}": round(float(np.percentile(bits, p)), 3) for p in (50, 75, 90, 95, 99)
    }
    route = routing_table(sur, (0.05, 0.10, 0.20, 0.30, 0.50))
    doc_mask = doc0.astype(bool)
    ex = decoded_examples(
        model, val, device=args.device, start=10_000, length=512, k=args.examples
    )

    summary = {
        "family": args.family,
        "tokens_scored": int(sur.size),
        "mean_bits_per_token": round(float(bits.mean()), 3),
        "percentiles_bits": pcts,
        "doc_start_mean_bits": round(float(bits[doc_mask].mean()), 3)
        if doc_mask.any()
        else None,
        "non_doc_start_mean_bits": round(float(bits[~doc_mask].mean()), 3),
        "routing_table": route,
    }
    print(json.dumps(summary, indent=2))
    print("\n--- highest-surprisal tokens (need the carrier) ---")
    for e in ex.get("highest_surprisal", []):
        print(f"  {e['bits']:5.1f} bits  ...{e['context']!r} -> {e['true_next']!r}")
    print("--- lowest-surprisal tokens (cheap monster lane handles) ---")
    for e in ex.get("lowest_surprisal", []):
        print(f"  {e['bits']:5.1f} bits  ...{e['context']!r} -> {e['true_next']!r}")

    Path(args.out).write_text(
        json.dumps({"config": vars(args), "summary": summary, "examples": ex}, indent=2)
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
