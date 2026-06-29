"""How many tokens *out* does a loss monster predict? Autoregressive rollout horizon.

W0 measured horizon-1 only (top-1/top-2 = candidates for the *immediate next* token). This
measures multi-token-ahead prediction: from a real FineFineWeb context, greedily roll the
model forward and check exact-match accuracy of the predicted token at each future offset
h = 1..K (free rollout — the model eats its own predictions, so error compounds, which is
exactly "how far out can it predict"). Reports the per-horizon decay vs the unigram floor.

Reloads the W0 checkpoints (`research/reports/loss_monsters/<family>.pt`), rebuilding each
model from its `graph_json` and loading the trained weights. Train-mode (eval-mode breaks
halt graphs — see W0 note).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.defaults import VOCAB_SIZE
from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _CORPUS_VAL,
    _RUNS_DB,
    _OUT_DIR,
    unigram_floor,
)


def _graph_json_for(result_id: str) -> str:
    conn = sqlite3.connect(f"file:{_RUNS_DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT graph_json FROM program_results WHERE result_id = ?", (result_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise ValueError(f"no graph_json for {result_id}")
    return str(row[0])


def _reload(ckpt_path: Path, seq: int, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    graph = graph_from_json(_graph_json_for(ckpt["result_id"]))
    model = compile_model_native_first(
        [graph] * int(ckpt["n_layers"]), vocab_size=VOCAB_SIZE, max_seq_len=seq
    )
    model.to(device).train()
    # Some ops (e.g. tropical router centroids) register buffers lazily on first
    # forward, so the freshly-compiled model lacks those keys until it runs once.
    with torch.no_grad():
        dummy = torch.zeros(  # pyright: ignore[reportPrivateImportUsage]
            1,
            min(8, seq),
            dtype=torch.long,
            device=device,  # pyright: ignore[reportPrivateImportUsage]
        )
        model(dummy)
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def rollout_horizon(
    model: torch.nn.Module,
    val: np.ndarray,
    *,
    ctx_len: int,
    horizon: int,
    n_contexts: int,
    batch: int,
    device: str,
    seed: int = 0,
) -> list[float]:
    """Greedy free rollout. Returns per-offset exact-match accuracy [acc@+1, .., acc@+K]."""
    gen = np.random.default_rng(seed)
    hits = np.zeros(horizon, dtype=np.float64)
    total = 0
    done = 0
    while done < n_contexts:
        b = min(batch, n_contexts - done)
        hi = val.shape[0] - ctx_len - horizon - 1
        starts = gen.integers(0, hi, size=b)
        ctx_idx = starts[:, None] + np.arange(ctx_len)[None, :]
        true_idx = starts[:, None] + ctx_len + np.arange(horizon)[None, :]
        cur = torch.as_tensor(  # pyright: ignore[reportPrivateImportUsage]
            np.ascontiguousarray(val[ctx_idx]),
            dtype=torch.int64,  # pyright: ignore[reportPrivateImportUsage]
            device=device,
        )
        truth = np.ascontiguousarray(val[true_idx])
        for h in range(horizon):
            logits = model(cur)
            nxt = logits[:, -1, :].argmax(dim=-1)  # [b]
            hits[h] += float((nxt.cpu().numpy() == truth[:, h]).sum())
            cur = torch.cat([cur, nxt[:, None]], dim=1)  # pyright: ignore[reportPrivateImportUsage]
        total += b
        done += b
    return [float(h / total) for h in hits]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families", nargs="*", default=None, help="default: all saved ckpts"
    )
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--n-contexts", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_horizon.json"))
    args = ap.parse_args()

    ckpts = sorted(_OUT_DIR.glob("*.pt"))
    if args.families:
        ckpts = [p for p in ckpts if p.stem in set(args.families)]
    if not ckpts:
        print("No checkpoints found in", _OUT_DIR)
        return 1

    val = np.load(_CORPUS_VAL, mmap_mode="r")
    seq = args.ctx_len + args.horizon + 1
    floor = unigram_floor(val[:2_000_000].astype(np.int64), VOCAB_SIZE)["unigram_top1"]
    print(f"unigram floor (any single token) = {floor:.4f}\n")
    print(f"{'family':24s} " + " ".join(f"+{h + 1:<5d}" for h in range(args.horizon)))

    results: list[dict[str, Any]] = []
    for ckpt in ckpts:
        try:
            model = _reload(ckpt, seq, args.device)
            acc = rollout_horizon(
                model,
                val,
                ctx_len=args.ctx_len,
                horizon=args.horizon,
                n_contexts=args.n_contexts,
                batch=args.batch,
                device=args.device,
            )
        except Exception as exc:  # loud per-candidate
            print(f"{ckpt.stem:24s} FAILED: {type(exc).__name__}: {exc}")
            results.append({"family": ckpt.stem, "error": str(exc)})
            continue
        print(f"{ckpt.stem:24s} " + " ".join(f"{a:.3f}" for a in acc), flush=True)
        results.append({"family": ckpt.stem, "horizon_acc": acc})

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
