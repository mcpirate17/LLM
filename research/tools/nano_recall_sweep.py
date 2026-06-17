"""In-context associative-recall capacity sweep (MQAR) — the real failure point.

Single-slot weight memorization never fails (it's a lookup table: a 113K model held
5120 noun->adj pairs at 100%). The capability that DOES degrade with load is
IN-CONTEXT recall: K key->value pairs are placed in the prompt (fresh per sequence,
so nothing is memorized in weights), one key is queried, and the model must retrieve
its value. Recall drops as K grows — this is the softmax-vs-SSM discriminator and the
project's binding wall. We sweep K and report the crossover where recall < threshold.

Sequence: [k1 v1 k2 v2 ... kK vK kq] -> predict v(q). Loss only on the final position.
Keys/values are disjoint token ranges; pairings are random each sequence (true
in-context binding). Chance = 1/n_values.

    python -m research.tools.nano_recall_sweep --dims 64 128 --loads 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from research.tools.nano_softmax_lab import _build

_N_KEYS = 128
_N_VALUES = 32  # chance recall = 1/32 ~ 0.031


def _batch(k_pairs: int, batch: int, device, gen):
    """[B, 2K+1] sequences + [B] targets (value bound to the queried key)."""
    key_off, val_off = 2, 2 + _N_KEYS  # 0,1 reserved
    seqs, tgts = [], []
    for _ in range(batch):
        keys = torch.randperm(_N_KEYS, generator=gen)[:k_pairs].tolist()
        vals = [int(torch.randint(0, _N_VALUES, (1,), generator=gen)) for _ in keys]
        seq = []
        for k, v in zip(keys, vals):
            seq += [key_off + k, val_off + v]
        qi = int(torch.randint(0, k_pairs, (1,), generator=gen))
        seq.append(key_off + keys[qi])  # query key
        seqs.append(seq)
        tgts.append(val_off + vals[qi])
    return (
        torch.tensor(seqs, device=device),
        torch.tensor(tgts, device=device),
    )


def _train_and_recall(
    dim,
    n_blocks,
    k_pairs,
    *,
    device,
    seed,
    steps,
    lr,
    batch_size,
    eval_batch,
    lane_factory=None,
) -> dict:
    vocab = 2 + _N_KEYS + _N_VALUES
    seq_len = 2 * k_pairs + 1
    torch.manual_seed(seed)
    model = _build(dim, n_blocks, vocab, seq_len, device, lane_factory=lane_factory)
    params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_loss = float("nan")
    model.train()
    for _ in range(steps):
        x, y = _batch(k_pairs, batch_size, device, gen)
        logits = model(x)[:, -1, :]  # predict at the query position
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        x, y = _batch(k_pairs, eval_batch, device, gen)
        pred = model(x)[:, -1, :].argmax(-1)
        recall = float((pred == y).float().mean())
    return {
        "k": k_pairs,
        "params": params,
        "recall": recall,
        "final_loss": round(final_loss, 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", nargs="*", type=int, default=[64, 128])
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--loads", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    chance = round(1 / _N_VALUES, 3)
    for dim in args.dims:
        print(
            f"\n=== dim={dim} multi-head softmax (chance={chance}, thr={args.threshold}) ==="
        )
        print(f"{'K_pairs':>7} {'params':>9} {'recall':>7} {'loss':>6}")
        cap = None
        for k in sorted(args.loads):
            t0 = time.time()
            runs = [
                _train_and_recall(
                    dim,
                    args.n_blocks,
                    k,
                    device=args.device,
                    seed=s,
                    steps=args.steps,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    eval_batch=args.eval_batch,
                )
                for s in range(args.seeds)
            ]
            rec = sum(r["recall"] for r in runs) / len(runs)
            loss = sum(r["final_loss"] for r in runs) / len(runs)
            mark = "" if rec >= args.threshold else "  <-- below threshold"
            print(
                f"{k:>7} {runs[0]['params']:>9,} {rec:>7.3f} {loss:>6.2f}"
                f"  ({round(time.time() - t0)}s){mark}"
            )
            if rec >= args.threshold:
                cap = k
        print(
            f"in-context recall capacity @ dim{dim}: largest K with recall>={args.threshold} = {cap}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
