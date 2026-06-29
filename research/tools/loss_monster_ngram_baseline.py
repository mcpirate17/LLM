"""Are the loss monsters just learned n-grams? Compare to literal bigram/trigram lookup.

The 11 monster families ensemble to ~no gain (redundant) — they seem to learn the same local
structure. This quantifies how much that structure is just the corpus's n-gram statistics:
build a plain bigram (1-token context) and trigram (2-token context) most-frequent-next table
from FineFineWeb train, measure next-token top-1 on val, and compare to the monsters' ~0.153
(full-context) and the 0.035 unigram floor.

If a monster barely beats the trigram, its "mechanism" is essentially an n-gram model — which
would explain why combining mechanisms (ensembling) does nothing. Cheap: pure counting, no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.defaults import VOCAB_SIZE
from research.tools.loss_monster_screen import _CORPUS_TRAIN, _CORPUS_VAL, _OUT_DIR


def build_ngram_table(toks: np.ndarray, order: int, vocab: int) -> dict[int, int]:
    """Map context-code (base-`vocab` of the prev order-1 tokens) -> most frequent next token."""
    t = toks.astype(np.int64)
    ctx = np.zeros(t.shape[0] - order + 1, dtype=np.int64)
    for j in range(order - 1):
        ctx = ctx * vocab + t[j : j + ctx.shape[0]]
    nxt = t[order - 1 :]
    key = ctx * vocab + nxt
    uniq, counts = np.unique(key, return_counts=True)
    u_ctx, u_nxt = uniq // vocab, uniq % vocab
    # for each ctx keep the next with the highest count: sort by (ctx, count) then dedup ctx
    order_idx = np.lexsort((counts, u_ctx))  # ascending; last per ctx = max count
    u_ctx_s, u_nxt_s = u_ctx[order_idx], u_nxt[order_idx]
    last = np.ones(u_ctx_s.shape[0], dtype=bool)
    last[:-1] = u_ctx_s[1:] != u_ctx_s[:-1]
    return {int(c): int(n) for c, n in zip(u_ctx_s[last], u_nxt_s[last])}


def eval_ngram(
    table: dict[int, int],
    toks: np.ndarray,
    order: int,
    vocab: int,
    fallback: int,
    n_eval: int,
) -> float:
    t = toks[: n_eval + order].astype(np.int64)
    ctx = np.zeros(t.shape[0] - order + 1, dtype=np.int64)
    for j in range(order - 1):
        ctx = ctx * vocab + t[j : j + ctx.shape[0]]
    nxt = t[order - 1 :]
    hits = 0
    for c, true in zip(ctx.tolist(), nxt.tolist()):
        if table.get(c, fallback) == true:
            hits += 1
    return hits / len(nxt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--build-tokens", type=int, default=40_000_000)
    ap.add_argument("--eval-tokens", type=int, default=200_000)
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_ngram_baseline.json"))
    args = ap.parse_args()

    train = np.load(_CORPUS_TRAIN, mmap_mode="r")[: args.build_tokens]
    val = np.load(_CORPUS_VAL, mmap_mode="r")
    counts = np.bincount(val[:2_000_000].astype(np.int64), minlength=VOCAB_SIZE)
    fallback = int(counts.argmax())
    floor = float(counts.max() / counts.sum())
    print(
        f"unigram floor (top-1) = {floor:.4f}   monster best single (full ctx) = ~0.153\n"
    )

    results = {"unigram_floor": floor}
    for order in args.orders:
        table = build_ngram_table(np.asarray(train), order, VOCAB_SIZE)
        acc = eval_ngram(
            table, np.asarray(val), order, VOCAB_SIZE, fallback, args.eval_tokens
        )
        results[f"{order}gram_top1"] = round(acc, 4)
        print(
            f"{order}-gram ({order - 1}-token context): top-1 = {acc:.4f}  "
            f"({len(table):,} contexts)"
        )

    Path(args.out).write_text(json.dumps({"config": vars(args), **results}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
