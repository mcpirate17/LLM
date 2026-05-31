"""Multi-axis x multi-lane x multi-seed nano probe over bAbI-for-SFT.

Answers: "which token-mixer LEARNS which capability axis better at nano scale?"
Train a fresh nano TinyLM on ONE bAbI category (train-each-separately), test on a
disjoint dedup'd split, repeat over seeds for mean +/- std. Score is closed-vocab
single-token argmax; the bar is the per-axis MAJORITY baseline (a frequency-
collapsed model can't beat it). Induction-vs-binding-vs-AR separation tells you
the architecture's inductive bias, the thing zero-shot probes can't see at nano.

Axes (clean closed-vocab categories):
  binding   = two-arg-relations      (6 rooms, majority ~0.175)
  induction = basic-induction        (4 colors, majority ~0.273)
  ar        = single-supporting-fact (6 rooms, majority ~0.198)

Reuses the verified helpers from babi_twoarg_cpu_probe. CPU-only, no DB writes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from research.tools.babi_twoarg_cpu_probe import (
    _accuracy,
    _answer_token,
    _encode_rows,
    _load_category,
    _split,
    _train,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

AXES = {
    "binding": "two-arg-relations",
    "induction": "basic-induction",
    "ar": "single-supporting-fact",
}


def _prep(cat: str):
    """Load + normalize a category; return (df, candidate_tokens, majority)."""
    df = _load_category(cat)
    df = df.copy()
    df["answer"] = df["answer"].astype(str).str.rstrip(".")  # 'office.' -> 'office'
    answers = sorted(df["answer"].unique())
    cand = [_answer_token(a) for a in answers]
    majority = df["answer"].value_counts().iloc[0] / len(df)
    return df, cand, majority, answers


def _one_run(df, cand, lane, dim, n_blocks, passes, lr, batch, seed, max_len):
    torch.manual_seed(seed)
    train_df, test_df = _split(df, 0.2, seed)
    tri, trp, tra = _encode_rows(train_df, max_len)
    test_ids, test_pos, test_ans = _encode_rows(test_df, max_len)
    model = _build_tinylm(
        _build_lane_factory(lane), dim=dim, n_blocks=n_blocks, use_ffn=True
    )
    _train(model, tri, trp, tra, passes, lr, batch, seed)
    return (
        _accuracy(model, tri, trp, tra, cand),
        _accuracy(model, test_ids, test_pos, test_ans, cand),
    )


def main() -> None:
    # guardrail: allow-complexity - orchestrates a small multi-seed CPU probe.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lanes",
        nargs="+",
        default=[
            "softmax_attention",
            "reciprocal_rank_attention",
            "semiring_reciprocal_attention",
        ],
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=120)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/babi_multiaxis_lane_probe.json"),
    )
    args = ap.parse_args()

    prepped = {ax: _prep(cat) for ax, cat in AXES.items()}
    results = {}
    print(f"{'lane':32s} " + " ".join(f"{ax:>22s}" for ax in AXES))
    print(f"{'(majority)':32s} " + " ".join(f"{prepped[ax][2]:>22.3f}" for ax in AXES))
    for lane in args.lanes:
        results[lane] = {}
        cells = []
        for ax in AXES:
            df, cand, majority, _ = prepped[ax]
            tests = []
            trains = []
            for seed in range(args.seeds):
                train_acc, test_acc = _one_run(
                    df,
                    cand,
                    lane,
                    args.dim,
                    args.n_blocks,
                    args.passes,
                    args.lr,
                    args.batch,
                    seed,
                    args.max_len,
                )
                trains.append(train_acc)
                tests.append(test_acc)
            mean, std = st.mean(tests), st.pstdev(tests)
            margin = mean - majority
            results[lane][ax] = {
                "test_mean": round(mean, 4),
                "test_std": round(std, 4),
                "train_mean": round(st.mean(trains), 4),
                "majority": round(majority, 4),
                "margin_over_majority": round(margin, 4),
                "tests": [round(t, 4) for t in tests],
            }
            cells.append(f"{mean:.3f}±{std:.3f}({margin:+.3f})")
        print(f"{lane:32s} " + " ".join(f"{c:>22s}" for c in cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": {
                    "seeds": args.seeds,
                    "dim": args.dim,
                    "n_blocks": args.n_blocks,
                    "passes": args.passes,
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\ncell = test_mean ± std (margin over majority).  wrote {args.out}")


if __name__ == "__main__":
    main()
