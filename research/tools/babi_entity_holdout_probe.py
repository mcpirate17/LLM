"""Relation/entity-holdout binding probe: does a nano mixer learn the
two-arg BINDING RULE, or just memorize answer frequencies?

Random-split test (babi_multiaxis_lane_probe) only checks "unseen prompts". This
is stricter: we hold out specific target relation -> answer entity bindings, then
test on rows requiring those bindings. The held answer entities still appear as
training targets for other relations, so their unembedding rows receive gradient;
the probe measures relation-conditioned rule learning rather than the impossible
"emit a never-trained answer token" variant.

Two scorings, both reported (honest about the known hardness of entity holdout —
a relation/entity binding is unseen even though the entity token is trained):
  strict   : argmax over ALL 6 rooms        (chance 1/6; tests full generalization)
  restricted: argmax over held binding entities (chance 1/k; "given it must be one
             of these entities, did it bind the relation correctly")

5 seeds; each seed draws a different held-out room pair AND a different init, so
std captures both entity-choice and init variance (the realistic noise). Only
binding is run — induction/AR sit at the majority floor on the random split
already, so entity-holdout there would be floor noise. Lanes: only those VERIFIED
to differ in forward pass (reciprocal_rank is a no-op == softmax, excluded).

CPU-only, no DB writes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import polars as pl
import torch

from research.tools.babi_twoarg_cpu_probe import (
    _accuracy,
    _answer_token,
    _encode_rows,
    _load_category,
    _train,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

CATEGORY = "two-arg-relations"
TARGET_PASSAGE_RE = re.compile(
    r"^The (?P<answer>\w+) is (?P<relation>north|south|east|west) of the (?P<anchor>\w+)\.$"
)


def _target_binding(row) -> tuple[str, str]:
    """Return the target (relation, answer entity) from the supporting passage."""
    passages = ast.literal_eval(row["answer_passage"])
    if len(passages) != 1:
        raise ValueError(f"expected one answer passage, got {passages!r}")
    match = TARGET_PASSAGE_RE.match(passages[0])
    if match is None:
        raise ValueError(f"cannot parse answer passage: {passages[0]!r}")
    answer = str(row["answer"]).rstrip(".")
    if match.group("answer") != answer:
        raise ValueError(
            f"answer passage/entity mismatch: passage={passages[0]!r} answer={answer!r}"
        )
    return match.group("relation"), answer


def _with_target_bindings(df):
    bindings = [_target_binding(row) for row in df.iter_rows(named=True)]
    return df.with_columns(
        pl.Series("_target_relation", [b[0] for b in bindings]),
        pl.Series("_target_entity", [b[1] for b in bindings]),
    )


def _binding_split(df, n_holdout, seed):
    """Hold out seeded relation/entity bindings, not whole answer entities."""
    g = torch.Generator().manual_seed(seed)
    df = _with_target_bindings(df)
    pairs = sorted(set(zip(df["_target_relation"], df["_target_entity"])))
    perm = torch.randperm(len(pairs), generator=g).tolist()
    held_pairs: list[tuple[str, str]] = []
    held_entities: set[str] = set()
    for idx in perm:
        relation, entity = pairs[idx]
        if entity in held_entities:
            continue
        held_pairs.append((relation, entity))
        held_entities.add(entity)
        if len(held_pairs) == n_holdout:
            break
    if len(held_pairs) < n_holdout:
        raise ValueError(f"could only select {len(held_pairs)} distinct held bindings")

    held_set = set(held_pairs)
    is_test = pl.Series(
        [
            (rel, ent) in held_set
            for rel, ent in zip(df["_target_relation"], df["_target_entity"])
        ]
    )
    train = df.filter(~is_test)
    test = df.filter(is_test)
    train_answers = set(train["answer"].cast(pl.String).str.strip_chars_end("."))
    missing = sorted(held_entities - train_answers)
    if missing:
        raise AssertionError(
            "held answer entities must appear as training targets for other "
            f"relations; missing={missing}"
        )
    held = [
        {"relation": relation, "entity": entity}
        for relation, entity in sorted(held_pairs)
    ]
    return train, test, held


def _majority_accuracy(train, test, candidates) -> tuple[str, float]:
    """Accuracy of the most frequent train answer within the candidate set."""
    cand = set(candidates)
    overlap = train.filter(pl.col("answer").is_in(cand))["answer"].to_list()
    if not overlap:
        raise ValueError(f"no training answers overlap candidates {sorted(cand)!r}")
    counts = Counter(overlap)  # insertion-ordered: ties go to the first-seen answer
    pred = str(max(counts.items(), key=lambda kv: kv[1])[0])
    return pred, float((test["answer"] == pred).mean())


def _adaptive_depth_stats(model, ids, batch) -> dict[str, float] | None:
    """Aggregate native adaptive-recursion depth counters over a dataset."""
    modules = [m for m in model.modules() if hasattr(m, "last_depth_counts")]
    if not modules:
        return None
    total = 0
    weighted_sum = 0.0
    zeros = 0
    max_seen = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            _ = model(ids[i : i + batch])
            for module in modules:
                depth = getattr(module, "last_depth_counts", None)
                if depth is None:
                    continue
                depth_cpu = depth.detach().cpu()
                total += int(depth_cpu.numel())
                weighted_sum += float(depth_cpu.float().sum().item())
                zeros += int((depth_cpu == 0).sum().item())
                max_seen = max(max_seen, int(depth_cpu.max().item()))
    if total == 0:
        return None
    return {
        "mean_depth": round(weighted_sum / total, 4),
        "skip_fraction": round(zeros / total, 4),
        "max_depth_seen": max_seen,
    }


def _one_run(
    df,
    rooms,
    room_toks,
    lane,
    n_holdout,
    dim,
    n_blocks,
    passes,
    lr,
    batch,
    seed,
    max_len,
    mtp_depth=0,
    mtp_weight=0.0,
):
    torch.manual_seed(seed)
    tr, test_df, held = _binding_split(df, n_holdout, seed)
    tri, trp, tra = _encode_rows(tr, max_len)
    test_ids, test_pos, test_ans = _encode_rows(test_df, max_len)
    model = _build_tinylm(
        _build_lane_factory(lane), dim=dim, n_blocks=n_blocks, use_ffn=True
    )
    _train(
        model,
        tri,
        trp,
        tra,
        passes,
        lr,
        batch,
        seed,
        mtp_depth=mtp_depth,
        mtp_weight=mtp_weight,
    )
    held_entities = sorted({h["entity"] for h in held})
    held_toks = [_answer_token(r) for r in held_entities]
    strict = _accuracy(model, test_ids, test_pos, test_ans, room_toks)
    restricted = _accuracy(model, test_ids, test_pos, test_ans, held_toks)
    adaptive_depth = _adaptive_depth_stats(model, test_ids, batch)
    strict_majority_pred, strict_majority = _majority_accuracy(tr, test_df, rooms)
    restricted_majority_pred, restricted_majority = _majority_accuracy(
        tr, test_df, held_entities
    )
    # train-on-seen sanity: accuracy on a held-out slice of TRAIN rooms
    tr_acc = _accuracy(model, tri, trp, tra, room_toks)
    return {
        "held": held,
        "held_entities": held_entities,
        "n_test": len(test_df),
        "n_train": len(tr),
        "strict": strict,
        "restricted": restricted,
        "adaptive_depth": adaptive_depth,
        "strict_majority_pred": strict_majority_pred,
        "strict_majority": strict_majority,
        "restricted_majority_pred": restricted_majority_pred,
        "restricted_majority": restricted_majority,
        "train": tr_acc,
    }


def main() -> None:
    # guardrail: allow-god-function - standalone CLI assembles one JSON report.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lanes",
        nargs="+",
        default=["softmax_attention", "semiring_reciprocal_attention"],
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--n-holdout",
        type=int,
        default=2,
        help="relation/entity bindings held out per seed",
    )
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument(
        "--mtp-depth",
        type=int,
        default=0,
        help="auxiliary multi-token prediction depth over the tokenized sequence",
    )
    ap.add_argument(
        "--mtp-weight",
        type=float,
        default=0.0,
        help="weight for the auxiliary multi-token prediction loss",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/babi_entity_holdout_probe.json"),
    )
    args = ap.parse_args()

    df = _load_category(CATEGORY)
    df = df.with_columns(pl.col("answer").cast(pl.String).str.strip_chars_end("."))
    rooms = sorted(df["answer"].unique().to_list())
    room_toks = [_answer_token(r) for r in rooms]
    strict_chance = 1.0 / len(rooms)

    print(
        f"binding holdout: {len(rooms)} rooms, hold out {args.n_holdout} "
        f"relation/entity bindings per seed, "
        f"{args.seeds} seeds"
    )
    print(f"chance: strict {strict_chance:.3f} (1/{len(rooms)})\n")
    print(
        f"{'lane':32s} {'strict(all6)':>20s} {'restricted(held)':>20s} "
        f"{'maj(strict)':>12s} {'maj(restr)':>12s} {'train':>12s}"
    )

    results = {}
    for lane in args.lanes:
        runs = [
            _one_run(
                df,
                rooms,
                room_toks,
                lane,
                args.n_holdout,
                args.dim,
                args.n_blocks,
                args.passes,
                args.lr,
                args.batch,
                s,
                args.max_len,
                mtp_depth=args.mtp_depth,
                mtp_weight=args.mtp_weight,
            )
            for s in range(args.seeds)
        ]
        strict = [r["strict"] for r in runs]
        restr = [r["restricted"] for r in runs]
        train = [r["train"] for r in runs]
        strict_majority = [r["strict_majority"] for r in runs]
        restricted_majority = [r["restricted_majority"] for r in runs]
        restricted_chances = [1.0 / len(r["held_entities"]) for r in runs]
        depth_stats = [r["adaptive_depth"] for r in runs if r["adaptive_depth"]]
        results[lane] = {
            "strict_mean": round(st.mean(strict), 4),
            "strict_std": round(st.pstdev(strict), 4),
            "restricted_mean": round(st.mean(restr), 4),
            "restricted_std": round(st.pstdev(restr), 4),
            "train_mean": round(st.mean(train), 4),
            "strict_chance_margin": round(st.mean(strict) - strict_chance, 4),
            "restricted_chance_mean": round(st.mean(restricted_chances), 4),
            "restricted_chance_margin": round(
                st.mean(restr) - st.mean(restricted_chances), 4
            ),
            "strict_majority_mean": round(st.mean(strict_majority), 4),
            "restricted_majority_mean": round(st.mean(restricted_majority), 4),
            "strict_margin_over_majority": round(
                st.mean(strict) - st.mean(strict_majority), 4
            ),
            "restricted_margin_over_majority": round(
                st.mean(restr) - st.mean(restricted_majority), 4
            ),
            "adaptive_depth_mean": (
                round(st.mean(d["mean_depth"] for d in depth_stats), 4)
                if depth_stats
                else None
            ),
            "adaptive_skip_fraction_mean": (
                round(st.mean(d["skip_fraction"] for d in depth_stats), 4)
                if depth_stats
                else None
            ),
            "adaptive_max_depth_seen": (
                max(d["max_depth_seen"] for d in depth_stats) if depth_stats else None
            ),
            "per_seed": runs,
        }
        r = results[lane]
        print(
            f"{lane:32s} "
            f"{r['strict_mean']:.3f}±{r['strict_std']:.3f}({r['strict_margin_over_majority']:+.3f})".rjust(
                20
            )
            + " "
            + f"{r['restricted_mean']:.3f}±{r['restricted_std']:.3f}({r['restricted_margin_over_majority']:+.3f})".rjust(
                20
            )
            + " "
            + f"{r['strict_majority_mean']:.3f}".rjust(12)
            + " "
            + f"{r['restricted_majority_mean']:.3f}".rjust(12)
            + " "
            + f"{r['train_mean']:.3f}".rjust(12)
        )
        if r["adaptive_depth_mean"] is not None:
            print(
                "  adaptive depth "
                f"mean={r['adaptive_depth_mean']:.3f} "
                f"skip={r['adaptive_skip_fraction_mean']:.3f} "
                f"max={r['adaptive_max_depth_seen']}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": vars(args) | {"out": str(args.out)},
                "rooms": rooms,
                "chance": {
                    "strict": round(strict_chance, 4),
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(
        f"\nstrict = over all 6 rooms; restricted = over held binding entities. "
        f"Parentheses are margin over per-seed train-majority baseline.\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
