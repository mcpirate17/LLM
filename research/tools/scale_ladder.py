"""Param scaling ladder on a SCALABLE binding corpus — predictivity calibration.

History: synthetic probes (poor proxy) -> tiny real corpus (memorized + trivial
metric: softmax@24K hit 1.0 vs gpt2 ref 0.6, because "predict *an* adjective" over
a 27-word vocab is free). Both failures share a cause — too little data + too easy
a metric. This version fixes both:

  * SCALABLE corpus: generate n_nouns nouns x n_adjectives adjectives, bind each
    noun to a fixed ``k`` adjectives, emit sentences "the {noun} was {adj_in_set}".
    Volume scales ~5 tok/param per rung so bigger models can't just memorize.
  * DISCRIMINATING metric (bound-set recall): "the {noun} was -> top1 in that noun's
    bound set". Chance ~ k/n_adjectives, so a non-binder scores near chance — unlike
    the trivial "is it an adjective" test. Held-out nouns test rule generalization.
  * MEMORIZATION guard: report in-dist recall, held-out recall, and final train loss;
    flag memorized = (in-dist high AND held-out ~chance) or train_loss ~ 0.

Medium vocab (~100 words) keeps sub-1M full LMs reachable AND keeps recall non-trivial.
Run softmax first to validate the curve discriminates; then candidate lanes + the L4
anchor (40M binding from runs.db + the pinned softmax refs) give the cheapest
predictive rung. If a rung shows memorization, raise --tokens-per-param.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from component_fab.harness.tiny_lm import lane_factory_for_baseline
from component_fab.harness.training_probe import build_tiny_lm

_REPO = Path(__file__).resolve().parents[1]
_DIMS = (32, 64, 128, 256, 512)
_N_BLOCKS = 2
_N_NOUNS, _N_ADJ, _K, _N_HELDOUT = 60, 40, 3, 12  # vocab ~ 100+2; chance ~ k/N_ADJ


def _make_binding_corpus(*, n_sentences: int, seed: int) -> dict:
    """Generate a noun->adjective binding corpus that scales with n_sentences."""
    rng = random.Random(seed)
    nouns = [f"n{i}" for i in range(_N_NOUNS)]
    adjs = [f"a{i}" for i in range(_N_ADJ)]
    bound = {nn: rng.sample(adjs, _K) for nn in nouns}  # each noun -> k adjectives
    held_out = nouns[:_N_HELDOUT]
    train_nouns = nouns[_N_HELDOUT:]
    vocab = sorted(["the", "was", *nouns, *adjs])
    stoi = {w: i for i, w in enumerate(vocab)}
    sentences = []
    for _ in range(n_sentences):
        nn_ = rng.choice(train_nouns)
        sentences.append(["the", nn_, "was", rng.choice(bound[nn_])])
    return {
        "sentences": sentences,
        "stoi": stoi,
        "vocab": vocab,
        "bound": {nn: frozenset(stoi[a] for a in adjs_) for nn, adjs_ in bound.items()},
        "adj_ids": frozenset(stoi[a] for a in adjs),
        "train_nouns": train_nouns,
        "held_out": held_out,
        "chance": round(_K / _N_ADJ, 3),
    }


def _lm_params(lane_factory, dim: int, vocab_size: int) -> int:
    m = build_tiny_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=_N_BLOCKS,
        max_seq_len=8,
        use_position_embedding=True,
        use_ffn=True,
        ffn_mult=4,
    )
    return sum(p.numel() for p in m.parameters())


def _train_and_score(
    lane_factory,
    dim: int,
    *,
    device: str,
    seed: int,
    tokens_per_param: int,
    lr: float = 3e-3,
    batch_size: int = 128,
) -> dict:
    vocab_probe = len(_make_binding_corpus(n_sentences=1, seed=0)["vocab"])
    params = _lm_params(lane_factory, dim, vocab_probe)
    # ~tokens_per_param tokens; 4 tokens/sentence. Floor keeps tiny rungs trainable.
    n_sentences = max(400, (tokens_per_param * params) // 4)
    corpus = _make_binding_corpus(n_sentences=n_sentences, seed=seed)
    stoi, vocab_size = corpus["stoi"], len(corpus["vocab"])
    torch.manual_seed(seed)
    model = build_tiny_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=_N_BLOCKS,
        max_seq_len=8,
        use_position_embedding=True,
        use_ffn=True,
        ffn_mult=4,
    ).to(device)
    data = torch.tensor(
        [[stoi[w] for w in s] for s in corpus["sentences"]], device=device
    )
    n = data.shape[0]
    n_steps = max(300, min(3000, n // batch_size * 4))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    final_loss = float("nan")
    for _ in range(n_steps):
        batch = data[torch.randint(0, n, (min(batch_size, n),), device=device)]
        logits = model(batch)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size), batch[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    model.eval()
    the_i, was_i = stoi["the"], stoi["was"]

    def _recall(nouns: list[str], bound_check: bool) -> float:
        hits = 0
        with torch.no_grad():
            for noun in nouns:
                top1 = int(
                    model(torch.tensor([[the_i, stoi[noun], was_i]], device=device))[
                        0, -1
                    ].argmax()
                )
                target = corpus["bound"][noun] if bound_check else corpus["adj_ids"]
                hits += int(top1 in target)
        return hits / max(1, len(nouns))

    in_dist = _recall(
        corpus["train_nouns"], bound_check=True
    )  # right adj for that noun
    held_out = _recall(
        corpus["held_out"], bound_check=False
    )  # any adj (rule generalizes)
    memorized = in_dist > 0.5 and held_out <= corpus["chance"] * 2
    return {
        "dim": dim,
        "params": params,
        "n_sentences": n_sentences,
        "n_steps": n_steps,
        "in_dist_bound_recall": round(in_dist, 3),
        "held_out_is_adj": round(held_out, 3),
        "final_loss": round(final_loss, 3),
        "memorized": memorized,
    }


def run_ladder(lane_name: str, *, device: str, seeds, tokens_per_param: int) -> dict:
    lane_factory = lane_factory_for_baseline(lane_name)
    chance = _make_binding_corpus(n_sentences=1, seed=0)["chance"]
    print(f"\n=== {lane_name}: binding ladder (chance bound-recall ~ {chance}) ===")
    rungs = []
    for dim in _DIMS:
        t0 = time.time()
        runs = [
            _train_and_score(
                lane_factory,
                dim,
                device=device,
                seed=s,
                tokens_per_param=tokens_per_param,
            )
            for s in seeds
        ]
        row = {
            "dim": dim,
            "params": runs[0]["params"],
            "n_sentences": runs[0]["n_sentences"],
            "in_dist_bound_recall": round(
                mean(r["in_dist_bound_recall"] for r in runs), 3
            ),
            "held_out_is_adj": round(mean(r["held_out_is_adj"] for r in runs), 3),
            "final_loss": round(mean(r["final_loss"] for r in runs), 3),
            "memorized_any": any(r["memorized"] for r in runs),
            "seconds": round(time.time() - t0, 1),
        }
        rungs.append(row)
        flag = " [MEMORIZED]" if row["memorized_any"] else ""
        print(
            f"  dim{dim:<4} params={row['params']:>9,} sents={row['n_sentences']:>7,} "
            f"bound_recall={row['in_dist_bound_recall']:.3f} held_out_adj={row['held_out_is_adj']:.3f} "
            f"loss={row['final_loss']:.2f}{flag} ({row['seconds']}s)"
        )
    return {"lane": lane_name, "chance": chance, "rungs": rungs}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lanes", nargs="*", default=["softmax_attention", "gpt2"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tokens-per-param", type=int, default=20)
    p.add_argument(
        "--output", default=str(_REPO / "reports" / "scale_ladder_binding.json")
    )
    args = p.parse_args(argv)
    out = {
        "device": args.device,
        "tokens_per_param": args.tokens_per_param,
        "config": {"n_nouns": _N_NOUNS, "n_adj": _N_ADJ, "k": _K},
        "ladders": [],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for lane in args.lanes:
        out["ladders"].append(
            run_ladder(
                lane,
                device=args.device,
                seeds=tuple(range(args.seeds)),
                tokens_per_param=args.tokens_per_param,
            )
        )
        out_path.write_text(json.dumps(out, indent=2))
    print(
        f"\nreport: {out_path}  | L4 anchor: research/data/scale_ladder/softmax_l4_anchor.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
