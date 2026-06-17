"""Memorization-capacity sweep: how many noun->adjective bindings can a fixed
multi-head softmax model hold before recall drops below threshold?

Each of N nouns is bound to ONE specific adjective (1:1), so "correct" is defined:
"the {noun} was" -> top1 == that noun's bound adjective. Distractor frames
("the {noun} {verb}") are mixed in so the model must use "was" (not just "the
{noun}") — otherwise the noun token alone would leak the adjective.

Sweep N (the binding load); training steps scale with N so each pair gets ample
passes (capacity, not under-training — watch train_loss). Report the recall curve
and the crossover: the largest N with recall >= --threshold (~0.55), i.e. the
point at which the model can no longer memorize the corpus. Optionally sweep
model dim too, to get capacity-vs-params.

    python -m research.tools.nano_capacity_sweep --dims 64 128 --loads 10 20 40 80 160 320
"""

from __future__ import annotations

import argparse
import random
import time
from statistics import mean

import torch
from torch import nn

from research.tools.nano_softmax_lab import _PAD, _build

_N_ADJ = 40  # adjective inventory; nouns reuse adjectives once N > _N_ADJ
_VERBS = ("sat", "ran", "jumped", "slept", "walked", "stood")


def _make_corpus(n_nouns: int, *, distractor_per_noun: int, seed: int):
    rng = random.Random(seed)
    nouns = [f"n{i}" for i in range(n_nouns)]
    adjs = [f"a{i}" for i in range(_N_ADJ)]
    bound = {nouns[i]: adjs[i % _N_ADJ] for i in range(n_nouns)}  # 1:1 (cyclic)
    vocab = [_PAD, "the", "was", *_VERBS, *nouns, *adjs]
    stoi = {w: i for i, w in enumerate(vocab)}
    sentences = []
    for noun in nouns:
        sentences.append(["the", noun, "was", bound[noun]])  # the binding
        for _ in range(distractor_per_noun):
            sentences.append(
                ["the", noun, rng.choice(_VERBS)]
            )  # forces "was" to matter
    return sentences, stoi, vocab, bound


def _train_and_recall(
    dim, n_blocks, n_nouns, *, device, seed, steps, lr, batch_size, distractor_per_noun
) -> dict:
    sentences, stoi, vocab, bound = _make_corpus(
        n_nouns, distractor_per_noun=distractor_per_noun, seed=seed
    )
    pad = stoi[_PAD]
    maxlen = max(len(s) for s in sentences)
    data = torch.tensor(
        [[stoi[w] for w in s] + [pad] * (maxlen - len(s)) for s in sentences],
        device=device,
    )
    torch.manual_seed(seed)
    model = _build(dim, n_blocks, len(vocab), maxlen, device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=pad)
    n = data.shape[0]
    final_loss = float("nan")
    model.train()
    for _ in range(steps):
        batch = data[torch.randint(0, n, (min(batch_size, n),), device=device)]
        logits = model(batch)
        loss = ce(logits[:, :-1, :].reshape(-1, len(vocab)), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    model.eval()
    the_i, was_i = stoi["the"], stoi["was"]
    hits = 0
    with torch.no_grad():
        for noun, adj in bound.items():
            top1 = int(
                model(torch.tensor([[the_i, stoi[noun], was_i]], device=device))[
                    0, -1
                ].argmax()
            )
            hits += int(top1 == stoi[adj])
    return {
        "n_nouns": n_nouns,
        "params": params,
        "recall": hits / n_nouns,
        "final_loss": round(final_loss, 3),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", nargs="*", type=int, default=[128])
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--loads", nargs="*", type=int, default=[10, 20, 40, 80, 160, 320])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--distractor-per-noun", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--steps-per-noun", type=int, default=40)
    p.add_argument("--min-steps", type=int, default=1500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    for dim in args.dims:
        print(f"\n=== dim={dim} (multi-head softmax, n_blocks={args.n_blocks}) ===")
        print(
            f"{'N_bind':>7} {'params':>9} {'recall':>7} {'loss':>6}  (threshold {args.threshold})"
        )
        capacity = None
        for n_nouns in sorted(args.loads):
            steps = max(args.min_steps, args.steps_per_noun * n_nouns)
            t0 = time.time()
            runs = [
                _train_and_recall(
                    dim,
                    args.n_blocks,
                    n_nouns,
                    device=args.device,
                    seed=s,
                    steps=steps,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    distractor_per_noun=args.distractor_per_noun,
                )
                for s in range(args.seeds)
            ]
            recall = mean(r["recall"] for r in runs)
            loss = mean(r["final_loss"] for r in runs)
            mark = "" if recall >= args.threshold else "  <-- below threshold"
            print(
                f"{n_nouns:>7} {runs[0]['params']:>9,} {recall:>7.3f} {loss:>6.2f}"
                f"  ({steps}st,{round(time.time() - t0)}s){mark}"
            )
            if recall >= args.threshold:
                capacity = n_nouns
        print(
            f"capacity @ dim{dim}: largest N with recall>={args.threshold} = {capacity}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
