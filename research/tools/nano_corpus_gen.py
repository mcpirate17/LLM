"""Generate an expandable nano binding corpus in the nano_corpus_v4 format.

The anti-memorization design (carried from v4): the model must learn that
"the {noun} **was**" -> adjective, NOT "the {noun}" -> adjective. So:
  * Bucket A: "the {noun} was {adj}"            -> the binding target (train nouns)
  * Bucket B: "the {noun} {verb}" / "I see the {noun}"  -> distractor frames where the
              noun is NOT followed by an adjective (so "was" is the binding cue)
  * Bucket C: "I see a {adj} {noun}"            -> adjectives in another construction
  * held-out nouns appear ONLY in Bucket B (their token exists) but never in A, so
    "the {held_out} was" tests rule generalization, not memorized pairs.

Scale it when a model memorizes: more words (--n-nouns/--n-adjectives/--n-verbs)
and more phrases (--bucket-a/-b/-c) grow the vocab + combinatorial space. Banks are
real common words (edit/extend below); generation errors loudly if you ask for more
than the bank holds. Output is drop-in for ``nano_softmax_lab --corpus``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Real common-word banks — extend these to go bigger.
NOUNS = [
    "boy",
    "girl",
    "man",
    "woman",
    "child",
    "dog",
    "cat",
    "bird",
    "mouse",
    "rabbit",
    "horse",
    "cow",
    "pig",
    "sheep",
    "fish",
    "car",
    "truck",
    "boat",
    "ship",
    "train",
    "plane",
    "chair",
    "table",
    "lamp",
    "book",
    "pen",
    "cup",
    "bowl",
    "box",
    "bag",
    "hat",
    "shoe",
    "ball",
    "tree",
    "flower",
    "house",
    "door",
    "window",
    "road",
    "clock",
]
ADJECTIVES = [
    "big",
    "small",
    "cold",
    "warm",
    "dry",
    "wet",
    "hard",
    "soft",
    "thin",
    "fat",
    "tall",
    "short",
    "fast",
    "slow",
    "old",
    "new",
    "clean",
    "dirty",
    "bright",
    "dark",
    "loud",
    "quiet",
    "sharp",
    "dull",
    "heavy",
    "light",
    "rough",
    "smooth",
    "round",
    "flat",
]
VERBS = [
    "sat",
    "ran",
    "jumped",
    "slept",
    "walked",
    "stood",
    "fell",
    "flew",
    "swam",
    "ate",
    "played",
    "rested",
    "moved",
    "waited",
    "looked",
]


def _bank(name: str, bank: list[str], k: int) -> list[str]:
    if k > len(bank):
        raise ValueError(
            f"asked for {k} {name} but bank has {len(bank)} — extend the {name.upper()} "
            f"list in nano_corpus_gen.py"
        )
    return bank[:k]


def generate(args) -> int:
    rng = random.Random(args.seed)
    nouns = _bank("nouns", NOUNS, args.n_nouns)
    adjs = _bank("adjectives", ADJECTIVES, args.n_adjectives)
    verbs = _bank("verbs", VERBS, args.n_verbs)
    held_out = nouns[: args.n_held_out]
    train_nouns = nouns[args.n_held_out :]

    lines: list[str] = []
    # Bucket A — binding target, train nouns only.
    lines.append(
        f"# Bucket A — target frame `the {{noun}} was {{adj}}`  (n={args.bucket_a})"
    )
    for _ in range(args.bucket_a):
        lines.append(f"the {rng.choice(train_nouns)} was {rng.choice(adjs)}")
    # Bucket B — distractor frames (ALL nouns incl. held-out; no adjective after noun).
    lines.append(f"# Bucket B — same nouns, other frames  (n={args.bucket_b})")
    for _ in range(args.bucket_b):
        noun = rng.choice(nouns)
        lines.append(
            rng.choice([f"the {noun} {rng.choice(verbs)}", f"I see the {noun}"])
        )
    # Bucket C — adjectives in another construction.
    lines.append(
        f"# Bucket C — same adjectives, other constructions  (n={args.bucket_c})"
    )
    for _ in range(args.bucket_c):
        lines.append(f"I see a {rng.choice(adjs)} {rng.choice(nouns)}")
    # Test prompts — held-out completions.
    lines.append(
        "# Test prompts (n=%d) — completion target: any adjective" % len(held_out)
    )
    for noun in held_out:
        lines.append(f"the {noun} was")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {out} — {args.bucket_a + args.bucket_b + args.bucket_c} train sentences, "
        f"{len(nouns)} nouns ({len(held_out)} held-out: {held_out}), {len(adjs)} adjectives, "
        f"{len(verbs)} verbs"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-nouns", type=int, default=20)
    p.add_argument("--n-adjectives", type=int, default=12)
    p.add_argument("--n-verbs", type=int, default=6)
    p.add_argument("--n-held-out", type=int, default=5)
    p.add_argument("--bucket-a", type=int, default=300)
    p.add_argument("--bucket-b", type=int, default=400)
    p.add_argument("--bucket-c", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", default=str(_REPO / "data" / "nano_corpus" / "nano_corpus_gen.txt")
    )
    return generate(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
