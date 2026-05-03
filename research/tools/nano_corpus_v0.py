"""Nano-corpus v0 — cheat-check controlled corpus.

Design (locked with Tim, 2026-05-03):

* 80 target-frame sentences:    `the {noun} was {adj}`
* 120 same-noun other-frames:   `the {noun} {verb}` / `I see the {noun}`
* 80 same-adj other-frames:     `the {adj} {noun} {verb}` / `I see a {adj} {noun}`
* 10 test prompts left in training (cheat-check):
                                `the {noun} was`

Lexicon (all single-token under tiktoken-gpt2):

* adjectives (10): fat thin big small wet dry warm cold soft hard
* nouns      (15): cat dog bird mouse rabbit man boy woman girl child car
                   ship lamp book chair
* verbs      (5):  ran jumped slept sat (+ "see" used in I-frame)
* function   (3):  the a I

Run: ``python -m research.tools.nano_corpus_v0``.
Output: ``research/reports/nano_corpus_v0.txt`` (plain text, readable
top-to-bottom) and ``nano_corpus_v0_test_prompts.txt``.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

ADJECTIVES: tuple[str, ...] = (
    # Original 10 (preserve existing v4 corpus byte-for-byte at default subset)
    "fat",
    "thin",
    "big",
    "small",
    "wet",
    "dry",
    "warm",
    "cold",
    "soft",
    "hard",
    # Extension for vocab sweeps (all single-token under cl100k_base, verified)
    "red",
    "blue",
    "green",
    "white",
    "black",
    "fast",
    "slow",
    "tall",
    "short",
    "fresh",
    "clean",
    "dirty",
    "sweet",
    "sharp",
    "deep",
    "light",
    "heavy",
    "loud",
    "quiet",
    "bright",
    "sad",
    "happy",
    "rich",
    "poor",
    "strong",
    "weak",
    "old",
    "new",
    "dark",
    "full",
)
# Default working subset; CLI can override via --n-adjectives.
DEFAULT_N_ADJECTIVES = 10

NOUNS: tuple[str, ...] = (
    "cat",
    "dog",
    "bird",
    "mouse",
    "rabbit",
    "man",
    "boy",
    "woman",
    "girl",
    "child",
    "car",
    "ship",
    "lamp",
    "book",
    "chair",
)

# Frames for bucket B — same-noun, other constructions.
B_FRAMES: tuple[str, ...] = (
    "the {noun} ran",
    "the {noun} jumped",
    "the {noun} slept",
    "the {noun} sat",
    "I see the {noun}",
)

# Frames for bucket C — same-adjective, other constructions.
C_FRAMES: tuple[str, ...] = (
    "the {adj} {noun} ran",
    "the {adj} {noun} sat",
    "I see a {adj} {noun}",
)


def _bucket_a(
    rng: random.Random,
    n: int,
    *,
    exclude_nouns: frozenset[str] = frozenset(),
    strict_selection: bool = False,
    n_adj_per_noun: int = 3,
    n_adjectives: int | None = None,
) -> list[str]:
    """Target frame: ``the {noun} was {adj}``.

    ``exclude_nouns`` is the held-out set; those nouns are NOT used in this
    bucket so they never co-occur with ``was`` in training.

    When ``strict_selection`` is True, each in-dist noun is assigned a fixed
    subset of ``n_adj_per_noun`` adjectives drawn deterministically from the
    full pool.  The subset assignment is `(noun_idx + i) % len(ADJECTIVES)`
    for `i in range(n_adj_per_noun)`, so noun_0 takes adjs[0:3], noun_1 takes
    adjs[1:4], etc.  This gives each noun a unique-but-overlapping signature.
    """
    in_dist = [n_ for n_ in NOUNS if n_ not in exclude_nouns]
    adj_pool_size = (
        len(ADJECTIVES)
        if n_adjectives is None
        else max(1, min(int(n_adjectives), len(ADJECTIVES)))
    )
    adj_pool = ADJECTIVES[:adj_pool_size]
    pairs: list[tuple[str, str]] = []
    if strict_selection:
        n_adj = max(1, min(int(n_adj_per_noun), adj_pool_size))
        for noun_idx, noun in enumerate(in_dist):
            adj_subset = [
                adj_pool[(noun_idx + i) % adj_pool_size] for i in range(n_adj)
            ]
            for adj in adj_subset:
                pairs.append((noun, adj))
    else:
        pairs = [(noun, adj) for noun in in_dist for adj in adj_pool]
    rng.shuffle(pairs)
    if strict_selection and len(pairs) < n:
        # Repeat the strict pairs to reach n; each (noun, adj) appears many times.
        repeats = (n + len(pairs) - 1) // len(pairs)
        expanded: list[tuple[str, str]] = []
        for _ in range(repeats):
            expanded.extend(pairs)
            rng.shuffle(expanded)
        pairs = expanded
    return [f"the {n_} was {a_}" for n_, a_ in pairs[:n]]


def _bucket_b(rng: random.Random, n: int) -> list[str]:
    """Same-noun, other-frame.  Cycle each noun across the 5 frames so every noun
    appears with every frame at least once before any frame repeats."""
    sentences: list[str] = []
    indices: list[tuple[int, int]] = [
        (ni, fi) for ni in range(len(NOUNS)) for fi in range(len(B_FRAMES))
    ]
    rng.shuffle(indices)
    repeats_needed = (n + len(indices) - 1) // len(indices)
    pool: list[tuple[int, int]] = []
    for _ in range(repeats_needed):
        pool.extend(indices)
        rng.shuffle(pool)
    for ni, fi in pool[:n]:
        sentences.append(B_FRAMES[fi].format(noun=NOUNS[ni]))
    return sentences


def _bucket_c(rng: random.Random, n: int) -> list[str]:
    """Same-adjective, other-construction.  Pair each adjective with random nouns
    across the 3 C frames so adjectives are seen attributively, not just predicatively."""
    triples: list[tuple[int, int, int]] = [
        (ai, ni, fi)
        for ai in range(len(ADJECTIVES))
        for ni in range(len(NOUNS))
        for fi in range(len(C_FRAMES))
    ]
    rng.shuffle(triples)
    sentences = [
        C_FRAMES[fi].format(adj=ADJECTIVES[ai], noun=NOUNS[ni])
        for ai, ni, fi in triples[:n]
    ]
    return sentences


def _test_prompts(noun_subset: Sequence[str]) -> list[str]:
    """One prompt per noun in ``noun_subset``."""
    return [f"the {n} was" for n in noun_subset]


def build_corpus(
    *,
    n_a: int = 80,
    n_b: int = 120,
    n_c: int = 80,
    seed: int = 0,
    test_nouns: Sequence[str] = (
        "cat",
        "dog",
        "bird",
        "mouse",
        "rabbit",
        "man",
        "boy",
        "woman",
        "car",
        "book",
    ),
) -> tuple[list[str], list[str]]:
    """Return ``(corpus_lines, test_prompts)``.

    ``corpus_lines`` includes the test prompts' completed forms (cheat check)
    interleaved with all three buckets, deterministic under ``seed``.
    """
    rng = random.Random(int(seed))
    a = _bucket_a(rng, n_a)
    b = _bucket_b(rng, n_b)
    c = _bucket_c(rng, n_c)
    return list(a) + list(b) + list(c), _test_prompts(test_nouns)


def _render_text(
    bucket_a: list[str], bucket_b: list[str], bucket_c: list[str], prompts: list[str]
) -> str:
    out: list[str] = []
    out.append(
        f"# Bucket A — target frame `the {{noun}} was {{adj}}`  (n={len(bucket_a)})"
    )
    out.extend(bucket_a)
    out.append("")
    out.append(f"# Bucket B — same nouns, other frames  (n={len(bucket_b)})")
    out.extend(bucket_b)
    out.append("")
    out.append(
        f"# Bucket C — same adjectives, other constructions  (n={len(bucket_c)})"
    )
    out.extend(bucket_c)
    out.append("")
    out.append(
        f"# Test prompts (n={len(prompts)}) — completion target: any of the 10 adjectives"
    )
    out.extend(prompts)
    return "\n".join(out)


def _coverage_summary(corpus: list[str]) -> str:
    from collections import Counter

    noun_counts = Counter()
    adj_counts = Counter()
    was_after_noun = Counter()
    for line in corpus:
        toks = line.split()
        for t in toks:
            if t in NOUNS:
                noun_counts[t] += 1
            if t in ADJECTIVES:
                adj_counts[t] += 1
        for i, t in enumerate(toks[:-2]):
            if t in NOUNS and toks[i + 1] == "was":
                was_after_noun[t] += 1
    lines = [
        "# Coverage summary",
        f"  total sentences: {len(corpus)}",
        "",
        "## Noun frequency (any position)",
    ]
    for n_, c in sorted(noun_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n_:>8}  {c:3d}")
    lines.append("")
    lines.append("## Adjective frequency (any position)")
    for a_, c in sorted(adj_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {a_:>8}  {c:3d}")
    lines.append("")
    lines.append("## `{noun} was` co-occurrence count (target frame)")
    for n_, c in sorted(was_after_noun.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n_:>8}  {c:3d}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=80)
    ap.add_argument("--n-b", type=int, default=120)
    ap.add_argument("--n-c", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-corpus",
        type=Path,
        default=Path("research/reports/nano_corpus_v0.txt"),
    )
    ap.add_argument(
        "--out-coverage",
        type=Path,
        default=Path("research/reports/nano_corpus_v0_coverage.txt"),
    )
    ap.add_argument(
        "--hold-out",
        nargs="*",
        default=(),
        help="Nouns to hold out from bucket A (the `was` frame). They still "
        "appear in buckets B and C so they have stable embeddings.",
    )
    ap.add_argument(
        "--test-nouns",
        nargs="*",
        default=(
            "cat",
            "dog",
            "bird",
            "mouse",
            "rabbit",
            "man",
            "boy",
            "woman",
            "car",
            "book",
        ),
        help="Nouns used in the `the {noun} was ___` test prompts.",
    )
    ap.add_argument(
        "--strict-selection",
        action="store_true",
        help="Each in-dist noun is restricted to a fixed n-adjective subset.",
    )
    ap.add_argument(
        "--n-adj-per-noun",
        type=int,
        default=3,
        help="When --strict-selection is set, how many adjectives each noun gets.",
    )
    ap.add_argument(
        "--n-adjectives",
        type=int,
        default=DEFAULT_N_ADJECTIVES,
        help="Use only the first N adjectives from the pool (1..30). Default 10.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    held = frozenset(args.hold_out)
    if held - set(NOUNS):
        raise SystemExit(f"unknown hold-out nouns: {held - set(NOUNS)}")
    rng = random.Random(int(args.seed))
    a = _bucket_a(
        rng,
        args.n_a,
        exclude_nouns=held,
        strict_selection=bool(args.strict_selection),
        n_adj_per_noun=int(args.n_adj_per_noun),
        n_adjectives=int(args.n_adjectives),
    )
    b = _bucket_b(rng, args.n_b)
    c = _bucket_c(rng, args.n_c)
    prompts = _test_prompts(args.test_nouns)

    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    args.out_corpus.write_text(_render_text(a, b, c, prompts))
    args.out_coverage.write_text(_coverage_summary(a + b + c))
    logger.info("wrote %s (hold_out=%s)", args.out_corpus, sorted(held))
    logger.info("wrote %s", args.out_coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
