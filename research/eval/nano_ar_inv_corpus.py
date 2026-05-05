"""Nano-AR-INV corpus generator — investigation-tier associative-recall probe.

Frame: ``the {noun} had a {adj} {object}`` — 6 single-token words under
cl100k_base. Each in-dist noun gets a fixed set of (adj, object) facts the
model sees repeated in training. Held-out nouns get one (adj, object) fact
that NEVER appears in training; the model must compositionally retrieve the
held-out fact at eval time using only its in-distribution priors.

Vocabulary is shared with NanoBind (nouns + adjectives) plus a new OBJECTS
pool of 30 single-token nouns. Tokenizer: cl100k_base. Same encoder as
``compile_model``.

Discriminator design:
  - In-dist exact retrieval (adj + object both correct): tests retrieval
    mechanism. Conv-only / SSM-only fail; attention passes.
  - Held-out class accuracy (adj is any adj, object is any object): tests
    whether the architecture learned the systematic frame structure.
  - Held-out exact retrieval: the breakthrough signal — true content-based
    addressing. Most architectures fail; only attention-class with adequate
    capacity passes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from research.tools.nano_corpus_v0 import ADJECTIVES, NOUNS

# Single-token under cl100k_base in mid-sentence form (' apple', ' marble', ...).
OBJECTS: tuple[str, ...] = (
    "apple",
    "marble",
    "seed",
    "hat",
    "ball",
    "book",
    "ring",
    "key",
    "coin",
    "stone",
    "leaf",
    "rock",
    "cup",
    "box",
    "flag",
    "sock",
    "bag",
    "pen",
    "card",
    "cake",
    "pie",
    "fish",
    "frog",
    "star",
    "flower",
    "shell",
    "feather",
    "bone",
    "crown",
    "wheel",
)

# Match nano_bind's held-out set so the two probes share noun semantics.
DEFAULT_HELD_OUT_NOUNS: tuple[str, ...] = ("cat", "book", "lamp", "child", "ship")

DEFAULT_N_PAIRS_PER_NOUN = 5  # facts per in-dist noun; pair-match grader credits any
DEFAULT_REPS = 10  # repetitions per fact in training corpus
DEFAULT_N_DISTRACTORS = 480

TARGET_FRAME = "the {noun} had a {adj} {object}"
DISTRACTOR_NOUN_FRAMES: tuple[str, ...] = (
    "the {noun} ran",
    "I see the {noun}",
    "the {noun} sat",
)
DISTRACTOR_ADJ_FRAMES: tuple[str, ...] = (
    "the {adj} sky was clear",
    "a {adj} day ended",
)
DISTRACTOR_OBJECT_FRAMES: tuple[str, ...] = (
    "I held a {object} today",
    "she found a {object} there",
)


@dataclass(frozen=True, slots=True)
class Fact:
    """One (noun, adj, object) binding."""

    noun: str
    adj: str
    obj: str
    held_out: bool

    def sentence(self) -> str:
        return TARGET_FRAME.format(noun=self.noun, adj=self.adj, object=self.obj)


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Deterministic facts + sentences ready to tokenize."""

    facts: tuple[Fact, ...]
    train_sentences: tuple[str, ...]
    # ``test_facts`` is deduplicated by noun — exactly one Fact per unique
    # in-dist noun + one per held-out noun. The (adj, obj) on the fact is
    # only used to identify "which fact's answer" for held-out exact grading;
    # in-dist grading uses ``trained_pairs_by_noun`` for any-pair-match.
    test_facts: tuple[Fact, ...]
    # noun → set of (adj, obj) tuples the model saw in training
    trained_pairs_by_noun: dict[str, frozenset[tuple[str, str]]]
    seed: int


def build_facts(
    *,
    seed: int,
    n_pairs_per_noun: int = DEFAULT_N_PAIRS_PER_NOUN,
    held_out_nouns: Sequence[str] = DEFAULT_HELD_OUT_NOUNS,
    n_adjectives: int = 10,
    n_objects: int = 15,
) -> tuple[Fact, ...]:
    """Assign deterministic (noun, adj, object) facts.

    In-dist nouns receive ``n_pairs_per_noun`` distinct facts each. Held-out
    nouns receive one fact each, drawn from the same vocabulary but using
    (adj, obj) tuples that do not collide with any in-dist fact for the same
    noun. The full set is unique by (noun, adj, obj) tuple.

    Determinism: index-based assignment from the seed-shuffled vocab so the
    same seed always produces the same facts.
    """
    rng = random.Random(int(seed))
    adj_pool = list(ADJECTIVES[: int(n_adjectives)])
    obj_pool = list(OBJECTS[: int(n_objects)])
    rng.shuffle(adj_pool)
    rng.shuffle(obj_pool)

    held = frozenset(held_out_nouns)
    in_dist = [n_ for n_ in NOUNS if n_ not in held]
    held_list = [n_ for n_ in NOUNS if n_ in held]

    facts: list[Fact] = []
    used: set[tuple[str, str]] = set()

    for noun_idx, noun in enumerate(in_dist):
        for k in range(n_pairs_per_noun):
            adj = adj_pool[(noun_idx + k) % len(adj_pool)]
            obj = obj_pool[(noun_idx + 2 * k + 1) % len(obj_pool)]
            facts.append(Fact(noun=noun, adj=adj, obj=obj, held_out=False))
            used.add((adj, obj))

    for noun_idx, noun in enumerate(held_list):
        for attempt in range(len(adj_pool) * len(obj_pool)):
            ai = (noun_idx * 7 + attempt) % len(adj_pool)
            oi = (noun_idx * 11 + attempt * 3) % len(obj_pool)
            adj = adj_pool[ai]
            obj = obj_pool[oi]
            if (adj, obj) not in used:
                facts.append(Fact(noun=noun, adj=adj, obj=obj, held_out=True))
                used.add((adj, obj))
                break
        else:
            raise ValueError(
                f"could not assign held-out fact for {noun!r} — pool exhausted"
            )

    return tuple(facts)


def build_corpus(
    *,
    seed: int,
    n_pairs_per_noun: int = DEFAULT_N_PAIRS_PER_NOUN,
    reps: int = DEFAULT_REPS,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    held_out_nouns: Sequence[str] = DEFAULT_HELD_OUT_NOUNS,
    n_adjectives: int = 10,
    n_objects: int = 15,
) -> CorpusSpec:
    """Assemble training corpus + dedup'd eval prompts + per-noun pair lookup.

    - In-dist facts repeated ``reps`` times (held-out facts NEVER appear).
    - Distractor sentences use the same vocab in non-fact frames so the model
      cannot key on raw token co-occurrence alone.
    - ``test_facts`` is deduplicated by noun (one prompt per noun) — for
      multi-pair-per-noun corpora the prompt is the same regardless of which
      specific (adj, obj) it carries.
    - ``trained_pairs_by_noun`` lets the grader credit any trained pair for
      that noun (combinatorial retrieval, not lexical-specific).
    """
    facts = build_facts(
        seed=seed,
        n_pairs_per_noun=n_pairs_per_noun,
        held_out_nouns=held_out_nouns,
        n_adjectives=n_adjectives,
        n_objects=n_objects,
    )
    in_dist_facts = [f for f in facts if not f.held_out]
    held_facts = [f for f in facts if f.held_out]

    rng = random.Random(int(seed) + 1)
    train_sentences: list[str] = []
    for fact in in_dist_facts:
        for _ in range(int(reps)):
            train_sentences.append(fact.sentence())

    distractor_pool: list[str] = []
    all_nouns = list(NOUNS)
    adj_subset = list(ADJECTIVES[: int(n_adjectives)])
    obj_subset = list(OBJECTS[: int(n_objects)])
    for frame in DISTRACTOR_NOUN_FRAMES:
        for noun in all_nouns:
            distractor_pool.append(frame.format(noun=noun))
    for frame in DISTRACTOR_ADJ_FRAMES:
        for adj in adj_subset:
            distractor_pool.append(frame.format(adj=adj))
    for frame in DISTRACTOR_OBJECT_FRAMES:
        for obj in obj_subset:
            distractor_pool.append(frame.format(object=obj))
    rng.shuffle(distractor_pool)
    train_sentences.extend(distractor_pool[: int(n_distractors)])

    rng.shuffle(train_sentences)

    # Dedup eval prompts by noun: one Fact per unique noun.
    seen_nouns: set[str] = set()
    eval_facts: list[Fact] = []
    for f in (*in_dist_facts, *held_facts):
        if f.noun not in seen_nouns:
            eval_facts.append(f)
            seen_nouns.add(f.noun)

    pairs_by_noun: dict[str, frozenset[tuple[str, str]]] = {}
    for f in in_dist_facts:
        existing = pairs_by_noun.get(f.noun, frozenset())
        pairs_by_noun[f.noun] = existing | {(f.adj, f.obj)}
    for f in held_facts:
        # Held-out: the single (adj, obj) is reserved as the breakthrough target.
        pairs_by_noun[f.noun] = frozenset({(f.adj, f.obj)})

    test_facts = tuple(eval_facts)
    return CorpusSpec(
        facts=facts,
        train_sentences=tuple(train_sentences),
        test_facts=test_facts,
        trained_pairs_by_noun=pairs_by_noun,
        seed=int(seed),
    )


def query_prompt(fact: Fact) -> str:
    """Eval prompt: ``the {noun} had a`` — model must produce (adj, object)."""
    return f"the {fact.noun} had a"


__all__ = [
    "OBJECTS",
    "DEFAULT_HELD_OUT_NOUNS",
    "DEFAULT_N_PAIRS_PER_NOUN",
    "DEFAULT_REPS",
    "DEFAULT_N_DISTRACTORS",
    "TARGET_FRAME",
    "Fact",
    "CorpusSpec",
    "build_facts",
    "build_corpus",
    "query_prompt",
]
