"""Nano controlled-holdout continuation probe.

WARNING — empirically validated failure-mode (2026-05-03):
This probe does NOT track ``wikitext_perplexity`` (Spearman ≈ 0 on top-50)
and gives high scores to architectures with PPL > 600 (composite < 30).
It measures *speed of overfitting a small curated synthetic corpus*, not
language capability. Do not wire as a leaderboard metric without an
audit of the entire ``language_control_*`` family.  See
``research/reports/nano_controlled_holdout_failuretest_*.json`` for evidence.

What the probe DOES measure reliably:
  * Whether an architecture has any sequence-mixing capability (no-mixer
    baselines correctly score ≈ 0).
  * Speed of overfitting a 4-class curated controlled-vocabulary corpus
    with held-out (noun, verb) and (adj, noun) combinations.

Design.  Class-aware curated real-word corpus.  Three eval buckets:

  * ``seen``           — (noun, verb) pair the model trained on
  * ``held_out_pair``  — class-aligned (noun, verb) pair that never
                          co-occurred in training; in-class generalisation
  * ``compositional``  — prefix ``the {adj} {noun}`` where the (adj, noun)
                          combination was never trained; multi-slot
                          composition

All three buckets report 4-way forced-choice accuracy.
"""

from __future__ import annotations

import gc
import logging
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import torch
import torch.nn as nn

from ._controlled_probe_utils import encode_controlled_text, next_token_loss
from .choice_scoring import concat_choice_tokens, grouped_choice_scores
from .utils import _get_tiktoken_encoder, clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

NANO_CONTROLLED_HOLDOUT_METRIC_VERSION = "nano_controlled_holdout_v1"

_DEFAULT_ACTIVE_VOCAB = 1000
_DEFAULT_TRAIN_STEPS = 100
_DEFAULT_BATCH = 32
_DEFAULT_LR = 1e-3
_DEFAULT_EVAL_PER_BUCKET = 24
_DEFAULT_HOLD_OUT_FRAC = 0.25
_DEFAULT_N_CLASSES = 4
_TIMEOUT_S = 120.0
_MAX_SEQ_LEN = 12
_PAD = 0
_BUCKETS: tuple[str, ...] = ("seen", "held_out_pair", "compositional")


# Curated role words — class-grouped so we can build a deterministic class-aware
# corpus.  Each role contributes the same number of words per class so distractor
# pools are balanced.  Words are short and tokenize cleanly under both ``gpt2``
# and ``cl100k_base`` BPE.

_NOUN_CLASSES: tuple[tuple[str, ...], ...] = (
    # animals
    ("dog", "cat", "horse", "bird", "mouse", "fox", "tiger", "rabbit"),
    # people
    ("man", "woman", "boy", "girl", "child", "baby", "friend", "student"),
    # workers
    ("cook", "doctor", "teacher", "driver", "farmer", "guard", "singer", "writer"),
    # objects
    ("car", "ship", "train", "clock", "lamp", "phone", "book", "chair"),
)

_VERB_CLASSES: tuple[tuple[str, ...], ...] = (
    # motion
    ("ran", "jumped", "walked", "swam", "climbed", "flew", "drove", "rolled"),
    # voice
    ("sang", "spoke", "laughed", "shouted", "whispered", "called", "cried", "smiled"),
    # work
    ("cooked", "wrote", "painted", "built", "fixed", "washed", "carried", "served"),
    # rest
    ("slept", "sat", "rested", "waited", "stood", "watched", "looked", "stayed"),
)

_ADJ_CLASSES: tuple[tuple[str, ...], ...] = (
    # size
    ("small", "big", "tall", "short", "wide", "thin", "huge", "tiny"),
    # mood
    ("happy", "sad", "kind", "brave", "calm", "angry", "proud", "shy"),
    # state
    ("clean", "dirty", "fresh", "warm", "cold", "soft", "hard", "old"),
    # color
    ("green", "red", "blue", "white", "black", "brown", "bright", "dark"),
)

_ADV_CLASSES: tuple[tuple[str, ...], ...] = (
    # speed
    ("fast", "slowly", "quickly", "again"),
    # voice
    ("loudly", "quietly", "softly", "boldly"),
    # quality
    ("well", "badly", "carefully", "easily"),
    # rest
    ("daily", "often", "still", "alone"),
)


@dataclass(frozen=True, slots=True)
class ControlledHoldoutItem:
    bucket: str
    prefix: str
    correct: str
    distractors: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class NanoControlledHoldoutCorpus:
    active_vocab_size: int
    n_classes: int
    vocabulary: tuple[str, ...]
    nouns: tuple[tuple[str, ...], ...]
    verbs: tuple[tuple[str, ...], ...]
    adjectives: tuple[tuple[str, ...], ...]
    adverbs: tuple[tuple[str, ...], ...]
    train_sentences: tuple[str, ...]
    eval_items: tuple[ControlledHoldoutItem, ...]
    n_train_pairs_seen: int
    n_train_pairs_held_out: int
    tokenizer: str
    tiktoken_encoding: str
    seed: int


@dataclass(slots=True)
class NanoControlledHoldoutResult:
    score: float
    seen_acc: float
    held_out_pair_acc: float
    compositional_acc: float
    generalisation_gap: float  # seen_acc - held_out_pair_acc; high = pure memorisation
    active_vocab_size: int
    n_train_steps: int
    n_train_sentences: int
    n_eval_per_bucket: dict[str, int]
    chance: float
    elapsed_ms: float
    status: str
    tokenizer: str
    tiktoken_encoding: str
    metric_version: str = NANO_CONTROLLED_HOLDOUT_METRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "nano_controlled_holdout_metric_version": self.metric_version,
            "nano_controlled_holdout_score": self.score,
            "nano_controlled_holdout_seen_acc": self.seen_acc,
            "nano_controlled_holdout_held_out_pair_acc": self.held_out_pair_acc,
            "nano_controlled_holdout_compositional_acc": self.compositional_acc,
            "nano_controlled_holdout_generalisation_gap": self.generalisation_gap,
            "nano_controlled_holdout_active_vocab_size": self.active_vocab_size,
            "nano_controlled_holdout_train_steps": self.n_train_steps,
            "nano_controlled_holdout_n_train_sentences": self.n_train_sentences,
            "nano_controlled_holdout_n_eval_seen": self.n_eval_per_bucket.get(
                "seen", 0
            ),
            "nano_controlled_holdout_n_eval_held_out_pair": self.n_eval_per_bucket.get(
                "held_out_pair", 0
            ),
            "nano_controlled_holdout_n_eval_compositional": self.n_eval_per_bucket.get(
                "compositional", 0
            ),
            "nano_controlled_holdout_chance": self.chance,
            "nano_controlled_holdout_elapsed_ms": self.elapsed_ms,
            "nano_controlled_holdout_status": self.status,
            "nano_controlled_holdout_tokenizer": self.tokenizer,
            "nano_controlled_holdout_tiktoken_encoding": self.tiktoken_encoding,
        }


def _dedupe_lower(words: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        cleaned = word.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)


@lru_cache(maxsize=8)
def _tokenizer_word_bank(
    target_size: int,
    *,
    tiktoken_encoding: str,
    max_token_id: int,
) -> tuple[str, ...]:
    """Return up to ``target_size`` short alphabetic words from the given BPE
    encoding's single-token vocabulary, padded with curated role words first."""
    enc = _get_tiktoken_encoder(tiktoken_encoding)
    seed_words: list[str] = []
    for groups in (_NOUN_CLASSES, _VERB_CLASSES, _ADJ_CLASSES, _ADV_CLASSES):
        for group in groups:
            seed_words.extend(group)
    words: list[str] = list(_dedupe_lower(seed_words))
    seen = set(words)
    cap = max(int(target_size), len(words))
    for token_id in range(min(int(max_token_id), int(enc.n_vocab))):
        try:
            raw = enc.decode_single_token_bytes(token_id)
            decoded = raw.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        word = decoded.strip().lower()
        if not (2 <= len(word) <= 12 and word.isalpha()):
            continue
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= cap:
            break
    return tuple(words[:cap])


def _encode_text(
    text: str,
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[int, ...]:
    return encode_controlled_text(
        text,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )


def _filter_encodable(
    words: Sequence[str],
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[str, ...]:
    out: list[str] = []
    for word in _dedupe_lower(words):
        try:
            _encode_text(
                " " + word,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                tiktoken_encoding=tiktoken_encoding,
            )
        except ValueError:
            continue
        out.append(word)
    return tuple(out)


def _trim_to_balanced_classes(
    classes: Sequence[Sequence[str]],
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
    max_per_class: int,
) -> tuple[tuple[str, ...], ...]:
    filtered: list[tuple[str, ...]] = []
    for group in classes:
        kept = _filter_encodable(
            group,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
        )
        if not kept:
            raise ValueError("class has no encodable words for current tokenizer")
        filtered.append(kept)
    n_per_class = min(max_per_class, *(len(group) for group in filtered))
    if n_per_class < 2:
        raise ValueError("not enough encodable role words to populate classes")
    return tuple(group[:n_per_class] for group in filtered)


def _select_eval_pairs(
    pool: Sequence[tuple[int, int]],
    rng: random.Random,
    n_target: int,
) -> list[tuple[int, int]]:
    if not pool:
        return []
    items = list(pool)
    rng.shuffle(items)
    return items[: int(min(n_target, len(items)))]


@dataclass(frozen=True, slots=True)
class _RoleTables:
    n_classes: int
    n_per: int
    n_per_adj: int
    n_per_adv: int
    nouns: tuple[tuple[str, ...], ...]
    verbs: tuple[tuple[str, ...], ...]
    adjectives: tuple[tuple[str, ...], ...]
    adverbs: tuple[tuple[str, ...], ...]

    def noun(self, idx: int) -> str:
        return self.nouns[idx // self.n_per][idx % self.n_per]

    def verb(self, idx: int) -> str:
        return self.verbs[idx // self.n_per][idx % self.n_per]

    def adj(self, idx: int) -> str:
        return self.adjectives[idx // self.n_per_adj][idx % self.n_per_adj]

    def adverb_for_class(self, cls: int, salt: int) -> str:
        return self.adverbs[cls][salt % self.n_per_adv]


def _select_role_tables(
    n_classes: int,
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> _RoleTables:
    nouns = _trim_to_balanced_classes(
        _NOUN_CLASSES[:n_classes],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        max_per_class=8,
    )
    verbs = _trim_to_balanced_classes(
        _VERB_CLASSES[:n_classes],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        max_per_class=8,
    )
    adjectives = _trim_to_balanced_classes(
        _ADJ_CLASSES[:n_classes],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        max_per_class=8,
    )
    adverbs = _trim_to_balanced_classes(
        _ADV_CLASSES[:n_classes],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        max_per_class=4,
    )
    return _RoleTables(
        n_classes=n_classes,
        n_per=len(nouns[0]),
        n_per_adj=len(adjectives[0]),
        n_per_adv=len(adverbs[0]),
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives,
        adverbs=adverbs,
    )


def _resolve_vocabulary(
    tables: _RoleTables,
    *,
    active_vocab_size: int,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[str, ...]:
    tok = (tokenizer or "tiktoken").strip().lower()
    if tok in ("byte", "bytes"):
        return tuple(
            _dedupe_lower(
                tuple(
                    w
                    for group in tables.nouns
                    + tables.verbs
                    + tables.adjectives
                    + tables.adverbs
                    for w in group
                )
                + ("the", "a", "is")
            )
        )
    enc_name = "gpt2" if tok == "gpt2" else tiktoken_encoding
    return _tokenizer_word_bank(
        int(active_vocab_size),
        tiktoken_encoding=enc_name,
        max_token_id=int(vocab_size),
    )


def _partition_pairs(
    rng: random.Random,
    *,
    n_classes: int,
    n_per_left: int,
    n_per_right: int,
    hold_out_frac: float,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Enumerate all in-class (left_id, right_id) pairs, shuffle, and split
    by ``hold_out_frac``.  Returns ``(seen_list, held_out_set)``."""
    pairs: list[tuple[int, int]] = []
    for cls in range(n_classes):
        for li in range(n_per_left):
            for rj in range(n_per_right):
                pairs.append((cls * n_per_left + li, cls * n_per_right + rj))
    rng.shuffle(pairs)
    n_held = max(1, int(round(len(pairs) * float(hold_out_frac))))
    return pairs[n_held:], set(pairs[:n_held])


def _build_train_sentences(
    tables: _RoleTables,
    *,
    seen_pairs: Sequence[tuple[int, int]],
    seen_an: Sequence[tuple[int, int]],
) -> tuple[str, ...]:
    sentences: list[str] = []
    for noun_id, verb_id in seen_pairs:
        cls = noun_id // tables.n_per
        adv = tables.adverb_for_class(cls, noun_id + verb_id)
        sentences.append(f"the {tables.noun(noun_id)} {tables.verb(verb_id)} {adv}")
        sentences.append(f"the {tables.noun(noun_id)} {tables.verb(verb_id)}")
    for adj_id, noun_id in sorted(seen_an):
        cls = adj_id // tables.n_per_adj
        verb_pool = [v for (n, v) in seen_pairs if n // tables.n_per == cls] or [
            cls * tables.n_per + (adj_id % tables.n_per)
        ]
        verb_id = verb_pool[(adj_id + noun_id) % len(verb_pool)]
        sentences.append(
            f"the {tables.adj(adj_id)} {tables.noun(noun_id)} {tables.verb(verb_id)}"
        )
    return tuple(sentences)


def _distractors_for(
    tables: _RoleTables,
    *,
    cls: int,
    verb_id: int,
    adv_word: str,
) -> tuple[str, str, str]:
    wrong_classes = [c for c in range(tables.n_classes) if c != cls]
    outs: list[str] = []
    for offset, other_cls in enumerate(wrong_classes[:3]):
        wrong_verb_id = other_cls * tables.n_per + (
            (verb_id + offset + 1) % tables.n_per
        )
        wrong_adv_word = (
            tables.adverb_for_class(other_cls, verb_id + offset)
            if offset == 2
            else adv_word
        )
        outs.append(f" {tables.verb(wrong_verb_id)} {wrong_adv_word}")
    while len(outs) < 3:
        outs.append(outs[-1])
    return tuple(outs)  # type: ignore[return-value]


def _pair_item(
    tables: _RoleTables,
    noun_id: int,
    verb_id: int,
    bucket: str,
) -> ControlledHoldoutItem:
    cls = noun_id // tables.n_per
    adv_word = tables.adverb_for_class(cls, noun_id + verb_id)
    return ControlledHoldoutItem(
        bucket=bucket,
        prefix=f"the {tables.noun(noun_id)}",
        correct=f" {tables.verb(verb_id)} {adv_word}",
        distractors=_distractors_for(
            tables, cls=cls, verb_id=verb_id, adv_word=adv_word
        ),
    )


def _build_eval_items(
    tables: _RoleTables,
    rng: random.Random,
    *,
    seen_pairs: Sequence[tuple[int, int]],
    held_out_pairs: set[tuple[int, int]],
    held_out_an: set[tuple[int, int]],
    n_eval_per_bucket: int,
) -> tuple[ControlledHoldoutItem, ...]:
    items: list[ControlledHoldoutItem] = []
    seen_pool = [
        (n, v) for (n, v) in seen_pairs if (n // tables.n_per) == (v // tables.n_per)
    ]
    for noun_id, verb_id in _select_eval_pairs(seen_pool, rng, int(n_eval_per_bucket)):
        items.append(_pair_item(tables, noun_id, verb_id, "seen"))

    held_out_pool = [
        (n, v)
        for (n, v) in held_out_pairs
        if (n // tables.n_per) == (v // tables.n_per)
    ]
    for noun_id, verb_id in _select_eval_pairs(
        held_out_pool, rng, int(n_eval_per_bucket)
    ):
        items.append(_pair_item(tables, noun_id, verb_id, "held_out_pair"))

    held_out_an_list = [
        (a, n)
        for (a, n) in held_out_an
        if (a // tables.n_per_adj) == (n // tables.n_per)
    ]
    rng.shuffle(held_out_an_list)
    for adj_id, noun_id in held_out_an_list[: int(n_eval_per_bucket)]:
        items.append(_compositional_item(tables, adj_id, noun_id, held_out_pairs))
    return tuple(items)


def _compositional_item(
    tables: _RoleTables,
    adj_id: int,
    noun_id: int,
    held_out_pairs: set[tuple[int, int]],
) -> ControlledHoldoutItem:
    cls = noun_id // tables.n_per
    candidate_verbs = [
        v for (n, v) in held_out_pairs if n == noun_id and (v // tables.n_per) == cls
    ]
    if not candidate_verbs:
        candidate_verbs = [cls * tables.n_per + ((noun_id + adj_id + 1) % tables.n_per)]
    verb_id = candidate_verbs[(adj_id + noun_id) % len(candidate_verbs)]
    adv_word = tables.adverb_for_class(cls, noun_id + verb_id + adj_id)
    return ControlledHoldoutItem(
        bucket="compositional",
        prefix=f"the {tables.adj(adj_id)} {tables.noun(noun_id)}",
        correct=f" {tables.verb(verb_id)} {adv_word}",
        distractors=_distractors_for(
            tables, cls=cls, verb_id=verb_id, adv_word=adv_word
        ),
    )


def build_nano_controlled_holdout_corpus(
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    vocab_size: int,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "gpt2",
    n_eval_per_bucket: int = _DEFAULT_EVAL_PER_BUCKET,
    hold_out_frac: float = _DEFAULT_HOLD_OUT_FRAC,
    n_classes: int = _DEFAULT_N_CLASSES,
    seed: int = 42,
) -> NanoControlledHoldoutCorpus:
    """Build a deterministic class-aware nano-HellaSwag corpus.

    Returns training sentences (each is a short real-word sentence the model
    will train next-token prediction on) plus evaluation items grouped into
    three difficulty buckets.  ``hold_out_frac`` controls what fraction of
    in-class (noun, verb) and (adj, noun) combinations are reserved for
    eval-only.
    """
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    tables = _select_role_tables(
        n_classes,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    vocabulary = _resolve_vocabulary(
        tables,
        active_vocab_size=active_vocab_size,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    rng = random.Random(int(seed))
    seen_pairs_list, held_out_pairs = _partition_pairs(
        rng,
        n_classes=n_classes,
        n_per_left=tables.n_per,
        n_per_right=tables.n_per,
        hold_out_frac=hold_out_frac,
    )
    seen_an_list, held_out_an = _partition_pairs(
        rng,
        n_classes=n_classes,
        n_per_left=tables.n_per_adj,
        n_per_right=tables.n_per,
        hold_out_frac=hold_out_frac,
    )
    train_sentences = _build_train_sentences(
        tables, seen_pairs=seen_pairs_list, seen_an=seen_an_list
    )
    eval_items = _build_eval_items(
        tables,
        rng,
        seen_pairs=seen_pairs_list,
        held_out_pairs=held_out_pairs,
        held_out_an=held_out_an,
        n_eval_per_bucket=n_eval_per_bucket,
    )
    return NanoControlledHoldoutCorpus(
        active_vocab_size=int(active_vocab_size),
        n_classes=n_classes,
        vocabulary=tuple(vocabulary[: int(active_vocab_size)]),
        nouns=tables.nouns,
        verbs=tables.verbs,
        adjectives=tables.adjectives,
        adverbs=tables.adverbs,
        train_sentences=train_sentences,
        eval_items=eval_items,
        n_train_pairs_seen=len(seen_pairs_list),
        n_train_pairs_held_out=len(held_out_pairs),
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        seed=int(seed),
    )


def _sentence_tokens(
    sentence: str,
    *,
    vocab_size: int,
    corpus: NanoControlledHoldoutCorpus,
) -> tuple[int, ...]:
    return _encode_text(
        sentence,
        vocab_size=vocab_size,
        tokenizer=corpus.tokenizer,
        tiktoken_encoding=corpus.tiktoken_encoding,
    )


def _make_train_batch(
    corpus: NanoControlledHoldoutCorpus,
    *,
    vocab_size: int,
    batch_size: int,
    device: str,
    rng: torch.Generator,
) -> torch.Tensor:
    if not corpus.train_sentences:
        raise ValueError("train_sentences is empty")
    indices = torch.randint(
        0, len(corpus.train_sentences), (int(batch_size),), generator=rng
    )
    seqs = [
        _sentence_tokens(
            corpus.train_sentences[int(idx.item())],
            vocab_size=vocab_size,
            corpus=corpus,
        )
        for idx in indices
    ]
    max_len = max(len(seq) for seq in seqs)
    batch = torch.full(
        (len(seqs), max_len),
        _PAD,
        dtype=torch.long,
        device=torch.device(device),
    )
    for i, seq in enumerate(seqs):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return batch


def _next_token_loss(
    model: nn.Module, batch: torch.Tensor, *, vocab_size: int
) -> torch.Tensor:
    return next_token_loss(model, batch, vocab_size=vocab_size, pad_id=_PAD)


def _bucket_choice_accuracy(
    model: nn.Module,
    items: Sequence[ControlledHoldoutItem],
    corpus: NanoControlledHoldoutCorpus,
    *,
    vocab_size: int,
    device: str,
) -> float:
    if not items:
        return 0.0
    grouped_sequences: list[list[tuple[int, ...]]] = []
    grouped_starts: list[list[int]] = []
    for item in items:
        prefix_tokens = _sentence_tokens(
            item.prefix, vocab_size=vocab_size, corpus=corpus
        )
        choices = (item.correct, *item.distractors)
        seqs: list[tuple[int, ...]] = []
        starts: list[int] = []
        for choice in choices:
            choice_tokens = _sentence_tokens(
                choice, vocab_size=vocab_size, corpus=corpus
            )
            seq, start = concat_choice_tokens(
                prefix_tokens, choice_tokens, max_seq_len=_MAX_SEQ_LEN
            )
            seqs.append(tuple(int(x) for x in seq))
            starts.append(int(start))
        grouped_sequences.append(seqs)
        grouped_starts.append(starts)
    scores = grouped_choice_scores(
        model,
        grouped_sequences,
        grouped_starts,
        vocab_size=vocab_size,
        device=device,
    )
    if not scores:
        return 0.0
    correct = sum(
        1 for row in scores if row and max(range(len(row)), key=row.__getitem__) == 0
    )
    return correct / len(scores)


def evaluate_nano_controlled_holdout(
    model: nn.Module,
    corpus: NanoControlledHoldoutCorpus,
    *,
    vocab_size: int,
    device: str,
) -> dict[str, float]:
    """Return per-bucket 4-way HellaSwag-style accuracy on the prepared corpus."""
    model.eval()
    out: dict[str, float] = {}
    for bucket in _BUCKETS:
        items = [item for item in corpus.eval_items if item.bucket == bucket]
        out[bucket] = _bucket_choice_accuracy(
            model, items, corpus, vocab_size=vocab_size, device=device
        )
    return out


def _empty_result(
    *,
    active_vocab_size: int,
    status: str,
    tokenizer: str,
    tiktoken_encoding: str,
    elapsed_ms: float,
) -> NanoControlledHoldoutResult:
    return NanoControlledHoldoutResult(
        score=0.0,
        seen_acc=0.0,
        held_out_pair_acc=0.0,
        compositional_acc=0.0,
        generalisation_gap=0.0,
        active_vocab_size=int(active_vocab_size),
        n_train_steps=0,
        n_train_sentences=0,
        n_eval_per_bucket={b: 0 for b in _BUCKETS},
        chance=0.25,
        elapsed_ms=round(elapsed_ms, 1),
        status=status,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )


def _train_one_step(
    model: nn.Module,
    corpus: NanoControlledHoldoutCorpus,
    opt: torch.optim.Optimizer,
    *,
    vocab_size: int,
    batch_size: int,
    device: str,
    rng: torch.Generator,
) -> str:
    """One forward/backward step.  Returns ``"ok"`` or ``"non_finite_loss"``."""
    batch = _make_train_batch(
        corpus,
        vocab_size=vocab_size,
        batch_size=batch_size,
        device=device,
        rng=rng,
    )
    opt.zero_grad(set_to_none=True)
    loss = _next_token_loss(model, batch, vocab_size=vocab_size)
    if not torch.isfinite(loss):
        return "non_finite_loss"
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return "ok"


def _train_probe(
    model: nn.Module,
    corpus: NanoControlledHoldoutCorpus,
    *,
    vocab_size: int,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    seed: int,
) -> tuple[int, str]:
    """Train ``model`` in place; caller is responsible for snapshot/restore."""
    model.train()
    opt = make_adamw(model.parameters(), lr=lr)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    steps = 0
    status = "ok"
    for step in range(int(n_train_steps)):
        if time.perf_counter() > deadline:
            status = "timeout"
            break
        step_status = _train_one_step(
            model,
            corpus,
            opt,
            vocab_size=vocab_size,
            batch_size=batch_size,
            device=device,
            rng=rng,
        )
        if step_status != "ok":
            status = step_status
            break
        steps = step + 1
    return steps, status


def _aggregate_score(bucket_acc: dict[str, float]) -> tuple[float, float, float, float]:
    """Compute (weighted-score, seen, held_out_pair, compositional).

    Weights generalisation and composition over memorisation: held-out buckets
    each get 0.4, seen only 0.2.  Chance-baseline subtracted so random→0.0.
    """
    seen = float(bucket_acc.get("seen", 0.0))
    held = float(bucket_acc.get("held_out_pair", 0.0))
    comp = float(bucket_acc.get("compositional", 0.0))
    chance = 0.25
    weighted = 0.2 * seen + 0.4 * held + 0.4 * comp
    score = max(0.0, (weighted - chance) / max(1.0 - chance, 1e-6))
    return score, seen, held, comp


def _train_and_score(
    model: nn.Module,
    corpus: NanoControlledHoldoutCorpus,
    *,
    vocab_size: int,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    seed: int,
) -> tuple[int, str, dict[str, float]]:
    """Snapshot, train, eval, restore.  Returns (steps, status, bucket_acc)."""
    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        steps, status = _train_probe(
            model,
            corpus,
            vocab_size=vocab_size,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            deadline=deadline,
            seed=seed,
        )
        if status == "non_finite_loss":
            bucket_acc = {b: 0.0 for b in _BUCKETS}
        else:
            bucket_acc = evaluate_nano_controlled_holdout(
                model, corpus, vocab_size=vocab_size, device=device
            )
        return steps, status, bucket_acc
    finally:
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def _finalize_result(
    corpus: NanoControlledHoldoutCorpus,
    *,
    bucket_acc: dict[str, float],
    steps: int,
    status: str,
    active_vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
    elapsed_ms: float,
) -> NanoControlledHoldoutResult:
    score, seen, held, comp = _aggregate_score(bucket_acc)
    counts = {
        bucket: sum(1 for item in corpus.eval_items if item.bucket == bucket)
        for bucket in _BUCKETS
    }
    return NanoControlledHoldoutResult(
        score=round(float(score), 4),
        seen_acc=round(seen, 4),
        held_out_pair_acc=round(held, 4),
        compositional_acc=round(comp, 4),
        generalisation_gap=round(seen - held, 4),
        active_vocab_size=int(active_vocab_size),
        n_train_steps=int(steps),
        n_train_sentences=len(corpus.train_sentences),
        n_eval_per_bucket=counts,
        chance=0.25,
        elapsed_ms=round(elapsed_ms, 1),
        status=status,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )


def nano_controlled_holdout_probe(
    model: nn.Module,
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    n_eval_per_bucket: int = _DEFAULT_EVAL_PER_BUCKET,
    hold_out_frac: float = _DEFAULT_HOLD_OUT_FRAC,
    n_classes: int = _DEFAULT_N_CLASSES,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "gpt2",
) -> NanoControlledHoldoutResult:
    """Train on real-word nano sentences, then evaluate three held-out buckets.

    Caller's model state is preserved.  Does not write leaderboard or
    program-result fields; use it for calibration before any wiring.
    """
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    vocab_size = int(getattr(model, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return _empty_result(
            active_vocab_size=active_vocab_size,
            status="missing_vocab_size",
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
    try:
        corpus = build_nano_controlled_holdout_corpus(
            active_vocab_size=active_vocab_size,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
            n_eval_per_bucket=n_eval_per_bucket,
            hold_out_frac=hold_out_frac,
            n_classes=n_classes,
            seed=seed,
        )
    except ValueError as exc:
        logger.debug("nano controlled-holdout corpus unavailable: %s", exc)
        return _empty_result(
            active_vocab_size=active_vocab_size,
            status="model_vocab_too_small",
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    steps, status, bucket_acc = _train_and_score(
        model,
        corpus,
        vocab_size=vocab_size,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        deadline=deadline,
        seed=seed,
    )
    return _finalize_result(
        corpus,
        bucket_acc=bucket_acc,
        steps=steps,
        status=status,
        active_vocab_size=active_vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


__all__ = [
    "NANO_CONTROLLED_HOLDOUT_METRIC_VERSION",
    "ControlledHoldoutItem",
    "NanoControlledHoldoutCorpus",
    "NanoControlledHoldoutResult",
    "build_nano_controlled_holdout_corpus",
    "evaluate_nano_controlled_holdout",
    "nano_controlled_holdout_probe",
]
