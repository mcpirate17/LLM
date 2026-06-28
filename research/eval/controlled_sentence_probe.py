"""Controlled sentence-shape diagnostic for nano-scale language capability.

This probe is intentionally a language-shape and step-curve diagnostic, not the
held-out compositional test.  It uses real word tokens in short multi-token
forms such as ``the dog ran fast`` plus mined bAbI / benchmark snippets to
check sentence surface coverage, training-step behavior, choice scoring, and
minimal-pair sanity.  ``nano_blimp_v3`` owns the slot/POS and binding holdout
questions.
"""

from __future__ import annotations

import gc
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from ._real_word_vocab import (
    REAL_WORD_ADJECTIVES,
    REAL_WORD_ADVERBS,
    REAL_WORD_BABI_CORE_WORDS,
    REAL_WORD_BASE_VERBS,
    REAL_WORD_FUNCTION_WORDS,
    REAL_WORD_NOUNS,
    REAL_WORD_VERBS,
)
from .choice_scoring import concat_choice_tokens, grouped_choice_scores
from ._controlled_probe_utils import (
    dedupe_lower_words as _dedupe,
    encode_controlled_text as _encode_text,
    next_token_loss,
)
from .utils import _get_tiktoken_encoder, clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

CONTROLLED_SENTENCE_METRIC_VERSION = "controlled_sentence_v2"
CONTROLLED_SENTENCE_PROBE_ROLE = "language_shape_diagnostic"

_DEFAULT_ACTIVE_VOCAB = 1000
_DEFAULT_TRAIN_STEPS = 100
_DEFAULT_BATCH = 32
_DEFAULT_LR = 1e-3
_DEFAULT_EVAL_ITEMS = 64
_TIMEOUT_S = 90.0
_MAX_SEQ_LEN = 12
_MAX_TEXT_WORDS = 6
_PAD = 0
_HELLASWAG_CACHE_FILE = (
    Path.home() / ".cache" / "aria" / "hellaswag" / "validation.json"
)
_BLIMP_CACHE_FILE = Path.home() / ".cache" / "aria" / "blimp" / "all_subtasks.json"
_CONTROLLED_SENTENCE_CACHE_DIR = Path.home() / ".cache" / "aria" / "controlled_sentence"
_BABI_CACHE_FILE = _CONTROLLED_SENTENCE_CACHE_DIR / "babi_qa_sentences.json"
_WORD_RE = re.compile(r"[a-z]+")
_BABI_CONTEXT_RE = re.compile(r"Context:\n(.*?)\n\nQuestion:", re.DOTALL)
_BABI_MOVE_RE = re.compile(
    r"^([a-z]+) ((?:moved|journeyed|travelled|went)(?: back)?) to the ([a-z]+)$"
)

_NOUNS = REAL_WORD_NOUNS
_VERBS = REAL_WORD_VERBS
_ADJECTIVES = REAL_WORD_ADJECTIVES
_ADVERBS = REAL_WORD_ADVERBS
_FUNCTION_WORDS = REAL_WORD_FUNCTION_WORDS
_BABI_CORE_WORDS = REAL_WORD_BABI_CORE_WORDS
_DROP_WORDS = {"title", "substep", "substeps"}
_NOISY_WORDS = {"substep", "substeps", "http", "https", "www"}
_BASE_VERBS = REAL_WORD_BASE_VERBS


@dataclass(frozen=True, slots=True)
class SentenceProbeItem:
    prefix: str
    correct: str
    distractors: tuple[str, str, str]
    good_sentence: str
    bad_order_sentence: str
    bad_binding_sentence: str
    source: str = "curated"


@dataclass(frozen=True, slots=True)
class SentenceProbeCorpus:
    active_vocab_size: int
    vocabulary: tuple[str, ...]
    nouns: tuple[str, ...]
    verbs: tuple[str, ...]
    adjectives: tuple[str, ...]
    adverbs: tuple[str, ...]
    train_sentences: tuple[str, ...]
    eval_items: tuple[SentenceProbeItem, ...]
    blimp_pairs: tuple[tuple[str, str], ...]
    tokenizer: str
    tiktoken_encoding: str
    source_counts: dict[str, int]


@dataclass(slots=True)
class ControlledSentenceResult:
    score: float
    nano_hellaswag_acc: float
    nano_blimp_order_acc: float
    nano_blimp_binding_acc: float
    active_vocab_size: int
    n_train_steps: int
    n_train_sentences: int
    n_eval_items: int
    n_blimp_pairs: int
    chance: float
    elapsed_ms: float
    status: str
    tokenizer: str
    tiktoken_encoding: str
    metric_version: str = CONTROLLED_SENTENCE_METRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "controlled_sentence_metric_version": self.metric_version,
            "controlled_sentence_probe_role": CONTROLLED_SENTENCE_PROBE_ROLE,
            "controlled_sentence_score": self.score,
            "controlled_sentence_nano_hellaswag_acc": self.nano_hellaswag_acc,
            "controlled_sentence_nano_blimp_order_acc": self.nano_blimp_order_acc,
            "controlled_sentence_nano_blimp_binding_acc": self.nano_blimp_binding_acc,
            "controlled_sentence_active_vocab_size": self.active_vocab_size,
            "controlled_sentence_train_steps": self.n_train_steps,
            "controlled_sentence_n_train_sentences": self.n_train_sentences,
            "controlled_sentence_n_eval_items": self.n_eval_items,
            "controlled_sentence_n_blimp_pairs": self.n_blimp_pairs,
            "controlled_sentence_chance": self.chance,
            "controlled_sentence_elapsed_ms": self.elapsed_ms,
            "controlled_sentence_status": self.status,
            "controlled_sentence_tokenizer": self.tokenizer,
            "controlled_sentence_tiktoken_encoding": self.tiktoken_encoding,
        }


def _words(text: str) -> tuple[str, ...]:
    return tuple(
        word
        for word in _WORD_RE.findall((text or "").lower())
        if word not in _DROP_WORDS
    )


def _has_noisy_words(text: str) -> bool:
    return any(word in _NOISY_WORDS for word in _WORD_RE.findall((text or "").lower()))


def _truncate_words(text: str, *, max_words: int, from_end: bool = False) -> str:
    words = list(_words(text))
    if not words:
        return ""
    if from_end:
        words = words[-int(max_words) :]
    else:
        words = words[: int(max_words)]
    return " ".join(words)


def _all_words_allowed(text: str, allowed: set[str]) -> bool:
    words = _words(text)
    return bool(words) and all(word in allowed for word in words)


def _bad_order_sentence(sentence: str) -> str:
    words = list(_words(sentence))
    if len(words) < 3:
        return ""
    words[0], words[1] = words[1], words[0]
    return " ".join(words)


def _sentence_with_min_words(text: str, *, min_words: int = 3) -> str:
    words = _words(text)
    if len(words) < int(min_words):
        return ""
    return " ".join(words)


def _is_short_sentence(text: str, *, min_words: int = 3, max_words: int = 12) -> bool:
    words = _words(text)
    return (
        int(min_words) <= len(words) <= int(max_words)
        and not _has_noisy_words(text)
        and len(set(words)) >= min(len(words), 3)
    )


def _split_simple_sentences(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for chunk in re.split(r"[.!?]+", text or ""):
        sentence = _sentence_with_min_words(chunk, min_words=3)
        if sentence and not _has_noisy_words(sentence):
            out.append(sentence)
    return tuple(out)


def _extract_babi_context_sentences(prompt: str) -> tuple[str, ...]:
    matches = _BABI_CONTEXT_RE.findall(prompt or "")
    if not matches:
        return ()
    # LM-Polygraph rows include a demonstration context first; the final
    # context is the actual example.
    return _split_simple_sentences(matches[-1])


@lru_cache(maxsize=1)
def _cached_hellaswag_rows() -> tuple[dict[str, Any], ...]:
    if not _HELLASWAG_CACHE_FILE.exists():
        return ()
    try:
        data = json.loads(_HELLASWAG_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(row for row in data if isinstance(row, dict))


@lru_cache(maxsize=1)
def _cached_blimp_pairs() -> tuple[tuple[str, str], ...]:
    if not _BLIMP_CACHE_FILE.exists():
        return ()
    try:
        data = json.loads(_BLIMP_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    pairs: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for examples in data.values():
            if not isinstance(examples, list):
                continue
            for row in examples:
                if isinstance(row, dict):
                    good = str(row.get("good") or row.get("sentence_good") or "")
                    bad = str(row.get("bad") or row.get("sentence_bad") or "")
                    if good and bad:
                        pairs.append((good, bad))
    return tuple(pairs)


@lru_cache(maxsize=1)
def _cached_babi_sentences() -> tuple[str, ...]:
    if _BABI_CACHE_FILE.exists():
        try:
            data = json.loads(_BABI_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return tuple(str(x) for x in data if isinstance(x, str))
        except (OSError, json.JSONDecodeError):
            pass
    try:
        from datasets import load_dataset

        ds = load_dataset("LM-Polygraph/babi_qa", "continuation", split="train")
    except Exception as exc:  # noqa: BLE001 - optional network/cache source
        logger.debug("controlled sentence bAbI source unavailable: %s", exc)
        return ()

    seen: set[str] = set()
    sentences: list[str] = []
    for row in ds:
        for sentence in _extract_babi_context_sentences(str(row.get("input") or "")):
            if sentence in seen:
                continue
            seen.add(sentence)
            sentences.append(sentence)
    try:
        _CONTROLLED_SENTENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _BABI_CACHE_FILE.write_text(json.dumps(sentences), encoding="utf-8")
    except OSError:
        pass
    return tuple(sentences)


@lru_cache(maxsize=1)
def _benchmark_word_counts() -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for sentence in _cached_babi_sentences():
        counter.update(_words(sentence))
    for row in _cached_hellaswag_rows():
        counter.update(_words(str(row.get("ctx") or "")))
        for ending in row.get("endings") or ():
            counter.update(_words(str(ending)))
    for good, bad in _cached_blimp_pairs():
        counter.update(_words(good))
        counter.update(_words(bad))
    return tuple(counter.most_common())


def _babi_move_parts(sentence: str) -> tuple[str, str] | None:
    match = _BABI_MOVE_RE.match(sentence)
    if not match:
        return None
    subject, verb_phrase, location = match.groups()
    return f"{subject} {verb_phrase} to the", location


def _build_babi_items(
    *,
    allowed_words: set[str],
    n_eval_items: int,
    seed: int,
    require_allowed: bool = True,
) -> tuple[SentenceProbeItem, ...]:
    sentences = list(_cached_babi_sentences())
    if not sentences:
        return ()
    parsed: list[tuple[str, str, str]] = []
    locations: list[str] = []
    for sentence in sentences:
        parts = _babi_move_parts(sentence)
        if parts is None:
            continue
        prefix, location = parts
        if require_allowed and not _all_words_allowed(
            f"{prefix} {location}", allowed_words
        ):
            continue
        parsed.append((sentence, prefix, location))
        locations.append(location)
    location_pool = sorted(set(locations))
    if len(location_pool) < 4:
        return ()
    rng = random.Random(int(seed) + 101)
    rng.shuffle(parsed)
    items: list[SentenceProbeItem] = []
    for sentence, prefix, location in parsed:
        distractors = [loc for loc in location_pool if loc != location]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        bad_binding = f"{prefix} {distractors[0]}"
        bad_order = _bad_order_sentence(sentence)
        if not bad_order or bad_order == sentence:
            continue
        items.append(
            SentenceProbeItem(
                prefix=prefix,
                correct=f" {location}",
                distractors=tuple(f" {loc}" for loc in distractors[:3]),  # type: ignore[arg-type]
                good_sentence=sentence,
                bad_order_sentence=bad_order,
                bad_binding_sentence=bad_binding,
                source="babi_qa",
            )
        )
        if len(items) >= int(n_eval_items):
            break
    return tuple(items)


def _build_babi_train_sentences(
    *,
    allowed_words: set[str],
    max_sentences: int,
    seed: int,
    excluded_sentences: frozenset[str],
) -> tuple[str, ...]:
    sentences = list(_cached_babi_sentences())
    if not sentences:
        return ()
    rng = random.Random(int(seed) + 137)
    rng.shuffle(sentences)
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        clean = _sentence_with_min_words(sentence, min_words=3)
        if not clean or clean in seen or clean in excluded_sentences:
            continue
        if not _all_words_allowed(clean, allowed_words):
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= int(max_sentences):
            break
    return tuple(out)


@lru_cache(maxsize=8)
def _tokenizer_word_bank(
    active_vocab_size: int,
    *,
    tiktoken_encoding: str,
    max_token_id: int,
) -> tuple[str, ...]:
    enc = _get_tiktoken_encoder(tiktoken_encoding)
    benchmark_words = tuple(word for word, _count in _benchmark_word_counts())
    seed_words = _dedupe(
        _FUNCTION_WORDS
        + _BABI_CORE_WORDS
        + _NOUNS
        + _VERBS
        + _ADJECTIVES
        + _ADVERBS
        + benchmark_words
    )
    words = list(seed_words)
    seen = set(words)
    target = max(int(active_vocab_size), len(words))
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
        if len(words) >= target:
            break
    return tuple(words[:target])


def _filter_encodable_words(
    words: Sequence[str],
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[str, ...]:
    out = []
    for word in _dedupe(words):
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


def _build_real_hellaswag_items(
    *,
    allowed_words: set[str],
    n_eval_items: int,
    seed: int,
    max_words: int,
    require_allowed: bool = True,
) -> tuple[SentenceProbeItem, ...]:
    rows = list(_cached_hellaswag_rows())
    if not rows:
        return ()
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    items: list[SentenceProbeItem] = []
    for row in rows:
        endings = row.get("endings") or ()
        if len(endings) != 4:
            continue
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        if not 0 <= label < 4:
            continue
        prefix = _truncate_words(
            str(row.get("ctx") or ""), max_words=max_words, from_end=True
        )
        choices = [
            _truncate_words(str(ending), max_words=max_words, from_end=False)
            for ending in endings
        ]
        if not prefix or any(not choice for choice in choices):
            continue
        if any(_has_noisy_words(part) for part in (prefix, *choices)):
            continue
        if len(set(choices)) != 4:
            continue
        correct = choices[label]
        distractors = tuple(
            choice for idx, choice in enumerate(choices) if idx != label
        )
        good_sentence = _sentence_with_min_words(f"{prefix} {correct}", min_words=4)
        bad_order = _bad_order_sentence(good_sentence)
        bad_binding = _sentence_with_min_words(
            f"{prefix} {distractors[0]}", min_words=4
        )
        candidates = (
            prefix,
            correct,
            *distractors,
            good_sentence,
            bad_order,
            bad_binding,
        )
        if not (
            _is_short_sentence(good_sentence, min_words=4, max_words=max_words * 2)
            and _is_short_sentence(bad_binding, min_words=4, max_words=max_words * 2)
        ):
            continue
        if require_allowed and any(
            not _all_words_allowed(candidate, allowed_words) for candidate in candidates
        ):
            continue
        if bad_order == good_sentence or bad_binding == good_sentence:
            continue
        items.append(
            SentenceProbeItem(
                prefix=prefix,
                correct=f" {correct}",
                distractors=tuple(f" {d}" for d in distractors),  # type: ignore[arg-type]
                good_sentence=good_sentence,
                bad_order_sentence=bad_order,
                bad_binding_sentence=bad_binding,
                source="hellaswag",
            )
        )
        if len(items) >= int(n_eval_items):
            break
    return tuple(items)


def _build_real_blimp_pairs(
    *,
    allowed_words: set[str],
    n_pairs: int,
    seed: int,
    max_words: int,
    require_allowed: bool = True,
) -> tuple[tuple[str, str], ...]:
    pairs = list(_cached_blimp_pairs())
    if not pairs:
        return ()
    rng = random.Random(int(seed) + 17)
    rng.shuffle(pairs)
    out: list[tuple[str, str]] = []
    for good_raw, bad_raw in pairs:
        good = _truncate_words(good_raw, max_words=max_words, from_end=False)
        bad = _truncate_words(bad_raw, max_words=max_words, from_end=False)
        if not good or not bad or good == bad:
            continue
        if not (
            _is_short_sentence(good, min_words=3, max_words=max_words)
            and _is_short_sentence(bad, min_words=3, max_words=max_words)
        ):
            continue
        if require_allowed and not (
            _all_words_allowed(good, allowed_words)
            and _all_words_allowed(bad, allowed_words)
        ):
            continue
        out.append((good, bad))
        if len(out) >= int(n_pairs):
            break
    return tuple(out)


def _build_real_train_sentences(
    *,
    allowed_words: set[str],
    max_sentences: int,
    seed: int,
    max_words: int,
    excluded_sentences: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    candidates: list[str] = []
    for row in _cached_hellaswag_rows():
        ctx = _truncate_words(
            str(row.get("ctx") or ""), max_words=max_words, from_end=True
        )
        endings = row.get("endings") or ()
        try:
            label = int(row.get("label"))
            ending = endings[label]
        except (IndexError, TypeError, ValueError):
            ending = ""
        if ctx:
            candidates.append(ctx)
        if ending:
            candidates.append(
                _truncate_words(f"{ctx} {ending}", max_words=max_words, from_end=False)
            )
    for good, _bad in _cached_blimp_pairs():
        candidates.append(_truncate_words(good, max_words=max_words, from_end=False))
    rng = random.Random(int(seed) + 29)
    rng.shuffle(candidates)
    out: list[str] = []
    seen: set[str] = set()
    for sentence in candidates:
        clean = _sentence_with_min_words(sentence, min_words=3)
        if not clean or clean in seen:
            continue
        if clean in excluded_sentences:
            continue
        if _has_noisy_words(clean):
            continue
        if not _all_words_allowed(clean, allowed_words):
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= int(max_sentences):
            break
    return tuple(out)


def build_sentence_probe_corpus(
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    vocab_size: int,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "gpt2",
    n_eval_items: int = _DEFAULT_EVAL_ITEMS,
    seed: int = 42,
    max_text_words: int = _MAX_TEXT_WORDS,
) -> SentenceProbeCorpus:
    """Build deterministic real-word diagnostic train/eval material.

    ``active_vocab_size`` controls the real token vocabulary used for this
    diagnostic.  Curated role words cover compact grammatical fallbacks, while
    mined bAbI / benchmark snippets and tokenizer words keep 1000-word settings
    broad enough for step-curve calibration.  Held-out slot/binding
    generalization lives in ``nano_blimp_v3``.
    """
    tok = (tokenizer or "tiktoken").strip().lower()
    if tok in ("byte", "bytes"):
        vocabulary = _dedupe(_FUNCTION_WORDS + _NOUNS + _VERBS + _ADJECTIVES + _ADVERBS)
    else:
        enc_name = "gpt2" if tok == "gpt2" else tiktoken_encoding
        vocabulary = _tokenizer_word_bank(
            int(active_vocab_size),
            tiktoken_encoding=enc_name,
            max_token_id=int(vocab_size),
        )

    allowed_words = set(vocabulary[: int(active_vocab_size)])
    eval_items = _build_babi_items(
        allowed_words=allowed_words,
        n_eval_items=n_eval_items,
        seed=seed,
    )
    min_real_eval = min(int(n_eval_items), 8)
    if len(eval_items) < min_real_eval:
        eval_items = _build_babi_items(
            allowed_words=allowed_words,
            n_eval_items=n_eval_items,
            seed=seed,
            require_allowed=False,
        )

    if len(eval_items) < min_real_eval:
        eval_items = _build_real_hellaswag_items(
            allowed_words=allowed_words,
            n_eval_items=n_eval_items,
            seed=seed,
            max_words=max_text_words,
        )
    if len(eval_items) < min_real_eval:
        eval_items = _build_real_hellaswag_items(
            allowed_words=allowed_words,
            n_eval_items=n_eval_items,
            seed=seed,
            max_words=max_text_words,
            require_allowed=False,
        )

    blimp_pairs = _build_real_blimp_pairs(
        allowed_words=allowed_words,
        n_pairs=max(int(n_eval_items), 1),
        seed=seed,
        max_words=max_text_words,
    )
    if len(blimp_pairs) < min_real_eval:
        blimp_pairs = _build_real_blimp_pairs(
            allowed_words=allowed_words,
            n_pairs=max(int(n_eval_items), 1),
            seed=seed,
            max_words=max_text_words,
            require_allowed=False,
        )

    excluded_sentences = frozenset(
        [item.good_sentence for item in eval_items]
        + [item.bad_order_sentence for item in eval_items]
        + [item.bad_binding_sentence for item in eval_items]
        + [sent for pair in blimp_pairs for sent in pair]
    )
    max_train_sentences = max(int(active_vocab_size), int(n_eval_items) * 8)
    train_source = "babi"
    train_sentences = _build_babi_train_sentences(
        allowed_words=allowed_words,
        max_sentences=max_train_sentences,
        seed=seed,
        excluded_sentences=excluded_sentences,
    )
    if len(train_sentences) < min(max_train_sentences, min_real_eval):
        train_source = "benchmark"
        train_sentences = _build_real_train_sentences(
            allowed_words=allowed_words,
            max_sentences=max_train_sentences,
            seed=seed,
            max_words=max_text_words,
            excluded_sentences=excluded_sentences,
        )

    if eval_items and train_sentences:
        source_counts = {
            "train_real": len(train_sentences),
            "train_babi": len(train_sentences) if train_source == "babi" else 0,
            "train_benchmark": len(train_sentences)
            if train_source == "benchmark"
            else 0,
            "babi_eval": sum(1 for item in eval_items if item.source == "babi_qa"),
            "hellaswag_eval": sum(
                1 for item in eval_items if item.source == "hellaswag"
            ),
            "blimp_pairs": len(blimp_pairs),
            "curated_train": 0,
        }
        return SentenceProbeCorpus(
            active_vocab_size=int(active_vocab_size),
            vocabulary=tuple(vocabulary[: int(active_vocab_size)]),
            nouns=(),
            verbs=(),
            adjectives=(),
            adverbs=(),
            train_sentences=train_sentences,
            eval_items=eval_items,
            blimp_pairs=blimp_pairs,
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
            source_counts=source_counts,
        )

    role_limit = max(8, min(48, int(active_vocab_size) // 12))
    nouns = _filter_encodable_words(
        _NOUNS[:role_limit],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    verbs = _filter_encodable_words(
        _VERBS[:role_limit],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    adjectives = _filter_encodable_words(
        _ADJECTIVES[:role_limit],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    adverbs = _filter_encodable_words(
        _ADVERBS[:role_limit],
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
    n_roles = min(len(nouns), len(verbs), len(adjectives), len(adverbs))
    if n_roles < 8:
        raise ValueError(
            "not enough encodable real words for controlled sentence probe"
        )
    nouns = nouns[:n_roles]
    verbs = verbs[:n_roles]
    adjectives = adjectives[:n_roles]
    adverbs = adverbs[:n_roles]

    curated_train_sentences: list[str] = []
    curated_eval_items: list[SentenceProbeItem] = []
    n_items = min(int(n_eval_items), n_roles)
    for i, noun in enumerate(nouns):
        verb = verbs[(i * 3 + 1) % n_roles]
        adjective = adjectives[(i * 5 + 2) % n_roles]
        adverb = adverbs[(i * 7 + 3) % n_roles]
        base = _BASE_VERBS.get(verb, verb[:-2] if verb.endswith("ed") else verb)
        curated_train_sentences.extend(
            (
                f"the {noun} {verb}",
                f"the {adjective} {noun} {verb}",
                f"the {noun} can {base}",
                f"the {noun} is {adjective}",
            )
        )
        if i >= n_items:
            continue
        distractors: list[str] = []
        for offset in (1, 2, 3):
            wrong_verb = verbs[((i + offset) * 3 + 1) % n_roles]
            wrong_adverb = adverbs[((i + offset) * 7 + 3) % n_roles]
            distractors.append(f" {wrong_verb} {wrong_adverb}")
        wrong_binding_verb = verbs[((i + 1) * 3 + 1) % n_roles]
        curated_eval_items.append(
            SentenceProbeItem(
                prefix=f"the {noun}",
                correct=f" {verb} {adverb}",
                distractors=tuple(distractors),  # type: ignore[arg-type]
                good_sentence=f"the {noun} {verb} {adverb}",
                bad_order_sentence=f"{verb} the {noun} {adverb}",
                bad_binding_sentence=f"the {noun} {wrong_binding_verb} {adverb}",
                source="curated",
            )
        )
    return SentenceProbeCorpus(
        active_vocab_size=int(active_vocab_size),
        vocabulary=tuple(vocabulary[: int(active_vocab_size)]),
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives,
        adverbs=adverbs,
        train_sentences=tuple(curated_train_sentences),
        eval_items=tuple(curated_eval_items),
        blimp_pairs=(),
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
        source_counts={
            "train_real": 0,
            "train_babi": 0,
            "train_benchmark": 0,
            "babi_eval": 0,
            "hellaswag_eval": 0,
            "blimp_pairs": 0,
            "curated_train": len(curated_train_sentences),
        },
    )


def _sentence_tokens(
    sentence: str,
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[int, ...]:
    return _encode_text(
        sentence,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )


def _make_train_batch(
    corpus: SentenceProbeCorpus,
    *,
    vocab_size: int,
    batch_size: int,
    device: str,
    rng: torch.Generator,
) -> torch.Tensor:
    indices = torch.randint(
        0, len(corpus.train_sentences), (int(batch_size),), generator=rng
    )
    seqs = [
        _sentence_tokens(
            corpus.train_sentences[int(idx.item())],
            vocab_size=vocab_size,
            tokenizer=corpus.tokenizer,
            tiktoken_encoding=corpus.tiktoken_encoding,
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


def _choice_accuracy(
    model: nn.Module,
    corpus: SentenceProbeCorpus,
    *,
    vocab_size: int,
    device: str,
) -> float:
    grouped_sequences: list[list[tuple[int, ...]]] = []
    grouped_starts: list[list[int]] = []
    for item in corpus.eval_items:
        prefix_tokens = _sentence_tokens(
            item.prefix,
            vocab_size=vocab_size,
            tokenizer=corpus.tokenizer,
            tiktoken_encoding=corpus.tiktoken_encoding,
        )
        choices = (item.correct, *item.distractors)
        seqs: list[tuple[int, ...]] = []
        starts: list[int] = []
        for choice in choices:
            choice_tokens = _sentence_tokens(
                choice,
                vocab_size=vocab_size,
                tokenizer=corpus.tokenizer,
                tiktoken_encoding=corpus.tiktoken_encoding,
            )
            seq, start = concat_choice_tokens(
                prefix_tokens,
                choice_tokens,
                max_seq_len=_MAX_SEQ_LEN,
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


def _pair_accuracy(
    model: nn.Module,
    good_sentences: Sequence[str],
    bad_sentences: Sequence[str],
    corpus: SentenceProbeCorpus,
    *,
    vocab_size: int,
    device: str,
) -> float:
    grouped_sequences = []
    grouped_starts = []
    for good, bad in zip(good_sentences, bad_sentences):
        good_tokens = _sentence_tokens(
            good,
            vocab_size=vocab_size,
            tokenizer=corpus.tokenizer,
            tiktoken_encoding=corpus.tiktoken_encoding,
        )
        bad_tokens = _sentence_tokens(
            bad,
            vocab_size=vocab_size,
            tokenizer=corpus.tokenizer,
            tiktoken_encoding=corpus.tiktoken_encoding,
        )
        grouped_sequences.append([good_tokens, bad_tokens])
        grouped_starts.append([0, 0])
    scores = grouped_choice_scores(
        model,
        grouped_sequences,
        grouped_starts,
        vocab_size=vocab_size,
        device=device,
    )
    if not scores:
        return 0.0
    return sum(1 for row in scores if row and row[0] > row[1]) / len(scores)


def evaluate_controlled_sentence_probe(
    model: nn.Module,
    corpus: SentenceProbeCorpus,
    *,
    vocab_size: int,
    device: str,
) -> tuple[float, float, float]:
    """Evaluate diagnostic choice and minimal-pair sentence-shape scores."""
    model.eval()
    hella = _choice_accuracy(model, corpus, vocab_size=vocab_size, device=device)
    order = _pair_accuracy(
        model,
        [item.good_sentence for item in corpus.eval_items],
        [item.bad_order_sentence for item in corpus.eval_items],
        corpus,
        vocab_size=vocab_size,
        device=device,
    )
    binding = _pair_accuracy(
        model,
        [pair[0] for pair in corpus.blimp_pairs]
        if corpus.blimp_pairs
        else [item.good_sentence for item in corpus.eval_items],
        [pair[1] for pair in corpus.blimp_pairs]
        if corpus.blimp_pairs
        else [item.bad_binding_sentence for item in corpus.eval_items],
        corpus,
        vocab_size=vocab_size,
        device=device,
    )
    return hella, order, binding


def controlled_sentence_probe(
    model: nn.Module,
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    n_eval_items: int = _DEFAULT_EVAL_ITEMS,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "gpt2",
) -> ControlledSentenceResult:
    """Train on real-word nano sentences, then evaluate sentence-shape forms.

    The caller's model state is preserved.  This function intentionally does
    not write leaderboard or program-result fields; it is for diagnostics and
    step-curve calibration only.  ``nano_blimp_v3`` owns held-out slot/binding
    generalization.
    """
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    vocab_size = int(getattr(model, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return ControlledSentenceResult(
            score=0.0,
            nano_hellaswag_acc=0.0,
            nano_blimp_order_acc=0.0,
            nano_blimp_binding_acc=0.0,
            active_vocab_size=int(active_vocab_size),
            n_train_steps=0,
            n_train_sentences=0,
            n_eval_items=0,
            n_blimp_pairs=0,
            chance=0.25,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="missing_vocab_size",
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
        )

    try:
        corpus = build_sentence_probe_corpus(
            active_vocab_size=active_vocab_size,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
            n_eval_items=n_eval_items,
            seed=seed,
        )
    except ValueError as exc:
        logger.debug("controlled sentence corpus unavailable: %s", exc)
        return ControlledSentenceResult(
            score=0.0,
            nano_hellaswag_acc=0.0,
            nano_blimp_order_acc=0.0,
            nano_blimp_binding_acc=0.0,
            active_vocab_size=int(active_vocab_size),
            n_train_steps=0,
            n_train_sentences=0,
            n_eval_items=0,
            n_blimp_pairs=0,
            chance=0.25,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
            tokenizer=tokenizer,
            tiktoken_encoding=tiktoken_encoding,
        )

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    steps = 0
    status = "ok"
    try:
        model.train()
        opt = make_adamw(model.parameters(), lr=lr)
        for step in range(int(n_train_steps)):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
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
                status = "non_finite_loss"
                break
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            steps = step + 1

        if status == "non_finite_loss":
            hella = order = binding = 0.0
        else:
            hella, order, binding = evaluate_controlled_sentence_probe(
                model,
                corpus,
                vocab_size=vocab_size,
                device=device,
            )
    finally:
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    hella_norm = max(0.0, (hella - 0.25) / 0.75)
    score = (hella_norm + order + binding) / 3.0
    return ControlledSentenceResult(
        score=round(float(score), 4),
        nano_hellaswag_acc=round(float(hella), 4),
        nano_blimp_order_acc=round(float(order), 4),
        nano_blimp_binding_acc=round(float(binding), 4),
        active_vocab_size=int(active_vocab_size),
        n_train_steps=int(steps),
        n_train_sentences=len(corpus.train_sentences),
        n_eval_items=len(corpus.eval_items),
        n_blimp_pairs=len(corpus.blimp_pairs),
        chance=0.25,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status=status,
        tokenizer=tokenizer,
        tiktoken_encoding=tiktoken_encoding,
    )
