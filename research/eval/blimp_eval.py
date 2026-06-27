"""BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation.

Downloads BLiMP from HuggingFace, caches processed examples locally,
and evaluates models via log-likelihood scoring (same method as HellaSwag).

For each minimal pair, we compute the mean log-prob of the grammatical vs
ungrammatical sentence. Accuracy = fraction where the grammatical sentence
scores higher. Reports per-subtask accuracy and overall mean.

BLiMP has 67 subtasks across 12 linguistic categories:
  Anaphor Agreement, Argument Structure, Binding, Control/Raising,
  Determiner-Noun Agreement, Ellipsis, Filler-Gap, Irregular Forms,
  Island Effects, NPI Licensing, Quantifiers, Subject-Verb Agreement

These models take token IDs and return logits (SynthesizedModel from compile_model).
Tokenization uses UTF-8 bytes mod vocab_size (same as hellaswag_eval, wikitext_eval).
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from .choice_scoring import grouped_choice_scores
from .utils import tokenize_string

logger = logging.getLogger(__name__)

_BLIMP_CACHE_DIR = Path.home() / ".cache" / "aria" / "blimp"
_CACHE_FILE = _BLIMP_CACHE_DIR / "all_subtasks.json"
_TIMEOUT_S = 120.0
_BLIMP_SCORE_BATCH_PAIRS = 25

# In-memory cache of the parsed BLiMP JSON. The disk cache holds ~10-30 MB of
# pairs across 67 subtasks; without this, every evaluate_blimp call re-reads
# and re-parses it. Keyed by file mtime so a cache rewrite invalidates.
_BLIMP_DATA_CACHE: Dict[str, List[Dict[str, str]]] | None = None
_BLIMP_DATA_CACHE_MTIME: int = 0
_TOKENIZED_SUBTASK_CACHE_MAX_ENTRIES = 8
_tokenized_subtask_cache: "OrderedDict[tuple[int, int, int, int, int], Dict[str, List[Dict[str, List[int]]]]]" = OrderedDict()


# ── Data loading ────────────────────────────────────────────────────────


_BLIMP_SUBTASKS = (
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
)


def _download_blimp() -> Dict[str, List[Dict[str, str]]]:
    """Download BLiMP and cache as JSON. Returns {subtask: [examples...]}.

    Each example: {"good": str, "bad": str}
    BLiMP uses one HuggingFace config per subtask (67 configs).
    """
    global _BLIMP_DATA_CACHE, _BLIMP_DATA_CACHE_MTIME
    if _CACHE_FILE.exists():
        mtime = int(_CACHE_FILE.stat().st_mtime_ns)
        if _BLIMP_DATA_CACHE is not None and _BLIMP_DATA_CACHE_MTIME == mtime:
            return _BLIMP_DATA_CACHE
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        _BLIMP_DATA_CACHE = data
        _BLIMP_DATA_CACHE_MTIME = mtime
        return data

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` package required for BLiMP evaluation. "
            "Install with: uv pip install datasets"
        )

    logger.info("Downloading BLiMP dataset (67 subtasks)...")
    subtasks: Dict[str, List[Dict[str, str]]] = {}

    for subtask_name in _BLIMP_SUBTASKS:
        try:
            ds = load_dataset(
                "nyu-mll/blimp", subtask_name, split="train", revision="main"
            )
            examples = [
                {"good": row["sentence_good"], "bad": row["sentence_bad"]} for row in ds
            ]
            subtasks[subtask_name] = examples
        except Exception as e:
            logger.warning("Failed to load BLiMP subtask %s: %s", subtask_name, e)

    _BLIMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(subtasks), encoding="utf-8")
    n_total = sum(len(v) for v in subtasks.values())
    logger.info(
        "BLiMP cached at %s (%d subtasks, %d total pairs)",
        _CACHE_FILE,
        len(subtasks),
        n_total,
    )
    return subtasks


def _get_subtask_examples(
    n_per_subtask: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Load and subsample n examples per subtask with deterministic shuffle."""
    import random

    all_subtasks = _download_blimp()
    rng = random.Random(seed)
    result = {}
    for subtask, examples in sorted(all_subtasks.items()):
        if n_per_subtask >= len(examples):
            result[subtask] = examples
        else:
            indices = rng.sample(range(len(examples)), n_per_subtask)
            result[subtask] = [examples[i] for i in indices[:n_per_subtask]]
    return result


def _get_tokenized_subtask_examples(
    n_per_subtask: int,
    *,
    vocab_size: int,
    max_seq_len: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, List[int]]]]:
    """Load, subsample, and tokenize BLiMP pairs once per eval configuration."""
    mtime = int(_CACHE_FILE.stat().st_mtime_ns) if _CACHE_FILE.exists() else 0
    cache_key = (
        int(mtime),
        int(n_per_subtask),
        int(seed),
        int(vocab_size),
        int(max_seq_len),
    )
    cached = _tokenized_subtask_cache.get(cache_key)
    if cached is not None:
        _tokenized_subtask_cache.move_to_end(cache_key)
        return cached

    subtasks = _get_subtask_examples(n_per_subtask, seed=seed)
    tokenized: Dict[str, List[Dict[str, List[int]]]] = {}
    for subtask, pairs in subtasks.items():
        tokenized_pairs: List[Dict[str, List[int]]] = []
        for pair in pairs:
            good = tokenize_string(pair["good"], vocab_size)
            bad = tokenize_string(pair["bad"], vocab_size)
            if len(good) > max_seq_len:
                good = good[:max_seq_len]
            if len(bad) > max_seq_len:
                bad = bad[:max_seq_len]
            tokenized_pairs.append({"good": good.tolist(), "bad": bad.tolist()})
        tokenized[subtask] = tokenized_pairs

    _tokenized_subtask_cache[cache_key] = tokenized
    _tokenized_subtask_cache.move_to_end(cache_key)
    while len(_tokenized_subtask_cache) > _TOKENIZED_SUBTASK_CACHE_MAX_ENTRIES:
        _tokenized_subtask_cache.popitem(last=False)
    return tokenized


# ── Scoring ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _score_token_subtasks_batched(
    model: nn.Module,
    subtasks: Dict[str, List[Dict[str, List[int]]]],
    vocab_size: int,
    device: str,
    *,
    deadline: float,
    batch_pairs: int = _BLIMP_SCORE_BATCH_PAIRS,
) -> tuple[Dict[str, int], Dict[str, int], bool]:
    """Score tokenized BLiMP pairs across subtasks in large chunks."""

    correct_by_subtask: Dict[str, int] = {name: 0 for name in subtasks}
    total_by_subtask: Dict[str, int] = {name: 0 for name in subtasks}
    chunk_pairs: List[Dict[str, List[int]]] = []
    chunk_subtasks: List[str] = []

    def flush() -> None:
        if not chunk_pairs:
            return
        grouped_sequences = [[pair["good"], pair["bad"]] for pair in chunk_pairs]
        grouped_starts = [[0, 0]] * len(chunk_pairs)
        pair_scores = grouped_choice_scores(
            model,
            grouped_sequences,
            grouped_starts,
            vocab_size=vocab_size,
            device=device,
        )
        for subtask_name, scores in zip(chunk_subtasks, pair_scores, strict=False):
            if len(scores) == 2:
                total_by_subtask[subtask_name] += 1
                if scores[0] > scores[1]:
                    correct_by_subtask[subtask_name] += 1
        chunk_pairs.clear()
        chunk_subtasks.clear()

    for subtask_name, pairs in sorted(subtasks.items()):
        for pair in pairs:
            if time.perf_counter() > deadline:
                flush()
                return correct_by_subtask, total_by_subtask, True
            chunk_pairs.append(pair)
            chunk_subtasks.append(subtask_name)
            if len(chunk_pairs) >= batch_pairs:
                flush()

    flush()
    return correct_by_subtask, total_by_subtask, False


# ── Result type ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class BLiMPResult:
    subtask_accuracies: Dict[str, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    n_subtasks: int = 0
    n_examples: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blimp_subtask_accuracies": self.subtask_accuracies,
            "blimp_overall_accuracy": self.overall_accuracy,
            "blimp_n_subtasks": self.n_subtasks,
            "blimp_n_examples": self.n_examples,
            "blimp_status": self.status,
            "blimp_elapsed_ms": self.elapsed_ms,
        }


# ── Main evaluation ────────────────────────────────────────────────────


def evaluate_blimp(
    model: nn.Module,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cuda",
    n_per_subtask: int = 50,
    max_seq_len: int = 512,
    timeout_s: float = _TIMEOUT_S,
) -> BLiMPResult:
    """Evaluate model on BLiMP minimal pairs. Zero-shot, no training.

    For each minimal pair, computes mean log-prob of both sentences.
    Accuracy = fraction where grammatical sentence scores higher.
    """
    t0 = time.perf_counter()
    result = BLiMPResult()

    was_training = model.training
    model.eval()

    try:
        subtasks = _get_tokenized_subtask_examples(
            n_per_subtask,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
    except Exception as e:
        result.status = f"data_failed: {e}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        model.train(was_training)
        return result

    try:
        total_correct = 0
        total_examples = 0

        correct_by_subtask, total_by_subtask, timed_out = _score_token_subtasks_batched(
            model,
            subtasks,
            vocab_size,
            device,
            deadline=t0 + timeout_s,
        )
        if timed_out:
            result.status = "timeout"

        for subtask_name in sorted(subtasks):
            total = total_by_subtask.get(subtask_name, 0)
            if total <= 0:
                continue
            correct = correct_by_subtask.get(subtask_name, 0)
            acc = correct / total
            result.subtask_accuracies[subtask_name] = round(acc, 4)
            total_correct += correct
            total_examples += total

        result.n_subtasks = len(result.subtask_accuracies)
        result.n_examples = total_examples
        result.overall_accuracy = round(total_correct / max(total_examples, 1), 4)

    except Exception as e:
        result.status = f"eval_failed: {e}"
    finally:
        model.train(was_training)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
