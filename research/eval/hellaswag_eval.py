"""HellaSwag commonsense reasoning evaluation.

Downloads HellaSwag from HuggingFace, caches processed examples locally,
and evaluates models via log-likelihood scoring (same method as lm-eval-harness).

For each 4-choice example, we concatenate context + each continuation,
run a forward pass, and pick the continuation with highest mean log-prob
over the continuation tokens.  Accuracy = fraction correct.

These models take token IDs and return logits (SynthesizedModel from compile_model).
Tokenization uses the same UTF-8 bytes mod vocab_size approach as wikitext_eval.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .choice_scoring import concat_choice_tokens, grouped_choice_scores
from ._probe_runtime import disable_native_probe_dispatch
from .utils import tokenize_string

logger = logging.getLogger(__name__)

_HELLASWAG_CACHE_DIR = Path.home() / ".cache" / "aria" / "hellaswag"
_CACHE_FILE = _HELLASWAG_CACHE_DIR / "validation.json"

# Stage budgets (importable — used by the unified backfill runner)
SCREENING_N_EXAMPLES = 50
INVESTIGATION_N_EXAMPLES = 100
VALIDATION_N_EXAMPLES = 200
_TOKENIZED_CACHE_MAX_ENTRIES = 4
_tokenized_examples_cache: "OrderedDict[tuple[int, int], List[Dict[str, Any]]]" = (
    OrderedDict()
)
_TOKENIZED_SUBSET_CACHE_MAX_ENTRIES = 16
_tokenized_subset_cache: "OrderedDict[tuple[int, int, int, int], List[Dict[str, Any]]]" = OrderedDict()
_NATIVE_SUBSET_CACHE_MAX_ENTRIES = 16
_native_subset_cache: "OrderedDict[tuple[int, int, int, int], tuple[List[List[int]], List[List[List[int]]], List[int]]]" = OrderedDict()


# ── Data loading ────────────────────────────────────────────────────────


def _download_hellaswag() -> List[Dict[str, Any]]:
    """Download HellaSwag validation split and cache as JSON.

    Each cached example: {"ctx": str, "endings": [str, str, str, str], "label": int}
    """
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` package required for HellaSwag evaluation. "
            "Install with: uv pip install datasets"
        )

    logger.info("Downloading HellaSwag validation split...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    examples = []
    for row in ds:
        examples.append(
            {
                "ctx": row["ctx"],
                "endings": row["endings"],
                "label": int(row["label"]),
            }
        )

    _HELLASWAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(examples), encoding="utf-8")
    logger.info("HellaSwag cached at %s (%d examples)", _CACHE_FILE, len(examples))
    return examples


def _tokenized_cache_key(vocab_size: int) -> tuple[int, int]:
    mtime = int(_CACHE_FILE.stat().st_mtime_ns) if _CACHE_FILE.exists() else 0
    return int(vocab_size), mtime


def _get_tokenized_examples(vocab_size: int) -> List[Dict[str, Any]]:
    cache_key = _tokenized_cache_key(vocab_size)
    cached = _tokenized_examples_cache.get(cache_key)
    if cached is not None:
        _tokenized_examples_cache.move_to_end(cache_key)
        return cached

    tokenized = []
    for example in _download_hellaswag():
        tokenized.append(
            {
                "ctx_tokens": tokenize_string(example["ctx"], vocab_size),
                "ending_tokens": tuple(
                    tokenize_string(ending, vocab_size) for ending in example["endings"]
                ),
                "label": int(example["label"]),
            }
        )

    _tokenized_examples_cache[cache_key] = tokenized
    _tokenized_examples_cache.move_to_end(cache_key)
    while len(_tokenized_examples_cache) > _TOKENIZED_CACHE_MAX_ENTRIES:
        _tokenized_examples_cache.popitem(last=False)
    return tokenized


def _get_tokenized_subset(
    n: int,
    *,
    vocab_size: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    tokenized = _get_tokenized_examples(vocab_size)
    if n >= len(tokenized):
        return tokenized
    cache_key = (*_tokenized_cache_key(vocab_size), int(n), int(seed))
    cached = _tokenized_subset_cache.get(cache_key)
    if cached is not None:
        _tokenized_subset_cache.move_to_end(cache_key)
        return cached
    import random

    rng = random.Random(seed)
    indices = list(range(len(tokenized)))
    rng.shuffle(indices)
    subset = [tokenized[i] for i in indices[:n]]
    _tokenized_subset_cache[cache_key] = subset
    _tokenized_subset_cache.move_to_end(cache_key)
    while len(_tokenized_subset_cache) > _TOKENIZED_SUBSET_CACHE_MAX_ENTRIES:
        _tokenized_subset_cache.popitem(last=False)
    return subset


def _get_native_subset_payload(
    n: int,
    *,
    vocab_size: int,
    seed: int = 42,
) -> tuple[List[List[int]], List[List[List[int]]], List[int]]:
    """Return native-friendly subset payloads cached across model evaluations.

    This avoids rebuilding nested Python lists via ``tolist()`` for every
    HellaSwag run over the same cached subset.
    """
    cache_key = (*_tokenized_cache_key(vocab_size), int(n), int(seed))
    cached = _native_subset_cache.get(cache_key)
    if cached is not None:
        _native_subset_cache.move_to_end(cache_key)
        return cached

    examples = _get_tokenized_subset(n, vocab_size=vocab_size, seed=seed)
    ctx_tokens = [ex["ctx_tokens"].tolist() for ex in examples]
    ending_tokens = [
        [ending.tolist() for ending in ex["ending_tokens"]] for ex in examples
    ]
    labels = [int(ex["label"]) for ex in examples]
    payload = (ctx_tokens, ending_tokens, labels)
    _native_subset_cache[cache_key] = payload
    _native_subset_cache.move_to_end(cache_key)
    while len(_native_subset_cache) > _NATIVE_SUBSET_CACHE_MAX_ENTRIES:
        _native_subset_cache.popitem(last=False)
    return payload


# ── Scoring ─────────────────────────────────────────────────────────────


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in text


def _clear_cuda_cache(device: str) -> None:
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _recommended_batch_examples(
    requested: int,
    vocab_size: int,
    max_seq_len: int,
    device: str,
    *,
    model_dim: int | None = None,
) -> int:
    batch_examples = max(1, int(requested))
    if not str(device).startswith("cuda"):
        # After removing the slow CPU SwiGLU/linear kernels, the best screening
        # throughput comes from very small example batches. Larger batches widen
        # the 4-choice fanout without paying back enough in kernel efficiency.
        return min(batch_examples, 2)
    # Each example expands to 4 candidate sequences. The scorer materializes a
    # large logits/log-prob tensor over roughly:
    #   4 * batch_examples * (max_seq_len - 1) * vocab_size
    # Target ~3 GiB of transient activation volume to avoid first-try OOMs.
    target_bytes = 3 * 1024 * 1024 * 1024
    bytes_per_scalar = 4
    denom = max(1, 4 * max(1, max_seq_len - 1) * max(1, vocab_size) * bytes_per_scalar)
    capped = max(1, target_bytes // denom)
    return min(batch_examples, int(capped))


def _score_example_batch(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    vocab_size: int,
    device: str,
    max_seq_len: int = 512,
) -> tuple[int, int]:
    """Score a batch of HellaSwag examples using the fastest correct path."""
    ctx_tokens = [ex["ctx_tokens"].tolist() for ex in examples]
    ending_tokens = [[t.tolist() for t in ex["ending_tokens"]] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    try:
        return _score_example_batch_native(
            model,
            ctx_tokens,
            ending_tokens,
            labels,
            vocab_size,
            device,
            max_seq_len=max_seq_len,
        )
    except Exception:
        logger.warning(
            "Native HellaSwag scorer failed; falling back to Python reference",
            exc_info=True,
        )
        return _score_example_batch_python(
            model,
            examples,
            vocab_size,
            device,
            max_seq_len=max_seq_len,
        )


def _score_example_batch_native(
    model: nn.Module,
    ctx_tokens: List[List[int]],
    ending_tokens: List[List[List[int]]],
    labels: List[int],
    vocab_size: int,
    device: str,
    max_seq_len: int = 512,
) -> tuple[int, int]:
    """Score a batch of HellaSwag examples using the native extension."""
    from ._eval_native import load_eval_native

    ext = load_eval_native()

    return ext.hellaswag_score_batch_native(
        model,
        ctx_tokens,
        ending_tokens,
        labels,
        vocab_size,
        str(device),
        max_seq_len,
    )


def _score_example_batch_python(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    vocab_size: int,
    device: str,
    max_seq_len: int = 512,
) -> tuple[int, int]:
    """Reference Python scorer used for parity and native fallback."""
    grouped_sequences: List[List[np.ndarray]] = []
    grouped_starts: List[List[int]] = []

    for ex in examples:
        ctx_tokens = ex["ctx_tokens"]
        ex_sequences: List[np.ndarray] = []
        ex_starts: List[int] = []
        for ending_tokens in ex["ending_tokens"]:
            full_tokens, start_pos = concat_choice_tokens(
                ctx_tokens,
                ending_tokens,
                max_seq_len=max_seq_len,
            )
            ex_sequences.append(full_tokens)
            ex_starts.append(start_pos)
        grouped_sequences.append(ex_sequences)
        grouped_starts.append(ex_starts)

    grouped_scores = grouped_choice_scores(
        model,
        grouped_sequences,
        grouped_starts,
        vocab_size=vocab_size,
        device=device,
    )

    correct = 0
    total = 0
    for ex, option_scores in zip(examples, grouped_scores):
        best_idx = max(range(len(option_scores)), key=option_scores.__getitem__)
        if option_scores[best_idx] == float("-inf"):
            continue
        total += 1
        if best_idx == int(ex["label"]):
            correct += 1

    return correct, total


def _run_hellaswag(
    model: nn.Module,
    vocab_size: int,
    device: str,
    n_examples: int,
    max_seq_len: int = 512,
    batch_examples: int = 16,
) -> Dict[str, Any]:
    """Core HellaSwag evaluation loop. Returns accuracy and metadata."""
    t0 = time.perf_counter()

    try:
        examples = _get_tokenized_subset(n_examples, vocab_size=vocab_size)
        native_ctx_tokens, native_ending_tokens, native_labels = (
            _get_native_subset_payload(n_examples, vocab_size=vocab_size)
        )
    except Exception as exc:
        return {
            "hellaswag_acc": None,
            "hellaswag_status": "data_failed",
            "error": str(exc),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    model.eval()
    correct = 0
    total = 0
    first_error: str | None = None
    effective_batch_examples = _recommended_batch_examples(
        requested=batch_examples,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        device=device,
        model_dim=getattr(model, "model_dim", None),
    )
    oom_retries = 0

    with disable_native_probe_dispatch(model, device=device):
        start = 0
        while start < len(examples):
            batch = examples[start : start + effective_batch_examples]
            batch_ctx_tokens = native_ctx_tokens[
                start : start + effective_batch_examples
            ]
            batch_ending_tokens = native_ending_tokens[
                start : start + effective_batch_examples
            ]
            batch_labels = native_labels[start : start + effective_batch_examples]
            try:
                batch_correct, batch_total = _score_example_batch_native(
                    model,
                    batch_ctx_tokens,
                    batch_ending_tokens,
                    batch_labels,
                    vocab_size,
                    device,
                    max_seq_len=max_seq_len,
                )
                correct += batch_correct
                total += batch_total
                start += effective_batch_examples
            except Exception:
                logger.warning(
                    "Native HellaSwag scorer failed; falling back to Python reference",
                    exc_info=True,
                )
                try:
                    batch_correct, batch_total = _score_example_batch_python(
                        model, batch, vocab_size, device, max_seq_len=max_seq_len
                    )
                    correct += batch_correct
                    total += batch_total
                    start += effective_batch_examples
                    continue
                except Exception as fallback_exc:
                    exc = fallback_exc
                if _is_cuda_oom(exc) and effective_batch_examples > 1:
                    oom_retries += 1
                    effective_batch_examples = max(1, effective_batch_examples // 2)
                    _clear_cuda_cache(device)
                    logger.warning(
                        "HellaSwag CUDA OOM at batch_examples=%d; retrying with %d",
                        len(batch),
                        effective_batch_examples,
                    )
                    continue
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
                logger.debug("HellaSwag batch failed, skipping", exc_info=True)
                start += max(1, effective_batch_examples)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if total == 0:
        result = {
            "hellaswag_acc": None,
            "hellaswag_correct": 0,
            "hellaswag_total": 0,
            "hellaswag_status": "all_failed",
            "hellaswag_n_examples": 0,
            "elapsed_ms": elapsed_ms,
        }
        if first_error is not None:
            result["error"] = first_error
        if oom_retries:
            result["hellaswag_oom_retries"] = oom_retries
        return result

    acc = round(correct / total, 4)
    result = {
        "hellaswag_acc": acc,
        "hellaswag_correct": correct,
        "hellaswag_total": total,
        "hellaswag_n_examples": n_examples,
        "hellaswag_status": "ok",
        "elapsed_ms": elapsed_ms,
    }
    if oom_retries:
        result["hellaswag_oom_retries"] = oom_retries
    return result


# ── Public API ──────────────────────────────────────────────────────────


def screening_hellaswag_eval(
    model: nn.Module,
    vocab_size: int,
    device: str,
    n_examples: int = SCREENING_N_EXAMPLES,
) -> Dict[str, Any]:
    """Non-invasive HellaSwag eval for screening — ~5-15s on GPU.

    Saves and restores model state so the live model is never mutated.
    Returns dict with ``hellaswag_acc``, ``hellaswag_status``, etc.
    """
    was_training = model.training
    try:
        result = _run_hellaswag(model, vocab_size, device, n_examples)
    finally:
        model.train(was_training)

    result["hellaswag_metric_version"] = "screening_hellaswag_v1"
    return result


def evaluate_hellaswag(
    model: nn.Module,
    vocab_size: int,
    device: str,
    n_examples: int = INVESTIGATION_N_EXAMPLES,
) -> Dict[str, Any]:
    """HellaSwag eval for investigation/validation stages.

    Does not save/restore model state (caller manages lifecycle).
    """
    result = _run_hellaswag(model, vocab_size, device, n_examples)
    result["hellaswag_metric_version"] = "hellaswag_v1"
    return result
