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
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.nn as nn

from .utils import batched_span_mean_log_probs, tokenize_string

logger = logging.getLogger(__name__)

_HELLASWAG_CACHE_DIR = Path.home() / ".cache" / "aria" / "hellaswag"
_CACHE_FILE = _HELLASWAG_CACHE_DIR / "validation.json"

# Stage budgets (importable — used by backfill_hellaswag)
SCREENING_N_EXAMPLES = 50
INVESTIGATION_N_EXAMPLES = 100
VALIDATION_N_EXAMPLES = 200


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


def _get_examples(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Load and subsample n examples with deterministic shuffle."""
    all_examples = _download_hellaswag()
    if n >= len(all_examples):
        return all_examples
    # Deterministic subsample via seeded shuffle
    import random

    rng = random.Random(seed)
    indices = list(range(len(all_examples)))
    rng.shuffle(indices)
    return [all_examples[i] for i in indices[:n]]


# ── Scoring ─────────────────────────────────────────────────────────────


def _score_example_batch(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    vocab_size: int,
    device: str,
    max_seq_len: int = 512,
) -> tuple[int, int]:
    """Score a batch of HellaSwag examples using one forward pass.

    Returns ``(n_correct, n_scored)``.
    """
    sequences = []
    start_positions: List[int] = []
    owner_idx: List[int] = []
    ending_idx: List[int] = []

    for ex_idx, ex in enumerate(examples):
        ctx_tokens = tokenize_string(ex["ctx"], vocab_size)
        for opt_idx, ending in enumerate(ex["endings"]):
            ending_tokens = tokenize_string(ending, vocab_size)
            full_tokens = (
                ctx_tokens
                if ending_tokens.size == 0
                else np.concatenate((ctx_tokens, ending_tokens))
            )

            if len(full_tokens) > max_seq_len:
                excess = len(full_tokens) - max_seq_len
                full_tokens = full_tokens[excess:]
                ctx_len = max(0, len(ctx_tokens) - excess)
            else:
                ctx_len = len(ctx_tokens)

            if ending_tokens.size == 0 or ctx_len >= len(full_tokens):
                full_tokens = full_tokens[:0]
                start_pos = 0
            else:
                start_pos = max(0, ctx_len - 1)

            sequences.append(full_tokens)
            start_positions.append(start_pos)
            owner_idx.append(ex_idx)
            ending_idx.append(opt_idx)

    scores = batched_span_mean_log_probs(
        model,
        sequences,
        start_positions,
        vocab_size=vocab_size,
        device=device,
    )

    grouped_scores = [[float("-inf")] * 4 for _ in examples]
    for score, ex_idx, opt_idx in zip(scores.tolist(), owner_idx, ending_idx):
        grouped_scores[ex_idx][opt_idx] = score

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
        examples = _get_examples(n_examples)
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

    for start in range(0, len(examples), max(1, batch_examples)):
        batch = examples[start : start + max(1, batch_examples)]
        try:
            batch_correct, batch_total = _score_example_batch(
                model, batch, vocab_size, device, max_seq_len
            )
            correct += batch_correct
            total += batch_total
        except Exception:
            logger.debug("HellaSwag batch failed, skipping", exc_info=True)
            continue

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if total == 0:
        return {
            "hellaswag_acc": None,
            "hellaswag_status": "all_failed",
            "hellaswag_n_examples": 0,
            "elapsed_ms": elapsed_ms,
        }

    acc = round(correct / total, 4)
    return {
        "hellaswag_acc": acc,
        "hellaswag_correct": correct,
        "hellaswag_total": total,
        "hellaswag_n_examples": n_examples,
        "hellaswag_status": "ok",
        "elapsed_ms": elapsed_ms,
    }


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
