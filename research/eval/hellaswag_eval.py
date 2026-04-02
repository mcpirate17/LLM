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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import tokenize_string

logger = logging.getLogger(__name__)

_HELLASWAG_CACHE_DIR = Path.home() / ".cache" / "aria" / "hellaswag"
_CACHE_FILE = _HELLASWAG_CACHE_DIR / "validation.json"

# Stage budgets (importable — used by backfill_hellaswag)
SCREENING_N_EXAMPLES = 50
INVESTIGATION_N_EXAMPLES = 100
VALIDATION_N_EXAMPLES = 200
FAST_LANE_N_EXAMPLES = 25


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


@torch.no_grad()
def _score_continuations(
    model: nn.Module,
    ctx: str,
    endings: List[str],
    vocab_size: int,
    device: str,
    max_seq_len: int = 512,
) -> int:
    """Score 4 continuations and return index of the best one.

    For each continuation:
    1. Tokenize context + continuation
    2. Forward pass to get logits
    3. Compute mean log-prob over continuation tokens only
    4. Pick the continuation with highest mean log-prob
    """
    ctx_tokens = tokenize_string(ctx, vocab_size)
    best_idx = 0
    best_score = float("-inf")

    for i, ending in enumerate(endings):
        ending_tokens = tokenize_string(ending, vocab_size)
        full_tokens = ctx_tokens + ending_tokens

        # Truncate from the left if too long (keep the ending visible)
        if len(full_tokens) > max_seq_len:
            excess = len(full_tokens) - max_seq_len
            full_tokens = full_tokens[excess:]
            # Recompute ctx length after truncation
            ctx_len = max(0, len(ctx_tokens) - excess)
        else:
            ctx_len = len(ctx_tokens)

        if len(ending_tokens) == 0 or ctx_len >= len(full_tokens):
            continue

        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        logits = model(input_ids)  # (1, seq_len, vocab_size)

        # Clamp logits to vocab_size if model outputs more
        if logits.shape[-1] > vocab_size:
            logits = logits[..., :vocab_size]

        # Log-probs for each position predicting the next token
        log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)

        # Mean log-prob over continuation tokens only
        # Position i predicts token i+1, so continuation starts at position ctx_len-1
        # predicting token ctx_len (first continuation token)
        start = max(0, ctx_len - 1)
        end = len(full_tokens) - 1  # last position that predicts a continuation token
        if start >= end:
            continue

        targets = torch.tensor(
            full_tokens[start + 1 : end + 1], dtype=torch.long, device=device
        )
        pred_log_probs = log_probs[start:end]
        token_scores = pred_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mean_ll = token_scores.mean().item()

        if mean_ll > best_score:
            best_score = mean_ll
            best_idx = i

    return best_idx


def _run_hellaswag(
    model: nn.Module,
    vocab_size: int,
    device: str,
    n_examples: int,
    max_seq_len: int = 512,
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

    for ex in examples:
        try:
            pred = _score_continuations(
                model, ex["ctx"], ex["endings"], vocab_size, device, max_seq_len
            )
            if pred == ex["label"]:
                correct += 1
            total += 1
        except Exception:
            # Skip examples that cause OOM or other errors
            logger.debug("HellaSwag example failed, skipping", exc_info=True)
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


def screening_hellaswag_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a normalized screening benchmark payload for persistence."""
    status = result.get("hellaswag_status")
    if not status:
        return None
    return {
        "screening_hellaswag": {
            "benchmark_family": "commonsense_reasoning",
            "metric_version": result.get("hellaswag_metric_version"),
            "status": status,
            "elapsed_ms": result.get("elapsed_ms"),
            "metrics": {
                "hellaswag_acc": result.get("hellaswag_acc"),
                "hellaswag_correct": result.get("hellaswag_correct"),
                "hellaswag_total": result.get("hellaswag_total"),
            },
        }
    }
