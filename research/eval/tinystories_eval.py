"""TinyStories validation for semantic coherence.

Downloads TinyStories from HuggingFace, micro-trains the candidate model,
then evaluates perplexity on the validation split. TinyStories uses simple
language patterns, so models that learn well here demonstrate basic
language modeling ability beyond synthetic data.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    tokenize_file,
    make_batches,
    micro_train_loop,
    compute_perplexity,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "aria" / "tinystories"
_DEFAULT_MAX_CHARS_TRAIN = 500_000
_DEFAULT_MAX_CHARS_VAL = 50_000


def _download_tinystories(
    max_chars_train: int = _DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
) -> tuple[Path, Path]:
    """Download and cache TinyStories train/val splits."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_path = _CACHE_DIR / "train.txt"
    val_path = _CACHE_DIR / "validation.txt"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("HuggingFace `datasets` required: pip install datasets")

    logger.info("Downloading TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", trust_remote_code=True)

    for split_name, path, max_chars in [
        ("train", train_path, max_chars_train),
        ("validation", val_path, max_chars_val),
    ]:
        texts = ds[split_name]["text"]
        combined = "\n".join(t for t in texts if t.strip())
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        path.write_text(combined, encoding="utf-8")

    logger.info("TinyStories cached at %s", _CACHE_DIR)
    return train_path, val_path


def evaluate_tinystories(
    model, vocab_size, device,
    n_train_steps=200, seq_len=128,
    n_train_batches=32, n_eval_batches=8,
    batch_size=4, lr=3e-4,
    max_chars_train=_DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val=_DEFAULT_MAX_CHARS_VAL,
) -> Dict[str, Any]:
    """Micro-train on TinyStories and evaluate perplexity."""
    t0 = time.perf_counter()
    try:
        train_path, val_path = _download_tinystories(max_chars_train, max_chars_val)
    except Exception as e:
        logger.warning("TinyStories download failed: %s", e)
        return {"tinystories_perplexity": None, "error": f"download_failed: {e}"}

    train_tokens = tokenize_file(train_path, vocab_size)
    val_tokens = tokenize_file(val_path, vocab_size)

    if len(train_tokens) < seq_len + 1 or len(val_tokens) < seq_len + 1:
        return {"tinystories_perplexity": None, "error": "insufficient_tokens"}

    train_batches = make_batches(train_tokens, batch_size, seq_len, n_train_batches, device)
    val_batches = make_batches(val_tokens, batch_size, seq_len, n_eval_batches, device, seed=123)

    if not train_batches or not val_batches:
        return {"tinystories_perplexity": None, "error": "batch_generation_failed"}

    pre_ppl = compute_perplexity(model, val_batches, vocab_size)
    train_final_loss = micro_train_loop(model, train_batches, vocab_size, n_train_steps, lr)
    post_ppl = compute_perplexity(model, val_batches, vocab_size)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Score: 0-1 normalized, higher = better
    tinystories_score = None
    if post_ppl is not None and post_ppl > 0:
        ratio = post_ppl / vocab_size
        tinystories_score = round(1.0 / (1.0 + math.log(max(ratio, 1e-6))), 4)
        tinystories_score = max(0.0, min(1.0, tinystories_score))

    return {
        "tinystories_perplexity": round(post_ppl, 2) if post_ppl is not None else None,
        "tinystories_pre_perplexity": round(pre_ppl, 2) if pre_ppl is not None else None,
        "tinystories_score": tinystories_score,
        "train_final_loss": round(train_final_loss, 6),
        "n_train_steps": n_train_steps,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "elapsed_ms": round(elapsed_ms, 1),
    }
