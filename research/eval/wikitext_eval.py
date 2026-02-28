"""WikiText perplexity evaluation for architecture robustness.

Downloads WikiText-2 (or WikiText-103) from HuggingFace, micro-trains
the candidate model on the train split, then evaluates perplexity on the
validation split. This tests whether an architecture can learn real
linguistic patterns, not just synthetic data.

Uses the existing CorpusTokenBatcher infrastructure via a cached text file.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

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

_WIKITEXT_CACHE_DIR = Path.home() / ".cache" / "aria" / "wikitext"

# Subset sizes (chars) to keep evaluation fast
_DEFAULT_MAX_CHARS_TRAIN = 500_000   # ~500KB of WikiText for micro-training
_DEFAULT_MAX_CHARS_VAL = 50_000      # ~50KB for validation perplexity


def _download_wikitext(
    variant: str = "wikitext-2-raw-v1",
    max_chars_train: int = _DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
) -> tuple[Path, Path]:
    """Download and cache WikiText train/val splits as plain text files.

    Returns (train_path, val_path) pointing to cached .txt files.
    """
    cache_dir = _WIKITEXT_CACHE_DIR / variant
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.txt"
    val_path = cache_dir / "validation.txt"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` package required for WikiText evaluation. "
            "Install with: pip install datasets"
        )

    logger.info("Downloading WikiText variant=%s ...", variant)
    ds = load_dataset("wikitext", variant, trust_remote_code=True)

    # Extract and truncate text
    for split_name, path, max_chars in [
        ("train", train_path, max_chars_train),
        ("validation", val_path, max_chars_val),
    ]:
        texts = ds[split_name]["text"]
        combined = "\n".join(t for t in texts if t.strip())
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        path.write_text(combined, encoding="utf-8")

    logger.info("WikiText cached at %s", cache_dir)
    return train_path, val_path


    # Tokenize
    train_tokens = tokenize_file(train_path, vocab_size)
    val_tokens = tokenize_file(val_path, vocab_size)

    if len(train_tokens) < seq_len + 1 or len(val_tokens) < seq_len + 1:
        return {
            "wikitext_perplexity": None,
            "error": "insufficient_tokens",
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
        }

    # Prepare batches
    train_batches = make_batches(
        train_tokens, train_batch_size, seq_len, n_train_batches, device, seed=42
    )
    val_batches = make_batches(
        val_tokens, eval_batch_size, seq_len, n_eval_batches, device, seed=123
    )

    if not train_batches or not val_batches:
        return {
            "wikitext_perplexity": None,
            "error": "batch_generation_failed",
        }

    # Pre-training perplexity (baseline for this model)
    pre_ppl = compute_perplexity(model, val_batches, vocab_size)

    # Micro-train
    train_final_loss = micro_train_loop(
        model, train_batches, vocab_size,
        n_steps=n_train_steps, lr=lr,
    )

    # Post-training perplexity
    post_ppl = compute_perplexity(model, val_batches, vocab_size)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Compute improvement ratio
    ppl_improvement = None
    if pre_ppl is not None and post_ppl is not None and pre_ppl > 0:
        ppl_improvement = round(post_ppl / pre_ppl, 4)

    # WikiText score: normalized to 0-1 range
    # Lower perplexity = better. Score = 1 / (1 + log(ppl/vocab_size))
    # This gives ~1.0 for very good models, ~0.0 for random
    wikitext_score = None
    if post_ppl is not None and post_ppl > 0:
        # Normalize: random baseline ≈ vocab_size perplexity
        ratio = post_ppl / vocab_size
        wikitext_score = round(1.0 / (1.0 + math.log(max(ratio, 1e-6))), 4)
        wikitext_score = max(0.0, min(1.0, wikitext_score))

    return {
        "wikitext_perplexity": round(post_ppl, 2) if post_ppl is not None else None,
        "wikitext_pre_perplexity": round(pre_ppl, 2) if pre_ppl is not None else None,
        "wikitext_score": wikitext_score,
        "wikitext_ppl_improvement": ppl_improvement,
        "train_final_loss": round(train_final_loss, 6),
        "variant": variant,
        "n_train_steps": n_train_steps,
        "seq_len": seq_len,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "elapsed_ms": round(elapsed_ms, 1),
    }
