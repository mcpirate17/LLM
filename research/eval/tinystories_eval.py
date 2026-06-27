"""TinyStories validation for semantic coherence.

Downloads TinyStories from HuggingFace, micro-trains the candidate model,
then evaluates perplexity on the validation split. TinyStories uses simple
language patterns, so models that learn well here demonstrate basic
language modeling ability beyond synthetic data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any

from .corpus_pipeline import (
    TextSplitSpec,
    cache_hf_text_splits,
    prepare_text_split_batches,
)
from .wikitext_eval import _finalize_ppl_result
from .utils import (
    micro_train_and_measure_perplexity,
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

    logger.info("Downloading TinyStories ...")
    paths = cache_hf_text_splits(
        cache_dir=_CACHE_DIR,
        dataset_name="roneneldan/TinyStories",
        split_specs=(
            TextSplitSpec("train", "train.txt", max_chars_train),
            TextSplitSpec("validation", "validation.txt", max_chars_val),
        ),
        trust_remote_code=True,
    )

    logger.info("TinyStories cached at %s", _CACHE_DIR)
    return paths["train"], paths["validation"]


def evaluate_tinystories(
    model,
    vocab_size,
    device,
    n_train_steps=200,
    seq_len=128,
    n_train_batches=32,
    n_eval_batches=8,
    batch_size=4,
    lr=3e-4,
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

    train_batches, val_batches, train_tokens, val_tokens = prepare_text_split_batches(
        namespace="tinystories",
        train_path=train_path,
        val_path=val_path,
        vocab_size=vocab_size,
        seq_len=seq_len,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        n_train_batches=n_train_batches,
        n_eval_batches=n_eval_batches,
        device=device,
    )

    if train_batches is None or val_batches is None:
        return {"tinystories_perplexity": None, "error": "insufficient_tokens"}
    if not train_batches or not val_batches:
        return {"tinystories_perplexity": None, "error": "batch_generation_failed"}

    pre_ppl, train_final_loss, post_ppl = micro_train_and_measure_perplexity(
        model,
        train_batches,
        val_batches,
        vocab_size,
        n_train_steps=n_train_steps,
        lr=lr,
    )

    result = _finalize_ppl_result(
        pre_ppl=pre_ppl,
        post_ppl=post_ppl,
        train_final_loss=train_final_loss,
        vocab_size=vocab_size,
        n_train_steps=n_train_steps,
        seq_len=seq_len,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        prefix="tinystories",
    )
    result["train_tokens"] = train_tokens
    result["val_tokens"] = val_tokens
    return result
