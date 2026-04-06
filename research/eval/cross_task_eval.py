"""Cross-task robustness evaluation.

Micro-trains a model on two different domains (code and natural language)
and compares perplexity. A robust architecture should perform reasonably
on both domains; a large gap indicates domain overfitting.

Uses WikiText-2 for natural language and a Python code subset for code.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Any

import torch

from .corpus_pipeline import (
    TextSplitSpec,
    cache_hf_text_splits,
    prepare_text_corpus_split_batches,
)
from .utils import (
    micro_train_loop,
    compute_perplexity,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "aria" / "cross_task"
_DEFAULT_MAX_CHARS = 200_000


def _download_code_corpus(max_chars: int = _DEFAULT_MAX_CHARS) -> Path:
    """Download and cache a Python code corpus."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "python_code.txt"

    if path.exists():
        return path

    logger.info("Downloading Python code corpus ...")
    paths = cache_hf_text_splits(
        cache_dir=_CACHE_DIR,
        dataset_name="codeparrot/github-code-clean",
        split_specs=(TextSplitSpec("train", "python_code.txt", max_chars),),
        streaming=True,
        trust_remote_code=True,
        load_kwargs={"languages": ["Python"]},
        sample_to_text=lambda sample: sample.get("code", ""),
    )
    return paths["train"]


def _download_nl_corpus(max_chars: int = _DEFAULT_MAX_CHARS) -> Path:
    """Download and cache a natural language corpus (WikiText-2)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "natural_language.txt"

    if path.exists():
        return path

    logger.info("Downloading WikiText-2 for NL corpus ...")
    paths = cache_hf_text_splits(
        cache_dir=_CACHE_DIR,
        dataset_name="wikitext",
        config_name="wikitext-2-raw-v1",
        split_specs=(TextSplitSpec("train", "natural_language.txt", max_chars),),
        trust_remote_code=True,
    )
    return paths["train"]


def evaluate_cross_task_robustness(
    make_model_fn,
    vocab_size: int,
    device: torch.device,
    n_train_steps: int = 200,
    batch_size: int = 4,
    seq_len: int = 128,
    n_train_batches: int = 32,
    n_eval_batches: int = 8,
    lr: float = 3e-4,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> Dict[str, Any]:
    """Evaluate cross-task robustness by training on code and NL separately.

    Args:
        make_model_fn: Callable that returns a fresh model instance.
        vocab_size: Model vocabulary size.
        device: Evaluation device.

    Returns:
        Dict with cross_task_score (0-1), per-domain perplexity, and gap.
    """
    t0 = time.perf_counter()

    try:
        code_path = _download_code_corpus(max_chars)
        nl_path = _download_nl_corpus(max_chars)
    except Exception as e:
        logger.warning("Cross-task corpus download failed: %s", e)
        return {"cross_task_score": None, "error": f"download_failed: {e}"}

    code_train_batches, code_val_batches, code_token_count = (
        prepare_text_corpus_split_batches(
            path=code_path,
            namespace="cross_task:code",
            vocab_size=vocab_size,
            seq_len=seq_len,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            n_train_batches=n_train_batches,
            n_eval_batches=n_eval_batches,
            device=device,
            train_fraction=0.9,
            train_seed=42,
            val_seed=99,
        )
    )
    nl_train_batches, nl_val_batches, nl_token_count = (
        prepare_text_corpus_split_batches(
            path=nl_path,
            namespace="cross_task:nl",
            vocab_size=vocab_size,
            seq_len=seq_len,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            n_train_batches=n_train_batches,
            n_eval_batches=n_eval_batches,
            device=device,
            train_fraction=0.9,
            train_seed=42,
            val_seed=99,
        )
    )

    if not all(
        [code_train_batches, code_val_batches, nl_train_batches, nl_val_batches]
    ):
        return {"cross_task_score": None, "error": "batch_generation_failed"}

    # Train fresh model on code
    code_model = make_model_fn().to(device)
    code_loss = micro_train_loop(
        code_model, code_train_batches, vocab_size, n_train_steps, lr
    )
    code_ppl = compute_perplexity(code_model, code_val_batches, vocab_size)
    del code_model

    # Train fresh model on NL
    nl_model = make_model_fn().to(device)
    nl_loss = micro_train_loop(
        nl_model, nl_train_batches, vocab_size, n_train_steps, lr
    )
    nl_ppl = compute_perplexity(nl_model, nl_val_batches, vocab_size)
    del nl_model

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Cross-task score: measure domain gap
    # Lower gap = more robust. Score = 1 / (1 + |log(code_ppl/nl_ppl)|)
    cross_task_score = None
    ppl_gap = None
    if code_ppl is not None and nl_ppl is not None and code_ppl > 0 and nl_ppl > 0:
        ppl_gap = round(abs(math.log(code_ppl / nl_ppl)), 4)
        cross_task_score = round(1.0 / (1.0 + ppl_gap), 4)

    return {
        "cross_task_score": cross_task_score,
        "code_perplexity": round(code_ppl, 2) if code_ppl is not None else None,
        "nl_perplexity": round(nl_ppl, 2) if nl_ppl is not None else None,
        "ppl_gap": ppl_gap,
        "code_train_loss": round(code_loss, 6),
        "nl_train_loss": round(nl_loss, 6),
        "n_train_steps": n_train_steps,
        "code_tokens": code_token_count,
        "nl_tokens": nl_token_count,
        "elapsed_ms": round(elapsed_ms, 1),
    }
