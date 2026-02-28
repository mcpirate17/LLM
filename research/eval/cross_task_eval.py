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

_CACHE_DIR = Path.home() / ".cache" / "aria" / "cross_task"
_DEFAULT_MAX_CHARS = 200_000


def _download_code_corpus(max_chars: int = _DEFAULT_MAX_CHARS) -> Path:
    """Download and cache a Python code corpus."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "python_code.txt"

    if path.exists():
        return path

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("HuggingFace `datasets` required: pip install datasets")

    logger.info("Downloading Python code corpus ...")
    # Use codeparrot/github-code-clean for Python snippets
    try:
        ds = load_dataset(
            "codeparrot/github-code-clean",
            languages=["Python"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        texts = []
        total_chars = 0
        for sample in ds:
            code = sample.get("code", "")
            if not code.strip():
                continue
            texts.append(code)
            total_chars += len(code)
            if total_chars >= max_chars:
                break
    except Exception:
        # Fallback: generate synthetic Python-like code
        logger.info("Falling back to synthetic Python corpus")
        texts = _generate_synthetic_python(max_chars)

    combined = "\n".join(texts)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    path.write_text(combined, encoding="utf-8")
    return path


def _generate_synthetic_python(max_chars: int) -> List[str]:
    """Generate synthetic Python code snippets as fallback."""
    snippets = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n",
        "for i in range(100):\n    if i % 3 == 0:\n        print('fizz')\n    elif i % 5 == 0:\n        print('buzz')\n",
        "import numpy as np\ndata = np.random.randn(100, 10)\nmean = np.mean(data, axis=0)\nstd = np.std(data, axis=0)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n",
        "with open('data.txt', 'r') as f:\n    lines = f.readlines()\n    for line in lines:\n        tokens = line.strip().split(',')\n        print(tokens)\n",
        "class BinaryTree:\n    def __init__(self):\n        self.root = None\n    def insert(self, val):\n        if self.root is None:\n            self.root = Node(val)\n",
        "def matrix_multiply(a, b):\n    rows_a, cols_a = len(a), len(a[0])\n    rows_b, cols_b = len(b), len(b[0])\n    result = [[0]*cols_b for _ in range(rows_a)]\n    for i in range(rows_a):\n        for j in range(cols_b):\n            for k in range(cols_a):\n                result[i][j] += a[i][k] * b[k][j]\n    return result\n",
    ]
    result = []
    total = 0
    while total < max_chars:
        for s in snippets:
            result.append(s)
            total += len(s)
            if total >= max_chars:
                break
    return result


def _download_nl_corpus(max_chars: int = _DEFAULT_MAX_CHARS) -> Path:
    """Download and cache a natural language corpus (WikiText-2)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "natural_language.txt"

    if path.exists():
        return path

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("HuggingFace `datasets` required: pip install datasets")

    logger.info("Downloading WikiText-2 for NL corpus ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    texts = ds["train"]["text"]
    combined = "\n".join(t for t in texts if t.strip())
    if len(combined) > max_chars:
        combined = combined[:max_chars]
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

    code_tokens = tokenize_file(code_path, vocab_size)
    nl_tokens = tokenize_file(nl_path, vocab_size)

    if len(code_tokens) < seq_len + 1 or len(nl_tokens) < seq_len + 1:
        return {"cross_task_score": None, "error": "insufficient_tokens"}

    # Split each corpus: 90% train, 10% val
    code_split = int(len(code_tokens) * 0.9)
    nl_split = int(len(nl_tokens) * 0.9)

    code_train_batches = make_batches(
        code_tokens[:code_split], batch_size, seq_len, n_train_batches, device, seed=42)
    code_val_batches = make_batches(
        code_tokens[code_split:], batch_size, seq_len, n_eval_batches, device, seed=99)
    nl_train_batches = make_batches(
        nl_tokens[:nl_split], batch_size, seq_len, n_train_batches, device, seed=42)
    nl_val_batches = make_batches(
        nl_tokens[nl_split:], batch_size, seq_len, n_eval_batches, device, seed=99)

    if not all([code_train_batches, code_val_batches, nl_train_batches, nl_val_batches]):
        return {"cross_task_score": None, "error": "batch_generation_failed"}

    # Train fresh model on code
    code_model = make_model_fn().to(device)
    code_loss = micro_train_loop(
        code_model, code_train_batches, vocab_size, n_train_steps, lr)
    code_ppl = compute_perplexity(code_model, code_val_batches, vocab_size)
    del code_model

    # Train fresh model on NL
    nl_model = make_model_fn().to(device)
    nl_loss = micro_train_loop(
        nl_model, nl_train_batches, vocab_size, n_train_steps, lr)
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
        "code_tokens": len(code_tokens),
        "nl_tokens": len(nl_tokens),
        "elapsed_ms": round(elapsed_ms, 1),
    }
