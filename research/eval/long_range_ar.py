"""Long-range associative recall — measures key-value retrieval across sequence lengths.

Parameterizes the base associative recall probe over increasing sequence
lengths by scaling the number of KV pairs.  At each length the model is
micro-trained on associative recall and evaluated for accuracy.

Score = AUC of accuracy across lengths (higher = better long-range recall).
Output column: ``robustness_long_ctx_assoc_score``
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .associative_recall import (
    _VOCAB_HI,
    _VOCAB_LO,
    _VOCAB_N,
    _eval_accuracy,
    _generate_ar_batch,
    _generate_eval_set,
    _get_special_tokens,
)
from .utils import make_adamw

logger = logging.getLogger(__name__)

_TIMEOUT_S = 90.0


@dataclass(slots=True)
class LongRangeARResult:
    """Result from long-range associative recall probe."""

    score: float = 0.0
    per_length: Dict[int, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "long_range_ar_score": self.score,
            "long_range_ar_per_length": self.per_length,
            "long_range_ar_elapsed_ms": self.elapsed_ms,
            "long_range_ar_status": self.status,
        }


def _n_pairs_for_seq_len(target_seq_len: int) -> int:
    """Compute n_pairs such that total sequence length ≈ target_seq_len.

    Each pair occupies 3 tokens (k1, k2, v) plus 4 overhead tokens
    (SEP, query_k1, query_k2, ANS).  So seq_len = 3 * n_pairs + 4.
    Clamp n_pairs to [5, _VOCAB_N // 3] so we stay within the
    256-token restricted vocab (need 3*n_pairs unique tokens).
    """
    n_pairs = max(5, (target_seq_len - 4) // 3)
    max_pairs = _VOCAB_N // 3  # 85 pairs max with 256-token vocab
    return min(n_pairs, max_pairs)


def _train_ar_at_length(
    model: nn.Module,
    n_pairs: int,
    n_train_steps: int,
    n_eval: int,
    lr: float,
    batch_size: int,
    device: str,
    deadline: float,
) -> Tuple[float, bool]:
    """Micro-train a deepcopy on AR at one sequence length.

    Returns (accuracy, timed_out).
    """
    probe_model = copy.deepcopy(model)
    probe_model.to(device)
    probe_model.train()

    sep_token, ans_token = _get_special_tokens(probe_model)
    eval_ids, eval_targets = _generate_eval_set(
        n_eval, n_pairs, sep_token, ans_token, device
    )
    opt = make_adamw(probe_model.parameters(), lr=lr)
    ans_pos = 3 * n_pairs + 3  # actual_seq_len - 1
    timed_out = False

    try:
        for step in range(1, n_train_steps + 1):
            if time.perf_counter() > deadline:
                timed_out = True
                break

            input_ids, targets = _generate_ar_batch(
                batch_size, n_pairs, sep_token, ans_token, device
            )
            opt.zero_grad(set_to_none=True)
            logits = probe_model(input_ids)
            pred_logits = logits[:, ans_pos, _VOCAB_LO:_VOCAB_HI]
            loss = F.cross_entropy(pred_logits, targets - _VOCAB_LO)

            if not torch.isfinite(loss):
                break

            loss.backward()
            nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
            opt.step()

        acc = _eval_accuracy(probe_model, eval_ids, eval_targets, batch_size)
    finally:
        del eval_ids, eval_targets, probe_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return acc, timed_out


def _auc_from_accuracies(accuracies: List[Tuple[int, float]]) -> float:
    """AUC of accuracy vs seq_len, normalized by range."""
    if len(accuracies) >= 2:
        area = 0.0
        for i in range(1, len(accuracies)):
            dt = accuracies[i][0] - accuracies[i - 1][0]
            area += 0.5 * dt * (accuracies[i - 1][1] + accuracies[i][1])
        max_range = accuracies[-1][0] - accuracies[0][0]
        return round(area / max(max_range, 1), 4)
    if len(accuracies) == 1:
        return round(accuracies[0][1], 4)
    return 0.0


def long_range_ar_score(
    model: nn.Module,
    seq_lens: tuple[int, ...] = (128, 256, 512, 1024),
    n_train_steps: int = 500,
    n_eval: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> LongRangeARResult:
    """Train deepcopies on AR at increasing sequence lengths.

    The original model is NOT modified — deepcopies are used.
    Score = AUC of final-accuracy across lengths, normalized to [0, 1].
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    result = LongRangeARResult()
    accuracies: List[Tuple[int, float]] = []

    for seq_len in sorted(seq_lens):
        if time.perf_counter() > deadline:
            result.status = "timeout"
            break

        n_pairs = _n_pairs_for_seq_len(seq_len)

        try:
            acc, timed_out = _train_ar_at_length(
                model, n_pairs, n_train_steps, n_eval, lr, batch_size, device, deadline
            )
            result.per_length[seq_len] = round(acc, 4)
            accuracies.append((seq_len, acc))
            if timed_out:
                result.status = "timeout"
                break
        except Exception as e:
            result.per_length[seq_len] = 0.0
            accuracies.append((seq_len, 0.0))
            logger.debug("long_range_ar: failed at seq_len=%d: %s", seq_len, e)

    result.score = _auc_from_accuracies(accuracies)
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
