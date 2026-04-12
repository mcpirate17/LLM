"""Passkey retrieval — measures ability to retrieve a planted token across distances.

Generates sequences:  [random tokens...] [PASSKEY=X at random position] [random tokens...] [QUERY]
The model must predict X after seeing the query token.

Tests at sequence lengths 256, 512, 1024, 2048.
Score = fraction of lengths where accuracy > chance.
Output column: ``robustness_long_ctx_passkey_score``
"""

from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_adamw

logger = logging.getLogger(__name__)

_TIMEOUT_S = 90.0
# Passkey and query markers — chosen outside common token ranges
_PASSKEY_MARKER = 2
_QUERY_MARKER = 3
# Vocab range for random fill and passkey values
_FILL_LO = 100
_FILL_HI = 356  # 256 possible passkey values
_FILL_N = _FILL_HI - _FILL_LO
# Chance baseline: 1/256 ≈ 0.004; we use 3x chance as threshold
_CHANCE = 1.0 / _FILL_N
_PASS_THRESHOLD = 3 * _CHANCE


@dataclass(slots=True)
class PasskeyResult:
    """Result from passkey retrieval probe."""

    score: float = 0.0
    per_length: Dict[int, float] = field(default_factory=dict)
    lengths_passed: int = 0
    lengths_tested: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passkey_score": self.score,
            "passkey_per_length": self.per_length,
            "passkey_lengths_passed": self.lengths_passed,
            "passkey_lengths_tested": self.lengths_tested,
            "passkey_elapsed_ms": self.elapsed_ms,
            "passkey_status": self.status,
        }


def _generate_passkey_batch(
    batch_size: int,
    seq_len: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of passkey retrieval sequences.

    Format: [fill...] [PASSKEY_MARKER] [passkey_value] [fill...] [QUERY_MARKER]
    Target: passkey_value (the token right after PASSKEY_MARKER)
    The QUERY_MARKER is at position seq_len-1; model must predict passkey_value there.
    """
    # Random fill tokens
    input_ids = torch.randint(
        _FILL_LO, _FILL_HI, (batch_size, seq_len), dtype=torch.long
    )

    # Random passkey values
    passkey_values = torch.randint(_FILL_LO, _FILL_HI, (batch_size,), dtype=torch.long)

    # Place passkey marker + value at random position in first 70% of sequence
    max_pos = max(1, int(seq_len * 0.7) - 2)
    positions = torch.randint(1, max_pos + 1, (batch_size,))

    for i in range(batch_size):
        p = positions[i].item()
        input_ids[i, p] = _PASSKEY_MARKER
        input_ids[i, p + 1] = passkey_values[i]

    # Query marker at end
    input_ids[:, -1] = _QUERY_MARKER

    return input_ids.to(device), passkey_values.to(device)


def _generate_passkey_eval_set(
    n_eval: int,
    seq_len: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed eval set with seed 42."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    try:
        return _generate_passkey_batch(n_eval, seq_len, device)
    finally:
        torch.random.set_rng_state(rng_state)


def _eval_passkey_accuracy(
    model: nn.Module,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    batch_size: int,
) -> float:
    """Evaluate passkey retrieval accuracy on eval set."""
    model.eval()
    correct = 0
    total = eval_ids.shape[0]
    query_pos = eval_ids.shape[1] - 1  # predict at last position

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inp = eval_ids[start:end]
            tgt = eval_targets[start:end]
            logits = model(inp)
            pred_logits = logits[:, query_pos, _FILL_LO:_FILL_HI]
            preds = pred_logits.argmax(dim=-1) + _FILL_LO
            correct += (preds == tgt).sum().item()

    return correct / max(total, 1)


def _train_passkey_at_length(
    model: nn.Module,
    seq_len: int,
    n_train_steps: int,
    n_eval: int,
    lr: float,
    batch_size: int,
    device: str,
    deadline: float,
) -> Tuple[float, bool]:
    """Micro-train a deepcopy on passkey retrieval at one length.

    Returns (accuracy, timed_out).
    """
    probe_model = copy.deepcopy(model)
    probe_model.to(device)
    probe_model.train()

    eval_ids, eval_targets = _generate_passkey_eval_set(n_eval, seq_len, device)
    opt = make_adamw(probe_model.parameters(), lr=lr)
    query_pos = seq_len - 1
    timed_out = False

    try:
        for step in range(1, n_train_steps + 1):
            if time.perf_counter() > deadline:
                timed_out = True
                break

            input_ids, targets = _generate_passkey_batch(batch_size, seq_len, device)
            opt.zero_grad(set_to_none=True)
            logits = probe_model(input_ids)
            pred_logits = logits[:, query_pos, _FILL_LO:_FILL_HI]
            loss = F.cross_entropy(pred_logits, targets - _FILL_LO)

            if not torch.isfinite(loss):
                break

            loss.backward()
            nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
            opt.step()

        acc = _eval_passkey_accuracy(probe_model, eval_ids, eval_targets, batch_size)
    finally:
        del eval_ids, eval_targets, probe_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return acc, timed_out


def passkey_retrieval_score(
    model: nn.Module,
    seq_lens: tuple[int, ...] = (256, 512, 1024, 2048),
    n_train_steps: int = 500,
    n_eval: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> PasskeyResult:
    """Micro-train deepcopies on passkey retrieval at increasing lengths.

    Score = fraction of lengths where final accuracy > 3x chance.
    The original model is NOT modified — deepcopies are used.
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    result = PasskeyResult()

    for seq_len in sorted(seq_lens):
        if time.perf_counter() > deadline:
            result.status = "timeout"
            break

        result.lengths_tested += 1
        try:
            acc, timed_out = _train_passkey_at_length(
                model, seq_len, n_train_steps, n_eval, lr, batch_size, device, deadline
            )
            result.per_length[seq_len] = round(acc, 4)
            if acc > _PASS_THRESHOLD:
                result.lengths_passed += 1
            if timed_out:
                result.status = "timeout"
                break
        except Exception as e:
            result.per_length[seq_len] = 0.0
            logger.debug("passkey: failed at seq_len=%d: %s", seq_len, e)

    if result.lengths_tested > 0:
        result.score = round(result.lengths_passed / result.lengths_tested, 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
