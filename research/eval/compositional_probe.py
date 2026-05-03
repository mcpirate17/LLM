"""Compositional generalization — train binary op, test ternary extension.

Tests whether the architecture supports compositional generalization: train the
model to apply ``f(a, b) = (a + b) mod V`` from sequences ``[OP][a][OP][b][QUERY]``,
then evaluate on ``[OP][a][OP][b][OP][c][QUERY]`` where the target is
``(a + b + c) mod V``. The model is never shown the 3-operand form during
training. Generalizing requires it to compose the binary operation, not memorize
the surface pattern.

Score = ternary accuracy (over chance) given the model first crossed a
binary-accuracy threshold (so we don't credit "didn't learn anything"). If the
model never learns the binary form (binary_acc < 5× chance), the probe returns
status='did_not_learn_binary' with score=0.

Output column: ``compositional_score``.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

_TIMEOUT_S = 90.0
_OP_MARKER = 6
_QUERY_MARKER = 7
# Calibration after the top-10 v2 run: vocab=128 was too hard — 0/10
# architectures learned the binary form at chance baseline 1/128 = 0.78%
# with 400 train steps. Reduced to 16 (chance = 6.25%) and 1500 train
# steps so we can actually distinguish "learned binary" vs "didn't" and
# then ask the harder question (ternary generalization).
_VOCAB_LO = 100
_VOCAB_HI = 116  # 16 number tokens
_VOCAB_N = _VOCAB_HI - _VOCAB_LO
_CHANCE = 1.0 / _VOCAB_N  # 0.0625
_BINARY_LEARNED_THRESHOLD = 3 * _CHANCE  # 0.1875 — meaningfully above chance
_DEFAULT_TRAIN_STEPS = 1500
_DEFAULT_EVAL_BATCHES = 8
_DEFAULT_BATCH = 16
_DEFAULT_LR = 1e-3


@dataclass(slots=True)
class CompositionalResult:
    score: float = 0.0
    binary_accuracy: float = 0.0
    ternary_accuracy: float = 0.0
    elapsed_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compositional_score": self.score,
            "compositional_binary_acc": self.binary_accuracy,
            "compositional_ternary_acc": self.ternary_accuracy,
            "compositional_elapsed_ms": self.elapsed_ms,
            "compositional_status": self.status,
        }


def _binary_seq(batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [OP][a][OP][b][QUERY] sequences. Target = (a+b) mod V at last pos."""
    a = torch.randint(0, _VOCAB_N, (batch_size,), device=device)
    b = torch.randint(0, _VOCAB_N, (batch_size,), device=device)
    target = ((a + b) % _VOCAB_N) + _VOCAB_LO
    seq = torch.empty(batch_size, 5, dtype=torch.long, device=device)
    seq[:, 0] = _OP_MARKER
    seq[:, 1] = a + _VOCAB_LO
    seq[:, 2] = _OP_MARKER
    seq[:, 3] = b + _VOCAB_LO
    seq[:, 4] = _QUERY_MARKER
    return seq, target


def _ternary_seq(batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [OP][a][OP][b][OP][c][QUERY] sequences. Target = (a+b+c) mod V."""
    a = torch.randint(0, _VOCAB_N, (batch_size,), device=device)
    b = torch.randint(0, _VOCAB_N, (batch_size,), device=device)
    c = torch.randint(0, _VOCAB_N, (batch_size,), device=device)
    target = ((a + b + c) % _VOCAB_N) + _VOCAB_LO
    seq = torch.empty(batch_size, 7, dtype=torch.long, device=device)
    seq[:, 0] = _OP_MARKER
    seq[:, 1] = a + _VOCAB_LO
    seq[:, 2] = _OP_MARKER
    seq[:, 3] = b + _VOCAB_LO
    seq[:, 4] = _OP_MARKER
    seq[:, 5] = c + _VOCAB_LO
    seq[:, 6] = _QUERY_MARKER
    return seq, target


def _eval_acc(
    model: nn.Module, batches: int, batch_size: int, ternary: bool, device: str
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for _ in range(batches):
            seq, tgt = (_ternary_seq if ternary else _binary_seq)(batch_size, device)
            logits = model(seq)
            pred_logits = logits[:, -1, _VOCAB_LO:_VOCAB_HI]
            preds = pred_logits.argmax(dim=-1) + _VOCAB_LO
            correct += int((preds == tgt).sum().item())
            total += int(tgt.shape[0])
    return correct / max(total, 1)


def compositional_score(
    model: nn.Module,
    *,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    n_eval_batches: int = _DEFAULT_EVAL_BATCHES,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> CompositionalResult:
    """Train binary, evaluate binary + ternary. Trains the *original* model
    in place (no deepcopy). The bundle driver runs this probe last so
    in-place mutation of the bundle's model is acceptable; this avoids the
    ``copy.deepcopy`` failure on models with non-leaf parameters
    (e.g. weight_norm parametrize)."""
    t0 = time.perf_counter()
    deadline = t0 + timeout_s

    # Snapshot state for restore. State_dict survives weight_norm where
    # deepcopy doesn't, so we can both train in place and revert at the end.
    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    model.train()
    opt = make_adamw(model.parameters(), lr=lr)

    try:
        for _step in range(n_train_steps):
            if time.perf_counter() > deadline:
                break
            seq, tgt = _binary_seq(batch_size, device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq)
            pred_logits = logits[:, -1, _VOCAB_LO:_VOCAB_HI]
            loss = F.cross_entropy(pred_logits, tgt - _VOCAB_LO)
            if not torch.isfinite(loss):
                break
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()

        binary_acc = _eval_acc(
            model, n_eval_batches, batch_size, ternary=False, device=device
        )
        ternary_acc = _eval_acc(
            model, n_eval_batches, batch_size, ternary=True, device=device
        )
    finally:
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if binary_acc < _BINARY_LEARNED_THRESHOLD:
        return CompositionalResult(
            score=0.0,
            binary_accuracy=round(binary_acc, 4),
            ternary_accuracy=round(ternary_acc, 4),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="did_not_learn_binary",
        )

    # Score: how well ternary tracks the binary capability — proxy for true
    # composition. score = ternary_acc / max(binary_acc, eps), capped at 1.
    score = min(1.0, ternary_acc / max(binary_acc, 1e-6))
    return CompositionalResult(
        score=round(float(score), 4),
        binary_accuracy=round(binary_acc, 4),
        ternary_accuracy=round(ternary_acc, 4),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status="ok",
    )
