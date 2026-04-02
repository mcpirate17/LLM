"""Binding Range probe — zero-shot measurement of copy-at-distance capability.

Generates sequences where token at position i equals token at position i-d.
The model (at its current weights, post screening training) must predict
token i given context. Accuracy vs distance d maps the architecture's
effective receptive field.

Why this works without training:
- After screening micro-training on language, the model has learned local
  copy/repeat patterns from natural text. The probe measures how far
  those patterns extend.
- A conv-3 model's accuracy drops to chance at d >= 4 because it literally
  cannot see tokens 4+ positions back.
- An attention model maintains accuracy across its full context window.
- A recurrent model shows smooth decay as the hidden state compresses
  distant information.

This is zero-shot: no deepcopy needed, no weight mutation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_RESTRICTED_VOCAB = 256
_TIMEOUT_S = 30.0


@dataclass(slots=True)
class BindingResult:
    """Result from binding range probe."""

    distance_accuracies: Dict[int, float] = None
    auc: float = 0.0
    status: str = "ok"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_distance_accuracies": self.distance_accuracies,
            "binding_auc": self.auc,
            "binding_status": self.status,
            "binding_elapsed_ms": self.elapsed_ms,
        }


def _generate_copy_batch(
    batch_size: int,
    distance: int,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """Generate sequences where token[i] == token[i - distance] for i >= distance.

    The first `distance` tokens are random. All subsequent tokens copy
    the token `distance` positions back. This creates a pure copy pattern
    that the model should predict if it can bind at that distance.
    """
    # Generate the first `distance` tokens randomly, then tile to fill
    prefix = torch.randint(1, _RESTRICTED_VOCAB, (batch_size, distance), device=device)
    # Repeat prefix to fill seq_len: token[i] = prefix[i % distance]
    repeats = (seq_len + distance - 1) // distance
    batch = prefix.repeat(1, repeats)[:, :seq_len]
    return batch


@torch.no_grad()
def binding_range_profile(
    model: nn.Module,
    distances: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    n_eval: int = 200,
    seq_len: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> BindingResult:
    """Measure copy-at-distance accuracy for each distance. Zero-shot.

    For each distance d, generates sequences where token[i] = token[i-d],
    then checks if the model's argmax prediction at position i matches
    token[i] for positions i >= d+1 (i.e., the model has seen both the
    source and at least one copy before predicting the next copy).

    Returns BindingResult with per-distance accuracies and AUC.
    """
    t0 = time.perf_counter()
    result = BindingResult(distance_accuracies={})

    was_training = model.training
    model.eval()

    try:
        for d in sorted(distances):
            if d + 2 > seq_len:
                result.distance_accuracies[d] = 0.0
                continue

            if time.perf_counter() - t0 > timeout_s:
                result.distance_accuracies[d] = 0.0
                result.status = "timeout"
                continue

            correct = 0
            total = 0
            remaining = n_eval

            while remaining > 0:
                bs = min(batch_size, remaining)
                input_ids = _generate_copy_batch(bs, d, seq_len, device)
                logits = model(input_ids)  # (B, seq_len, V)

                # Check predictions at positions d+1 through seq_len-1
                # Position i predicts token i+1, so logits[:, i, :] predicts input_ids[:, i+1]
                # We want positions where both source and copy pattern are established
                start = d  # logits at position d predicts token d+1 = copy of token 1
                end = seq_len - 1  # last valid prediction position

                if start >= end:
                    remaining -= bs
                    continue

                pred_logits = logits[:, start:end, :_RESTRICTED_VOCAB]
                targets = input_ids[:, start + 1 : end + 1]

                preds = pred_logits.argmax(dim=-1)  # (B, n_positions)
                matches = (preds == targets).float()
                correct += matches.sum().item()
                total += matches.numel()
                remaining -= bs

            acc = correct / max(total, 1)
            result.distance_accuracies[d] = round(acc, 4)

    except Exception as e:
        result.status = f"eval_failed: {e}"
    finally:
        model.train(was_training)

    if result.distance_accuracies:
        vals = list(result.distance_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
