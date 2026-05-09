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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ._probe_runtime import disable_native_probe_dispatch

logger = logging.getLogger(__name__)

_RESTRICTED_VOCAB = 256
_TIMEOUT_S = 30.0
_COPY_INDEX_CACHE_LIMIT = 32
_COPY_INDEX_CACHE: "OrderedDict[tuple[int, int, str], torch.Tensor]" = OrderedDict()


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
            "binding_screening_auc": self.auc,
            "binding_status": self.status,
            "binding_elapsed_ms": self.elapsed_ms,
        }


def _generate_copy_batch(
    batch: torch.Tensor,
    prefix_buf: torch.Tensor,
    distance: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences where token[i] == token[i - distance] for i >= distance.

    The first `distance` tokens are random. All subsequent tokens copy
    the token `distance` positions back. This creates a pure copy pattern
    that the model should predict if it can bind at that distance.
    """
    batch_size, seq_len = batch.shape
    device = batch.device
    cache_key = (int(seq_len), int(distance), str(device))
    copy_idx = _COPY_INDEX_CACHE.get(cache_key)
    if copy_idx is None:
        copy_idx = torch.arange(seq_len, device=device, dtype=torch.long).remainder(
            distance
        )
        _COPY_INDEX_CACHE[cache_key] = copy_idx
        while len(_COPY_INDEX_CACHE) > _COPY_INDEX_CACHE_LIMIT:
            _COPY_INDEX_CACHE.popitem(last=False)
    else:
        _COPY_INDEX_CACHE.move_to_end(cache_key)

    prefix = prefix_buf[:batch_size, :distance]
    prefix.copy_(
        torch.randint(
            1,
            _RESTRICTED_VOCAB,
            prefix.shape,
            device=device,
            generator=generator,
        )
    )
    batch.copy_(prefix[:, copy_idx])
    return batch


@torch.inference_mode()
def binding_range_profile(
    model: nn.Module,
    distances: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    n_eval: int = 200,
    seq_len: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
    seed: int | None = None,
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
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    was_training = model.training
    model.eval()

    try:
        with disable_native_probe_dispatch(model, device=device):
            max_distance = max((d for d in distances if d > 0), default=1)
            batch_buf = torch.empty(
                (batch_size, seq_len), dtype=torch.long, device=device
            )
            prefix_buf = torch.empty(
                (batch_size, max_distance), dtype=torch.long, device=device
            )
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

                start = d
                end = seq_len - 1
                if start >= end:
                    result.distance_accuracies[d] = 0.0
                    continue

                while remaining > 0:
                    bs = min(batch_size, remaining)
                    input_ids = _generate_copy_batch(
                        batch_buf[:bs],
                        prefix_buf[:bs],
                        d,
                        generator=generator,
                    )
                    logits = model(input_ids)  # (B, seq_len, V)

                    # Check predictions at positions d+1 through seq_len-1
                    # Position i predicts token i+1, so logits[:, i, :] predicts input_ids[:, i+1]
                    # We want positions where both source and copy pattern are established
                    pred_logits = logits[:, start:end, :_RESTRICTED_VOCAB]
                    targets = input_ids[:, start + 1 : end + 1]

                    preds = pred_logits.argmax(dim=-1)  # (B, n_positions)
                    correct += preds.eq(targets).sum().item()
                    total += targets.numel()
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
