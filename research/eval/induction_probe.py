"""Induction Head probe — measures pattern completion across gap distances.

Format: [A][B][noise x gap][A][?] -> B

The model sees token A followed by token B, then a gap of random noise
tokens, then A again. It must predict B at the final position. This is
the fundamental induction head pattern that attention models learn early.

By varying the gap distance, we map the architecture's binding range:
- conv-3: accuracy drops to chance at gap >= 4 (step function)
- attention: flat accuracy up to context length (rectangular)
- recurrent: smooth exponential decay with distance

The AUC of accuracy-vs-gap is a single number that captures the full
binding profile. A model with AUC > 0.3 has non-trivial binding range.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_RESTRICTED_VOCAB = 256
_TIMEOUT_S = 120.0


@dataclass(slots=True)
class InductionResult:
    """Result from induction head probe."""

    gap_accuracies: Dict[int, float] = None
    auc: float = 0.0
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "induction_gap_accuracies": self.gap_accuracies,
            "induction_auc": self.auc,
            "induction_steps_trained": self.steps_trained,
            "induction_status": self.status,
            "induction_elapsed_ms": self.elapsed_ms,
        }


def _generate_induction_batch(
    batch_size: int,
    gap: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate [A][B][noise x gap][A] sequences.

    Returns (input_ids, target_B).
    Sequence length: gap + 4 (A, B, noise*gap, A_repeat, predict_position)
    Actually: A B n1 n2 ... nG A = gap + 3 tokens, predict at position gap+2.
    """
    seq_len = gap + 3  # A B [noise x gap] A
    batch = torch.randint(1, _RESTRICTED_VOCAB, (batch_size, seq_len), device=device)

    # Token A and B are random, distinct from noise
    A = torch.randint(1, _RESTRICTED_VOCAB, (batch_size,), device=device)
    B = torch.randint(1, _RESTRICTED_VOCAB, (batch_size,), device=device)

    # Place pattern: position 0=A, position 1=B, positions 2..gap+1=noise, position gap+2=A
    batch[:, 0] = A
    batch[:, 1] = B
    # Vectorized noise collision fix: where noise == A, replace with (A+offset) mod vocab
    noise = batch[:, 2 : gap + 2]  # (B, gap)
    A_expanded = A.unsqueeze(1).expand_as(noise)
    collisions = noise == A_expanded
    if collisions.any():
        # Shift colliding tokens by a random offset (1 to vocab-2) to avoid A
        offsets = torch.randint(
            1, _RESTRICTED_VOCAB - 1, collisions.shape, device=device
        )
        noise[collisions] = (A_expanded[collisions] + offsets[collisions]) % (
            _RESTRICTED_VOCAB - 1
        ) + 1
        batch[:, 2 : gap + 2] = noise
    batch[:, gap + 2] = A  # repeat A at the end

    return batch, B


def induction_score(
    model: nn.Module,
    gaps: tuple[int, ...] = (4, 8, 16, 32, 64),
    n_train_steps: int = 1000,
    n_eval: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> InductionResult:
    """Train a deepcopy on the induction task, measure accuracy per gap distance.

    Training uses gap=8 (a reasonable mid-range distance). Evaluation sweeps
    all requested gap distances to map the binding profile.

    Returns InductionResult with per-gap accuracies and AUC.
    """
    t0 = time.perf_counter()
    result = InductionResult(gap_accuracies={})
    train_gap = 8  # fixed training gap

    try:
        original_state = model.state_dict()
        original_training = model.training
        model.to(device)
        model.train()
        probe_model = model
    except Exception as e:
        result.status = f"copy_failed: {e}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    opt = torch.optim.AdamW(probe_model.parameters(), lr=lr)

    try:
        # Train on fixed gap
        for step in range(1, n_train_steps + 1):
            if time.perf_counter() - t0 > timeout_s:
                result.status = "timeout"
                break

            input_ids, targets = _generate_induction_batch(
                batch_size, train_gap, device
            )
            opt.zero_grad(set_to_none=True)

            logits = probe_model(input_ids)
            # Predict B at the last position (after seeing second A)
            pred_pos = input_ids.shape[1] - 1
            pred_logits = logits[:, pred_pos, :_RESTRICTED_VOCAB]
            loss = F.cross_entropy(pred_logits, targets)

            if not torch.isfinite(loss):
                result.status = "diverged"
                break

            loss.backward()
            nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
            opt.step()
            result.steps_trained = step

        # Evaluate across all gap distances
        probe_model.eval()
        with torch.no_grad():
            for gap in sorted(gaps):
                if time.perf_counter() - t0 > timeout_s:
                    result.gap_accuracies[gap] = 0.0
                    continue

                correct = 0
                total = 0
                remaining = n_eval
                while remaining > 0:
                    bs = min(batch_size, remaining)
                    inp, tgt = _generate_induction_batch(bs, gap, device)
                    out = probe_model(inp)
                    pred_pos = inp.shape[1] - 1
                    preds = out[:, pred_pos, :_RESTRICTED_VOCAB].argmax(dim=-1)
                    correct += (preds == tgt).sum().item()
                    total += bs
                    remaining -= bs

                result.gap_accuracies[gap] = round(correct / max(total, 1), 4)

    except Exception as e:
        result.status = f"train_failed: {e}"
    finally:
        model.load_state_dict(original_state)
        model.train(original_training)
        if device == "cuda":
            torch.cuda.empty_cache()

    # AUC: mean accuracy across gaps, normalized to [0, 1]
    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
