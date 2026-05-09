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

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

_RESTRICTED_VOCAB = 256
_TIMEOUT_S = 120.0


def _amp_context(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
            "induction_screening_auc": self.auc,
            "induction_steps_trained": self.steps_trained,
            "induction_status": self.status,
            "induction_elapsed_ms": self.elapsed_ms,
        }


def _generate_induction_batch(
    batch_size: int,
    gap: int,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate [A][B][noise x gap][A] sequences.

    Returns (input_ids, target_B).
    Sequence length: gap + 4 (A, B, noise*gap, A_repeat, predict_position)
    Actually: A B n1 n2 ... nG A = gap + 3 tokens, predict at position gap+2.
    """
    seq_len = gap + 3  # A B [noise x gap] A

    # Token A and B are random; noise excludes A by construction.
    A = torch.randint(
        1,
        _RESTRICTED_VOCAB,
        (batch_size,),
        device=device,
        generator=generator,
    )
    B = torch.randint(
        1,
        _RESTRICTED_VOCAB,
        (batch_size,),
        device=device,
        generator=generator,
    )

    noise_raw = torch.randint(
        1,
        _RESTRICTED_VOCAB - 1,
        (batch_size, gap),
        device=device,
        generator=generator,
    )
    noise = noise_raw + (noise_raw >= A.unsqueeze(1)).to(noise_raw.dtype)

    batch = torch.empty(
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Place pattern: position 0=A, position 1=B, positions 2..gap+1=noise, position gap+2=A
    batch[:, 0] = A
    batch[:, 1] = B
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
    seed: int | None = None,
) -> InductionResult:
    """Train a deepcopy on the induction task, measure accuracy per gap distance.

    Training uses gap=8 (a reasonable mid-range distance). Evaluation sweeps
    all requested gap distances to map the binding profile.

    Returns InductionResult with per-gap accuracies and AUC.
    """
    t0 = time.perf_counter()
    result = InductionResult(gap_accuracies={})
    train_gap = 8  # fixed training gap
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    try:
        probe_model = copy.deepcopy(model)
        probe_model.to(device)
        probe_model.train()
    except Exception as e:
        result.status = f"copy_failed: {e}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    opt = make_adamw(probe_model.parameters(), lr=lr)

    try:
        with disable_native_probe_dispatch(probe_model, device=device):
            # Train on fixed gap
            for step in range(1, n_train_steps + 1):
                if time.perf_counter() - t0 > timeout_s:
                    result.status = "timeout"
                    break

                input_ids, targets = _generate_induction_batch(
                    batch_size, train_gap, device, generator=generator
                )
                opt.zero_grad(set_to_none=True)

                with _amp_context(device):
                    logits = probe_model(input_ids)
                    # Predict B at the last position (after seeing second A)
                    pred_pos = input_ids.shape[1] - 1
                    pred_logits = logits[:, pred_pos, :_RESTRICTED_VOCAB]
                    loss = F.cross_entropy(pred_logits.float(), targets)

                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break

                loss.backward()
                clip_grad_norm(probe_model.parameters(), 1.0)
                opt.step()
                result.steps_trained = step

            # Evaluate across all gap distances
            probe_model.eval()
            with torch.inference_mode():
                for gap in sorted(gaps):
                    if time.perf_counter() - t0 > timeout_s:
                        result.gap_accuracies[gap] = 0.0
                        continue

                    correct = 0
                    total = 0
                    remaining = n_eval
                    while remaining > 0:
                        bs = min(batch_size, remaining)
                        inp, tgt = _generate_induction_batch(
                            bs,
                            gap,
                            device,
                            generator=generator,
                        )
                        out = probe_model(inp)
                        pred_pos = inp.shape[1] - 1
                        preds = out[:, pred_pos, :_RESTRICTED_VOCAB].argmax(dim=-1)
                        correct += (preds == tgt).sum().item()
                        total += tgt.numel()
                        remaining -= bs

                    result.gap_accuracies[gap] = round(correct / max(total, 1), 4)

    except Exception as e:
        result.status = f"train_failed: {e}"
    finally:
        del probe_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # AUC: mean accuracy across gaps, normalized to [0, 1]
    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
