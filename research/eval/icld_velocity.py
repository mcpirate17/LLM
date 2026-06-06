"""In-Context Loss Decay (ICLD) Velocity.

Measures how rapidly per-token loss decreases across sequence position on
synthetic Dyck (balanced-bracket) sequences. A reasoning-capable architecture
shows a steeply *negative* slope of loss vs position — each new context token
materially reduces loss on the next prediction. A bag-of-tokens model is
flat or noisy.

The metric is a single forward pass plus an OLS slope. No training is
required; the value reflects the model's *current weights* (at-init for
freshly-rebuilt models, post-screening for live-pipeline calls).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from ._trajectory_datasets import DYCK_VOCAB_SIZE, dyck_sequences

logger = logging.getLogger(__name__)

# Discard the very first MIN_CONTEXT positions when fitting the slope —
# at position 0 there's no context yet, so loss there is uninformative
# noise that drags the slope estimate.
_MIN_CONTEXT = 4


@dataclass(slots=True)
class ICLDResult:
    velocity: Optional[float] = None  # slope of loss vs position (negative is good)
    early_loss: Optional[float] = None  # mean loss in early positions
    late_loss: Optional[float] = None  # mean loss in late positions
    delta_loss: Optional[float] = None  # late - early (negative is good)
    seq_len: Optional[int] = None
    status: str = "init"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, float | int | str | None]:
        return {
            "fp_icld_velocity": self.velocity,
            "fp_icld_early_loss": self.early_loss,
            "fp_icld_late_loss": self.late_loss,
            "fp_icld_delta_loss": self.delta_loss,
            "fp_icld_seq_len": self.seq_len,
            "fp_icld_status": self.status,
            "fp_icld_elapsed_ms": self.elapsed_ms,
        }


def compute_icld_velocity(
    model: nn.Module,
    *,
    seq_len: int = 64,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    seed: int = 1234,
) -> ICLDResult:
    """Run the ICLD probe and return the per-position loss slope.

    Args:
        model: ``SynthesizedModel``-compatible module that emits
            next-token logits when called as ``model(input_ids)``.
        seq_len: probe sequence length. Longer = more positions = lower
            slope variance, at the cost of more compute.
        batch_size: number of independent Dyck sequences averaged over.
        device: cuda or cpu.
        seed: deterministic seed for the synthetic Dyck batch.
    """
    result = ICLDResult(seq_len=seq_len, status="failed")
    t0 = time.perf_counter()

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    device_str = str(dev)

    if seq_len <= _MIN_CONTEXT + 4:
        result.status = "seq_len_too_short"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    try:
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        ids = dyck_sequences(
            batch_size=batch_size,
            seq_len=seq_len,
            device=dev,
            generator=gen,
        )

        model.eval()
        with disable_native_probe_dispatch(model, device=device_str), torch.no_grad():
            logits = model(ids)  # (B, S, V)
            if logits.dim() != 3:
                result.status = "unexpected_logits_shape"
                return result

            # Targets are next-token: targets[t] = ids[t+1].
            # Per-position loss is at positions 0..S-2 predicting 1..S-1.
            shifted_logits = logits[:, :-1, :DYCK_VOCAB_SIZE]
            shifted_targets = ids[:, 1:]
            B, Sm1, V = shifted_logits.shape
            losses = F.cross_entropy(
                shifted_logits.reshape(-1, V),
                shifted_targets.reshape(-1),
                reduction="none",
            ).reshape(B, Sm1)  # (B, S-1)
            # Mean across batch — per-position curve.
            per_pos = losses.mean(dim=0)  # (S-1,)

            # Fit slope on positions [_MIN_CONTEXT .. end] to skip the
            # context-starved leading positions.
            xs = torch.arange(per_pos.numel(), dtype=torch.float32, device=dev)
            mask = xs >= _MIN_CONTEXT
            x_fit = xs[mask]
            y_fit = per_pos[mask]
            mean_x = x_fit.mean()
            mean_y = y_fit.mean()
            cov = ((x_fit - mean_x) * (y_fit - mean_y)).sum()
            var_x = ((x_fit - mean_x) ** 2).sum().clamp_min(1e-12)
            slope = float((cov / var_x).item())

            # Early/late loss snapshot for interpretability.
            split = (per_pos.numel() + _MIN_CONTEXT) // 2
            early_loss = float(per_pos[_MIN_CONTEXT:split].mean().item())
            late_loss = float(per_pos[split:].mean().item())

            result.velocity = slope
            result.early_loss = early_loss
            result.late_loss = late_loss
            result.delta_loss = late_loss - early_loss
            result.status = "ok"
    except RuntimeError as exc:
        logger.warning("ICLD probe failed: %s", exc)
        result.status = f"failed: {exc.__class__.__name__}"
    finally:
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return result
