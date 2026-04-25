"""Jacobian Effective Receptive Field (ERF) Density.

Measures how much information from each input position influences the
final-layer representation of the last token. A reasoning-capable architecture
maintains a dense, high-variance ERF across the context window — distant
positions retain the ability to influence the last-token output. Architectures
that bottleneck information transfer show sparse / exponentially-decaying ERF.

Computed at any single forward pass (no training required), but the returned
values reflect the model's *current weights*. For backfilled rows on rebuilt
models, this is at-init; for rows from live screening eval, this is post-step-750.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ._probe_runtime import disable_native_probe_dispatch
from .fingerprint_sensitivity import forward_model_from_embed

logger = logging.getLogger(__name__)


@dataclass
class JacobianERFResult:
    density: Optional[float] = None
    variance: Optional[float] = None
    decay_slope: Optional[float] = None
    last_position_norm: Optional[float] = None
    first_position_norm: Optional[float] = None
    status: str = "init"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, float | str | None]:
        return {
            "fp_jacobian_erf_density": self.density,
            "fp_jacobian_erf_variance": self.variance,
            "fp_jacobian_erf_decay_slope": self.decay_slope,
            "fp_jacobian_erf_last_norm": self.last_position_norm,
            "fp_jacobian_erf_first_norm": self.first_position_norm,
            "fp_jacobian_erf_status": self.status,
            "fp_jacobian_erf_elapsed_ms": self.elapsed_ms,
        }


def compute_jacobian_erf(
    model: nn.Module,
    *,
    seq_len: int = 64,
    vocab_size: int = 32000,
    device: str | torch.device = "cuda",
    n_samples: int = 4,
    nonzero_threshold: float = 1e-3,
) -> JacobianERFResult:
    """Run the Jacobian ERF probe.

    Backwards-pass ``J(last_token_output, input_embeddings)`` and reduce it to
    a per-position influence vector. Density, variance, and the slope of
    log-influence vs position-distance-from-last summarize whether the
    architecture preserves long-range information.

    Args:
        model: ``SynthesizedModel``-compatible module exposing ``.embed`` and
            either ``_fingerprint_forward_from_embed`` or ``layers``.
        seq_len: probe sequence length.
        vocab_size: probe vocabulary size for token sampling.
        device: cuda or cpu.
        n_samples: batch size for averaging across random sequences.
        nonzero_threshold: per-position influence below this counts as zero
            for the density measurement.
    """
    result = JacobianERFResult(status="failed")
    t0 = time.perf_counter()

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    device_str = str(dev)

    try:
        model.eval()
        with (
            disable_native_probe_dispatch(model, device=device_str),
            torch.enable_grad(),
        ):
            ids = torch.randint(0, vocab_size, (n_samples, seq_len), device=dev)
            embed = model.embed(ids).detach().requires_grad_(True)

            x = forward_model_from_embed(model, embed)
            if not x.requires_grad:
                result.status = "output_no_grad"
                return result

            # Reduce to last-token output, sum across batch and feature dim
            # so a single backward gives the gradient w.r.t. embed for that
            # signal. Using sum instead of mean keeps the magnitude raw.
            last_output = x[:, -1, :]
            grad_target = last_output.sum()

            (jacobian,) = torch.autograd.grad(
                grad_target,
                embed,
                retain_graph=False,
                create_graph=False,
            )

            # jacobian shape: (B, S, D_embed). Per-position influence is the
            # L1 norm across embed dim, then averaged across batch.
            per_position = jacobian.abs().sum(dim=-1).mean(dim=0)  # (S,)

            density = float((per_position > nonzero_threshold).float().mean().item())
            variance = float(per_position.var(unbiased=False).item())
            last_norm = float(per_position[-1].item())
            first_norm = float(per_position[0].item())

            # Decay slope: regress log(influence) on distance-from-last.
            # Steeper negative slope = stronger decay → bottlenecked info flow.
            distances = torch.arange(
                seq_len, device=dev, dtype=torch.float32
            )  # 0..S-1 from front
            distances = (seq_len - 1) - distances  # 0 at last position, S-1 at first
            log_inf = torch.log(per_position.clamp_min(1e-12))
            mean_d = distances.mean()
            mean_l = log_inf.mean()
            cov = ((distances - mean_d) * (log_inf - mean_l)).sum()
            var_d = ((distances - mean_d) ** 2).sum().clamp_min(1e-12)
            decay_slope = float((cov / var_d).item())

            result.density = density
            result.variance = variance
            result.decay_slope = decay_slope
            result.last_position_norm = last_norm
            result.first_position_norm = first_norm
            result.status = "ok"
    except RuntimeError as exc:
        logger.warning("Jacobian ERF probe failed: %s", exc)
        result.status = f"failed: {exc.__class__.__name__}"
    finally:
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return result
