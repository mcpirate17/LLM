"""Tiered pretest probes adapted from research/eval/ screening pipeline.

The research screening pipeline tiers probes from cheapest to most expensive:

- **S0.5**: forward-only stability + causality gates. No training.
  Adapted from ``research/eval/sandbox.py:_run_causality_gate`` and the
  stability probe. Catches acausal information leakage and pathological
  numerical behavior in milliseconds.
- **S1.0 AR**: short-training associative recall. Adapted from
  ``research/eval/associative_recall.py``. Discriminates retrieval-capable
  architectures from lossy-state ones (Mamba, RWKV, conv) — the SAME
  distinction the research probe makes.

Operate on the fab's ``[B, L, D]`` continuous-vector setting (no vocab).
The autonomous validator runs S0.5 first; if it passes, runs S1.0 AR.
A component must clear S0.5 to be considered for AR; AR results then
feed the composite score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from .training_probe import train_lane_head

logger = logging.getLogger(__name__)


# ----------------------------- S0.5 gates -----------------------------


@dataclass(frozen=True, slots=True)
class S05GateResult:
    stability_passed: bool
    causality_passed: bool
    max_output_abs: float
    max_first_half_drift: float
    output_finite: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.stability_passed and self.causality_passed


def causality_stability_gate(
    lane_block: nn.Module,
    *,
    seq_len: int = 32,
    dim: int = 32,
    batch_size: int = 2,
    causality_threshold: float = 0.05,
    seed: int = 0,
) -> S05GateResult:
    """Combined S0.5 stability + causality gate (forward-only, no training).

    Stability: forward on random input must produce finite output.
    Causality: scramble the second half of the input; the first half of
    the output must be unchanged within ``causality_threshold`` (max abs).
    Adapted from ``research/eval/sandbox.py:_run_causality_gate``.
    """
    torch.manual_seed(seed)
    lane_block.eval()
    try:
        with torch.no_grad():
            x = torch.randn(batch_size, seq_len, dim)
            y = lane_block(x)
            max_abs = float(y.abs().max().item())
            finite = bool(torch.isfinite(y).all().item())

            x_mod = x.clone()
            midpoint = seq_len // 2
            x_mod[:, midpoint:] = torch.randn(batch_size, seq_len - midpoint, dim)
            y_mod = lane_block(x_mod)
            first_half_drift = float(
                (y[:, :midpoint] - y_mod[:, :midpoint]).abs().max().item()
            )
    except Exception as exc:  # noqa: BLE001
        return S05GateResult(
            stability_passed=False,
            causality_passed=False,
            max_output_abs=0.0,
            max_first_half_drift=float("inf"),
            output_finite=False,
            notes=(f"{type(exc).__name__}: {exc}",),
        )

    return S05GateResult(
        stability_passed=finite and max_abs < 1e6,
        causality_passed=finite and first_half_drift < causality_threshold,
        max_output_abs=max_abs,
        max_first_half_drift=first_half_drift,
        output_finite=finite,
    )


# ----------------------------- S1.0 AR --------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    name: str
    sample_fn: Callable[
        [int, int, int, torch.Generator],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    pass_threshold: float
    n_train_steps: int = 60
    learning_rate: float = 3e-3
    batch_size: int = 8


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    probe_name: str
    initial_query_mse: float
    final_query_mse: float
    baseline_mse: float
    relative_recall: float
    passes: bool
    trained_successfully: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _sample_ar_task(
    batch_size: int,
    seq_len: int,
    dim: int,
    generator: torch.Generator,
    *,
    n_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate AR task: [k0, v0, k1, v1, ..., k_{N-1}, v_{N-1}, q_key, q_target].

    Returns (input, target, query_mask).
    - input[:, 2N] = q_key (copy of one of the keys)
    - target[:, 2N+1] = v_i where k_i was the key copied to q_key
    - query_mask[:, 2N+1] = 1; other positions = 0
    """
    min_len = 2 * n_pairs + 2
    if seq_len < min_len:
        raise ValueError(f"seq_len {seq_len} too short for AR with {n_pairs} pairs")

    x = torch.randn(batch_size, seq_len, dim, generator=generator)
    target = x.clone()
    mask = torch.zeros((batch_size, seq_len), dtype=x.dtype)
    chosen = torch.randint(0, n_pairs, (batch_size,), generator=generator)
    for b in range(batch_size):
        k_idx = int(chosen[b].item())
        x[b, 2 * n_pairs] = x[b, 2 * k_idx]
        target[b, 2 * n_pairs + 1] = x[b, 2 * k_idx + 1]
    mask[:, 2 * n_pairs + 1] = 1.0
    return x, target, mask


def make_ar_probe(
    n_pairs: int,
    *,
    name: str | None = None,
    pass_threshold: float = 0.5,
    n_train_steps: int = 60,
) -> CapabilityProbe:
    name = name or f"ar_{n_pairs}pairs"
    return CapabilityProbe(
        name=name,
        sample_fn=lambda b, l, d, g: _sample_ar_task(b, l, d, g, n_pairs=n_pairs),
        pass_threshold=pass_threshold,
        n_train_steps=n_train_steps,
    )


# Two AR difficulty levels — matches the S1.0 AR vs deeper AR distinction
# (user's "ar0.5, ar1.0" terminology):
DEFAULT_CAPABILITY_PROBES: tuple[CapabilityProbe, ...] = (
    make_ar_probe(n_pairs=2, name="ar_easy", n_train_steps=40),
    make_ar_probe(n_pairs=5, name="ar_medium", n_train_steps=60),
)


def _eval_loss(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    with torch.no_grad():
        y = model(x)
    diff = (y - target).pow(2).sum(dim=-1)
    masked = diff * mask
    n = mask.sum().clamp_min(1.0)
    return float((masked.sum() / n).item())


def _baseline_mse(
    lane_block: nn.Module,
    probe: CapabilityProbe,
    *,
    seq_len: int,
    dim: int,
    seed: int,
) -> float:
    """Loss with no training — just initial forward."""
    gen = torch.Generator().manual_seed(seed)
    x, target, mask = probe.sample_fn(probe.batch_size, seq_len, dim, gen)
    return _eval_loss(lane_block, x, target, mask)


def train_and_score(
    lane_block: nn.Module,
    probe: CapabilityProbe,
    *,
    seq_len: int,
    dim: int,
    seed: int = 0,
) -> CapabilityResult:
    """Train ``lane_block`` briefly on ``probe`` and report retrieval quality."""
    initial = _baseline_mse(
        lane_block, probe, seq_len=seq_len, dim=dim, seed=seed + 999
    )
    gen = torch.Generator().manual_seed(seed)

    def sample() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, target, mask = probe.sample_fn(probe.batch_size, seq_len, dim, gen)
        return x, (target, mask)

    def masked_mse(
        y: torch.Tensor, target_mask: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        target, mask = target_mask
        diff = (y - target).pow(2).sum(dim=-1)
        return (diff * mask).sum() / mask.sum().clamp_min(1.0)

    final = initial
    trained = True
    try:
        lane_block.train()
        trace = train_lane_head(
            lane_block,
            lane_block.parameters(),
            sample,
            masked_mse,
            n_train_steps=probe.n_train_steps,
            learning_rate=probe.learning_rate,
        )
        final = trace.final_loss
    except Exception:  # noqa: BLE001 - one bad lane must not abort the cohort
        logger.warning(
            "capability probe %r failed; scoring 0.0", probe.name, exc_info=True
        )
        trained = False
    lane_block.eval()
    relative = 1.0 - (final / max(initial, 1e-12))
    relative = max(0.0, min(1.0, relative))
    passes = trained and relative >= probe.pass_threshold
    return CapabilityResult(
        probe_name=probe.name,
        initial_query_mse=initial,
        final_query_mse=final,
        baseline_mse=initial,
        relative_recall=relative,
        passes=passes,
        trained_successfully=trained,
    )
