"""Mixing-quality subscore: how broadly AND globally a lane mixes positions.

The user objective names "maximum mixing" explicitly, but ``ranking.py`` had no
mixing axis at all. Mixing has two complementary facets, both intrinsic (random
inputs, init-time, no training, no DB):

- **reach** — does a perturbation at one position propagate FAR? Reuses
  :func:`measure_mix_speed` (``mixes_globally`` / ``is_pure_local`` /
  ``mix_half_life``). Separates a global mixer (softmax-attn, half-life 0) from a
  local op (conv k=3, pure-local) and a no-mix op (rmsnorm, peak response 0).
- **breadth** — how many DISTINCT predecessor positions does each output
  aggregate, and how independently? Computed from the finite-diff
  :func:`influence_matrix` as ``offdiag_mass × normalized_effective_rank``.

The product is deliberate: it scores an identity lane (no off-diagonal mass) and
a "collapse-to-one-summary-then-broadcast" lane (rank ~1) both LOW, while
rewarding a mixer that is broad AND diverse. Reach alone cannot tell a degenerate
global mixer from a good one; breadth is the discriminator among global mixers.

``mixing_subscore ∈ [0,1]`` is gated so a dead lane (zero response) scores 0. It
is an *intrinsic capability* signal — the composite in ``ranking.py`` gates the
credit behind ``STABILITY_FLOOR`` so a lane that mixes but cannot bind/induce
never wins on mixing alone (same pattern as the orthogonality lift).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .mix_speed import (
    MixSpeedScorecard,
    influence_matrix,
    measure_mix_speed,
)


@dataclass(frozen=True, slots=True)
class MixingQualityScorecard:
    mixing_subscore: float
    reach_subscore: float
    breadth_subscore: float
    offdiag_mass_fraction: float
    effective_rank: float
    mixes_globally: bool
    is_pure_local: bool
    mix_half_life: float
    peak_response_magnitude: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def _reach_subscore(card: MixSpeedScorecard) -> float:
    """Global > mid > local > dead, from the shared mix-speed classification."""
    if card.peak_response_magnitude <= 0.0:
        return 0.0
    if card.mixes_globally:
        return 0.9
    if card.is_pure_local:
        return 0.1
    return 0.5  # mixes beyond the local window but not globally


def _breadth_signals(infl: torch.Tensor) -> tuple[float, float, float]:
    """Off-diagonal mass fraction + effective rank of the influence matrix.

    ``infl[i, j]`` = response at output ``j`` when input ``i`` is perturbed
    (row = inject, col = response). The diagonal is self-response; every
    off-diagonal entry is cross-token influence. For a causal lane the
    ``j > i`` off-diagonal is the genuine "aggregate from a predecessor" mass
    and the ``j < i`` off-diagonal is ~0 (an acausal leak, caught by S0.5) — so
    total off-diagonal mass is a convention-free mixing fraction.

    Returns ``(breadth, offdiag_mass_fraction, effective_rank)``. ``breadth``
    is the product of the off-diagonal fraction and the normalized effective
    rank (participation ratio ``(Σσ)²/Σσ²`` scaled by ``L``): broad AND diverse.
    """
    L = int(infl.shape[0])
    total = float(infl.sum().item())
    if total <= 1e-12 or L <= 1:
        return 0.0, 0.0, 0.0
    diag = float(torch.diagonal(infl).sum().item())
    offdiag_mass = max(0.0, min(1.0, 1.0 - diag / total))
    # Effective rank via the participation ratio of singular values.
    s = torch.linalg.svdvals(infl)
    s_sum = float(s.sum().item())
    s2_sum = float((s * s).sum().item())
    if s_sum <= 0.0 or s2_sum <= 1e-15:
        erank = 1.0
    else:
        erank = (s_sum * s_sum) / s2_sum
    erank_norm = max(0.0, min(1.0, erank / L))
    breadth = offdiag_mass * erank_norm
    return breadth, offdiag_mass, erank


def measure_mixing_quality(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    feature_dim: int = 32,
    seq_len: int = 24,
    n_trials: int = 4,
    delta_scale: float = 1e-2,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> MixingQualityScorecard:
    """Reach (mix-speed) + breadth (influence matrix) → ``mixing_subscore``.

    Two intrinsic measurements on random inputs; no training, no registry. The
    forward fn must map ``[B, L, D]`` to ``[B, L, D]``.
    """
    reach_card = measure_mix_speed(
        forward_fn,
        seq_len=max(seq_len, 16),
        feature_dim=feature_dim,
        delta_scale=delta_scale,
        n_trials=max(n_trials, 4),
        device=device,
        dtype=dtype,
        seed=seed,
    )
    infl = influence_matrix(
        forward_fn,
        seq_len=seq_len,
        feature_dim=feature_dim,
        batch_size=1,
        delta_scale=delta_scale,
        n_trials=n_trials,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    reach = _reach_subscore(reach_card)
    breadth, offdiag_mass, erank = _breadth_signals(infl)

    if reach_card.peak_response_magnitude <= 0.0:
        mixing = 0.0
    else:
        mixing = 0.5 * reach + 0.5 * breadth

    return MixingQualityScorecard(
        mixing_subscore=round(mixing, 4),
        reach_subscore=round(reach, 4),
        breadth_subscore=round(breadth, 4),
        offdiag_mass_fraction=round(offdiag_mass, 4),
        effective_rank=round(erank, 4),
        mixes_globally=bool(reach_card.mixes_globally),
        is_pure_local=bool(reach_card.is_pure_local),
        mix_half_life=reach_card.mix_half_life,
        peak_response_magnitude=round(float(reach_card.peak_response_magnitude), 4),
    )


def mixing_scorecard_to_dict(card: MixingQualityScorecard) -> dict[str, Any]:
    half_life = card.mix_half_life
    return {
        "mixing_subscore": card.mixing_subscore,
        "mixing_reach_subscore": card.reach_subscore,
        "mixing_breadth_subscore": card.breadth_subscore,
        "mixing_offdiag_mass_fraction": card.offdiag_mass_fraction,
        "mixing_effective_rank": card.effective_rank,
        "mixing_mixes_globally": card.mixes_globally,
        "mixing_is_pure_local": card.is_pure_local,
        "mixing_half_life": None if math.isinf(half_life) else half_life,
        "mixing_peak_response": card.peak_response_magnitude,
        "mixing_notes": list(card.notes),
    }
