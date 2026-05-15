"""Routing-health metric for routers / gates / lane-blenders.

Probes intrinsic behavior of a router under random inputs:
- ``routing_entropy_mean / std``: average per-token entropy of the
  routing distribution. Low mean = router is making confident choices;
  high std = entropy varies a lot across tokens / batches.
- ``load_balance_cv``: coefficient of variation of per-lane load.
  ``0.0`` = perfectly balanced, ``> 0.5`` = severely imbalanced.
- ``mode_collapse_propensity``: fraction of tokens for which the router
  picks the same dominant lane across batches. ``> 0.3`` indicates a
  router that ignores input.
- ``active_lane_fraction``: fraction of lanes that receive non-trivial
  load (above ``1 / (n_lanes * 4)``).

The router is invoked with a softmax-routable signal and is expected
to return either a probability tensor of shape ``[..., n_lanes]`` or a
hard one-hot of the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class RoutingHealthScorecard:
    routing_entropy_mean: float
    routing_entropy_std: float
    load_balance_cv: float
    mode_collapse_propensity: float
    active_lane_fraction: float
    n_lanes: int
    n_trials: int


def _ensure_probability(weights: torch.Tensor) -> torch.Tensor:
    if (weights < 0).any():
        weights = weights.clamp_min(0.0)
    sums = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / sums


def _entropy_per_token(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1e-12)
    return -(safe * safe.log()).sum(dim=-1)


def _load_balance_cv(per_lane_load: torch.Tensor) -> float:
    mean = per_lane_load.mean()
    if mean.item() <= 0.0:
        return 0.0
    return float(per_lane_load.std(unbiased=False).item() / mean.item())


def _mode_collapse_fraction(weights_by_trial: torch.Tensor) -> float:
    chosen = weights_by_trial.argmax(dim=-1)
    first = chosen[0]
    matches = (chosen == first).all(dim=0)
    return float(matches.float().mean().item())


def _active_lane_fraction(per_lane_load: torch.Tensor) -> float:
    n_lanes = per_lane_load.shape[0]
    threshold = 1.0 / max(n_lanes * 4, 1)
    return float((per_lane_load > threshold).float().mean().item())


def _random_input(
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    generator: torch.Generator,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        seq_len,
        feature_dim,
        generator=generator,
        dtype=dtype,
        device=device,
    )


def _route_once(
    router_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    n_lanes: int,
) -> torch.Tensor:
    with torch.no_grad():
        raw = router_fn(x)
    if raw.shape[-1] != n_lanes:
        raise ValueError(
            f"router must emit shape [..., {n_lanes}]; got {tuple(raw.shape)}"
        )
    return _ensure_probability(raw)


def measure_routing_health(
    router_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_lanes: int,
    seq_len: int = 64,
    feature_dim: int = 32,
    batch_size: int = 4,
    n_trials: int = 8,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> RoutingHealthScorecard:
    """Probe ``router_fn`` for entropy stability, load balance, and collapse."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def sampler() -> torch.Tensor:
        return _random_input(
            batch_size=batch_size,
            seq_len=seq_len,
            feature_dim=feature_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    entropies: list[torch.Tensor] = []
    per_lane_loads: list[torch.Tensor] = []
    weight_snapshots: list[torch.Tensor] = []

    fixed_input = sampler()
    for trial in range(n_trials):
        x = fixed_input if trial == 0 else sampler()
        weights = _route_once(router_fn, x, n_lanes=n_lanes)
        entropies.append(_entropy_per_token(weights).flatten())
        per_lane_loads.append(weights.sum(dim=tuple(range(weights.ndim - 1))).cpu())
        if x is fixed_input:
            weight_snapshots.append(weights.detach().cpu())
    for _ in range(min(4, n_trials)):
        weight_snapshots.append(
            _route_once(router_fn, sampler(), n_lanes=n_lanes).detach().cpu()
        )

    entropy_concat = torch.cat(entropies)
    summed_load = torch.stack(per_lane_loads).sum(dim=0)
    normalized_load = summed_load / summed_load.sum().clamp_min(1e-12)
    snapshots = torch.stack(weight_snapshots)

    return RoutingHealthScorecard(
        routing_entropy_mean=float(entropy_concat.mean().item()),
        routing_entropy_std=float(entropy_concat.std(unbiased=False).item()),
        load_balance_cv=_load_balance_cv(normalized_load),
        mode_collapse_propensity=_mode_collapse_fraction(snapshots),
        active_lane_fraction=_active_lane_fraction(normalized_load),
        n_lanes=n_lanes,
        n_trials=n_trials,
    )
