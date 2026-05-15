"""Smoke + behavior tests for component_fab.metrics.routing_health."""

from __future__ import annotations

import math

import torch
from torch import nn

from component_fab.metrics.routing_health import measure_routing_health


class _UniformRouter(nn.Module):
    def __init__(self, n_lanes: int) -> None:
        super().__init__()
        self.n_lanes = n_lanes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((*x.shape[:-1], self.n_lanes), 1.0 / self.n_lanes)


class _CollapsedRouter(nn.Module):
    def __init__(self, n_lanes: int) -> None:
        super().__init__()
        self.n_lanes = n_lanes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*x.shape[:-1], self.n_lanes)
        out[..., 0] = 1.0
        return out


class _LearnedSoftmaxRouter(nn.Module):
    def __init__(self, dim: int, n_lanes: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, n_lanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=-1)


def test_uniform_router_has_max_entropy_and_zero_load_cv() -> None:
    n_lanes = 4
    card = measure_routing_health(
        _UniformRouter(n_lanes),
        n_lanes=n_lanes,
        seq_len=32,
        feature_dim=8,
        batch_size=2,
        n_trials=4,
    )
    assert card.routing_entropy_mean == pytest_approx(math.log(n_lanes))
    assert card.load_balance_cv < 1e-5
    assert card.active_lane_fraction == 1.0


def test_collapsed_router_has_zero_entropy_and_full_collapse() -> None:
    n_lanes = 4
    card = measure_routing_health(
        _CollapsedRouter(n_lanes),
        n_lanes=n_lanes,
        seq_len=32,
        feature_dim=8,
        batch_size=2,
        n_trials=4,
    )
    assert card.routing_entropy_mean < 1e-4
    assert card.load_balance_cv > 1.0
    assert card.mode_collapse_propensity == 1.0
    assert card.active_lane_fraction == 1.0 / n_lanes


def test_learned_softmax_router_partially_collapses_at_init() -> None:
    n_lanes = 3
    card = measure_routing_health(
        _LearnedSoftmaxRouter(dim=16, n_lanes=n_lanes),
        n_lanes=n_lanes,
        seq_len=32,
        feature_dim=16,
        batch_size=2,
        n_trials=4,
    )
    assert 0.0 < card.routing_entropy_mean < math.log(n_lanes)
    assert card.routing_entropy_std >= 0.0


def test_shape_violation_raises() -> None:
    class _WrongShape(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*x.shape[:-1], 5)

    try:
        measure_routing_health(_WrongShape(), n_lanes=3)
    except ValueError:
        return
    raise AssertionError("router emitting wrong K should raise ValueError")


def pytest_approx(expected: float, tol: float = 1e-3):
    class _Approx:
        def __eq__(self, actual: object) -> bool:
            try:
                return abs(float(actual) - expected) <= tol  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False

    return _Approx()
