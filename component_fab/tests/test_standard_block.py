"""Smoke + behavior tests for component_fab.harness.standard_block."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.harness.standard_block import (
    CanonicalLaneEnvironment,
    make_canonical_lanes,
    make_compression_test_block,
    make_lane_test_block,
    make_routing_test_block,
)
from component_fab.metrics.mix_speed import measure_mix_speed


class _IdentityLane(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DenseSoftmaxRouter(nn.Module):
    def __init__(self, dim: int, n_lanes: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, n_lanes)

    def forward(
        self, normalized: torch.Tensor, lane_outputs: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.softmax(self.gate(normalized), dim=-1).unsqueeze(-1)
        return (weights * lane_outputs).sum(dim=2)


def test_canonical_lanes_have_three_distinct_regimes() -> None:
    env = make_canonical_lanes(dim=16)
    assert isinstance(env, CanonicalLaneEnvironment)
    x = torch.randn(2, 8, 16)
    a = env.local(x)
    b = env.global_(x)
    c = env.stateful(x)
    assert a.shape == x.shape
    assert b.shape == x.shape
    assert c.shape == x.shape
    assert not torch.allclose(a, b)
    assert not torch.allclose(b, c)


def test_lane_test_block_preserves_shape_and_adds_residual() -> None:
    block = make_lane_test_block(_IdentityLane(), dim=16)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_routing_test_block_runs_with_dense_router() -> None:
    dim = 16
    router = _DenseSoftmaxRouter(dim=dim, n_lanes=3)
    block = make_routing_test_block(router, dim=dim)
    x = torch.randn(2, 8, dim)
    y = block(x)
    assert y.shape == x.shape


def test_compression_test_block_paired_compress_restore() -> None:
    compress = nn.Linear(16, 4)
    restore = nn.Linear(4, 16)
    block = make_compression_test_block(compress, restore)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_lane_block_works_with_mix_speed() -> None:
    block = make_lane_test_block(_IdentityLane(), dim=16).eval()
    card = measure_mix_speed(block, seq_len=32, feature_dim=16, n_trials=2)
    assert card.is_pure_local
