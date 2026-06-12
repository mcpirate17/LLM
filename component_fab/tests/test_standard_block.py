"""Smoke + behavior tests for component_fab.harness.standard_block."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.harness.standard_block import make_lane_test_block
from component_fab.metrics.mix_speed import measure_mix_speed


class _IdentityLane(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_lane_test_block_preserves_shape_and_adds_residual() -> None:
    block = make_lane_test_block(_IdentityLane(), dim=16)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_lane_block_works_with_mix_speed() -> None:
    block = make_lane_test_block(_IdentityLane(), dim=16).eval()
    card = measure_mix_speed(block, seq_len=32, feature_dim=16, n_trials=2)
    assert card.is_pure_local
