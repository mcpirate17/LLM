"""Regression tests: deep recursion blocks stay finite at random init.

Before the pre-norm fix (2026-06-04) ``RecursiveDepthBlock`` /
``RecursiveDepthRouterBlock`` re-fed an unnormalized residual stream into the
mixer every step, so its magnitude grew geometrically with depth x mixer-gain
and NaN'd deep / high-gain variants — the bulk of the measured-screen
``unstable`` rejects. Pre-norming the recursion input bounds each step's
contribution, so the stream is finite and depth-invariant. These tests pin that.
"""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.block_templates import (
    RecursiveDepthBlock,
    RecursiveDepthRouterBlock,
)


class _GainMixer(nn.Module):
    """A mixer with gain > 1 — the family that compounded to NaN."""

    def __init__(self, dim: int, gain: float = 3.0) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * self.lin(x)


def _gain_factory(gain: float):
    return lambda d: _GainMixer(d, gain)


def test_recursive_depth_router_finite_and_bounded_at_deep_high_gain() -> None:
    dim = 32
    absmaxes = []
    for depth in (4, 8, 16):
        torch.manual_seed(1)
        block = RecursiveDepthRouterBlock(_gain_factory(3.0), dim, max_depth=depth)
        x = torch.randn(2, 24, dim, requires_grad=True)
        out = block(x)
        grad = torch.autograd.grad(out.sum(), x)[0]
        assert torch.isfinite(out).all(), f"forward NaN/inf at depth={depth}"
        assert torch.isfinite(grad).all(), f"backward NaN/inf at depth={depth}"
        absmaxes.append(out.abs().max().item())
    # Pre-norm makes the output magnitude depth-invariant rather than geometric.
    assert max(absmaxes) < 4.0 * min(absmaxes)


def test_recursive_depth_block_finite_at_deep_high_gain() -> None:
    dim = 32
    for depth in (4, 8, 16):
        torch.manual_seed(1)
        block = RecursiveDepthBlock(_gain_factory(3.0), dim, max_depth=depth)
        x = torch.randn(2, 24, dim, requires_grad=True)
        out = block(x)
        grad = torch.autograd.grad(out.sum(), x)[0]
        assert torch.isfinite(out).all(), f"forward NaN/inf at depth={depth}"
        assert torch.isfinite(grad).all(), f"backward NaN/inf at depth={depth}"


def test_recursion_blocks_preserve_shape() -> None:
    dim = 16
    x = torch.randn(2, 8, dim)
    assert RecursiveDepthBlock(_gain_factory(1.0), dim, max_depth=3)(x).shape == x.shape
    assert (
        RecursiveDepthRouterBlock(_gain_factory(1.0), dim, max_depth=4)(x).shape
        == x.shape
    )
