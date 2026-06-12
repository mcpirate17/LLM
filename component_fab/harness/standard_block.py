"""Bog-standard test blocks for grading new components under identical conditions.

``LaneTestBlock`` is the canonical lane wrapper:
``input -> rmsnorm -> lane -> residual_add``. The wrappers are pure
``torch.nn.Module`` subclasses with no dependency on research/synthesis.
A component under test only needs to satisfy the ``[B, L, D] -> [B, L, D]``
shape contract.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .primitives import RMSNorm


class _LocalConvLane(nn.Module):
    """Symmetric (non-causal) local conv reference lane (range-audit baseline)."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class LaneTestBlock(nn.Module):
    """``input -> rmsnorm -> lane -> residual``."""

    def __init__(self, lane: nn.Module, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lane = lane

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lane(self.norm(x))


def make_lane_test_block(lane: nn.Module, dim: int) -> LaneTestBlock:
    return LaneTestBlock(lane, dim)


def lane_forward_for_mix_speed(
    lane: nn.Module, dim: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a lane in a ``LaneTestBlock`` and return a torch fn for mix_speed."""
    block = make_lane_test_block(lane, dim).eval()
    return block.__call__
