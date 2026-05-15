"""Bog-standard test blocks for grading new components under identical conditions.

Each fab category gets a canonical wrapper:
- ``LaneTestBlock``: ``input -> rmsnorm -> lane -> residual_add``.
- ``RoutingTestBlock``: ``input -> rmsnorm -> router(canonical_lanes) -> output``
  with three fixed reference lanes (local, global, stateful) so different
  routers are graded against the same downstream substrate.
- ``CompressionTestBlock``: ``input -> compress -> restore -> output``.

The wrappers are pure ``torch.nn.Module`` subclasses with no dependency
on research/synthesis. A component under test only needs to satisfy the
shape contract declared in the corresponding rules JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CanonicalLaneEnvironment:
    """Three reference lanes covering distinct mixing regimes.

    Routers placed inside ``RoutingTestBlock`` route over these lanes.
    The lanes are deliberately simple and span the lane substrate so that
    a router's grading reflects its routing behavior, not the lane bodies.
    """

    local: nn.Module
    global_: nn.Module
    stateful: nn.Module


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return self.weight * (x / scale)


class _LocalConvLane(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class _GlobalMeanLane(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1, keepdim=True).expand_as(x))


class _CausalRunningMeanLane(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        weights = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(
            1, -1, 1
        )
        return self.proj(x.cumsum(dim=1) / weights)


def make_canonical_lanes(dim: int) -> CanonicalLaneEnvironment:
    return CanonicalLaneEnvironment(
        local=_LocalConvLane(dim),
        global_=_GlobalMeanLane(dim),
        stateful=_CausalRunningMeanLane(dim),
    )


class LaneTestBlock(nn.Module):
    """``input -> rmsnorm -> lane -> residual``."""

    def __init__(self, lane: nn.Module, dim: int) -> None:
        super().__init__()
        self.norm = _RMSNorm(dim)
        self.lane = lane

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lane(self.norm(x))


class RoutingTestBlock(nn.Module):
    """``input -> rmsnorm -> router(canonical_lanes) -> output``.

    The router is any module callable as
    ``router(normalized_input, lane_outputs) -> combined`` where
    ``lane_outputs`` is a tensor of shape ``[B, L, K, D]``. This contract
    matches dense, top-k, and hard-argmax routers alike.
    """

    def __init__(
        self,
        router: nn.Module,
        environment: CanonicalLaneEnvironment,
        dim: int,
    ) -> None:
        super().__init__()
        self.norm = _RMSNorm(dim)
        self.router = router
        self.lanes = nn.ModuleList(
            [environment.local, environment.global_, environment.stateful]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        lane_outputs = torch.stack([lane(normalized) for lane in self.lanes], dim=2)
        return self.router(normalized, lane_outputs)


class CompressionTestBlock(nn.Module):
    """``input -> compress -> restore -> output``."""

    def __init__(self, compress: nn.Module, restore: nn.Module) -> None:
        super().__init__()
        self.compress = compress
        self.restore = restore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.restore(self.compress(x))


def make_lane_test_block(lane: nn.Module, dim: int) -> LaneTestBlock:
    return LaneTestBlock(lane, dim)


def make_routing_test_block(
    router: nn.Module,
    dim: int,
    environment: CanonicalLaneEnvironment | None = None,
) -> RoutingTestBlock:
    return RoutingTestBlock(router, environment or make_canonical_lanes(dim), dim)


def make_compression_test_block(
    compress: nn.Module, restore: nn.Module
) -> CompressionTestBlock:
    return CompressionTestBlock(compress, restore)


def lane_forward_for_mix_speed(
    lane: nn.Module, dim: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a lane in a ``LaneTestBlock`` and return a torch fn for mix_speed."""
    block = make_lane_test_block(lane, dim).eval()
    return block.__call__
