"""Shared name -> lane-factory mapping for benchmark/audit reference baselines.

``tools/run_range_audit`` and ``tools/run_surprise_memory_bench`` each kept
overlapping literal dicts of the same constructors; both now import this
mapping and extend it with their tool-specific entries/labels.

Factories take ``dim`` and return a fresh lane. Note: lanes whose
constructors intentionally raise ``NotImplementedError`` (LegendreSSMLane,
PowerSemiringMemoryLane) must not be added here.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from component_fab.generator.memory_primitives import (
    PadicSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)
from component_fab.generator.primitive_templates import (
    LinearStateSpaceLane,
    TropicalAttention,
)

LaneFactory = Callable[[int], nn.Module]

REFERENCE_LANES: dict[str, LaneFactory] = {
    "tropical_attention": lambda dim: TropicalAttention(dim),
    "linear_ssm": lambda dim: LinearStateSpaceLane(dim),
    "tropical_surprise_memory": lambda dim: TropicalSurpriseMemoryLane(dim),
    "padic_surprise_memory": lambda dim: PadicSurpriseMemoryLane(dim),
}
