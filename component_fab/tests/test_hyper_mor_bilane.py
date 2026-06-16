"""Tests for the integrated Native Hyperbolic Surprise MoR lane."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.hyper_mor_bilane import (
    HyperbolicMoRSurpriseRefineMLPBiLane,
)
from component_fab.generator.primitive_templates import HyperbolicAttention


def _lane(dim: int = 64, floor: float | None = None, router_hidden: int = 16):
    cls = type(
        "HyperMoRTest",
        (HyperbolicMoRSurpriseRefineMLPBiLane,),
        {"ROUTER_HIDDEN": router_hidden},
    )
    lane = cls(dim, memory_dim=16, max_recursive_steps=5)
    if floor is not None:
        lane.SURPRISE_FLOOR = floor
    return lane


def test_builds_with_three_mechanisms() -> None:
    lane = _lane()
    # lane_a is the surprise-MoR trunk (carries the halting router + ponder cost).
    assert hasattr(lane.lane_a, "halt_head")
    assert lane.last_ponder_cost is not None or lane.lane_a.last_ponder_cost is None
    # lane_b is the hyperbolic addressing pathway with a learned curvature.
    assert isinstance(lane.lane_b, HyperbolicAttention)
    assert lane.lane_b.log_curvature.requires_grad
    # gate is the 1-logit sigmoid blend.
    assert lane.gate.out_features == 1


def test_forward_shape_and_grad_reaches_both_mechanisms() -> None:
    lane = _lane()
    x = torch.randn(2, 24, 64, requires_grad=True)
    y = lane(x)
    assert y.shape == (2, 24, 64)
    y.sum().backward()
    # curvature (hyperbolic) AND the halting router (MoR) both receive gradient.
    assert lane.lane_b.log_curvature.grad is not None
    assert lane.lane_b.log_curvature.grad.abs().sum() > 0
    halt_grad = lane.lane_a.halt_head[0].weight.grad
    assert halt_grad is not None and halt_grad.abs().sum() > 0


def test_surprise_floor_protects_the_novel_trunk() -> None:
    # Even with the gate forced fully toward hyperbolic (logit -> -inf => gate 0),
    # the floor keeps >= SURPRISE_FLOOR of the mix on the surprise-MoR trunk.
    floor = 0.3
    lane = _lane(floor=floor)
    with torch.no_grad():
        lane.gate.weight.zero_()
        lane.gate.bias.fill_(-50.0)  # sigmoid -> ~0  => trunk weight -> floor
    x = torch.randn(2, 16, 64)
    lane(x)
    assert lane.last_trunk_frac == pytest.approx(floor, abs=1e-3)


def test_floor_zero_recovers_plain_gate() -> None:
    lane = _lane(floor=0.0)
    with torch.no_grad():
        lane.gate.weight.zero_()
        lane.gate.bias.fill_(50.0)  # sigmoid -> ~1 => trunk weight -> 1
    lane(torch.randn(1, 8, 64))
    assert lane.last_trunk_frac == pytest.approx(1.0, abs=1e-3)
