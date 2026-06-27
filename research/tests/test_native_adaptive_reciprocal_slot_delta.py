from __future__ import annotations

import torch

from research.tools._scaling_lanes import NativeAdaptiveReciprocalSlotDeltaLane
from research.tools.scaling_blimp_study import _build_lane_factory


def test_native_adaptive_reciprocal_slot_delta_smoke() -> None:
    torch.manual_seed(0)
    lane = _build_lane_factory("native_adaptive_reciprocal_slot_delta")(16)

    assert isinstance(lane, NativeAdaptiveReciprocalSlotDeltaLane)
    assert lane.slot.use_delta_update is True
    assert lane.native.lane_a.max_recursive_steps == 4

    x = torch.randn(2, 9, 16, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    metrics = lane.last_gate_metrics
    assert metrics["raw_gate_mean"].shape == (3,)
    assert metrics["effective_gate_mean"].shape == (3,)
    assert metrics["weighted_branch_rms"].shape == (3,)
    assert metrics["gate_entropy"] > 0.0
    assert metrics["effective_gate_mean"][0] >= lane.GATE_FLOOR

    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert lane.gate.weight.grad is not None
    assert lane.reciprocal.reciprocal_logit_scale.grad is not None
    assert any(p.grad is not None for p in lane.slot.parameters())
