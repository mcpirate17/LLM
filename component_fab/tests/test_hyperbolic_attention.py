"""HyperbolicAttention lane: Lorentz-model distance scoring, learned curvature.

Guards the properties the nano gate relies on: strict causality, finite output +
gradients (the acosh-near-1 region is the numerically delicate part), learnable
curvature/temperature, and param-parity with the reciprocal twin it's meant to
replace (so the gate comparison is controlled, not confounded by capacity).
"""

from __future__ import annotations

import torch

from component_fab.generator.primitive_templates import (
    HyperbolicAttention,
    ReciprocalRankAttention,
)


def test_forward_shape_and_finite():
    lane = HyperbolicAttention(64, use_rope=True)
    y = lane(torch.randn(2, 24, 64))
    assert y.shape == (2, 24, 64)
    assert torch.isfinite(y).all()


def test_strict_causality():
    torch.manual_seed(0)
    lane = HyperbolicAttention(64, use_rope=True)
    x = torch.randn(2, 32, 64)
    x2 = x.clone()
    x2[:, 16:] = torch.randn_like(x2[:, 16:])
    with torch.no_grad():
        a, b = lane(x)[:, :16], lane(x2)[:, :16]
    assert torch.allclose(a, b, atol=1e-5)


def test_gradients_finite_through_acosh():
    lane = HyperbolicAttention(64, use_rope=True)
    x = torch.randn(2, 16, 64, requires_grad=True)
    lane(x).sum().backward()
    for p in lane.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.isfinite(x.grad).all()


def test_curvature_and_temp_learnable():
    lane = HyperbolicAttention(64)
    names = {n for n, _ in lane.named_parameters()}
    assert "log_curvature" in names and "log_temp" in names
    assert lane.log_curvature.requires_grad and lane.log_temp.requires_grad


def test_param_matched_to_reciprocal_twin():
    # Same q/k/v + 2 scalar geometry params vs reciprocal's 1 → within a hair,
    # so the gate compares mechanism, not capacity.
    hyp = sum(p.numel() for p in HyperbolicAttention(128).parameters())
    recip = sum(
        p.numel() for p in ReciprocalRankAttention(128, use_rope=True).parameters()
    )
    assert abs(hyp - recip) <= 2
