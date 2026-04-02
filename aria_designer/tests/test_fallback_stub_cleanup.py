"""Regression tests for placeholder fallback cleanup."""

from __future__ import annotations

import torch
import pytest

from aria_designer.components.positional.rope_rotate.kernel_fallback import (
    ComponentHandler as RopeRotateHandler,
)
from aria_designer.components.routing.adaptive_rank_gate.kernel_fallback import (
    ComponentHandler as AdaptiveRankGateHandler,
)
from aria_designer.components.routing.difficulty_blend_3way.kernel_fallback import (
    ComponentHandler as DifficultyBlendHandler,
)
from aria_designer.components.routing.score_depth_blend.kernel_fallback import (
    ComponentHandler as ScoreDepthBlendHandler,
)
from aria_designer.components.routing.signal_conditioned_compression.kernel_fallback import (
    ComponentHandler as SignalCompressionHandler,
)
from aria_designer.components.math_space.low_rank_proj.kernel_fallback import (
    ComponentHandler as LowRankProjHandler,
)
from aria_designer.components.math_space.clifford_attention.kernel_fallback import (
    ComponentHandler as CliffordAttentionHandler,
)


@pytest.mark.unit
def test_rope_rotate_fallback_is_not_passthrough():
    handler = RopeRotateHandler()
    x = torch.arange(1, 33, dtype=torch.float32).reshape(1, 4, 8)
    y = handler.forward({"x": x}, {})["y"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "handler_cls,inputs",
    [
        (AdaptiveRankGateHandler, {"x": torch.randn(1, 8, 32)}),
        (DifficultyBlendHandler, {"x": torch.randn(1, 8, 32)}),
        (
            ScoreDepthBlendHandler,
            {"x": torch.randn(1, 8, 32), "scores": torch.randn(1, 8, 3)},
        ),
        (
            SignalCompressionHandler,
            {"x": torch.randn(1, 8, 32), "routing_signal": torch.randn(1, 8, 1)},
        ),
    ],
)
def test_placeholder_fallbacks_now_compute_real_outputs(handler_cls, inputs):
    handler = handler_cls()
    x = inputs["x"]
    y = handler.forward(inputs, {})["y"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)


@pytest.mark.unit
def test_low_rank_proj_fallback_passthrough_without_params():
    """low_rank_proj delegates to research.mathspaces which returns identity when no params."""
    handler = LowRankProjHandler()
    x = torch.randn(1, 8, 32)
    y = handler.forward({"x": x}, {})["y"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.unit
def test_clifford_attention_fallback_produces_real_output():
    """clifford_attention now delegates to research.mathspaces instead of raising."""
    handler = CliffordAttentionHandler()
    x = torch.randn(1, 8, 32)
    y = handler.forward({"x": x}, {})["y"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
