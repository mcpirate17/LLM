"""Behavioral tests for depth_token_mask fallback."""

import torch

from aria_designer.components.routing.depth_token_mask.kernel_fallback import (
    ComponentHandler,
)


def test_depth_token_mask_zeroes_some_tokens_when_capacity_is_limited():
    handler = ComponentHandler()
    x = torch.ones(2, 8, 16)

    y = handler.forward({"x": x}, {"capacity_factor": 0.5})["y"]

    dropped = (y.abs().sum(dim=-1) == 0).sum().item()
    assert dropped > 0
    assert y.shape == x.shape
