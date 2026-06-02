"""Behavior tests for the MoR-style learnable recursion-router surprise lane.

Locks the three properties that motivate the lane (design note
``research/notes/mor_native_recursion_router_2026-06-01.md``):
1. the halting router is *differentiable* — gradient reaches it (the native
   fixed-threshold gate gets exactly zero gradient, the bug this fixes);
2. the routed recursion stays strictly causal (no look-ahead leak);
3. the expected-depth ponder cost is finite and exposed for a compute penalty.
"""

from __future__ import annotations

import torch

from component_fab.generator.mor_surprise_memory import MoRSemiringSurpriseMemoryLane


def _lane(**kw) -> MoRSemiringSurpriseMemoryLane:
    return MoRSemiringSurpriseMemoryLane(64, memory_dim=16, **kw)


def test_forward_shape_and_finite() -> None:
    lane = _lane(max_recursive_steps=4)
    y = lane(torch.randn(4, 20, 64))
    assert y.shape == (4, 20, 64)
    assert torch.isfinite(y).all()
    assert lane.last_ponder_cost is not None
    assert torch.isfinite(lane.last_ponder_cost)
    assert float(lane.last_ponder_cost.detach()) > 0.0


def test_router_receives_gradient() -> None:
    """The differentiability fix: the halting head must get a real gradient."""
    lane = _lane(max_recursive_steps=4, surprise_conditioned=True)
    y = lane(torch.randn(4, 12, 64))
    (y.pow(2).mean() + lane.last_ponder_cost).backward()
    assert lane.halt_head.weight.grad is not None
    assert lane.halt_head.weight.grad.abs().max().item() > 0.0
    assert lane.halt_head.bias.grad is not None
    assert lane.halt_head.bias.grad.abs().item() > 0.0


def test_causal_no_lookahead() -> None:
    lane = _lane(max_recursive_steps=4).eval()
    torch.manual_seed(0)
    x = torch.randn(3, 16, 64)
    with torch.no_grad():
        base = lane(x)
    worst = 0.0
    for p in range(x.shape[1] - 1):
        z = x.clone()
        z[:, p + 1 :] = torch.randn(3, x.shape[1] - 1 - p, 64)
        with torch.no_grad():
            out = lane(z)
        worst = max(worst, (base[:, : p + 1] - out[:, : p + 1]).abs().max().item())
    assert worst < 1e-4, f"look-ahead leak: max delta {worst}"


def test_surprise_conditioning_ablation_builds() -> None:
    """Both router variants build; surprise-cond adds exactly one input feature."""
    surp = _lane(surprise_conditioned=True)
    plain = _lane(surprise_conditioned=False)
    assert surp.halt_head.in_features == plain.halt_head.in_features + 1
    assert plain(torch.randn(2, 8, 64)).shape == (2, 8, 64)


def test_depth_one_reduces_cleanly() -> None:
    lane = _lane(max_recursive_steps=1)
    y = lane(torch.randn(2, 8, 64))
    assert y.shape == (2, 8, 64)
    assert torch.isfinite(y).all()
