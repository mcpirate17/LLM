"""Tests for NM-C7 recurrent-depth weight-shared refinement.

Pins the spec: identity-at-init (``W=I`` ⟹ ``h_k=x`` ⟹ ``out=x`` for any gate),
the per-token depth weights are the p-adic Lorentzian gate (bounded-reciprocal
inverse-distance — NOT the ``F.softmax(depth_logits)`` router that collapsed 6/8
``recursive_depth_router`` instances at scale), the gate resists single-depth
collapse across varied tokens, gradient reaches the shared block + gate knobs,
the operator is NOT a softmax-attention twin (NM-11 measured detector), and it
exposes a finite NM-10 physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.recurrent_depth_refine import (
    RecurrentDepthRefine,
    _lorentzian_weights,
    recurrent_depth_param_count,
)


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = RecurrentDepthRefine(dim=8, max_depth=3)
    x = torch.randn(2, 10, 8)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 3, 5, 8, 16])
@pytest.mark.parametrize("max_depth", [1, 2, 4])
def test_identity_at_init(d: int, max_depth: int) -> None:
    """``W=I`` ⟹ every recursion depth returns ``x`` ⟹ ``out=x`` for any gate."""
    mix = RecurrentDepthRefine(dim=d, max_depth=max_depth)
    x = torch.randn(3, 5, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"d={d}, max_depth={max_depth}"


def test_param_count_is_one_shared_block_not_stacked() -> None:
    """Compaction claim: ONE shared ``D×D`` block (+ gate scalars), not n_layers stacked."""
    d, n_layers = 64, 8
    mix = RecurrentDepthRefine(dim=d, max_depth=4)
    assert mix.num_parameters == recurrent_depth_param_count(d, 4)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters
    assert mix.num_parameters < n_layers * d * d  # ÷ n_layers at the block level
    assert mix.W.numel() == d * d


def test_depth_weights_are_inverse_distance_not_softmax() -> None:
    """The depth gate is a bounded-reciprocal (Lorentzian) partition of unity —
    the weight on an anchor decreases with ``|val - anchor|`` (inverse-distance),
    the exact collapse-proof gate of ``_op_padic_depth_route``."""
    mix = RecurrentDepthRefine(dim=8, max_depth=4)
    x = torch.randn(3, 7, 8)
    w = mix.depth_weights(x)
    assert w.shape == (3, 7, 4)
    assert (w >= 0).all() and (w <= 1).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(3, 7), atol=1e-6)
    # Weight rank must be the REVERSE of distance rank (inverse-distance ⟹ not softmax).
    val_row = torch.tensor([0.7])  # one token's standardized valuation
    anchors = mix.depth_anchors.detach()
    w_one = _lorentzian_weights(val_row, anchors, mix.route_log_sharpness.detach())
    dist = (val_row.unsqueeze(-1) - anchors).abs().squeeze(0)
    order_dist = torch.argsort(dist)  # nearest anchor first
    order_w = torch.argsort(w_one.squeeze(0), descending=True)  # heaviest weight first
    assert torch.equal(order_w, order_dist), (
        f"Lorentzian weight must rank anchors by inverse distance; "
        f"dist_order={order_dist.tolist()}, weight_order={order_w.tolist()}"
    )


def test_gate_resists_single_depth_collapse() -> None:
    """The documented pathology: a softmax depth-router collapsed 6/8 to one
    depth. The p-adic gate spreads weight across depths for a batch of varied
    tokens (per-token ultrametric structure), so no single depth dominates all."""
    torch.manual_seed(0)
    mix = RecurrentDepthRefine(dim=16, max_depth=4)
    x = torch.randn(8, 32, 16) * 3.0  # varied magnitudes ⟹ varied p-adic valuations
    w = mix.depth_weights(x)  # (8, 32, 4)
    max_w = w.max(dim=-1).values  # (8, 32)
    assert max_w.mean() < 0.95  # not one-hot collapsed for all tokens
    entropy = -(w * (w + 1e-9).log()).sum(dim=-1)  # (8, 32)
    assert entropy.mean() > 0.05  # weight is genuinely spread


def test_backward_flows_to_shared_block_and_gate() -> None:
    mix = RecurrentDepthRefine(dim=16, max_depth=3)
    with torch.no_grad():
        mix.W.add_(0.1 * torch.randn_like(mix.W))  # nudge off identity
    x = torch.randn(2, 6, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mix.W.grad is not None and mix.W.grad.abs().sum() > 0
    # Gate knobs are learned (the depth weighting adapts), unlike a fixed router.
    assert mix.depth_anchors.grad is not None and mix.depth_anchors.grad.abs().sum() > 0
    assert (
        mix.route_log_sharpness.grad is not None
        and mix.route_log_sharpness.grad.abs() > 0
    )


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the refine is per-token (pointwise ⟹
    ``cross_token_mixing=0``) and its depth gate is p-adic Lorentzian
    (inverse-distance, not exponentiated dot-product). Confirmed not a softmax-
    attention twin — this is the structural guarantee the softmax depth-router
    collapse cannot recur through this gate."""
    mix = RecurrentDepthRefine(dim=32, max_depth=3)
    with torch.no_grad():
        mix.W.add_(0.3 * torch.randn_like(mix.W))  # active, not identity
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mix)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f}, "
        f"const={props.constant_token_preservation:.3f}, "
        f"convex={props.convex_range_fraction:.3f})"
    )
    assert props.cross_token_mixing < 0.1  # pointwise refinement, not attention


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the refine exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside Monarch/Butterfly/Ternary."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = RecurrentDepthRefine(dim=16, max_depth=3)
    with torch.no_grad():
        mix.W.add_(0.3 * torch.randn_like(mix.W))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
