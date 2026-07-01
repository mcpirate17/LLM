"""Tests for the Tier-1 novel-math invention lanes.

Covers the two new mechanisms (NM-T1-2 sheaf diffusion, NM-T1-3 fractional
integral): shape/finiteness, the structural claims that make each anti-softmax
(power-law kernel; causal overlap-agreement), strict causality, and end-to-end
dispatch through the invention codegen.
"""

from __future__ import annotations

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.novel_math_primitives import (
    FractionalIntegralMemoryLane,
    MeraRenormMixerLane,
    OctonionicMixerLane,
    SheafDiffusionMixerLane,
    SignedExpanderMixerLane,
    octonion_mul,
)
from component_fab.generator.reversible_primitives import ReversibleCouplingMixerLane
from component_fab.generator.routing_primitives import (
    AuctionCapacityRoutedLane,
    AuctionCapacityRouter,
)
from component_fab.inventor.mechanism_catalog import (
    enumerate_invention_specs,
    is_invention_spec,
)
from component_fab.proposer.algebraic_properties import measure_algebraic_properties


def _fwd_bwd_finite(lane: torch.nn.Module, shape: tuple[int, int, int]) -> torch.Tensor:
    x = torch.randn(*shape, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    return y


# --------------------------------------------------------------------------- #
# NM-T1-3 — fractional-integral memory
# --------------------------------------------------------------------------- #


def test_fractional_shape_and_grad_finite() -> None:
    torch.manual_seed(0)
    lane = FractionalIntegralMemoryLane(32, kernel_len=64)
    _fwd_bwd_finite(lane, (2, 40, 32))


def test_fractional_alpha_in_unit_interval() -> None:
    lane = FractionalIntegralMemoryLane(16)
    alpha = lane.alphas()
    assert alpha.shape == (16,)
    assert bool((alpha > 0.0).all()) and bool((alpha < 1.0).all())


def test_fractional_kernel_is_normalized_and_decaying() -> None:
    lane = FractionalIntegralMemoryLane(8, kernel_len=32)
    w = lane.kernel()
    assert w.shape == (8, 32)
    # Positive, normalized per channel.
    assert bool((w >= 0.0).all())
    assert torch.allclose(w.sum(dim=-1), torch.ones(8), atol=1e-5)
    # Monotone non-increasing in lag (power-law profile w_k ∝ k**(alpha-1), α<1).
    diffs = w[:, 1:] - w[:, :-1]
    assert bool((diffs <= 1e-6).all())


def test_fractional_larger_alpha_means_longer_memory() -> None:
    """α → 1 flattens the kernel toward a running average; α → 0 concentrates on
    the current token. So the mass on lags beyond 0 must increase with α."""
    lane = FractionalIntegralMemoryLane(2, kernel_len=64)
    with torch.no_grad():
        # channel 0 → small α, channel 1 → large α
        lane.alpha_logit.copy_(torch.tensor([-3.0, 3.0]))
        w = lane.kernel()
        mass_beyond_0 = 1.0 - w[:, 0]
        assert float(mass_beyond_0[1]) > float(mass_beyond_0[0])
        # small-α channel is nearly a delta (mostly current token)
        assert float(w[0, 0]) > 0.8


def test_fractional_is_causal() -> None:
    torch.manual_seed(1)
    lane = FractionalIntegralMemoryLane(12, kernel_len=32)
    x_a = torch.randn(1, 16, 12)
    x_b = x_a.clone()
    x_b[:, 8:] += torch.randn(1, 8, 12)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :8], lane(x_b)[:, :8], atol=1e-5)


# --------------------------------------------------------------------------- #
# NM-T1-2 — sheaf diffusion mixer
# --------------------------------------------------------------------------- #


def test_sheaf_shape_and_grad_finite() -> None:
    torch.manual_seed(0)
    lane = SheafDiffusionMixerLane(32, window=6, n_steps=3)
    _fwd_bwd_finite(lane, (2, 24, 32))


def test_sheaf_is_causal() -> None:
    torch.manual_seed(2)
    lane = SheafDiffusionMixerLane(16, window=4, n_steps=3)
    x_a = torch.randn(1, 12, 16)
    x_b = x_a.clone()
    x_b[:, 6:] += torch.randn(1, 6, 16)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :6], lane(x_b)[:, :6], atol=1e-5)


def test_sheaf_restriction_is_not_identity() -> None:
    """Anti-collapse: the restriction map R must stay non-degenerate (not the
    identity), otherwise the diffusion has no sheaf structure to enforce."""
    torch.manual_seed(0)
    lane = SheafDiffusionMixerLane(16)
    eye = torch.eye(16)
    assert not torch.allclose(lane.restrict.weight, eye, atol=1e-2)


def test_sheaf_diffusion_actually_mixes() -> None:
    """With diffusion active the output must differ from a pure readout of the
    input (n_steps=0 has no effect) — i.e. the agreement step does work."""
    torch.manual_seed(3)
    lane = SheafDiffusionMixerLane(16, window=4, n_steps=3)
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        mixed = lane(x)
        no_diffusion = lane.out(x)  # what forward returns if alpha·update == 0
    assert not torch.allclose(mixed, no_diffusion, atol=1e-3)


# --------------------------------------------------------------------------- #
# NM-T1-4 — MERA renormalization mixer
# --------------------------------------------------------------------------- #


def test_mera_shape_and_grad_finite() -> None:
    torch.manual_seed(0)
    lane = MeraRenormMixerLane(32, n_levels=3)
    _fwd_bwd_finite(lane, (2, 20, 32))


def test_mera_is_causal() -> None:
    torch.manual_seed(1)
    lane = MeraRenormMixerLane(16, n_levels=3)
    x_a = torch.randn(1, 16, 16)
    x_b = x_a.clone()
    x_b[:, 10:] += torch.randn(1, 6, 16)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :10], lane(x_b)[:, :10], atol=1e-5)


def test_mera_receptive_field_is_multiscale() -> None:
    """n_levels=3 → causal lookback 2**3 - 1 = 7. Perturbing token 0 changes the
    output at position 7 (lag 7, in reach) but not position 8 (lag 8, out of
    reach) — proving the receptive field doubles across levels."""
    torch.manual_seed(0)
    lane = MeraRenormMixerLane(8, n_levels=3)
    x_a = torch.randn(1, 12, 8)
    x_b = x_a.clone()
    x_b[:, 0] += 1.0
    with torch.no_grad():
        ya, yb = lane(x_a), lane(x_b)
    assert not torch.allclose(ya[:, 7], yb[:, 7], atol=1e-6)  # in reach
    assert torch.allclose(ya[:, 8], yb[:, 8], atol=1e-6)  # out of reach


# --------------------------------------------------------------------------- #
# NM-9 — octonionic non-associative mixer
# --------------------------------------------------------------------------- #


def test_octonion_mul_is_normed_division_algebra() -> None:
    """|xy| = |x||y| — the defining property that keeps the mix norm-bounded."""
    torch.manual_seed(0)
    x, y = torch.randn(500, 8), torch.randn(500, 8)
    prod_norm = octonion_mul(x, y).norm(dim=-1)
    factor_norm = x.norm(dim=-1) * y.norm(dim=-1)
    assert torch.allclose(prod_norm, factor_norm, atol=1e-4)


def test_octonion_mul_is_non_associative() -> None:
    """(xy)z != x(yz) — the novel structure a semiring/associative scan lacks."""
    torch.manual_seed(1)
    x, y, z = torch.randn(500, 8), torch.randn(500, 8), torch.randn(500, 8)
    left = octonion_mul(octonion_mul(x, y), z)
    right = octonion_mul(x, octonion_mul(y, z))
    assert not torch.allclose(left, right, atol=1e-3)


def test_octonionic_shape_and_grad_finite() -> None:
    torch.manual_seed(0)
    lane = OctonionicMixerLane(24)
    _fwd_bwd_finite(lane, (4, 16, 24))


def test_octonionic_requires_multiple_of_eight() -> None:
    import pytest

    with pytest.raises(ValueError):
        OctonionicMixerLane(20)


def test_octonionic_is_causal() -> None:
    torch.manual_seed(2)
    lane = OctonionicMixerLane(16)
    x_a = torch.randn(1, 16, 16)
    x_b = x_a.clone()
    x_b[:, 9:] += torch.randn(1, 7, 16)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :9], lane(x_b)[:, :9], atol=1e-5)


def test_octonionic_mixes_but_is_not_a_softmax_twin() -> None:
    """Genuinely mixes tokens yet is anti-softmax-twin: it is NOT a convex
    averager (a token-constant input is not preserved) and is non-linear."""
    torch.manual_seed(0)
    lane = OctonionicMixerLane(24)
    props = measure_algebraic_properties(lane, dim=24, n_seeds=3)
    assert props.cross_token_mixing > 0.5  # it does mix across tokens
    assert not props.is_softmax_twin()  # but not by convex averaging
    assert props.constant_token_preservation < 0.7  # breaks partition-of-unity


# --------------------------------------------------------------------------- #
# MiniMax-M3-align M3X-M2 — signed expander mixer
# --------------------------------------------------------------------------- #


def test_signed_expander_shape_and_grad_finite() -> None:
    """MiniMax-M3-align M3X-M2."""
    torch.manual_seed(0)
    lane = SignedExpanderMixerLane(32, degree=8)
    _fwd_bwd_finite(lane, (2, 18, 32))


def test_signed_expander_is_causal() -> None:
    """MiniMax-M3-align M3X-M2."""
    torch.manual_seed(1)
    lane = SignedExpanderMixerLane(24, degree=8)
    x_a = torch.randn(1, 14, 24)
    x_b = x_a.clone()
    x_b[:, 8:] += torch.randn(1, 6, 24)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :8], lane(x_b)[:, :8], atol=1e-5)


def test_signed_expander_has_regular_gap_and_compact_params() -> None:
    """MiniMax-M3-align M3X-M2."""
    lane = SignedExpanderMixerLane(32, degree=8)
    adj = lane.channel_adjacency()
    assert adj.shape == (32, 32)
    assert torch.allclose(adj.sum(dim=-1), torch.ones(32), atol=1e-6)
    assert bool((adj > 0).sum(dim=-1).eq(lane.degree).all())
    assert float(lane.spectral_gap().detach()) > 0.2
    param_count = sum(p.numel() for p in lane.parameters())
    assert param_count < 32 * 32 // 4


def test_signed_expander_mixes_but_is_not_a_softmax_twin() -> None:
    """MiniMax-M3-align M3X-M2."""
    torch.manual_seed(0)
    lane = SignedExpanderMixerLane(32, degree=8)
    props = measure_algebraic_properties(lane, dim=32, n_seeds=3)
    assert props.cross_token_mixing > 0.1
    assert props.softmax_twin_score < 0.4
    assert not props.is_softmax_twin()


# --------------------------------------------------------------------------- #
# MiniMax-M3-align M3X-R1 — auction capacity router
# --------------------------------------------------------------------------- #


def test_auction_capacity_router_is_hard_and_balanced() -> None:
    """MiniMax-M3-align M3X-R1."""
    torch.manual_seed(0)
    router = AuctionCapacityRouter(12, n_experts=4)
    x = torch.randn(2, 16, 12)
    weights = router.route_weights(x)
    assert weights.shape == (2, 16, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 16), atol=1e-6)
    hard = weights.detach()
    assert torch.allclose(hard, hard.round(), atol=1e-6)
    per_batch_load = weights.sum(dim=1)
    assert bool((per_batch_load <= router.capacity(16)).all())
    assert torch.allclose(per_batch_load, torch.full((2, 4), 4.0), atol=1e-6)


def test_auction_capacity_lane_shape_grad_and_router_grad_finite() -> None:
    """MiniMax-M3-align M3X-R1."""
    torch.manual_seed(1)
    lane = AuctionCapacityRoutedLane(16, n_experts=4)
    x = torch.randn(2, 12, 16, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lane.router.bid_proj.weight.grad is not None
    assert torch.isfinite(lane.router.bid_proj.weight.grad).all()


def test_auction_capacity_lane_is_causal() -> None:
    """MiniMax-M3-align M3X-R1."""
    torch.manual_seed(2)
    lane = AuctionCapacityRoutedLane(16, n_experts=4)
    x_a = torch.randn(1, 16, 16)
    x_b = x_a.clone()
    x_b[:, 9:] += torch.randn(1, 7, 16)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :9], lane(x_b)[:, :9], atol=1e-5)


def test_auction_capacity_mixes_but_is_not_a_softmax_twin() -> None:
    """MiniMax-M3-align M3X-R1."""
    torch.manual_seed(0)
    lane = AuctionCapacityRoutedLane(16, n_experts=4)
    props = measure_algebraic_properties(lane, dim=16, n_seeds=3)
    assert props.cross_token_mixing > 0.05
    assert props.softmax_twin_score < 0.4
    assert not props.is_softmax_twin()


# --------------------------------------------------------------------------- #
# NM-9 verification — the symplectic residual mixer lane is real, not a stub
# --------------------------------------------------------------------------- #


def test_symplectic_residual_mixer_is_real() -> None:
    specs = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    assert "symplectic_residual_mixer" in specs
    module = generate_module_from_spec(specs["symplectic_residual_mixer"], dim=16)
    # A real lane, not an identity/stub: finite fwd/bwd and it transforms input.
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with torch.no_grad():
        assert not torch.allclose(module(x.detach()), x.detach())


# --------------------------------------------------------------------------- #
# Blueprints + end-to-end dispatch
# --------------------------------------------------------------------------- #


def test_blueprints_enumerated_and_gate_clean() -> None:
    mechanisms = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    for mech in (
        "fractional_integral_memory",
        "sheaf_consistent_slot_mixer",
        "mera_block",
        "octonionic_mixer",
        "signed_expander_mixer",
        "auction_capacity_router",
        "reversible_coupling_mixer",
    ):
        assert mech in mechanisms
        assert is_invention_spec(mechanisms[mech])


def test_codegen_dispatches_novel_lanes() -> None:
    specs = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    x = torch.randn(2, 8, 16)
    expected = {
        "fractional_integral_memory": FractionalIntegralMemoryLane,
        "sheaf_consistent_slot_mixer": SheafDiffusionMixerLane,
        "mera_block": MeraRenormMixerLane,
        "octonionic_mixer": OctonionicMixerLane,
        "signed_expander_mixer": SignedExpanderMixerLane,
        "auction_capacity_router": AuctionCapacityRoutedLane,
        "reversible_coupling_mixer": ReversibleCouplingMixerLane,
    }
    for mech, cls in expected.items():
        module = generate_module_from_spec(specs[mech], dim=16)
        assert isinstance(module, cls)
        assert module(x).shape == x.shape
