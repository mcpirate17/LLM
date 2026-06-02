from __future__ import annotations

import torch

from component_fab.generator.native_surprise_memory import (
    NativeAtlasPolySurpriseMemoryLane,
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringBiLaneSurpriseMemoryLane,
    NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTriLaneSurpriseMemoryLane,
    NativeContextGatedSurpriseMemoryLane,
    NativeReadBeforeWriteSurpriseMemoryLane,
    NativeSemiringRopeSurpriseMemoryLane,
    NativeSemiringRopeTitansMACSurpriseMemoryLane,
    NativeSemiringSurpriseMemoryLane,
    NativeSemiringTitansMACSurpriseMemoryLane,
    NativeTitansMACSurpriseMemoryLane,
    _NativeAdaptiveSemiringSurpriseScan,
    _NativeThreeLaneBlend,
    _NativeSemiringSurpriseScan,
    _NativeSurpriseScan,
    _NativeTwoLaneBlend,
)


def test_native_surprise_scan_gradcheck() -> None:
    torch.manual_seed(0)
    bsz, seq_len, memory_dim = 1, 3, 2
    q = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    k = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    v = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    write = torch.sigmoid(
        torch.randn(bsz, seq_len, dtype=torch.double)
    ).requires_grad_()
    forget = (
        torch.sigmoid(torch.randn(bsz, seq_len, memory_dim, dtype=torch.double)) * 0.1
    )
    forget = forget.detach().requires_grad_()
    momentum = torch.tensor(0.4, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda *args: _NativeSurpriseScan.apply(*args),
        (q, k, v, write, forget, momentum),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_native_semiring_surprise_scan_gradcheck() -> None:
    torch.manual_seed(1)
    bsz, seq_len, memory_dim = 1, 3, 2
    q = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    k = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    v = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    write = torch.sigmoid(
        torch.randn(bsz, seq_len, dtype=torch.double)
    ).requires_grad_()
    forget = (
        torch.sigmoid(torch.randn(bsz, seq_len, memory_dim, dtype=torch.double)) * 0.1
    )
    forget = forget.detach().requires_grad_()
    momentum = torch.tensor(0.35, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    balance = torch.tensor(0.75, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda *args: _NativeSemiringSurpriseScan.apply(*args),
        (q, k, v, write, forget, momentum, beta, balance),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_native_adaptive_semiring_surprise_scan_gradcheck() -> None:
    torch.manual_seed(3)
    bsz, seq_len, memory_dim = 1, 3, 2
    q = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    k = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    v = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    write = torch.sigmoid(
        torch.randn(bsz, seq_len, dtype=torch.double)
    ).requires_grad_()
    forget = (
        torch.sigmoid(torch.randn(bsz, seq_len, memory_dim, dtype=torch.double)) * 0.1
    )
    forget = forget.detach().requires_grad_()
    momentum = torch.tensor(0.35, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    balance = torch.tensor(0.75, dtype=torch.double, requires_grad=True)
    low = torch.tensor(0.0, dtype=torch.double)
    high = torch.tensor(0.01, dtype=torch.double)

    def only_y(*args):
        y, _depth = _NativeAdaptiveSemiringSurpriseScan.apply(*args)
        return y

    assert torch.autograd.gradcheck(
        only_y,
        (q, k, v, write, forget, momentum, beta, balance, low, high, 3),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_native_lane_blends_gradcheck() -> None:
    torch.manual_seed(2)
    shape = (1, 3, 4)
    a = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    b = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    c = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    logit = torch.randn(1, 3, 1, dtype=torch.double, requires_grad=True)
    logits = torch.randn(1, 3, 3, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda *args: _NativeTwoLaneBlend.apply(*args),
        (a, b, logit),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.autograd.gradcheck(
        lambda *args: _NativeThreeLaneBlend.apply(*args),
        (a, b, c, logits),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_native_surprise_lanes_are_finite_and_trainable() -> None:
    for cls in (
        NativeReadBeforeWriteSurpriseMemoryLane,
        NativeContextGatedSurpriseMemoryLane,
        NativeAtlasPolySurpriseMemoryLane,
        NativeTitansMACSurpriseMemoryLane,
        NativeSemiringSurpriseMemoryLane,
        NativeSemiringRopeSurpriseMemoryLane,
        NativeSemiringTitansMACSurpriseMemoryLane,
        NativeSemiringRopeTitansMACSurpriseMemoryLane,
        NativeBalancedSemiringTitansMACSurpriseMemoryLane,
        NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
        NativeBalancedSemiringBiLaneSurpriseMemoryLane,
        NativeBalancedSemiringTriLaneSurpriseMemoryLane,
        NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
        NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    ):
        torch.manual_seed(0)
        lane = cls(16)
        x = torch.randn(3, 7, 16, requires_grad=True)
        y = lane(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert any(p.grad is not None for p in lane.parameters())


def test_tuned_semiring_mac_factory_name() -> None:
    from research.tools.scaling_blimp_study import _build_lane_factory

    lane = _build_lane_factory(
        "native_semiring_rope_titans_mac_m32_g1_t1_surprise_memory"
    )(16)
    assert isinstance(lane, NativeSemiringRopeTitansMACSurpriseMemoryLane)
    assert lane.memory_dim == 32
    assert lane.semiring_temp.item() == 1.0
    assert torch.allclose(lane.dim_gate.bias, torch.ones_like(lane.dim_gate.bias))

    qkn = _build_lane_factory(
        "native_semiring_rope_titans_mac_qkn_m32_g0_t1_surprise_memory"
    )(16)
    assert isinstance(qkn, NativeSemiringRopeTitansMACSurpriseMemoryLane)
    assert qkn.memory_dim == 32
    assert qkn.qk_norm is True

    balanced = _build_lane_factory(
        "native_semiring_rope_titans_mac_bal_m32_g0_t1_b1_surprise_memory"
    )(16)
    assert isinstance(balanced, NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane)
    assert balanced.memory_dim == 32
    assert balanced.recursive_balance_logit is not None

    bilane = _build_lane_factory(
        "native_semiring_bal_bilane_m32_g0_t1_b1_surprise_memory"
    )(16)
    trilane = _build_lane_factory(
        "native_semiring_bal_trilane_m32_g0_t1_b1_surprise_memory"
    )(16)
    assert isinstance(bilane, NativeBalancedSemiringBiLaneSurpriseMemoryLane)
    assert isinstance(trilane, NativeBalancedSemiringTriLaneSurpriseMemoryLane)

    adaptive = _build_lane_factory(
        "native_semiring_rope_titans_mac_adapt_m32_g0_t1_b1_l1_h5_r4_surprise_memory"
    )(16)
    adaptive_bi = _build_lane_factory(
        "native_semiring_adapt_bilane_m32_g0_t1_b1_l1_h5_r4_surprise_memory"
    )(16)
    adaptive_bi_bp = _build_lane_factory(
        "native_semiring_adapt_bilane_m32_g0_t1_b1_l20bp_h200bp_r4_surprise_memory"
    )(16)
    assert isinstance(adaptive, NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane)
    assert isinstance(adaptive_bi, NativeAdaptiveSemiringBiLaneSurpriseMemoryLane)
    assert isinstance(adaptive_bi_bp, NativeAdaptiveSemiringBiLaneSurpriseMemoryLane)
    assert adaptive.max_recursive_steps == 4
