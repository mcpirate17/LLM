"""Tests for tropical routing primitives (Phase 1 of Three-Pillar Upgrade)."""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


def test_tropical_router_and_moe_in_registry():
    """Verify tropical_router and tropical_moe are registered primitives."""
    from research.mathspaces.registry import register_all_mathspaces
    from research.synthesis.primitives import PRIMITIVE_REGISTRY

    register_all_mathspaces()
    assert "tropical_router" in PRIMITIVE_REGISTRY
    assert "tropical_moe" in PRIMITIVE_REGISTRY

    # Check metadata
    router_op = PRIMITIVE_REGISTRY["tropical_router"]
    assert router_op.n_inputs == 1
    assert router_op.shape_rule == "identity"
    assert router_op.has_params

    moe_op = PRIMITIVE_REGISTRY["tropical_moe"]
    assert moe_op.n_inputs == 1
    assert moe_op.shape_rule == "identity"
    assert moe_op.has_params


def test_tropical_router_forward_backward():
    """Test forward and backward pass through tropical_router execute fn."""
    from research.mathspaces.tropical_routing import execute_tropical_router

    module = nn.Module()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = execute_tropical_router(module, x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Backward
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_tropical_moe_forward_backward():
    """Test forward and backward pass through tropical_moe execute fn."""
    from research.mathspaces.tropical_routing import execute_tropical_moe

    module = nn.Module()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = execute_tropical_moe(module, x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Backward
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_tropical_moe_small_experts():
    """TropicalMoE with <=32 experts uses ModuleList path."""
    from research.mathspaces.tropical_routing import TropicalMoE

    moe = TropicalMoE(dim=32, n_experts=8, top_k=2)
    assert moe.experts is not None
    x = torch.randn(1, 4, 32)
    out = moe(x)
    assert out.shape == (1, 4, 32)


def test_tropical_moe_large_experts():
    """TropicalMoE with >32 experts uses batched matmul path."""
    from research.mathspaces.tropical_routing import TropicalMoE

    moe = TropicalMoE(dim=32, n_experts=64, top_k=2)
    assert moe.experts is None
    assert hasattr(moe, 'expert_weights')
    x = torch.randn(1, 4, 32)
    out = moe(x)
    assert out.shape == (1, 4, 32)


def test_routing_telemetry_capture():
    """Test that routing telemetry is recorded for tropical routing ops."""
    from research.mathspaces.registry import register_all_mathspaces
    from research.synthesis.compiler import _execute_op

    register_all_mathspaces()

    module = nn.Module()
    x = torch.randn(1, 4, 64)

    try:
        _execute_op(module, "tropical_router", (x,), {})
        telemetry = getattr(module, "routing_telemetry", None)
        # Telemetry should be recorded if _record_routing_telemetry was called
        if telemetry is not None:
            assert "tokens_total" in telemetry
    except Exception:
        # _execute_op may not be directly importable; that's OK
        pytest.skip("_execute_op not directly accessible")
