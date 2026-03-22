"""Tests for softmax, layernorm, rmsnorm backward kernels and autograd Functions.

Tests:
1. Forward correctness vs PyTorch reference
2. torch.autograd.gradcheck for each op
3. Multi-op chain: input -> layernorm -> relu -> softmax
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure cython bridge is importable
_cython_dir = str(Path(__file__).resolve().parents[1] / "runtime" / "native" / "cython")
if _cython_dir not in sys.path:
    sys.path.insert(0, _cython_dir)

try:
    import aria_bridge

    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(not HAS_BRIDGE, reason="Cython bridge not available"),
]


# ── Helpers ──────────────────────────────────────────────────────────


def _rand2d(batch, dim, requires_grad=False):
    """Create a random 2D float32 tensor."""
    t = torch.randn(batch, dim, dtype=torch.float64)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _rand1d(dim, requires_grad=False):
    """Create a random 1D float32 tensor (for gamma/beta)."""
    t = torch.randn(dim, dtype=torch.float64)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ── Bridge-level backward tests ─────────────────────────────────────


class TestBridgeSoftmaxBackward:
    def test_softmax_backward_shape(self):
        batch, dim = 4, 8
        x = np.random.randn(batch, dim).astype(np.float32)
        y = aria_bridge.dispatch_softmax(x)
        grad_out = np.ones_like(y)
        grad_in = aria_bridge.dispatch_softmax_backward(grad_out, y)
        assert grad_in.shape == (batch, dim)

    def test_softmax_backward_sums_to_zero(self):
        """For uniform grad_out, softmax backward should produce gradients
        that sum to zero along each row (since softmax outputs sum to 1)."""
        batch, dim = 4, 8
        x = np.random.randn(batch, dim).astype(np.float32)
        y = aria_bridge.dispatch_softmax(x)
        grad_out = np.ones((batch, dim), dtype=np.float32)
        grad_in = aria_bridge.dispatch_softmax_backward(grad_out, y)
        # Each row of grad_in should sum approximately to 0
        row_sums = grad_in.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-5)


class TestBridgeLayernormBackward:
    def test_layernorm_backward_shapes(self):
        batch, dim = 4, 8
        x = np.random.randn(batch, dim).astype(np.float32)
        gamma = np.ones(dim, dtype=np.float32)
        grad_out = np.random.randn(batch, dim).astype(np.float32)
        grad_in, grad_gamma, grad_beta = aria_bridge.dispatch_layernorm_backward(
            grad_out, x, gamma
        )
        assert grad_in.shape == (batch, dim)
        assert grad_gamma.shape == (dim,)
        assert grad_beta.shape == (dim,)


class TestBridgeRmsnormBackward:
    def test_rmsnorm_backward_shapes(self):
        batch, dim = 4, 8
        x = np.random.randn(batch, dim).astype(np.float32)
        gamma = np.ones(dim, dtype=np.float32)
        grad_out = np.random.randn(batch, dim).astype(np.float32)
        grad_in, grad_gamma = aria_bridge.dispatch_rmsnorm_backward(grad_out, x, gamma)
        assert grad_in.shape == (batch, dim)
        assert grad_gamma.shape == (dim,)


class TestHasBackward:
    def test_norm_ops_have_backward(self):
        assert aria_bridge.has_backward("softmax")
        assert aria_bridge.has_backward("layernorm")
        assert aria_bridge.has_backward("rmsnorm")


# ── Autograd forward correctness ────────────────────────────────────


class TestAutogradForwardCorrectness:
    def test_softmax_forward_matches_torch(self):
        from research.scientist.native_autograd import NativeSoftmax

        batch, dim = 4, 16
        x = torch.randn(batch, dim)
        expected = torch.softmax(x, dim=-1)
        result = NativeSoftmax.apply(x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_layernorm_forward_matches_torch(self):
        from research.scientist.native_autograd import NativeLayernorm

        batch, dim = 4, 16
        x = torch.randn(batch, dim)
        gamma = torch.ones(dim)
        beta = torch.zeros(dim)
        ln = torch.nn.LayerNorm(dim, elementwise_affine=False)
        expected = ln(x)
        result = NativeLayernorm.apply(x, gamma, beta)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_forward_matches_torch(self):
        from research.scientist.native_autograd import NativeRmsnorm

        batch, dim = 4, 16
        x = torch.randn(batch, dim)
        gamma = torch.ones(dim)
        # RMSNorm: x * gamma / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
        expected = x / rms * gamma
        result = NativeRmsnorm.apply(x, gamma)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


# ── Autograd gradcheck ──────────────────────────────────────────────


class TestAutogradGradcheck:
    def test_softmax_gradcheck(self):
        from research.scientist.native_autograd import NativeSoftmax

        batch, dim = 2, 4
        x = _rand2d(batch, dim, requires_grad=True)
        # Larger tolerance needed because C kernels operate in float32
        assert torch.autograd.gradcheck(
            NativeSoftmax.apply,
            (x,),
            eps=1e-3,
            atol=1e-2,
            rtol=1e-2,
            nondet_tol=1e-3,
        )

    def test_layernorm_gradcheck(self):
        from research.scientist.native_autograd import NativeLayernorm

        batch, dim = 2, 4
        x = _rand2d(batch, dim, requires_grad=True)
        gamma = _rand1d(dim, requires_grad=True)
        beta = _rand1d(dim, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeLayernorm.apply,
            (x, gamma, beta),
            eps=1e-3,
            atol=1e-2,
            rtol=1e-2,
            nondet_tol=1e-3,
        )

    def test_rmsnorm_gradcheck(self):
        from research.scientist.native_autograd import NativeRmsnorm

        batch, dim = 2, 4
        x = _rand2d(batch, dim, requires_grad=True)
        gamma = _rand1d(dim, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeRmsnorm.apply,
            (x, gamma),
            eps=1e-3,
            atol=1e-2,
            rtol=1e-2,
            nondet_tol=1e-3,
        )


# ── Registry tests ──────────────────────────────────────────────────


class TestRegistry:
    def test_supported_ops_include_norms(self):
        from research.scientist.native_autograd import NATIVE_AUTOGRAD_SUPPORTED_OPS

        assert "softmax" in NATIVE_AUTOGRAD_SUPPORTED_OPS
        assert "layernorm" in NATIVE_AUTOGRAD_SUPPORTED_OPS
        assert "rmsnorm" in NATIVE_AUTOGRAD_SUPPORTED_OPS

    def test_dispatch_softmax(self):
        from research.scientist.native_autograd import native_autograd_dispatch

        x = torch.randn(2, 8, requires_grad=True)
        y = native_autograd_dispatch("softmax", x)
        assert y.requires_grad
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_dispatch_layernorm(self):
        from research.scientist.native_autograd import native_autograd_dispatch

        x = torch.randn(2, 8, requires_grad=True)
        gamma = torch.ones(8, requires_grad=True)
        beta = torch.zeros(8, requires_grad=True)
        y = native_autograd_dispatch("layernorm", x, gamma, beta)
        assert y.requires_grad
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert gamma.grad is not None
        assert beta.grad is not None

    def test_dispatch_rmsnorm(self):
        from research.scientist.native_autograd import native_autograd_dispatch

        x = torch.randn(2, 8, requires_grad=True)
        gamma = torch.ones(8, requires_grad=True)
        y = native_autograd_dispatch("rmsnorm", x, gamma)
        assert y.requires_grad
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert gamma.grad is not None


# ── Multi-op chain test ─────────────────────────────────────────────


class TestMultiOpChain:
    def test_layernorm_relu_softmax_chain(self):
        """Test gradient flow through: input -> layernorm -> relu -> softmax."""
        from research.scientist.native_autograd import native_autograd_dispatch

        batch, dim = 2, 8
        x = torch.randn(batch, dim, requires_grad=True)
        gamma = torch.ones(dim, requires_grad=True)
        beta = torch.zeros(dim, requires_grad=True)

        # Forward
        h = native_autograd_dispatch("layernorm", x, gamma, beta)
        h = native_autograd_dispatch("relu", h)
        y = native_autograd_dispatch("softmax", h)

        # Backward
        loss = y.sum()
        loss.backward()

        # All inputs should have gradients
        assert x.grad is not None
        assert gamma.grad is not None
        assert beta.grad is not None
        assert x.grad.shape == x.shape
        assert gamma.grad.shape == gamma.shape
        assert beta.grad.shape == beta.shape

        # Gradients should be finite
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(gamma.grad).all()
        assert torch.isfinite(beta.grad).all()
