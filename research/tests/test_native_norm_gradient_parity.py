"""Parity tests: native C backward kernels for softmax, layernorm, rmsnorm
vs PyTorch autograd.

Each test:
  1. Runs the forward + backward in PyTorch (ground truth)
  2. Calls the C backward kernel
  3. Checks that C gradients match PyTorch within f32 tolerance
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch required for parity tests", allow_module_level=True)


ATOL = 1e-4
RTOL = 1e-4

# Norm backward accumulates reductions — slightly wider tolerance for larger dims
ATOL_NORM = 2e-4
RTOL_NORM = 2e-4


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str = "",
                   atol: float = ATOL, rtol: float = RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


# ── Softmax backward ──────────────────────────────────────────────────

class TestSoftmaxBackward:

    @pytest.mark.parametrize("batch,dim", [
        (1, 8), (4, 16), (8, 64), (2, 128), (16, 256), (1, 3),
    ])
    def test_softmax_backward_vs_torch(self, native_lib, batch, dim):
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)

        # PyTorch reference
        x_t = torch.tensor(x_np, requires_grad=True)
        y_t = torch.softmax(x_t, dim=-1)
        grad_out_np = np.random.randn(batch, dim).astype(np.float32)
        y_t.backward(torch.tensor(grad_out_np))
        expected_grad_in = x_t.grad.numpy()

        # C kernel: needs forward output + grad_out -> grad_in
        y_np = y_t.detach().numpy().copy()
        grad_in = np.empty((batch, dim), dtype=np.float32)

        fn = native_lib.aria_softmax_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(grad_out_np.ctypes.data, y_np.ctypes.data,
           grad_in.ctypes.data, batch, dim)

        _assert_close(grad_in, expected_grad_in,
                      f"softmax_backward batch={batch} dim={dim}")

    def test_softmax_backward_uniform_grad(self, native_lib):
        """When grad_out is uniform, softmax backward should produce zero gradients."""
        batch, dim = 4, 32
        np.random.seed(123)
        x_np = np.random.randn(batch, dim).astype(np.float32)

        # Forward
        fwd = native_lib.aria_softmax_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        fwd.restype = None
        y_np = np.empty((batch, dim), dtype=np.float32)
        fwd(x_np.ctypes.data, y_np.ctypes.data, batch, dim)

        # grad_out = constant per row -> grad_in should be ~0
        grad_out = np.ones((batch, dim), dtype=np.float32) * 3.7
        grad_in = np.empty((batch, dim), dtype=np.float32)

        fn = native_lib.aria_softmax_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, y_np.ctypes.data, grad_in.ctypes.data, batch, dim)

        # sum(y) = 1 per row, so dot = constant * 1 = constant
        # gi = y * (constant - constant) = 0
        _assert_close(grad_in, np.zeros_like(grad_in),
                      "softmax_backward uniform grad", atol=1e-6)


# ── LayerNorm backward ────────────────────────────────────────────────

class TestLayerNormBackward:

    @pytest.mark.parametrize("batch,dim", [
        (1, 8), (4, 16), (8, 64), (2, 128), (16, 256), (1, 3),
    ])
    def test_layernorm_backward_vs_torch(self, native_lib, batch, dim):
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.random.randn(dim).astype(np.float32) * 0.5 + 1.0
        beta_np = np.random.randn(dim).astype(np.float32) * 0.1
        eps = 1e-5

        # PyTorch reference
        x_t = torch.tensor(x_np, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, requires_grad=True)
        beta_t = torch.tensor(beta_np, requires_grad=True)
        y_t = torch.nn.functional.layer_norm(x_t, (dim,), gamma_t, beta_t, eps=eps)
        grad_out_np = np.random.randn(batch, dim).astype(np.float32)
        y_t.backward(torch.tensor(grad_out_np))
        expected_grad_in = x_t.grad.numpy()
        expected_grad_gamma = gamma_t.grad.numpy()
        expected_grad_beta = beta_t.grad.numpy()

        # C kernel
        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)
        grad_beta = np.empty(dim, dtype=np.float32)

        fn = native_lib.aria_layernorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out_np.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data, grad_beta.ctypes.data,
           batch, dim, eps)

        _assert_close(grad_in, expected_grad_in,
                      f"layernorm_backward grad_in batch={batch} dim={dim}",
                      atol=ATOL_NORM, rtol=RTOL_NORM)
        _assert_close(grad_gamma, expected_grad_gamma,
                      f"layernorm_backward grad_gamma batch={batch} dim={dim}",
                      atol=ATOL_NORM, rtol=RTOL_NORM)
        _assert_close(grad_beta, expected_grad_beta,
                      f"layernorm_backward grad_beta batch={batch} dim={dim}",
                      atol=ATOL_NORM, rtol=RTOL_NORM)

    def test_layernorm_backward_gamma_ones_beta_zeros(self, native_lib):
        """With gamma=1, beta=0, simplifies to plain normalization gradient."""
        batch, dim = 4, 32
        eps = 1e-5
        np.random.seed(99)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.ones(dim, dtype=np.float32)
        beta_np = np.zeros(dim, dtype=np.float32)

        x_t = torch.tensor(x_np, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, requires_grad=True)
        beta_t = torch.tensor(beta_np, requires_grad=True)
        y_t = torch.nn.functional.layer_norm(x_t, (dim,), gamma_t, beta_t, eps=eps)
        grad_out_np = np.random.randn(batch, dim).astype(np.float32)
        y_t.backward(torch.tensor(grad_out_np))

        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)
        grad_beta = np.empty(dim, dtype=np.float32)

        fn = native_lib.aria_layernorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out_np.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data, grad_beta.ctypes.data,
           batch, dim, eps)

        _assert_close(grad_in, x_t.grad.numpy(),
                      "layernorm_backward identity gamma",
                      atol=ATOL_NORM, rtol=RTOL_NORM)


# ── RMSNorm backward ──────────────────────────────────────────────────

class TestRMSNormBackward:

    @pytest.mark.parametrize("batch,dim", [
        (1, 8), (4, 16), (8, 64), (2, 128), (16, 256), (1, 3),
    ])
    def test_rmsnorm_backward_vs_torch(self, native_lib, batch, dim):
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.random.randn(dim).astype(np.float32) * 0.5 + 1.0
        eps = 1e-5

        # PyTorch reference: manual RMSNorm forward + autograd
        x_t = torch.tensor(x_np, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, requires_grad=True)
        rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
        y_t = gamma_t * x_t / rms

        grad_out_np = np.random.randn(batch, dim).astype(np.float32)
        y_t.backward(torch.tensor(grad_out_np))
        expected_grad_in = x_t.grad.numpy()
        expected_grad_gamma = gamma_t.grad.numpy()

        # C kernel
        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)

        fn = native_lib.aria_rmsnorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out_np.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data,
           batch, dim, eps)

        _assert_close(grad_in, expected_grad_in,
                      f"rmsnorm_backward grad_in batch={batch} dim={dim}",
                      atol=ATOL_NORM, rtol=RTOL_NORM)
        _assert_close(grad_gamma, expected_grad_gamma,
                      f"rmsnorm_backward grad_gamma batch={batch} dim={dim}",
                      atol=ATOL_NORM, rtol=RTOL_NORM)

    def test_rmsnorm_backward_gamma_ones(self, native_lib):
        """With gamma=1, simplifies to x/rms normalization gradient."""
        batch, dim = 4, 32
        eps = 1e-5
        np.random.seed(77)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.ones(dim, dtype=np.float32)

        x_t = torch.tensor(x_np, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, requires_grad=True)
        rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
        y_t = gamma_t * x_t / rms

        grad_out_np = np.random.randn(batch, dim).astype(np.float32)
        y_t.backward(torch.tensor(grad_out_np))

        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)

        fn = native_lib.aria_rmsnorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out_np.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data,
           batch, dim, eps)

        _assert_close(grad_in, x_t.grad.numpy(),
                      "rmsnorm_backward identity gamma",
                      atol=ATOL_NORM, rtol=RTOL_NORM)


# ── Finite difference cross-check ─────────────────────────────────────

class TestNumericalGradientCheck:
    """Cross-validate backward kernels against finite-difference approximation."""

    def test_softmax_numerical(self, native_lib):
        batch, dim = 2, 8
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        grad_out = np.ones((batch, dim), dtype=np.float32)

        # Forward
        y_np = np.empty((batch, dim), dtype=np.float32)
        fwd = native_lib.aria_softmax_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        fwd.restype = None
        fwd(x_np.ctypes.data, y_np.ctypes.data, batch, dim)

        # Backward
        grad_in = np.empty((batch, dim), dtype=np.float32)
        fn = native_lib.aria_softmax_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, y_np.ctypes.data, grad_in.ctypes.data, batch, dim)

        # Numerical gradient
        eps = 1e-3
        num_grad = np.zeros_like(x_np)
        x_flat = x_np.ravel()
        for i in range(len(x_flat)):
            x_plus = x_flat.copy(); x_plus[i] += eps
            x_minus = x_flat.copy(); x_minus[i] -= eps
            y_plus = np.empty_like(y_np)
            y_minus = np.empty_like(y_np)
            fwd(x_plus.ctypes.data, y_plus.ctypes.data, batch, dim)
            fwd(x_minus.ctypes.data, y_minus.ctypes.data, batch, dim)
            num_grad.ravel()[i] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

        _assert_close(grad_in, num_grad, "softmax numerical", atol=1e-3, rtol=1e-3)

    def test_layernorm_numerical(self, native_lib):
        batch, dim = 2, 8
        eps = 1e-5
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.ones(dim, dtype=np.float32)
        bias_np = np.zeros(dim, dtype=np.float32)
        grad_out = np.ones((batch, dim), dtype=np.float32)

        # Backward via C
        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)
        grad_beta = np.empty(dim, dtype=np.float32)
        fn = native_lib.aria_layernorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data, grad_beta.ctypes.data,
           batch, dim, eps)

        # Numerical gradient w.r.t. input
        fwd = native_lib.aria_layernorm_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fwd.restype = None

        delta = 1e-3
        num_grad = np.zeros_like(x_np)
        x_flat = x_np.ravel()
        for i in range(len(x_flat)):
            x_plus = x_flat.copy(); x_plus[i] += delta
            x_minus = x_flat.copy(); x_minus[i] -= delta
            y_plus = np.empty_like(x_np)
            y_minus = np.empty_like(x_np)
            fwd(x_plus.ctypes.data, gamma_np.ctypes.data, bias_np.ctypes.data,
                y_plus.ctypes.data, batch, dim, eps)
            fwd(x_minus.ctypes.data, gamma_np.ctypes.data, bias_np.ctypes.data,
                y_minus.ctypes.data, batch, dim, eps)
            num_grad.ravel()[i] = (y_plus.sum() - y_minus.sum()) / (2 * delta)

        _assert_close(grad_in, num_grad, "layernorm numerical", atol=1e-3, rtol=1e-3)

    def test_rmsnorm_numerical(self, native_lib):
        batch, dim = 2, 8
        eps = 1e-5
        np.random.seed(42)
        x_np = np.random.randn(batch, dim).astype(np.float32)
        gamma_np = np.ones(dim, dtype=np.float32)
        grad_out = np.ones((batch, dim), dtype=np.float32)

        # Backward via C
        grad_in = np.empty((batch, dim), dtype=np.float32)
        grad_gamma = np.empty(dim, dtype=np.float32)
        fn = native_lib.aria_rmsnorm_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(grad_out.ctypes.data, x_np.ctypes.data, gamma_np.ctypes.data,
           grad_in.ctypes.data, grad_gamma.ctypes.data,
           batch, dim, eps)

        # Numerical gradient w.r.t. input
        fwd = native_lib.aria_rmsnorm_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fwd.restype = None

        delta = 1e-3
        num_grad = np.zeros_like(x_np)
        x_flat = x_np.ravel()
        for i in range(len(x_flat)):
            x_plus = x_flat.copy(); x_plus[i] += delta
            x_minus = x_flat.copy(); x_minus[i] -= delta
            y_plus = np.empty_like(x_np)
            y_minus = np.empty_like(x_np)
            fwd(x_plus.ctypes.data, gamma_np.ctypes.data,
                y_plus.ctypes.data, batch, dim, eps)
            fwd(x_minus.ctypes.data, gamma_np.ctypes.data,
                y_minus.ctypes.data, batch, dim, eps)
            num_grad.ravel()[i] = (y_plus.sum() - y_minus.sum()) / (2 * delta)

        _assert_close(grad_in, num_grad, "rmsnorm numerical", atol=1e-3, rtol=1e-3)
