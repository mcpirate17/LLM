"""Parity tests: native C backward (gradient) kernels vs NumPy reference.

Tests every backward kernel in libaria_native_runtime.so against a manually
computed NumPy gradient. Tolerance: atol=2e-5, rtol=2e-5 for f32 ops that
use SIMD approximate exp/sigmoid.
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest


ATOL = 2e-5
RTOL = 2e-5
# GELU backward compounds multiple approximate ops, needs slightly wider tol
ATOL_GELU = 5e-5
RTOL_GELU = 5e-5


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str = "",
                   atol: float = ATOL, rtol: float = RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


# ── Unary backward ops ────────────────────────────────────────────────

class TestReluBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.input = np.random.randn(n).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_in = np.empty(n, dtype=np.float32)

    def test_relu_backward(self):
        fn = self.lib.aria_relu_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.input.ctypes.data,
           self.grad_in.ctypes.data, self.n)
        expected = self.grad_out * (self.input > 0).astype(np.float32)
        _assert_close(self.grad_in, expected, "relu_backward")


class TestSigmoidBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        input_data = np.random.randn(n).astype(np.float32)
        # Compute forward output (sigmoid uses output, not input)
        self.output = (1.0 / (1.0 + np.exp(-input_data))).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_in = np.empty(n, dtype=np.float32)

    def test_sigmoid_backward(self):
        fn = self.lib.aria_sigmoid_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.output.ctypes.data,
           self.grad_in.ctypes.data, self.n)
        expected = self.grad_out * self.output * (1.0 - self.output)
        _assert_close(self.grad_in, expected, "sigmoid_backward")


class TestTanhBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        input_data = np.random.randn(n).astype(np.float32)
        # Compute forward output (tanh uses output, not input)
        self.output = np.tanh(input_data).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_in = np.empty(n, dtype=np.float32)

    def test_tanh_backward(self):
        fn = self.lib.aria_tanh_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.output.ctypes.data,
           self.grad_in.ctypes.data, self.n)
        expected = self.grad_out * (1.0 - self.output * self.output)
        _assert_close(self.grad_in, expected, "tanh_backward")


class TestGeluBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.input = np.random.randn(n).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_in = np.empty(n, dtype=np.float32)

    def test_gelu_backward(self):
        fn = self.lib.aria_gelu_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.input.ctypes.data,
           self.grad_in.ctypes.data, self.n)

        # Reference: exact GELU backward
        x = self.input.astype(np.float64)
        coeff = np.sqrt(2.0 / np.pi)
        cubic = 0.044715
        inner = coeff * (x + cubic * x ** 3)
        t = np.tanh(inner)
        d_inner = coeff * (1.0 + 3.0 * cubic * x ** 2)
        dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * d_inner
        expected = (self.grad_out.astype(np.float64) * dgelu).astype(np.float32)
        _assert_close(self.grad_in, expected, "gelu_backward",
                      atol=ATOL_GELU, rtol=RTOL_GELU)


class TestSiluBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.input = np.random.randn(n).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_in = np.empty(n, dtype=np.float32)

    def test_silu_backward(self):
        fn = self.lib.aria_silu_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.input.ctypes.data,
           self.grad_in.ctypes.data, self.n)

        # Reference: SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        x = self.input.astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-x))
        dsilu = sig * (1.0 + x * (1.0 - sig))
        expected = (self.grad_out.astype(np.float64) * dsilu).astype(np.float32)
        _assert_close(self.grad_in, expected, "silu_backward")


# ── Binary backward ops ──────────────────────────────────────────────

class TestAddBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_a = np.empty(n, dtype=np.float32)
        self.grad_b = np.empty(n, dtype=np.float32)

    def test_add_backward(self):
        fn = self.lib.aria_add_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.grad_a.ctypes.data,
           self.grad_b.ctypes.data, self.n)
        _assert_close(self.grad_a, self.grad_out, "add_backward grad_a")
        _assert_close(self.grad_b, self.grad_out, "add_backward grad_b")


class TestMulBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.a = np.random.randn(n).astype(np.float32)
        self.b = np.random.randn(n).astype(np.float32)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_a = np.empty(n, dtype=np.float32)
        self.grad_b = np.empty(n, dtype=np.float32)

    def test_mul_backward(self):
        fn = self.lib.aria_mul_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.a.ctypes.data, self.b.ctypes.data,
           self.grad_a.ctypes.data, self.grad_b.ctypes.data, self.n)
        _assert_close(self.grad_a, self.grad_out * self.b, "mul_backward grad_a")
        _assert_close(self.grad_b, self.grad_out * self.a, "mul_backward grad_b")


class TestSubBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.grad_out = np.random.randn(n).astype(np.float32)
        self.grad_a = np.empty(n, dtype=np.float32)
        self.grad_b = np.empty(n, dtype=np.float32)

    def test_sub_backward(self):
        fn = self.lib.aria_sub_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.grad_out.ctypes.data, self.grad_a.ctypes.data,
           self.grad_b.ctypes.data, self.n)
        _assert_close(self.grad_a, self.grad_out, "sub_backward grad_a")
        _assert_close(self.grad_b, -self.grad_out, "sub_backward grad_b")


# ── Matmul backward ──────────────────────────────────────────────────

class TestMatmulBackward:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("M,K,N", [(2, 3, 2), (16, 32, 16), (64, 64, 64), (1, 128, 1), (4, 8, 4)])
    def test_matmul_backward(self, M, K, N):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        grad_out = np.random.randn(M, N).astype(np.float32)
        grad_A = np.zeros((M, K), dtype=np.float32)
        grad_B = np.zeros((K, N), dtype=np.float32)

        fn = self.lib.aria_matmul_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, A.ctypes.data, B.ctypes.data,
           grad_A.ctypes.data, grad_B.ctypes.data, M, K, N)

        # Reference: grad_A = grad_out @ B^T, grad_B = A^T @ grad_out
        expected_grad_A = grad_out @ B.T
        expected_grad_B = A.T @ grad_out
        _assert_close(grad_A, expected_grad_A, f"matmul_backward grad_A {M}x{K}x{N}")
        _assert_close(grad_B, expected_grad_B, f"matmul_backward grad_B {M}x{K}x{N}")


# ── Numerical gradient check (finite differences) ────────────────────

class TestNumericalGradientCheck:
    """Cross-validate backward kernels against finite-difference approximation."""

    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib
        np.random.seed(123)

    def _numerical_grad(self, forward_fn, x, eps=1e-3):
        """Central finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (forward_fn(x_plus) - forward_fn(x_minus)) / (2 * eps)
        return grad

    def test_relu_numerical(self):
        n = 32
        x = np.random.randn(n).astype(np.float32)
        # Avoid points near zero where relu is non-differentiable
        x[np.abs(x) < 0.1] = 0.5
        grad_out = np.ones(n, dtype=np.float32)
        grad_in = np.empty(n, dtype=np.float32)

        fn = self.lib.aria_relu_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, x.ctypes.data, grad_in.ctypes.data, n)

        def forward(xi):
            y = np.empty(n, dtype=np.float32)
            fwd = self.lib.aria_relu_f32
            fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fwd.restype = None
            fwd(xi.ctypes.data, y.ctypes.data, n)
            return y.sum()

        num_grad = self._numerical_grad(forward, x)
        _assert_close(grad_in, num_grad, "relu numerical", atol=1e-3, rtol=1e-3)

    def test_sigmoid_numerical(self):
        n = 32
        x = np.random.randn(n).astype(np.float32)
        grad_out = np.ones(n, dtype=np.float32)

        # Forward to get output
        output = np.empty(n, dtype=np.float32)
        fwd = self.lib.aria_sigmoid_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fwd.restype = None
        fwd(x.ctypes.data, output.ctypes.data, n)

        grad_in = np.empty(n, dtype=np.float32)
        fn = self.lib.aria_sigmoid_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, output.ctypes.data, grad_in.ctypes.data, n)

        def forward(xi):
            y = np.empty(n, dtype=np.float32)
            fwd(xi.ctypes.data, y.ctypes.data, n)
            return y.sum()

        num_grad = self._numerical_grad(forward, x)
        _assert_close(grad_in, num_grad, "sigmoid numerical", atol=1e-3, rtol=1e-3)

    def test_matmul_numerical(self):
        """Verify matmul backward with finite differences on a small case."""
        M, K, N = 3, 4, 2
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        grad_out = np.ones((M, N), dtype=np.float32)

        grad_A = np.zeros((M, K), dtype=np.float32)
        grad_B = np.zeros((K, N), dtype=np.float32)
        fn = self.lib.aria_matmul_backward_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(grad_out.ctypes.data, A.ctypes.data, B.ctypes.data,
           grad_A.ctypes.data, grad_B.ctypes.data, M, K, N)

        fwd = self.lib.aria_matmul_f32
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fwd.restype = None

        eps = 1e-3
        # Numerical grad w.r.t. A
        num_grad_A = np.zeros_like(A)
        A_flat = A.ravel()
        for i in range(len(A_flat)):
            A_plus = A_flat.copy(); A_plus[i] += eps
            A_minus = A_flat.copy(); A_minus[i] -= eps
            C_plus = np.zeros((M, N), dtype=np.float32)
            C_minus = np.zeros((M, N), dtype=np.float32)
            fwd(A_plus.ctypes.data, B.ctypes.data, C_plus.ctypes.data, M, K, N)
            fwd(A_minus.ctypes.data, B.ctypes.data, C_minus.ctypes.data, M, K, N)
            num_grad_A.ravel()[i] = (C_plus.sum() - C_minus.sum()) / (2 * eps)

        _assert_close(grad_A, num_grad_A, "matmul grad_A numerical", atol=1e-3, rtol=1e-3)
