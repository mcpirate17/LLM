"""Parity tests: native C kernels vs NumPy/PyTorch reference implementations.

Tests every kernel in libaria_native_runtime.so against a reference implementation.
Tolerance: atol=1e-5, rtol=1e-5 for f32 operations.
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest

pytestmark = pytest.mark.native


ATOL = 1e-5
RTOL = 1e-5

# Wider tolerance for kernels using fast approximate SIMD exp (~1e-6 max rel error
# per element, but compounds through sigmoid/silu/softmax chains).
ATOL_APPROX = 2e-5
RTOL_APPROX = 2e-5


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str = "",
                   approx: bool = False):
    atol = ATOL_APPROX if approx else ATOL
    rtol = RTOL_APPROX if approx else RTOL
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


# ── Unary ops ─────────────────────────────────────────────────────────

class TestUnaryOps:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.x = np.random.randn(n).astype(np.float32)
        self.y = np.empty(n, dtype=np.float32)

    def _call_unary(self, name):
        fn = getattr(self.lib, f'aria_{name}_f32')
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.x.ctypes.data, self.y.ctypes.data, self.n)
        return self.y.copy()

    def test_relu(self):
        result = self._call_unary('relu')
        expected = np.maximum(self.x, 0.0)
        _assert_close(result, expected, "relu")

    def test_sigmoid(self):
        result = self._call_unary('sigmoid')
        expected = 1.0 / (1.0 + np.exp(-self.x))
        _assert_close(result, expected, "sigmoid", approx=True)

    def test_tanh(self):
        result = self._call_unary('tanh')
        expected = np.tanh(self.x)
        _assert_close(result, expected, "tanh")

    def test_exp(self):
        # Clamp to avoid overflow
        x_safe = np.clip(self.x, -10, 10).astype(np.float32)
        fn = self.lib.aria_exp_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(x_safe.ctypes.data, self.y.ctypes.data, self.n)
        expected = np.exp(x_safe)
        _assert_close(self.y, expected, "exp", approx=True)

    def test_silu(self):
        result = self._call_unary('silu')
        expected = self.x / (1.0 + np.exp(-self.x))
        _assert_close(result, expected, "silu", approx=True)

    def test_gelu(self):
        result = self._call_unary('gelu')
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = np.sqrt(2.0 / np.pi).astype(np.float32)
        inner = coeff * (self.x + 0.044715 * self.x ** 3)
        expected = 0.5 * self.x * (1.0 + np.tanh(inner))
        _assert_close(result, expected.astype(np.float32), "gelu")

    def test_sin(self):
        fn = self.lib.aria_sin_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.x.ctypes.data, self.y.ctypes.data, self.n)
        expected = np.sin(self.x)
        _assert_close(self.y, expected, "sin")

    def test_cos(self):
        fn = self.lib.aria_cos_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.x.ctypes.data, self.y.ctypes.data, self.n)
        expected = np.cos(self.x)
        _assert_close(self.y, expected, "cos")


# ── Binary ops ────────────────────────────────────────────────────────

class TestBinaryOps:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.a = np.random.randn(n).astype(np.float32)
        self.b = np.random.randn(n).astype(np.float32)
        self.y = np.empty(n, dtype=np.float32)

    def _call_binary(self, name):
        fn = getattr(self.lib, f'aria_{name}_f32')
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.a.ctypes.data, self.b.ctypes.data, self.y.ctypes.data, self.n)
        return self.y.copy()

    def test_add(self):
        result = self._call_binary('add')
        _assert_close(result, self.a + self.b, "add")

    def test_mul(self):
        result = self._call_binary('mul')
        _assert_close(result, self.a * self.b, "mul")

    def test_sub(self):
        result = self._call_binary('sub')
        _assert_close(result, self.a - self.b, "sub")


# ── Reductions ────────────────────────────────────────────────────────

class TestReductions:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.x = np.random.randn(n).astype(np.float32)

    def test_sum(self):
        fn = self.lib.aria_sum_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        fn.restype = ctypes.c_float
        result = fn(self.x.ctypes.data, self.n)
        expected = np.sum(self.x, dtype=np.float64)  # Higher precision ref
        assert abs(result - expected) < max(ATOL, abs(expected) * 0.001), f"sum: {result} vs {expected}"

    def test_mean(self):
        fn = self.lib.aria_mean_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        fn.restype = ctypes.c_float
        result = fn(self.x.ctypes.data, self.n)
        expected = np.mean(self.x, dtype=np.float64)
        assert abs(result - expected) < max(ATOL, abs(expected) * 0.001), f"mean: {result} vs {expected}"


# ── Matmul ────────────────────────────────────────────────────────────

class TestMatmul:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("M,K,N", [(2, 3, 2), (16, 32, 16), (64, 64, 64), (1, 128, 1)])
    def test_matmul(self, M, K, N):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        fn = self.lib.aria_matmul_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, K, N)
        expected = A @ B
        _assert_close(C, expected, f"matmul {M}x{K}x{N}")


# ── Linear ────────────────────────────────────────────────────────────

class TestLinear:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,din,dout", [(1, 16, 8), (4, 32, 64), (8, 64, 32)])
    def test_linear_with_bias(self, batch, din, dout):
        np.random.seed(42)
        x = np.random.randn(batch, din).astype(np.float32)
        W = np.random.randn(dout, din).astype(np.float32)
        bias = np.random.randn(dout).astype(np.float32)
        y = np.empty((batch, dout), dtype=np.float32)
        fn = self.lib.aria_linear_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(x.ctypes.data, W.ctypes.data, bias.ctypes.data,
           y.ctypes.data, batch, din, dout)
        expected = x @ W.T + bias
        _assert_close(y, expected, f"linear {batch}x{din}x{dout}")

    def test_linear_no_bias(self):
        np.random.seed(42)
        batch, din, dout = 4, 16, 8
        x = np.random.randn(batch, din).astype(np.float32)
        W = np.random.randn(dout, din).astype(np.float32)
        y = np.empty((batch, dout), dtype=np.float32)
        fn = self.lib.aria_linear_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(x.ctypes.data, W.ctypes.data, None,
           y.ctypes.data, batch, din, dout)
        expected = x @ W.T
        _assert_close(y, expected, "linear no bias")


# ── RMSNorm ───────────────────────────────────────────────────────────

class TestRMSNorm:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,dim", [(1, 16), (4, 64), (8, 128)])
    def test_rmsnorm(self, batch, dim):
        np.random.seed(42)
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5
        fn = self.lib.aria_rmsnorm_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(x.ctypes.data, w.ctypes.data, y.ctypes.data, batch, dim, eps)
        # Reference: y = x / rms(x) * weight
        rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps)
        expected = (x / rms) * w
        _assert_close(y, expected, f"rmsnorm {batch}x{dim}")


# ── Softmax ───────────────────────────────────────────────────────────

class TestSoftmax:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,dim", [(1, 8), (4, 32), (8, 128)])
    def test_softmax(self, batch, dim):
        np.random.seed(42)
        x = np.random.randn(batch, dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        fn = self.lib.aria_softmax_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(x.ctypes.data, y.ctypes.data, batch, dim)
        # Reference: stable softmax
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)
        expected = e_x / e_x.sum(axis=1, keepdims=True)
        _assert_close(y, expected, f"softmax {batch}x{dim}", approx=True)

    def test_softmax_sums_to_one(self):
        np.random.seed(42)
        batch, dim = 4, 64
        x = np.random.randn(batch, dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        fn = self.lib.aria_softmax_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(x.ctypes.data, y.ctypes.data, batch, dim)
        for b in range(batch):
            assert abs(y[b].sum() - 1.0) < 1e-5, f"softmax batch {b} sum != 1"


# ── LayerNorm ─────────────────────────────────────────────────────────

class TestLayerNorm:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,dim", [(1, 16), (4, 64), (8, 128)])
    def test_layernorm(self, batch, dim):
        np.random.seed(42)
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5
        fn = self.lib.aria_layernorm_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        fn.restype = None
        fn(x.ctypes.data, w.ctypes.data, b.ctypes.data,
           y.ctypes.data, batch, dim, eps)
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        expected = normed * w + b
        _assert_close(y, expected.astype(np.float32), f"layernorm {batch}x{dim}")


# ── Transpose ─────────────────────────────────────────────────────────

class TestTranspose:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("rows,cols", [(2, 3), (16, 32), (64, 64), (1, 128)])
    def test_transpose(self, rows, cols):
        np.random.seed(42)
        x = np.random.randn(rows, cols).astype(np.float32)
        y = np.empty((cols, rows), dtype=np.float32)
        fn = self.lib.aria_transpose2d_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        fn(x.ctypes.data, y.ctypes.data, rows, cols)
        _assert_close(y, x.T, f"transpose {rows}x{cols}")


# ── Concat/Split round-trip ───────────────────────────────────────────

class TestConcatSplit:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    def test_concat_split_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(200).astype(np.float32)
        c = np.random.randn(50).astype(np.float32)

        # Concat
        inputs = (ctypes.c_void_p * 3)(a.ctypes.data, b.ctypes.data, c.ctypes.data)
        sizes = (ctypes.c_int64 * 3)(100, 200, 50)
        output = np.empty(350, dtype=np.float32)
        fn_concat = self.lib.aria_concat_f32
        fn_concat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        fn_concat.restype = None
        fn_concat(inputs, sizes, 3, output.ctypes.data)
        expected = np.concatenate([a, b, c])
        _assert_close(output, expected, "concat")

        # Split back
        out_a = np.empty(100, dtype=np.float32)
        out_b = np.empty(200, dtype=np.float32)
        out_c = np.empty(50, dtype=np.float32)
        outputs = (ctypes.c_void_p * 3)(out_a.ctypes.data, out_b.ctypes.data, out_c.ctypes.data)
        fn_split = self.lib.aria_split_f32
        fn_split.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        fn_split.restype = None
        fn_split(output.ctypes.data, outputs, sizes, 3)
        _assert_close(out_a, a, "split[0]")
        _assert_close(out_b, b, "split[1]")
        _assert_close(out_c, c, "split[2]")


# ── Registry ──────────────────────────────────────────────────────────

class TestRegistry:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    def test_registry_init_and_count(self):
        self.lib.aria_registry_init()
        fn = self.lib.aria_registry_count
        fn.restype = ctypes.c_int32
        count = fn()
        assert count >= 9, f"Expected >= 9 registered kernels, got {count}"

    @pytest.mark.parametrize("op", ["relu", "gelu", "silu", "sigmoid", "tanh", "exp", "add", "mul", "sub"])
    def test_registry_has_builtin(self, op):
        self.lib.aria_registry_init()
        fn = self.lib.aria_registry_is_native
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = ctypes.c_int32
        assert fn(op.encode()) == 1, f"{op} not registered"

    def test_registry_missing_op(self):
        self.lib.aria_registry_init()
        fn = self.lib.aria_registry_is_native
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = ctypes.c_int32
        assert fn(b"nonexistent_fantasy_op") == 0
