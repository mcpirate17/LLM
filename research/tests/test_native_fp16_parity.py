"""Parity tests: fp16 C kernels vs f32 C kernels.

Each fp16 kernel should produce results close to the f32 kernel applied
to the same data, with tolerance accounting for fp16 precision loss.

fp16 has ~3.3 decimal digits of precision (vs f32's ~7.2), so we use
atol=1e-2 / rtol=5e-3 which is generous but catches real bugs.
"""

from __future__ import annotations

import ctypes
import numpy as np
import pytest

pytestmark = pytest.mark.native


# fp16 tolerances — wider than f32 due to representation limits
ATOL_F16 = 1e-2
RTOL_F16 = 5e-3


def _f32_to_f16_array(arr: np.ndarray) -> np.ndarray:
    """Convert f32 array to fp16 stored as uint16."""
    return arr.astype(np.float16).view(np.uint16)


def _f16_to_f32_array(arr: np.ndarray) -> np.ndarray:
    """Convert uint16 fp16 array back to f32."""
    return arr.view(np.float16).astype(np.float32)


def _assert_close_f16(
    actual_f16: np.ndarray, expected_f32: np.ndarray, label: str = ""
):
    """Compare fp16 output (as uint16) against f32 reference."""
    actual_f32 = _f16_to_f32_array(actual_f16)
    np.testing.assert_allclose(
        actual_f32, expected_f32, atol=ATOL_F16, rtol=RTOL_F16, err_msg=label
    )


# ── Unary fp16 ops ────────────────────────────────────────────────────


class TestUnaryF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        # Use moderate range to stay in fp16 representable range
        self.x_f32 = np.random.uniform(-3, 3, n).astype(np.float32)
        self.x_f16 = _f32_to_f16_array(self.x_f32)
        self.y_f16 = np.empty(n, dtype=np.uint16)
        self.y_f32 = np.empty(n, dtype=np.float32)

    def _call_f32(self, name):
        fn = getattr(self.lib, f"aria_{name}_f32")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.x_f32.ctypes.data, self.y_f32.ctypes.data, self.n)

    def _call_f16(self, name):
        fn = getattr(self.lib, f"aria_{name}_f16")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        fn(self.x_f16.ctypes.data, self.y_f16.ctypes.data, self.n)

    def test_relu_f16(self):
        self._call_f32("relu")
        self._call_f16("relu")
        _assert_close_f16(self.y_f16, self.y_f32, "relu_f16")

    def test_gelu_f16(self):
        self._call_f32("gelu")
        self._call_f16("gelu")
        _assert_close_f16(self.y_f16, self.y_f32, "gelu_f16")

    def test_silu_f16(self):
        self._call_f32("silu")
        self._call_f16("silu")
        _assert_close_f16(self.y_f16, self.y_f32, "silu_f16")

    def test_sigmoid_f16(self):
        self._call_f32("sigmoid")
        self._call_f16("sigmoid")
        _assert_close_f16(self.y_f16, self.y_f32, "sigmoid_f16")


# ── Binary fp16 ops ───────────────────────────────────────────────────


class TestBinaryF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib, array_size):
        self.lib = native_lib
        self.n = n = array_size
        np.random.seed(42)
        self.a_f32 = np.random.uniform(-2, 2, n).astype(np.float32)
        self.b_f32 = np.random.uniform(-2, 2, n).astype(np.float32)
        self.a_f16 = _f32_to_f16_array(self.a_f32)
        self.b_f16 = _f32_to_f16_array(self.b_f32)
        self.y_f16 = np.empty(n, dtype=np.uint16)
        self.y_f32 = np.empty(n, dtype=np.float32)

    def _call_f32(self, name):
        fn = getattr(self.lib, f"aria_{name}_f32")
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        fn.restype = None
        fn(
            self.a_f32.ctypes.data,
            self.b_f32.ctypes.data,
            self.y_f32.ctypes.data,
            self.n,
        )

    def _call_f16(self, name):
        fn = getattr(self.lib, f"aria_{name}_f16")
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        fn.restype = None
        fn(
            self.a_f16.ctypes.data,
            self.b_f16.ctypes.data,
            self.y_f16.ctypes.data,
            self.n,
        )

    def test_add_f16(self):
        self._call_f32("add")
        self._call_f16("add")
        _assert_close_f16(self.y_f16, self.y_f32, "add_f16")

    def test_mul_f16(self):
        self._call_f32("mul")
        self._call_f16("mul")
        _assert_close_f16(self.y_f16, self.y_f32, "mul_f16")


# ── Matmul fp16 ──────────────────────────────────────────────────────


class TestMatmulF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("M,K,N", [(2, 3, 2), (16, 32, 16), (8, 8, 8)])
    def test_matmul_f16(self, M, K, N):
        np.random.seed(42)
        A_f32 = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
        B_f32 = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
        A_f16 = _f32_to_f16_array(A_f32)
        B_f16 = _f32_to_f16_array(B_f32)
        C_f16 = np.empty((M, N), dtype=np.uint16)
        C_f32 = np.zeros((M, N), dtype=np.float32)

        # f32 reference
        fn32 = self.lib.aria_matmul_f32
        fn32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn32.restype = None
        fn32(A_f32.ctypes.data, B_f32.ctypes.data, C_f32.ctypes.data, M, K, N)

        # fp16
        fn16 = self.lib.aria_matmul_f16
        fn16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn16.restype = None
        fn16(A_f16.ctypes.data, B_f16.ctypes.data, C_f16.ctypes.data, M, K, N)

        # Wider tolerance for matmul due to accumulated error from K dot-products
        actual = _f16_to_f32_array(C_f16)
        np.testing.assert_allclose(
            actual, C_f32, atol=5e-2, rtol=1e-2, err_msg=f"matmul_f16 {M}x{K}x{N}"
        )


# ── Softmax fp16 ─────────────────────────────────────────────────────


class TestSoftmaxF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,dim", [(1, 8), (4, 32), (2, 64)])
    def test_softmax_f16(self, batch, dim):
        np.random.seed(42)
        x_f32 = np.random.uniform(-3, 3, (batch, dim)).astype(np.float32)
        x_f16 = _f32_to_f16_array(x_f32)
        y_f32 = np.empty((batch, dim), dtype=np.float32)
        y_f16 = np.empty((batch, dim), dtype=np.uint16)

        fn32 = self.lib.aria_softmax_f32
        fn32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn32.restype = None
        fn32(x_f32.ctypes.data, y_f32.ctypes.data, batch, dim)

        fn16 = self.lib.aria_softmax_f16
        fn16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn16.restype = None
        fn16(x_f16.ctypes.data, y_f16.ctypes.data, batch, dim)

        _assert_close_f16(y_f16, y_f32, f"softmax_f16 {batch}x{dim}")

    def test_softmax_f16_sums_to_one(self):
        np.random.seed(42)
        batch, dim = 4, 32
        x_f32 = np.random.uniform(-3, 3, (batch, dim)).astype(np.float32)
        x_f16 = _f32_to_f16_array(x_f32)
        y_f16 = np.empty((batch, dim), dtype=np.uint16)

        fn16 = self.lib.aria_softmax_f16
        fn16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn16.restype = None
        fn16(x_f16.ctypes.data, y_f16.ctypes.data, batch, dim)

        y_f32 = _f16_to_f32_array(y_f16)
        for b in range(batch):
            row_sum = y_f32[b].sum()
            assert abs(row_sum - 1.0) < 0.02, (
                f"softmax_f16 batch {b} sum={row_sum:.4f}, expected ~1.0"
            )


# ── RMSNorm fp16 ─────────────────────────────────────────────────────


class TestRMSNormF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib

    @pytest.mark.parametrize("batch,dim", [(1, 16), (4, 64), (2, 128)])
    def test_rmsnorm_f16(self, batch, dim):
        np.random.seed(42)
        x_f32 = np.random.uniform(-2, 2, (batch, dim)).astype(np.float32)
        w_f32 = np.random.uniform(0.5, 1.5, dim).astype(np.float32)
        x_f16 = _f32_to_f16_array(x_f32)
        w_f16 = _f32_to_f16_array(w_f32)
        y_f32 = np.empty((batch, dim), dtype=np.float32)
        y_f16 = np.empty((batch, dim), dtype=np.uint16)
        eps = 1e-5

        fn32 = self.lib.aria_rmsnorm_f32
        fn32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        fn32.restype = None
        fn32(x_f32.ctypes.data, w_f32.ctypes.data, y_f32.ctypes.data, batch, dim, eps)

        fn16 = self.lib.aria_rmsnorm_f16
        fn16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        fn16.restype = None
        fn16(x_f16.ctypes.data, w_f16.ctypes.data, y_f16.ctypes.data, batch, dim, eps)

        _assert_close_f16(y_f16, y_f32, f"rmsnorm_f16 {batch}x{dim}")


# ── Registry fp16 support ────────────────────────────────────────────


class TestRegistryF16:
    @pytest.fixture(autouse=True)
    def setup(self, native_lib):
        self.lib = native_lib
        self.lib.aria_registry_init()

    @pytest.mark.parametrize("op", ["relu", "gelu", "silu", "sigmoid"])
    def test_registry_has_f16_unary(self, op):
        """Verify that fp16 ops are registered (callable via direct C symbol)."""
        # Just verify the C symbol exists and is callable
        fn = getattr(self.lib, f"aria_{op}_f16")
        assert fn is not None, f"aria_{op}_f16 symbol not found"

    @pytest.mark.parametrize("op", ["add", "mul"])
    def test_registry_has_f16_binary(self, op):
        fn = getattr(self.lib, f"aria_{op}_f16")
        assert fn is not None, f"aria_{op}_f16 symbol not found"

    def test_registry_has_matmul_f16(self):
        fn = getattr(self.lib, "aria_matmul_f16")
        assert fn is not None

    def test_registry_has_softmax_f16(self):
        fn = getattr(self.lib, "aria_softmax_f16")
        assert fn is not None

    def test_registry_has_rmsnorm_f16(self):
        fn = getattr(self.lib, "aria_rmsnorm_f16")
        assert fn is not None
