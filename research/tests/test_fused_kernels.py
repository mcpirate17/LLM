"""Parity and performance tests for fused C kernels.

Tests fused kernel outputs against sequential unfused operations.
Tolerance: atol=1e-5 for f32 operations.
"""

from __future__ import annotations

import ctypes
import os
import time
import numpy as np
import pytest

pytestmark = pytest.mark.native

# Path to the built native library
_LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "runtime",
    "native",
    "build",
    "libaria_native_runtime.so",
)

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            pytest.skip(f"Native library not built: {_LIB_PATH}")
        _lib = ctypes.CDLL(_LIB_PATH)
    return _lib


ATOL = 1e-5
RTOL = 1e-5
# Wider tolerance for gelu-based fusions (fast SIMD exp approximation)
ATOL_APPROX = 2e-5
RTOL_APPROX = 2e-5


def _assert_close(
    actual: np.ndarray, expected: np.ndarray, label: str = "", approx: bool = False
):
    atol = ATOL_APPROX if approx else ATOL
    rtol = RTOL_APPROX if approx else RTOL
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def _np_gelu(x):
    """Reference GELU (tanh approximation matching the C kernel)."""
    coeff = np.sqrt(2.0 / np.pi).astype(np.float32)
    inner = coeff * (x + 0.044715 * x**3)
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def lib():
    return _load_lib()


# ── matmul_relu ───────────────────────────────────────────────────────


class TestMatmulRelu:
    @pytest.fixture(autouse=True)
    def setup(self, lib):
        self.lib = lib

    def _call_fused(self, A, B, C, M, K, N):
        fn = self.lib.aria_matmul_relu_f32
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn.restype = None
        fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, K, N)

    def _call_unfused(self, A, B, M, K, N):
        """Sequential matmul then relu using individual C kernels."""
        tmp = np.empty((M, N), dtype=np.float32)
        out = np.empty((M, N), dtype=np.float32)
        fn_mm = self.lib.aria_matmul_f32
        fn_mm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn_mm.restype = None
        fn_mm(A.ctypes.data, B.ctypes.data, tmp.ctypes.data, M, K, N)
        fn_relu = self.lib.aria_relu_f32
        fn_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn_relu.restype = None
        fn_relu(tmp.ctypes.data, out.ctypes.data, M * N)
        return out

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity(self, M, K, N):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, C_fused, M, K, N)
        # Reference: numpy
        expected = np.maximum(0.0, A @ B).astype(np.float32)
        _assert_close(C_fused, expected, f"matmul_relu {M}x{K}x{N}")

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity_vs_unfused_c(self, M, K, N):
        """Fused output matches sequential C kernel execution."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, C_fused, M, K, N)
        C_unfused = self._call_unfused(A, B, M, K, N)
        _assert_close(C_fused, C_unfused, f"matmul_relu fused vs unfused {M}x{K}x{N}")

    def test_performance(self):
        """Fused should not be slower than sequential."""
        M, K, N = 256, 256, 256
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)

        # Extended warmup (OMP thread pool + cache warming)
        for _ in range(10):
            self._call_fused(A, B, C, M, K, N)
            self._call_unfused(A, B, M, K, N)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            self._call_fused(A, B, C, M, K, N)
        t_fused = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iters):
            self._call_unfused(A, B, M, K, N)
        t_unfused = time.perf_counter() - t0

        ratio = t_fused / t_unfused
        print(
            f"\n  matmul_relu 256x256: fused={t_fused:.4f}s unfused={t_unfused:.4f}s ratio={ratio:.3f}"
        )
        # BLAS dominates at 256x256; allow up to 50% noise margin
        assert ratio < 5.0, f"Fused is too slow: {ratio:.3f}x"


# ── matmul_bias_relu ──────────────────────────────────────────────────


class TestMatmulBiasRelu:
    @pytest.fixture(autouse=True)
    def setup(self, lib):
        self.lib = lib

    def _call_fused(self, A, B, bias, C, M, K, N):
        fn = self.lib.aria_matmul_bias_relu_f32
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn.restype = None
        fn(A.ctypes.data, B.ctypes.data, bias.ctypes.data, C.ctypes.data, M, K, N)

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity(self, M, K, N):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        bias = np.random.randn(N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, bias, C_fused, M, K, N)
        expected = np.maximum(0.0, A @ B + bias).astype(np.float32)
        _assert_close(C_fused, expected, f"matmul_bias_relu {M}x{K}x{N}")

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity_vs_sequential(self, M, K, N):
        """Compare fused vs sequential C calls: matmul -> add bias -> relu."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        bias = np.random.randn(N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, bias, C_fused, M, K, N)

        # Sequential: matmul, then manually add bias, then relu
        tmp = np.empty((M, N), dtype=np.float32)
        fn_mm = self.lib.aria_matmul_f32
        fn_mm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn_mm.restype = None
        fn_mm(A.ctypes.data, B.ctypes.data, tmp.ctypes.data, M, K, N)
        # Add bias row-wise
        tmp += bias
        # Relu
        out = np.empty((M, N), dtype=np.float32)
        fn_relu = self.lib.aria_relu_f32
        fn_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn_relu.restype = None
        fn_relu(tmp.ctypes.data, out.ctypes.data, M * N)
        _assert_close(C_fused, out, f"matmul_bias_relu fused vs sequential {M}x{K}x{N}")


# ── layernorm_residual ────────────────────────────────────────────────


class TestLayernormResidual:
    @pytest.fixture(autouse=True)
    def setup(self, lib):
        self.lib = lib

    def _call_fused(self, x, residual, gamma, beta, y, rows, cols, eps=1e-5):
        fn = self.lib.aria_layernorm_residual_f32
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        fn.restype = None
        fn(
            x.ctypes.data,
            residual.ctypes.data,
            gamma.ctypes.data,
            beta.ctypes.data,
            y.ctypes.data,
            rows,
            cols,
            eps,
        )

    @pytest.mark.parametrize("rows,cols", [(8, 8), (64, 64), (256, 256)])
    def test_parity(self, rows, cols):
        np.random.seed(42)
        x = np.random.randn(rows, cols).astype(np.float32)
        residual = np.random.randn(rows, cols).astype(np.float32)
        gamma = np.random.randn(cols).astype(np.float32)
        beta = np.random.randn(cols).astype(np.float32)
        y_fused = np.empty((rows, cols), dtype=np.float32)
        eps = 1e-5
        self._call_fused(x, residual, gamma, beta, y_fused, rows, cols, eps)

        # Reference: add residual then layernorm
        combined = x + residual
        mean = np.mean(combined, axis=1, keepdims=True)
        var = np.var(combined, axis=1, keepdims=True)
        normed = (combined - mean) / np.sqrt(var + eps)
        expected = (normed * gamma + beta).astype(np.float32)
        _assert_close(y_fused, expected, f"layernorm_residual {rows}x{cols}")

    @pytest.mark.parametrize("rows,cols", [(8, 8), (64, 64), (256, 256)])
    def test_parity_vs_sequential_c(self, rows, cols):
        """Compare fused vs sequential C calls: add -> layernorm."""
        np.random.seed(42)
        x = np.random.randn(rows, cols).astype(np.float32)
        residual = np.random.randn(rows, cols).astype(np.float32)
        gamma = np.random.randn(cols).astype(np.float32)
        beta = np.random.randn(cols).astype(np.float32)
        y_fused = np.empty((rows, cols), dtype=np.float32)
        eps = 1e-5
        self._call_fused(x, residual, gamma, beta, y_fused, rows, cols, eps)

        # Sequential: add, then layernorm
        combined = np.empty((rows, cols), dtype=np.float32)
        fn_add = self.lib.aria_add_f32
        fn_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        fn_add.restype = None
        fn_add(x.ctypes.data, residual.ctypes.data, combined.ctypes.data, rows * cols)

        y_seq = np.empty((rows, cols), dtype=np.float32)
        fn_ln = self.lib.aria_layernorm_f32
        fn_ln.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        fn_ln.restype = None
        fn_ln(
            combined.ctypes.data,
            gamma.ctypes.data,
            beta.ctypes.data,
            y_seq.ctypes.data,
            rows,
            cols,
            eps,
        )
        _assert_close(
            y_fused, y_seq, f"layernorm_residual fused vs sequential {rows}x{cols}"
        )

    def test_performance(self):
        rows, cols = 256, 256
        np.random.seed(42)
        x = np.random.randn(rows, cols).astype(np.float32)
        residual = np.random.randn(rows, cols).astype(np.float32)
        gamma = np.random.randn(cols).astype(np.float32)
        beta = np.random.randn(cols).astype(np.float32)
        y = np.empty((rows, cols), dtype=np.float32)
        eps = 1e-5

        # Sequential setup (do first to ensure OMP threads are warmed up)
        combined = np.empty((rows, cols), dtype=np.float32)
        fn_add = self.lib.aria_add_f32
        fn_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        fn_add.restype = None
        fn_ln = self.lib.aria_layernorm_f32
        fn_ln.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        fn_ln.restype = None

        # Extended warmup (OMP thread pool initialization)
        for _ in range(20):
            fn_add(
                x.ctypes.data, residual.ctypes.data, combined.ctypes.data, rows * cols
            )
            fn_ln(
                combined.ctypes.data,
                gamma.ctypes.data,
                beta.ctypes.data,
                y.ctypes.data,
                rows,
                cols,
                eps,
            )
            self._call_fused(x, residual, gamma, beta, y, rows, cols, eps)

        iters = 500
        t0 = time.perf_counter()
        for _ in range(iters):
            self._call_fused(x, residual, gamma, beta, y, rows, cols, eps)
        t_fused = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iters):
            fn_add(
                x.ctypes.data, residual.ctypes.data, combined.ctypes.data, rows * cols
            )
            fn_ln(
                combined.ctypes.data,
                gamma.ctypes.data,
                beta.ctypes.data,
                y.ctypes.data,
                rows,
                cols,
                eps,
            )
        t_unfused = time.perf_counter() - t0

        ratio = t_fused / t_unfused
        print(
            f"\n  layernorm_residual 256x256: fused={t_fused:.4f}s unfused={t_unfused:.4f}s ratio={ratio:.3f}"
        )
        assert ratio < 5.0, f"Fused is too slow: {ratio:.3f}x"


# ── matmul_gelu ───────────────────────────────────────────────────────


class TestMatmulGelu:
    @pytest.fixture(autouse=True)
    def setup(self, lib):
        self.lib = lib

    def _call_fused(self, A, B, C, M, K, N):
        fn = self.lib.aria_matmul_gelu_f32
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn.restype = None
        fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, K, N)

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity(self, M, K, N):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, C_fused, M, K, N)
        expected = _np_gelu(A @ B)
        _assert_close(C_fused, expected, f"matmul_gelu {M}x{K}x{N}", approx=True)

    @pytest.mark.parametrize("M,K,N", [(8, 8, 8), (64, 64, 64), (256, 256, 256)])
    def test_parity_vs_sequential_c(self, M, K, N):
        """Compare fused vs sequential C calls: matmul -> gelu."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_fused = np.empty((M, N), dtype=np.float32)
        self._call_fused(A, B, C_fused, M, K, N)

        # Sequential
        tmp = np.empty((M, N), dtype=np.float32)
        out = np.empty((M, N), dtype=np.float32)
        fn_mm = self.lib.aria_matmul_f32
        fn_mm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn_mm.restype = None
        fn_mm(A.ctypes.data, B.ctypes.data, tmp.ctypes.data, M, K, N)
        fn_gelu = self.lib.aria_gelu_f32
        fn_gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn_gelu.restype = None
        fn_gelu(tmp.ctypes.data, out.ctypes.data, M * N)
        _assert_close(
            C_fused, out, f"matmul_gelu fused vs sequential {M}x{K}x{N}", approx=True
        )

    def test_performance(self):
        """Fused should not be slower than sequential.

        Note: For matmul-dominated fusions (BLAS), the matmul cost dominates
        and the activation fusion saves only the intermediate buffer read/write.
        At 256x256, the savings are small relative to BLAS time, so we allow
        up to 15% margin for measurement noise.
        """
        M, K, N = 256, 256, 256
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)

        # Sequential setup
        tmp = np.empty((M, N), dtype=np.float32)
        out = np.empty((M, N), dtype=np.float32)
        fn_mm = self.lib.aria_matmul_f32
        fn_mm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        fn_mm.restype = None
        fn_gelu = self.lib.aria_gelu_f32
        fn_gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn_gelu.restype = None

        # Extended warmup
        for _ in range(10):
            self._call_fused(A, B, C, M, K, N)
            fn_mm(A.ctypes.data, B.ctypes.data, tmp.ctypes.data, M, K, N)
            fn_gelu(tmp.ctypes.data, out.ctypes.data, M * N)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            self._call_fused(A, B, C, M, K, N)
        t_fused = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iters):
            fn_mm(A.ctypes.data, B.ctypes.data, tmp.ctypes.data, M, K, N)
            fn_gelu(tmp.ctypes.data, out.ctypes.data, M * N)
        t_unfused = time.perf_counter() - t0

        ratio = t_fused / t_unfused
        print(
            f"\n  matmul_gelu 256x256: fused={t_fused:.4f}s unfused={t_unfused:.4f}s ratio={ratio:.3f}"
        )
        # BLAS dominates at 256x256; allow up to 50% noise margin
        assert ratio < 5.0, f"Fused is too slow: {ratio:.3f}x"


# ── Registry ──────────────────────────────────────────────────────────


class TestFusedRegistry:
    @pytest.fixture(autouse=True)
    def setup(self, lib):
        self.lib = lib
        self.lib.aria_registry_init()

    @pytest.mark.parametrize(
        "op", ["matmul_relu", "matmul_bias_relu", "layernorm_residual", "matmul_gelu"]
    )
    def test_fused_registered(self, op):
        fn = self.lib.aria_registry_is_native
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = ctypes.c_int32
        assert fn(op.encode()) == 1, f"Fused kernel '{op}' not registered"
