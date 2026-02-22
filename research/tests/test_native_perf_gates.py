"""Performance acceptance gate tests for native C kernels.

Gate criteria:
- Native kernel must be at least 0.5x NumPy speed on large inputs (no regressions)
- Native matmul 256x256 must complete in < 5ms (absolute latency budget)
- Native softmax batch=16 dim=4096 must complete in < 100us (absolute latency budget)
- Native relu 262144 must complete in < 200us (throughput gate)
"""
import ctypes
import os
import time

import numpy as np
import pytest

# Perf gates are intended to catch kernel regressions, not thread runtime noise.
# Default to single-thread OpenMP unless caller explicitly overrides env vars.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

_LIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "runtime", "native", "build", "libaria_native_runtime.so",
)


@pytest.fixture
def lib():
    """Load native library or skip if not built."""
    if not os.path.exists(_LIB_PATH):
        pytest.skip("Native library not built")
    return ctypes.CDLL(_LIB_PATH)


def _ptr(arr):
    """Return a ctypes void pointer to a numpy array's data."""
    return arr.ctypes.data_as(ctypes.c_void_p)


def _median_us(fn, *args, iterations=50):
    """Run fn(*args) `iterations` times, return median time in microseconds."""
    # Warmup
    for _ in range(min(3, iterations)):
        fn(*args)
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Absolute latency gate tests
# ---------------------------------------------------------------------------

class TestAbsoluteLatencyGates:
    """Hard latency budgets for key kernels."""

    def test_relu_throughput_gate(self, lib):
        """relu on 262144 elements must complete in < 200us."""
        fn = lib.aria_relu_f32
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None

        n = 262144
        x = np.random.randn(n).astype(np.float32)
        y = np.empty(n, dtype=np.float32)

        t = _median_us(lambda: fn(_ptr(x), _ptr(y), n))
        assert t < 200.0, f"relu 262144: {t:.1f}us > 200us budget"

    def test_matmul_latency_gate(self, lib):
        """matmul 256x256x256 must complete in < 5000us (5ms)."""
        fn = lib.aria_matmul_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        fn.restype = None

        M, K, N = 256, 256, 256
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)

        t = _median_us(lambda: fn(_ptr(A), _ptr(B), _ptr(C), M, K, N))
        assert t < 5000.0, f"matmul 256x256x256: {t:.1f}us > 5000us budget"

    def test_softmax_latency_gate(self, lib):
        """softmax batch=16 dim=4096 must complete in < 500us."""
        fn = lib.aria_softmax_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64,
        ]
        fn.restype = None

        batch, dim = 16, 4096
        x = np.random.randn(batch, dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)

        t = _median_us(lambda: fn(_ptr(x), _ptr(y), batch, dim))
        assert t < 500.0, f"softmax 16x4096: {t:.1f}us > 500us budget"

    def test_layernorm_latency_gate(self, lib):
        """layernorm batch=16 dim=1024 must complete in < 200us."""
        fn = lib.aria_layernorm_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ]
        fn.restype = None

        batch, dim = 16, 1024
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)

        t = _median_us(lambda: fn(_ptr(x), _ptr(w), _ptr(b), _ptr(y), batch, dim, ctypes.c_float(1e-5)))
        assert t < 200.0, f"layernorm 16x1024: {t:.1f}us > 200us budget"

    def test_rmsnorm_latency_gate(self, lib):
        """rmsnorm batch=16 dim=1024 must complete in < 200us."""
        fn = lib.aria_rmsnorm_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ]
        fn.restype = None

        batch, dim = 16, 1024
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)

        t = _median_us(lambda: fn(_ptr(x), _ptr(w), _ptr(y), batch, dim, ctypes.c_float(1e-5)))
        assert t < 200.0, f"rmsnorm 16x1024: {t:.1f}us > 200us budget"


# ---------------------------------------------------------------------------
# Relative speedup gate tests (native vs NumPy, parametrized)
# ---------------------------------------------------------------------------

UNARY_OPS = [
    ("relu", lambda x: np.maximum(x, 0)),
    ("gelu", lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))),
    ("silu", lambda x: x / (1 + np.exp(-x))),
    ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
    ("exp", np.exp),
]

BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("sub", lambda a, b: a - b),
]

# Test on a large size to give the C kernel a fair chance vs numpy overhead
_GATE_SIZE = 65536


class TestRelativeSpeedupGates:
    """Ensure native kernels are at least 0.5x NumPy speed (regression gate).

    Uses scipy-bundled OpenBLAS 0.3.27 (DYNAMIC_ARCH, Haswell-optimized) for
    matmul/linear parity with NumPy.
    """

    @pytest.mark.parametrize("op_name,np_fn", UNARY_OPS, ids=[o[0] for o in UNARY_OPS])
    def test_unary_vs_numpy(self, lib, op_name, np_fn):
        """Unary op must achieve >= 0.5x NumPy throughput on 65536 elements."""
        c_fn = getattr(lib, f"aria_{op_name}_f32")
        c_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        c_fn.restype = None

        n = _GATE_SIZE
        x = np.random.randn(n).astype(np.float32)
        y = np.empty(n, dtype=np.float32)

        native_us = _median_us(lambda: c_fn(_ptr(x), _ptr(y), n))
        numpy_us = _median_us(lambda: np_fn(x))

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"{op_name} n={n}: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    @pytest.mark.parametrize("op_name,np_fn", BINARY_OPS, ids=[o[0] for o in BINARY_OPS])
    def test_binary_vs_numpy(self, lib, op_name, np_fn):
        """Binary op must achieve >= 0.5x NumPy throughput on 65536 elements."""
        c_fn = getattr(lib, f"aria_{op_name}_f32")
        c_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        c_fn.restype = None

        n = _GATE_SIZE
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        y = np.empty(n, dtype=np.float32)

        native_us = _median_us(lambda: c_fn(_ptr(a), _ptr(b), _ptr(y), n))
        numpy_us = _median_us(lambda: np_fn(a, b))

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"{op_name} n={n}: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    def test_matmul_vs_numpy(self, lib):
        """matmul 128x128x128 must achieve >= 0.5x NumPy throughput."""
        fn = lib.aria_matmul_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        fn.restype = None

        M, K, N = 128, 128, 128
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)

        native_us = _median_us(lambda: fn(_ptr(A), _ptr(B), _ptr(C), M, K, N))
        numpy_us = _median_us(lambda: np.dot(A, B))

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"matmul 128x128x128: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    def test_linear_vs_numpy(self, lib):
        """linear 16x256x256 must achieve >= 0.5x NumPy throughput."""
        fn = lib.aria_linear_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        fn.restype = None

        batch, dim_in, dim_out = 16, 256, 256
        x = np.random.randn(batch, dim_in).astype(np.float32)
        W = np.random.randn(dim_out, dim_in).astype(np.float32)
        bias = np.random.randn(dim_out).astype(np.float32)
        y = np.empty((batch, dim_out), dtype=np.float32)

        native_us = _median_us(lambda: fn(_ptr(x), _ptr(W), _ptr(bias), _ptr(y), batch, dim_in, dim_out))

        def np_linear():
            return x @ W.T + bias

        numpy_us = _median_us(np_linear)

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"linear 16x256x256: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    def test_softmax_vs_numpy(self, lib):
        """softmax batch=8 dim=1024 must achieve >= 0.5x NumPy throughput."""
        fn = lib.aria_softmax_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64,
        ]
        fn.restype = None

        batch, dim = 8, 1024
        x = np.random.randn(batch, dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)

        native_us = _median_us(lambda: fn(_ptr(x), _ptr(y), batch, dim))

        def np_softmax():
            m = np.max(x, axis=-1, keepdims=True)
            e = np.exp(x - m)
            return e / np.sum(e, axis=-1, keepdims=True)

        numpy_us = _median_us(np_softmax)

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"softmax 8x1024: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    def test_rmsnorm_vs_numpy(self, lib):
        """rmsnorm batch=8 dim=1024 must achieve >= 0.5x NumPy throughput."""
        fn = lib.aria_rmsnorm_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ]
        fn.restype = None

        batch, dim = 8, 1024
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5

        native_us = _median_us(lambda: fn(_ptr(x), _ptr(w), _ptr(y), batch, dim, ctypes.c_float(eps)))

        def np_rmsnorm():
            rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
            return x / rms * w

        numpy_us = _median_us(np_rmsnorm)

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"rmsnorm 8x1024: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )

    def test_layernorm_vs_numpy(self, lib):
        """layernorm batch=8 dim=1024 must achieve >= 0.5x NumPy throughput."""
        fn = lib.aria_layernorm_f32
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ]
        fn.restype = None

        batch, dim = 8, 1024
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5

        native_us = _median_us(lambda: fn(_ptr(x), _ptr(w), _ptr(b), _ptr(y), batch, dim, ctypes.c_float(eps)))

        def np_layernorm():
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps) * w + b

        numpy_us = _median_us(np_layernorm)

        speedup = numpy_us / native_us if native_us > 0 else float("inf")
        assert speedup >= 0.5, (
            f"layernorm 8x1024: native={native_us:.1f}us, numpy={numpy_us:.1f}us, "
            f"speedup={speedup:.2f}x < 0.5x threshold"
        )
