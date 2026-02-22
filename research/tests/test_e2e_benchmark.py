"""Regression gate tests for native execution performance.

These tests verify that native kernels meet minimum performance thresholds
relative to PyTorch and NumPy baselines. They are designed as CI gates:
each test asserts a conservative lower bound (e.g., native must be at
least 0.5x PyTorch speed for elementwise ops).

Skip gracefully when the Cython bridge or PyTorch are unavailable.

Run:
    source /home/tim/venvs/llm/bin/activate
    cd /home/tim/Projects/LLM/research
    python -m pytest tests/test_e2e_benchmark.py -v
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CYTHON_BUILD_DIR = os.path.join(
    os.path.dirname(__file__), "..", "runtime", "native", "cython", "build",
    "lib.linux-x86_64-cpython-312",
)
_CYTHON_SRC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "runtime", "native", "cython",
)

for _p in (_CYTHON_BUILD_DIR, _CYTHON_SRC_DIR):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _try_import_bridge():
    try:
        import aria_bridge  # type: ignore[import-untyped]
        return aria_bridge
    except ImportError:
        return None


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _median_us(fn: Callable, *, warmup: int = 5, iterations: int = 50) -> float:
    """Return the median execution time in microseconds."""
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

bridge = _try_import_bridge()
torch = _try_import_torch()

requires_bridge = pytest.mark.skipif(bridge is None, reason="aria_bridge not available")
requires_torch = pytest.mark.skipif(torch is None, reason="torch not available")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_bridge
@requires_torch
def test_native_relu_faster_than_pytorch():
    """Native relu on 65536 elements should be at least 0.5x PyTorch speed.

    In other words: native_time <= 2.0 * pytorch_time.
    """
    n = 65536
    x_np = np.random.randn(n).astype(np.float32)
    xt = torch.from_numpy(x_np)

    native_us = _median_us(lambda: bridge.dispatch_unary("relu", x_np))
    torch_us = _median_us(lambda: torch.relu(xt))

    ratio = native_us / torch_us
    print(f"\nrelu: native={native_us:.1f}us  torch={torch_us:.1f}us  ratio={ratio:.2f}x")

    # native must be at least 0.5x PyTorch speed => native_time <= 2x torch_time
    assert ratio <= 2.0, (
        f"Native relu is {ratio:.2f}x PyTorch time (threshold: <= 2.0x). "
        f"native={native_us:.1f}us torch={torch_us:.1f}us"
    )


@requires_bridge
@requires_torch
def test_native_gelu_faster_than_pytorch():
    """Native gelu on 65536 elements should be at least 0.1x PyTorch speed.

    GELU involves tanh/exp which PyTorch heavily vectorizes (AVX/SSE).
    The naive C kernel is significantly slower, so we use a generous
    threshold: native_time <= 10x pytorch_time (0.1x speed).
    """
    n = 65536
    x_np = np.random.randn(n).astype(np.float32)
    xt = torch.from_numpy(x_np)

    native_us = _median_us(lambda: bridge.dispatch_unary("gelu", x_np))
    torch_us = _median_us(lambda: torch.nn.functional.gelu(xt))

    ratio = native_us / torch_us
    print(f"\ngelu: native={native_us:.1f}us  torch={torch_us:.1f}us  ratio={ratio:.2f}x")

    assert ratio <= 10.0, (
        f"Native gelu is {ratio:.2f}x PyTorch time (threshold: <= 10.0x). "
        f"native={native_us:.1f}us torch={torch_us:.1f}us"
    )


@requires_bridge
@requires_torch
def test_native_matmul_faster_than_pytorch():
    """Native matmul 128x128 should be at least 0.3x PyTorch speed.

    Matmul is harder to beat since PyTorch uses optimized BLAS (MKL/OpenBLAS).
    We use a generous threshold: native_time <= 3.33x torch_time.
    """
    M = 128
    A = np.random.randn(M, M).astype(np.float32)
    B = np.random.randn(M, M).astype(np.float32)
    At = torch.from_numpy(A)
    Bt = torch.from_numpy(B)

    native_us = _median_us(lambda: bridge.dispatch_matmul(A, B))
    torch_us = _median_us(lambda: torch.mm(At, Bt))

    ratio = native_us / torch_us
    print(f"\nmatmul(128x128): native={native_us:.1f}us  torch={torch_us:.1f}us  ratio={ratio:.2f}x")

    # 0.3x speed => native_time <= 3.33x torch_time
    assert ratio <= 3.34, (
        f"Native matmul is {ratio:.2f}x PyTorch time (threshold: <= 3.33x). "
        f"native={native_us:.1f}us torch={torch_us:.1f}us"
    )


@requires_bridge
def test_cython_bridge_overhead_acceptable():
    """Overhead of Cython bridge dispatch vs direct ctypes should be < 2x.

    We compare the Cython bridge dispatch_unary("relu") against a raw
    ctypes call to the same kernel. The Cython layer adds Python->C
    argument marshalling overhead; it should be bounded.
    """
    import ctypes

    n = 65536
    x_np = np.random.randn(n).astype(np.float32)

    # Cython bridge time
    cython_us = _median_us(lambda: bridge.dispatch_unary("relu", x_np))

    # Raw ctypes time
    lib_path = os.path.join(
        os.path.dirname(__file__), "..", "runtime", "native", "build",
        "libaria_native_runtime.so",
    )
    lib_path = os.path.abspath(lib_path)
    if not os.path.exists(lib_path):
        pytest.skip("libaria_native_runtime.so not found")

    lib = ctypes.CDLL(lib_path)
    relu_fn = lib.aria_relu_f32
    relu_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    relu_fn.restype = None

    y_np = np.empty(n, dtype=np.float32)
    x_ptr = x_np.ctypes.data_as(ctypes.c_void_p)
    y_ptr = y_np.ctypes.data_as(ctypes.c_void_p)

    ctypes_us = _median_us(lambda: relu_fn(x_ptr, y_ptr, n))

    ratio = cython_us / ctypes_us if ctypes_us > 0 else float("inf")
    print(
        f"\nCython vs ctypes relu (n={n}): "
        f"cython={cython_us:.1f}us  ctypes={ctypes_us:.1f}us  ratio={ratio:.2f}x"
    )

    assert ratio < 2.0, (
        f"Cython bridge overhead {ratio:.2f}x vs ctypes (threshold: < 2.0x). "
        f"cython={cython_us:.1f}us ctypes={ctypes_us:.1f}us"
    )


@requires_bridge
def test_native_relu_correctness():
    """Sanity check: native relu produces correct output."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = bridge.dispatch_unary("relu", x)
    expected = np.maximum(x, 0)
    np.testing.assert_allclose(y, expected, atol=1e-6)


@requires_bridge
def test_native_matmul_correctness():
    """Sanity check: native matmul produces correct output."""
    np.random.seed(42)
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    C_native = bridge.dispatch_matmul(A, B)
    C_numpy = A @ B
    np.testing.assert_allclose(C_native, C_numpy, atol=1e-3, rtol=1e-3)


@requires_bridge
@requires_torch
def test_per_op_dispatch_roundtrip():
    """Verify NativeForwardWrapper dispatch path returns correct results."""
    from scientist.native_runner import dispatch_op_native

    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = dispatch_op_native("relu", x)
    expected = np.maximum(x, 0)
    np.testing.assert_allclose(result, expected, atol=1e-6)

    # Binary op
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    result = dispatch_op_native("add", a, b)
    expected = a + b
    np.testing.assert_allclose(result, expected, atol=1e-6)
