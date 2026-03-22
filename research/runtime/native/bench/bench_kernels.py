#!/usr/bin/env python3
"""Performance benchmark suite for aria native C kernels.

Benchmarks each kernel against NumPy and (optionally) PyTorch baselines.
Reports median times and speedup ratios.

Acceptance gate: every native kernel must be at least 0.5x NumPy speed
on the largest tested size; exit code 1 if any gate fails.

Usage:
    python bench_kernels.py              # full suite
    python bench_kernels.py --quick      # reduced iterations for CI
"""

import ctypes
import os
import sys
import time
import argparse

import numpy as np

_LIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "build",
    "libaria_native_runtime.so",
)

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def bench_fn(fn, *args, iterations=100):
    """Benchmark a callable, return median time in microseconds."""
    # Warmup
    for _ in range(min(5, iterations)):
        fn(*args)
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # ns -> us
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Library loader + typed wrapper helpers
# ---------------------------------------------------------------------------


def load_library(path):
    """Load the native shared library and return the ctypes handle."""
    lib = ctypes.CDLL(path)
    return lib


def _ptr(arr):
    """Return a ctypes void pointer to a numpy array's data."""
    return arr.ctypes.data_as(ctypes.c_void_p)


def _f32_ptr(arr):
    """Return a ctypes float pointer to a numpy float32 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _setup_unary(lib, name):
    """Configure argtypes/restype for a unary kernel."""
    fn = getattr(lib, f"aria_{name}_f32")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    return fn


def _setup_binary(lib, name):
    """Configure argtypes/restype for a binary kernel."""
    fn = getattr(lib, f"aria_{name}_f32")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    return fn


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_unary(lib, op_name, np_fn, sizes, iterations, torch_fn=None):
    """Benchmark a unary elementwise kernel at various sizes."""
    results = []
    c_fn = _setup_unary(lib, op_name)

    for n in sizes:
        x = np.random.randn(n).astype(np.float32)
        y = np.empty(n, dtype=np.float32)

        # Native
        native_us = bench_fn(
            lambda _x=x, _y=y, _n=n: c_fn(_ptr(_x), _ptr(_y), _n),
            iterations=iterations,
        )

        # NumPy
        numpy_us = bench_fn(lambda _x=x: np_fn(_x), iterations=iterations)

        # PyTorch (optional)
        torch_us = None
        if torch_fn is not None:
            try:
                import torch

                xt = torch.from_numpy(x)
                torch_us = bench_fn(lambda _xt=xt: torch_fn(_xt), iterations=iterations)
            except ImportError:
                pass

        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": op_name,
                "size": n,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


def bench_binary(lib, op_name, np_fn, sizes, iterations, torch_fn=None):
    """Benchmark a binary elementwise kernel at various sizes."""
    results = []
    c_fn = _setup_binary(lib, op_name)

    for n in sizes:
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        y = np.empty(n, dtype=np.float32)

        native_us = bench_fn(
            lambda _a=a, _b=b, _y=y, _n=n: c_fn(_ptr(_a), _ptr(_b), _ptr(_y), _n),
            iterations=iterations,
        )

        numpy_us = bench_fn(lambda _a=a, _b=b: np_fn(_a, _b), iterations=iterations)

        torch_us = None
        if torch_fn is not None:
            try:
                import torch

                at = torch.from_numpy(a)
                bt = torch.from_numpy(b)
                torch_us = bench_fn(
                    lambda _at=at, _bt=bt: torch_fn(_at, _bt), iterations=iterations
                )
            except ImportError:
                pass

        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": op_name,
                "size": n,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


def bench_matmul(lib, sizes_mat, iterations):
    """Benchmark dense matmul: C[M,N] = A[M,K] @ B[K,N]."""
    fn = lib.aria_matmul_f32
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    fn.restype = None

    results = []
    for M, K, N in sizes_mat:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)

        native_us = bench_fn(
            lambda _A=A, _B=B, _C=C, _M=M, _K=K, _N=N: fn(
                _ptr(_A), _ptr(_B), _ptr(_C), _M, _K, _N
            ),
            iterations=iterations,
        )

        numpy_us = bench_fn(lambda _A=A, _B=B: np.dot(_A, _B), iterations=iterations)

        torch_us = None
        try:
            import torch

            At = torch.from_numpy(A)
            Bt = torch.from_numpy(B)
            torch_us = bench_fn(
                lambda _At=At, _Bt=Bt: torch.mm(_At, _Bt), iterations=iterations
            )
        except ImportError:
            pass

        size_str = f"{M}x{K}x{N}"
        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": "matmul",
                "size": size_str,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


def bench_linear(lib, sizes_linear, iterations):
    """Benchmark linear projection: y = x @ W^T + bias."""
    fn = lib.aria_linear_f32
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

    results = []
    for batch, dim_in, dim_out in sizes_linear:
        x = np.random.randn(batch, dim_in).astype(np.float32)
        W = np.random.randn(dim_out, dim_in).astype(np.float32)
        bias = np.random.randn(dim_out).astype(np.float32)
        y = np.empty((batch, dim_out), dtype=np.float32)

        native_us = bench_fn(
            lambda _x=x, _W=W, _b=bias, _y=y, _ba=batch, _di=dim_in, _do=dim_out: fn(
                _ptr(_x), _ptr(_W), _ptr(_b), _ptr(_y), _ba, _di, _do
            ),
            iterations=iterations,
        )

        def np_linear(_x=x, _W=W, _b=bias):
            return _x @ _W.T + _b

        numpy_us = bench_fn(np_linear, iterations=iterations)

        torch_us = None
        try:
            import torch

            xt = torch.from_numpy(x)
            Wt = torch.from_numpy(W)
            bt = torch.from_numpy(bias)
            torch_us = bench_fn(
                lambda _xt=xt, _Wt=Wt, _bt=bt: torch.nn.functional.linear(
                    _xt, _Wt, _bt
                ),
                iterations=iterations,
            )
        except ImportError:
            pass

        size_str = f"{batch}x{dim_in}x{dim_out}"
        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": "linear",
                "size": size_str,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


def bench_rmsnorm(lib, sizes_norm, iterations):
    """Benchmark RMSNorm: y = x / rms(x) * weight."""
    fn = lib.aria_rmsnorm_f32
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    fn.restype = None

    results = []
    for batch, dim in sizes_norm:
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5

        native_us = bench_fn(
            lambda _x=x, _w=w, _y=y, _b=batch, _d=dim: fn(
                _ptr(_x), _ptr(_w), _ptr(_y), _b, _d, ctypes.c_float(eps)
            ),
            iterations=iterations,
        )

        def np_rmsnorm(_x=x, _w=w):
            rms = np.sqrt(np.mean(_x * _x, axis=-1, keepdims=True) + eps)
            return _x / rms * _w

        numpy_us = bench_fn(np_rmsnorm, iterations=iterations)

        size_str = f"{batch}x{dim}"
        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")

        results.append(
            {
                "op": "rmsnorm",
                "size": size_str,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": None,
                "speedup_np": speedup_np,
                "speedup_pt": None,
            }
        )
    return results


def bench_softmax(lib, sizes_sm, iterations):
    """Benchmark softmax: y = exp(x - max) / sum(exp(x - max))."""
    fn = lib.aria_softmax_f32
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    fn.restype = None

    results = []
    for batch, dim in sizes_sm:
        x = np.random.randn(batch, dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)

        native_us = bench_fn(
            lambda _x=x, _y=y, _b=batch, _d=dim: fn(_ptr(_x), _ptr(_y), _b, _d),
            iterations=iterations,
        )

        def np_softmax(_x=x):
            m = np.max(_x, axis=-1, keepdims=True)
            e = np.exp(_x - m)
            return e / np.sum(e, axis=-1, keepdims=True)

        numpy_us = bench_fn(np_softmax, iterations=iterations)

        torch_us = None
        try:
            import torch

            xt = torch.from_numpy(x)
            torch_us = bench_fn(
                lambda _xt=xt: torch.nn.functional.softmax(_xt, dim=-1),
                iterations=iterations,
            )
        except ImportError:
            pass

        size_str = f"{batch}x{dim}"
        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": "softmax",
                "size": size_str,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


def bench_layernorm(lib, sizes_ln, iterations):
    """Benchmark LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias."""
    fn = lib.aria_layernorm_f32
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    fn.restype = None

    results = []
    for batch, dim in sizes_ln:
        x = np.random.randn(batch, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        y = np.empty((batch, dim), dtype=np.float32)
        eps = 1e-5

        native_us = bench_fn(
            lambda _x=x, _w=w, _b=b, _y=y, _ba=batch, _d=dim: fn(
                _ptr(_x), _ptr(_w), _ptr(_b), _ptr(_y), _ba, _d, ctypes.c_float(eps)
            ),
            iterations=iterations,
        )

        def np_layernorm(_x=x, _w=w, _b=b):
            mean = np.mean(_x, axis=-1, keepdims=True)
            var = np.var(_x, axis=-1, keepdims=True)
            return (_x - mean) / np.sqrt(var + eps) * _w + _b

        numpy_us = bench_fn(np_layernorm, iterations=iterations)

        torch_us = None
        try:
            import torch

            xt = torch.from_numpy(x)
            wt = torch.from_numpy(w)
            bt = torch.from_numpy(b)
            torch_us = bench_fn(
                lambda _xt=xt, _wt=wt, _bt=bt, _d=dim: torch.nn.functional.layer_norm(
                    _xt, [_d], _wt, _bt, eps
                ),
                iterations=iterations,
            )
        except ImportError:
            pass

        size_str = f"{batch}x{dim}"
        speedup_np = numpy_us / native_us if native_us > 0 else float("inf")
        speedup_pt = torch_us / native_us if torch_us and native_us > 0 else None

        results.append(
            {
                "op": "layernorm",
                "size": size_str,
                "native_us": native_us,
                "numpy_us": numpy_us,
                "torch_us": torch_us,
                "speedup_np": speedup_np,
                "speedup_pt": speedup_pt,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(all_results):
    """Print a formatted table of benchmark results and return gate failure count."""
    has_torch = any(r["torch_us"] is not None for r in all_results)

    if has_torch:
        hdr = (
            f"{'Op':<12} {'Size':<14} {'Native(us)':<12} {'NumPy(us)':<12} "
            f"{'Torch(us)':<12} {'vs NumPy':<10} {'vs Torch':<10} {'Gate':<6}"
        )
        sep = "-" * 88
    else:
        hdr = (
            f"{'Op':<12} {'Size':<14} {'Native(us)':<12} {'NumPy(us)':<12} "
            f"{'vs NumPy':<10} {'Gate':<6}"
        )
        sep = "-" * 66

    print(f"\n{hdr}")
    print(sep)

    gate_failures = 0
    for r in all_results:
        gate = "PASS" if r["speedup_np"] >= 0.5 else "FAIL"
        if gate == "FAIL":
            gate_failures += 1

        size_str = str(r["size"])
        if has_torch:
            torch_str = (
                f"{r['torch_us']:<12.1f}"
                if r["torch_us"] is not None
                else f"{'n/a':<12}"
            )
            pt_str = (
                f"{r['speedup_pt']:<10.2f}"
                if r["speedup_pt"] is not None
                else f"{'n/a':<10}"
            )
            print(
                f"{r['op']:<12} {size_str:<14} {r['native_us']:<12.1f} "
                f"{r['numpy_us']:<12.1f} {torch_str} {r['speedup_np']:<10.2f} "
                f"{pt_str} {gate:<6}"
            )
        else:
            print(
                f"{r['op']:<12} {size_str:<14} {r['native_us']:<12.1f} "
                f"{r['numpy_us']:<12.1f} {r['speedup_np']:<10.2f} {gate:<6}"
            )

    print(sep)
    print(f"Total: {len(all_results)} benchmarks, {gate_failures} gate failures")
    if gate_failures:
        print("RESULT: FAIL -- some kernels below 0.5x NumPy regression threshold")
    else:
        print("RESULT: PASS -- all kernels meet acceptance gate (>= 0.5x NumPy)")
    return gate_failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark native C kernels")
    parser.add_argument("--quick", action="store_true", help="Fewer iterations for CI")
    args = parser.parse_args()

    iterations = 20 if args.quick else 100

    if not os.path.exists(_LIB_PATH):
        print(f"SKIP: native library not built at {_LIB_PATH}")
        return 0

    lib = load_library(_LIB_PATH)
    np.random.seed(42)

    # ----- Size configurations -----
    sizes_elem = [256, 4096, 65536, 262144]
    sizes_mat = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
    ]
    sizes_linear = [
        (1, 64, 64),
        (8, 128, 128),
        (16, 256, 256),
        (32, 512, 256),
    ]
    sizes_norm = [
        (1, 256),
        (8, 512),
        (16, 1024),
        (32, 2048),
    ]
    sizes_softmax = [
        (1, 256),
        (8, 1024),
        (16, 4096),
        (32, 4096),
    ]
    sizes_layernorm = [
        (1, 256),
        (8, 512),
        (16, 1024),
        (32, 2048),
    ]

    all_results = []

    # ---- PyTorch function imports (optional) ----
    try:
        import torch

        torch_relu = torch.relu
        torch_sigmoid = torch.sigmoid
        torch_exp = torch.exp

        def torch_gelu(x):
            return torch.nn.functional.gelu(x)

        def torch_silu(x):
            return torch.nn.functional.silu(x)
    except ImportError:
        torch_relu = None
        torch_sigmoid = None
        torch_exp = None
        torch_gelu = None
        torch_silu = None

    # ---- Unary ops ----
    print("Benchmarking unary elementwise ops...")
    unary_ops = [
        ("relu", lambda x: np.maximum(x, 0), torch_relu),
        (
            "gelu",
            lambda x: (
                0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            ),
            torch_gelu,
        ),
        ("silu", lambda x: x / (1 + np.exp(-x)), torch_silu),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x)), torch_sigmoid),
        ("exp", np.exp, torch_exp),
    ]
    for op_name, np_fn, torch_fn in unary_ops:
        all_results.extend(
            bench_unary(lib, op_name, np_fn, sizes_elem, iterations, torch_fn)
        )

    # ---- Binary ops ----
    print("Benchmarking binary elementwise ops...")
    try:
        import torch as _t

        torch_add = _t.add
        torch_mul = _t.mul
        torch_sub = _t.sub
    except ImportError:
        torch_add = torch_mul = torch_sub = None

    binary_ops = [
        ("add", lambda a, b: a + b, torch_add),
        ("mul", lambda a, b: a * b, torch_mul),
        ("sub", lambda a, b: a - b, torch_sub),
    ]
    for op_name, np_fn, torch_fn in binary_ops:
        all_results.extend(
            bench_binary(lib, op_name, np_fn, sizes_elem, iterations, torch_fn)
        )

    # ---- Matmul ----
    print("Benchmarking matmul...")
    all_results.extend(bench_matmul(lib, sizes_mat, iterations))

    # ---- Linear ----
    print("Benchmarking linear projection...")
    all_results.extend(bench_linear(lib, sizes_linear, iterations))

    # ---- RMSNorm ----
    print("Benchmarking rmsnorm...")
    all_results.extend(bench_rmsnorm(lib, sizes_norm, iterations))

    # ---- Softmax ----
    print("Benchmarking softmax...")
    all_results.extend(bench_softmax(lib, sizes_softmax, iterations))

    # ---- LayerNorm ----
    print("Benchmarking layernorm...")
    all_results.extend(bench_layernorm(lib, sizes_layernorm, iterations))

    # ---- Report ----
    gate_failures = print_report(all_results)
    return 1 if gate_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
