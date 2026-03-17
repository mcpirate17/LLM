#!/usr/bin/env python3
"""End-to-end benchmark: native execution paths vs PyTorch baseline.

Compares four execution paths for compiled computation graphs:
1. PyTorch forward pass (baseline) -- standard _execute_op -> _OP_DISPATCH
2. Per-op native dispatch via NativeForwardWrapper -> Cython bridge -> C kernels
3. Cython bridge direct -- raw aria_bridge kernel calls
4. Rust scheduler (if available) -- full graph execution via dispatch_graph_native

Also includes an op-level comparison table (relu, gelu, matmul) across all backends.

Usage:
    source /home/tim/venvs/llm/bin/activate
    cd /home/tim/Projects/LLM/research
    python runtime/native/bench/bench_e2e.py
    python runtime/native/bench/bench_e2e.py --quick   # fewer iterations for CI
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def bench_fn(fn: Callable, *, warmup: int = 5, iterations: int = 100) -> float:
    """Benchmark a callable; return median time in microseconds."""
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
# Graph builder helper
# ---------------------------------------------------------------------------


def _make_relu_graph(model_dim: int = 64):
    """Create a minimal computation graph: input -> relu -> output."""
    # Ensure the research dir is on sys.path (imports are e.g. synthesis.graph).
    research_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if research_root not in sys.path:
        sys.path.insert(0, research_root)

    from synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    relu_id = g.add_op("relu", [inp])
    g.set_output(relu_id)
    return g


def _make_multi_op_graph(model_dim: int = 64):
    """input -> relu -> gelu -> add(relu, gelu) -> output."""
    research_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if research_root not in sys.path:
        sys.path.insert(0, research_root)

    from synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    relu_id = g.add_op("relu", [inp])
    gelu_id = g.add_op("gelu", [inp])
    add_id = g.add_op("add", [relu_id, gelu_id])
    g.set_output(add_id)
    return g


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_pytorch_forward(
    graph,
    *,
    vocab_size: int,
    max_seq_len: int,
    batch: int,
    seq_len: int,
    iterations: int,
) -> Optional[float]:
    """Benchmark 1: PyTorch compiled model forward pass."""
    try:
        import torch
        from scientist.native_runner import compile_model_native_first as compile_model
    except ImportError:
        return None

    model = compile_model([graph], vocab_size=vocab_size, max_seq_len=max_seq_len)
    model.eval()
    x = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        return bench_fn(lambda: model(x), iterations=iterations)


def bench_pytorch_layer_forward(
    graph,
    *,
    batch: int,
    seq_len: int,
    model_dim: int,
    iterations: int,
) -> Optional[float]:
    """Benchmark PyTorch compiled layer (no embed/lm_head overhead)."""
    try:
        import torch
        from synthesis.compiler import CompiledLayer
    except ImportError:
        return None

    layer = CompiledLayer(graph)
    layer.eval()
    x = torch.randn(batch, seq_len, model_dim)

    with torch.no_grad():
        return bench_fn(lambda: layer(x), iterations=iterations)


def bench_native_per_op(
    *,
    op_name: str = "relu",
    n: int = 512,
    iterations: int = 100,
) -> Optional[float]:
    """Benchmark 2: Per-op native dispatch via NativeForwardWrapper path."""
    try:
        from scientist.native_runner import dispatch_op_native
    except (ImportError, RuntimeError):
        return None

    x = np.random.randn(n).astype(np.float32)
    try:
        dispatch_op_native(op_name, x)  # warm-up / verify it works
    except Exception:
        return None

    return bench_fn(lambda: dispatch_op_native(op_name, x), iterations=iterations)


def bench_cython_bridge_direct(
    *,
    op_name: str = "relu",
    n: int = 65536,
    iterations: int = 100,
) -> Optional[float]:
    """Benchmark 3: Raw Cython bridge kernel call."""
    cython_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cython",
        "build",
        "lib.linux-x86_64-cpython-312",
    )
    if cython_dir not in sys.path:
        sys.path.insert(0, cython_dir)
    # Also try the cython source dir (in case .so is there).
    cython_src = os.path.join(os.path.dirname(__file__), "..", "cython")
    if cython_src not in sys.path:
        sys.path.insert(0, cython_src)

    try:
        import aria_bridge  # type: ignore[import-untyped]
    except ImportError:
        return None

    x = np.random.randn(n).astype(np.float32)
    try:
        aria_bridge.dispatch_unary(op_name, x)  # verify
    except Exception:
        return None

    return bench_fn(
        lambda: aria_bridge.dispatch_unary(op_name, x),
        iterations=iterations,
    )


def bench_rust_scheduler(
    graph,
    *,
    batch: int,
    seq_len: int,
    model_dim: int,
    iterations: int,
) -> Optional[float]:
    """Benchmark 4: Rust scheduler full graph execution."""
    try:
        from scientist.native_runner import dispatch_graph_native
    except (ImportError, RuntimeError):
        return None

    x = np.random.randn(batch, seq_len, model_dim).astype(np.float32)
    try:
        dispatch_graph_native(graph, x)  # verify
    except Exception:
        return None

    return bench_fn(
        lambda: dispatch_graph_native(graph, x),
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Op-level comparison
# ---------------------------------------------------------------------------


def bench_op_comparison(*, n: int = 65536, iterations: int = 100) -> List[Dict]:
    """Compare individual ops across native, numpy, and pytorch."""
    results: List[Dict] = []

    # Import backends
    cython_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cython",
        "build",
        "lib.linux-x86_64-cpython-312",
    )
    if cython_dir not in sys.path:
        sys.path.insert(0, cython_dir)
    cython_src = os.path.join(os.path.dirname(__file__), "..", "cython")
    if cython_src not in sys.path:
        sys.path.insert(0, cython_src)

    try:
        import aria_bridge  # type: ignore

        has_bridge = True
    except ImportError:
        has_bridge = False

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    # -- Unary ops --
    unary_ops = [
        ("relu", lambda x: np.maximum(x, 0)),
        (
            "gelu",
            lambda x: (
                0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            ),
        ),
    ]
    for op_name, np_fn in unary_ops:
        x_np = np.random.randn(n).astype(np.float32)
        row: Dict[str, Any] = {"op": op_name, "n": n}

        # NumPy
        row["numpy_us"] = bench_fn(lambda: np_fn(x_np), iterations=iterations)

        # Native (Cython bridge)
        if has_bridge:
            try:
                row["native_us"] = bench_fn(
                    lambda: aria_bridge.dispatch_unary(op_name, x_np),
                    iterations=iterations,
                )
            except Exception:
                row["native_us"] = None
        else:
            row["native_us"] = None

        # PyTorch
        if has_torch:
            xt = torch.from_numpy(x_np)
            if op_name == "relu":
                row["torch_us"] = bench_fn(
                    lambda: torch.relu(xt), iterations=iterations
                )
            elif op_name == "gelu":
                row["torch_us"] = bench_fn(
                    lambda: torch.nn.functional.gelu(xt),
                    iterations=iterations,
                )
            else:
                row["torch_us"] = None
        else:
            row["torch_us"] = None

        results.append(row)

    # -- Matmul --
    M = 128
    A_np = np.random.randn(M, M).astype(np.float32)
    B_np = np.random.randn(M, M).astype(np.float32)
    row_mm: Dict[str, Any] = {"op": f"matmul({M}x{M})", "n": M * M}

    row_mm["numpy_us"] = bench_fn(lambda: np.dot(A_np, B_np), iterations=iterations)

    if has_bridge:
        try:
            row_mm["native_us"] = bench_fn(
                lambda: aria_bridge.dispatch_matmul(A_np, B_np),
                iterations=iterations,
            )
        except Exception:
            row_mm["native_us"] = None
    else:
        row_mm["native_us"] = None

    if has_torch:
        At = torch.from_numpy(A_np)
        Bt = torch.from_numpy(B_np)
        row_mm["torch_us"] = bench_fn(lambda: torch.mm(At, Bt), iterations=iterations)
    else:
        row_mm["torch_us"] = None

    results.append(row_mm)

    return results


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------


def _fmt(val: Optional[float], width: int = 10) -> str:
    if val is None:
        return "n/a".center(width)
    return f"{val:.1f}".rjust(width)


def _ratio(val: Optional[float], base: Optional[float]) -> str:
    if val is None or base is None or base == 0:
        return "n/a".center(8)
    return f"{val / base:.2f}x".rjust(8)


def print_path_report(
    pt_full: Optional[float],
    pt_layer: Optional[float],
    native_perop: Optional[float],
    cython_direct: Optional[float],
    rust_sched: Optional[float],
    *,
    model_dim: int,
    seq_len: int,
) -> None:
    print(f"\nNative Execution Benchmark (dim={model_dim}, seq={seq_len})")
    print("=" * 60)
    print(f"{'Path':<30} | {'Median (us)':>11} | {'vs PT layer':>11}")
    print("-" * 30 + "-+-" + "-" * 11 + "-+-" + "-" * 11)

    baseline = pt_layer  # use layer forward as the baseline

    rows = [
        ("PyTorch full model fwd", pt_full),
        ("PyTorch layer fwd (baseline)", pt_layer),
        ("Native per-op dispatch", native_perop),
        ("Cython bridge direct", cython_direct),
        ("Rust scheduler (if avail)", rust_sched),
    ]
    for label, val in rows:
        med = _fmt(val, 11)
        ratio = (
            _ratio(baseline, val)
            if val is not None and baseline is not None
            else "n/a".center(11)
        )
        if val is not None and baseline is not None and val > 0:
            ratio = f"{baseline / val:.2f}x".rjust(11)
        else:
            ratio = "n/a".rjust(11)
        if label.startswith("PyTorch layer"):
            ratio = "1.00x".rjust(11)
        print(f"{label:<30} | {med} | {ratio}")


def print_op_report(op_results: List[Dict]) -> None:
    print("\nOp-level comparison (elementwise n=65536, matmul 128x128):")
    print("-" * 78)
    print(
        f"{'Op':<16} | {'Native (us)':>11} | {'NumPy (us)':>11} | "
        f"{'PyTorch (us)':>12} | {'Native/PT':>10}"
    )
    print(
        "-" * 16
        + "-+-"
        + "-" * 11
        + "-+-"
        + "-" * 11
        + "-+-"
        + "-" * 12
        + "-+-"
        + "-" * 10
    )

    for r in op_results:
        native_s = _fmt(r.get("native_us"), 11)
        numpy_s = _fmt(r.get("numpy_us"), 11)
        torch_s = _fmt(r.get("torch_us"), 12)
        nat = r.get("native_us")
        pt = r.get("torch_us")
        if nat is not None and pt is not None and pt > 0:
            ratio_s = f"{nat / pt:.2f}x".rjust(10)
        else:
            ratio_s = "n/a".rjust(10)
        print(f"{r['op']:<16} | {native_s} | {numpy_s} | {torch_s} | {ratio_s}")
    print("-" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E native execution benchmark")
    parser.add_argument("--quick", action="store_true", help="Fewer iterations for CI")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--seq", type=int, default=8, help="Sequence length")
    parser.add_argument("--vocab", type=int, default=512, help="Vocab size")
    args = parser.parse_args()

    iters = 20 if args.quick else 100
    model_dim = args.dim
    seq_len = args.seq
    vocab_size = args.vocab
    batch = 1

    np.random.seed(42)

    # Ensure research package is importable
    research_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if research_root not in sys.path:
        sys.path.insert(0, research_root)

    # Build graph
    print(f"Building graph: input(dim={model_dim}) -> relu -> output")
    graph = _make_relu_graph(model_dim)

    # ── Path-level benchmarks ──────────────────────────────────────

    print("Benchmarking PyTorch full model forward...")
    pt_full = bench_pytorch_forward(
        graph,
        vocab_size=vocab_size,
        max_seq_len=seq_len * 2,
        batch=batch,
        seq_len=seq_len,
        iterations=iters,
    )

    print("Benchmarking PyTorch layer forward...")
    pt_layer = bench_pytorch_layer_forward(
        graph,
        batch=batch,
        seq_len=seq_len,
        model_dim=model_dim,
        iterations=iters,
    )

    print("Benchmarking native per-op dispatch...")
    native_perop = bench_native_per_op(
        op_name="relu",
        n=model_dim * seq_len,
        iterations=iters,
    )

    print("Benchmarking Cython bridge direct...")
    cython_direct = bench_cython_bridge_direct(
        op_name="relu",
        n=model_dim * seq_len,
        iterations=iters,
    )

    print("Benchmarking Rust scheduler...")
    rust_sched = bench_rust_scheduler(
        graph,
        batch=batch,
        seq_len=seq_len,
        model_dim=model_dim,
        iterations=iters,
    )

    print_path_report(
        pt_full,
        pt_layer,
        native_perop,
        cython_direct,
        rust_sched,
        model_dim=model_dim,
        seq_len=seq_len,
    )

    # ── Op-level comparison ────────────────────────────────────────

    print("\nRunning op-level comparison...")
    op_results = bench_op_comparison(n=65536, iterations=iters)
    print_op_report(op_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
