"""Benchmark: SubgraphDispatcher (single Rust call) vs per-op dispatch (N Python->C calls).

Quantifies the latency profile of batching entire graph execution into a single
Rust scheduler call versus dispatching each op individually through the Cython bridge.

Key insight: per-op dispatch has N Python->C roundtrips for N ops, while subgraph
dispatch has 1 Python->Rust call that executes all N ops in native space. However,
the subgraph path currently includes IR JSON serialization + Rust parsing overhead
on each call, which adds fixed cost.

We benchmark three paths:
1. Per-op dispatch: N individual Cython bridge calls (Python loop + N C calls)
2. Subgraph dispatch (end-to-end): IR conversion + Rust scheduler (realistic path)
3. Subgraph dispatch (pre-converted IR): Rust scheduler only (amortized path)

The pre-converted IR path shows the "steady-state" benefit when IR is cached,
while the end-to-end path shows the current realistic cost.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

# Skip if native runtime is not available.
aria_scheduler = pytest.importorskip(
    "research.scientist.aria_scheduler",
    reason="aria_scheduler.so (Rust PyO3 module) not built or not importable",
)

from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import graph_to_native_ir_json
from research.scientist.native_runner import (
    dispatch_graph_native,
    dispatch_op_native,
    SubgraphDispatcher,
)

pytestmark = pytest.mark.native


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _make_small_chain(dim: int = 256) -> ComputationGraph:
    """input -> relu -> sigmoid -> output  (2 non-input ops)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    sig = g.add_op("sigmoid", [relu])
    g.set_output(sig)
    return g


def _make_medium_chain(dim: int = 256) -> ComputationGraph:
    """input -> relu -> sigmoid -> tanh -> exp -> sigmoid -> output  (5 non-input ops)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    n1 = g.add_op("relu", [inp])
    n2 = g.add_op("sigmoid", [n1])
    n3 = g.add_op("tanh", [n2])
    n4 = g.add_op("exp", [n3])
    n5 = g.add_op("sigmoid", [n4])
    g.set_output(n5)
    return g


def _make_diamond(dim: int = 256) -> ComputationGraph:
    """input -> relu, input -> sigmoid; add(relu, sigmoid) -> output  (3 non-input ops)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    sig = g.add_op("sigmoid", [inp])
    add = g.add_op("add", [relu, sig])
    g.set_output(add)
    return g


def _make_large_chain(dim: int = 256) -> ComputationGraph:
    """10 chained unary ops: relu -> sigmoid -> tanh -> ... -> exp -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    ops_seq = [
        "relu",
        "sigmoid",
        "tanh",
        "relu",
        "sigmoid",
        "tanh",
        "relu",
        "sigmoid",
        "tanh",
        "exp",
    ]
    prev = inp
    for op_name in ops_seq:
        prev = g.add_op(op_name, [prev])
    g.set_output(prev)
    return g


# ---------------------------------------------------------------------------
# Execution paths
# ---------------------------------------------------------------------------


def _execute_per_op(graph: ComputationGraph, x: np.ndarray) -> np.ndarray:
    """Per-op dispatch: one Python->C call per op node (N roundtrips)."""
    topo = graph.topological_order()
    buffers = {}
    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            buffers[nid] = x.copy()
            continue
        inputs = [buffers[iid] for iid in node.input_ids]
        buffers[nid] = dispatch_op_native(node.op_name, *inputs)
    return buffers[graph._output_node_id]


def _execute_subgraph_e2e(graph: ComputationGraph, x: np.ndarray) -> np.ndarray:
    """Subgraph dispatch end-to-end: IR conversion + Rust scheduler."""
    return dispatch_graph_native(graph, x)


def _execute_subgraph_preconverted(ir_json: str, x_list: list) -> list:
    """Subgraph dispatch with pre-converted IR: Rust scheduler only."""
    if hasattr(aria_scheduler, "execute_graph_with_stats"):
        result = aria_scheduler.execute_graph_with_stats(ir_json, x_list)
        return result["output"]
    return aria_scheduler.execute_graph(ir_json, x_list)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _benchmark(fn, *, warmup: int = 20, iterations: int = 200):
    """Run fn() for warmup+iterations rounds, return (mean_ms, std_ms, times_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt * 1000.0)
    return statistics.mean(times), statistics.stdev(times), times


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 256
ITERATIONS = 200
WARMUP = 30


# ---------------------------------------------------------------------------
# Tests: Correctness
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Verify all three paths produce identical results."""

    @pytest.mark.parametrize(
        "builder",
        [
            _make_small_chain,
            _make_medium_chain,
            _make_diamond,
            _make_large_chain,
        ],
        ids=["small", "medium", "diamond", "large"],
    )
    def test_paths_agree(self, builder):
        g = builder(DIM)
        x = np.clip(np.random.randn(DIM), -2.0, 2.0).astype(np.float32)

        per_op = _execute_per_op(g, x)
        subgraph = _execute_subgraph_e2e(g, x)

        ir_json = graph_to_native_ir_json(g)
        preconv = np.array(
            _execute_subgraph_preconverted(ir_json, x.ravel().tolist()),
            dtype=np.float32,
        )

        np.testing.assert_allclose(per_op, subgraph, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(per_op, preconv, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Benchmarks
# ---------------------------------------------------------------------------


class TestSmallChainBenchmark:
    """input -> relu -> sigmoid (2 non-input ops)."""

    def test_benchmark(self):
        g = _make_small_chain(DIM)
        x = np.random.randn(DIM).astype(np.float32)
        ir_json = graph_to_native_ir_json(g)
        x_list = x.ravel().tolist()

        per_op_mean, per_op_std, _ = _benchmark(
            lambda: _execute_per_op(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_e2e_mean, sub_e2e_std, _ = _benchmark(
            lambda: _execute_subgraph_e2e(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_pre_mean, sub_pre_std, _ = _benchmark(
            lambda: _execute_subgraph_preconverted(ir_json, x_list),
            warmup=WARMUP,
            iterations=ITERATIONS,
        )

        print(
            f"\n  Small chain (2 ops, dim={DIM}):"
            f"\n    Per-op (N Cython calls): {per_op_mean:.4f} +/- {per_op_std:.4f} ms"
            f"\n    Subgraph (end-to-end):   {sub_e2e_mean:.4f} +/- {sub_e2e_std:.4f} ms"
            f"\n    Subgraph (pre-conv IR):  {sub_pre_mean:.4f} +/- {sub_pre_std:.4f} ms"
            f"\n    IR overhead:             {sub_e2e_mean - sub_pre_mean:.4f} ms"
            f"\n    Speedup (pre-conv):      {per_op_mean / sub_pre_mean:.2f}x"
        )

        # For 2 ops, we just document the overhead. No strict assertion.
        # The pre-converted path should be fast.
        assert sub_pre_mean < 1.0, "Pre-converted subgraph should be sub-millisecond"


class TestMediumChainBenchmark:
    """input -> relu -> sigmoid -> tanh -> exp -> sigmoid (5 non-input ops)."""

    def test_benchmark(self):
        g = _make_medium_chain(DIM)
        x = np.clip(np.random.randn(DIM), -3.0, 3.0).astype(np.float32)
        ir_json = graph_to_native_ir_json(g)
        x_list = x.ravel().tolist()

        per_op_mean, per_op_std, _ = _benchmark(
            lambda: _execute_per_op(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_e2e_mean, sub_e2e_std, _ = _benchmark(
            lambda: _execute_subgraph_e2e(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_pre_mean, sub_pre_std, _ = _benchmark(
            lambda: _execute_subgraph_preconverted(ir_json, x_list),
            warmup=WARMUP,
            iterations=ITERATIONS,
        )

        print(
            f"\n  Medium chain (5 ops, dim={DIM}):"
            f"\n    Per-op (N Cython calls): {per_op_mean:.4f} +/- {per_op_std:.4f} ms"
            f"\n    Subgraph (end-to-end):   {sub_e2e_mean:.4f} +/- {sub_e2e_std:.4f} ms"
            f"\n    Subgraph (pre-conv IR):  {sub_pre_mean:.4f} +/- {sub_pre_std:.4f} ms"
            f"\n    IR overhead:             {sub_e2e_mean - sub_pre_mean:.4f} ms"
            f"\n    Speedup (pre-conv):      {per_op_mean / sub_pre_mean:.2f}x"
        )

        assert sub_pre_mean < 1.0, "Pre-converted subgraph should be sub-millisecond"


class TestDiamondBenchmark:
    """input -> relu, input -> sigmoid; add(relu, sigmoid) (3 non-input ops, diamond)."""

    def test_benchmark(self):
        g = _make_diamond(DIM)
        x = np.random.randn(DIM).astype(np.float32)
        ir_json = graph_to_native_ir_json(g)
        x_list = x.ravel().tolist()

        per_op_mean, per_op_std, _ = _benchmark(
            lambda: _execute_per_op(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_e2e_mean, sub_e2e_std, _ = _benchmark(
            lambda: _execute_subgraph_e2e(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_pre_mean, sub_pre_std, _ = _benchmark(
            lambda: _execute_subgraph_preconverted(ir_json, x_list),
            warmup=WARMUP,
            iterations=ITERATIONS,
        )

        print(
            f"\n  Diamond (3 ops, dim={DIM}):"
            f"\n    Per-op (N Cython calls): {per_op_mean:.4f} +/- {per_op_std:.4f} ms"
            f"\n    Subgraph (end-to-end):   {sub_e2e_mean:.4f} +/- {sub_e2e_std:.4f} ms"
            f"\n    Subgraph (pre-conv IR):  {sub_pre_mean:.4f} +/- {sub_pre_std:.4f} ms"
            f"\n    IR overhead:             {sub_e2e_mean - sub_pre_mean:.4f} ms"
            f"\n    Speedup (pre-conv):      {per_op_mean / sub_pre_mean:.2f}x"
        )

        assert sub_pre_mean < 1.0, "Pre-converted subgraph should be sub-millisecond"


class TestLargeChainBenchmark:
    """10 chained unary ops -- this is where the roundtrip reduction should matter most."""

    def test_benchmark(self):
        g = _make_large_chain(DIM)
        x = np.clip(np.random.randn(DIM), -2.0, 2.0).astype(np.float32)
        ir_json = graph_to_native_ir_json(g)
        x_list = x.ravel().tolist()

        per_op_mean, per_op_std, _ = _benchmark(
            lambda: _execute_per_op(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_e2e_mean, sub_e2e_std, _ = _benchmark(
            lambda: _execute_subgraph_e2e(g, x), warmup=WARMUP, iterations=ITERATIONS
        )
        sub_pre_mean, sub_pre_std, _ = _benchmark(
            lambda: _execute_subgraph_preconverted(ir_json, x_list),
            warmup=WARMUP,
            iterations=ITERATIONS,
        )

        print(
            f"\n  Large chain (10 ops, dim={DIM}):"
            f"\n    Per-op (N Cython calls): {per_op_mean:.4f} +/- {per_op_std:.4f} ms"
            f"\n    Subgraph (end-to-end):   {sub_e2e_mean:.4f} +/- {sub_e2e_std:.4f} ms"
            f"\n    Subgraph (pre-conv IR):  {sub_pre_mean:.4f} +/- {sub_pre_std:.4f} ms"
            f"\n    IR overhead:             {sub_e2e_mean - sub_pre_mean:.4f} ms"
            f"\n    Speedup (pre-conv):      {per_op_mean / sub_pre_mean:.2f}x"
        )

        assert sub_pre_mean < 1.0, "Pre-converted subgraph should be sub-millisecond"


# ---------------------------------------------------------------------------
# Tests: SubgraphDispatcher class
# ---------------------------------------------------------------------------


class TestSubgraphDispatcherClass:
    """Test the SubgraphDispatcher class itself."""

    def test_all_native_detection(self):
        g = _make_small_chain(DIM)
        supported = {"relu", "sigmoid", "tanh", "exp", "add", "mul", "sub"}
        dispatcher = SubgraphDispatcher(g, supported)
        assert dispatcher.all_native is True

    def test_unsupported_op_detection(self):
        g = _make_small_chain(DIM)
        dispatcher = SubgraphDispatcher(g, {"relu"})  # sigmoid missing
        assert dispatcher.all_native is False
        result = dispatcher.try_dispatch(np.random.randn(DIM).astype(np.float32))
        assert result is None

    def test_try_dispatch_produces_correct_result(self):
        g = _make_diamond(DIM)
        supported = {"relu", "sigmoid", "tanh", "exp", "add", "mul", "sub"}
        dispatcher = SubgraphDispatcher(g, supported)
        x = np.random.randn(DIM).astype(np.float32)
        result = dispatcher.try_dispatch(x)
        assert result is not None
        expected = _execute_per_op(g, x)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_stats_tracking(self):
        g = _make_small_chain(DIM)
        dispatcher = SubgraphDispatcher(g, {"relu", "sigmoid"})
        x = np.random.randn(DIM).astype(np.float32)
        dispatcher.try_dispatch(x)
        dispatcher.try_dispatch(x)
        stats = dispatcher.stats
        assert stats["all_native"] is True
        assert stats["subgraph_dispatches"] == 2
        assert stats["subgraph_fallbacks"] == 0


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestScalingSummary:
    """Summary benchmark showing overhead profile across graph sizes."""

    def test_scaling_trend(self):
        configs = [
            ("small_chain (2 ops)", _make_small_chain),
            ("diamond (3 ops)", _make_diamond),
            ("medium_chain (5 ops)", _make_medium_chain),
            ("large_chain (10 ops)", _make_large_chain),
        ]

        print("\n" + "=" * 82)
        print(
            f"  Subgraph vs Per-Op Dispatch Benchmark (dim={DIM}, {ITERATIONS} iters)"
        )
        print("=" * 82)
        print(
            f"  {'Graph':<22} {'Per-Op':>10} {'Sub E2E':>10} {'Sub Pre':>10}"
            f" {'IR Cost':>10} {'Pre Speedup':>12}"
        )
        print(f"  {'':22} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'':>12}")
        print("-" * 82)

        for name, builder in configs:
            g = builder(DIM)
            x = np.clip(np.random.randn(DIM), -2.0, 2.0).astype(np.float32)
            ir_json = graph_to_native_ir_json(g)
            x_list = x.ravel().tolist()

            per_op_mean, _, _ = _benchmark(
                lambda g=g, x=x: _execute_per_op(g, x),
                warmup=WARMUP,
                iterations=ITERATIONS,
            )
            sub_e2e_mean, _, _ = _benchmark(
                lambda g=g, x=x: _execute_subgraph_e2e(g, x),
                warmup=WARMUP,
                iterations=ITERATIONS,
            )
            sub_pre_mean, _, _ = _benchmark(
                lambda ir=ir_json, xl=x_list: _execute_subgraph_preconverted(ir, xl),
                warmup=WARMUP,
                iterations=ITERATIONS,
            )

            ir_cost = sub_e2e_mean - sub_pre_mean
            speedup = per_op_mean / sub_pre_mean if sub_pre_mean > 0 else float("inf")

            print(
                f"  {name:<22} {per_op_mean:>10.4f} {sub_e2e_mean:>10.4f}"
                f" {sub_pre_mean:>10.4f} {ir_cost:>10.4f} {speedup:>10.2f}x"
            )

        print("=" * 82)
        print(
            "  Note: 'Sub E2E' includes IR JSON conversion on every call."
            "\n  'Sub Pre' uses pre-converted IR (amortized path)."
            "\n  'IR Cost' = E2E - Pre = overhead of graph_to_native_ir_json()."
        )
        print("=" * 82)
