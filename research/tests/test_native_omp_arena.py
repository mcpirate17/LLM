"""Tests for arena allocation safety under OpenMP parallel dispatch.

The C kernels use `#pragma omp parallel for if(n > 16384)` for large tensors,
meaning OpenMP threads write into arena-allocated buffers.  This module verifies:

1. Large tensors (>16384 elements) that trigger OpenMP produce correct results
   with arena-allocated output buffers (no corruption from parallel writes).
2. Multiple concurrent graph executions (Python threads) each get isolated
   arenas, so no cross-execution corruption occurs.
3. Repeated execution with arena reuse (via separate calls) is deterministic.

The allocation-then-dispatch pattern is safe because:
- Each `execute_with_arena()` creates a fresh Arena (bump allocator)
- All arena allocations happen sequentially in topological order
- OpenMP parallelism is confined within a single kernel call writing to a
  single pre-allocated buffer with read-only inputs
- 64-byte alignment avoids false sharing between cache lines

Requirements:
- aria_scheduler.so must be built (Rust PyO3 module)
- libaria_native_runtime.so must be built with OpenMP (ARIA_HAS_OPENMP)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

aria_scheduler = pytest.importorskip(
    "research.scientist.aria_scheduler",
    reason="aria_scheduler.so (Rust PyO3 module) not built or not importable",
)

from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import graph_to_native_ir_json

pytestmark = pytest.mark.native


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OMP_THRESHOLD = 16384  # Matches ARIA_OMP_THRESHOLD in kernels.c


def _make_relu_graph(dim: int) -> ComputationGraph:
    """input(dim) -> relu -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    g.set_output(relu)
    return g


def _make_chain_graph(dim: int) -> ComputationGraph:
    """input(dim) -> relu -> sigmoid -> tanh -> output (3-op chain)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    sig = g.add_op("sigmoid", [relu])
    th = g.add_op("tanh", [sig])
    g.set_output(th)
    return g


def _make_residual_graph(dim: int) -> ComputationGraph:
    """input -> relu; add(input, relu) -> sigmoid -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    add = g.add_op("add", [inp, relu])
    sig = g.add_op("sigmoid", [add])
    g.set_output(sig)
    return g


def _make_diamond_graph(dim: int) -> ComputationGraph:
    """input -> sigmoid, input -> tanh; mul(sigmoid, tanh) -> relu -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    sig = g.add_op("sigmoid", [inp])
    th = g.add_op("tanh", [inp])
    mul = g.add_op("mul", [sig, th])
    relu = g.add_op("relu", [mul])
    g.set_output(relu)
    return g


def _numpy_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _numpy_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Test 1: Large tensors that trigger OpenMP (unary ops)
# ---------------------------------------------------------------------------


class TestOpenMPUnaryLargeTensors:
    """Verify correctness with tensors > 16384 elements (OpenMP threshold)."""

    @pytest.mark.parametrize(
        "dim",
        [
            OMP_THRESHOLD + 1,  # Just above threshold
            OMP_THRESHOLD * 2,  # 32768
            OMP_THRESHOLD * 4,  # 65536
            OMP_THRESHOLD * 8,  # 131072
        ],
    )
    def test_relu_large_tensor(self, dim: int):
        """relu on large tensors: negatives zeroed, positives unchanged."""
        graph = _make_relu_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)
        x_list = x.tolist()

        result = aria_scheduler.execute_graph(ir_json, x_list)

        expected = _numpy_relu(x)
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
            err_msg=f"relu mismatch at dim={dim} (OpenMP should be active)",
        )

    @pytest.mark.parametrize(
        "dim",
        [
            OMP_THRESHOLD + 1,
            OMP_THRESHOLD * 4,
        ],
    )
    def test_sigmoid_large_tensor(self, dim: int):
        """sigmoid on large tensors: output in (0, 1)."""
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        sig = g.add_op("sigmoid", [inp])
        g.set_output(sig)
        ir_json = graph_to_native_ir_json(g)

        rng = np.random.RandomState(123)
        x = rng.randn(dim).astype(np.float32) * 5.0
        x_list = x.tolist()

        result = np.array(
            aria_scheduler.execute_graph(ir_json, x_list), dtype=np.float32
        )

        expected = _numpy_sigmoid(x)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"sigmoid mismatch at dim={dim}",
        )
        # All values must be in [0, 1] (exact 0.0/1.0 possible due to f32 saturation)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    @pytest.mark.parametrize(
        "dim",
        [
            OMP_THRESHOLD + 1,
            OMP_THRESHOLD * 4,
        ],
    )
    def test_tanh_large_tensor(self, dim: int):
        """tanh on large tensors: output in (-1, 1)."""
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        th = g.add_op("tanh", [inp])
        g.set_output(th)
        ir_json = graph_to_native_ir_json(g)

        rng = np.random.RandomState(456)
        x = rng.randn(dim).astype(np.float32) * 3.0
        x_list = x.tolist()

        result = np.array(
            aria_scheduler.execute_graph(ir_json, x_list), dtype=np.float32
        )

        expected = np.tanh(x)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"tanh mismatch at dim={dim}",
        )
        # f32 saturation can produce exact -1.0/1.0 for large inputs
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# Test 2: Large tensors with multi-op chains (binary + unary)
# ---------------------------------------------------------------------------


class TestOpenMPMultiOpLargeTensors:
    """Multi-op graphs with large tensors triggering OpenMP at each step."""

    def test_chain_large_tensor(self):
        """relu -> sigmoid -> tanh on 65536 elements."""
        dim = OMP_THRESHOLD * 4
        graph = _make_chain_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(789)
        x = rng.randn(dim).astype(np.float32)
        x_list = x.tolist()

        result = np.array(
            aria_scheduler.execute_graph(ir_json, x_list), dtype=np.float32
        )

        expected = np.tanh(_numpy_sigmoid(_numpy_relu(x)))
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="chain (relu->sigmoid->tanh) mismatch on large tensor",
        )

    def test_residual_large_tensor(self):
        """input -> relu; add(input, relu) -> sigmoid on 32768 elements."""
        dim = OMP_THRESHOLD * 2
        graph = _make_residual_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(321)
        x = rng.randn(dim).astype(np.float32)
        x_list = x.tolist()

        result = np.array(
            aria_scheduler.execute_graph(ir_json, x_list), dtype=np.float32
        )

        relu_x = _numpy_relu(x)
        expected = _numpy_sigmoid(x + relu_x)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="residual graph mismatch on large tensor",
        )

    def test_diamond_large_tensor(self):
        """Diamond: sigmoid + tanh -> mul -> relu on 65536 elements."""
        dim = OMP_THRESHOLD * 4
        graph = _make_diamond_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(654)
        x = rng.randn(dim).astype(np.float32) * 3.0
        x_list = x.tolist()

        result = np.array(
            aria_scheduler.execute_graph(ir_json, x_list), dtype=np.float32
        )

        expected = _numpy_relu(_numpy_sigmoid(x) * np.tanh(x))
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="diamond graph mismatch on large tensor",
        )


# ---------------------------------------------------------------------------
# Test 3: Arena stats with large tensors
# ---------------------------------------------------------------------------


class TestArenaStatsLargeTensors:
    """Verify arena allocation bookkeeping is correct for large tensors."""

    def test_arena_stats_large_relu(self):
        """Arena stats are populated correctly for large relu graph."""
        dim = OMP_THRESHOLD * 4
        graph = _make_relu_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(111)
        x = rng.randn(dim).astype(np.float32).tolist()

        stats = aria_scheduler.execute_graph_with_stats(ir_json, x)

        # 2 nodes: input + relu, both arena-allocated
        assert stats["arena_alloc_count"] == 2
        assert stats["heap_fallback_count"] == 0
        assert stats["arena_bytes_used"] > 0
        assert stats["arena_capacity"] >= stats["arena_bytes_used"]
        assert len(stats["output"]) == dim

    def test_arena_stats_diamond(self):
        """Diamond graph: 5 nodes all arena-allocated."""
        dim = OMP_THRESHOLD * 2
        graph = _make_diamond_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(222)
        x = rng.randn(dim).astype(np.float32).tolist()

        stats = aria_scheduler.execute_graph_with_stats(ir_json, x)

        # 5 nodes: input, sigmoid, tanh, mul, relu
        assert stats["arena_alloc_count"] == 5
        assert stats["heap_fallback_count"] == 0
        # Each node allocates dim * 4 bytes + alignment padding
        min_expected_bytes = 5 * dim * 4
        assert stats["arena_bytes_used"] >= min_expected_bytes


# ---------------------------------------------------------------------------
# Test 4: Concurrent graph executions (arena-per-execution isolation)
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    """Verify that concurrent graph executions don't corrupt each other.

    Each call to execute_graph() / execute_with_arena() creates its own
    Arena instance, so there is no shared mutable state between executions.
    The Python GIL serializes the calls, but this test verifies the
    isolation pattern is correct and would catch any accidental sharing
    of global state (e.g., static buffers, registry corruption).
    """

    def test_concurrent_relu_threads(self):
        """8 threads execute relu on different inputs; all get correct results."""
        dim = OMP_THRESHOLD * 2
        graph = _make_relu_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)
        num_threads = 8

        results = [None] * num_threads
        errors = [None] * num_threads

        def worker(thread_id: int):
            try:
                rng = np.random.RandomState(thread_id * 1000)
                x = rng.randn(dim).astype(np.float32)
                x_list = x.tolist()

                output = aria_scheduler.execute_graph(ir_json, x_list)

                expected = _numpy_relu(x)
                np.testing.assert_allclose(
                    output,
                    expected,
                    atol=1e-6,
                    err_msg=f"thread {thread_id} relu mismatch",
                )
                results[thread_id] = True
            except Exception as e:
                errors[thread_id] = e

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for i in range(num_threads):
            if errors[i] is not None:
                raise AssertionError(f"Thread {i} failed: {errors[i]}") from errors[i]
            assert results[i] is True, f"Thread {i} did not complete"

    def test_concurrent_mixed_graphs(self):
        """4 threads execute different graph topologies concurrently."""
        dim = OMP_THRESHOLD * 2

        graph_fns = [
            (_make_relu_graph, "relu"),
            (_make_chain_graph, "chain"),
            (_make_residual_graph, "residual"),
            (_make_diamond_graph, "diamond"),
        ]

        expected_fns = {
            "relu": lambda x: _numpy_relu(x),
            "chain": lambda x: np.tanh(_numpy_sigmoid(_numpy_relu(x))),
            "residual": lambda x: _numpy_sigmoid(x + _numpy_relu(x)),
            "diamond": lambda x: _numpy_relu(_numpy_sigmoid(x) * np.tanh(x)),
        }

        results = {}
        errors = {}
        lock = threading.Lock()

        def worker(make_fn, name, seed):
            try:
                graph = make_fn(dim=dim)
                ir_json = graph_to_native_ir_json(graph)

                rng = np.random.RandomState(seed)
                x = rng.randn(dim).astype(np.float32) * 3.0
                x_list = x.tolist()

                output = np.array(
                    aria_scheduler.execute_graph(ir_json, x_list),
                    dtype=np.float32,
                )
                expected = expected_fns[name](x)
                np.testing.assert_allclose(
                    output,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{name} graph mismatch in concurrent execution",
                )
                with lock:
                    results[name] = True
            except Exception as e:
                with lock:
                    errors[name] = e

        threads = []
        for i, (fn, name) in enumerate(graph_fns):
            t = threading.Thread(target=worker, args=(fn, name, i * 100))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for name in [n for _, n in graph_fns]:
            if name in errors:
                raise AssertionError(
                    f"Graph '{name}' failed: {errors[name]}"
                ) from errors[name]
            assert results.get(name) is True, f"Graph '{name}' did not complete"

    def test_concurrent_repeated_execution(self):
        """Each of 4 threads executes the same graph 20 times, verifying determinism."""
        dim = OMP_THRESHOLD * 2
        graph = _make_chain_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)
        num_threads = 4
        iterations = 20

        errors = [None] * num_threads

        def worker(thread_id: int):
            try:
                rng = np.random.RandomState(thread_id * 500)
                x = rng.randn(dim).astype(np.float32)
                x_list = x.tolist()

                expected = np.tanh(_numpy_sigmoid(_numpy_relu(x)))

                for iteration in range(iterations):
                    output = np.array(
                        aria_scheduler.execute_graph(ir_json, x_list),
                        dtype=np.float32,
                    )
                    np.testing.assert_allclose(
                        output,
                        expected,
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=(
                            f"thread {thread_id}, iteration {iteration}: "
                            f"non-deterministic result"
                        ),
                    )
            except Exception as e:
                errors[thread_id] = e

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        for i in range(num_threads):
            if errors[i] is not None:
                raise AssertionError(
                    f"Thread {i} non-deterministic: {errors[i]}"
                ) from errors[i]


# ---------------------------------------------------------------------------
# Test 5: ThreadPoolExecutor stress test
# ---------------------------------------------------------------------------


class TestThreadPoolStress:
    """High-concurrency stress test using ThreadPoolExecutor."""

    def test_threadpool_stress_16_workers(self):
        """16 concurrent workers, each executing a large graph once."""
        dim = OMP_THRESHOLD * 2
        graph = _make_residual_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        def task(seed: int) -> bool:
            rng = np.random.RandomState(seed)
            x = rng.randn(dim).astype(np.float32)
            x_list = x.tolist()

            output = np.array(
                aria_scheduler.execute_graph(ir_json, x_list),
                dtype=np.float32,
            )
            expected = _numpy_sigmoid(x + _numpy_relu(x))
            np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
            return True

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(task, seed) for seed in range(16)]
            for f in as_completed(futures):
                assert f.result() is True


# ---------------------------------------------------------------------------
# Test 6: Bit-exact determinism across runs
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify that repeated executions with the same input produce identical output."""

    def test_relu_deterministic_large(self):
        """Same input -> same relu output, 10 runs."""
        dim = OMP_THRESHOLD * 4
        graph = _make_relu_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(999)
        x = rng.randn(dim).astype(np.float32).tolist()

        first_result = aria_scheduler.execute_graph(ir_json, x)
        for run in range(10):
            result = aria_scheduler.execute_graph(ir_json, x)
            assert result == first_result, f"Run {run} differs from first run"

    def test_chain_deterministic_large(self):
        """Same input -> same chain output, 10 runs."""
        dim = OMP_THRESHOLD * 4
        graph = _make_chain_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        rng = np.random.RandomState(888)
        x = rng.randn(dim).astype(np.float32).tolist()

        first_result = aria_scheduler.execute_graph(ir_json, x)
        for run in range(10):
            result = aria_scheduler.execute_graph(ir_json, x)
            assert result == first_result, f"Run {run} differs from first run"


# ---------------------------------------------------------------------------
# Test 7: Below-threshold (no OpenMP) vs above-threshold correctness match
# ---------------------------------------------------------------------------


class TestThresholdBoundary:
    """Results must be identical whether OpenMP kicks in or not."""

    @pytest.mark.parametrize("op_name", ["relu", "sigmoid", "tanh"])
    def test_below_vs_above_threshold(self, op_name: str):
        """Compare results at dim=1024 (no OMP) vs dim=32768 (with OMP)."""
        small_dim = 1024
        large_dim = OMP_THRESHOLD * 2

        # Build graphs
        def make_graph(dim):
            g = ComputationGraph(model_dim=dim)
            inp = g.add_input()
            op = g.add_op(op_name, [inp])
            g.set_output(op)
            return g

        # Use the same data pattern (first 1024 elements)
        rng = np.random.RandomState(777)
        x_small = rng.randn(small_dim).astype(np.float32)

        # For large: tile the small data to fill
        x_large = np.tile(x_small, large_dim // small_dim).astype(np.float32)

        ir_small = graph_to_native_ir_json(make_graph(small_dim))
        ir_large = graph_to_native_ir_json(make_graph(large_dim))

        result_small = np.array(
            aria_scheduler.execute_graph(ir_small, x_small.tolist()),
            dtype=np.float32,
        )
        result_large = np.array(
            aria_scheduler.execute_graph(ir_large, x_large.tolist()),
            dtype=np.float32,
        )

        # The first 1024 elements of the large result should match the small result
        np.testing.assert_allclose(
            result_large[:small_dim],
            result_small,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{op_name}: below-threshold and above-threshold results differ",
        )
