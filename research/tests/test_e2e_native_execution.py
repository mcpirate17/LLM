"""End-to-end tests for the full native execution pipeline.

Pipeline: ComputationGraph -> native_ir.v1 -> Rust scheduler -> C kernels -> output

These tests verify that the complete chain works:
1. Python ComputationGraph is built with the synthesis API
2. native_ir_converter produces valid IR JSON
3. The Rust scheduler (aria_scheduler) parses, sorts, and executes the IR
4. C kernels (libaria_native_runtime) perform the actual computation
5. Results match expected values

Requirements:
- aria_scheduler.so must be built (Rust PyO3 module)
- libaria_native_runtime.so must be built (C kernel library)
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

# Skip entire module if the Rust scheduler is not available.
# The module lives inside the scientist package as a PyO3 .so.
aria_scheduler = pytest.importorskip(
    "research.scientist.aria_scheduler",
    reason="aria_scheduler.so (Rust PyO3 module) not built or not importable",
)

from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import (
    graph_to_native_ir,
    graph_to_native_ir_json,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_relu_graph(dim: int = 4) -> ComputationGraph:
    """input(dim) -> relu -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    g.set_output(relu)
    return g


def _make_chain_graph(dim: int = 4) -> ComputationGraph:
    """input(dim) -> relu -> exp -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    exp = g.add_op("exp", [relu])
    g.set_output(exp)
    return g


def _make_sigmoid_graph(dim: int = 4) -> ComputationGraph:
    """input(dim) -> sigmoid -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    sig = g.add_op("sigmoid", [inp])
    g.set_output(sig)
    return g


def _make_tanh_graph(dim: int = 4) -> ComputationGraph:
    """input(dim) -> tanh -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    th = g.add_op("tanh", [inp])
    g.set_output(th)
    return g


def _make_add_self_graph(dim: int = 4) -> ComputationGraph:
    """input(dim) -> relu; add(input, relu) -> output (residual pattern)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    add = g.add_op("add", [inp, relu])
    g.set_output(add)
    return g


def _make_mul_branches_graph(dim: int = 4) -> ComputationGraph:
    """input -> sigmoid, input -> tanh; mul(sigmoid, tanh) -> output (gating pattern)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    sig = g.add_op("sigmoid", [inp])
    th = g.add_op("tanh", [inp])
    mul = g.add_op("mul", [sig, th])
    g.set_output(mul)
    return g


def _make_long_chain_graph(dim: int = 4) -> ComputationGraph:
    """input -> relu -> sigmoid -> tanh -> exp -> output (4-op chain)"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    sig = g.add_op("sigmoid", [relu])
    th = g.add_op("tanh", [sig])
    exp = g.add_op("exp", [th])
    g.set_output(exp)
    return g


def _make_diamond_graph(dim: int = 4) -> ComputationGraph:
    """input -> relu -> sigmoid; input -> tanh; add(sigmoid, tanh) -> output"""
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    sig = g.add_op("sigmoid", [relu])
    th = g.add_op("tanh", [inp])
    add = g.add_op("add", [sig, th])
    g.set_output(add)
    return g


def _make_double_residual_graph(dim: int = 4) -> ComputationGraph:
    """input -> relu -> add(input, relu) -> sigmoid -> add(relu_add, sigmoid) -> output

    Two sequential residual connections.
    """
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    add1 = g.add_op("add", [inp, relu])
    sig = g.add_op("sigmoid", [add1])
    add2 = g.add_op("add", [add1, sig])
    g.set_output(add2)
    return g


# ---------------------------------------------------------------------------
# Test 1: Rust scheduler parses IR from ComputationGraph
# ---------------------------------------------------------------------------


class TestParseConvertedIR:
    def test_rust_scheduler_parses_converted_ir(self):
        """ComputationGraph -> native_ir.v1 JSON -> parse_graph_ir succeeds."""
        graph = _make_relu_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        result = aria_scheduler.parse_graph_ir(ir_json)

        # parse_graph_ir returns a summary string like "2 nodes"
        assert "2 nodes" in result

    def test_parse_chain_graph(self):
        """Three-node chain parses correctly."""
        graph = _make_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        result = aria_scheduler.parse_graph_ir(ir_json)
        assert "3 nodes" in result

    def test_parse_rejects_invalid_json(self):
        """Malformed JSON is rejected with a clear error."""
        with pytest.raises(Exception):
            aria_scheduler.parse_graph_ir("{bad json")

    def test_ir_roundtrip_schema_version(self):
        """Converted IR has the required schema_version field."""
        graph = _make_relu_graph()
        ir_doc = graph_to_native_ir(graph)
        assert ir_doc["schema_version"] == "native_ir.v1"
        assert ir_doc["model_dim"] == 4
        assert len(ir_doc["nodes"]) == 2
        assert len(ir_doc["edges"]) == 1


# ---------------------------------------------------------------------------
# Test 2: Topological order matches Python graph
# ---------------------------------------------------------------------------


class TestTopologicalOrder:
    def test_rust_scheduler_topological_order(self):
        """Rust topo order matches Python ComputationGraph.topological_order()."""
        graph = _make_relu_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        rust_order = aria_scheduler.topological_order(ir_json)
        python_order = graph.topological_order()

        assert list(rust_order) == python_order

    def test_chain_topological_order(self):
        """Chain graph: input(0) -> relu(1) -> exp(2)."""
        graph = _make_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        rust_order = aria_scheduler.topological_order(ir_json)
        assert list(rust_order) == [0, 1, 2]

    def test_topological_order_length(self):
        """Number of nodes in topo order matches graph node count."""
        graph = _make_chain_graph(dim=8)
        ir_json = graph_to_native_ir_json(graph)

        rust_order = aria_scheduler.topological_order(ir_json)
        assert len(rust_order) == len(graph.nodes)


# ---------------------------------------------------------------------------
# Test 3: Execute graph — relu
# ---------------------------------------------------------------------------


class TestExecuteRelu:
    def test_rust_execute_graph_relu(self):
        """input -> relu: negative values clamped to zero."""
        graph = _make_relu_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 2.0, 3.5]
        result = aria_scheduler.execute_graph(ir_json, x)

        assert result == [0.0, 0.0, 2.0, 3.5]

    def test_relu_all_negative(self):
        """All-negative input produces all zeros."""
        graph = _make_relu_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        result = aria_scheduler.execute_graph(ir_json, [-5.0, -3.0, -0.1, -100.0])
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_relu_all_positive(self):
        """All-positive input passes through unchanged."""
        graph = _make_relu_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [1.0, 2.0, 3.0, 4.0]
        result = aria_scheduler.execute_graph(ir_json, x)
        assert result == pytest.approx(x)


# ---------------------------------------------------------------------------
# Test 4: Execute graph — relu -> exp chain
# ---------------------------------------------------------------------------


class TestExecuteChain:
    def test_rust_execute_graph_chain(self):
        """input -> relu -> exp: exp(relu(x))."""
        graph = _make_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 1.0, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        # relu([-1, 0, 1, 2]) = [0, 0, 1, 2]
        # exp([0, 0, 1, 2])   = [1.0, 1.0, e, e^2]
        expected = [1.0, 1.0, math.exp(1.0), math.exp(2.0)]

        assert result == pytest.approx(expected, rel=1e-5)

    def test_chain_with_large_values(self):
        """Chain handles moderately large positive values."""
        graph = _make_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [0.0, 0.5, 1.0, 3.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [math.exp(0.0), math.exp(0.5), math.exp(1.0), math.exp(3.0)]
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 5: Execute graph — additional unary ops (sigmoid, tanh)
# ---------------------------------------------------------------------------


class TestExecuteUnaryOps:
    def test_sigmoid(self):
        """input -> sigmoid."""
        graph = _make_sigmoid_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-2.0, 0.0, 1.0, 5.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [1.0 / (1.0 + math.exp(-v)) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_tanh(self):
        """input -> tanh."""
        graph = _make_tanh_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 0.5, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [math.tanh(v) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 6: dispatch_graph_native (full Python -> Rust -> C pipeline)
# ---------------------------------------------------------------------------


class TestDispatchGraphNative:
    def test_dispatch_graph_native_function(self):
        """dispatch_graph_native: Python entry point through full pipeline."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_relu_graph(dim=4)
        x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = np.array([0.0, 0.0, 2.0, 3.5], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_dispatch_graph_native_chain(self):
        """dispatch_graph_native with relu -> exp chain."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_chain_graph(dim=4)
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = np.array([1.0, 1.0, math.exp(1.0), math.exp(2.0)], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 7: Throughput benchmark (sanity, not a strict gate)
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_e2e_benchmark_relu_throughput(self):
        """Run relu on 65536 elements through the Rust scheduler.

        This is a sanity check that large inputs work, not a perf gate.
        """
        dim = 65536
        graph = _make_relu_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        x = list(np.random.randn(dim).astype(np.float32))

        t0 = time.perf_counter()
        result = aria_scheduler.execute_graph(ir_json, x)
        dt = time.perf_counter() - t0

        # Basic correctness: negatives are zeroed
        assert len(result) == dim
        for i in range(dim):
            if x[i] < 0:
                assert result[i] == 0.0
            else:
                assert result[i] == pytest.approx(x[i], rel=1e-5)

        # Just log the latency — no strict threshold
        print(f"\n  relu({dim}) via Rust scheduler: {dt * 1000:.2f} ms")

    def test_chain_throughput(self):
        """relu -> exp on 8192 elements."""
        dim = 8192
        graph = _make_chain_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        # Use small positive values to avoid exp overflow
        x = list(np.clip(np.random.randn(dim).astype(np.float32), -5.0, 5.0))

        t0 = time.perf_counter()
        result = aria_scheduler.execute_graph(ir_json, x)
        dt = time.perf_counter() - t0

        assert len(result) == dim
        print(f"\n  relu->exp({dim}) via Rust scheduler: {dt * 1000:.2f} ms")


# ---------------------------------------------------------------------------
# Test 8: Multi-node graphs with binary ops
# ---------------------------------------------------------------------------


class TestMultiNodeBinaryOps:
    def test_add_residual(self):
        """input -> relu; add(input, relu) -> output (residual connection)."""
        graph = _make_add_self_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 2.0, 3.5]
        result = aria_scheduler.execute_graph(ir_json, x)

        # relu(x) = [0, 0, 2, 3.5]; input + relu = [-1, 0, 4, 7]
        expected = [x[i] + max(0, x[i]) for i in range(len(x))]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_mul_gating(self):
        """input -> sigmoid, input -> tanh; mul(sigmoid, tanh) -> output."""
        graph = _make_mul_branches_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 1.0, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [(1.0 / (1.0 + math.exp(-v))) * math.tanh(v) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_sub_binary(self):
        """input -> relu; sub(input, relu) -> output: x - relu(x)."""
        g = ComputationGraph(model_dim=4)
        inp = g.add_input()
        relu = g.add_op("relu", [inp])
        sub = g.add_op("sub", [inp, relu])
        g.set_output(sub)

        ir_json = graph_to_native_ir_json(g)
        x = [-3.0, -1.0, 0.0, 5.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        # sub(x, relu(x)): for negative x, relu=0 so sub=x; for positive, sub=0
        expected = [v - max(0, v) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Diamond and branching topologies
# ---------------------------------------------------------------------------


class TestDiamondTopology:
    def test_diamond_add(self):
        """input -> relu -> sigmoid; input -> tanh; add(sigmoid, tanh)."""
        graph = _make_diamond_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 1.0, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [(1.0 / (1.0 + math.exp(-max(0, v)))) + math.tanh(v) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_diamond_topological_order(self):
        """Diamond graph has 5 nodes, topo order visits them correctly."""
        graph = _make_diamond_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        order = aria_scheduler.topological_order(ir_json)
        assert len(order) == 5

        # input must come first, add must come last
        assert order[0] == 0
        assert order[-1] == 4

    def test_diamond_parse(self):
        """Diamond graph with 5 nodes parses correctly."""
        graph = _make_diamond_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        result = aria_scheduler.parse_graph_ir(ir_json)
        assert "5 nodes" in result


# ---------------------------------------------------------------------------
# Test 10: Long chains (4+ ops)
# ---------------------------------------------------------------------------


class TestLongChain:
    def test_four_op_chain(self):
        """input -> relu -> sigmoid -> tanh -> exp: 4-op sequential chain."""
        graph = _make_long_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [0.5, 1.0, -0.5, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [math.exp(math.tanh(1.0 / (1.0 + math.exp(-max(0, v))))) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_four_op_chain_topological_order(self):
        """4-op chain has 5 nodes in strict linear order."""
        graph = _make_long_chain_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        order = aria_scheduler.topological_order(ir_json)
        assert list(order) == [0, 1, 2, 3, 4]

    def test_three_op_chain_sigmoid_tanh_relu(self):
        """input -> sigmoid -> tanh -> relu: 3-op chain."""
        g = ComputationGraph(model_dim=4)
        inp = g.add_input()
        sig = g.add_op("sigmoid", [inp])
        th = g.add_op("tanh", [sig])
        relu = g.add_op("relu", [th])
        g.set_output(relu)

        ir_json = graph_to_native_ir_json(g)
        x = [-2.0, 0.0, 1.0, 3.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        expected = [max(0, math.tanh(1.0 / (1.0 + math.exp(-v)))) for v in x]
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 11: Double residual (sequential binary ops)
# ---------------------------------------------------------------------------


class TestDoubleResidual:
    def test_double_residual(self):
        """input -> relu -> add(input, relu) -> sigmoid -> add(add1, sigmoid).

        Two residual connections in sequence, testing intermediate buffer reuse.
        """
        graph = _make_double_residual_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        x = [-1.0, 0.0, 1.0, 2.0]
        result = aria_scheduler.execute_graph(ir_json, x)

        # Step through:
        # relu(x) = [0, 0, 1, 2]
        # add1 = x + relu(x) = [-1, 0, 2, 4]
        # sigmoid(add1) = [sigmoid(-1), 0.5, sigmoid(2), sigmoid(4)]
        # add2 = add1 + sigmoid(add1)
        expected = []
        for v in x:
            relu_v = max(0, v)
            add1 = v + relu_v
            sig = 1.0 / (1.0 + math.exp(-add1))
            add2 = add1 + sig
            expected.append(add2)

        assert result == pytest.approx(expected, rel=1e-5)

    def test_double_residual_node_count(self):
        """Double residual has 5 nodes."""
        graph = _make_double_residual_graph(dim=4)
        ir_json = graph_to_native_ir_json(graph)

        result = aria_scheduler.parse_graph_ir(ir_json)
        assert "5 nodes" in result


# ---------------------------------------------------------------------------
# Test 12: dispatch_graph_native with multi-node graphs
# ---------------------------------------------------------------------------


class TestDispatchGraphNativeMultiNode:
    def test_dispatch_residual(self):
        """dispatch_graph_native with residual add pattern."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_add_self_graph(dim=4)
        x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = np.array([v + max(0, v) for v in x], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dispatch_diamond(self):
        """dispatch_graph_native with diamond topology."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_diamond_graph(dim=4)
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = np.array(
            [(1.0 / (1.0 + math.exp(-max(0, v)))) + math.tanh(v) for v in x],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dispatch_long_chain(self):
        """dispatch_graph_native with 4-op chain."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_long_chain_graph(dim=4)
        x = np.array([0.5, 1.0, -0.5, 2.0], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = np.array(
            [math.exp(math.tanh(1.0 / (1.0 + math.exp(-max(0, v))))) for v in x],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dispatch_double_residual(self):
        """dispatch_graph_native with double residual pattern."""
        from research.scientist.native_runner import dispatch_graph_native

        graph = _make_double_residual_graph(dim=4)
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = dispatch_graph_native(graph, x)

        expected = []
        for v in x:
            relu_v = max(0, v)
            add1 = v + relu_v
            sig = 1.0 / (1.0 + math.exp(-add1))
            add2 = add1 + sig
            expected.append(add2)
        np.testing.assert_allclose(
            result, np.array(expected, dtype=np.float32), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Test 13: Multi-node throughput benchmark
# ---------------------------------------------------------------------------


class TestMultiNodeBenchmark:
    def test_diamond_throughput(self):
        """Diamond topology on 16384 elements."""
        dim = 16384
        graph = _make_diamond_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        x = list(np.clip(np.random.randn(dim).astype(np.float32), -5.0, 5.0))

        t0 = time.perf_counter()
        result = aria_scheduler.execute_graph(ir_json, x)
        dt = time.perf_counter() - t0

        assert len(result) == dim
        print(f"\n  diamond({dim}) via Rust scheduler: {dt * 1000:.2f} ms")

    def test_double_residual_throughput(self):
        """Double residual on 16384 elements."""
        dim = 16384
        graph = _make_double_residual_graph(dim=dim)
        ir_json = graph_to_native_ir_json(graph)

        x = list(np.clip(np.random.randn(dim).astype(np.float32), -3.0, 3.0))

        t0 = time.perf_counter()
        result = aria_scheduler.execute_graph(ir_json, x)
        dt = time.perf_counter() - t0

        assert len(result) == dim
        print(f"\n  double_residual({dim}) via Rust scheduler: {dt * 1000:.2f} ms")
