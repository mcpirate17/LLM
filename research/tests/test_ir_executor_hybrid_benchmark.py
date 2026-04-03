from __future__ import annotations

import statistics
import time

import torch

from research.synthesis.graph import ComputationGraph
from research.synthesis.ir_executor import IRExecutor


def _make_mixed_chain_graph(model_dim: int = 256) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    n1 = graph.add_op("relu", [inp])
    n2 = graph.add_op("sigmoid", [n1])
    n3 = graph.add_op("tanh", [n2])
    n4 = graph.add_op("exp", [n3])
    out = graph.add_op("cumsum", [n4])
    graph.set_output(out)
    return graph


def _median_ms(fn, *, warmup: int = 40, iterations: int = 200) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1_000_000.0)
    return statistics.median(samples)


def _build_executors(
    graph: ComputationGraph,
) -> tuple[IRExecutor, IRExecutor, IRExecutor]:
    ir = graph.lower_to_ir()

    python_only = IRExecutor(ir, source_graph=None)

    hybrid_per_op = IRExecutor(ir, source_graph=graph)
    hybrid_per_op._native_chain_segments = ()
    hybrid_per_op._native_chain_segments_by_plan_index = {}

    hybrid_chain = IRExecutor(ir, source_graph=graph)
    return python_only, hybrid_per_op, hybrid_chain


def test_ir_executor_hybrid_benchmark_smoke(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors):
            x = tensors[0]
            if op_name == "relu":
                self._dispatch_count += 1
                return torch.relu(x)
            if op_name == "sigmoid":
                self._dispatch_count += 1
                return torch.sigmoid(x)
            if op_name == "tanh":
                self._dispatch_count += 1
                return torch.tanh(x)
            if op_name == "exp":
                self._dispatch_count += 1
                return torch.exp(x)
            self._fallback_count += 1
            return None

        @property
        def stats(self):
            return {
                "native_dispatches": self._dispatch_count,
                "fallbacks": self._fallback_count,
            }

    class FakeDispatcher:
        def __init__(self, graph, supported_ops):
            self.graph = graph
            self.supported_ops = supported_ops
            op_names = [
                node.op_name for node in graph.nodes.values() if not node.is_input
            ]
            self.all_native = all(op_name in supported_ops for op_name in op_names)

        def try_dispatch(self, x):
            out = x
            for node_id in self.graph.topological_order():
                node = self.graph.nodes[node_id]
                if node.is_input:
                    continue
                if node.op_name == "relu":
                    out = torch.relu(out)
                elif node.op_name == "sigmoid":
                    out = torch.sigmoid(out)
                elif node.op_name == "tanh":
                    out = torch.tanh(out)
                elif node.op_name == "exp":
                    out = torch.exp(out)
                else:
                    return None
            return out

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {
            "supported": ["relu", "sigmoid", "tanh", "exp"]
        },
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)

    graph = _make_mixed_chain_graph()
    python_only, hybrid_per_op, hybrid_chain = _build_executors(graph)

    x = torch.randn(8, 128, graph.model_dim, dtype=torch.float32)

    expected = python_only(x)
    per_op_out = hybrid_per_op(x)
    chain_out = hybrid_chain(x)
    assert torch.allclose(per_op_out, expected, rtol=1e-4, atol=1e-5)
    assert torch.allclose(chain_out, expected, rtol=1e-4, atol=1e-5)

    python_ms = _median_ms(lambda: python_only(x))
    per_op_ms = _median_ms(lambda: hybrid_per_op(x))
    chain_ms = _median_ms(lambda: hybrid_chain(x))

    print(
        "\nIRExecutor hybrid benchmark"
        f"\n  python_only_ms={python_ms:.4f}"
        f"\n  hybrid_per_op_ms={per_op_ms:.4f}"
        f"\n  hybrid_chain_ms={chain_ms:.4f}"
        f"\n  per_op_speedup={python_ms / per_op_ms:.3f}x"
        f"\n  chain_speedup={python_ms / chain_ms:.3f}x"
        f"\n  chain_vs_per_op={per_op_ms / chain_ms:.3f}x"
    )

    assert python_ms > 0.0
    assert per_op_ms > 0.0
    assert chain_ms > 0.0
