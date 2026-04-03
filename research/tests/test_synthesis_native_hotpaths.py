from __future__ import annotations

import torch

from research.synthesis.graph import ComputationGraph
from research.synthesis.ir_executor import IRExecutor
from research.synthesis.native_analysis import reset_native_analysis_bindings
from research.synthesis.validator import validate_graph


def _make_relu_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)
    return graph


def _make_chain_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    relu = graph.add_op("relu", [inp])
    out = graph.add_op("exp", [relu])
    graph.set_output(out)
    return graph


def test_structural_analysis_uses_native_runtime_when_available():
    graph = _make_relu_graph()
    dead = graph.add_op("relu", [graph.input_node.id])
    assert dead in graph.nodes

    reset_native_analysis_bindings()
    analysis = graph._analysis_ir().analyze_structure(include_reachable=True)

    assert analysis.backend in {"native", "aria_core"}
    assert analysis.has_gradient_path is True
    assert analysis.reachable_count == 2
    assert analysis.depth == 1


def test_structural_analysis_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(native_analysis, "_try_import_aria_core", lambda: None)
    reset_native_analysis_bindings()

    graph = _make_relu_graph()
    analysis = graph._analysis_ir().analyze_structure(include_reachable=True)

    assert analysis.backend == "python"
    assert analysis.has_gradient_path is True
    assert analysis.reachable_count == 2
    assert analysis.depth == 1


def test_ir_executor_prefers_native_subgraph_dispatch(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch

    class FakeDispatcher:
        def __init__(self, graph, supported_ops):
            self.graph = graph
            self.supported_ops = supported_ops
            self.all_native = True

        def try_dispatch(self, x):
            return x + 5

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["relu"]},
    )
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)

    graph = _make_relu_graph()
    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    x = torch.zeros(1, 3, 8)
    out = executor(x)

    assert torch.equal(out, x + 5)
    assert executor.execution_stats["last_execution_path"] == "native_subgraph"
    assert executor.execution_stats["native_subgraph_dispatches"] == 1


def test_ir_executor_falls_back_to_python_loop(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": []},
    )

    graph = _make_relu_graph()
    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    x = torch.tensor([[[-1.0] * 8, [2.0] * 8]], dtype=torch.float32)
    out = executor(x)

    assert torch.equal(out, torch.relu(x))
    assert executor.execution_stats["last_execution_path"] == "python_ir_loop"
    assert executor.execution_stats["python_ir_loop_fallbacks"] == 1
    assert executor.execution_stats["native_subgraph_available"] is False


def test_ir_executor_uses_partial_native_wrapper_for_mixed_graph(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self.model = model
            self.supported_ops = supported_ops
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors):
            if op_name == "relu":
                self._dispatch_count += 1
                return tensors[0] + 1
            self._fallback_count += 1
            return None

        @property
        def stats(self):
            return {
                "native_dispatches": self._dispatch_count,
                "fallbacks": self._fallback_count,
            }

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["relu"]},
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    relu = graph.add_op("relu", [inp])
    out = graph.add_op("sigmoid", [relu])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    x = torch.zeros(1, 2, 8)
    out = executor(x)

    assert torch.allclose(out, torch.sigmoid(x + 1))
    assert (
        executor.execution_stats["last_execution_path"]
        == "hybrid_native_python_ir_loop"
    )
    assert executor.execution_stats["partial_native_available"] is True
    assert executor.execution_stats["partial_native_dispatches"] == 1
    assert executor.execution_stats["hybrid_native_python_ir_loops"] == 1


def test_ir_executor_dispatches_native_unary_chain_segments(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors):
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
                else:
                    return None
            return out

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["relu", "sigmoid"]},
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    relu = graph.add_op("relu", [inp])
    sig = graph.add_op("sigmoid", [relu])
    out = graph.add_op("cumsum", [sig])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": []},
    )
    python_executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    x = torch.tensor([[[-1.0] * 8, [2.0] * 8]], dtype=torch.float32)
    out = executor(x)
    expected = python_executor(x)

    assert torch.allclose(out, expected)
    assert executor.execution_stats["native_chain_segments"] == 1
    assert executor.execution_stats["native_chain_dispatches"] == 1
    assert (
        executor.execution_stats["last_execution_path"]
        == "hybrid_native_python_ir_loop"
    )


def test_topological_order_uses_rust_scheduler_when_aria_core_unavailable(
    monkeypatch,
):
    import research.synthesis.native_topology as native_topology

    calls = {}

    class FakeRustScheduler:
        @staticmethod
        def topological_order(ir_json):
            calls["ir_json"] = ir_json
            return [0, 1, 2]

    monkeypatch.setattr(native_topology, "_try_import_aria_core", lambda: None)
    monkeypatch.setattr(
        native_topology, "_try_import_rust_scheduler", lambda: FakeRustScheduler()
    )
    monkeypatch.setattr(
        native_topology,
        "_graph_to_native_ir_json",
        lambda graph: "fake-native-ir",
    )

    graph = _make_chain_graph()

    assert graph.topological_order() == [0, 1, 2]
    assert calls["ir_json"] == "fake-native-ir"


def test_topological_order_falls_back_to_python_when_native_unavailable(monkeypatch):
    import research.synthesis.native_topology as native_topology

    calls = {"python": 0}
    original_python_topology = native_topology._python_topological_order

    def _tracked_python_topology(graph):
        calls["python"] += 1
        return original_python_topology(graph)

    monkeypatch.setattr(native_topology, "_try_import_aria_core", lambda: None)
    monkeypatch.setattr(native_topology, "_try_import_rust_scheduler", lambda: None)
    monkeypatch.setattr(
        native_topology, "_python_topological_order", _tracked_python_topology
    )

    graph = _make_chain_graph()

    assert graph.topological_order() == [0, 1, 2]
    assert calls["python"] == 1


def test_validate_graph_uses_native_validation_summary_when_available():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    a = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    b = graph.add_op("linear_proj", [a], config={"out_dim": 8})
    c = graph.add_op("linear_proj", [b], config={"out_dim": 8})
    d = graph.add_op("linear_proj", [c], config={"out_dim": 8})
    graph.set_output(d)

    reset_native_analysis_bindings()
    result = validate_graph(graph)

    assert result.n_parameterized_ops == 4
    assert any("projection chain" in warning.lower() for warning in result.warnings)


def test_validate_graph_validation_summary_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis
    import research.synthesis.native_validation as native_validation

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(
        native_validation, "_load_native_graph_analysis_lib", lambda: None
    )
    reset_native_analysis_bindings()

    graph = ComputationGraph(8)
    inp = graph.add_input()
    a = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    b = graph.add_op("linear_proj", [a], config={"out_dim": 8})
    c = graph.add_op("linear_proj", [b], config={"out_dim": 8})
    d = graph.add_op("linear_proj", [c], config={"out_dim": 8})
    graph.set_output(d)

    result = validate_graph(graph)

    assert result.n_parameterized_ops == 4
    assert any("projection chain" in warning.lower() for warning in result.warnings)


def test_validate_graph_reuses_shared_structural_analysis(monkeypatch):
    class FakeAnalysis:
        has_gradient_path = True
        reachable_count = 2
        depth = 1
        has_cycle = False
        param_estimate = 0

    class FakeIR:
        def analyze_structure(self, *, include_reachable=False):
            assert include_reachable is True
            return FakeAnalysis()

    graph = _make_relu_graph()
    monkeypatch.setattr(graph, "_analysis_ir", lambda: FakeIR())
    monkeypatch.setattr(
        graph,
        "depth",
        lambda: (_ for _ in ()).throw(AssertionError("depth() should not be called")),
    )
    monkeypatch.setattr(
        graph,
        "n_params_estimate",
        lambda: (_ for _ in ()).throw(
            AssertionError("n_params_estimate() should not be called")
        ),
    )
    monkeypatch.setattr(
        graph,
        "get_reachable_nodes",
        lambda: (_ for _ in ()).throw(
            AssertionError("get_reachable_nodes() should not be called")
        ),
    )
    monkeypatch.setattr(
        graph,
        "has_gradient_path",
        lambda: (_ for _ in ()).throw(
            AssertionError("has_gradient_path() should not be called")
        ),
    )

    result = validate_graph(graph)

    assert result.valid is True
    assert result.depth == 1
    assert result.has_gradient_path is True


def test_dead_parameterized_mask_uses_native_runtime(monkeypatch):
    import numpy as np

    import research.synthesis.native_dim_flow as native_dim_flow

    calls = {}

    class FakeLib:
        def aria_graph_dead_parameterized_mask(
            self,
            n_nodes,
            reachable_ptr,
            parameterized_ptr,
            dead_ptr,
        ):
            reachable = np.ctypeslib.as_array(reachable_ptr, shape=(n_nodes,))
            parameterized = np.ctypeslib.as_array(parameterized_ptr, shape=(n_nodes,))
            dead = np.ctypeslib.as_array(dead_ptr, shape=(n_nodes,))
            dead[:] = ((reachable == 0) & (parameterized != 0)).astype(np.int32)
            calls["n_nodes"] = n_nodes
            return 0

    monkeypatch.setattr(
        native_dim_flow,
        "_load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    result = native_dim_flow.dead_parameterized_mask(
        reachable_mask=np.array([1, 0, 0, 1], dtype=np.int32),
        parameterized_flags=np.array([0, 1, 0, 1], dtype=np.int32),
    )

    assert calls["n_nodes"] == 4
    assert result.backend == "native"
    assert result.mask.tolist() == [False, True, False, False]
