from __future__ import annotations

import json

import torch

from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_graph, compile_model
from research.synthesis.reference_architectures import build_reference
from research.synthesis.ir_executor import IRExecutor
from research.synthesis.native_analysis import reset_native_analysis_bindings
from research.synthesis.validator import validate_graph
from research.tests.test_component_graphs import graph_token_routing


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


def test_compile_graph_prefers_bound_dispatcher_for_parameterized_graph(monkeypatch):
    import research.synthesis.native_compile as native_compile
    import research.synthesis.native_bound_graph as native_bound_graph
    import research.scientist.native.autograd as native_autograd

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self.model = model
            self.supported_ops = supported_ops

        def dispatch(self, op_name, *tensors, **kwargs):
            return None

        @property
        def stats(self):
            return {"native_dispatches": 0, "fallbacks": 0}

    class FakeBoundDispatcher:
        def __init__(self, graph, *, flat_ops, ir_node_ids, supported_ops):
            assert flat_ops
            assert ir_node_ids
            self.all_native = True
            self.stats = {"all_native": True, "native_backward_supported": False}

        def try_dispatch(self, x):
            return x + 7

    class FailingPlainDispatcher:
        def __init__(self, graph, supported_ops):
            raise AssertionError(
                "plain dispatcher should not be used for parameterized graphs"
            )

    monkeypatch.setattr(
        native_compile, "get_supported_native_ops", lambda graph: {"rmsnorm"}
    )
    monkeypatch.setattr(
        native_bound_graph, "BoundNativeSubgraphDispatcher", FakeBoundDispatcher
    )
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FailingPlainDispatcher)
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rmsnorm", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.zeros(1, 2, 8)
    y = layer(x)

    assert torch.equal(y, x + 7)
    assert layer._subgraph_dispatcher is not None
    assert layer._native_forward_wrapper is not None
    assert layer.ops[str(out)]._cached_native_wrapper is layer._native_forward_wrapper


def test_compile_graph_uses_plain_dispatcher_for_non_parameterized_graph(monkeypatch):
    import research.synthesis.native_compile as native_compile
    import research.synthesis.native_bound_graph as native_bound_graph
    import research.scientist.native.autograd as native_autograd

    class FakePlainDispatcher:
        def __init__(self, graph, supported_ops):
            self.all_native = True

        def try_dispatch(self, x):
            return x + 3

    class FailingBoundDispatcher:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "bound dispatcher should not be used for non-parameterized graphs"
            )

    monkeypatch.setattr(
        native_compile, "get_supported_native_ops", lambda graph: {"relu"}
    )
    monkeypatch.setattr(
        native_bound_graph, "BoundNativeSubgraphDispatcher", FailingBoundDispatcher
    )
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakePlainDispatcher)

    graph = _make_relu_graph()
    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.zeros(1, 2, 8)
    y = layer(x)

    assert torch.equal(y, x + 3)


def test_bound_dispatcher_disables_after_runtime_failure(monkeypatch):
    import research.synthesis.native_bound_graph as native_bound_graph
    import research.synthesis.native_compile as native_compile
    import research.scientist.native.autograd as native_autograd

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors, **kwargs):
            self._dispatch_count += 1
            return tensors[0] + 1

        @property
        def stats(self):
            return {
                "native_dispatches": self._dispatch_count,
                "fallbacks": self._fallback_count,
            }

    class FailingBoundDispatcher:
        def __init__(self, graph, *, flat_ops, ir_node_ids, supported_ops):
            self._all_native = True
            self._runtime_enabled = True
            self._backward_native = False
            self._dispatch_count = 0
            self._fallback_count = 0

        @property
        def all_native(self):
            return self._all_native and self._runtime_enabled

        @property
        def stats(self):
            return {
                "all_native": self._all_native,
                "runtime_enabled": self._runtime_enabled,
                "subgraph_dispatches": self._dispatch_count,
                "subgraph_fallbacks": self._fallback_count,
                "native_backward_supported": self._backward_native,
            }

        def try_dispatch(self, x):
            if not self._runtime_enabled:
                return None
            self._runtime_enabled = False
            self._fallback_count += 1
            return None

    monkeypatch.setattr(
        native_compile, "get_supported_native_ops", lambda graph: {"rmsnorm"}
    )
    monkeypatch.setattr(
        native_bound_graph, "BoundNativeSubgraphDispatcher", FailingBoundDispatcher
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rmsnorm", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.zeros(1, 2, 8)
    y1 = layer(x)
    y2 = layer(x)

    assert torch.equal(y1, x + 1)
    assert torch.equal(y2, x + 1)
    assert layer._subgraph_dispatcher.stats["subgraph_fallbacks"] == 1
    assert layer._subgraph_dispatcher.stats["runtime_enabled"] is False
    assert layer._native_forward_wrapper.stats["native_dispatches"] == 2


def test_bound_dispatcher_rmsnorm_uses_flattened_rows(monkeypatch):
    import research.synthesis.native_compile as native_compile

    monkeypatch.setattr(
        native_compile, "get_supported_native_ops", lambda graph: {"rmsnorm"}
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rmsnorm", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    dispatcher = layer._subgraph_dispatcher
    assert dispatcher is not None

    x = torch.zeros(2, 3, 8)
    plan = dispatcher._plan_for_input(x)
    nodes = json.loads(plan.ir_json)["nodes"]
    rmsnorm_node = next(node for node in nodes if node["op_name"] == "rmsnorm")

    assert rmsnorm_node["config"]["batch"] == 6
    assert rmsnorm_node["config"]["dim"] == 8


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

    assert torch.allclose(out, expected, rtol=1e-3, atol=1e-4)
    assert executor.execution_stats["native_chain_segments"] == 1
    assert executor.execution_stats["native_chain_dispatches"] == 1
    assert (
        executor.execution_stats["last_execution_path"]
        == "hybrid_native_python_ir_loop"
    )


def test_ir_executor_dispatches_bound_native_parameter_chain(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch
    import research.synthesis.native_bound_segments as native_bound_segments

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors, **kwargs):
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
            self.all_native = False

        def try_dispatch(self, x):
            return None

    def _fake_multi_input_dispatch(ir_json, inputs, *, output_shape=None):
        x, linear_weight, linear_bias, gate_weight, gate_bias = inputs
        flat = x.reshape(-1, x.shape[-1])
        gated = torch.nn.functional.linear(flat, linear_weight, linear_bias)
        gate = torch.sigmoid(torch.nn.functional.linear(flat, gate_weight, gate_bias))
        out = torch.relu(gated * gate)
        return out.reshape(output_shape)

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["gated_linear", "relu"]},
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)
    monkeypatch.setattr(
        native_bound_segments,
        "dispatch_graph_native_multi_input_cached",
        _fake_multi_input_dispatch,
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    gated = graph.add_op("gated_linear", [inp], {"out_dim": 8})
    relu = graph.add_op("relu", [gated])
    out = graph.add_op("cumsum", [relu])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": []},
    )
    python_executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    python_executor.load_state_dict(executor.state_dict(), strict=False)

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    out = executor(x)
    expected = python_executor(x)

    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5)
    assert executor.execution_stats["native_chain_segments"] == 1
    assert executor.execution_stats["native_chain_dispatches"] == 1


def test_ir_executor_skips_bound_native_parameter_chain_for_rank1_input(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch
    import research.synthesis.native_bound_segments as native_bound_segments

    class FakeWrapper:
        def __init__(self, model, supported_ops):
            self._dispatch_count = 0
            self._fallback_count = 0

        def dispatch(self, op_name, *tensors, **kwargs):
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
            self.all_native = False

        def try_dispatch(self, x):
            return None

    def _unexpected_multi_input_dispatch(ir_json, inputs, *, output_shape=None):
        raise AssertionError("bound native chain should be skipped for rank-1 input")

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["gated_linear", "relu"]},
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)
    monkeypatch.setattr(
        native_bound_segments,
        "dispatch_graph_native_multi_input_cached",
        _unexpected_multi_input_dispatch,
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    gated = graph.add_op("gated_linear", [inp], {"out_dim": 8})
    relu = graph.add_op("relu", [gated])
    out = graph.add_op("cumsum", [relu])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    dispatcher = executor._native_chain_segments[0].dispatcher

    assert executor.execution_stats["native_chain_segments"] == 1
    assert dispatcher.try_dispatch(torch.randn(8, dtype=torch.float32)) is None


def test_fingerprint_representations_preserve_rank_under_no_grad():
    model = compile_model(
        [build_reference("gpt2", d_model=64)],
        vocab_size=1000,
        max_seq_len=32,
    ).eval()
    input_ids = torch.randint(0, 1000, (8, 32))

    with torch.no_grad():
        logits, reps = model._fingerprint_representations(input_ids)

    assert logits.shape == (8, 32, 1000)
    assert reps.shape == (8, 32, 64)


def test_ir_executor_prefers_bound_native_weighted_subgraph(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch
    import research.synthesis.native_bound_graph as native_bound_graph

    def _fake_multi_input_dispatch(ir_json, inputs, *, output_shape=None):
        x, weight = inputs
        flat = x.reshape(-1, x.shape[-1])
        out = torch.nn.functional.linear(flat, weight).reshape(output_shape)
        return torch.relu(out)

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["linear_proj", "relu"]},
    )
    monkeypatch.setattr(
        native_bound_graph,
        "dispatch_graph_native_multi_input_cached",
        _fake_multi_input_dispatch,
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], {"out_dim": 8})
    out = graph.add_op("relu", [proj])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    for param in executor.parameters():
        param.requires_grad_(False)
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    y = executor(x)

    weight = executor._subgraph_dispatcher._flat_ops[
        executor._subgraph_dispatcher._node_id_to_ir_idx[proj]
    ].weight
    expected = torch.relu(torch.nn.functional.linear(x, weight))

    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-5)
    assert executor.execution_stats["last_execution_path"] == "native_subgraph"
    assert executor.execution_stats["native_subgraph_dispatches"] == 1


def test_bound_native_dispatch_skips_host_bridge_for_non_cpu_tensors(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch
    import research.synthesis.native_bound_graph as native_bound_graph

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["linear_proj", "relu"]},
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], {"out_dim": 8})
    out = graph.add_op("relu", [proj])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    dispatcher = executor._subgraph_dispatcher
    assert dispatcher is not None

    monkeypatch.setattr(
        native_bound_graph,
        "supports_host_array_bridge",
        lambda *values: False,
    )
    monkeypatch.setattr(
        native_bound_graph,
        "dispatch_graph_native_multi_input_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host bridge should be skipped")
        ),
    )

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    result = dispatcher.try_dispatch(x)

    assert result is None
    assert dispatcher.stats["subgraph_dispatches"] == 0
    assert (
        dispatcher.stats["last_refusal_reason"]
        == "host_array_bridge_unsupported_device"
    )


def test_ir_executor_dispatches_single_input_native_subgraphs(monkeypatch):
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
            values = {}
            for node_id in self.graph.topological_order():
                node = self.graph.nodes[node_id]
                if node.is_input:
                    values[node_id] = x
                elif node.op_name == "relu":
                    values[node_id] = torch.relu(values[node.input_ids[0]])
                elif node.op_name == "gelu":
                    values[node_id] = torch.nn.functional.gelu(
                        values[node.input_ids[0]]
                    )
                elif node.op_name == "add":
                    values[node_id] = (
                        values[node.input_ids[0]] + values[node.input_ids[1]]
                    )
                else:
                    return None
            return values[self.graph.output_node.id]

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["relu", "gelu", "add"]},
    )
    monkeypatch.setattr(native_autograd, "NativeForwardWrapper", FakeWrapper)
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    relu = graph.add_op("relu", [inp])
    gelu = graph.add_op("gelu", [inp])
    add = graph.add_op("add", [relu, gelu])
    out = graph.add_op("cumsum", [add])
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

    assert torch.allclose(out, expected, rtol=1e-3, atol=1e-4)
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


def test_native_wrapper_softmax_attention_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("softmax_attention", {}, shape, shape, 8)
    python_op = CompiledOp("softmax_attention", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-6, rtol=1e-6)


def test_native_wrapper_selective_scan_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("selective_scan", {}, shape, shape, 8)
    python_op = CompiledOp("selective_scan", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_wrapper_state_space_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("state_space", {}, shape, shape, 8)
    python_op = CompiledOp("state_space", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_wrapper_gated_delta_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=16)
    native_op = CompiledOp("gated_delta", {}, shape, shape, 16)
    python_op = CompiledOp("gated_delta", {}, shape, shape, 16)
    python_op.load_state_dict(native_op.state_dict())
    native_op._gated_delta_heads = 4
    python_op._gated_delta_heads = 4
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 37, 16)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_compile_graph_uses_bound_native_subgraph_for_weighted_chain_without_prewarm():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("selective_scan", [n2])
    n4 = graph.add_op("state_space", [n3])
    n5 = graph.add_op("gated_delta", [n4])
    graph.set_output(n5)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 2, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_weighted_chain():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("selective_scan", [n2])
    n4 = graph.add_op("state_space", [n3])
    n5 = graph.add_op("gated_delta", [n4])
    graph.set_output(n5)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 2, 8, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n3)].A_log.grad is not None
    assert layer.ops[str(n4)].ssm_A.grad is not None
    assert layer.ops[str(n5)].q_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_rwkv_time_mixing_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rwkv_time_mixing", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_rwkv_time_mixing():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rwkv_time_mixing", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    op = layer.ops[str(out)]
    assert op.w_decay.grad is not None
    assert op.u_bonus.grad is not None
    assert op.W_k.grad is not None
    assert op.W_v.grad is not None
    assert op.W_r.grad is not None
    assert op.W_o.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_transformer_standard():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("linear_proj", [n4], {"out_dim": 32})
    n6 = graph.add_op("gelu", [n5])
    n7 = graph.add_op("linear_proj", [n6], {"out_dim": 8})
    out = graph.add_op("add", [n3, n7])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_transformer_standard():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("linear_proj", [n4], {"out_dim": 32})
    n6 = graph.add_op("gelu", [n5])
    n7 = graph.add_op("linear_proj", [n6], {"out_dim": 8})
    out = graph.add_op("add", [n3, n7])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n1)].bias.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n4)].weight.grad is not None
    assert layer.ops[str(n4)].bias.grad is not None
    assert layer.ops[str(n5)].weight.grad is not None
    assert layer.ops[str(n7)].weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_ssm_rwkv_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("rwkv_time_mixing", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("rwkv_channel", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_rwkv():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("rwkv_time_mixing", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("rwkv_channel", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n1)].bias.grad is not None
    assert layer.ops[str(n2)].W_o.grad is not None
    assert layer.ops[str(n5)].mix_k.grad is not None
    assert layer.ops[str(n5)].mix_r.grad is not None
    assert layer.ops[str(n5)].key_proj.weight.grad is not None
    assert layer.ops[str(n5)].receptance_proj.weight.grad is not None
    assert layer.ops[str(n5)].value_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_swiglu_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("swiglu_mlp", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_swiglu():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("swiglu_mlp", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n4)].weight.grad is not None
    assert layer.ops[str(n5)].gate_proj.weight.grad is not None
    assert layer.ops[str(n5)].up_proj.weight.grad is not None
    assert layer.ops[str(n5)].down_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_linear_attention_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("linear_attention", [n1])
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_uses_bound_native_subgraph_for_adaptive_recursion_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("adaptive_recursion", [n1], {"max_depth": 3})
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_falls_back_cleanly_for_adaptive_recursion_backward():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("adaptive_recursion", [n1], {"max_depth": 3})
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert x.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 0
    assert stats["subgraph_fallbacks"] == 1
    assert stats["native_backward_supported"] is False
    assert stats["last_refusal_reason"] == "native_backward_unavailable"


def test_compile_graph_uses_bound_native_subgraph_for_token_routing_inference():
    graph, _, _ = graph_token_routing()
    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, graph.model_dim))

    assert y.shape == (1, 3, graph.model_dim)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_uses_bound_native_subgraph_for_ssm_state_space_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("state_space", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv_only", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_state_space():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("state_space", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv_only", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n2)].ssm_A.grad is not None
    assert layer.ops[str(n5)].conv_dw.weight.grad is not None
    assert layer.ops[str(n5)].conv_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_ir_executor_skips_plain_subgraph_dispatch_for_mixed_parameterized_graph():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("split2", [n1], {"n_splits": 2})
    n3 = graph.add_op("split2", [n1], {"n_splits": 2})
    n4 = graph.add_op("gelu", [n2])
    n5 = graph.add_op("tanh", [n3])
    n6 = graph.add_op("concat", [n4, n5])
    out = graph.add_op("add", [inp, n6])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert isinstance(layer, IRExecutor)
    assert getattr(layer, "_subgraph_dispatcher", None) is None

    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer.execution_stats["native_subgraph_available"] is False
    assert layer.execution_stats["native_setup_reason"] in {
        "partial_native_segments",
        "per_op_native_wrapper",
    }


def test_compile_graph_uses_bound_native_subgraph_for_ssm_mamba_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("conv1d_seq", [n1])
    n3 = graph.add_op("silu", [n2])
    n4 = graph.add_op("selective_scan", [n3])
    n5 = graph.add_op("gated_linear", [n4], {"out_dim": 8})
    out = graph.add_op("add", [inp, n5])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_mamba():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("conv1d_seq", [n1])
    n3 = graph.add_op("silu", [n2])
    n4 = graph.add_op("selective_scan", [n3])
    n5 = graph.add_op("gated_linear", [n4], {"out_dim": 8})
    out = graph.add_op("add", [inp, n5])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].conv_weight.grad is not None
    assert layer.ops[str(n4)].A_log.grad is not None
    assert layer.ops[str(n5)].linear_weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_hybrid_ssm_attention_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv1d_seq", [n4])
    n6 = graph.add_op("selective_scan", [n5])
    n7 = graph.add_op("add", [n3, n6])
    n8 = graph.add_op("rmsnorm", [n7])
    n9 = graph.add_op("swiglu_mlp", [n8])
    out = graph.add_op("add", [n7, n9])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_hybrid_ssm_attention():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv1d_seq", [n4])
    n6 = graph.add_op("selective_scan", [n5])
    n7 = graph.add_op("add", [n3, n6])
    n8 = graph.add_op("rmsnorm", [n7])
    n9 = graph.add_op("swiglu_mlp", [n8])
    out = graph.add_op("add", [n7, n9])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n5)].conv_weight.grad is not None
    assert layer.ops[str(n6)].A_log.grad is not None
    assert layer.ops[str(n9)].gate_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True
