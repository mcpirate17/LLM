from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from research.synthesis.compiler import compile_graph, compile_model
from research.synthesis.grammar import GrammarConfig, _validate_graph
from research.synthesis.graph import ComputationGraph, ComputationGraphIR
from research.synthesis.graph_features import extract_graph_features
from research.synthesis.native_compile import get_supported_native_ops
from research.synthesis.workflow_converter import workflow_to_computation_graph


def test_graph_features_use_canonical_primitive_names():
    graph_json = {
        "model_dim": 64,
        "nodes": {
            "0": {"id": 0, "op_name": "input", "input_ids": []},
            "1": {"id": 1, "op_name": "route_lanes", "input_ids": [0]},
            "2": {"id": 2, "op_name": "rope_rotate", "input_ids": [1]},
            "3": {"id": 3, "op_name": "softmax_attention", "input_ids": [2]},
            "4": {"id": 4, "op_name": "layernorm", "input_ids": [3]},
        },
        "metadata": {},
    }

    features = extract_graph_features(graph_json)

    assert features["op_gated_lane_blend"] == 1.0
    assert features["op_layernorm"] == 1.0
    assert features["has_rope"] == 1.0
    assert features["has_attention"] == 1.0
    assert features["has_norm"] == 1.0
    assert features["cat_parameterized"] == 2.0
    assert features["cat_mixing"] == 1.0
    assert features["cat_functional"] == 1.0
    assert "op_route_lanes" not in features


def test_validate_graph_rejects_too_shallow_ops_without_mutation():
    graph = ComputationGraph(64)
    inp = graph.add_input()
    gated = graph.add_op("gated_lane_blend", [inp], {"n_lanes": 3})
    graph.set_output(gated)

    with pytest.raises(ValueError, match="min_layer_depth=2"):
        _validate_graph(graph, GrammarConfig(model_dim=64))

    assert graph.nodes[gated].op_name == "gated_lane_blend"


def test_ir_gradient_path_uses_sparse_traversal():
    ir = ComputationGraphIR(
        model_dim=64,
        op_codes=np.array([0, 1, 2], dtype=np.int32),
        input_indices=np.array([[-1, -1], [0, -1], [1, -1]], dtype=np.int32),
        output_node_idx=2,
        configs=[{}, {}, {}],
    )
    assert ir.has_gradient_path() is True


def test_workflow_converter_rejects_unwired_output_node():
    workflow = {
        "nodes": [
            {"id": "in", "component_type": "io/input"},
            {"id": "out", "component_type": "io/output_head"},
        ],
        "edges": [],
        "metadata": {"model_dim": 64},
    }

    with pytest.raises(ValueError, match="has no incoming edge"):
        workflow_to_computation_graph(workflow)


def test_compile_graph_prefers_native_subgraph_dispatch(monkeypatch):
    import research.synthesis.ir_executor as ir_executor_mod

    class FakeIRExecutor(nn.Module):
        def __init__(self, ir, source_graph=None):
            super().__init__()
            self.ir = ir
            self.source_graph = source_graph

        def forward(self, x):
            return x

    monkeypatch.setattr(ir_executor_mod, "IRExecutor", FakeIRExecutor)

    graph = ComputationGraph(32)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)

    module = compile_graph(graph, use_ir=True)

    assert isinstance(module, FakeIRExecutor)
    assert module.source_graph is graph


def test_compile_graph_falls_back_to_ir_executor_when_native_unavailable(monkeypatch):
    import research.synthesis.ir_executor as ir_executor_mod

    class FakeIRExecutor(nn.Module):
        def __init__(self, ir, source_graph=None):
            super().__init__()
            self.ir = ir
            self.source_graph = source_graph

    monkeypatch.setattr(ir_executor_mod, "IRExecutor", FakeIRExecutor)

    graph = ComputationGraph(32)
    inp = graph.add_input()
    out = graph.add_op("topk_gate", [inp], {"k": 1})
    graph.set_output(out)

    module = compile_graph(graph, use_ir=True)

    assert isinstance(module, FakeIRExecutor)


def test_compile_model_uses_fast_path_selection_per_layer(monkeypatch):
    import research.synthesis.compiler as compiler_mod

    class MarkerModule(nn.Module):
        def __init__(self, graph):
            super().__init__()
            self.graph = graph

        def forward(self, x):
            return x

    monkeypatch.setattr(
        compiler_mod,
        "_compile_layer_module",
        lambda graph, prefer_fast_path: MarkerModule(graph),
    )

    graph = ComputationGraph(16)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)

    model = compile_model([graph], vocab_size=32, max_seq_len=8, use_ir=True)

    assert isinstance(model.layers[0], MarkerModule)


def test_compile_graph_attaches_native_subgraph_dispatcher(monkeypatch):
    import research.scientist.native.autograd as native_autograd
    import research.scientist.native.dispatch as native_dispatch
    from research.synthesis.ir_executor import IRExecutor

    class FakeDispatcher:
        def __init__(self, graph, supported_ops):
            self.graph = graph
            self.supported_ops = supported_ops
            self.all_native = True

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["relu"]},
    )
    monkeypatch.setattr(native_autograd, "SubgraphDispatcher", FakeDispatcher)

    graph = ComputationGraph(32)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)

    module = compile_graph(graph, use_ir=True)

    assert isinstance(module, IRExecutor)
    assert isinstance(module._subgraph_dispatcher, FakeDispatcher)


def test_get_supported_native_ops_caches_probe_result(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    calls = {"count": 0}

    def _fake_check(graphs, native_lib=None):
        calls["count"] += 1
        return {"supported": ["relu"]}

    monkeypatch.setattr(native_dispatch, "_check_native_op_support", _fake_check)

    graph = ComputationGraph(32)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)

    assert get_supported_native_ops(graph) == {"relu"}
    assert get_supported_native_ops(graph) == {"relu"}
    assert calls["count"] == 1


def test_graph_uses_native_analysis_when_available(monkeypatch):
    import research.synthesis.native_analysis as native_analysis

    graph = ComputationGraph(16)
    inp = graph.add_input()
    mid = graph.add_op("relu", [inp])
    out = graph.add_op("gelu", [mid])
    graph.set_output(out)

    class FakeAriaCore:
        @staticmethod
        def analyze_graph(n_nodes, edges, op_codes, output_idx, input_idx):
            return {
                "valid": True,
                "reachable_nodes": [0, 1, 2],
                "max_depth": 7,
                "has_input_path": False,
            }

    native_analysis.reset_native_analysis_bindings()
    monkeypatch.setattr(
        native_analysis, "_try_import_aria_core", lambda: FakeAriaCore()
    )

    assert graph.get_reachable_nodes() == {inp, mid, out}
    assert graph.depth() == 7
    assert graph.has_gradient_path() is False
