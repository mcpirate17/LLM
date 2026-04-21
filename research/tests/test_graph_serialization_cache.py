from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aria_designer.api.app.workflow_graph_cache import materialize_workflow_graph
from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import graph_to_native_ir_json
from research.synthesis.native_topology import _build_canonical_topology_inputs
from research.synthesis.serializer import graph_from_json, graph_to_json


def _build_graph() -> ComputationGraph:
    graph = ComputationGraph(64)
    inp = graph.add_input()
    node = inp
    for _ in range(4):
        node = graph.add_op("linear_proj", [node], {"out_dim": 64})
        node = graph.add_op("relu", [node], {})
    graph.set_output(node)
    return graph


def _load_workflow_fixture() -> dict:
    fixture = (
        Path(__file__).resolve().parents[2]
        / "aria_designer"
        / "ui"
        / "public"
        / "examples"
        / "transformer_mini.json"
    )
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_graph_to_json_is_cached(monkeypatch):
    graph = _build_graph()
    import research.synthesis.serializer as serializer_mod

    calls = {"n": 0}
    original = serializer_mod.dumps_json

    def counted(payload):
        calls["n"] += 1
        return original(payload)

    monkeypatch.setattr(serializer_mod, "dumps_json", counted)

    first = graph_to_json(graph)
    second = graph_to_json(graph)

    assert first == second
    assert calls["n"] == 1


def test_graph_to_native_ir_json_is_cached(monkeypatch):
    graph = _build_graph()
    import research.synthesis.native_ir_converter as native_ir_mod

    calls = {"n": 0}
    original = native_ir_mod.dumps_json

    def counted(payload):
        calls["n"] += 1
        return original(payload)

    monkeypatch.setattr(native_ir_mod, "dumps_json", counted)

    first = graph_to_native_ir_json(graph)
    second = graph_to_native_ir_json(graph)

    assert first == second
    assert calls["n"] == 1


def test_graph_from_json_uses_cached_parse_and_returns_copy(monkeypatch):
    graph = _build_graph()
    payload = graph_to_json(graph)
    import research.synthesis.serializer as serializer_mod

    calls = {"n": 0}
    original = serializer_mod.loads_json

    def counted(raw):
        calls["n"] += 1
        return original(raw)

    monkeypatch.setattr(serializer_mod, "loads_json", counted)

    first = graph_from_json(payload)
    second = graph_from_json(payload, model_dim=128)
    second.metadata["changed"] = True

    assert calls["n"] == 1
    assert first is not second
    assert first.model_dim == 64
    assert second.model_dim == 128
    assert "changed" not in first.metadata


def test_canonical_topology_inputs_are_cached_and_invalidated():
    graph = _build_graph()
    first = _build_canonical_topology_inputs(graph)
    second = _build_canonical_topology_inputs(graph)

    assert first is second

    graph.add_op("relu", [graph.output_node.id], {})
    refreshed = _build_canonical_topology_inputs(graph)
    assert refreshed is not first


def test_materialize_workflow_graph_reuses_same_workflow_object():
    workflow = _load_workflow_fixture()
    graph_a, id_map_a = materialize_workflow_graph(
        workflow,
        128,
        return_id_map=True,
    )
    graph_b, id_map_b = materialize_workflow_graph(
        workflow,
        128,
        return_id_map=True,
    )

    assert graph_a is graph_b
    assert id_map_a == id_map_b
    assert id_map_a is not id_map_b


def test_dispatch_forward_saved_uses_supplied_ir_json(monkeypatch):
    from research.scientist.native.dispatch import dispatch_graph_forward_native_saved

    graph = _build_graph()
    fake_output = np.ones((2, 4, 64), dtype=np.float32)
    captured = {}

    def fake_prepare(target_graph, input_data, *, ir_json=None):
        captured["graph"] = target_graph
        captured["ir_json"] = ir_json
        return np.asarray(input_data, dtype=np.float32), ir_json or "generated"

    def fake_execute(*, graph_json, x_np, graph):
        captured["graph_json"] = graph_json
        return {
            "output": fake_output,
            "saved_activations": {0: fake_output.reshape(-1)},
            "ir_json": graph_json,
        }

    monkeypatch.setattr(
        "research.scientist.native.dispatch._prepare_graph_input",
        fake_prepare,
    )
    monkeypatch.setattr(
        "research.scientist.native.dispatch._execute_rust_graph_forward_saved",
        fake_execute,
    )

    result = dispatch_graph_forward_native_saved(
        graph,
        fake_output,
        ir_json="cached-ir-json",
    )

    assert captured["graph"] is graph
    assert captured["ir_json"] == "cached-ir-json"
    assert captured["graph_json"] == "cached-ir-json"
    assert result["ir_json"] == "cached-ir-json"
