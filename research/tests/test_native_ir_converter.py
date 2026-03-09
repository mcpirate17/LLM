"""Tests for the ComputationGraph -> native_ir.v1 converter."""

from __future__ import annotations

import json

import pytest

from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import (
    graph_to_native_ir,
    graph_to_native_ir_json,
)

pytestmark = pytest.mark.native


def _make_simple_graph(model_dim: int = 64) -> ComputationGraph:
    """input -> relu -> output (3 nodes: input, relu, identity-as-output)."""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    r = g.add_op("relu", [inp])
    g.set_output(r)
    return g


def _make_multi_input_graph(model_dim: int = 64) -> ComputationGraph:
    """input -> (relu, gelu) -> add -> output."""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    r = g.add_op("relu", [inp])
    s = g.add_op("gelu", [inp])
    a = g.add_op("add", [r, s])
    g.set_output(a)
    return g


# ── Structural conversion tests ──────────────────────────────────────


def test_simple_graph_converts_to_valid_ir():
    """A simple graph should produce a well-formed native_ir.v1 document."""
    g = _make_simple_graph()
    ir = graph_to_native_ir(g)

    assert ir["schema_version"] == "native_ir.v1"
    assert ir["model_dim"] == 64
    assert isinstance(ir["nodes"], list)
    assert isinstance(ir["edges"], list)
    assert ir["output_node_id"] == g._output_node_id


def test_converter_produces_correct_edges():
    """Edges should be derived from input_ids of each node."""
    g = _make_simple_graph()
    ir = graph_to_native_ir(g)

    # relu (node 1) has input_ids=[0], so there's one edge: 0->1
    edges = ir["edges"]
    assert len(edges) == 1
    assert edges[0] == {"source": 0, "target": 1}


def test_converter_strips_output_shape():
    """IR nodes must NOT contain output_shape (schema uses additionalProperties: false)."""
    g = _make_simple_graph()
    ir = graph_to_native_ir(g)

    for node in ir["nodes"]:
        assert "output_shape" not in node


def test_converter_roundtrip_node_count():
    """Number of IR nodes should match number of graph nodes."""
    g = _make_multi_input_graph()
    ir = graph_to_native_ir(g)

    assert len(ir["nodes"]) == len(g.nodes)


def test_multi_input_graph():
    """Graph with a binary op (add with 2 inputs) should produce correct edges."""
    g = _make_multi_input_graph()
    ir = graph_to_native_ir(g)

    # Nodes: 0=input, 1=relu, 2=gelu, 3=add
    assert len(ir["nodes"]) == 4

    # Edges: relu<-input, gelu<-input, add<-relu, add<-gelu
    edge_pairs = {(e["source"], e["target"]) for e in ir["edges"]}
    assert (0, 1) in edge_pairs  # input -> relu
    assert (0, 2) in edge_pairs  # input -> gelu
    assert (1, 3) in edge_pairs  # relu -> add
    assert (2, 3) in edge_pairs  # gelu -> add
    assert len(edge_pairs) == 4


def test_nodes_sorted_by_id():
    """IR nodes should be sorted by node ID."""
    g = _make_multi_input_graph()
    ir = graph_to_native_ir(g)

    ids = [n["id"] for n in ir["nodes"]]
    assert ids == sorted(ids)


def test_is_input_and_is_output_always_present():
    """is_input and is_output should always be present on every node."""
    g = _make_simple_graph()
    ir = graph_to_native_ir(g)

    for node in ir["nodes"]:
        assert "is_input" in node
        assert "is_output" in node
        assert isinstance(node["is_input"], bool)
        assert isinstance(node["is_output"], bool)


def test_json_serialization_roundtrips():
    """graph_to_native_ir_json should produce valid JSON that matches the dict."""
    g = _make_simple_graph()
    ir_dict = graph_to_native_ir(g)
    ir_json = graph_to_native_ir_json(g)

    parsed = json.loads(ir_json)
    assert parsed == ir_dict


# ── Schema validation tests ──────────────────────────────────────────


def test_ir_validator_accepts_converted_graph():
    """validate_ir should return no errors for a correctly converted graph."""
    from research.runtime.native.ir_validator import validate_ir

    g = _make_simple_graph()
    ir = graph_to_native_ir(g)
    errors = validate_ir(ir)

    assert errors == [], f"Unexpected validation errors: {errors}"


def test_ir_validator_accepts_multi_input_graph():
    """validate_ir should accept a graph with binary ops."""
    from research.runtime.native.ir_validator import validate_ir

    g = _make_multi_input_graph()
    ir = graph_to_native_ir(g)
    errors = validate_ir(ir)

    assert errors == [], f"Unexpected validation errors: {errors}"


def test_ir_validator_rejects_missing_schema_version():
    """A document missing schema_version should fail validation."""
    from research.runtime.native.ir_validator import validate_ir

    ir = {
        "model_dim": 64,
        "nodes": [{"id": 0, "op_name": "input", "input_ids": [], "config": {}}],
        "edges": [],
        "output_node_id": 0,
    }
    errors = validate_ir(ir)
    assert len(errors) > 0
    assert any("schema_version" in e for e in errors)


def test_ir_validator_rejects_invalid_node_with_extra_field():
    """A node with an extra field would fail with jsonschema (additionalProperties: false).

    Without the jsonschema package installed, the manual fallback validator does
    not check additionalProperties.  Either way, the converter must never emit
    output_shape -- that is covered by test_converter_strips_output_shape.

    When jsonschema IS available, this test confirms rejection.  When it is NOT
    available, we verify the converter itself prevents the problem.
    """
    from research.runtime.native.ir_validator import HAS_JSONSCHEMA, validate_ir

    ir = {
        "schema_version": "native_ir.v1",
        "model_dim": 64,
        "nodes": [
            {
                "id": 0,
                "op_name": "input",
                "input_ids": [],
                "config": {},
                "is_input": True,
                "is_output": False,
                "output_shape": {"batch": "B", "seq": "S", "dim": 64},
            }
        ],
        "edges": [],
        "output_node_id": 0,
    }
    errors = validate_ir(ir)
    if HAS_JSONSCHEMA:
        # Full schema validation catches the extra field
        assert len(errors) > 0
    else:
        # Manual fallback does not check additionalProperties;
        # the converter is responsible for never including output_shape.
        # Verify our converter never produces this field.
        g = _make_simple_graph()
        converted = graph_to_native_ir(g)
        for node in converted["nodes"]:
            assert "output_shape" not in node


def test_ir_validator_rejects_missing_edges():
    """A document missing the edges field should fail validation."""
    from research.runtime.native.ir_validator import validate_ir

    ir = {
        "schema_version": "native_ir.v1",
        "model_dim": 64,
        "nodes": [{"id": 0, "op_name": "input", "input_ids": [], "config": {}}],
        "output_node_id": 0,
    }
    errors = validate_ir(ir)
    assert len(errors) > 0
    assert any("edges" in e for e in errors)


def test_ir_validator_rejects_nonexistent_output_node_id():
    """output_node_id referencing a missing node should fail structural validation."""
    from research.runtime.native.ir_validator import validate_ir

    ir = {
        "schema_version": "native_ir.v1",
        "model_dim": 64,
        "nodes": [{"id": 0, "op_name": "input", "input_ids": [], "config": {}}],
        "edges": [],
        "output_node_id": 99,
    }
    errors = validate_ir(ir)
    assert len(errors) > 0
    assert any("output_node_id" in e for e in errors)
