"""Tests for the research survivor importer."""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Also need research/ on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runtime.importer import graph_to_workflow
from runtime.bridge import workflow_to_graph, validate_workflow_graph
from research.synthesis.graph import ComputationGraph


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_simple_graph(model_dim=256):
    """linear → gelu → linear (with residual)."""
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    l1 = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    act = g.add_op("gelu", [l1])
    l2 = g.add_op("linear_proj", [act], {"out_dim": model_dim})
    res = g.add_op("add", [inp, l2])
    g.set_output(res)
    return g


def _make_attention_graph(model_dim=128):
    """Q/K/V → matmul → softmax → matmul → proj."""
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    q = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    k = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    v = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    attn = g.add_op("matmul", [q, k])
    sm = g.add_op("softmax_last", [attn])
    av = g.add_op("matmul", [sm, v])
    proj = g.add_op("linear_proj", [av], {"out_dim": model_dim})
    g.set_output(proj)
    return g


def _make_minimal_graph(model_dim=256):
    """Minimal: input → linear → output."""
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    l1 = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    g.set_output(l1)
    return g


# ── Tests: graph → workflow conversion ───────────────────────────────

def test_simple_conversion():
    g = _make_simple_graph()
    wf = graph_to_workflow(g, workflow_id="test_1", name="Test Simple")
    assert wf["workflow_id"] == "test_1"
    assert wf["name"] == "Test Simple"
    assert wf["schema_version"] == "workflow_graph.v1"
    assert len(wf["nodes"]) == 6  # input + 4 ops + output
    assert len(wf["edges"]) == 6  # 5 data edges + 1 output edge


def test_attention_conversion():
    g = _make_attention_graph()
    wf = graph_to_workflow(g)
    # input + 7 ops + output = 9 nodes
    assert len(wf["nodes"]) == 9


def test_metadata_populated():
    g = _make_simple_graph()
    wf = graph_to_workflow(g, metadata={"custom_key": "value"})
    assert wf["metadata"]["model_dim"] == 256
    assert wf["metadata"]["source"] == "research_import"
    assert wf["metadata"]["graph_fingerprint"] == g.fingerprint()
    assert wf["metadata"]["custom_key"] == "value"


def test_auto_layout():
    g = _make_simple_graph()
    wf = graph_to_workflow(g)
    for node in wf["nodes"]:
        assert "position" in node["ui_meta"]
        assert "x" in node["ui_meta"]["position"]
        assert "y" in node["ui_meta"]["position"]


def test_io_nodes_present():
    g = _make_minimal_graph()
    wf = graph_to_workflow(g)
    types = {n["component_type"] for n in wf["nodes"]}
    assert "io/input" in types
    assert "io/output_head" in types


def test_params_preserved():
    # Output must match model_dim, so use a chain: linear(512) → linear(256)
    g = ComputationGraph(model_dim=256)
    inp = g.add_input()
    l1 = g.add_op("linear_proj", [inp], {"out_dim": 512})
    l2 = g.add_op("linear_proj", [l1], {"out_dim": 256})
    g.set_output(l2)
    wf = graph_to_workflow(g)
    linear_nodes = [n for n in wf["nodes"] if n["component_type"].endswith("/linear_proj")]
    assert len(linear_nodes) == 2
    # First linear should have out_dim=512
    dims = sorted([n["params"]["out_dim"] for n in linear_nodes])
    assert 512 in dims


# ── Tests: round-trip (graph → workflow → graph) ─────────────────────

def test_roundtrip_simple():
    g1 = _make_simple_graph()
    wf = graph_to_workflow(g1)
    g2 = workflow_to_graph(wf, model_dim=256)
    assert g1.fingerprint() == g2.fingerprint()
    assert g1.n_ops() == g2.n_ops()
    assert g1.depth() == g2.depth()


def test_roundtrip_attention():
    g1 = _make_attention_graph(model_dim=128)
    wf = graph_to_workflow(g1)
    g2 = workflow_to_graph(wf, model_dim=128)
    assert g1.fingerprint() == g2.fingerprint()
    assert g1.n_ops() == g2.n_ops()


def test_roundtrip_minimal():
    g1 = _make_minimal_graph()
    wf = graph_to_workflow(g1)
    g2 = workflow_to_graph(wf, model_dim=256)
    assert g1.fingerprint() == g2.fingerprint()


def test_roundtrip_validates():
    g = _make_simple_graph()
    wf = graph_to_workflow(g)
    result = validate_workflow_graph(wf, model_dim=256)
    assert result["valid"] is True


# ── Tests: JSON serialization ────────────────────────────────────────

def test_json_serializable():
    g = _make_simple_graph()
    wf = graph_to_workflow(g)
    json_str = json.dumps(wf)
    parsed = json.loads(json_str)
    assert parsed["workflow_id"] == wf["workflow_id"]
    assert len(parsed["nodes"]) == len(wf["nodes"])
