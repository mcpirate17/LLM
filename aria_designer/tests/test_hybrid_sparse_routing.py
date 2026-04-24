from __future__ import annotations

import yaml
from pathlib import Path

from aria_designer.runtime.bridge import validate_workflow_graph, workflow_to_graph


def _hybrid_workflow():
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "hybrid_sparse_router",
        "name": "Hybrid Sparse Router",
        "nodes": _hybrid_nodes(),
        "edges": _hybrid_edges(),
    }


def _node(node_id, component_type, params=None):
    return {
        "id": node_id,
        "component_type": component_type,
        "params": params or {},
        "ui_meta": {},
    }


def _hybrid_nodes():
    return [
        _node("in", "graph_input"),
        _node("default", "routing/default_path"),
        _node("gate", "routing/hybrid_token_gate", {"threshold": 0.45}),
        _node(
            "span",
            "routing/sparse_span_builder",
            {"span_width": 3, "fallback_behavior": "default_path"},
        ),
        _node(
            "router",
            "routing/hybrid_sparse_router",
            {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.45},
        ),
        _node("lane", "routing/lane_conditioned_block", {"lane_id": 1}),
        _node("merge", "add"),
        _node("out", "graph_output"),
    ]


def _edge(edge_id, source, target, source_port, target_port):
    return {
        "id": edge_id,
        "source": source,
        "source_port": source_port,
        "target": target,
        "target_port": target_port,
    }


def _hybrid_edges():
    return [
        _edge("e0", "in", "default", "out", "x"),
        _edge("e1", "in", "gate", "out", "x"),
        _edge("e2", "gate", "span", "y", "x"),
        _edge("e3", "span", "router", "y", "x"),
        _edge("e4", "router", "lane", "y", "x"),
        _edge("e5", "default", "merge", "y", "a"),
        _edge("e6", "lane", "merge", "y", "b"),
        _edge("e7", "merge", "out", "y", "x"),
    ]


def test_validate_hybrid_sparse_workflow_succeeds_and_suggests():
    result = validate_workflow_graph(_hybrid_workflow(), model_dim=64)
    assert result["valid"] is True
    assert result.get("design_suggestions")


def test_validate_hybrid_sparse_workflow_requires_default_path():
    workflow = _hybrid_workflow()
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != "default"]
    workflow["edges"] = [
        edge for edge in workflow["edges"] if edge["source"] != "default"
    ]
    result = validate_workflow_graph(workflow, model_dim=64)
    assert result["valid"] is False
    assert "default_path" in result["error"]


def test_workflow_to_graph_lower_hybrid_sparse_components():
    graph = workflow_to_graph(_hybrid_workflow(), model_dim=64)
    op_names = [node.op_name for node in graph.nodes.values() if not node.is_input]
    assert "hybrid_token_gate" in op_names
    assert "sparse_span_builder" in op_names
    assert "hybrid_sparse_router" in op_names


def test_hybrid_sparse_router_workflow_matches_manifest_template_id():
    manifest_path = (
        Path(__file__).resolve().parent.parent
        / "components"
        / "routing"
        / "hybrid_sparse_router"
        / "manifest.yaml"
    )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    template = manifest["templates"][0]
    workflow = _hybrid_workflow()
    assert template["id"] == "hybrid_sparse_triplet_router"
    assert template["workflow"] == workflow["workflow_id"]
