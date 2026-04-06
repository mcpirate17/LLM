from __future__ import annotations

import yaml
from pathlib import Path

from aria_designer.runtime.bridge import validate_workflow_graph, workflow_to_graph


def _hybrid_workflow():
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "hybrid_sparse_router",
        "name": "Hybrid Sparse Router",
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}, "ui_meta": {}},
            {
                "id": "default",
                "component_type": "routing/default_path",
                "params": {},
                "ui_meta": {},
            },
            {
                "id": "gate",
                "component_type": "routing/hybrid_token_gate",
                "params": {"threshold": 0.45},
                "ui_meta": {},
            },
            {
                "id": "span",
                "component_type": "routing/sparse_span_builder",
                "params": {"span_width": 3, "fallback_behavior": "default_path"},
                "ui_meta": {},
            },
            {
                "id": "router",
                "component_type": "routing/hybrid_sparse_router",
                "params": {
                    "span_width": 3,
                    "lane_count": 3,
                    "confidence_threshold": 0.45,
                },
                "ui_meta": {},
            },
            {
                "id": "lane",
                "component_type": "routing/lane_conditioned_block",
                "params": {"lane_id": 1},
                "ui_meta": {},
            },
            {"id": "merge", "component_type": "add", "params": {}, "ui_meta": {}},
            {
                "id": "out",
                "component_type": "graph_output",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "source_port": "out",
                "target": "default",
                "target_port": "x",
            },
            {
                "id": "e1",
                "source": "in",
                "source_port": "out",
                "target": "gate",
                "target_port": "x",
            },
            {
                "id": "e2",
                "source": "gate",
                "source_port": "y",
                "target": "span",
                "target_port": "x",
            },
            {
                "id": "e3",
                "source": "span",
                "source_port": "y",
                "target": "router",
                "target_port": "x",
            },
            {
                "id": "e4",
                "source": "router",
                "source_port": "y",
                "target": "lane",
                "target_port": "x",
            },
            {
                "id": "e5",
                "source": "default",
                "source_port": "y",
                "target": "merge",
                "target_port": "a",
            },
            {
                "id": "e6",
                "source": "lane",
                "source_port": "y",
                "target": "merge",
                "target_port": "b",
            },
            {
                "id": "e7",
                "source": "merge",
                "source_port": "y",
                "target": "out",
                "target_port": "x",
            },
        ],
    }


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
