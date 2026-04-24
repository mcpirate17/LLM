from __future__ import annotations
from research.synthesis.graph import ComputationGraph
from research.synthesis.workflow_converter import workflow_to_computation_graph
from aria_designer.runtime.importer import graph_to_workflow


def test_research_import_canonicalization():
    # Create a simple graph in research
    graph = ComputationGraph(128)
    x = graph.add_input()

    # Add some common ops
    # Note: research op names are often just the leaf name
    y = graph.add_op("linear_proj", [x], {"out_dim": 128})
    z = graph.add_op("relu", [y], {})
    graph.set_output(z)

    # Convert to workflow using the importer
    wf = graph_to_workflow(graph, workflow_id="test_wf", name="Test Research Workflow")

    # Verify canonicalization
    node_types = {n["component_type"] for n in wf["nodes"]}

    # Should contain canonical IDs, not just leaf names or wrong categories
    assert "io/input" in node_types
    assert "linear_algebra/linear_proj" in node_types
    assert "math/relu" in node_types
    assert "io/output_head" in node_types

    # Ensure no unprefixed IDs
    for ct in node_types:
        assert "/" in ct, f"Component type '{ct}' is not canonicalized"


def test_workflow_import_canonicalizes_long_conv_hyen_alias():
    workflow = {
        "workflow_id": "test_long_conv_hyen_alias",
        "metadata": {"model_dim": 128},
        "nodes": [
            {"id": "in", "component_type": "io/input", "params": {}},
            {"id": "mix", "component_type": "mixing/long_conv_hyen", "params": {}},
            {"id": "out", "component_type": "io/output_head", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "in", "target": "mix"},
            {"id": "e1", "source": "mix", "target": "out"},
        ],
    }

    graph = workflow_to_computation_graph(workflow)
    op_names = {node.op_name for node in graph.nodes.values()}
    assert "long_conv_hyena" in op_names
    assert "long_conv_hyen" not in op_names
