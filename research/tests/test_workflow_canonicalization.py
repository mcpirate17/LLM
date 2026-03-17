from __future__ import annotations
from research.synthesis.graph import ComputationGraph
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
