"""Cross-format compatibility tests for ComputationGraph.

Verifies correct conversion between:
  1. ComputationGraph (Python object)
  2. native_ir.v1 (JSON schema for Rust scheduler)
  3. Round-trip through to_dict/from_dict
"""

import pytest

from research.synthesis.graph import ComputationGraph

# ---------------------------------------------------------------------------
# Inline converter: graph_to_native_ir
#
# The canonical module (research.synthesis.native_ir_converter) is being
# built by another agent.  We provide a minimal reference implementation
# here so the tests are self-contained.
# ---------------------------------------------------------------------------

try:
    from research.synthesis.native_ir_converter import graph_to_native_ir
except ImportError:

    def graph_to_native_ir(g: ComputationGraph) -> dict:
        """Convert a ComputationGraph to native_ir.v1 format."""
        topo = g.topological_order()
        nodes = []
        edges = []

        for nid in topo:
            node = g.nodes[nid]
            ir_node = {
                "id": node.id,
                "op_name": node.op_name,
                "input_ids": list(node.input_ids),
                "config": dict(node.config),
            }
            if node.is_input:
                ir_node["is_input"] = True
            if node.is_output:
                ir_node["is_output"] = True
            nodes.append(ir_node)

            # Build explicit edges from input_ids
            for inp_id in node.input_ids:
                edges.append({"source": inp_id, "target": node.id})

        return {
            "schema_version": "native_ir.v1",
            "model_dim": g.model_dim,
            "nodes": nodes,
            "edges": edges,
            "output_node_id": g._output_node_id,
        }


from research.runtime.native.ir_validator import validate_ir

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _make_simple_graph() -> ComputationGraph:
    """input(64) -> relu -> output"""
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    g.set_output(relu)
    return g


def _make_binary_graph() -> ComputationGraph:
    """input(64) -> relu, input handled via fork:
       input(64) -> relu
       input(64) -> gelu
       [relu, gelu] -> add -> output

    Since ComputationGraph supports only one input node, we fork from
    a single input.
    """
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    relu = g.add_op("relu", [inp])
    gelu = g.add_op("gelu", [inp])
    add = g.add_op("add", [relu, gelu])
    g.set_output(add)
    return g


def _make_linear_graph() -> ComputationGraph:
    """input(64) -> linear_proj_down(64->32) -> relu -> linear_proj_up(32->64) -> output

    Uses linear_proj_down / linear_proj_up from the primitive registry.
    set_output requires output dim == model_dim, so we project back up.
    """
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    lin1 = g.add_op("linear_proj_down", [inp], config={"out_dim": 32})
    relu = g.add_op("relu", [lin1])
    lin2 = g.add_op("linear_proj_up", [relu], config={"out_dim": 64})
    g.set_output(lin2)
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDictRoundtrip:

    def test_graph_to_dict_roundtrip(self):
        """to_dict -> from_dict -> to_dict produces identical dict."""
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            d1 = g.to_dict()
            g2 = ComputationGraph.from_dict(d1)
            d2 = g2.to_dict()
            assert d1 == d2, f"Round-trip failed for {factory.__name__}"


class TestNativeIRStructure:

    def test_graph_to_native_ir_has_required_fields(self):
        """Converted IR has all fields required by the schema."""
        g = _make_simple_graph()
        ir = graph_to_native_ir(g)
        for field in ("schema_version", "model_dim", "nodes", "edges", "output_node_id"):
            assert field in ir, f"Missing required field: {field}"

    def test_graph_to_native_ir_node_count_matches(self):
        """Same number of nodes in both formats."""
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            ir = graph_to_native_ir(g)
            assert len(ir["nodes"]) == len(g.nodes), (
                f"Node count mismatch for {factory.__name__}: "
                f"IR has {len(ir['nodes'])}, graph has {len(g.nodes)}"
            )

    def test_graph_to_native_ir_edges_match_input_ids(self):
        """Each edge corresponds to an input_id reference in the graph."""
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            ir = graph_to_native_ir(g)

            # Collect expected edges from ComputationGraph input_ids
            expected_edges = set()
            for node in g.nodes.values():
                for inp_id in node.input_ids:
                    expected_edges.add((inp_id, node.id))

            # Collect actual edges from IR
            actual_edges = set()
            for edge in ir["edges"]:
                actual_edges.add((edge["source"], edge["target"]))

            assert expected_edges == actual_edges, (
                f"Edge mismatch for {factory.__name__}: "
                f"expected {expected_edges}, got {actual_edges}"
            )

    def test_native_ir_schema_version_correct(self):
        """Converted IR has schema_version 'native_ir.v1'."""
        g = _make_simple_graph()
        ir = graph_to_native_ir(g)
        assert ir["schema_version"] == "native_ir.v1"

    def test_native_ir_no_extra_node_fields(self):
        """Converted nodes must not have 'output_shape' (schema forbids it via additionalProperties: false)."""
        allowed_fields = {"id", "op_name", "input_ids", "config", "is_input", "is_output"}
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            ir = graph_to_native_ir(g)
            for node in ir["nodes"]:
                extra = set(node.keys()) - allowed_fields
                assert not extra, (
                    f"Node {node['id']} in {factory.__name__} has extra fields: {extra}"
                )

    def test_all_ops_in_ir_are_valid_strings(self):
        """Every op_name in IR is a non-empty string."""
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            ir = graph_to_native_ir(g)
            for node in ir["nodes"]:
                assert isinstance(node["op_name"], str), (
                    f"op_name is not a string: {node['op_name']}"
                )
                assert len(node["op_name"]) > 0, (
                    f"op_name is empty for node {node['id']}"
                )


class TestNativeIRValidation:

    def test_native_ir_validator_accepts_simple_graph(self):
        """validate_ir returns no errors for simple graph."""
        g = _make_simple_graph()
        ir = graph_to_native_ir(g)
        errors = validate_ir(ir)
        assert errors == [], f"Validator errors for simple graph: {errors}"

    def test_native_ir_validator_accepts_binary_graph(self):
        """validate_ir returns no errors for binary op graph."""
        g = _make_binary_graph()
        ir = graph_to_native_ir(g)
        errors = validate_ir(ir)
        assert errors == [], f"Validator errors for binary graph: {errors}"

    def test_native_ir_validator_accepts_linear_graph(self):
        """validate_ir returns no errors for linear graph."""
        g = _make_linear_graph()
        ir = graph_to_native_ir(g)
        errors = validate_ir(ir)
        assert errors == [], f"Validator errors for linear graph: {errors}"


class TestTopologicalOrderConsistency:

    def test_graph_topological_order_matches_native_ir(self):
        """Topo order from ComputationGraph matches IR node ordering.

        The native IR nodes are emitted in topological order by the
        converter. Verify that the ordering is consistent: for every
        edge (source -> target), source appears before target in the
        node list.
        """
        for factory in (_make_simple_graph, _make_binary_graph, _make_linear_graph):
            g = factory()
            ir = graph_to_native_ir(g)

            # Build position map from IR node order
            ir_node_ids = [n["id"] for n in ir["nodes"]]
            pos = {nid: i for i, nid in enumerate(ir_node_ids)}

            # Verify topo invariant: sources before targets
            for edge in ir["edges"]:
                assert pos[edge["source"]] < pos[edge["target"]], (
                    f"Topological order violated in {factory.__name__}: "
                    f"node {edge['source']} (pos {pos[edge['source']]}) "
                    f"should come before node {edge['target']} (pos {pos[edge['target']]})"
                )

            # Also verify ComputationGraph.topological_order() is consistent
            graph_topo = g.topological_order()
            graph_pos = {nid: i for i, nid in enumerate(graph_topo)}
            for node in g.nodes.values():
                for inp_id in node.input_ids:
                    assert graph_pos[inp_id] < graph_pos[node.id], (
                        f"Graph topo order violated: "
                        f"input {inp_id} should come before {node.id}"
                    )
