"""Convert ComputationGraph to/from native_ir.v1 format.

The native_ir.v1 schema differs from ComputationGraph.to_dict() in several ways:
- ``schema_version`` field required (const "native_ir.v1")
- ``nodes`` is an array, not a dict keyed by string IDs
- ``edges`` must be explicitly listed (derived from ``input_ids``)
- ``output_shape`` is NOT allowed on nodes (``additionalProperties: false``)
- ``is_input`` / ``is_output`` are optional booleans (always included for clarity)

The Rust scheduler's ``GraphIR`` (graph.rs) consumes this same format.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .graph import ComputationGraph


def graph_to_native_ir(graph: ComputationGraph) -> dict:
    """Convert a ComputationGraph to a native_ir.v1 document.

    The returned dict validates against ``schemas/native_ir.v1.json`` and can
    be serialized directly to JSON for the Rust scheduler.
    """
    reachable = graph.get_reachable_nodes()
    if len(reachable) != len(graph.nodes):
        dead_count = len(graph.nodes) - len(reachable)
        raise ValueError(
            f"Graph contains {dead_count} unreachable nodes (dead branches)"
        )

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        ir_node: Dict[str, Any] = {
            "id": node.id,
            "op_name": node.op_name,
            "input_ids": list(node.input_ids),
            "config": dict(node.config),
            "is_input": node.is_input,
            "is_output": node.is_output,
        }
        # NOTE: output_shape is intentionally omitted -- the schema uses
        # additionalProperties: false on node objects, so including it
        # would cause validation failure.
        nodes.append(ir_node)

        # Derive edges from input_ids
        for inp_id in node.input_ids:
            edges.append({"source": inp_id, "target": node.id})

    return {
        "schema_version": "native_ir.v1",
        "model_dim": graph.model_dim,
        "nodes": nodes,
        "edges": edges,
        "output_node_id": graph._output_node_id,
    }


def graph_to_native_ir_json(graph: ComputationGraph) -> str:
    """Convert a ComputationGraph and serialize to a compact JSON string."""
    return json.dumps(graph_to_native_ir(graph), separators=(",", ":"))
