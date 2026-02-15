"""
JSON Serialization for Computation Graphs

Serialize/deserialize graphs for storage, sharing, and the lab notebook.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .graph import ComputationGraph


def graph_to_json(graph: ComputationGraph) -> str:
    """Serialize a computation graph to JSON."""
    return json.dumps(graph.to_dict(), separators=(",", ":"))


def graph_from_json(json_str: str) -> ComputationGraph:
    """Deserialize a computation graph from JSON."""
    d = json.loads(json_str)
    return ComputationGraph.from_dict(d)


def graphs_to_json(graphs: List[ComputationGraph]) -> str:
    """Serialize a list of graphs (for a multi-layer model)."""
    return json.dumps([g.to_dict() for g in graphs], separators=(",", ":"))


def graphs_from_json(json_str: str) -> List[ComputationGraph]:
    """Deserialize a list of graphs."""
    data = json.loads(json_str)
    return [ComputationGraph.from_dict(d) for d in data]


def graph_to_mermaid(graph: ComputationGraph) -> str:
    """Convert a graph to Mermaid diagram syntax for visualization."""
    lines = ["graph TD"]
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        label = node.op_name
        shape_str = f"{node.output_shape.seq},{node.output_shape.dim}"

        if node.is_input:
            lines.append(f"    n{nid}([\"INPUT\\n({shape_str})\"])")
        elif node.is_output:
            lines.append(f"    n{nid}[[\"OUTPUT: {label}\\n({shape_str})\"]]")
        else:
            lines.append(f"    n{nid}[\"{label}\\n({shape_str})\"]")

        for inp_id in node.input_ids:
            lines.append(f"    n{inp_id} --> n{nid}")

    return "\n".join(lines)


def graph_summary(graph: ComputationGraph) -> Dict:
    """Compact summary of a graph for the lab notebook."""
    ops_used = set()
    for node in graph.nodes.values():
        if not node.is_input:
            ops_used.add(node.op_name)

    return {
        "fingerprint": graph.fingerprint(),
        "model_dim": graph.model_dim,
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "params_estimate": graph.n_params_estimate(),
        "ops_used": sorted(ops_used),
        "has_gradient_path": graph.has_gradient_path(),
    }
