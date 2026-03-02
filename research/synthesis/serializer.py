"""
JSON Serialization for Computation Graphs

Serialize/deserialize graphs for storage, sharing, and the lab notebook.
"""

from __future__ import annotations

import json
from typing import Dict

from .graph import ComputationGraph


def graph_to_json(graph: ComputationGraph) -> str:
    """Serialize a computation graph to JSON."""
    return json.dumps(graph.to_dict(), separators=(",", ":"))


def graph_from_json(json_str: str) -> ComputationGraph:
    """Deserialize a computation graph from JSON."""
    d = json.loads(json_str)
    return ComputationGraph.from_dict(d)


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
