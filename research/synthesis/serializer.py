"""JSON serialization for ComputationGraph."""

from __future__ import annotations

from typing import Dict

from ._json_compat import dumps_json, loads_json
from .graph import ComputationGraph


def graph_to_json(graph: ComputationGraph) -> str:
    """Serialize a computation graph to JSON."""
    return dumps_json(graph.to_dict())


def graph_from_json(json_str: str) -> ComputationGraph:
    """Deserialize a computation graph from JSON."""
    d = loads_json(json_str)
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
