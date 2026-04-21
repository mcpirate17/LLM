"""JSON serialization for ComputationGraph."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from ._json_compat import dumps_json, loads_json
from .graph import ComputationGraph


def graph_to_json(graph: ComputationGraph) -> str:
    """Serialize a computation graph to JSON."""
    cached = getattr(graph, "_cache", {}).get("graph_json")
    if cached is not None:
        return cached
    payload = dumps_json(graph.to_dict())
    if hasattr(graph, "_cache"):
        graph._cache["graph_json"] = payload
    return payload


@lru_cache(maxsize=256)
def _graph_dict_from_json_cached(json_str: str) -> dict:
    return loads_json(json_str)


def graph_from_json(
    json_str: str,
    model_dim: Optional[int] = None,
) -> ComputationGraph:
    """Deserialize a computation graph from JSON.

    Caches the decoded JSON document and rebuilds a fresh graph per call so
    callers can mutate the result without contaminating shared cache state.
    """
    graph = ComputationGraph.from_dict(_graph_dict_from_json_cached(json_str))
    if model_dim is not None:
        graph.model_dim = int(model_dim)
    return graph


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
