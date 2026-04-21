from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict

from research.synthesis.graph import ComputationGraph
from research.synthesis.workflow_converter import workflow_to_computation_graph

_MAX_CACHE_ENTRIES = 32


@dataclass
class _WorkflowGraphCacheEntry:
    workflow: Dict[str, Any]
    model_dim: int
    graph: ComputationGraph
    id_map: Dict[str, int]


_WORKFLOW_GRAPH_CACHE: "OrderedDict[tuple[int, int], _WorkflowGraphCacheEntry]" = (
    OrderedDict()
)


def materialize_workflow_graph(
    workflow: Dict[str, Any],
    model_dim: int,
    *,
    return_id_map: bool = False,
) -> Any:
    """Convert a workflow dict to ``ComputationGraph`` with a small identity cache.

    The source-of-truth object remains workflow JSON. This only avoids repeated
    conversion when the same in-memory payload is reused across a save/evaluate
    request path.
    """

    model_dim = int(model_dim)
    key = (id(workflow), model_dim)
    cached = _WORKFLOW_GRAPH_CACHE.get(key)
    if cached is not None and cached.workflow is workflow:
        _WORKFLOW_GRAPH_CACHE.move_to_end(key)
        if return_id_map:
            return cached.graph, dict(cached.id_map)
        return cached.graph

    graph, id_map = workflow_to_computation_graph(
        workflow,
        model_dim,
        return_id_map=True,
    )
    _WORKFLOW_GRAPH_CACHE[key] = _WorkflowGraphCacheEntry(
        workflow=workflow,
        model_dim=model_dim,
        graph=graph,
        id_map=dict(id_map),
    )
    if len(_WORKFLOW_GRAPH_CACHE) > _MAX_CACHE_ENTRIES:
        _WORKFLOW_GRAPH_CACHE.popitem(last=False)
    if return_id_map:
        return graph, dict(id_map)
    return graph
