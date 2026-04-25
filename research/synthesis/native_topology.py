from __future__ import annotations

import heapq
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional

from ..scientist.native.core import _try_import_rust_scheduler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .graph import ComputationGraph


@lru_cache(maxsize=1)
def _try_import_aria_core():
    try:
        import aria_core

        return aria_core
    except Exception:
        return None


def _graph_to_native_ir_json(graph: "ComputationGraph") -> str:
    from .native_ir_converter import graph_to_native_ir_json

    return graph_to_native_ir_json(graph)


def _build_canonical_topology_inputs(
    graph: "ComputationGraph",
) -> tuple[int, list[tuple[int, int]], list[str], list[str], list[object]]:
    cached = getattr(graph, "_cache", {}).get("canonical_topology_inputs")
    if cached is not None:
        return cached

    n_nodes = graph._next_id
    op_names = [""] * n_nodes
    config_strs = [""] * n_nodes
    node_inputs: list[object] = [()] * n_nodes
    edges: list[tuple[int, int]] = []

    for nid, node in graph.nodes.items():
        op_names[nid] = node.op_name
        config_strs[nid] = node._config_repr
        node_inputs[nid] = node.input_ids
        for iid in node.input_ids:
            edges.append((iid, nid))

    payload = (n_nodes, edges, op_names, config_strs, node_inputs)
    if hasattr(graph, "_cache"):
        graph._cache["canonical_topology_inputs"] = payload
    return payload


def _topological_order_with_aria_core(graph: "ComputationGraph") -> Optional[List[int]]:
    aria_core = _try_import_aria_core()
    if aria_core is None or not hasattr(aria_core, "canonical_topo_sort"):
        return None

    try:
        n_nodes, edges, op_names, config_strs, node_inputs = (
            _build_canonical_topology_inputs(graph)
        )
        order = aria_core.canonical_topo_sort(
            n_nodes, edges, op_names, config_strs, node_inputs
        )
    except Exception as exc:
        logger.debug("aria_core.canonical_topo_sort failed: %s", exc)
        return None

    return [int(nid) for nid in order if int(nid) in graph.nodes]


def _topological_order_with_rust_scheduler(
    graph: "ComputationGraph",
) -> Optional[List[int]]:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "topological_order"):
        return None

    try:
        order = rust.topological_order(_graph_to_native_ir_json(graph))
    except Exception as exc:
        logger.debug("aria_scheduler.topological_order failed: %s", exc)
        return None

    return [int(nid) for nid in order if int(nid) in graph.nodes]


def _python_topological_order(graph: "ComputationGraph") -> List[int]:
    in_degree = {nid: len(node.input_ids) for nid, node in graph.nodes.items()}
    children = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for iid in node.input_ids:
            children[iid].append(nid)

    static_keys: Dict[int, tuple[str, str, int]] = {}
    for nid, node in graph.nodes.items():
        static_keys[nid] = (node.op_name, node._config_repr, nid)

    order: List[int] = []
    canonical_id_map: Dict[int, int] = {}
    ready: list[tuple[str, tuple[int, ...], str, int]] = []

    for nid, deg in in_degree.items():
        if deg == 0:
            op_name, config_str, orig_id = static_keys[nid]
            heapq.heappush(ready, (op_name, (), config_str, orig_id))

    while ready:
        _, _, _, node_id = heapq.heappop(ready)
        canonical_id_map[node_id] = len(order)
        order.append(node_id)

        for child_id in children[node_id]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                child = graph.nodes[child_id]
                input_keys = tuple(canonical_id_map[iid] for iid in child.input_ids)
                op_name, config_str, orig_id = static_keys[child_id]
                heapq.heappush(ready, (op_name, input_keys, config_str, orig_id))

    if len(order) < len(graph.nodes):
        return sorted(graph.nodes.keys())
    return order


def compute_topological_order(graph: "ComputationGraph") -> List[int]:
    order = _topological_order_with_aria_core(graph)
    if order is not None:
        return order

    order = _topological_order_with_rust_scheduler(graph)
    if order is not None:
        return order

    return _python_topological_order(graph)
