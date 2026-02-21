"""
Aria Patch Engine — materializes patch operations into workflow changes.

Supports:
  - add_node: Insert a new node into the workflow
  - remove_node: Remove a node and its connected edges
  - replace_node: Swap a node's component type (re-wire in place)
  - rewire: Add, remove, or modify edges
  - mutate_param: Change a node's parameters

Each operation is validated before application. The engine returns
the modified workflow or raises PatchError on failure.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4


class PatchError(Exception):
    """Raised when a patch operation fails."""
    def __init__(self, op_index: int, op_type: str, message: str):
        self.op_index = op_index
        self.op_type = op_type
        super().__init__(f"Patch op #{op_index} ({op_type}): {message}")


def apply_patch_ops(
    workflow: Dict[str, Any],
    ops: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply a list of patch operations to a workflow, returning the modified workflow.

    The input workflow is not mutated; a deep copy is made first.

    Args:
        workflow: workflow_graph.v1 JSON dict
        ops: list of patch operation dicts (from AriaPatchProposalModel.ops)

    Returns:
        Modified workflow dict

    Raises:
        PatchError: if any operation fails validation
    """
    wf = copy.deepcopy(workflow)

    # Build mutable indices
    nodes_by_id = {n["id"]: n for n in wf.get("nodes", [])}
    edges_by_id = {e.get("id", f"_e{i}"): e for i, e in enumerate(wf.get("edges", []))}

    for i, op_dict in enumerate(ops):
        op_type = op_dict.get("op", "")
        payload = op_dict.get("payload", {})
        node_id = op_dict.get("node_id")
        edge_id = op_dict.get("edge_id")

        if op_type == "add_node":
            _apply_add_node(i, nodes_by_id, edges_by_id, payload)
        elif op_type == "remove_node":
            _apply_remove_node(i, node_id, nodes_by_id, edges_by_id)
        elif op_type == "replace_node":
            _apply_replace_node(i, node_id, nodes_by_id, payload)
        elif op_type == "rewire":
            _apply_rewire(i, edge_id, nodes_by_id, edges_by_id, payload)
        elif op_type == "mutate_param":
            _apply_mutate_param(i, node_id, nodes_by_id, payload)
        else:
            raise PatchError(i, op_type, f"Unknown operation type: {op_type}")

    # Rebuild lists from dicts
    wf["nodes"] = list(nodes_by_id.values())
    wf["edges"] = list(edges_by_id.values())

    return wf


def _apply_add_node(
    idx: int,
    nodes: Dict[str, Dict],
    edges: Dict[str, Dict],
    payload: Dict[str, Any],
):
    """Add a new node. Payload must include: id, component_type. Optional: params, ui_meta, edges."""
    node_id = payload.get("id")
    if not node_id:
        node_id = f"aria_{uuid4().hex[:8]}"

    if node_id in nodes:
        raise PatchError(idx, "add_node", f"Node '{node_id}' already exists")

    component_type = payload.get("component_type")
    if not component_type:
        raise PatchError(idx, "add_node", "Missing 'component_type' in payload")

    nodes[node_id] = {
        "id": node_id,
        "component_type": component_type,
        "params": payload.get("params", {}),
        "ui_meta": payload.get("ui_meta", {}),
    }

    # Optionally add edges connecting the new node
    for edge_spec in payload.get("edges", []):
        eid = edge_spec.get("id", f"aria_e_{uuid4().hex[:6]}")
        edges[eid] = {
            "id": eid,
            "source": edge_spec.get("source", ""),
            "source_port": edge_spec.get("source_port", "out"),
            "target": edge_spec.get("target", ""),
            "target_port": edge_spec.get("target_port", "in"),
        }


def _apply_remove_node(
    idx: int,
    node_id: Optional[str],
    nodes: Dict[str, Dict],
    edges: Dict[str, Dict],
):
    """Remove a node and all its connected edges."""
    if not node_id:
        raise PatchError(idx, "remove_node", "Missing 'node_id'")

    if node_id not in nodes:
        raise PatchError(idx, "remove_node", f"Node '{node_id}' not found")

    del nodes[node_id]

    # Remove all edges connected to this node
    to_remove = [
        eid for eid, e in edges.items()
        if e.get("source") == node_id or e.get("target") == node_id
    ]
    for eid in to_remove:
        del edges[eid]


def _apply_replace_node(
    idx: int,
    node_id: Optional[str],
    nodes: Dict[str, Dict],
    payload: Dict[str, Any],
):
    """Replace a node's component type and optionally its params."""
    if not node_id:
        raise PatchError(idx, "replace_node", "Missing 'node_id'")

    if node_id not in nodes:
        raise PatchError(idx, "replace_node", f"Node '{node_id}' not found")

    new_type = payload.get("component_type")
    if not new_type:
        raise PatchError(idx, "replace_node", "Missing 'component_type' in payload")

    nodes[node_id]["component_type"] = new_type

    if "params" in payload:
        nodes[node_id]["params"] = payload["params"]


def _apply_rewire(
    idx: int,
    edge_id: Optional[str],
    nodes: Dict[str, Dict],
    edges: Dict[str, Dict],
    payload: Dict[str, Any],
):
    """Add, remove, or modify an edge.

    Payload actions:
      - {"action": "add", "source": ..., "target": ..., ...}: add new edge
      - {"action": "remove"}: remove edge by edge_id
      - {"action": "modify", "source": ..., "target": ..., ...}: modify existing edge
    """
    action = payload.get("action", "modify")

    if action == "add":
        eid = edge_id or f"aria_e_{uuid4().hex[:6]}"
        source = payload.get("source")
        target = payload.get("target")
        if not source or not target:
            raise PatchError(idx, "rewire", "Missing 'source' or 'target' for add action")
        if source not in nodes:
            raise PatchError(idx, "rewire", f"Source node '{source}' not found")
        if target not in nodes:
            raise PatchError(idx, "rewire", f"Target node '{target}' not found")

        edges[eid] = {
            "id": eid,
            "source": source,
            "source_port": payload.get("source_port", "out"),
            "target": target,
            "target_port": payload.get("target_port", "in"),
        }

    elif action == "remove":
        if not edge_id:
            raise PatchError(idx, "rewire", "Missing 'edge_id' for remove action")
        if edge_id not in edges:
            raise PatchError(idx, "rewire", f"Edge '{edge_id}' not found")
        del edges[edge_id]

    elif action == "modify":
        if not edge_id:
            raise PatchError(idx, "rewire", "Missing 'edge_id' for modify action")
        if edge_id not in edges:
            raise PatchError(idx, "rewire", f"Edge '{edge_id}' not found")

        edge = edges[edge_id]
        if "source" in payload:
            if payload["source"] not in nodes:
                raise PatchError(idx, "rewire", f"Source node '{payload['source']}' not found")
            edge["source"] = payload["source"]
        if "target" in payload:
            if payload["target"] not in nodes:
                raise PatchError(idx, "rewire", f"Target node '{payload['target']}' not found")
            edge["target"] = payload["target"]
        if "source_port" in payload:
            edge["source_port"] = payload["source_port"]
        if "target_port" in payload:
            edge["target_port"] = payload["target_port"]

    else:
        raise PatchError(idx, "rewire", f"Unknown rewire action: {action}")


def _apply_mutate_param(
    idx: int,
    node_id: Optional[str],
    nodes: Dict[str, Dict],
    payload: Dict[str, Any],
):
    """Mutate parameters on a node. Payload: {"key": "value", ...} to set."""
    if not node_id:
        raise PatchError(idx, "mutate_param", "Missing 'node_id'")

    if node_id not in nodes:
        raise PatchError(idx, "mutate_param", f"Node '{node_id}' not found")

    params = nodes[node_id].get("params", {})

    for key, value in payload.items():
        if value is None:
            # None means delete the param
            params.pop(key, None)
        else:
            params[key] = value

    nodes[node_id]["params"] = params
