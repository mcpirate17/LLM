from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _append_edge(
    edges: List[Dict[str, Any]],
    source: str,
    target: str,
    source_port: str = "out",
    target_port: str = "in",
) -> None:
    if not source or not target or source == target:
        return
    exists = any(
        e.get("source") == source
        and e.get("target") == target
        and (e.get("source_port") or "out") == (source_port or "out")
        and (e.get("target_port") or "in") == (target_port or "in")
        for e in edges
    )
    if exists:
        return
    edges.append(
        {
            "id": f"aria_e_{uuid4().hex[:6]}",
            "source": source,
            "source_port": source_port or "out",
            "target": target,
            "target_port": target_port or "in",
        }
    )


def _contains_token(component_type: str, token: str) -> bool:
    return token in str(component_type or "").lower()


def _remove_edge_once(
    edges: List[Dict[str, Any]],
    source: str,
    target: str,
) -> Optional[Dict[str, Any]]:
    for idx, edge in enumerate(edges):
        if edge.get("source") == source and edge.get("target") == target:
            return edges.pop(idx)
    return None


def _last_non_new_sink(
    nodes: List[Dict[str, Any]],
    source_ids: set[str],
    excluded_ids: set[str],
) -> Optional[Dict[str, Any]]:
    for node in reversed(nodes):
        node_id = node.get("id")
        if (
            node_id not in source_ids
            and node_id not in excluded_ids
            and not _contains_token(node.get("component_type", ""), "output")
        ):
            return node
    return None


def _node_edge_indexes(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    new_id: str,
) -> tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]], set[str]
]:
    incoming = [edge for edge in edges if edge.get("target") == new_id]
    outgoing = [edge for edge in edges if edge.get("source") == new_id]
    output_node = next(
        (
            node
            for node in reversed(nodes)
            if node.get("id") != new_id
            and _contains_token(node.get("component_type", ""), "output")
        ),
        None,
    )
    source_ids = {str(edge.get("source")) for edge in edges}
    return incoming, outgoing, output_node, source_ids


def _auto_connect_added_nodes(
    workflow: Dict[str, Any],
    added_node_ids: List[str],
    insertion_hints: Optional[Dict[str, Dict[str, str | None]]] = None,
) -> None:
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    if not nodes:
        return

    nodes_by_id = {str(n.get("id")): n for n in nodes}

    for new_id in added_node_ids:
        _auto_connect_added_node(
            nodes,
            edges,
            nodes_by_id,
            str(new_id),
            insertion_hints,
        )


def _auto_connect_added_node(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    new_id: str,
    insertion_hints: Optional[Dict[str, Dict[str, str | None]]],
) -> None:
    node = nodes_by_id.get(str(new_id))
    if not node:
        return
    ctype = str(node.get("component_type", ""))
    if _contains_token(ctype, "input") or _contains_token(ctype, "output"):
        return

    incoming, outgoing, output_node, source_ids = _node_edge_indexes(
        nodes, edges, str(new_id)
    )

    if incoming and outgoing:
        return

    hint = (insertion_hints or {}).get(str(new_id))
    if _connect_from_insertion_hint(
        edges, nodes_by_id, new_id, hint, incoming, outgoing, output_node
    ):
        return

    if _connect_missing_incoming(
        nodes, edges, new_id, outgoing, output_node, source_ids
    ):
        return

    if incoming and (not outgoing):
        src = str(incoming[-1].get("source", ""))
        if output_node:
            while _remove_edge_once(edges, src, str(output_node.get("id", ""))):
                pass
            _append_edge(
                edges, str(new_id), str(output_node.get("id", "")), "out", "in"
            )
            return

    if _insert_before_output(edges, new_id, output_node):
        return
    _connect_from_sink_or_input(nodes, edges, new_id, source_ids)


def _connect_from_insertion_hint(
    edges: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    new_id: str,
    hint: object,
    incoming: List[Dict[str, Any]],
    outgoing: List[Dict[str, Any]],
    output_node: Optional[Dict[str, Any]],
) -> bool:
    if not isinstance(hint, dict) or incoming or outgoing:
        return False
    after_id = hint.get("after_node_id")
    before_id = hint.get("before_node_id")
    if after_id and after_id in nodes_by_id:
        if before_id and before_id in nodes_by_id:
            _remove_edge_once(edges, str(after_id), str(before_id))
            _append_edge(edges, str(after_id), str(new_id))
            _append_edge(edges, str(new_id), str(before_id))
            return True
        _append_edge(edges, str(after_id), str(new_id))
        if output_node:
            _append_edge(edges, str(new_id), str(output_node.get("id", "")))
        return True
    if before_id and before_id in nodes_by_id:
        _append_edge(edges, str(new_id), str(before_id))
        return True
    return False


def _connect_missing_incoming(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    new_id: str,
    outgoing: List[Dict[str, Any]],
    output_node: Optional[Dict[str, Any]],
    source_ids: set[str],
) -> bool:
    if not outgoing or not output_node:
        return False
    if not any(edge.get("target") == output_node.get("id") for edge in outgoing):
        return False
    in_to_output = [
        edge
        for edge in edges
        if edge.get("target") == output_node.get("id") and edge.get("source") != new_id
    ]
    if in_to_output:
        old = in_to_output[-1]
        _remove_edge_object(edges, old)
        _append_edge(
            edges,
            str(old.get("source", "")),
            str(new_id),
            str(old.get("source_port") or "out"),
            "in",
        )
        return True
    sink = _last_non_new_sink(
        nodes, source_ids, {str(new_id), str(output_node.get("id"))}
    )
    if sink:
        _append_edge(edges, str(sink.get("id", "")), str(new_id))
        return True
    return False


def _remove_edge_object(edges: List[Dict[str, Any]], edge: Dict[str, Any]) -> None:
    try:
        edges.remove(edge)
    except ValueError:
        pass


def _insert_before_output(
    edges: List[Dict[str, Any]],
    new_id: str,
    output_node: Optional[Dict[str, Any]],
) -> bool:
    if not output_node:
        return False
    inc_to_output = [
        edge for edge in edges if edge.get("target") == output_node.get("id")
    ]
    if not inc_to_output:
        return False
    old = inc_to_output[-1]
    _remove_edge_object(edges, old)
    _append_edge(
        edges,
        str(old.get("source", "")),
        str(new_id),
        str(old.get("source_port") or "out"),
        "in",
    )
    _append_edge(
        edges,
        str(new_id),
        str(output_node.get("id", "")),
        "out",
        str(old.get("target_port") or "in"),
    )
    return True


def _connect_from_sink_or_input(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    new_id: str,
    source_ids: set[str],
) -> None:
    sink = _last_non_new_sink(nodes, source_ids, {str(new_id)})
    if sink:
        _append_edge(edges, str(sink.get("id", "")), str(new_id))
        return
    inputs = [
        node
        for node in nodes
        if _contains_token(node.get("component_type", ""), "input")
        and node.get("id") != new_id
    ]
    if inputs:
        _append_edge(edges, str(inputs[0].get("id", "")), str(new_id))


def _auto_layout_workflow(
    workflow: Dict[str, Any],
    insertion_hints: Optional[Dict[str, Dict[str, str | None]]] = None,
) -> None:
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    if not nodes:
        return

    node_ids = [str(n.get("id")) for n in nodes if n.get("id") is not None]
    if not node_ids:
        return

    indeg: Dict[str, int] = {nid: 0 for nid in node_ids}
    outs: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for e in edges:
        s = str(e.get("source", ""))
        t = str(e.get("target", ""))
        if s in outs and t in indeg:
            outs[s].append(t)
            indeg[t] += 1

    depths: Dict[str, int] = {nid: 0 for nid in node_ids}
    queue = [nid for nid in node_ids if indeg[nid] == 0]

    for n in nodes:
        nid = str(n.get("id"))
        if _contains_token(n.get("component_type", ""), "input"):
            depths[nid] = 0

    q_idx = 0
    while q_idx < len(queue):
        u = queue[q_idx]
        q_idx += 1
        for v in outs.get(u, []):
            depths[v] = max(depths.get(v, 0), depths.get(u, 0) + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    max_depth = max(depths.values()) if depths else 0
    for n in nodes:
        nid = str(n.get("id"))
        if _contains_token(n.get("component_type", ""), "output"):
            depths[nid] = max_depth + 1

    for n in nodes:
        nid = str(n.get("id"))
        hint = (insertion_hints or {}).get(nid)
        if isinstance(hint, dict):
            after_id = hint.get("after_node_id")
            if after_id and after_id in depths:
                depths[nid] = depths[after_id] + 1

    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for n in nodes:
        nid = str(n.get("id"))
        d = int(depths.get(nid, 0))
        by_depth.setdefault(d, []).append(n)

    for group in by_depth.values():

        def _y(node: Dict[str, Any]) -> float:
            pos = (node.get("ui_meta") or {}).get("position") or {}
            try:
                return float(pos.get("y", 0))
            except Exception:
                logger.debug(
                    "Failed to parse node y-position, defaulting to 0.0", exc_info=True
                )
                return 0.0

        group.sort(key=_y)

    x_step = 260
    y_step = 140
    x0 = 90
    y0 = 120

    for d, group in sorted(by_depth.items(), key=lambda kv: kv[0]):
        for idx, n in enumerate(group):
            ui_meta = n.setdefault("ui_meta", {})
            ui_meta["position"] = {"x": x0 + d * x_step, "y": y0 + idx * y_step}


def postprocess_patched_workflow(
    workflow: Dict[str, Any],
    added_node_ids: List[str],
    insertion_hints: Optional[Dict[str, Dict[str, str | None]]] = None,
) -> Dict[str, Any]:
    if not isinstance(workflow, dict):
        return workflow
    _auto_connect_added_nodes(workflow, added_node_ids, insertion_hints=insertion_hints)
    _auto_layout_workflow(workflow, insertion_hints=insertion_hints)
    return workflow
