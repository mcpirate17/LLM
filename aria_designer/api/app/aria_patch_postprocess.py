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
        node = nodes_by_id.get(str(new_id))
        if not node:
            continue
        ctype = str(node.get("component_type", ""))
        if _contains_token(ctype, "input") or _contains_token(ctype, "output"):
            continue

        incoming, outgoing, output_node, source_ids = _node_edge_indexes(
            nodes, edges, str(new_id)
        )

        if incoming and outgoing:
            continue

        hint = (insertion_hints or {}).get(str(new_id))
        if isinstance(hint, dict) and not incoming and not outgoing:
            after_id = hint.get("after_node_id")
            before_id = hint.get("before_node_id")
            if after_id and after_id in nodes_by_id:
                if before_id and before_id in nodes_by_id:
                    _remove_edge_once(edges, str(after_id), str(before_id))
                    _append_edge(edges, str(after_id), str(new_id))
                    _append_edge(edges, str(new_id), str(before_id))
                    continue
                _append_edge(edges, str(after_id), str(new_id))
                if output_node:
                    _append_edge(edges, str(new_id), str(output_node.get("id", "")))
                continue
            if before_id and before_id in nodes_by_id:
                _append_edge(edges, str(new_id), str(before_id))
                continue

        if (not incoming) and outgoing:
            out_to_output = [
                e
                for e in outgoing
                if output_node and e.get("target") == output_node.get("id")
            ]
            if out_to_output and output_node:
                in_to_output = [
                    e
                    for e in edges
                    if e.get("target") == output_node.get("id")
                    and e.get("source") != new_id
                ]
                if in_to_output:
                    old = in_to_output[-1]
                    try:
                        edges.remove(old)
                    except ValueError:
                        pass
                    _append_edge(
                        edges,
                        str(old.get("source", "")),
                        str(new_id),
                        str(old.get("source_port") or "out"),
                        "in",
                    )
                    continue

                sink = _last_non_new_sink(
                    nodes, source_ids, {str(new_id), str(output_node.get("id"))}
                )
                if sink:
                    _append_edge(edges, str(sink.get("id", "")), str(new_id))
                    continue

        if incoming and (not outgoing):
            src = str(incoming[-1].get("source", ""))
            if output_node:
                while _remove_edge_once(edges, src, str(output_node.get("id", ""))):
                    pass
                _append_edge(
                    edges, str(new_id), str(output_node.get("id", "")), "out", "in"
                )
                continue

        if output_node:
            inc_to_output = [
                e for e in edges if e.get("target") == output_node.get("id")
            ]
            if inc_to_output:
                old = inc_to_output[-1]
                try:
                    edges.remove(old)
                except ValueError:
                    pass
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
                continue

        sink = _last_non_new_sink(nodes, source_ids, {str(new_id)})
        if sink:
            _append_edge(edges, str(sink.get("id", "")), str(new_id))
            continue

        inputs = [
            n
            for n in nodes
            if _contains_token(n.get("component_type", ""), "input")
            and n.get("id") != new_id
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
