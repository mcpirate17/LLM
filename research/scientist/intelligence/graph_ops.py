from __future__ import annotations

import json
from typing import Any, Iterable, List


def _normalize_extracted_ops(ops: Any, *, unique: bool) -> List[str]:
    if not isinstance(ops, list):
        return []
    cleaned = [
        str(op).strip() for op in ops if str(op).strip() and str(op).strip() != "input"
    ]
    if unique:
        return sorted(set(cleaned))
    return cleaned


def _iter_graph_nodes(graph_json: Any) -> Iterable[Any]:
    if not isinstance(graph_json, dict):
        return ()
    nodes = graph_json.get("nodes") or {}
    if isinstance(nodes, dict):
        return (node for node in nodes.values() if isinstance(node, (dict, str)))
    if isinstance(nodes, list):
        return (node for node in nodes if isinstance(node, (dict, str)))
    return ()


def _extract_graph_ops(graph_json: Any, *, unique: bool) -> List[str]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return []

    ops: List[str] = []
    for node in _iter_graph_nodes(graph_json):
        if isinstance(node, dict):
            op = node.get("op_name") or node.get("op_type") or node.get("op") or ""
        else:
            op = node
        op_name = str(op).strip()
        if op_name and op_name != "input":
            ops.append(op_name)
    return _normalize_extracted_ops(ops, unique=unique)


def extract_graph_ops(graph_payload: Any) -> List[str]:
    """Return graph op names with multiplicity preserved when possible."""
    return _extract_graph_ops(graph_payload, unique=False)


def extract_unique_graph_ops(graph_payload: Any) -> List[str]:
    return _extract_graph_ops(graph_payload, unique=True)


def extract_unique_graph_ops_batch(graph_payloads: Iterable[Any]) -> List[List[str]]:
    return [_extract_graph_ops(payload, unique=True) for payload in graph_payloads]
