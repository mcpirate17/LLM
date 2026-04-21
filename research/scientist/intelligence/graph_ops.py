from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, List

from ..native.core import _try_import_rust_scheduler

logger = logging.getLogger(__name__)

_OP_NAME_PATTERN = re.compile(r'"op_name"\s*:\s*"([^"]+)"')
_OP_TYPE_PATTERN = re.compile(r'"op_type"\s*:\s*"([^"]+)"')
_OP_KEY_PATTERN = re.compile(r'"op"\s*:\s*"([^"]+)"')
_NODES_LIST_PATTERN = re.compile(r'"nodes"\s*:\s*\[')


def _normalize_extracted_ops(ops: Any, *, unique: bool) -> List[str]:
    if not isinstance(ops, list):
        return []
    cleaned = [
        str(op).strip() for op in ops if str(op).strip() and str(op).strip() != "input"
    ]
    if unique:
        return sorted(set(cleaned))
    return cleaned


def _iter_graph_nodes(graph_json: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(graph_json, dict):
        return ()
    nodes = graph_json.get("nodes") or {}
    if isinstance(nodes, dict):
        return (node for node in nodes.values() if isinstance(node, dict))
    if isinstance(nodes, list):
        return (node for node in nodes if isinstance(node, dict))
    return ()


def _payload_requires_python_fallback(payload: Any) -> bool:
    if isinstance(payload, str):
        return bool(_NODES_LIST_PATTERN.search(payload) or '"op_type"' in payload)
    if not isinstance(payload, dict):
        return True
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        return True
    if not isinstance(nodes, dict):
        return True
    return any(
        isinstance(node, dict) and node.get("op_type") for node in nodes.values()
    )


def _extract_graph_ops_python(graph_json: Any, *, unique: bool) -> List[str]:
    if isinstance(graph_json, str):
        if (
            '"op_name"' not in graph_json
            and '"op_type"' not in graph_json
            and '"op"' not in graph_json
        ):
            return []
        ops = [
            *[
                op
                for op in _OP_NAME_PATTERN.findall(graph_json)
                if op and op != "input"
            ],
            *[
                op
                for op in _OP_TYPE_PATTERN.findall(graph_json)
                if op and op != "input"
            ],
            *[op for op in _OP_KEY_PATTERN.findall(graph_json) if op and op != "input"],
        ]
        if ops:
            return _normalize_extracted_ops(ops, unique=unique)
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return []

    ops = [
        str(node.get("op_name") or node.get("op_type") or node.get("op") or "").strip()
        for node in _iter_graph_nodes(graph_json)
        if str(
            node.get("op_name") or node.get("op_type") or node.get("op") or ""
        ).strip()
        and str(
            node.get("op_name") or node.get("op_type") or node.get("op") or ""
        ).strip()
        != "input"
    ]
    return _normalize_extracted_ops(ops, unique=unique)


def extract_graph_ops(graph_payload: Any) -> List[str]:
    """Return graph op names with multiplicity preserved when possible."""
    return _extract_graph_ops_python(graph_payload, unique=False)


def _extract_unique_graph_ops_python(graph_json: Any) -> List[str]:
    return _extract_graph_ops_python(graph_json, unique=True)


def extract_unique_graph_ops(graph_payload: Any) -> List[str]:
    return _extract_graph_ops_python(graph_payload, unique=True)


def extract_unique_graph_ops_batch(graph_payloads: Iterable[Any]) -> List[List[str]]:
    payloads = list(graph_payloads)
    if not payloads:
        return []

    results: List[List[str] | None] = [None] * len(payloads)
    native_serialized: List[str] = []
    native_indexes: List[int] = []

    for index, payload in enumerate(payloads):
        if _payload_requires_python_fallback(payload):
            results[index] = _extract_graph_ops_python(payload, unique=True)
            continue
        if isinstance(payload, str):
            native_serialized.append(payload)
            native_indexes.append(index)
            continue
        try:
            native_serialized.append(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            native_indexes.append(index)
        except (TypeError, ValueError):
            results[index] = []

    rust = _try_import_rust_scheduler()
    if (
        native_serialized
        and rust is not None
        and hasattr(rust, "extract_graph_ops_batch")
    ):
        result = rust.extract_graph_ops_batch(native_serialized)
        if isinstance(result, list):
            for index, ops in zip(native_indexes, result):
                results[index] = _normalize_extracted_ops(ops, unique=True)

    for index, payload in enumerate(payloads):
        if results[index] is None:
            results[index] = _extract_graph_ops_python(payload, unique=True)

    return [ops or [] for ops in results]
