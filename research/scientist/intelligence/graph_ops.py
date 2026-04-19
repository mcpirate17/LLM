from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List

from ..native.core import _try_import_rust_scheduler

logger = logging.getLogger(__name__)


def _extract_unique_graph_ops_python(graph_json: Any) -> List[str]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(graph_json, dict):
        return []
    nodes = graph_json.get("nodes") or {}
    ops = {
        node.get("op_name", "")
        for node in nodes.values()
        if node.get("op_name", "") and node.get("op_name", "") != "input"
    }
    return sorted(ops)


def extract_unique_graph_ops_batch(graph_payloads: Iterable[Any]) -> List[List[str]]:
    payloads = list(graph_payloads)
    if not payloads:
        return []

    serialized: List[str] = []
    for payload in payloads:
        if isinstance(payload, str):
            serialized.append(payload)
        else:
            try:
                serialized.append(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                )
            except (TypeError, ValueError):
                serialized.append("")

    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, "extract_graph_ops_batch"):
        result = rust.extract_graph_ops_batch(serialized)
        if isinstance(result, list):
            return [
                sorted(
                    {
                        str(op).strip()
                        for op in ops
                        if str(op).strip() and str(op).strip() != "input"
                    }
                )
                if isinstance(ops, list)
                else []
                for ops in result
            ]

    return [_extract_unique_graph_ops_python(payload) for payload in payloads]
