"""Shared parsing utilities for the eval pipeline.

Extracts common patterns duplicated across analytics.py, notebook.py, etc.
"""
import json
import math
from typing import Any, Optional


def safe_json_load(value: Any) -> Any:
    """Parse JSON string, returning None on failure. Passes through non-strings."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_parse_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None for None/NaN/Inf/unparseable."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def compute_opcode_histogram(graph_json: str) -> dict[str, int]:
    """Build op_name → count histogram from a graph JSON string.

    Returns empty dict on parse failure.
    """
    data = safe_json_load(graph_json)
    if not isinstance(data, dict):
        return {}
    nodes = data.get("nodes", {})
    if isinstance(nodes, dict):
        items = nodes.values()
    elif isinstance(nodes, list):
        items = nodes
    else:
        return {}
    hist: dict[str, int] = {}
    for node in items:
        if isinstance(node, dict):
            op = node.get("op_name") or node.get("op") or node.get("type")
            if op:
                hist[op] = hist.get(op, 0) + 1
    return hist
