from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from ..json_utils import fast_dumps as _json_dumps
from ..json_utils import fast_loads as _json_loads

_EMPTY_FEATURE_PAYLOAD: tuple[str, tuple[str, ...], tuple[str, ...], str, str, str] = (
    "",
    (),
    (),
    "[]",
    "[]",
    "[]",
)


def _extract_graph_feature_payload_python(
    graph_json: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str, str]:
    """Python fallback for notebook graph features.

    This function is the only write-time extraction boundary that still needs a
    native implementation if notebook ingest volume becomes dominant.
    """
    if not isinstance(graph_json, str) or not graph_json.strip():
        return _EMPTY_FEATURE_PAYLOAD
    try:
        graph = _json_loads(graph_json)
    except Exception:
        return _EMPTY_FEATURE_PAYLOAD
    if not isinstance(graph, dict):
        return _EMPTY_FEATURE_PAYLOAD

    metadata = graph.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    nodes = graph.get("nodes", {})
    if isinstance(nodes, dict):
        node_map = {
            str(key): value for key, value in nodes.items() if isinstance(value, dict)
        }
        node_items = tuple(node_map.items())
    elif isinstance(nodes, list):
        node_map = {}
        node_items_list: List[Tuple[str, Dict[str, Any]]] = []
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", index))
            node_map[node_id] = node
            node_items_list.append((node_id, node))
        node_items = tuple(node_items_list)
    else:
        node_map = {}
        node_items = ()

    op_names: set[str] = set()
    pair_signatures: set[str] = set()
    for _, node in node_items:
        op_name = str(node.get("op_name") or "").strip()
        if not op_name or op_name == "input":
            continue
        op_names.add(op_name)
        input_ids = node.get("input_ids", ())
        if not isinstance(input_ids, (list, tuple)):
            continue
        for raw_parent in input_ids:
            parent = node_map.get(str(raw_parent))
            if not isinstance(parent, dict):
                continue
            parent_op = str(parent.get("op_name") or "").strip()
            if parent_op and parent_op != "input":
                pair_signatures.add(f"{parent_op}->{op_name}")

    template_name = str(
        metadata.get("template") or metadata.get("template_name") or ""
    ).strip()
    templates = metadata.get("templates_used")
    motifs = metadata.get("motifs_used")
    slot_usage = metadata.get("template_slot_usage")

    templates_json = _json_dumps(
        [str(item) for item in templates if item is not None]
        if isinstance(templates, list)
        else []
    )
    motifs_json = _json_dumps(
        [str(item) for item in motifs if item is not None]
        if isinstance(motifs, list)
        else []
    )
    slot_usage_json = _json_dumps(
        [item for item in slot_usage if isinstance(item, dict)]
        if isinstance(slot_usage, list)
        else []
    )
    return (
        template_name,
        tuple(sorted(op_names)),
        tuple(sorted(pair_signatures)),
        templates_json,
        motifs_json,
        slot_usage_json,
    )


@lru_cache(maxsize=8192)
def extract_graph_feature_payload(
    graph_json: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str, str]:
    return _extract_graph_feature_payload_python(graph_json)


def build_graph_feature_rows(
    *,
    result_id: str,
    graph_fingerprint: str,
    graph_json: str,
) -> dict[str, Any]:
    (
        template_name,
        op_names,
        pair_signatures,
        templates_json,
        motifs_json,
        slot_usage_json,
    ) = extract_graph_feature_payload(graph_json)
    created_at = time.time()
    return {
        "feature_row": (
            result_id,
            graph_fingerprint,
            template_name,
            templates_json,
            motifs_json,
            slot_usage_json,
            len(op_names),
            len(pair_signatures),
            created_at,
        ),
        "op_rows": [(result_id, graph_fingerprint, op_name) for op_name in op_names],
        "pair_rows": [
            (result_id, graph_fingerprint, signature) for signature in pair_signatures
        ],
    }
