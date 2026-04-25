from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List

from ..json_utils import fast_dumps as _json_dumps
from ..json_utils import fast_loads as _json_loads
from ..native.core import _try_import_rust_scheduler

_EMPTY_FEATURE_PAYLOAD: tuple[str, tuple[str, ...], tuple[str, ...], str, str, str] = (
    "",
    (),
    (),
    "[]",
    "[]",
    "[]",
)
_EMPTY_JSON_ARRAY = "[]"


def _extract_graph_feature_payload_native(
    graph_json: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str, str] | None:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_graph_feature_payload"):
        return None
    try:
        result = rust.extract_graph_feature_payload(graph_json)
    except Exception:
        return None
    if not isinstance(result, (list, tuple)) or len(result) != 6:
        return None
    (
        template_name,
        op_names,
        pair_signatures,
        templates_json,
        motifs_json,
        slot_usage,
    ) = result
    return (
        str(template_name or ""),
        tuple(str(op).strip() for op in (op_names or ()) if str(op).strip()),
        tuple(
            str(signature).strip()
            for signature in (pair_signatures or ())
            if str(signature).strip()
        ),
        str(templates_json or _EMPTY_JSON_ARRAY),
        str(motifs_json or _EMPTY_JSON_ARRAY),
        str(slot_usage or _EMPTY_JSON_ARRAY),
    )


def _node_lookup_key(raw_value: Any) -> str:
    return raw_value if isinstance(raw_value, str) else str(raw_value)


def _iter_graph_nodes(
    nodes: Any,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(nodes, dict):
        node_map = {
            str(key): value for key, value in nodes.items() if isinstance(value, dict)
        }
        return node_map, list(node_map.values())
    if isinstance(nodes, list):
        node_map = {
            str(node.get("id", index)): node
            for index, node in enumerate(nodes)
            if isinstance(node, dict)
        }
        return node_map, list(node_map.values())
    return {}, []


def _collect_graph_ops_and_pairs(
    node_map: Dict[str, Dict[str, Any]],
    node_values: List[Dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    op_names: set[str] = set()
    pair_signatures: set[str] = set()
    for node in node_values:
        op_name = str(node.get("op_name") or node.get("op") or "").strip()
        if not op_name or op_name == "input":
            continue
        op_names.add(op_name)
        input_ids = node.get("input_ids")
        if not isinstance(input_ids, (list, tuple)):
            continue
        for raw_parent in input_ids:
            parent = node_map.get(_node_lookup_key(raw_parent))
            if not isinstance(parent, dict):
                continue
            parent_op = str(parent.get("op_name") or parent.get("op") or "").strip()
            if parent_op and parent_op != "input":
                pair_signatures.add(f"{parent_op}->{op_name}")
    return tuple(sorted(op_names)), tuple(sorted(pair_signatures))


def _clean_string_array_json(payload: Any) -> str:
    if not isinstance(payload, list):
        return _EMPTY_JSON_ARRAY
    cleaned = [
        item if isinstance(item, str) else str(item)
        for item in payload
        if item is not None
    ]
    return _json_dumps(cleaned) if cleaned else _EMPTY_JSON_ARRAY


def _clean_dict_array_json(payload: Any) -> str:
    if not isinstance(payload, list):
        return _EMPTY_JSON_ARRAY
    cleaned = [item for item in payload if isinstance(item, dict)]
    return _json_dumps(cleaned) if cleaned else _EMPTY_JSON_ARRAY


def _extract_graph_metadata_payload(
    metadata: Dict[str, Any],
) -> tuple[str, str, str, str]:
    template_name = metadata.get("template") or metadata.get("template_name") or ""
    if not isinstance(template_name, str):
        template_name = str(template_name)
    template_name = template_name.strip()
    templates_used = metadata.get("templates_used")
    if not template_name and isinstance(templates_used, list):
        for candidate in templates_used:
            if isinstance(candidate, str) and candidate.strip():
                template_name = candidate.strip()
                break
    return (
        template_name,
        _clean_string_array_json(templates_used),
        _clean_string_array_json(metadata.get("motifs_used")),
        _clean_dict_array_json(metadata.get("template_slot_usage")),
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

    metadata = graph.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    node_map, node_values = _iter_graph_nodes(graph.get("nodes"))
    op_names, pair_signatures = _collect_graph_ops_and_pairs(node_map, node_values)
    template_name, templates_json, motifs_json, slot_usage_json = (
        _extract_graph_metadata_payload(metadata)
    )
    return (
        template_name,
        op_names,
        pair_signatures,
        templates_json,
        motifs_json,
        slot_usage_json,
    )


def _first_template_from_json(templates_json: str) -> str:
    if not templates_json or templates_json == _EMPTY_JSON_ARRAY:
        return ""
    try:
        names = _json_loads(templates_json)
    except Exception:
        return ""
    if not isinstance(names, list):
        return ""
    for candidate in names:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


@lru_cache(maxsize=8192)
def extract_graph_feature_payload(
    graph_json: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str, str]:
    native_payload = _extract_graph_feature_payload_native(graph_json)
    payload = (
        native_payload
        if native_payload is not None
        else _extract_graph_feature_payload_python(graph_json)
    )
    (
        template_name,
        op_names,
        pair_signatures,
        templates_json,
        motifs_json,
        slot_usage_json,
    ) = payload
    if not template_name:
        template_name = _first_template_from_json(templates_json)
    return (
        template_name,
        op_names,
        pair_signatures,
        templates_json,
        motifs_json,
        slot_usage_json,
    )


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
