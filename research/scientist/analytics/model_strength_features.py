"""Graph/config feature extraction for model-strength analysis.

Pure-Python helpers that turn a program result's ``config_json`` and
``graph_json`` payloads into flat feature dicts. Split out of
``model_strength`` to keep that module under the god-file limit; behaviour is
unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..json_utils import fast_loads as _json_loads
from .dynamic_component_features import dynamic_component_feature_summary


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = _json_loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _entropy_from_weights(weights: dict[str, Any]) -> float:
    vals = [
        max(float(value), 0.0)
        for value in weights.values()
        if isinstance(value, (int, float))
    ]
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    probs = [value / total for value in vals if value > 0.0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def parse_config_features(config_json: Any) -> dict[str, Any]:
    config = _safe_json_loads(config_json)
    out: dict[str, Any] = {
        "cfg_stage1_steps": config.get("stage1_steps"),
        "cfg_stage1_batch_size": config.get("stage1_batch_size"),
        "cfg_stage1_lr": config.get("stage1_lr"),
        "cfg_model_dim": config.get("model_dim"),
        "cfg_n_layers": config.get("n_layers"),
        "cfg_n_programs": config.get("n_programs"),
        "cfg_graphs_weighted": config.get("n_graphs_weighted"),
    }
    for prefix in ("category_weights", "op_weights", "template_weights"):
        weights = config.get(prefix)
        if not isinstance(weights, dict):
            continue
        out[f"{prefix}_entropy"] = _entropy_from_weights(weights)
        numeric_items = {
            str(key): float(value)
            for key, value in weights.items()
            if isinstance(value, (int, float))
        }
        if numeric_items:
            out[f"{prefix}_max_weight"] = max(numeric_items.values())
            out[f"{prefix}_min_weight"] = min(numeric_items.values())
        for key, value in numeric_items.items():
            out[f"{prefix}::{key}"] = value
    return out


def _iter_nodes(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nodes = graph.get("nodes") or {}
    if isinstance(nodes, dict):
        return [
            (str(node_id), node)
            for node_id, node in nodes.items()
            if isinstance(node, dict)
        ]
    if isinstance(nodes, list):
        return [
            (str(node.get("id", idx)), node)
            for idx, node in enumerate(nodes)
            if isinstance(node, dict)
        ]
    return []


def _graph_topology_features(
    nodes: list[tuple[str, dict[str, Any]]],
) -> tuple[list[str], list[str], list[str]]:
    by_id = {node_id: node for node_id, node in nodes}
    children = {node_id: [] for node_id, _ in nodes}
    indegree = {node_id: 0 for node_id, _ in nodes}
    depths = {node_id: 0 for node_id, _ in nodes}
    ops: list[str] = []
    depth_ops: list[tuple[str, str]] = []
    pairs: set[str] = set()
    for node_id, node in nodes:
        op = str(node.get("op_name") or "").strip()
        inputs = node.get("input_ids") or []
        if isinstance(inputs, list):
            valid_parents: list[str] = []
            for parent in inputs:
                parent_id = str(parent)
                if parent_id not in by_id:
                    continue
                valid_parents.append(parent_id)
                children[parent_id].append(node_id)
            indegree[node_id] = len(valid_parents)
            if op and op not in {"input", "output"}:
                for parent_id in valid_parents:
                    parent_node = by_id.get(parent_id) or {}
                    parent_op = str(parent_node.get("op_name") or "").strip()
                    if parent_op and parent_op not in {"input", "output"}:
                        a, b = sorted((parent_op, op))
                        pairs.add(f"{a}+{b}")
        if not op or op in {"input", "output"}:
            continue
        ops.append(op)
        depth_ops.append((node_id, op))
    op_depth_buckets = _op_depth_buckets(depth_ops, children, indegree, depths)
    return sorted(set(ops)), sorted(pairs), sorted(op_depth_buckets)


def _op_depth_buckets(
    depth_ops: list[tuple[str, str]],
    children: dict[str, list[str]],
    indegree: dict[str, int],
    depths: dict[str, int],
) -> set[str]:
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    while queue:
        node_id = queue.popleft()
        next_depth = depths[node_id] + 1
        for child_id in children.get(node_id, ()):
            if next_depth > depths.get(child_id, 0):
                depths[child_id] = next_depth
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)

    max_depth = max(depths.values(), default=0)
    op_depth_buckets: set[str] = set()
    for node_id, op in depth_ops:
        depth = depths.get(node_id, 0)
        if max_depth <= 1:
            bucket = "middle"
        else:
            rel = depth / max(max_depth, 1)
            if rel <= 0.33:
                bucket = "early"
            elif rel <= 0.66:
                bucket = "middle"
            else:
                bucket = "late"
        op_depth_buckets.add(f"{bucket}:{op}")
    return op_depth_buckets


def _metadata_sequence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _slot_usage_features(
    metadata: dict[str, Any],
    primary_template: str,
) -> tuple[list[str], list[str], list[str]]:
    slot_entries = metadata.get("template_slot_usage") or []
    slot_keys: list[str] = []
    slot_motifs: list[str] = []
    slot_components: list[str] = []
    if not isinstance(slot_entries, list):
        return slot_keys, slot_motifs, slot_components
    for slot in slot_entries:
        if not isinstance(slot, dict):
            continue
        slot_key = str(
            slot.get("slot_key")
            or f"{slot.get('template_name', primary_template or 'unknown')}.slot{slot.get('slot_index', 0)}"
        )
        slot_keys.append(slot_key)
        selected_motif = str(slot.get("selected_motif") or "").strip()
        selected_class = str(slot.get("selected_motif_class") or "").strip()
        if selected_motif:
            slot_motifs.append(selected_motif)
            slot_components.append(f"{slot_key}:{selected_motif}")
        elif selected_class:
            slot_components.append(f"{slot_key}:{selected_class}")
    return slot_keys, slot_motifs, slot_components


def _pattern_feature_flags(
    *,
    unique_ops: list[str],
    templates: list[str],
    slot_keys: list[str],
    dynamic_feature_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pattern_has_attention": int(any("attention" in op for op in unique_ops)),
        "pattern_has_moe": int(any("moe" in op or "router" in op for op in unique_ops)),
        "pattern_has_routing": int(
            any(
                token in op
                for op in unique_ops
                for token in ("route", "gate", "router")
            )
        ),
        "pattern_has_ssm": int(
            any(
                token in op
                for op in unique_ops
                for token in ("scan", "state_space", "rwkv")
            )
        ),
        "pattern_has_math_space": int(
            any(
                token in op
                for op in unique_ops
                for token in ("tropical", "padic", "clifford", "hyp_")
            )
        ),
        "pattern_has_residual": int("add" in unique_ops),
        "pattern_has_norm": int(any("norm" in op for op in unique_ops)),
        "pattern_multi_template": int(len(templates) > 1),
        "pattern_slot_telemetry": int(bool(slot_keys)),
        **dynamic_feature_flags,
    }


def _metadata_features(
    metadata: dict[str, Any], unique_ops: list[str]
) -> dict[str, Any]:
    templates = metadata.get("templates_used") or []
    motifs = metadata.get("motifs_used") or []
    dynamic_component_tokens, dynamic_feature_flags = dynamic_component_feature_summary(
        metadata
    )
    primary_template = str(
        metadata.get("primary_template")
        or (templates[0] if isinstance(templates, list) and templates else "")
    )
    template_names = _metadata_sequence(templates)
    motif_names = _metadata_sequence(motifs)
    slot_keys, slot_motifs, slot_components = _slot_usage_features(
        metadata,
        primary_template,
    )
    feature_flags = _pattern_feature_flags(
        unique_ops=unique_ops,
        templates=template_names,
        slot_keys=slot_keys,
        dynamic_feature_flags=dynamic_feature_flags,
    )
    return {
        "primary_template": primary_template,
        "templates_used": template_names,
        "motifs_used": motif_names,
        "dynamic_components": dynamic_component_tokens,
        "slot_keys": slot_keys,
        "slot_motifs": slot_motifs,
        "slot_components": slot_components,
        **feature_flags,
    }


def graph_features(graph_json: Any) -> dict[str, Any]:
    graph = _safe_json_loads(graph_json)
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    unique_ops, op_pairs, depth_ops = _graph_topology_features(_iter_nodes(graph))
    return {
        **_metadata_features(metadata, unique_ops),
        "ops": unique_ops,
        "op_pairs": op_pairs,
        "depth_ops": depth_ops,
    }
