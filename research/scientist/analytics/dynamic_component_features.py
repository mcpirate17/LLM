from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_BRANCH_LOWERING = "trunk_sidecar_merge_v1"


def _metadata_list(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata.get(key)
    return value if isinstance(value, list) else []


def _dynamic_component_records(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    current = [
        item
        for item in _metadata_list(metadata, "dynamic_components_used")
        if isinstance(item, Mapping)
    ]
    if current:
        return current
    return [
        item
        for item in _metadata_list(metadata, "dynamic_templates_used")
        if isinstance(item, Mapping)
        and isinstance(item.get("component_descriptor"), Mapping)
    ]


def _dynamic_component_tokens(records: list[Mapping[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for record in records:
        descriptor = record.get("component_descriptor")
        if not isinstance(descriptor, Mapping):
            descriptor = {}
        component_id = str(
            record.get("component_id") or descriptor.get("component_id") or ""
        )
        lowering = str(record.get("lowering") or descriptor.get("lowering") or "")
        if component_id:
            tokens.append(component_id)
        if lowering:
            tokens.append(f"lowering:{lowering}")
    return tokens


def dynamic_component_feature_summary(
    metadata: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    dynamic_templates = _metadata_list(metadata, "dynamic_templates_used")
    dynamic_components = _dynamic_component_records(metadata)
    tokens = _dynamic_component_tokens(dynamic_components)
    flags = {
        "pattern_dynamic_template": int(bool(dynamic_templates)),
        "pattern_dynamic_component": int(bool(dynamic_components)),
        "pattern_dynamic_branch_component": int(
            any(token == f"lowering:{_BRANCH_LOWERING}" for token in tokens)
        ),
    }
    return tokens, flags
