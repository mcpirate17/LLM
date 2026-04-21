"""
Direct component catalog helpers shared by research and aria_designer.

This module is the authoritative reader for aria_designer/runtime/component_mapping.yaml.
Hot-path callers should use these functions directly instead of routing through the
older singleton registry compatibility layer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import yaml

from .primitives import canonicalize_op_name

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_MAPPING_FILE = _PROJECT_ROOT / "aria_designer" / "runtime" / "component_mapping.yaml"
IO_COMPONENTS = frozenset(
    {"graph_input", "graph_output", "input", "output", "output_head"}
)


def component_leaf(component_type: str) -> str:
    token = str(component_type or "").strip().lower()
    if not token:
        return ""
    return token.split("/")[-1]


def component_category(component_type: str) -> str:
    token = str(component_type or "").strip().lower()
    if not token or "/" not in token:
        return ""
    return token.split("/", 1)[0]


@lru_cache(maxsize=None)
def _load_mapping_file(mapping_path: str) -> dict[str, Any]:
    path = Path(mapping_path)
    if not path.exists():
        warnings.warn(
            f"component_mapping.yaml not found at {path}; component mappings unavailable",
            stacklevel=2,
        )
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        warnings.warn(
            f"failed to load component mappings from {path}; component mappings unavailable",
            stacklevel=2,
        )
        return {}


def load_component_mapping(mapping_file: Path | None = None) -> dict[str, Any]:
    path = str((mapping_file or _MAPPING_FILE).resolve())
    return _load_mapping_file(path)


def category_execution_class(
    component_type: str, default: str = "primitive_candidate"
) -> str:
    config = load_component_mapping()
    category = (
        component_category(component_type) or str(component_type or "").strip().lower()
    )
    return str(config.get("category_execution_class", {}).get(category, default))


def component_execution_class(component_type: str, default: str = "") -> str:
    config = load_component_mapping()
    return str(
        config.get("component_execution_class", {}).get(
            component_leaf(component_type), default
        )
    )


@lru_cache(maxsize=1)
def passthrough_components() -> frozenset[str]:
    config = load_component_mapping()
    return frozenset(config.get("passthrough_components", []))


@lru_cache(maxsize=1)
def source_components() -> frozenset[str]:
    config = load_component_mapping()
    return frozenset(config.get("source_components", []))


@lru_cache(maxsize=1)
def template_lowered_components() -> frozenset[str]:
    config = load_component_mapping()
    return frozenset(config.get("template_lowered_components", []))


def is_passthrough_component(component_type: str) -> bool:
    return component_leaf(component_type) in passthrough_components()


def is_source_component(component_type: str) -> bool:
    return component_leaf(component_type) in source_components()


def is_template_lowered_component(component_type: str) -> bool:
    return component_leaf(component_type) in template_lowered_components()


def get_primitive_name(component_type: str) -> str:
    leaf = component_leaf(component_type)
    if not leaf:
        return "identity"
    return canonicalize_op_name(leaf)
