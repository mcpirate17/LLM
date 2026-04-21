"""
Compatibility shim for the older singleton component-registry API.

Hot-path callers should import from ``research.synthesis.component_catalog``
directly. This module remains only to preserve existing imports while the rest
of the tree is migrated away from the registry facade.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .component_catalog import (
    get_primitive_name,
    is_passthrough_component,
    is_source_component,
    load_component_mapping,
    passthrough_components,
    source_components,
    template_lowered_components,
)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_MAPPING_FILE = _PROJECT_ROOT / "aria_designer" / "runtime" / "component_mapping.yaml"


class ComponentRegistry:
    """Legacy facade over the direct component catalog helpers."""

    def __init__(self, mapping_file: Optional[Path] = None):
        self.mapping_file = mapping_file or _MAPPING_FILE
        self.config: Dict[str, Any] = {}
        self.category_execution_class: Dict[str, str] = {}
        self.component_execution_class: Dict[str, str] = {}
        self.passthrough_components: Set[str] = set()
        self.source_components: Set[str] = set()
        self.template_lowered_components: Set[str] = set()

        self.load()

    def load(self):
        self.config = load_component_mapping(self.mapping_file)
        self.category_execution_class = dict(
            self.config.get("category_execution_class", {})
        )
        self.component_execution_class = dict(
            self.config.get("component_execution_class", {})
        )
        self.passthrough_components = set(passthrough_components())
        self.source_components = set(source_components())
        self.template_lowered_components = set(template_lowered_components())

    def get_primitive_name(self, component_type: str) -> str:
        return get_primitive_name(component_type)

    def is_passthrough(self, component_type: str) -> bool:
        return is_passthrough_component(component_type)

    def is_source(self, component_type: str) -> bool:
        return is_source_component(component_type)


# Global instance
registry = ComponentRegistry()


def fe_type_to_op_name(fe_type: str) -> str:
    """Compatibility helper for existing code."""
    return get_primitive_name(fe_type)
