"""
Component Registry: Single source of truth for component type mappings.
Maps frontend component types to backend primitive names (leaf extraction).
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Configuration file location
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_MAPPING_FILE = _PROJECT_ROOT / "aria_designer" / "runtime" / "component_mapping.yaml"


class ComponentRegistry:
    """Registry for mapping frontend components to backend primitives."""

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
        """Load mapping configuration from YAML."""
        if not self.mapping_file.exists():
            self._set_defaults()
            return

        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        except Exception:
            self._set_defaults()
            return

        self.category_execution_class = self.config.get("category_execution_class", {})
        self.component_execution_class = self.config.get(
            "component_execution_class", {}
        )
        self.passthrough_components = set(self.config.get("passthrough_components", []))
        self.source_components = set(self.config.get("source_components", []))
        self.template_lowered_components = set(
            self.config.get("template_lowered_components", [])
        )

    def _set_defaults(self):
        """Minimal fallback — YAML should always be present at runtime."""
        import warnings

        warnings.warn(
            f"component_mapping.yaml not found at {self.mapping_file}; "
            "component mappings will be unavailable",
            stacklevel=2,
        )

    def get_primitive_name(self, component_type: str) -> str:
        """Map component type to primitive name (leaf extraction)."""
        if not component_type:
            return "identity"
        return component_type.split("/")[-1]

    def is_passthrough(self, component_type: str) -> bool:
        """Check if component is a passthrough (identity)."""
        leaf_id = component_type.split("/")[-1]
        return leaf_id in self.passthrough_components

    def is_source(self, component_type: str) -> bool:
        """Check if component is a data source."""
        leaf_id = component_type.split("/")[-1]
        return leaf_id in self.source_components


# Global instance
registry = ComponentRegistry()


def fe_type_to_op_name(fe_type: str) -> str:
    """Compatibility helper for existing code."""
    return registry.get_primitive_name(fe_type)
