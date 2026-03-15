"""
Unified Component Registry for aria_designer runtime.
Consolidates logic from compiler.py, bridge.py, and research/ component_registry.
"""

import logging
import os
import yaml
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_COMPONENTS_ROOT = _HERE.parent / "components"
_MAPPING_FILE = _HERE / "component_mapping.yaml"

class ComponentRegistry:
    def __init__(self, components_dir: Optional[Path] = None, mapping_file: Optional[Path] = None):
        self.components_dir = components_dir or _COMPONENTS_ROOT
        self.mapping_file = mapping_file or _MAPPING_FILE
        self.handlers = {}
        self.manifests = {}
        
        # Mapping config (from component_mapping.yaml)
        self.mapping_config = {}
        self.aliases = {}
        self.approximate_alias_notes = {}
        self.passthrough_components = set()
        self.source_components = set()
        self.template_lowered_components = set()
        self.category_execution_class = {}

        self.load_mappings()

    def load_mappings(self):
        """Load mapping configuration from YAML."""
        if not self.mapping_file.exists():
            return

        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                self.mapping_config = yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Failed to parse manifest YAML: %s", self.mapping_file, exc_info=True)
            return

        self.aliases = self.mapping_config.get("aliases", {})
        self.approximate_alias_notes = self.mapping_config.get("approximate_alias_notes", {})
        self.passthrough_components = set(self.mapping_config.get("passthrough_components", []))
        self.source_components = set(self.mapping_config.get("source_components", []))
        self.template_lowered_components = set(self.mapping_config.get("template_lowered_components", []))
        self.category_execution_class = self.mapping_config.get("category_execution_class", {})

    def _resolve_component_dir(self, component_type: str) -> Optional[Path]:
        """Resolve component type to its directory on disk."""
        # Check aliases first
        component_type = self.aliases.get(component_type, component_type)
        
        parts = component_type.split("/")
        if len(parts) == 2:
            cat, cid = parts
            component_dir = self.components_dir / cat / cid
            if component_dir.is_dir():
                return component_dir
            # Category mismatch — fall through to search by op name alone
            parts = [cid]

        # Search across categories if only ID provided or category was wrong
        cid = parts[0]
        if self.components_dir.exists():
            for category_dir in self.components_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                component_dir = category_dir / cid
                if component_dir.is_dir():
                    return component_dir
        return None

    def get_manifest(self, component_type: str) -> Optional[Dict[str, Any]]:
        if component_type in self.manifests:
            return self.manifests[component_type]

        component_dir = self._resolve_component_dir(component_type)
        if not component_dir:
            return None

        manifest_path = component_dir / "manifest.yaml"
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            self.manifests[component_type] = manifest
            return manifest
        except Exception:
            logger.warning("Failed to import component manifest: %s", component_type, exc_info=True)
            return None

    def get_handler(self, component_type: str):
        if component_type in self.handlers:
            return self.handlers[component_type]

        component_dir = self._resolve_component_dir(component_type)
        if not component_dir:
            return None

        path = component_dir / "kernel_fallback.py"
        if path.exists():
            return self._load_handler(component_type, path)
        return None

    def _load_handler(self, component_type: str, path: Path):
        cid = path.parent.name
        spec = importlib.util.spec_from_file_location(f"handler_{cid}", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        handler_class = getattr(module, "ComponentHandler", None)
        self.handlers[component_type] = handler_class
        return handler_class

    def get_primitive_name(self, component_type: str) -> str:
        """Map component type to research primitive name."""
        leaf_id = component_type.split("/")[-1]
        return self.aliases.get(leaf_id, leaf_id)

    def is_passthrough(self, component_type: str) -> bool:
        leaf_id = component_type.split("/")[-1]
        return leaf_id in self.passthrough_components

    def is_source(self, component_type: str) -> bool:
        leaf_id = component_type.split("/")[-1]
        return leaf_id in self.source_components

# Default instance
registry = ComponentRegistry()
