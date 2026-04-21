from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from aria_designer.component_identity import canonicalize_component_id


class RuntimeComponentCatalog:
    """Shared runtime catalog for manifest + fallback handler resolution.

    The runtime compiler only needs two lookups:
    - `get_manifest(component_type)`
    - `get_handler(component_type)`

    Keeping those in one module avoids a second private registry
    implementation inside `runtime/compiler.py`.
    """

    def __init__(self, components_dir: str | Path):
        self.components_dir = Path(components_dir)
        self._dir_cache: Dict[str, Optional[Path]] = {}
        self._manifest_cache: Dict[str, Dict[str, Any]] = {}
        self._handler_cache: Dict[str, Any] = {}

    def _resolve_component_dir(self, component_type: str) -> Optional[Path]:
        token = str(component_type or "")
        if token in self._dir_cache:
            return self._dir_cache[token]

        canonical = canonicalize_component_id(token)
        if canonical in self._dir_cache:
            resolved = self._dir_cache[canonical]
            self._dir_cache[token] = resolved
            return resolved

        resolved = self._scan_component_dir(canonical)
        self._dir_cache[token] = resolved
        self._dir_cache[canonical] = resolved
        return resolved

    def _scan_component_dir(self, component_type: str) -> Optional[Path]:
        parts = str(component_type or "").split("/")
        if len(parts) == 2:
            category, component_id = parts
            component_dir = self.components_dir / category / component_id
            if component_dir.is_dir():
                return component_dir
            parts = [component_id]

        if not parts or not parts[0]:
            return None

        component_id = parts[0]
        try:
            categories = list(self.components_dir.iterdir())
        except OSError:
            return None

        for category_dir in categories:
            if not category_dir.is_dir():
                continue
            component_dir = category_dir / component_id
            if component_dir.is_dir():
                return component_dir
        return None

    def get_manifest(self, component_type: str) -> Optional[Dict[str, Any]]:
        token = str(component_type or "")
        if token in self._manifest_cache:
            return self._manifest_cache[token]

        component_dir = self._resolve_component_dir(token)
        if component_dir is None:
            return None

        manifest_path = component_dir / "manifest.yaml"
        if not manifest_path.exists():
            return None

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        if not isinstance(manifest, dict):
            return None

        self._manifest_cache[token] = manifest
        return manifest

    def get_handler(self, component_type: str):
        token = str(component_type or "")
        if token in self._handler_cache:
            return self._handler_cache[token]

        component_dir = self._resolve_component_dir(token)
        if component_dir is None:
            return None

        path = component_dir / "kernel_fallback.py"
        if not path.exists():
            return None

        handler = self._load_handler(path)
        self._handler_cache[token] = handler
        return handler

    @staticmethod
    def _load_handler(path: Path):
        component_id = path.parent.name
        spec = importlib.util.spec_from_file_location(f"handler_{component_id}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load fallback handler from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ComponentHandler
