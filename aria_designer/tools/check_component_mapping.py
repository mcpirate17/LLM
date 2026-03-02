#!/usr/bin/env python3
"""
Fail CI when component manifests are missing integration mapping classification.

Rules:
- Component is OK if it maps directly to PRIMITIVE_REGISTRY.
- Component is OK if it maps via alias.
- Component is OK if it is an IO component.
- Otherwise it must have a non-primitive execution class via:
  - component_execution_class[component_id], or
  - category_execution_class[category].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
COMPONENTS_ROOT = ROOT / "components"
MAPPING_FILE = ROOT / "runtime" / "component_mapping.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from research.synthesis.primitives import PRIMITIVE_REGISTRY  # noqa: E402
from runtime.bridge import _IO_COMPONENTS  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _component_manifests() -> List[Tuple[str, str, Path]]:
    rows: List[Tuple[str, str, Path]] = []
    for manifest_path in sorted(COMPONENTS_ROOT.glob("*/*/manifest.yaml")):
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        cid = str(data.get("id") or manifest_path.parent.name)
        category = manifest_path.parent.parent.name
        rows.append((cid, category, manifest_path))
    return rows


def _execution_class(cid: str, category: str, mapping: Dict[str, Any]) -> str:
    by_component = mapping.get("component_execution_class") or {}
    by_category = mapping.get("category_execution_class") or {}
    if cid in by_component:
        return str(by_component[cid])
    if category in by_category:
        return str(by_category[category])
    return "primitive"


def main() -> int:
    mapping = _load_yaml(MAPPING_FILE)
    aliases = mapping.get("aliases") or {}
    allowed_classes = {"primitive", "primitive_candidate", "composite", "data_control", "control", "io"}

    errors: List[str] = []
    unmapped_count = 0

    for cid, category, manifest_path in _component_manifests():
        leaf = cid.split("/")[-1]
        exec_class = _execution_class(leaf, category, mapping)
        if exec_class not in allowed_classes:
            errors.append(
                f"{manifest_path}: invalid execution class '{exec_class}' for '{leaf}'"
            )
            continue

        if leaf in _IO_COMPONENTS or leaf in PRIMITIVE_REGISTRY or leaf in aliases:
            continue

        unmapped_count += 1
        if exec_class in {"primitive", "io"}:
            errors.append(
                f"{manifest_path}: unmapped component '{leaf}' missing explicit non-primitive "
                f"classification in runtime/component_mapping.yaml"
            )

    if errors:
        print("Component mapping check failed:")
        for err in errors:
            print(f"- {err}")
        print(f"\nUnmapped components examined: {unmapped_count}")
        return 1

    print("Component mapping check passed.")
    print(f"Unmapped components classified: {unmapped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
