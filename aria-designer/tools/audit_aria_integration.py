#!/usr/bin/env python3
"""
Audit Aria Designer component coverage against research/ primitives.

Outputs:
  - docs/integration_component_audit.json
  - docs/integration_component_audit.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
COMPONENTS_ROOT = ROOT / "components"
OUT_JSON = ROOT / "docs" / "integration_component_audit.json"
OUT_MD = ROOT / "docs" / "integration_component_audit.md"


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from research.synthesis.primitives import PRIMITIVE_REGISTRY  # noqa: E402
from runtime.bridge import (  # noqa: E402
    _COMPONENT_ALIASES,
    _IO_COMPONENTS,
    _PASSTHROUGH_COMPONENTS,
    _SOURCE_COMPONENTS,
    _TEMPLATE_LOWERED_COMPONENTS,
)


@dataclass
class ComponentAudit:
    component_id: str
    category: str
    folder: str
    primitive_status: str
    primitive_name: Optional[str]
    native_impl: List[str]
    has_python_fallback: bool
    param_count: int


def _detect_native_impl(component_dir: Path) -> List[str]:
    impl = []
    if (component_dir / "kernel.c").exists():
        impl.append("c")
    if (component_dir / "kernel.cpp").exists() or (component_dir / "kernel.cc").exists():
        impl.append("cpp")
    if (component_dir / "kernel.rs").exists():
        impl.append("rust")
    if (component_dir / "kernel.pyx").exists():
        impl.append("cython")
    return impl


def _resolve_primitive(component_id: str, category: str) -> tuple[str, Optional[str]]:
    leaf = component_id.split("/")[-1]
    cat_prefixed = f"{category}/{leaf}"
    candidates = [component_id, leaf, cat_prefixed]

    for candidate in candidates:
        cid = candidate.split("/")[-1]
        if cid in _IO_COMPONENTS:
            return "io", None
        if cid in _SOURCE_COMPONENTS:
            return "source", None
        if cid in _TEMPLATE_LOWERED_COMPONENTS:
            return "template", None
        if cid in _PASSTHROUGH_COMPONENTS:
            return "passthrough", None
        if cid in PRIMITIVE_REGISTRY:
            return "direct", cid
        if cid in _COMPONENT_ALIASES:
            return "alias", _COMPONENT_ALIASES[cid]
    return "unmapped", None


def _load_manifests() -> List[ComponentAudit]:
    audits: List[ComponentAudit] = []
    for manifest_path in sorted(COMPONENTS_ROOT.glob("*/*/manifest.yaml")):
        component_dir = manifest_path.parent
        category = component_dir.parent.name
        folder = component_dir.name

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        component_id = str(manifest.get("id") or folder)
        status, primitive = _resolve_primitive(component_id, category)

        audits.append(
            ComponentAudit(
                component_id=component_id,
                category=category,
                folder=folder,
                primitive_status=status,
                primitive_name=primitive,
                native_impl=_detect_native_impl(component_dir),
                has_python_fallback=(component_dir / "kernel_fallback.py").exists(),
                param_count=len(manifest.get("params") or {}),
            )
        )
    return audits


def _build_report(audits: List[ComponentAudit]) -> Dict[str, Any]:
    by_status = Counter(a.primitive_status for a in audits)
    by_category_unmapped = Counter(a.category for a in audits if a.primitive_status == "unmapped")
    native_count = sum(1 for a in audits if a.native_impl)
    fallback_count = sum(1 for a in audits if a.has_python_fallback)

    unmapped_by_cat: Dict[str, List[str]] = defaultdict(list)
    for a in audits:
        if a.primitive_status == "unmapped":
            unmapped_by_cat[a.category].append(a.component_id)

    return {
        "summary": {
            "total_components": len(audits),
            "mapped_direct": by_status.get("direct", 0),
            "mapped_alias": by_status.get("alias", 0),
            "mapped_source": by_status.get("source", 0),
            "mapped_template": by_status.get("template", 0),
            "mapped_passthrough": by_status.get("passthrough", 0),
            "io_components": by_status.get("io", 0),
            "unmapped_components": by_status.get("unmapped", 0),
            "native_kernel_components": native_count,
            "python_fallback_components": fallback_count,
            "primitive_registry_size": len(PRIMITIVE_REGISTRY),
        },
        "unmapped_by_category": {k: sorted(v) for k, v in sorted(unmapped_by_cat.items())},
        "unmapped_category_counts": dict(sorted(by_category_unmapped.items())),
        "components": [asdict(a) for a in audits],
    }


def _write_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Aria Integration Component Audit",
        "",
        "## Summary",
        f"- Total components: {s['total_components']}",
        f"- Direct primitive mapping: {s['mapped_direct']}",
        f"- Alias primitive mapping: {s['mapped_alias']}",
        f"- Source-lowered mapping: {s['mapped_source']}",
        f"- Template-lowered mapping: {s['mapped_template']}",
        f"- Passthrough-lowered mapping: {s['mapped_passthrough']}",
        f"- IO passthrough components: {s['io_components']}",
        f"- Unmapped components: {s['unmapped_components']}",
        f"- Components with native kernel files (C/C++/Rust/Cython): {s['native_kernel_components']}",
        f"- Components with Python fallback: {s['python_fallback_components']}",
        "",
        "## Unmapped Components By Category",
    ]

    for cat, count in report["unmapped_category_counts"].items():
        lines.append(f"- {cat}: {count}")

    lines.append("")
    lines.append("## Highest Priority Gaps")

    priority_cats = ("mixing", "routing", "data_io", "data_transform", "blocks", "control_flow")
    for cat in priority_cats:
        items = report["unmapped_by_category"].get(cat, [])
        if not items:
            continue
        preview = ", ".join(items[:8])
        more = "" if len(items) <= 8 else f", ... (+{len(items) - 8} more)"
        lines.append(f"- {cat}: {preview}{more}")

    return "\n".join(lines) + "\n"


def main() -> int:
    audits = _load_manifests()
    report = _build_report(audits)

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(_write_markdown(report), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
