"""Component property coverage audit helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class PropertyIssue:
    code: str
    message: str
    severity: str = "warning"


def _is_split_or_filter(manifest: Dict[str, Any]) -> bool:
    text = f"{manifest.get('id', '')} {manifest.get('name', '')}".lower()
    return ("split" in text) or ("filter" in text)


def _is_sink_component(manifest: Dict[str, Any]) -> bool:
    text = f"{manifest.get('id', '')} {manifest.get('name', '')}"
    tags = " ".join(manifest.get("tags", []) or [])
    both = f"{text} {tags}".lower()
    return any(k in both for k in ["output", "writer", "sink", "export"])


def _param_issues(param_name: str, schema: Dict[str, Any]) -> List[PropertyIssue]:
    issues: List[PropertyIssue] = []
    if "type" not in schema:
        issues.append(PropertyIssue("param_missing_type", f"Param '{param_name}' is missing type", "error"))
    if "default" not in schema:
        issues.append(PropertyIssue("param_missing_default", f"Param '{param_name}' is missing default"))
    if schema.get("type") == "enum" and not schema.get("options"):
        issues.append(PropertyIssue("enum_missing_options", f"Param '{param_name}' is enum without options", "error"))
    if not schema.get("description"):
        issues.append(PropertyIssue("param_missing_description", f"Param '{param_name}' has no description"))
    return issues


def analyze_manifest(manifest: Dict[str, Any], source_path: str = "") -> Dict[str, Any]:
    params = manifest.get("params") or {}
    issues: List[PropertyIssue] = []

    if not manifest.get("description"):
        issues.append(PropertyIssue("missing_description", "Component has no description"))

    if not isinstance(params, dict):
        issues.append(PropertyIssue("params_not_object", "Manifest params is not a mapping", "error"))
        params = {}

    for name, schema in params.items():
        if not isinstance(schema, dict):
            issues.append(PropertyIssue("param_schema_invalid", f"Param '{name}' schema is not an object", "error"))
            continue
        issues.extend(_param_issues(name, schema))

    if _is_split_or_filter(manifest):
        keys = set(params.keys())
        if not ({"split_scope", "filter_scope", "split_axis", "axis", "mode"} & keys):
            issues.append(PropertyIssue(
                "ambiguous_scope",
                "Split/filter component should expose token-vs-feature/data scope explicitly",
            ))

    if _is_sink_component(manifest) and len(params) == 0:
        issues.append(PropertyIssue("sink_missing_config", "Sink/output component should expose output configuration"))

    # If component has no inputs and no params, users cannot configure source behavior.
    if not manifest.get("inputs") and len(params) == 0:
        issues.append(PropertyIssue("source_missing_config", "Source component has no configurable parameters"))

    severity_rank = {"error": 2, "warning": 1}
    score = max((severity_rank.get(i.severity, 1) for i in issues), default=0)
    status = "ok" if not issues else ("error" if score >= 2 else "warning")

    return {
        "id": manifest.get("id"),
        "name": manifest.get("name"),
        "category": manifest.get("category"),
        "source_path": source_path,
        "property_count": len(params),
        "has_help": bool(manifest.get("help_md")),
        "status": status,
        "issues": [i.__dict__ for i in issues],
    }


def audit_components(root: Path) -> Dict[str, Any]:
    manifests: List[Tuple[str, Dict[str, Any]]] = []
    for path in sorted(root.rglob("manifest.yaml")):
        with path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        manifests.append((str(path), manifest))

    by_component = [analyze_manifest(m, p) for p, m in manifests]

    total = len(by_component)
    warnings = sum(1 for row in by_component if row["status"] == "warning")
    errors = sum(1 for row in by_component if row["status"] == "error")

    return {
        "summary": {
            "total_components": total,
            "ok": total - warnings - errors,
            "warnings": warnings,
            "errors": errors,
        },
        "components": by_component,
    }
