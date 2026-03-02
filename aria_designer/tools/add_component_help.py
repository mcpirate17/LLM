#!/usr/bin/env python3
"""Add help_md to all component manifests.

Generates a short technical + plain language blurb from existing manifest
fields (description, inputs/outputs, params).
"""
from __future__ import annotations

from pathlib import Path
import yaml

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"


def _format_ports(ports):
    if not ports:
        return "none"
    parts = []
    for p in ports:
        name = p.get("name", "")
        dtype = p.get("dtype", "")
        shape = p.get("shape", [])
        shape_str = "[{}]".format(", ".join(str(s) for s in shape)) if shape else ""
        parts.append(f"{name}:{dtype}{shape_str}")
    return ", ".join(parts)


def build_help(manifest: dict) -> str:
    desc = (manifest.get("description") or "").strip()
    name = manifest.get("name") or manifest.get("id")
    inputs = _format_ports(manifest.get("inputs"))
    outputs = _format_ports(manifest.get("outputs"))
    params = manifest.get("params") or {}
    param_list = ", ".join(params.keys()) if params else "none"

    tech = desc if desc else f"{name} component."
    plain = desc if desc else f"{name} transforms its inputs into outputs."

    lines = [
        "### Technical",
        tech,
        "",
        f"Inputs: {inputs}.",
        f"Outputs: {outputs}.",
        f"Params: {param_list}.",
        "",
        "### Plain Language",
        plain,
        "",
        "Use when you need this specific transformation in a workflow.",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    manifests = sorted(COMPONENTS_ROOT.rglob("manifest.yaml"))
    updated = 0
    for path in manifests:
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("help_md"):
            continue
        data["help_md"] = build_help(data)
        path.write_text(yaml.dump(data, sort_keys=False, width=120))
        updated += 1
    print(f"Updated {updated} manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
