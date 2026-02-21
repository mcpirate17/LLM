#!/usr/bin/env python3
"""Fill missing param descriptions across component manifests."""
from __future__ import annotations

from pathlib import Path
import re
import yaml

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"


def _humanize(name: str) -> str:
    text = name.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.capitalize()


def _infer_desc(param_name: str, schema: dict) -> str:
    typ = schema.get("type", "value")
    if typ == "enum" and schema.get("options"):
        opts = ", ".join(str(o) for o in schema["options"])
        return f"Select { _humanize(param_name).lower() } ({opts})."
    if typ in {"integer", "float"}:
        return f"Numeric value for { _humanize(param_name).lower() }."
    if typ == "boolean":
        return f"Enable or disable { _humanize(param_name).lower() }."
    return f"Configuration for { _humanize(param_name).lower() }."


def main() -> int:
    updated = 0
    for path in sorted(COMPONENTS_ROOT.rglob("manifest.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = data.get("params") or {}
        changed = False
        if isinstance(params, dict):
            for name, schema in params.items():
                if not isinstance(schema, dict):
                    continue
                if schema.get("description"):
                    continue
                schema["description"] = _infer_desc(name, schema)
                changed = True
        if changed:
            path.write_text(yaml.dump(data, sort_keys=False, width=120), encoding="utf-8")
            updated += 1
    print(f"Updated {updated} manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
