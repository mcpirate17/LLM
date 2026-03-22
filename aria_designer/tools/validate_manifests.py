#!/usr/bin/env python3
"""Validate all component manifests against the JSON Schema.

Usage:
    python tools/validate_manifests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Ensure parent package is importable
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.app.loader import REQUIRED_KEYS, VALID_CATEGORIES  # noqa: E402

SCHEMA_PATH = _PROJECT_ROOT / "schemas" / "component_manifest.v1.schema.json"
COMPONENTS_ROOT = _PROJECT_ROOT / "components"

# Try jsonschema if available, fallback to basic validation
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

VALID_DTYPES = {
    "tensor",
    "scalar",
    "index",
    "mask",
    "complex_tensor",
    "dataset",
    "list",
    "record",
}


def validate_basic(manifest: dict, path: Path) -> list[str]:
    """Basic validation without jsonschema library."""
    errors = []
    missing = REQUIRED_KEYS - set(manifest.keys())
    if missing:
        errors.append(f"{path}: missing keys: {missing}")
    if manifest.get("category") not in VALID_CATEGORIES:
        errors.append(f"{path}: invalid category '{manifest.get('category')}'")
    if not manifest.get("outputs"):
        errors.append(f"{path}: must have at least one output")

    # Validate dtypes in ports
    for section in ["inputs", "outputs"]:
        ports = manifest.get(section)
        if isinstance(ports, list):
            for i, port in enumerate(ports):
                if not isinstance(port, dict):
                    continue
                dtype = port.get("dtype")
                if dtype not in VALID_DTYPES:
                    errors.append(f"{path}: {section}[{i}] has invalid dtype '{dtype}'")

    return errors


def main():
    schema = None
    if HAS_JSONSCHEMA and SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)

    manifests = sorted(COMPONENTS_ROOT.rglob("manifest.yaml"))
    print(f"Found {len(manifests)} manifest files")

    errors = []
    for path in manifests:
        try:
            with open(path) as f:
                manifest = yaml.safe_load(f)
        except Exception as e:
            errors.append(f"{path}: YAML parse error: {e}")
            continue

        if manifest is None:
            errors.append(f"{path}: empty manifest")
            continue

        if schema and HAS_JSONSCHEMA:
            try:
                jsonschema.validate(manifest, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"{path}: schema error: {e.message}")
        else:
            errors.extend(validate_basic(manifest, path))

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    else:
        print(f"All {len(manifests)} manifests valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
