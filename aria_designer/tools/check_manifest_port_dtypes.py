#!/usr/bin/env python3
"""Fail if any manifest uses port dtypes outside schema enum.

Usage:
    python tools/check_manifest_port_dtypes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "component_manifest.v1.schema.json"
COMPONENTS_ROOT = Path(__file__).parent.parent / "components"


def _load_allowed_dtypes() -> set[str]:
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)

    dtype_enum = schema.get("$defs", {}).get("port", {}).get("properties", {}).get("dtype", {}).get("enum")
    if not isinstance(dtype_enum, list) or not dtype_enum:
        raise ValueError("Could not read non-empty $defs.port.properties.dtype.enum from schema")

    bad_entries = [entry for entry in dtype_enum if not isinstance(entry, str)]
    if bad_entries:
        raise ValueError("Schema dtype enum contains non-string entries")

    return set(dtype_enum)


def _validate_manifest_port_dtypes(manifest: dict, manifest_path: Path, allowed_dtypes: set[str]) -> list[str]:
    errors: list[str] = []
    for section in ("inputs", "outputs"):
        ports = manifest.get(section, [])
        if not isinstance(ports, list):
            errors.append(f"{manifest_path}: `{section}` must be a list")
            continue

        for idx, port in enumerate(ports):
            if not isinstance(port, dict):
                errors.append(f"{manifest_path}: `{section}[{idx}]` must be an object")
                continue

            dtype = port.get("dtype")
            port_name = port.get("name", f"{section}[{idx}]")
            if not isinstance(dtype, str):
                errors.append(f"{manifest_path}: `{section}[{idx}]` ({port_name}) missing string dtype")
                continue

            if dtype not in allowed_dtypes:
                errors.append(
                    f"{manifest_path}: `{section}[{idx}]` ({port_name}) dtype '{dtype}' not in schema enum {sorted(allowed_dtypes)}"
                )

    return errors


def main() -> int:
    try:
        allowed_dtypes = _load_allowed_dtypes()
    except Exception as exc:
        print(f"ERROR: failed loading schema enum: {exc}")
        return 1

    manifests = sorted(COMPONENTS_ROOT.rglob("manifest.yaml"))
    print(f"Checking port dtypes in {len(manifests)} manifest files")
    print(f"Allowed dtypes: {sorted(allowed_dtypes)}")

    errors: list[str] = []
    for manifest_path in manifests:
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = yaml.safe_load(handle)
        except Exception as exc:
            errors.append(f"{manifest_path}: YAML parse error: {exc}")
            continue

        if not isinstance(manifest, dict):
            errors.append(f"{manifest_path}: manifest root must be an object")
            continue

        errors.extend(_validate_manifest_port_dtypes(manifest, manifest_path, allowed_dtypes))

    if errors:
        print(f"\nFound {len(errors)} dtype validation errors:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1

    print("All manifest port dtypes conform to schema enum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
