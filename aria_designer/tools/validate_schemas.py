#!/usr/bin/env python3
"""Validate JSON schemas for structural correctness.

Usage:
    python tools/validate_schemas.py
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMAS_ROOT = Path(__file__).parent.parent / "schemas"

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def main() -> int:
    schema_files = sorted(SCHEMAS_ROOT.glob("*.schema.json"))
    print(f"Found {len(schema_files)} schema files")

    errors: list[str] = []
    for path in schema_files:
        try:
            with open(path) as f:
                schema = json.load(f)
        except Exception as e:
            errors.append(f"{path}: JSON parse error: {e}")
            continue

        if HAS_JSONSCHEMA:
            try:
                jsonschema.Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                errors.append(f"{path}: schema error: {e.message}")

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print(f"All {len(schema_files)} schemas valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
