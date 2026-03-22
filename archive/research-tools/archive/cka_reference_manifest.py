from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_REFERENCE_FAMILIES = {"transformer", "ssm", "conv"}


@dataclass
class ManifestValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _ensure_dict(value: Any, field_name: str, errors: List[str]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{field_name} must be an object")
        return {}
    return value


def _ensure_list_of_strings(
    value: Any, field_name: str, errors: List[str]
) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field_name} must be a list of strings")
        return []
    return value


def validate_manifest(manifest: Dict[str, Any]) -> ManifestValidation:
    errors: List[str] = []
    warnings: List[str] = []

    required_fields = [
        "artifact_version",
        "created_at",
        "code_version",
        "python_torch_versions",
        "reference_families",
        "probe_protocol_hash",
        "activation_schema",
        "quality_flags",
    ]

    for field_name in required_fields:
        if field_name not in manifest:
            errors.append(f"missing required field: {field_name}")

    artifact_version = manifest.get("artifact_version")
    if "artifact_version" in manifest and not _is_non_empty_string(artifact_version):
        errors.append("artifact_version must be a non-empty string")

    created_at = manifest.get("created_at")
    if "created_at" in manifest:
        if not _is_non_empty_string(created_at):
            errors.append("created_at must be a non-empty string")
        else:
            try:
                datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                errors.append("created_at must be ISO-8601 format")

    code_version = manifest.get("code_version")
    if "code_version" in manifest and not _is_non_empty_string(code_version):
        errors.append("code_version must be a non-empty string")

    if "probe_protocol_hash" in manifest and not _is_non_empty_string(
        manifest.get("probe_protocol_hash")
    ):
        errors.append("probe_protocol_hash must be a non-empty string")

    versions = _ensure_dict(
        manifest.get("python_torch_versions"), "python_torch_versions", errors
    )
    if versions:
        if not _is_non_empty_string(versions.get("python")):
            errors.append("python_torch_versions.python must be a non-empty string")
        if not _is_non_empty_string(versions.get("torch")):
            errors.append("python_torch_versions.torch must be a non-empty string")

    families = _ensure_list_of_strings(
        manifest.get("reference_families"), "reference_families", errors
    )
    if families:
        normalized = {family.strip().lower() for family in families if family.strip()}
        if not normalized:
            errors.append("reference_families must not be empty")
        missing_families = EXPECTED_REFERENCE_FAMILIES - normalized
        if missing_families:
            warnings.append(
                "reference_families is missing expected entries: "
                + ", ".join(sorted(missing_families))
            )

    activation_schema = _ensure_dict(
        manifest.get("activation_schema"), "activation_schema", errors
    )
    if activation_schema:
        if not _is_non_empty_string(activation_schema.get("representation_format")):
            errors.append(
                "activation_schema.representation_format must be a non-empty string"
            )
        if (
            not isinstance(activation_schema.get("vector_dim"), int)
            or activation_schema.get("vector_dim") <= 0
        ):
            errors.append("activation_schema.vector_dim must be a positive integer")

    quality_flags = _ensure_dict(manifest.get("quality_flags"), "quality_flags", errors)
    if quality_flags:
        invalid_flags = [
            name for name, value in quality_flags.items() if not isinstance(value, bool)
        ]
        if invalid_flags:
            errors.append(
                "quality_flags values must be booleans: "
                + ", ".join(sorted(invalid_flags))
            )

    return ManifestValidation(valid=not errors, errors=errors, warnings=warnings)


def build_stub_manifest(version: str, code_version: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "artifact_version": version,
        "created_at": now,
        "code_version": code_version,
        "python_torch_versions": {
            "python": "3.11",
            "torch": "2.x",
        },
        "reference_families": ["transformer", "ssm", "conv"],
        "probe_protocol_hash": "replace-with-real-hash",
        "activation_schema": {
            "representation_format": "projected_activation_matrix",
            "vector_dim": 512,
            "layer_policy": "final_hidden",
        },
        "quality_flags": {
            "deterministic_seeds": False,
            "probe_hash_verified": False,
            "family_coverage_complete": False,
        },
    }


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _print_validation(result: ManifestValidation) -> None:
    if result.valid:
        print("Manifest is valid.")
    else:
        print("Manifest is invalid.")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold and validate CKA reference artifact manifests"
    )
    parser.add_argument(
        "--validate", type=Path, help="Path to an existing manifest.json to validate"
    )
    parser.add_argument(
        "--out", type=Path, help="Write a scaffold manifest.json to this path"
    )
    parser.add_argument(
        "--version", default="v1", help="Artifact version for scaffold generation"
    )
    parser.add_argument(
        "--code-version", default="unknown", help="Code version for scaffold generation"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures during validation",
    )
    args = parser.parse_args()

    if args.validate is None and args.out is None:
        parser.error("must provide at least one action: --validate and/or --out")

    if args.out is not None:
        scaffold = build_stub_manifest(
            version=args.version, code_version=args.code_version
        )
        _write_manifest(args.out, scaffold)
        print(f"Wrote scaffold manifest: {args.out}")

    if args.validate is not None:
        try:
            manifest = _load_manifest(args.validate)
        except Exception as exc:
            print(f"Failed to load manifest: {exc}")
            return 2

        result = validate_manifest(manifest)
        _print_validation(result)

        if not result.valid:
            return 1
        if args.strict and result.warnings:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
