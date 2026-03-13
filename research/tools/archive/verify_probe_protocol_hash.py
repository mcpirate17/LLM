from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _canonical_json(payload: Any) -> str:
    normalized = _normalize(payload)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)


def _compute_hash(payload: Any, algorithm: str) -> str:
    canonical = _canonical_json(payload).encode("utf-8")
    digest = hashlib.new(algorithm)
    digest.update(canonical)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return data


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify manifest probe_protocol_hash against canonicalized probe protocol JSON"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--probe-spec", required=True, help="Path to probe_protocol.json")
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm accepted by hashlib (default: sha256)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write computed hash back to manifest on mismatch",
    )
    args = parser.parse_args()

    try:
        manifest_path = Path(args.manifest)
        probe_spec_path = Path(args.probe_spec)

        manifest = _load_json(manifest_path)
        probe_spec = _load_json(probe_spec_path)

        computed_hash = _compute_hash(probe_spec, args.algorithm)
        current_hash = manifest.get("probe_protocol_hash", "")

        print(f"manifest probe_protocol_hash: {current_hash}")
        print(f"computed probe_protocol_hash: {computed_hash}")

        if current_hash == computed_hash:
            print("probe protocol hash is consistent")
            return 0

        if args.update_manifest:
            manifest["probe_protocol_hash"] = computed_hash
            _save_json(manifest_path, manifest)
            print("manifest probe_protocol_hash updated")
            return 0

        print("probe protocol hash mismatch")
        return 1
    except Exception as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
