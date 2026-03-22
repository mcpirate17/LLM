from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def canonical_probe_json(payload: Any) -> str:
    normalized = _normalize(payload)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)


def compute_probe_protocol_hash(payload: Any, algorithm: str = "sha256") -> str:
    canonical = canonical_probe_json(payload).encode("utf-8")
    digest = hashlib.new(algorithm)
    digest.update(canonical)
    return digest.hexdigest()


def _load_payload(args: argparse.Namespace) -> Any:
    if args.spec_json:
        return json.loads(args.spec_json)

    if args.spec_file:
        raw = Path(args.spec_file).read_text(encoding="utf-8")
        return json.loads(raw)

    if not sys.stdin.isatty():
        raw_stdin = sys.stdin.read()
        if raw_stdin.strip():
            return json.loads(raw_stdin)

    raise ValueError("Provide probe spec via --spec-json, --spec-file, or stdin")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute deterministic hash for probe protocol JSON"
    )
    parser.add_argument(
        "--spec-file", help="Path to JSON file containing probe protocol spec"
    )
    parser.add_argument("--spec-json", help="Inline probe protocol JSON string")
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm name accepted by hashlib (default: sha256)",
    )
    parser.add_argument(
        "--print-canonical",
        action="store_true",
        help="Also print canonical JSON before hash",
    )
    args = parser.parse_args()

    try:
        payload = _load_payload(args)
        canonical = canonical_probe_json(payload)
        digest = hashlib.new(args.algorithm)
        digest.update(canonical.encode("utf-8"))
        if args.print_canonical:
            print(canonical)
        print(digest.hexdigest())
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
