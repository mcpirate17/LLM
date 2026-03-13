from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _tool_path(repo_root: Path, script_name: str) -> str:
    return str((repo_root / "tools" / script_name).resolve())


def _extract_hash(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("hash command produced empty output")
    return lines[-1]


def _set_manifest_hash(manifest_path: Path, digest: str) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    manifest["probe_protocol_hash"] = digest
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command CKA artifact integrity flow: scaffold, hash, verify, validate"
    )
    parser.add_argument("--artifact-dir", default="artifacts/cka_references/v1", help="Artifact pack directory")
    parser.add_argument("--version", default="v1", help="Artifact version for scaffold step")
    parser.add_argument("--code-version", default="unknown", help="Code version for scaffold step")
    parser.add_argument(
        "--scaffold-if-missing",
        action="store_true",
        help="Scaffold manifest.json if missing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict manifest validation",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root path containing the tools directory",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = repo_root / artifact_dir
    manifest_path = artifact_dir / "manifest.json"
    probe_path = artifact_dir / "probe_protocol.json"

    if not probe_path.exists():
        print(f"error: missing probe spec: {probe_path}")
        return 2

    if not manifest_path.exists():
        if not args.scaffold_if_missing:
            print(f"error: missing manifest: {manifest_path} (use --scaffold-if-missing)")
            return 2
        scaffold_cmd = [
            sys.executable,
            _tool_path(repo_root, "cka_reference_manifest.py"),
            "--out",
            str(manifest_path),
            "--version",
            args.version,
            "--code-version",
            args.code_version,
        ]
        scaffold_result = _run(scaffold_cmd)
        if scaffold_result.stdout.strip():
            print(scaffold_result.stdout.strip())

    hash_cmd = [
        sys.executable,
        _tool_path(repo_root, "probe_protocol_hash.py"),
        "--spec-file",
        str(probe_path),
    ]
    hash_result = _run(hash_cmd)
    digest = _extract_hash(hash_result.stdout)
    _set_manifest_hash(manifest_path, digest)
    print(f"probe_protocol_hash set to: {digest}")

    verify_cmd = [
        sys.executable,
        _tool_path(repo_root, "verify_probe_protocol_hash.py"),
        "--manifest",
        str(manifest_path),
        "--probe-spec",
        str(probe_path),
    ]
    verify_result = _run(verify_cmd)
    if verify_result.stdout.strip():
        print(verify_result.stdout.strip())

    validate_cmd = [
        sys.executable,
        _tool_path(repo_root, "cka_reference_manifest.py"),
        "--validate",
        str(manifest_path),
    ]
    if args.strict:
        validate_cmd.append("--strict")
    validate_result = _run(validate_cmd)
    if validate_result.stdout.strip():
        print(validate_result.stdout.strip())

    print("cka artifact integrity check complete")
    return 0

    

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.strip() if exc.stdout else ""
        stderr = exc.stderr.strip() if exc.stderr else ""
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
