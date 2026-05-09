from __future__ import annotations

"""Verify a split DB backup bundle without replacing live files."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from research.scientist.notebook.artifact_store import NotebookArtifactStore
from research.tools.db_health import assert_sqlite_health


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _extract_bundle(bundle: Path, destination: Path) -> None:
    _run(
        [
            "tar",
            "-I",
            "zstd",
            "-xf",
            str(bundle),
            "-C",
            str(destination),
        ]
    )


def _verify_manifest_files(root: Path, manifest: dict[str, Any]) -> list[str]:
    verified: list[str] = []
    for record in manifest.get("files") or []:
        rel = Path(str(record["path"]))
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(record.get("size") or 0)
        if path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch for {rel}")
        expected_hash = record.get("sha256")
        if expected_hash and _sha256_file(path) != expected_hash:
            raise ValueError(f"sha256 mismatch for {rel}")
        verified.append(str(rel))
    return verified


def _verify_artifacts(extracted_root: Path, *, limit: int) -> dict[str, int]:
    runs_db = extracted_root / "research" / "runs.db"
    if not runs_db.is_file():
        return {"checked": 0, "available": 0}
    store = NotebookArtifactStore(runs_db)
    with sqlite3.connect(str(runs_db)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            available = conn.execute(
                "SELECT COUNT(*) FROM notebook_artifacts"
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM notebook_artifacts
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {"checked": 0, "available": 0}
    checked = 0
    for row in rows:
        store.read_bytes(dict(row))
        checked += 1
    return {"checked": checked, "available": int(available or 0)}


def verify_bundle(
    *,
    bundle: Path,
    manifest: Path,
    extract_dir: Path | None = None,
    artifact_sample_limit: int = 10,
) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = manifest.resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

    temp_ctx = None
    if extract_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="split-bundle-drill-")
        extract_root = Path(temp_ctx.name)
    else:
        extract_root = extract_dir
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)
    try:
        _extract_bundle(bundle, extract_root)
        verified_files = _verify_manifest_files(extract_root, manifest_payload)

        db_checks: list[str] = []
        for rel in (
            "research/runs.db",
            "research/lab_notebook.db",
            "research/events.db",
        ):
            path = extract_root / rel
            if path.is_file():
                assert_sqlite_health(path, label=rel)
                db_checks.append(rel)

        artifact_checks = _verify_artifacts(
            extract_root,
            limit=max(0, artifact_sample_limit),
        )
        return {
            "bundle": str(bundle),
            "manifest": str(manifest),
            "verified_files": len(verified_files),
            "db_checks": db_checks,
            "artifact_checks": artifact_checks,
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--artifact-sample-limit", type=int, default=10)
    args = parser.parse_args(argv)
    report = verify_bundle(
        bundle=args.bundle,
        manifest=args.manifest,
        extract_dir=args.extract_dir,
        artifact_sample_limit=args.artifact_sample_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
