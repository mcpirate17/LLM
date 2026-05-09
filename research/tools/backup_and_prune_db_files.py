from __future__ import annotations

"""Upload local DB backup files to Google Drive, then prune verified copies."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DEST = "gdrive:LLM/db-backups"
DEFAULT_STAGING = Path("research/tmp/db-backup-upload")


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def discover_db_backup_files(root: Path, *, include_live_db: bool) -> list[Path]:
    candidates: set[Path] = set()
    patterns = (
        "lab_notebook.db.snap_*",
        "lab_notebook.db.pre_*",
        "lab_notebook.db.corrupt_*",
        "lab_notebook.db.*_20*",
        "db_backups/**/*",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                candidates.add(path)
    if include_live_db:
        for name in ("lab_notebook.db", "runs.db", "events.db"):
            live = root / name
            if live.is_file():
                candidates.add(live)
    return sorted(candidates)


def discover_notebook_artifacts(root: Path) -> list[Path]:
    artifact_root = root / "artifacts" / "notebook"
    if not artifact_root.exists():
        return []
    return sorted(path for path in artifact_root.rglob("*") if path.is_file())


def discover_runtime_event_files(root: Path) -> list[Path]:
    event_root = root / "runtime_events"
    if not event_root.exists():
        return []
    return sorted(path for path in event_root.rglob("*") if path.is_file())


def discover_live_bundle_files(root: Path, *, include_artifacts: bool) -> list[Path]:
    files: list[Path] = []
    for name in ("lab_notebook.db", "runs.db", "events.db"):
        live = root / name
        if live.is_file():
            files.append(live)
    files.extend(discover_runtime_event_files(root))
    split_manifest = root / "db_split_manifest.md"
    if split_manifest.is_file():
        files.append(split_manifest)
    if include_artifacts:
        files.extend(discover_notebook_artifacts(root))
    return sorted(files)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root.parent)),
        "size": stat.st_size,
        "allocated_bytes": getattr(stat, "st_blocks", 0) * 512,
        "mtime": stat.st_mtime,
        "sha256": _sha256_file(path),
    }


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def create_bundle(
    *,
    root: Path,
    staging_root: Path,
    include_live_db: bool,
    include_artifacts: bool,
    latest_only: bool,
) -> dict[str, Any]:
    db_files = discover_db_backup_files(root, include_live_db=False)
    if latest_only:
        files = discover_live_bundle_files(root, include_artifacts=include_artifacts)
    else:
        artifact_files = discover_notebook_artifacts(root) if include_artifacts else []
        history_files = discover_db_backup_files(
            root,
            include_live_db=include_live_db,
        )
        files = sorted({*history_files, *artifact_files})
    stamp = _timestamp()
    staging = staging_root / stamp
    staging.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": stamp,
        "root": str(root.resolve()),
        "include_live_db": include_live_db,
        "include_artifacts": include_artifacts,
        "latest_only": latest_only,
        "files": [_file_record(path, root) for path in files],
        "prunable_files": [_file_record(path, root) for path in db_files],
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    file_list_path = staging / "files.txt"
    file_list_path.write_text(
        "\n".join(str(path.relative_to(root.parent)) for path in files) + "\n"
    )
    bundle_path = staging / "db-backups.tar.zst"
    if files:
        _run(
            [
                "tar",
                "--sparse",
                "-I",
                "zstd -19 -T0",
                "-cf",
                str(bundle_path),
                "-C",
                str(root.parent),
                "-T",
                str(file_list_path),
            ],
            cwd=root.parent,
        )
    else:
        bundle_path.write_bytes(b"")
    return {
        "stamp": stamp,
        "staging": str(staging),
        "manifest": str(manifest_path),
        "bundle": str(bundle_path),
        "files": [str(path) for path in files],
        "prunable_files": [str(path) for path in db_files],
    }


def upload_and_verify(*, staging: Path, destination: str, stamp: str) -> str:
    remote = f"{destination.rstrip('/')}/{stamp}"
    _run(["rclone", "copy", str(staging), remote, "--progress"], cwd=Path.cwd())
    _run(["rclone", "check", str(staging), remote, "--size-only"], cwd=Path.cwd())
    return remote


def prune_files(files: list[str], *, root: Path) -> list[str]:
    live_paths = {
        (root / "lab_notebook.db").resolve(),
        (root / "runs.db").resolve(),
        (root / "events.db").resolve(),
    }
    removed: list[str] = []
    for raw in files:
        path = Path(raw)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in live_paths:
            continue
        if not path.exists():
            continue
        path.unlink()
        removed.append(str(path))
    db_backups = root / "db_backups"
    if db_backups.exists():
        for dirpath, dirnames, filenames in os.walk(db_backups, topdown=False):
            current = Path(dirpath)
            if not dirnames and not filenames:
                try:
                    current.rmdir()
                except OSError:
                    pass
        try:
            if not any(db_backups.iterdir()):
                db_backups.rmdir()
        except OSError:
            pass
    tmp_root = root / "tmp" / "db-backup-upload"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument("--destination", default=DEFAULT_DEST)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--include-live-db", action="store_true")
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="Bundle historical DB backups too. Default uploads only the live DB/artifacts.",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--prune-after-verify", action="store_true")
    args = parser.parse_args(argv)

    bundle = create_bundle(
        root=args.root,
        staging_root=args.staging_root,
        include_live_db=args.include_live_db,
        include_artifacts=args.include_artifacts,
        latest_only=not args.all_history,
    )
    result: dict[str, Any] = {"bundle": bundle, "remote": None, "removed": []}
    if args.upload:
        remote = upload_and_verify(
            staging=Path(bundle["staging"]),
            destination=args.destination,
            stamp=str(bundle["stamp"]),
        )
        result["remote"] = remote
        if args.prune_after_verify:
            result["removed"] = prune_files(bundle["prunable_files"], root=args.root)
    elif args.prune_after_verify:
        raise SystemExit("--prune-after-verify requires --upload")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
