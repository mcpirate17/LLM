"""Shared database backup helpers for research tools."""

from __future__ import annotations

import shutil
from pathlib import Path


def backup_database(
    db_path: Path,
    backup_name: str,
    *,
    project_root: Path,
    google_backup_root: Path,
    dry_run: bool = False,
) -> dict[str, str]:
    local_dir = project_root / "research/db_backups" / backup_name
    google_dir = google_backup_root / backup_name
    local_target = local_dir / db_path.name
    google_target = google_dir / db_path.name
    if dry_run:
        return {"local": str(local_target), "google_drive": str(google_target)}
    local_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, local_target)
    google_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, google_target)
    return {"local": str(local_target), "google_drive": str(google_target)}
