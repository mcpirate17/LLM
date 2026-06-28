"""Run provenance helpers for component_fab reports.

The fab loop can only be trusted when the report says exactly what produced it:
code revision, argv, Python/runtime details, schema versions, and policy config
versions. This module is deliberately best-effort; missing git metadata must not
break local exploratory runs.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from component_fab.state.schema_versions import SCHEMA_VERSIONS


@dataclass(frozen=True, slots=True)
class RunProvenance:
    run_id: str
    created_at_utc: str
    argv: tuple[str, ...]
    cwd: str
    python_version: str
    platform: str
    git_sha: str | None
    git_dirty: bool | None
    schema_versions: dict[str, str] = field(default_factory=lambda: dict(SCHEMA_VERSIONS))
    config_versions: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["argv"] = list(self.argv)
        return payload


def _git(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    return result.stdout.strip()


def build_run_provenance(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | str | None = None,
    config_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready provenance block for a component_fab artifact."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    status = _git(["status", "--porcelain"], cwd=root)
    return RunProvenance(
        run_id=uuid.uuid4().hex,
        created_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        argv=tuple(sys.argv[1:] if argv is None else argv),
        cwd=os.getcwd(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_sha=_git(["rev-parse", "HEAD"], cwd=root),
        git_dirty=None if status is None else bool(status),
        config_versions=dict(config_versions or {}),
    ).to_json()
