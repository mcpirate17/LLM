from __future__ import annotations

"""Helpers for reading graph_json whether inline or artifact-backed."""

import sqlite3
from pathlib import Path
from typing import Any

from .artifact_store import (
    ARTIFACT_POINTER_KEY,
    NotebookArtifactStore,
    parse_artifact_pointer,
)


def is_nonempty_graph_json(value: Any) -> bool:
    if parse_artifact_pointer(value):
        return True
    if value is None:
        return False
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    return text.strip() not in {"", "{}"}


def resolve_graph_json_value(
    conn: sqlite3.Connection,
    db_path: str | Path,
    value: Any,
) -> str:
    pointer = parse_artifact_pointer(value)
    if pointer is None:
        if value is None:
            return ""
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    artifact_id = pointer[ARTIFACT_POINTER_KEY]
    row = conn.execute(
        "SELECT * FROM notebook_artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    if row is not None:
        metadata: dict[str, Any] = dict(row)
    elif pointer.get("path"):
        metadata = {
            "artifact_id": artifact_id,
            "path": pointer["path"],
            "compression": pointer.get("compression") or "zstd",
        }
    else:
        raise ValueError(f"graph artifact metadata not found: {artifact_id}")
    return NotebookArtifactStore(db_path).read_bytes(metadata).decode("utf-8")
