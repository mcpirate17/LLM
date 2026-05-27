from __future__ import annotations

"""Helpers for reading graph_json whether inline or artifact-backed."""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from .artifact_store import (
    ARTIFACT_POINTER_KEY,
    NotebookArtifactStore,
    parse_artifact_pointer,
)

logger = logging.getLogger(__name__)

# Process-lifetime dedup so callers that scan the same table on every cache
# refresh (observability, exp_coverage, mathspace impact, ...) don't flood the
# log when artifacts have been pruned from disk. Keyed by `category`; we only
# re-warn when the missing-row count grows.
_warned_missing_artifact_counts: dict[str, int] = {}
_warned_missing_artifacts_lock = threading.Lock()


def warn_missing_artifacts_once(
    category: str,
    count: int,
    examples: Sequence[str] = (),
) -> None:
    """Emit one WARNING per process per category when artifacts are missing.

    Repeat calls with an equal-or-smaller count log at DEBUG instead, so the
    same scan of the same dead pointers doesn't keep firing WARNING every
    cache rebuild. A larger count (new pruning, new rows) re-arms the warning.
    """
    if count <= 0:
        return
    with _warned_missing_artifacts_lock:
        prev = _warned_missing_artifact_counts.get(category, 0)
        if count <= prev:
            logger.debug(
                "%s: %d missing graph artifact(s) (already reported; max=%d)",
                category,
                count,
                prev,
            )
            return
        _warned_missing_artifact_counts[category] = count
    examples_msg = "; ".join(examples)
    if examples_msg:
        logger.warning(
            "%s: %d missing graph artifact(s). examples=%s",
            category,
            count,
            examples_msg,
        )
    else:
        logger.warning("%s: %d missing graph artifact(s)", category, count)


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
