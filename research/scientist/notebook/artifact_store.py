from __future__ import annotations

"""Compressed artifact storage for bulky notebook payloads."""

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import zstandard as zstd

from research.defaults import NOTEBOOK_ARTIFACTS_DIR


ARTIFACT_POINTER_KEY = "_notebook_artifact"
DEFAULT_ARTIFACT_ROOT = Path(NOTEBOOK_ARTIFACTS_DIR)


def is_artifact_pointer(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get(ARTIFACT_POINTER_KEY))
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return is_artifact_pointer(json.loads(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    return False


def parse_artifact_pointer(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        pointer = value
    elif isinstance(value, bytes):
        try:
            pointer = json.loads(value.decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    elif isinstance(value, str):
        try:
            pointer = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    else:
        return None
    if not isinstance(pointer, dict) or not pointer.get(ARTIFACT_POINTER_KEY):
        return None
    return pointer


def artifact_pointer_json(artifact_id: str, *, path: str) -> str:
    return json.dumps(
        {
            ARTIFACT_POINTER_KEY: artifact_id,
            "path": path,
            "compression": "zstd",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class NotebookArtifactStore:
    """Write and read compressed notebook artifacts."""

    def __init__(self, db_path: str | Path, *, root: str | Path | None = None):
        db_path = Path(db_path)
        if root is None:
            if str(db_path) == ":memory:":
                root = DEFAULT_ARTIFACT_ROOT
            else:
                root = db_path.parent / "artifacts" / "notebook"
        self.root = Path(root)

    @staticmethod
    def _stable_component(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
        return safe[:120] or "unknown"

    def _relative_path(
        self,
        *,
        table_name: str,
        row_pk: str,
        column_name: str,
        content_type: str,
    ) -> Path:
        ext = "json.zst" if content_type == "application/json" else "bin.zst"
        return (
            Path(self._stable_component(table_name))
            / self._stable_component(row_pk)
            / f"{self._stable_component(column_name)}.{ext}"
        )

    @staticmethod
    def encode_payload(payload: Any, *, content_type: str) -> bytes:
        if isinstance(payload, bytes):
            return payload
        if content_type == "application/json":
            if isinstance(payload, str):
                return payload.encode("utf-8")
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        if isinstance(payload, str):
            return payload.encode("utf-8")
        raise TypeError(f"unsupported artifact payload type: {type(payload).__name__}")

    def write(
        self,
        *,
        table_name: str,
        row_pk: str,
        column_name: str,
        payload: Any,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        raw = self.encode_payload(payload, content_type=content_type)
        compressor = zstd.ZstdCompressor(level=10)
        compressed = compressor.compress(raw)
        rel_path = self._relative_path(
            table_name=table_name,
            row_pk=row_pk,
            column_name=column_name,
            content_type=content_type,
        )
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_bytes(compressed)
        tmp_path.replace(path)
        return {
            "artifact_id": uuid.uuid4().hex,
            "table_name": table_name,
            "row_pk": row_pk,
            "column_name": column_name,
            "path": str(rel_path),
            "compression": "zstd",
            "content_type": content_type,
            "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
            "sha256_compressed": hashlib.sha256(compressed).hexdigest(),
            "uncompressed_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "created_at": time.time(),
        }

    def read_bytes(self, metadata: dict[str, Any]) -> bytes:
        rel_path = str(metadata.get("path") or "")
        if not rel_path:
            raise FileNotFoundError("artifact metadata has no path")
        path = self.root / rel_path
        compressed = path.read_bytes()
        expected_compressed = metadata.get("sha256_compressed")
        if expected_compressed:
            actual = hashlib.sha256(compressed).hexdigest()
            if actual != expected_compressed:
                raise ValueError(f"compressed artifact hash mismatch for {path}")
        raw = zstd.ZstdDecompressor().decompress(compressed)
        expected_raw = metadata.get("sha256_uncompressed")
        if expected_raw:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_raw:
                raise ValueError(f"artifact hash mismatch for {path}")
        return raw

    def read_json(self, metadata: dict[str, Any]) -> Any:
        raw = self.read_bytes(metadata)
        return json.loads(raw.decode("utf-8"))
