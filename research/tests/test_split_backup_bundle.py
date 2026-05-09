from __future__ import annotations

import hashlib
import json
import sqlite3

import zstandard as zstd

from research.tools.backup_and_prune_db_files import create_bundle
from research.tools.restore_split_bundle_drill import verify_bundle


def _write_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (value TEXT)")
        conn.execute("INSERT INTO t VALUES ('ok')")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_latest_split_bundle_includes_active_dbs_events_artifacts_and_hashes(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    _write_db(root / "lab_notebook.db")
    _write_db(root / "runs.db")
    (root / "db_split_manifest.md").write_text("manifest", encoding="utf-8")
    event_dir = root / "runtime_events"
    event_dir.mkdir()
    (event_dir / "segment-000001.ndjson").write_text("{}\n", encoding="utf-8")
    artifact_dir = root / "artifacts" / "notebook" / "entries" / "row"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "metadata_json.json.zst").write_bytes(b"artifact")

    bundle = create_bundle(
        root=root,
        staging_root=root / "tmp" / "db-backup-upload",
        include_live_db=True,
        include_artifacts=True,
        latest_only=True,
    )

    manifest = json.loads(
        (
            root / "tmp" / "db-backup-upload" / bundle["stamp"] / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    records = {item["path"]: item for item in manifest["files"]}
    assert "research/lab_notebook.db" in records
    assert "research/runs.db" in records
    assert "research/runtime_events/segment-000001.ndjson" in records
    assert "research/artifacts/notebook/entries/row/metadata_json.json.zst" in records
    assert "research/db_split_manifest.md" in records
    assert records["research/runs.db"]["sha256"]


def test_restore_split_bundle_drill_verifies_hashes_dbs_and_artifacts(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    _write_db(root / "lab_notebook.db")
    runs_db = root / "runs.db"
    artifact_rel = "training_curves/rid/curve_json.json.zst"
    raw = b'[{"step":0,"loss":1.0}]'
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    artifact_path = root / "artifacts" / "notebook" / artifact_rel
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(compressed)
    with sqlite3.connect(runs_db) as conn:
        conn.execute(
            """
            CREATE TABLE notebook_artifacts (
                artifact_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                row_pk TEXT NOT NULL,
                column_name TEXT NOT NULL,
                path TEXT NOT NULL,
                compression TEXT NOT NULL,
                content_type TEXT NOT NULL,
                sha256_uncompressed TEXT NOT NULL,
                sha256_compressed TEXT NOT NULL,
                uncompressed_bytes INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """INSERT INTO notebook_artifacts VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "artifact-1",
                "training_curves",
                "rid",
                "curve_json",
                artifact_rel,
                "zstd",
                "application/json",
                _sha256(raw),
                _sha256(compressed),
                len(raw),
                len(compressed),
                1.0,
            ),
        )

    bundle = create_bundle(
        root=root,
        staging_root=root / "tmp" / "db-backup-upload",
        include_live_db=True,
        include_artifacts=True,
        latest_only=True,
    )
    staging = root / "tmp" / "db-backup-upload" / bundle["stamp"]

    report = verify_bundle(
        bundle=staging / "db-backups.tar.zst",
        manifest=staging / "manifest.json",
        extract_dir=tmp_path / "extract",
    )

    assert "research/runs.db" in report["db_checks"]
    assert "research/lab_notebook.db" in report["db_checks"]
    assert report["artifact_checks"]["checked"] == 1
