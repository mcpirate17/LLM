from __future__ import annotations

import sqlite3

from research.scientist.api_routes import _observability_core as obs
from research.scientist.notebook.artifact_store import NotebookArtifactStore


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("_load_program_rows should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeNotebook:
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


def test_load_program_rows_streams_and_normalizes_payload():
    nb = _FakeNotebook(
        [
            {
                "graph_json": '{"nodes":[{"op_name":"gelu"}]}',
                "stage0_passed": 1,
                "stage1_passed": 0,
                "loss_ratio": 0.2,
                "error_type": "shape_mismatch",
                "failure_op": "gelu",
                "failure_details_json": '{"failure_op":"gelu"}',
            },
            {
                "graph_json": "",
                "stage0_passed": 1,
                "stage1_passed": 1,
                "loss_ratio": None,
                "error_type": None,
                "failure_op": None,
                "failure_details_json": None,
            },
        ]
    )

    rows = obs._load_program_rows(nb, "all")

    assert rows == [
        {
            "graph_json": '{"nodes":[{"op_name":"gelu"}]}',
            "stage0_passed": True,
            "stage1_passed": False,
            "loss_ratio": 0.2,
            "error_type": "shape_mismatch",
            "failure_op": "gelu",
            "failure_details_json": '{"failure_op":"gelu"}',
        }
    ]


def test_load_program_rows_resolves_artifact_backed_graph_json(tmp_path):
    db_path = tmp_path / "runs.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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
    graph_json = '{"nodes":[{"op_name":"gelu"},{"op_name":"linear_proj"}]}'
    store = NotebookArtifactStore(db_path)
    metadata = store.write(
        table_name="program_results",
        row_pk="rid-artifact",
        column_name="graph_json",
        payload=graph_json,
        content_type="application/json",
    )
    conn.execute(
        """
        INSERT INTO notebook_artifacts (
            artifact_id, table_name, row_pk, column_name, path, compression,
            content_type, sha256_uncompressed, sha256_compressed,
            uncompressed_bytes, compressed_bytes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(
            metadata[key]
            for key in (
                "artifact_id",
                "table_name",
                "row_pk",
                "column_name",
                "path",
                "compression",
                "content_type",
                "sha256_uncompressed",
                "sha256_compressed",
                "uncompressed_bytes",
                "compressed_bytes",
                "created_at",
            )
        ),
    )
    conn.commit()
    pointer = (
        '{"_notebook_artifact":"'
        + metadata["artifact_id"]
        + '","compression":"zstd","path":"'
        + metadata["path"]
        + '"}'
    )

    class _Notebook:
        def __init__(self):
            self.conn = conn
            self.db_path = db_path

    rows = obs._load_program_rows(
        _Notebook(),
        "all",
        db_path,
    )
    assert rows == []

    class _Cursor:
        def __iter__(self):
            return iter(
                [
                    {
                        "graph_json": pointer,
                        "stage0_passed": 1,
                        "stage1_passed": 1,
                        "loss_ratio": 0.3,
                        "error_type": None,
                        "failure_op": None,
                        "failure_details_json": None,
                    }
                ]
            )

    class _Conn:
        def execute(self, query, params=()):
            if "FROM program_results" in query:
                return _Cursor()
            return conn.execute(query, params)

    nb = _Notebook()
    nb.conn = _Conn()
    rows = obs._load_program_rows(nb, "all", db_path)
    assert rows[0]["graph_json"] == graph_json
    conn.close()
