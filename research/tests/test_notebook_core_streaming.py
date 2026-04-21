from __future__ import annotations

from research.scientist.notebook import notebook_core as core


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("core backfill should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.executemany_calls = []

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)

    def executemany(self, sql, seq_of_parameters):
        self.executemany_calls.append((sql, list(seq_of_parameters)))

    def commit(self):
        return None


class _FakeNotebookCore(core._NotebookCore):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)
        self._batch_depth = 0
        self._dashboard_summary_cache = {}
        self._dashboard_summary_cache_expires_at = 0.0
        self._template_observability_cache = {}
        self._template_observability_cache_expires_at = 0.0


def test_backfill_missing_graph_features_streams_cursor_rows(monkeypatch):
    notebook = _FakeNotebookCore(
        [
            {
                "result_id": "rid1",
                "graph_fingerprint": "fp1",
                "graph_json": '{"nodes": {"0": {"op_name": "gelu"}}}',
            }
        ]
    )

    monkeypatch.setattr(
        core,
        "build_graph_feature_rows",
        lambda **_kwargs: {
            "feature_row": (
                "rid1",
                "fp1",
                "tpl",
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                "[]",
                "[]",
                "[]",
            ),
            "op_rows": [("rid1", "fp1", "gelu")],
            "pair_rows": [("rid1", "fp1", "gelu->gelu")],
        },
    )

    count = notebook._backfill_missing_graph_features(limit=10)

    assert count == 1
    assert len(notebook.conn.executemany_calls) == 5
