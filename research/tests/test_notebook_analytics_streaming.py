from __future__ import annotations

from research.scientist.notebook.notebook_analytics import _AnalyticsMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("streaming helpers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeAnalytics(_AnalyticsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def flush_writes(self):
        return None

    def _ensure_graph_features(self):
        return None


def test_query_op_stats_sql_streams_cursor_rows_without_fetchall():
    analytics = _FakeAnalytics(
        [
            {
                "op_name": "linear_proj",
                "n_used": 4,
                "n_stage0_passed": 4,
                "n_stage05_passed": 3,
                "n_stage1_passed": 2,
                "avg_loss_ratio": 0.75,
                "avg_novelty": 0.5,
                "avg_novelty_confidence": 0.9,
            }
        ]
    )

    rows = analytics._query_op_stats_sql("1=1", ())

    assert rows == [
        {
            "op_name": "linear_proj",
            "n_used": 4,
            "n_stage0_passed": 4,
            "n_stage05_passed": 3,
            "n_stage1_passed": 2,
            "avg_loss_ratio": 0.75,
            "avg_novelty": 0.5,
            "avg_novelty_confidence": 0.9,
        }
    ]
