from __future__ import annotations

from research.scientist.notebook.notebook_entries import _EntriesMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("entry readers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeEntries(_EntriesMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


def test_get_training_curve_streams_rows_without_fetchall():
    entries = _FakeEntries(
        [
            {"step": 1, "loss": 0.5, "grad_norm": 1.2, "step_time_ms": 3.4},
            {"step": 2, "loss": 0.4, "grad_norm": 1.0, "step_time_ms": 3.2},
        ]
    )

    rows = entries.get_training_curve("rid1")

    assert rows == [
        {"step": 1, "loss": 0.5, "grad_norm": 1.2, "step_time_ms": 3.4},
        {"step": 2, "loss": 0.4, "grad_norm": 1.0, "step_time_ms": 3.2},
    ]


def test_get_entries_streams_rows_without_fetchall():
    entries = _FakeEntries([{"entry_type": "note", "content": "hello"}])

    rows = entries.get_entries(limit=10)

    assert rows == [{"entry_type": "note", "content": "hello"}]
