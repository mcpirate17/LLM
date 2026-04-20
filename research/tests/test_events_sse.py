from __future__ import annotations

from unittest.mock import patch

from research.scientist.api_routes import events_bp


class _RunnerStub:
    def __init__(self, events):
        self._events = list(events)
        self.calls = []

    def get_events(self, timeout):
        self.calls.append(timeout)
        return list(self._events)


def test_sse_stream_idles_without_busy_loop():
    sleeps = []
    with (
        patch.object(events_bp, "get_runner", side_effect=[None, None]),
        patch.object(events_bp.time, "sleep", side_effect=lambda s: sleeps.append(s)),
    ):
        stream = events_bp._iter_sse_events(
            notebook_path="research/lab_notebook.db",
            sse_timeout=30.0,
        )
        assert next(stream) == "event: keepalive\ndata: {}\n\n"
        assert next(stream) == "event: keepalive\ndata: {}\n\n"
    assert sleeps == [5.0]


def test_sse_stream_rechecks_runner_after_idle_tick():
    runner = _RunnerStub([{"type": "progress", "data": {"step": 3}}])
    with (
        patch.object(events_bp, "get_runner", side_effect=[None, runner]),
        patch.object(events_bp.time, "sleep"),
    ):
        stream = events_bp._iter_sse_events(
            notebook_path="research/lab_notebook.db",
            sse_timeout=7.0,
        )
        assert next(stream) == "event: keepalive\ndata: {}\n\n"
        assert next(stream) == 'event: progress\ndata: {"step":3}\n\n'
        assert runner.calls == [7.0]
