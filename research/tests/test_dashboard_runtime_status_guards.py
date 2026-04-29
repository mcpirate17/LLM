from __future__ import annotations

from concurrent.futures import Future
import threading
from unittest.mock import MagicMock, patch

from research.scientist.api_routes import _helpers
from research.scientist.runner.core import _CoreMixin


def test_resolve_runner_status_does_not_start_projector_by_default():
    nb = MagicMock()
    runner = MagicMock()
    runner.is_running = False
    runner.progress.to_dict.return_value = {"status": "idle"}

    with (
        patch.object(
            _helpers,
            "get_registry_running_experiment_snapshot",
            return_value=None,
        ),
        patch.object(
            _helpers,
            "get_external_running_experiment_snapshot",
            return_value=None,
        ),
        patch.object(
            _helpers,
            "get_projected_running_experiment_snapshot",
        ) as projected,
    ):
        result = _helpers.resolve_runner_status(nb, runner)

    assert result["is_running"] is False
    projected.assert_not_called()


def test_investigation_eval_future_blocks_completion_until_done():
    runner = object.__new__(_CoreMixin)
    runner._lock = threading.Lock()
    runner._pending_investigation_eval_futures = []

    future = Future()
    runner._register_investigation_eval_future(
        exp_id="exp-1",
        future=future,
        kind="benchmark",
        source_result_id="rid-1",
    )

    timed_out = runner._wait_for_investigation_eval_futures("exp-1", timeout_s=0)
    assert timed_out == [
        {
            "source_result_id": "rid-1",
            "kind": "benchmark",
            "status": "timed_out",
        }
    ]

    future = Future()
    future.set_result(None)
    runner._register_investigation_eval_future(
        exp_id="exp-1",
        future=future,
        kind="v2-probe",
        source_result_id="rid-2",
    )

    completed = runner._wait_for_investigation_eval_futures("exp-1", timeout_s=1)
    assert completed == [
        {
            "source_result_id": "rid-2",
            "kind": "v2-probe",
            "status": "completed",
        }
    ]
    assert runner._pending_investigation_eval_futures == []
