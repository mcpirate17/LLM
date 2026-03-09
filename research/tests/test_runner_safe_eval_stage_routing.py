from __future__ import annotations

import os
from contextlib import contextmanager
import threading
from unittest.mock import patch

import pytest

from research.scientist.runner import ExperimentRunner, LiveProgress

pytestmark = pytest.mark.unit


@contextmanager
def _env(**values):
    prev = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def test_safe_eval_stage_default_probe_enabled_primary_disabled():
    runner = ExperimentRunner.__new__(ExperimentRunner)

    with _env(
        NATIVE_RUNNER_ABI_PRIMARY_STAGES="",
        NATIVE_RUNNER_ABI_PROBE_STAGES="",
    ), patch("research.scientist.runner.screening.safe_eval", return_value=object()) as mocked:
        runner._safe_eval_for_stage(
            object(),
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=8,
            vocab_size=128,
            device="cpu",
        )
        kwargs = mocked.call_args.kwargs
        assert kwargs["abi_infer_probe"] is True
        assert kwargs["abi_infer_primary"] is False


def test_safe_eval_stage_primary_routing_for_selected_stage():
    runner = ExperimentRunner.__new__(ExperimentRunner)

    with _env(
        NATIVE_RUNNER_ABI_PRIMARY_STAGES="candidate_screening,evolution_fitness",
        NATIVE_RUNNER_ABI_PROBE_STAGES="candidate_screening",
    ), patch("research.scientist.runner.screening.safe_eval", return_value=object()) as mocked:
        runner._safe_eval_for_stage(
            object(),
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=8,
            vocab_size=128,
            device="cpu",
        )
        kwargs = mocked.call_args.kwargs
        assert kwargs["abi_infer_probe"] is True
        assert kwargs["abi_infer_primary"] is True


def test_safe_eval_stage_probe_can_be_disabled_per_stage():
    runner = ExperimentRunner.__new__(ExperimentRunner)

    with _env(
        NATIVE_RUNNER_ABI_PRIMARY_STAGES="*",
        NATIVE_RUNNER_ABI_PROBE_STAGES="ablation",
    ), patch("research.scientist.runner.screening.safe_eval", return_value=object()) as mocked:
        runner._safe_eval_for_stage(
            object(),
            stage_tag="graph_candidate_gen",
            batch_size=2,
            seq_len=8,
            vocab_size=128,
            device="cpu",
        )
        kwargs = mocked.call_args.kwargs
        assert kwargs["abi_infer_probe"] is False
        assert kwargs["abi_infer_primary"] is True


def test_safe_eval_stage_updates_progress_with_abi_probe_payload():
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner._lock = threading.RLock()
    runner._progress = LiveProgress()

    class _Res:
        native_abi_probe = {"attempted": True, "succeeded": True, "parity_pass": True}

    with _env(
        NATIVE_RUNNER_ABI_PRIMARY_STAGES="candidate_screening",
        NATIVE_RUNNER_ABI_PROBE_STAGES="candidate_screening",
    ), patch("research.scientist.runner.screening.safe_eval", return_value=_Res()):
        runner._safe_eval_for_stage(
            object(),
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=8,
            vocab_size=128,
            device="cpu",
        )

    native_runner = runner._progress.native_runner or {}
    assert native_runner.get("abi_last_stage") == "candidate_screening"
    assert (native_runner.get("abi_last_probe") or {}).get("parity_pass") is True


def test_safe_eval_stage_records_parity_result_when_sampled():
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner._lock = threading.RLock()
    runner._progress = LiveProgress()

    class _Res:
        native_abi_probe = {"attempted": True, "parity_attempted": True, "parity_pass": False}

    with _env(
        NATIVE_RUNNER_ABI_PRIMARY_STAGES="candidate_screening",
        NATIVE_RUNNER_ABI_PROBE_STAGES="candidate_screening",
    ), patch("research.scientist.runner.screening.safe_eval", return_value=_Res()), patch(
        "research.scientist.runner.record_native_abi_parity_result"
    ) as mock_record:
        runner._safe_eval_for_stage(
            object(),
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=8,
            vocab_size=128,
            device="cpu",
        )

        mock_record.assert_called_once_with(False)
