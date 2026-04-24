"""Smoke tests for the investigation thread split methods.

Validates that the extracted helper methods on _ExecutionInvestigationMixin
behave correctly in isolation.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
from unittest.mock import patch


from research.scientist.runner.execution_investigation import (
    _ExecutionInvestigationMixin,
    _SKIP_INFRA,
)


class _StubRunner(_ExecutionInvestigationMixin):
    """Concrete subclass without __slots__ so we can set attributes."""

    def __init__(self, **overrides):
        self._update_progress = MagicMock()
        self._emit_event = MagicMock()
        self._publish_terminal_event = MagicMock()
        self._fail_experiment_compat = MagicMock()
        self._live_training_context = None
        for k, v in overrides.items():
            setattr(self, k, v)


def _make_mixin(**overrides) -> _ExecutionInvestigationMixin:
    """Create a minimal mixin instance with mocked self.* dependencies."""
    return _StubRunner(**overrides)


class TestInvestigateFingerprintCompletion:
    """Tests for _investigate_fingerprint_completion."""

    def test_returns_expected_types_with_none_model(self):
        """When best_inv_model is None, should return (False, False, source)."""
        mixin = _make_mixin()
        source = {"_behavioral_fingerprint": {"some": "data"}, "result_id": "abc123"}
        completed, attempted, returned_source = (
            mixin._investigate_fingerprint_completion(
                source_result_id="abc12345",
                source=source,
                best_inv_model=None,
                config=MagicMock(max_seq_len=128, model_dim=256, vocab_size=32000),
                dev=MagicMock(),
                nb=MagicMock(),
            )
        )
        assert completed is False
        assert attempted is False
        assert returned_source is source

    def test_returns_expected_types_with_none_fp_dict(self):
        """When source has no _behavioral_fingerprint, should skip."""
        mixin = _make_mixin()
        source = {"result_id": "abc123"}  # no _behavioral_fingerprint
        model = MagicMock()
        completed, attempted, returned_source = (
            mixin._investigate_fingerprint_completion(
                source_result_id="abc12345",
                source=source,
                best_inv_model=model,
                config=MagicMock(max_seq_len=128, model_dim=256, vocab_size=32000),
                dev=MagicMock(),
                nb=MagicMock(),
            )
        )
        assert completed is False
        assert attempted is False
        assert returned_source is source

    def test_returns_expected_types_with_both_none(self):
        """When both model and fp_dict are None, should return early."""
        mixin = _make_mixin()
        source = {"result_id": "abc123"}
        completed, attempted, returned_source = (
            mixin._investigate_fingerprint_completion(
                source_result_id="abc12345",
                source=source,
                best_inv_model=None,
                config=MagicMock(max_seq_len=128, model_dim=256, vocab_size=32000),
                dev=MagicMock(),
                nb=MagicMock(),
            )
        )
        assert isinstance(completed, bool)
        assert isinstance(attempted, bool)
        assert completed is False
        assert attempted is False


class TestHandleInvestigationInfraFailure:
    """Tests for _handle_investigation_infra_failure."""

    def test_all_infra_failure_returns_true(self):
        """When all candidates failed with infra errors, should return True."""
        nb = MagicMock()
        mixin = _make_mixin()

        results = {
            "investigation_results": [],  # no successful results
            "infra_failures": [
                {
                    "result_id": "abc12345",
                    "n_programs": 3,
                    "errors": ["CUDA error: device-side assert triggered"],
                },
                {
                    "result_id": "def67890",
                    "n_programs": 3,
                    "errors": ["out of memory"],
                },
            ],
        }

        result = mixin._handle_investigation_infra_failure(
            exp_id="exp_test_12345678",
            results=results,
            nb=nb,
            t_start=time.time(),
        )

        assert result is True
        mixin._fail_experiment_compat.assert_called_once()
        nb.flush_writes.assert_called_once()
        mixin._update_progress.assert_called_once()
        mixin._emit_event.assert_called_once_with(
            "investigation_completed",
            {
                "experiment_id": "exp_test_12345678",
                "status": "infra_error",
                "infra_failures": results["infra_failures"],
            },
        )

    def test_no_infra_failure_returns_false(self):
        """When there are normal investigation results, should return False."""
        nb = MagicMock()
        mixin = _make_mixin()

        results = {
            "investigation_results": [{"result_id": "abc", "robustness": 0.8}],
            "infra_failures": [],
        }

        result = mixin._handle_investigation_infra_failure(
            exp_id="exp_test_12345678",
            results=results,
            nb=nb,
            t_start=time.time(),
        )

        assert result is False
        nb.fail_experiment.assert_not_called()

    def test_mixed_results_returns_false(self):
        """When there are both real results and infra failures, not all-infra."""
        nb = MagicMock()
        mixin = _make_mixin()

        results = {
            "investigation_results": [{"result_id": "abc", "robustness": 0.5}],
            "infra_failures": [
                {"result_id": "def", "n_programs": 3, "errors": ["CUDA error"]},
            ],
        }

        result = mixin._handle_investigation_infra_failure(
            exp_id="exp_test_12345678",
            results=results,
            nb=nb,
            t_start=time.time(),
        )

        assert result is False
        nb.fail_experiment.assert_not_called()


class TestSummarizeAndCheckInfra:
    """Tests for _summarize_and_check_infra."""

    def test_returns_skip_infra_when_all_infra_failures(self):
        """All programs failed with CUDA errors -> _SKIP_INFRA."""
        mixin = _make_mixin(
            _investigation_loss_multiplier=lambda s, b: None,
        )
        tp_results = [
            {
                "passed": False,
                "loss_ratio": None,
                "error": "CUDA error: device-side assert",
            },
            {"passed": False, "loss_ratio": None, "error": "CUDA out of memory"},
        ]
        results: dict = {}

        ret = mixin._summarize_and_check_infra(
            "abc12345",
            tp_results,
            {"loss_ratio": 0.5},
            MagicMock(investigation_max_loss_ratio_multiplier=3.0),
            results,
        )

        assert ret is _SKIP_INFRA
        assert "infra_failures" in results
        assert len(results["infra_failures"]) == 1

    def test_returns_summary_when_some_pass(self):
        """At least one program passed -> returns InvestigationProgramSummary."""
        mixin = _make_mixin(
            _investigation_loss_multiplier=lambda s, b: 1.0,
        )
        tp_results = [
            {"passed": True, "loss_ratio": 0.5, "error": None},
            {"passed": False, "loss_ratio": None, "error": "CUDA error"},
        ]
        results: dict = {}

        ret = mixin._summarize_and_check_infra(
            "abc12345",
            tp_results,
            {"loss_ratio": 0.5},
            MagicMock(investigation_max_loss_ratio_multiplier=3.0),
            results,
        )

        assert ret is not _SKIP_INFRA
        assert hasattr(ret, "n_passed")
        assert ret.n_passed == 1


class TestInvestigationModelReconstruction:
    """Tests for investigation model reconstruction fallback behavior."""

    def test_reconstruct_investigation_model_falls_back_for_byte_unsafe_graph(self):
        mixin = _make_mixin()
        config = MagicMock(model_dim=256, n_layers=4, vocab_size=32000)
        sentinel = object()

        with (
            patch(
                "research.scientist.runner.execution_investigation.graph_from_json",
                return_value=MagicMock(name="graph"),
            ) as graph_from_json,
            patch(
                "research.scientist.runner.execution_investigation.find_byte_safety_violations",
                return_value=["Byte-unsafe op 'mod_topk'"],
            ),
            patch(
                "research.scientist.runner.execution_investigation._compile_model_legacy",
                return_value=sentinel,
            ) as legacy_compile,
            patch(
                "research.scientist.runner.execution_investigation.compile_model",
            ) as native_compile,
        ):
            model = mixin._reconstruct_investigation_model(
                source_result_id="9cfd1d97-336",
                model_source="fingerprint_refine",
                graph_json_str="{}",
                arch_spec_json_str=None,
                config=config,
                tp_max_seq=256,
            )

        assert model is sentinel
        graph = graph_from_json.return_value
        legacy_compile.assert_called_once_with(
            [graph] * config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=256,
        )
        native_compile.assert_not_called()
