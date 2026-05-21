from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.scientist.api_routes import _helpers as api_helpers
from research.scientist.api_routes import _strategy_briefing
from research.scientist.notebook.notebook_leaderboard import _LeaderboardMixin
from research.scientist.runner.execution_experiment_phase3 import (
    _ExecutionExperimentPhase3Mixin,
)
from research.scientist.runner import _helpers_metrics

pytestmark = pytest.mark.unit


def _raise_runtime_error(*_args, **_kwargs):
    raise RuntimeError("telemetry failed")


def test_fingerprint_lookup_propagates_query_failures():
    class _Conn:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("lookup failed")

    nb = SimpleNamespace(conn=_Conn())

    with pytest.raises(RuntimeError, match="lookup failed"):
        _ExecutionExperimentPhase3Mixin()._lookup_existing_fingerprints(nb, {"fp1"})


def test_native_runner_progress_propagates_telemetry_failures(monkeypatch):
    monkeypatch.setattr(
        api_helpers,
        "native_runner_capability_report",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="telemetry failed"):
        api_helpers.with_native_runner_progress({"status": "running"})


def test_runner_native_progress_report_propagates_telemetry_failures(monkeypatch):
    monkeypatch.setattr(
        "research.scientist.native.telemetry.native_runner_capability_report",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="telemetry failed"):
        _helpers_metrics._native_runner_progress_report()


def test_briefing_data_is_recomputed_for_each_notebook(monkeypatch):
    monkeypatch.setattr(
        _strategy_briefing,
        "compute_compression_opportunities",
        lambda _coverage: {"summary": {}},
    )
    monkeypatch.setattr(_strategy_briefing, "compute_sparse_evidence", lambda _nb: {})
    monkeypatch.setattr(_strategy_briefing, "sparse_coverage_summary", lambda _data: {})
    monkeypatch.setattr(
        _strategy_briefing,
        "_build_recommendation_evidence",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(_strategy_briefing, "_build_ref_comparison", lambda _nb: {})

    class _Rows:
        def fetchall(self):
            return []

    class _Conn:
        def execute(self, *_args, **_kwargs):
            return _Rows()

    class _Notebook:
        conn = _Conn()

        def __init__(self, total_experiments):
            self.total_experiments = total_experiments

        def get_dashboard_headline_summary(self):
            return {
                "total_experiments": self.total_experiments,
                "total_programs_evaluated": 0,
                "stage1_survivors": 0,
            }

    class _Analytics:
        def learning_trajectory(self):
            return {}

        def compression_coverage(self):
            return {}

        def compression_primitive_effectiveness(self):
            return {}

        def sparse_coverage(self):
            return {}

    first = _strategy_briefing.gather_briefing_data(_Notebook(1), _Analytics(), [])
    second = _strategy_briefing.gather_briefing_data(_Notebook(2), _Analytics(), [])

    assert first["summary"]["total_experiments"] == 1
    assert second["summary"]["total_experiments"] == 2


def test_wikitext_improvement_alias_keeps_zero_primary_value():
    entry = {
        "wikitext_ppl_improvement_ratio": 0.0,
        "wikitext_improvement_ratio": 0.75,
    }

    _LeaderboardMixin()._apply_wikitext_improvement_alias(entry, {})

    assert entry["improvement_ratio"] == 0.0
