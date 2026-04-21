from __future__ import annotations

from research.scientist.notebook.notebook_advanced_analytics import (
    _AdvancedAnalyticsMixin,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("advanced analytics readers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeAdvancedAnalytics(_AdvancedAnalyticsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


def test_list_scaffold_profile_results_streams_and_decodes_metrics():
    analytics = _FakeAdvancedAnalytics(
        [
            {
                "run_id": "run123",
                "family": "gpt2_attn",
                "metrics_json": '{"loss_ratio": 0.52, "sandbox_passed": true}',
            }
        ]
    )

    rows = analytics.list_scaffold_profile_results(run_id="run123", limit=10)

    assert rows == [
        {
            "run_id": "run123",
            "family": "gpt2_attn",
            "metrics_json": '{"loss_ratio": 0.52, "sandbox_passed": true}',
            "metrics": {"loss_ratio": 0.52, "sandbox_passed": True},
        }
    ]


def test_get_attribution_reports_streams_and_decodes_json_fields():
    analytics = _FakeAdvancedAnalytics(
        [
            {
                "supporting_experiments": '["exp1"]',
                "ablation_experiments": '["exp2"]',
                "report_json": '{"winner": "exp1"}',
            }
        ]
    )

    rows = analytics.get_attribution_reports(limit=10)

    assert rows == [
        {
            "supporting_experiments": ["exp1"],
            "ablation_experiments": ["exp2"],
            "report_json": {"winner": "exp1"},
        }
    ]
