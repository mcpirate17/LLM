"""Streaming contracts for notebook mixin readers.

Every reader must iterate the sqlite cursor (never fetchall()) and decode
its JSON side-columns. One file per mixin used to repeat the same fake
cursor/conn scaffolding seven times; consolidated 2026-06-12.
"""

from __future__ import annotations

from research.scientist.notebook import notebook_core as core
from research.scientist.notebook.notebook_advanced_analytics import (
    _AdvancedAnalyticsMixin,
)
from research.scientist.notebook.notebook_analytics import _AnalyticsMixin
from research.scientist.notebook.notebook_campaigns import _CampaignsMixin
from research.scientist.notebook.notebook_entries import _EntriesMixin
from research.scientist.notebook.notebook_knowledge import _KnowledgeMixin
from research.scientist.notebook.notebook_references import _ReferencesMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("streaming readers should not call fetchall()")


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


class _FakeAnalytics(_AnalyticsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def flush_writes(self):
        return None

    def _ensure_graph_features(self):
        return None


class _FakeCampaigns(_CampaignsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


class _FakeEntries(_EntriesMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


class _FakeKnowledge(_KnowledgeMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


class _FakeReferences(_ReferencesMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


class _FakeAdvancedAnalytics(_AdvancedAnalyticsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


class _FakeNotebookCore(core._NotebookCore):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)
        self._batch_depth = 0
        self._dashboard_summary_cache = {}
        self._dashboard_summary_cache_expires_at = 0.0
        self._template_observability_cache = {}
        self._template_observability_cache_expires_at = 0.0


# ── analytics ────────────────────────────────────────────────────────


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


# ── campaigns ────────────────────────────────────────────────────────


def test_get_campaign_hypotheses_streams_and_decodes_metadata():
    campaigns = _FakeCampaigns(
        [
            {
                "hypothesis_id": "hyp1",
                "metadata_json": '{"priority": "high"}',
            }
        ]
    )

    rows = campaigns.get_campaign_hypotheses("camp1")

    assert rows == [
        {
            "hypothesis_id": "hyp1",
            "metadata_json": '{"priority": "high"}',
            "metadata": {"priority": "high"},
        }
    ]


def test_get_metrics_streams_rows_without_fetchall():
    campaigns = _FakeCampaigns(
        [
            {
                "metric_name": "loss",
                "metric_value": 0.5,
            }
        ]
    )

    rows = campaigns.get_metrics("loss", limit=10)

    assert rows == [{"metric_name": "loss", "metric_value": 0.5}]


# ── entries ──────────────────────────────────────────────────────────


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


# ── knowledge ────────────────────────────────────────────────────────


def test_get_insights_streams_and_decodes_evidence_json():
    knowledge = _FakeKnowledge(
        [
            {
                "content": "test evidence",
                "evidence_json": '{"test":"fisher_exact","p_value":0.001}',
            }
        ]
    )

    rows = knowledge.get_insights(limit=1)

    assert rows == [
        {
            "content": "test evidence",
            "evidence_json": {"test": "fisher_exact", "p_value": 0.001},
        }
    ]


def test_get_pending_selection_insight_trials_streams_and_decodes_json_fields():
    knowledge = _FakeKnowledge(
        [
            {
                "trial_id": "trial1",
                "insight_ids_json": '["i1","i2"]',
                "chosen_result_ids_json": '["r1"]',
                "metadata_json": '{"reason":"test"}',
            }
        ]
    )

    rows = knowledge.get_pending_selection_insight_trials(limit=10)

    assert rows == [
        {
            "trial_id": "trial1",
            "insight_ids_json": ["i1", "i2"],
            "chosen_result_ids_json": ["r1"],
            "metadata_json": {"reason": "test"},
        }
    ]


# ── references ───────────────────────────────────────────────────────


def test_get_decisions_streams_and_decodes_json_fields():
    refs = _FakeReferences(
        [
            {
                "decision_type": "next_experiment_plan",
                "evidence_ids": '["e1"]',
                "alternatives_considered": '[{"mode": "synthesis"}]',
                "evidence_pack_json": '{"mode": "refinement"}',
            }
        ]
    )

    rows = refs.get_decisions(decision_type="next_experiment_plan")

    assert rows == [
        {
            "decision_type": "next_experiment_plan",
            "evidence_ids": ["e1"],
            "alternatives_considered": [{"mode": "synthesis"}],
            "evidence_pack_json": '{"mode": "refinement"}',
            "evidence_pack": {"mode": "refinement"},
        }
    ]


def test_get_selection_decisions_streams_and_decodes_json_fields():
    refs = _FakeReferences(
        [
            {
                "context": "mode_selection",
                "candidate_pool_summary_json": '{"candidate_count": 2}',
                "score_breakdown_json": '[{"score": 0.8}]',
                "policy_json": '{"name": "ucb"}',
                "chosen_experiments_json": '[{"experiment_id": "exp1"}]',
                "trigger_json": '{"triggered": true}',
            }
        ]
    )

    rows = refs.get_selection_decisions(context="mode_selection", limit=5)

    assert rows == [
        {
            "context": "mode_selection",
            "candidate_pool_summary_json": {"candidate_count": 2},
            "score_breakdown_json": [{"score": 0.8}],
            "policy_json": {"name": "ucb"},
            "chosen_experiments_json": [{"experiment_id": "exp1"}],
            "trigger_json": {"triggered": True},
        }
    ]


# ── advanced analytics ───────────────────────────────────────────────


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


# ── notebook core ────────────────────────────────────────────────────


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
