from __future__ import annotations

from research.scientist.notebook.notebook_references import _ReferencesMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("reference readers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeReferences(_ReferencesMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


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
