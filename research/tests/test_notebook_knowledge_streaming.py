from __future__ import annotations

from research.scientist.notebook.notebook_knowledge import _KnowledgeMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("knowledge readers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeKnowledge(_KnowledgeMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


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
