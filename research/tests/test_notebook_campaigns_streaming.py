from __future__ import annotations

from research.scientist.notebook.notebook_campaigns import _CampaignsMixin


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("campaign readers should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeCampaigns(_CampaignsMixin):
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


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
