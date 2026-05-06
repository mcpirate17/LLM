import json

from research.scientist.api_routes._fingerprint_failures import (
    fingerprint_failure_summary,
)


def _entry(tier: str) -> dict:
    return {
        "tier": tier,
        "fingerprint_json": json.dumps(
            {
                "fingerprint_completed_post_investigation": False,
                "cka_source": "deferred",
            }
        ),
    }


def test_screening_deferred_post_investigation_fingerprint_is_pending_not_failed():
    summary = fingerprint_failure_summary(_entry("screening"))

    assert summary["failed"] is False
    assert not any(
        check["field"] == "fingerprint_completed_post_investigation"
        for check in summary["failed_checks"]
    )


def test_exact_replay_deferred_post_investigation_fingerprint_is_not_failed():
    entry = _entry("validation")
    entry["model_source"] = "exact_graph_replay"

    summary = fingerprint_failure_summary(entry)

    assert summary["failed"] is False
    assert not any(
        check["field"] == "fingerprint_completed_post_investigation"
        for check in summary["failed_checks"]
    )


def test_backfill_observation_deferred_post_investigation_fingerprint_is_not_failed():
    entry = _entry("validation")
    entry["trust_label"] = "backfill_observation"
    entry["result_cohort"] = "backfill"

    summary = fingerprint_failure_summary(entry)

    assert summary["failed"] is False
    assert not any(
        check["field"] == "fingerprint_completed_post_investigation"
        for check in summary["failed_checks"]
    )


def test_validation_deferred_post_investigation_fingerprint_is_failed():
    summary = fingerprint_failure_summary(_entry("validation"))

    assert summary["failed"] is True
    assert {
        "field": "fingerprint_completed_post_investigation",
        "label": "Post-investigation fingerprint",
        "status": "incomplete",
    } in summary["failed_checks"]


def test_incomplete_investigation_tier_marks_post_investigation_fingerprint_failed():
    summary = fingerprint_failure_summary(
        _entry("investigation_fingerprint_incomplete")
    )

    assert summary["failed"] is True
    assert {
        "field": "fingerprint_completed_post_investigation",
        "label": "Post-investigation fingerprint",
        "status": "incomplete",
    } in summary["failed_checks"]
