from research.tools.shadow_screening_gating_audit import (
    TimingEstimate,
    _summarize_policy,
)


def test_shadow_policy_miss_uses_score_without_blimp():
    rows = [
        {
            "result_id": "still-passes",
            "current_score": 70.0,
            "score_without_blimp": 65.0,
            "cheap_soft_signal": False,
        },
        {
            "result_id": "would-miss",
            "current_score": 70.0,
            "score_without_blimp": 55.0,
            "cheap_soft_signal": False,
        },
    ]
    summary = _summarize_policy(
        rows,
        threshold=62.7,
        policy="never",
        timing=TimingEstimate(current_ms=100.0, no_blimp_ms=60.0, blimp_ms=40.0),
    )
    assert summary["current_pass_count"] == 2
    assert summary["shadow_pass_count"] == 1
    assert summary["missed_current_pass_count"] == 1
    assert summary["missed_examples"][0]["result_id"] == "would-miss"


def test_shadow_policy_margin_preserves_threshold_passes():
    rows = [
        {
            "result_id": "near-threshold",
            "current_score": 70.0,
            "score_without_blimp": 55.0,
            "cheap_soft_signal": False,
        }
    ]
    summary = _summarize_policy(
        rows,
        threshold=62.7,
        policy="soft_signal_or_within_10",
        timing=TimingEstimate(current_ms=100.0, no_blimp_ms=60.0, blimp_ms=40.0),
    )
    assert summary["run_blimp_count"] == 1
    assert summary["missed_current_pass_count"] == 0
