from __future__ import annotations

from research.scientist.leaderboard_scoring import compute_composite_v14


def _score_for_nb(value: float) -> float:
    result = compute_composite_v14(
        decompose=True,
        tier="validation",
        language_control_s10_binding_score=value,
    )
    return result["breakdown"]["cl_s10_nb_bucket"]


def test_language_control_s10_nb_bucket_points() -> None:
    assert _score_for_nb(0.64) == 0.0
    assert _score_for_nb(0.65) == 6.25
    assert _score_for_nb(0.75) == 12.5
    assert _score_for_nb(0.85) == 18.75
    assert _score_for_nb(0.95) == 25.0


def test_language_control_investigation_binding_bucket_points() -> None:
    result = compute_composite_v14(
        decompose=True,
        tier="validation",
        language_control_investigation_binding_score=0.91,
    )

    assert result["breakdown"]["cl_investigation_nb_bucket"] == 18.75
