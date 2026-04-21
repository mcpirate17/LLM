from __future__ import annotations

import pytest

from research.scientist.api_routes._ml_influence_status import build_ml_influence_status


pytestmark = pytest.mark.unit


def test_ml_influence_status_resolves_screening_floor_from_saved_report(monkeypatch):
    monkeypatch.setattr(
        "research.scientist.api_routes._ml_influence_status.load_predictor_metrics_report",
        lambda: {
            "ensemble_calibrated": {
                "saved_runtime_artifact_evaluation": {"selected_threshold": 0.27}
            }
        },
    )

    status = build_ml_influence_status()

    assert status["defaults"][
        "resolved_screening_ensemble_p_pass_floor"
    ] == pytest.approx(0.27)
    assert (
        status["defaults"]["resolved_screening_ensemble_p_pass_floor_source"]
        == "ensemble.saved_runtime_artifact_evaluation.selected_threshold"
    )


def test_ml_influence_status_prefers_temporal_f1_floor(monkeypatch):
    monkeypatch.setattr(
        "research.scientist.api_routes._ml_influence_status.load_predictor_metrics_report",
        lambda: {
            "ensemble_calibrated": {
                "saved_runtime_artifact_evaluation": {"selected_threshold": 0.27},
                "temporal_holdout_evaluation": {
                    "operating_points": {"f1": {"threshold": 0.32}}
                },
            }
        },
    )

    status = build_ml_influence_status()

    assert status["defaults"][
        "resolved_screening_ensemble_p_pass_floor"
    ] == pytest.approx(0.32)
    assert (
        status["defaults"]["resolved_screening_ensemble_p_pass_floor_source"]
        == "ensemble.temporal_holdout_evaluation.operating_points.f1.threshold"
    )
