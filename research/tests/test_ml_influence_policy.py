from __future__ import annotations

import pytest

from research.scientist.ml_influence_policy import build_ml_influence_policy
from research.scientist.runner import RunConfig


pytestmark = pytest.mark.unit


def test_policy_prefers_saved_runtime_artifact_metrics_over_legacy_aliases():
    config = RunConfig()
    config.gbm_prescreener_enabled = True

    report = {
        "ensemble_calibrated": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {
                    "roc_auc": 0.82,
                    "precision_ppv": 0.46,
                }
            },
            "val_metrics_selected_threshold": {
                "roc_auc": 0.51,
                "precision_ppv": 0.21,
            },
        },
        "gbm_gate": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {"roc_auc": 0.79}
            },
            "val_metrics_selected_threshold": {"roc_auc": 0.12},
        },
        "graph_predictor": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {"roc_auc": 0.76}
            },
            "derived_val_classification_metrics": {"roc_auc": 0.11},
        },
        "investigation_predictor": {
            "spearman_rho": 0.1,
            "n_test": 50,
        },
    }

    policy = build_ml_influence_policy(config=config, report=report)

    assert policy["components"]["screening_ensemble"]["proven"] is True
    assert policy["components"]["screening_ensemble"]["allowed"] is True
    assert policy["components"]["screening_ensemble"]["roc_auc"] == pytest.approx(0.82)
    assert policy["components"]["screening_ensemble"]["precision_ppv"] == pytest.approx(
        0.46
    )
    assert policy["components"]["gbm_gate"]["roc_auc"] == pytest.approx(0.79)
    assert policy["components"]["gbm_gate"]["quality_tier"] == "usable"
    assert policy["components"]["graph_predictor"]["roc_auc"] == pytest.approx(0.76)
    assert policy["components"]["graph_predictor"]["quality_tier"] == "usable"


def test_policy_prefers_temporal_metrics_when_present():
    config = RunConfig()
    config.gbm_prescreener_enabled = True

    report = {
        "ensemble_calibrated": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {
                    "roc_auc": 0.91,
                    "precision_ppv": 0.62,
                }
            },
            "temporal_holdout_evaluation": {
                "selected_metrics": {
                    "roc_auc": 0.78,
                    "precision_ppv": 0.38,
                }
            },
        },
        "gbm_gate": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {"roc_auc": 0.88}
            },
            "temporal_holdout_evaluation": {"selected_metrics": {"roc_auc": 0.80}},
        },
        "graph_predictor": {
            "saved_runtime_artifact_evaluation": {
                "selected_metrics": {"roc_auc": 0.77}
            },
            "temporal_holdout_evaluation": {"selected_metrics": {"roc_auc": 0.53}},
        },
        "investigation_predictor": {
            "spearman_rho": 0.27,
            "n_test": 150,
        },
    }

    policy = build_ml_influence_policy(config=config, report=report)

    assert policy["components"]["screening_ensemble"]["proven"] is False
    assert policy["components"]["screening_ensemble"]["allowed"] is False
    assert policy["components"]["screening_ensemble"]["roc_auc"] == pytest.approx(0.78)
    assert (
        policy["components"]["screening_ensemble"]["metric_source"]
        == "temporal_holdout_evaluation"
    )
    assert policy["components"]["gbm_gate"]["roc_auc"] == pytest.approx(0.80)
    assert (
        policy["components"]["gbm_gate"]["metric_source"]
        == "temporal_holdout_evaluation"
    )
    assert policy["components"]["graph_predictor"]["roc_auc"] == pytest.approx(0.53)
    assert policy["components"]["graph_predictor"]["quality_tier"] == "weak"


def test_policy_falls_back_to_legacy_metrics_when_saved_eval_is_missing():
    config = RunConfig()
    config.gbm_prescreener_enabled = True

    report = {
        "ensemble_calibrated": {
            "val_metrics_selected_threshold": {
                "roc_auc": 0.81,
                "precision_ppv": 0.45,
            }
        },
        "gbm_gate": {"val_metrics_selected_threshold": {"roc_auc": 0.77}},
        "graph_predictor": {"derived_val_classification_metrics": {"roc_auc": 0.74}},
        "investigation_predictor": {
            "spearman_rho": 0.27,
            "n_test": 150,
        },
    }

    policy = build_ml_influence_policy(config=config, report=report)

    assert policy["components"]["screening_ensemble"]["proven"] is True
    assert policy["components"]["screening_ensemble"]["allowed"] is True
    assert policy["components"]["screening_ensemble"]["roc_auc"] == pytest.approx(0.81)
    assert policy["components"]["gbm_gate"]["roc_auc"] == pytest.approx(0.77)
    assert policy["components"]["graph_predictor"]["roc_auc"] == pytest.approx(0.74)
    assert policy["components"]["investigation_predictor"]["proven"] is True
