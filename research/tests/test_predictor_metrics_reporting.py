from __future__ import annotations

import json
import numpy as np

from research.scientist.api_routes._ml_influence_status import build_ml_influence_status
from research.scientist.ml_influence_policy import build_ml_influence_policy
from research.scientist.intelligence.metrics_utils import (
    binary_classification_metrics,
    safe_binary_roc_auc,
)
from research.tools import report_predictor_metrics


def test_safe_binary_roc_auc_returns_zero_for_single_class_labels():
    y_true = np.ones(8, dtype=np.int32)
    y_score = np.linspace(0.1, 0.9, num=8, dtype=np.float64)

    assert safe_binary_roc_auc(y_true, y_score) == 0.0
    assert binary_classification_metrics(y_true, y_score)["roc_auc"] == 0.0


def test_component_scores_uses_screening_predictor_corpus(monkeypatch):
    seen: dict[str, str] = {}

    def fake_rows(db_path: str):
        seen["db_path"] = db_path
        return [
            {
                "graph_json": {"nodes": {}},
                "stage0_any_passed": True,
                "stage1_any_passed": True,
                "canonical_fingerprint": "fp-pos-a",
            },
            {
                "graph_json": {"nodes": {}},
                "stage0_any_passed": True,
                "stage1_any_passed": True,
                "canonical_fingerprint": "fp-pos-b",
            },
            {
                "graph_json": {"nodes": {}},
                "stage0_any_passed": True,
                "stage1_any_passed": False,
                "canonical_fingerprint": "fp-neg-a",
            },
            {
                "graph_json": {"nodes": {}},
                "stage0_any_passed": True,
                "stage1_any_passed": False,
                "canonical_fingerprint": "fp-neg-b",
            },
        ]

    class DummyEnsemble:
        gbm = None
        graph_pred = None
        bayesian = None
        interaction = None
        _score_mean = np.zeros(2, dtype=np.float64)
        _score_std = np.ones(2, dtype=np.float64)
        w_ensemble = np.zeros(2, dtype=np.float64)
        b_ensemble = 0.0

    monkeypatch.setattr(
        report_predictor_metrics,
        "load_deduped_screening_predictor_rows",
        fake_rows,
    )
    monkeypatch.setattr(report_predictor_metrics, "load_op_stats", lambda _db: {})

    y_true, y_score, split_stats = report_predictor_metrics._component_scores(
        DummyEnsemble(),
        "research/lab_notebook.db",
        sample_limit=100,
    )

    assert seen["db_path"] == "research/lab_notebook.db"
    assert len(y_true) == len(y_score) > 0
    assert set(np.unique(y_true)) == {0, 1}
    assert "n_unique_graphs" in split_stats


def test_ml_influence_status_reflects_conservative_defaults(tmp_path, monkeypatch):
    report_path = tmp_path / "predictor_metrics_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ensemble_calibrated": {
                    "val_metrics_selected_threshold": {"roc_auc": 0.87}
                },
                "gbm_gate": {"val_metrics_selected_threshold": {"roc_auc": 0.82}},
                "graph_predictor": {
                    "derived_val_classification_metrics": {"roc_auc": 0.80}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy._PREDICTOR_REPORT_PATH",
        report_path,
    )

    status = build_ml_influence_status()

    assert status["defaults"]["gbm_prescreener_enabled"] is True
    assert status["defaults"]["use_learned_candidate_weights"] is False
    assert status["defaults"]["use_screening_signal_weights"] is False
    assert status["defaults"]["use_learned_grammar_weights"] is False
    assert status["components"]["screening_ensemble"]["quality_tier"] == "strong"


def test_ml_influence_policy_blocks_unproven_generation_weights():
    policy = build_ml_influence_policy()

    assert policy["components"]["learned_candidate_weights"]["allowed"] is False
    assert policy["components"]["screening_signal_weights"]["allowed"] is False
    assert policy["components"]["learned_grammar_weights"]["allowed"] is False
