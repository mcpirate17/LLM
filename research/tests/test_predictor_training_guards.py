from __future__ import annotations

import sys
import numpy as np
import pytest
from types import SimpleNamespace

from research.scientist.intelligence import gnn_predictor as gp
from research.scientist.intelligence.gnn_predictor import GraphPredictor
from research.scientist.intelligence.predictor import (
    EnsemblePredictor,
    _calibrate_ensemble,
    evaluate_gbm_induction,
)
from research.scientist.runner import RunConfig
from research.scientist.runner.execution_experiment_phase3 import (
    _ExecutionExperimentPhase3Mixin,
    _resolve_p_pass_floor,
)


def test_graph_predictor_train_skips_single_class_corpus(monkeypatch, tmp_path):
    rows = [
        {
            "graph_json": '{"nodes":{"0":{"op_name":"linear_proj"}}}',
            "stage1_any_passed": 1,
            "stage0_any_passed": 1,
            "stage05_any_passed": 1,
            "wikitext_perplexity_best": 10.0,
            "loss_ratio_best": 0.5,
            "induction_screening_auc_500": 0.01,
            "n_rows": 1,
            "canonical_fingerprint": f"fp{i}",
        }
        for i in range(32)
    ]

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._load_op_profiles",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._load_pair_stability",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor.load_screening_predictor_corpus_rows",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor.extract_topology_features",
        lambda *_args, **_kwargs: {"feat_a": 1.0, "feat_b": 0.0},
    )

    model = GraphPredictor.train(
        notebook_db=tmp_path / "lab_notebook.db",
        profiling_db=tmp_path / "component_profiles.db",
    )

    assert not model.is_fitted()
    assert model.w_gate.size == 0


def test_gbm_prescreener_ignores_nonpositive_temporal_threshold():
    config = RunConfig(gbm_prescreener_enabled=True)
    ensemble = SimpleNamespace(gate_threshold=0.6673569091703704)
    report = {
        "ensemble_calibrated": {
            "temporal_holdout_evaluation": {
                "operating_points": {"f1": {"threshold": 0.0}}
            }
        }
    }

    floor, source = _resolve_p_pass_floor(ensemble, config, report)

    assert floor == pytest.approx(0.6673569091703704)
    assert source == "ensemble.gate_threshold"


def test_augment_imodel_features_prefers_pair_stats():
    class _FakeIModel:
        _trained = True

        def predict_pair_stats(self, edge_pairs):
            assert edge_pairs == [("linear_proj", "gelu"), ("add", "rmsnorm")]
            return (0.25, 0.5, 0.75)

        def predict_stability(self, left, right):
            raise AssertionError("scalar pair scoring should not run")

        def predict_loss(self, left, right):
            raise AssertionError("scalar pair scoring should not run")

    features = gp._augment_imodel_features_from_pairs(
        {"topo_n_ops": 2.0},
        [("linear_proj", "gelu"), ("add", "rmsnorm")],
        _FakeIModel(),
    )

    assert features == {
        "topo_n_ops": 2.0,
        "imodel_min_stability": 0.25,
        "imodel_mean_stability": 0.5,
        "imodel_mean_loss": 0.75,
    }


def test_calibrate_ensemble_skips_single_class_rows(monkeypatch):
    rows = [
        {
            "graph_json": '{"nodes":{"0":{"op_name":"linear_proj"}}}',
            "stage0_any_passed": 1,
            "stage1_any_passed": 1,
            "n_rows": 1,
            "canonical_fingerprint": f"fp{i}",
        }
        for i in range(120)
    ]

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor_ensemble._load_screening_predictor_corpus_rows",
        lambda *_args, **_kwargs: rows,
    )

    ensemble = EnsemblePredictor(
        gbm=None,
        graph_pred=None,
        bayesian=None,
        interaction=None,
    )
    _calibrate_ensemble(ensemble, "unused.sqlite")

    assert ensemble.w_ensemble.size == 0
    assert ensemble._calibration_metrics["error"] == "insufficient_class_balance"
    assert ensemble.gate_threshold == 0.5
    np.testing.assert_allclose(ensemble._score_mean, np.zeros(0, dtype=np.float32))


def test_planning_score_promotes_quality_when_pass_is_equal():
    ensemble = EnsemblePredictor()

    ensemble.predict_gate = lambda **_kwargs: 0.4
    ensemble.predict_induction_screening_auc = lambda **_kwargs: 0.01
    ensemble.predict_induction_learner_prob = lambda **_kwargs: 0.5
    ensemble.gbm = SimpleNamespace(
        is_fitted=lambda: True,
        predict_quality_score=lambda features: (
            0.8 if features == {"feat": 1.0} else 0.2
        ),
    )
    ensemble.graph_pred = SimpleNamespace(
        is_fitted=lambda: True,
        predict_rank=lambda graph_json: 6.0 if graph_json == {"id": "good"} else 50.0,
        predict_loss=lambda graph_json: 0.15 if graph_json == {"id": "good"} else 0.65,
    )

    good = ensemble.predict_planning_score(
        graph_json={"id": "good"},
        graph_features={"feat": 1.0},
    )
    bad = ensemble.predict_planning_score(
        graph_json={"id": "bad"},
        graph_features={"feat": 0.0},
    )

    assert good["p_pass"] == bad["p_pass"] == 0.4
    assert good["predicted_quality_score"] > bad["predicted_quality_score"]
    assert good["planning_score"] > bad["planning_score"]


def test_graph_predictor_rank_is_clipped_to_trained_log_bounds():
    model = GraphPredictor(
        w_gate=np.zeros(1, dtype=np.float32),
        b_gate=0.0,
        w_rank=np.array([-10.0], dtype=np.float32),
        b_rank=0.0,
        rank_log_min=float(np.log(6.0)),
        rank_log_max=float(np.log(60.0)),
        feature_names=["feat_a"],
        feature_mean=np.zeros(1, dtype=np.float32),
        feature_std=np.ones(1, dtype=np.float32),
        op_profiles={},
        pair_stability={},
        _trained=True,
    )
    original = GraphPredictor._extract_and_normalize
    try:
        GraphPredictor._extract_and_normalize = lambda self, _graph: np.array(
            [10.0], dtype=np.float64
        )
        pred = model.predict_rank({"graph": "ood"})
    finally:
        GraphPredictor._extract_and_normalize = original
    assert pred == np.float64(6.0)


class _DummyPhase3Runner(_ExecutionExperimentPhase3Mixin):
    pass


def test_gbm_prescreener_gates_on_p_pass_not_planning_score(monkeypatch):
    runner = _DummyPhase3Runner()
    config = RunConfig(gbm_prescreener_enabled=True, gbm_gate_threshold=0.1)
    results = {"funnel_counts": {}}
    recorded = []

    class _Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": {}}

    class _Ensemble:
        def is_fitted(self):
            return True

        def diagnostics(self):
            return {"n_components": 2}

        def predict_planning_score(self, *, graph_json=None, graph_features=None):
            if graph_json["graph_id"] == "keep_me":
                return {
                    "planning_score": 0.05,
                    "p_pass": 0.30,
                    "p_induction_learner": 0.0,
                    "predicted_induction_screening_auc": 0.0,
                }
            return {
                "planning_score": 0.60,
                "p_pass": 0.05,
                "p_induction_learner": 1.0,
                "predicted_induction_screening_auc": 0.03,
            }

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor.load_runtime_ensemble",
        lambda **_kwargs: _Ensemble(),
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.component_is_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features",
        lambda _graph_dict: {"feat": 1.0},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )

    notebook = SimpleNamespace(
        db_path="unused.sqlite",
        record_program_result=lambda **kwargs: recorded.append(kwargs),
    )
    kept = runner._run_gbm_prescreener(
        nb=notebook,
        graphs=[_Graph("keep_me"), _Graph("skip_me")],
        config=config,
        exp_id="exp-1",
        results=results,
    )

    assert [graph.graph_id for graph in kept] == ["keep_me"]
    assert results["funnel_counts"]["gbm_prescreener_skipped"] == 1
    assert recorded[0]["metrics"]["predicted_p_s1"] == 0.05


def test_gbm_prescreener_uses_temporal_f1_floor_by_default(monkeypatch):
    runner = _DummyPhase3Runner()
    config = RunConfig(gbm_prescreener_enabled=True)
    results = {"funnel_counts": {}}

    class _Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": {}}

    class _Ensemble:
        gate_threshold = 0.27

        def is_fitted(self):
            return True

        def diagnostics(self):
            return {"n_components": 2}

        def predict_planning_score(self, *, graph_json=None, graph_features=None):
            return {
                "planning_score": 0.9,
                "p_pass": 0.26 if graph_json["graph_id"] == "skip_me" else 0.28,
                "p_induction_learner": 0.0,
                "predicted_induction_screening_auc": 0.0,
            }

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor.load_runtime_ensemble",
        lambda **_kwargs: _Ensemble(),
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.component_is_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.load_predictor_metrics_report",
        lambda: {
            "ensemble_calibrated": {
                "temporal_holdout_evaluation": {
                    "operating_points": {"f1": {"threshold": 0.32}}
                }
            }
        },
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features",
        lambda _graph_dict: {"feat": 1.0},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )

    notebook = SimpleNamespace(
        db_path="unused.sqlite",
        record_program_result=lambda **kwargs: None,
    )
    kept = runner._run_gbm_prescreener(
        nb=notebook,
        graphs=[_Graph("keep_me"), _Graph("skip_me")],
        config=config,
        exp_id="exp-1",
        results=results,
    )

    assert [graph.graph_id for graph in kept] == []
    assert results["screening_ensemble_p_pass_floor"] == 0.32
    assert (
        results["screening_ensemble_p_pass_floor_source"]
        == "ensemble.temporal_holdout_evaluation.operating_points.f1.threshold"
    )


def test_gbm_prescreener_prefers_explicit_screening_floor_over_deprecated_alias(
    monkeypatch,
):
    runner = _DummyPhase3Runner()
    config = RunConfig(
        gbm_prescreener_enabled=True,
        screening_ensemble_p_pass_floor=0.4,
        gbm_gate_threshold=0.1,
    )
    results = {"funnel_counts": {}}

    class _Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": {}}

    class _Ensemble:
        gate_threshold = 0.27

        def is_fitted(self):
            return True

        def diagnostics(self):
            return {"n_components": 2}

        def predict_planning_score(self, *, graph_json=None, graph_features=None):
            return {
                "planning_score": 0.9,
                "p_pass": 0.35 if graph_json["graph_id"] == "skip_me" else 0.45,
                "p_induction_learner": 0.0,
                "predicted_induction_screening_auc": 0.0,
            }

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor.load_runtime_ensemble",
        lambda **_kwargs: _Ensemble(),
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.component_is_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features",
        lambda _graph_dict: {"feat": 1.0},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )

    notebook = SimpleNamespace(
        db_path="unused.sqlite",
        record_program_result=lambda **kwargs: None,
    )
    kept = runner._run_gbm_prescreener(
        nb=notebook,
        graphs=[_Graph("keep_me"), _Graph("skip_me")],
        config=config,
        exp_id="exp-1",
        results=results,
    )

    assert [graph.graph_id for graph in kept] == ["keep_me"]
    assert results["screening_ensemble_p_pass_floor"] == 0.4
    assert (
        results["screening_ensemble_p_pass_floor_source"]
        == "config.screening_ensemble_p_pass_floor"
    )


@pytest.mark.parametrize(
    "case_id, composite_table, expected_order, expected_used",
    [
        # Production composite head says C > A > B even though planning says A > B > C.
        # The rank head (Spearman 0.82) wins.
        ("usable", {"A": 50.0, "B": 10.0, "C": 90.0}, ["C", "A", "B"], True),
        # All three return the unusable sentinel: ordering reverts to planning_score.
        ("unusable", {"A": 1e6, "B": 1e6, "C": 1e6}, ["A", "B", "C"], False),
    ],
)
def test_gbm_prescreener_reorders_by_rank_composite(
    monkeypatch, case_id, composite_table, expected_order, expected_used
):
    runner = _DummyPhase3Runner()
    config = RunConfig(gbm_prescreener_enabled=True, gbm_gate_threshold=0.1)
    results = {"funnel_counts": {}}
    planning_table = {"A": 0.9, "B": 0.8, "C": 0.7}

    class _Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": {}}

    class _GBM:
        def is_fitted(self):
            return True

        def predict_rank_composite(self, features):
            return composite_table[features["graph_id"]]

    class _Ensemble:
        gbm = _GBM()

        def is_fitted(self):
            return True

        def diagnostics(self):
            return {"n_components": 2}

        def predict_planning_score(self, *, graph_json=None, graph_features=None):
            return {
                "planning_score": planning_table[graph_json["graph_id"]],
                "p_pass": 0.5,
                "p_induction_learner": 0.0,
                "predicted_induction_screening_auc": 0.0,
            }

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor.load_runtime_ensemble",
        lambda **_kwargs: _Ensemble(),
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.component_is_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features_bundle",
        lambda graph_dict: ({"graph_id": graph_dict["graph_id"]}, []),
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )

    notebook = SimpleNamespace(
        db_path="unused.sqlite", record_program_result=lambda **kwargs: None
    )
    kept = runner._run_gbm_prescreener(
        nb=notebook,
        graphs=[_Graph("A"), _Graph("B"), _Graph("C")],
        config=config,
        exp_id=f"exp-rank-{case_id}",
        results=results,
    )

    assert [g.graph_id for g in kept] == expected_order
    assert results["screening_rank_composite_used"] is expected_used
    assert results["funnel_counts"]["gbm_prescreener_skipped"] == 0


def test_gbm_prescreener_applies_ar_binding_reranker_after_rank_composite(
    monkeypatch,
):
    runner = _DummyPhase3Runner()
    config = RunConfig(
        gbm_prescreener_enabled=True,
        gbm_gate_threshold=0.1,
        ar_binding_overlay_enabled=True,
    )
    results = {"funnel_counts": {}}
    planning_table = {"A": 0.9, "B": 0.8, "C": 0.7}
    composite_table = {"A": 50.0, "B": 10.0, "C": 90.0}
    reranker_input = []

    class _Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id
            self.metadata = {}

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": {}}

    class _GBM:
        def is_fitted(self):
            return True

        def predict_rank_composite(self, features):
            return composite_table[features["graph_id"]]

    class _Ensemble:
        gbm = _GBM()

        def is_fitted(self):
            return True

        def diagnostics(self):
            return {"n_components": 2}

        def predict_planning_score(self, *, graph_json=None, graph_features=None):
            return {
                "planning_score": planning_table[graph_json["graph_id"]],
                "p_pass": 0.5,
                "p_induction_learner": 0.0,
                "predicted_induction_screening_auc": 0.0,
            }

    def fake_ar_binding_rerank(graphs):
        reranker_input.extend(g.graph_id for g in graphs)
        return [graphs[2], graphs[1], graphs[0]], {
            "used": True,
            "scored": 3,
            "holdout_required": 0,
            "score_min": -0.1,
            "score_max": 0.6,
        }

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor.load_runtime_ensemble",
        lambda **_kwargs: _Ensemble(),
    )
    monkeypatch.setattr(
        "research.scientist.ml_influence_policy.component_is_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features_bundle",
        lambda graph_dict: ({"graph_id": graph_dict["graph_id"]}, []),
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.ar_binding_reranker.rerank_graphs_by_ar_binding",
        fake_ar_binding_rerank,
    )

    notebook = SimpleNamespace(
        db_path="unused.sqlite", record_program_result=lambda **kwargs: None
    )
    kept = runner._run_gbm_prescreener(
        nb=notebook,
        graphs=[_Graph("A"), _Graph("B"), _Graph("C")],
        config=config,
        exp_id="exp-ar-overlay-rank",
        results=results,
    )

    assert reranker_input == ["C", "A", "B"]
    assert [g.graph_id for g in kept] == ["B", "A", "C"]
    assert results["screening_rank_composite_used"] is True
    assert results["screening_ar_binding_overlay_used"] is True
    assert results["screening_ar_binding_overlay_scored"] == 3
    assert results["screening_ar_binding_overlay_score_max"] == 0.6


def test_evaluate_gbm_induction_uses_single_corpus_source(monkeypatch):
    rows = []
    for i in range(60):
        rows.append(
            {
                "canonical_fingerprint": f"fp-{i}",
                "graph_json": {"nodes": {"0": {"op_name": "linear_proj"}}},
                "induction_screening_auc_500": 0.03 if i % 2 == 0 else 0.0,
                "n_rows": 1,
            }
        )

    class _DummyDataset:
        def __init__(
            self, data, label=None, weight=None, feature_name=None, reference=None
        ):
            self.data = data
            self.label = label

    class _DummyModel:
        def __init__(self, objective):
            self.objective = objective

        def predict(self, X):
            n = len(X)
            if self.objective == "binary":
                return np.linspace(0.1, 0.9, num=n, dtype=np.float64)
            return np.linspace(0.0, 0.04, num=n, dtype=np.float64)

        def feature_importance(self, _kind):
            return np.array([1.0, 0.5], dtype=np.float64)

    class _DummyLGB:
        @staticmethod
        def Dataset(data, label=None, weight=None, feature_name=None, reference=None):
            return _DummyDataset(data, label, weight, feature_name, reference)

        @staticmethod
        def train(
            params, train_set, num_boost_round=None, valid_sets=None, callbacks=None
        ):
            return _DummyModel(params["objective"])

        @staticmethod
        def early_stopping(*_args, **_kwargs):
            return object()

        @staticmethod
        def log_evaluation(*_args, **_kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "lightgbm", _DummyLGB)
    monkeypatch.setattr(
        "research.scientist.intelligence.predictor_gbm._load_screening_predictor_corpus_rows",
        lambda _db_path, validate=True: rows,
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.load_op_stats",
        lambda _db_path: {},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.extract_graph_features",
        lambda _graph_dict: {"feat_a": 1.0},
    )
    monkeypatch.setattr(
        "research.synthesis.graph_features.enrich_with_op_stats",
        lambda *_args, **_kwargs: None,
    )

    metrics = evaluate_gbm_induction("unused.sqlite")

    assert metrics.get("error") != "feature_row_mismatch"
    assert metrics["n_total"] == 60
    assert "learner_auc" in metrics
