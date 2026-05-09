from __future__ import annotations

import sys
import numpy as np
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


def test_extract_topology_features_prefers_single_native_bridge(monkeypatch):
    ctx = gp._NativeTopologyContext(
        op_profiles_json="{}",
        pair_stability_json="{}",
        op_metadata_json="{}",
    )
    calls = {"single": 0}

    class _FakeRust:
        def extract_topology_features_native(self, *args):
            calls["single"] += 1
            return '{"topo_n_ops": 3.0}'

        def extract_topology_features_batch_native(self, *args):
            raise AssertionError("single-graph path should not use batch bridge")

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._try_import_rust_scheduler",
        lambda: _FakeRust(),
    )

    features = gp.extract_topology_features(
        {"nodes": {"0": {"op_name": "input"}, "1": {"op_name": "linear_proj"}}},
        {},
        {},
        native_ctx=ctx,
    )

    assert calls["single"] == 1
    assert features == {
        "topo_n_ops": 3.0,
        "imodel_min_stability": 0.5,
        "imodel_mean_stability": 0.5,
        "imodel_mean_loss": 0.7,
    }


def test_extract_topology_features_prefers_single_direct_feature_map_bridge(
    monkeypatch,
):
    ctx = gp._NativeTopologyContext(
        op_profiles_json="{}",
        pair_stability_json="{}",
        op_metadata_json="{}",
    )
    calls = {"single": 0}

    class _FakeRust:
        def extract_topology_feature_map_native_py(self, *args):
            calls["single"] += 1
            return {"topo_n_ops": 3.0}

        def extract_topology_features_native(self, *args):
            raise AssertionError("single-graph path should not use JSON bridge")

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._try_import_rust_scheduler",
        lambda: _FakeRust(),
    )

    features = gp.extract_topology_features(
        {"nodes": {"0": {"op_name": "input"}, "1": {"op_name": "linear_proj"}}},
        {},
        {},
        native_ctx=ctx,
    )

    assert calls["single"] == 1
    assert features == {
        "topo_n_ops": 3.0,
        "imodel_min_stability": 0.5,
        "imodel_mean_stability": 0.5,
        "imodel_mean_loss": 0.7,
    }


def test_extract_topology_features_prefers_fused_single_native_imodel_bridge(
    monkeypatch,
):
    ctx = gp._NativeTopologyContext(
        op_profiles_json="{}",
        pair_stability_json="{}",
        op_metadata_json="{}",
    )
    calls = {"fused": 0}

    class _FakeRust:
        def extract_topology_feature_map_with_imodel_native_py(self, *args):
            calls["fused"] += 1
            return {
                "topo_n_ops": 2.0,
                "imodel_min_stability": 0.2,
                "imodel_mean_stability": 0.4,
                "imodel_mean_loss": 0.6,
            }

        def extract_topology_features_native(self, *args):
            raise AssertionError("fused single-graph path should bypass base bridge")

    class _FakeIModel:
        _trained = True
        op_names = ["linear_proj", "gelu"]
        u = np.ones((2, 2), dtype=np.float32)
        v = np.ones((2, 2), dtype=np.float32)
        W_s = np.eye(2, dtype=np.float32)
        W_l = np.eye(2, dtype=np.float32)
        b_s = 0.0
        b_l = 0.5

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._try_import_rust_scheduler",
        lambda: _FakeRust(),
    )

    features = gp.extract_topology_features(
        {"nodes": {"0": {"op_name": "linear_proj"}, "1": {"op_name": "gelu"}}},
        {},
        {},
        imodel=_FakeIModel(),
        native_ctx=ctx,
    )

    assert calls["fused"] == 1
    assert features == {
        "topo_n_ops": 2.0,
        "imodel_min_stability": 0.2,
        "imodel_mean_stability": 0.4,
        "imodel_mean_loss": 0.6,
    }


def test_extract_topology_feature_batch_uses_edge_pair_batch_for_imodel(monkeypatch):
    class _FakeIModel:
        _trained = True

        def predict_stability(self, left, right):
            return 0.75 if left == "linear_proj" else 0.5

        def predict_loss(self, left, right):
            return 0.25 if right == "gelu" else 0.4

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._extract_topology_base_features_native",
        lambda *_args, **_kwargs: [{"topo_n_ops": 2.0}, {"topo_n_ops": 3.0}],
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._extract_edge_op_pairs_batch_native",
        lambda _payloads: [
            [("linear_proj", "gelu")],
            [("linear_proj", "add"), ("add", "rmsnorm")],
        ],
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._edge_op_pairs_with_fallback",
        lambda _payload: (_ for _ in ()).throw(
            AssertionError("per-graph edge pair fallback should not run")
        ),
    )

    features = gp._extract_topology_features_batch(
        ['{"nodes":{"0":{"op_name":"linear_proj"}}}'] * 2,
        {},
        {},
        imodel=_FakeIModel(),
    )

    assert features[0]["imodel_min_stability"] == 0.75
    assert features[0]["imodel_mean_loss"] == 0.25
    assert features[1]["imodel_mean_stability"] == 0.625
    assert features[1]["imodel_mean_loss"] == 0.4


def test_extract_topology_feature_batch_prefers_fused_native_imodel_bridge(monkeypatch):
    class _FakeRust:
        def extract_topology_feature_maps_with_imodel_batch_native_py(self, *args):
            return [
                {
                    "topo_n_ops": 2.0,
                    "imodel_min_stability": 0.2,
                    "imodel_mean_stability": 0.4,
                    "imodel_mean_loss": 0.6,
                }
            ]

    class _FakeIModel:
        _trained = True
        op_names = ["linear_proj", "gelu"]
        u = np.ones((2, 2), dtype=np.float32)
        v = np.ones((2, 2), dtype=np.float32)
        W_s = np.eye(2, dtype=np.float32)
        W_l = np.eye(2, dtype=np.float32)
        b_s = 0.0
        b_l = 0.5

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._try_import_rust_scheduler",
        lambda: _FakeRust(),
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._extract_topology_base_features_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("base feature native path should not run")
        ),
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._extract_edge_op_pairs_batch_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("edge-pair batch path should not run")
        ),
    )

    features = gp._extract_topology_features_batch(
        ['{"nodes":{"0":{"op_name":"linear_proj"}}}'],
        {},
        {},
        imodel=_FakeIModel(),
    )

    assert features == [
        {
            "topo_n_ops": 2.0,
            "imodel_min_stability": 0.2,
            "imodel_mean_stability": 0.4,
            "imodel_mean_loss": 0.6,
        }
    ]


def test_extract_topology_feature_batch_prefers_direct_feature_maps_bridge(monkeypatch):
    class _FakeRust:
        def extract_topology_feature_maps_batch_native_py(self, *args):
            return [{"topo_n_ops": 2.0, "pair_mean_stability": 0.7}]

    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._try_import_rust_scheduler",
        lambda: _FakeRust(),
    )
    monkeypatch.setattr(
        "research.scientist.intelligence.gnn_predictor._extract_edge_op_pairs_batch_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("edge-pair batch path should not run")
        ),
    )

    features = gp._extract_topology_features_batch(
        ['{"nodes":{"0":{"op_name":"linear_proj"}}}'],
        {},
        {},
    )

    assert features == [
        {
            "topo_n_ops": 2.0,
            "pair_mean_stability": 0.7,
            "imodel_min_stability": 0.5,
            "imodel_mean_stability": 0.5,
            "imodel_mean_loss": 0.7,
        }
    ]


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
