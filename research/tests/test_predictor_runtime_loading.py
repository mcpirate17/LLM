"""Regression tests for persisted predictor runtime loading."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.scientist.intelligence.gnn_predictor import GraphPredictor
from research.scientist.intelligence.interaction_model import InteractionModel
from research.scientist.intelligence.op_embeddings import OpEmbeddings
from research.scientist.intelligence.predictor import load_runtime_ensemble


@pytest.mark.unit
def test_graph_predictor_roundtrip_from_artifact(tmp_path):
    """Saved topology predictor artifacts reload without retraining."""
    artifact = tmp_path / "graph_predictor.npz"
    model = GraphPredictor(
        w_gate=np.array([0.1, -0.2], dtype=np.float64),
        b_gate=0.3,
        w_rank=np.array([0.4, 0.5], dtype=np.float64),
        b_rank=1.7,
        w_loss=np.array([-0.8, 0.9], dtype=np.float64),
        b_loss=0.6,
        feature_names=["feat_a", "feat_b"],
        feature_mean=np.array([1.0, 2.0], dtype=np.float64),
        feature_std=np.array([0.5, 1.5], dtype=np.float64),
        n_train=42,
        _trained=True,
        _train_metrics={"val_acc": 0.91, "val_loss": 0.12},
    )

    model.save(artifact)
    loaded = GraphPredictor.load(artifact, profiling_db=tmp_path / "missing.db")

    assert loaded.is_fitted()
    assert loaded.feature_names == ["feat_a", "feat_b"]
    assert loaded.n_train == 42
    assert loaded._train_metrics == {"val_acc": 0.91, "val_loss": 0.12}
    assert loaded.op_profiles == {}
    assert loaded.pair_stability == {}
    np.testing.assert_allclose(loaded.w_gate, model.w_gate)
    np.testing.assert_allclose(loaded.w_rank, model.w_rank)
    np.testing.assert_allclose(loaded.w_loss, model.w_loss)
    np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
    np.testing.assert_allclose(loaded.feature_std, model.feature_std)


@pytest.mark.unit
def test_interaction_model_roundtrip_from_artifact(tmp_path):
    artifact = tmp_path / "interaction_model.npz"
    model = InteractionModel(
        u=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        v=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        W_s=np.eye(2, dtype=np.float32),
        W_l=np.array([[1.0, -1.0], [0.25, 0.5]], dtype=np.float32),
        b_s=0.15,
        b_l=0.85,
        op_names=["add", "mul"],
        op_to_idx={"add": 0, "mul": 1},
        _trained=True,
        _timestamp=123.0,
        _train_metrics={"best_loss": 0.42},
    )

    model.save(artifact)
    loaded = InteractionModel.load(artifact)

    assert loaded._trained is True
    assert loaded.op_names == ["add", "mul"]
    assert loaded.op_to_idx == {"add": 0, "mul": 1}
    assert loaded._train_metrics == {"best_loss": 0.42}
    np.testing.assert_allclose(loaded.u, model.u)
    np.testing.assert_allclose(loaded.v, model.v)
    np.testing.assert_allclose(loaded.W_s, model.W_s)
    np.testing.assert_allclose(loaded.W_l, model.W_l)


@pytest.mark.unit
def test_interaction_model_predict_pair_stats_matches_scalar_predictions():
    model = InteractionModel(
        u=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        v=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        W_s=np.eye(2, dtype=np.float32),
        W_l=np.array([[1.0, -1.0], [0.25, 0.5]], dtype=np.float32),
        b_s=0.15,
        b_l=0.85,
        op_names=["add", "mul"],
        op_to_idx={"add": 0, "mul": 1},
        _trained=True,
    )
    edge_pairs = [("add", "mul"), ("mul", "add"), ("unknown", "mul")]

    stats = model.predict_pair_stats(edge_pairs)
    scalar_stabilities = [
        model.predict_stability(left, right) for left, right in edge_pairs
    ]
    scalar_losses = [model.predict_loss(left, right) for left, right in edge_pairs]

    assert stats is not None
    assert stats[0] == pytest.approx(min(scalar_stabilities))
    assert stats[1] == pytest.approx(float(np.mean(scalar_stabilities)))
    assert stats[2] == pytest.approx(float(np.mean(scalar_losses)))


@pytest.mark.unit
def test_op_embeddings_roundtrip_from_artifact(tmp_path):
    artifact = tmp_path / "op_embeddings.npz"
    model = OpEmbeddings(
        embeddings=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        op_names=["add", "relu"],
        op_to_idx={"add": 0, "relu": 1},
        _trained=True,
        _timestamp=456.0,
    )

    model.save(artifact)
    loaded = OpEmbeddings.load(artifact)

    assert loaded._trained is True
    assert loaded.op_names == ["add", "relu"]
    assert loaded.op_to_idx == {"add": 0, "relu": 1}
    np.testing.assert_allclose(loaded.embeddings, model.embeddings)


@pytest.mark.unit
def test_load_runtime_ensemble_uses_persisted_state_and_cache(tmp_path):
    """Runtime ensemble loading stays load-only and reuses cached artifacts."""
    state_dir = tmp_path / "runtime_state"
    state_dir.mkdir()
    np.savez_compressed(
        state_dir / "ensemble_state.npz",
        w_ensemble=np.array([0.25, -0.75], dtype=np.float64),
        score_mean=np.array([1.0, 2.0], dtype=np.float64),
        score_std=np.array([3.0, 4.0], dtype=np.float64),
    )
    (state_dir / "ensemble_state.json").write_text(
        json.dumps({"b_ensemble": 0.5, "n_score_dims": 2}),
        encoding="utf-8",
    )

    load_runtime_ensemble.cache_clear()
    loaded = load_runtime_ensemble(
        state_dir=str(state_dir),
        profiling_db=str(tmp_path / "missing.db"),
    )
    cached = load_runtime_ensemble(
        state_dir=str(state_dir),
        profiling_db=str(tmp_path / "missing.db"),
    )

    assert loaded is cached
    assert loaded.gbm is None
    assert loaded.graph_pred is None
    assert loaded.interaction is None
    assert loaded.bayesian is None
    assert loaded.b_ensemble == 0.5
    assert loaded._n_score_dims == 2
    np.testing.assert_allclose(loaded.w_ensemble, np.array([0.25, -0.75]))
    np.testing.assert_allclose(loaded._score_mean, np.array([1.0, 2.0]))
    np.testing.assert_allclose(loaded._score_std, np.array([3.0, 4.0]))
