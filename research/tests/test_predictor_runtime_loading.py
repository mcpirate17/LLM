"""Regression tests for persisted predictor runtime loading."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.scientist.intelligence.gnn_predictor import GraphPredictor
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
