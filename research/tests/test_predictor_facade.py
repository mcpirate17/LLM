from __future__ import annotations

import research.scientist.intelligence.predictor as predictor_facade
from research.scientist.intelligence.predictor_ensemble import (
    EnsemblePredictor,
    load_runtime_ensemble,
    train_ensemble,
)
from research.scientist.intelligence.predictor_gbm import (
    GBMPredictor,
    train_gbm,
)
from research.scientist.intelligence.predictor_ridge import (
    PerformancePredictor,
    train,
)


def test_predictor_facade_reexports_primary_predictor_symbols():
    assert predictor_facade.PerformancePredictor is PerformancePredictor
    assert predictor_facade.GBMPredictor is GBMPredictor
    assert predictor_facade.EnsemblePredictor is EnsemblePredictor
    assert predictor_facade.train is train
    assert predictor_facade.train_gbm is train_gbm
    assert predictor_facade.train_ensemble is train_ensemble
    assert predictor_facade.load_runtime_ensemble is load_runtime_ensemble


def test_predictor_facade_declares_stable_export_surface():
    exported = set(predictor_facade.__all__)

    assert "PerformancePredictor" in exported
    assert "GBMPredictor" in exported
    assert "EnsemblePredictor" in exported
    assert "train" in exported
    assert "train_gbm" in exported
    assert "train_ensemble" in exported
    assert "load_runtime_ensemble" in exported
