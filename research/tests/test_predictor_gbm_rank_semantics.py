from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import pytest

from research.scientist.intelligence.predictor_gbm import GBMPredictor


pytestmark = pytest.mark.unit


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, _x):
        return [self.value]


def test_gbm_predict_quality_score_uses_explicit_head_semantics():
    model = GBMPredictor(
        feature_names=["feat_a"],
        rank_model_ppl=_ConstantModel(math.log(5.0)),
        rank_model_composite=_ConstantModel(80.0),
        rank_ppl_log_min=math.log(1.0),
        rank_ppl_log_max=math.log(100.0),
        train_metrics={
            "rank_heads": {
                "ppl": {"spearman": 0.4, "ndcg_at_top_decile": 0.6},
                "composite": {"spearman": 0.8, "ndcg_at_top_decile": 0.9},
            }
        },
    )

    assert model.predict_rank({"feat_a": 1.0}) == pytest.approx(5.0)
    assert model.predict_rank_ppl({"feat_a": 1.0}) == pytest.approx(5.0)
    assert model.predict_rank_composite({"feat_a": 1.0}) == pytest.approx(80.0)

    # ppl head maps through exp(-ppl/25), composite maps through score/100.
    expected = (float(math.exp(-5.0 / 25.0)) + 0.8) / 2.0
    assert model.predict_quality_score({"feat_a": 1.0}) == pytest.approx(expected)


def test_gbm_load_ignores_legacy_mixed_rank_artifact(tmp_path, monkeypatch):
    class _FakeBooster:
        def __init__(self, model_file):
            self.model_file = model_file

        def predict(self, _x):
            return [0.5]

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(Booster=_FakeBooster))

    (tmp_path / "gbm_gate_model.txt").write_text("gate", encoding="utf-8")
    (tmp_path / "gbm_rank_model.txt").write_text("legacy", encoding="utf-8")
    (tmp_path / "gbm_predictor.json").write_text(
        json.dumps(
            {
                "feature_names": ["feat_a"],
                "gate_feature_names": ["feat_a"],
                "n_train": 10,
                "gate_threshold": 0.4,
                "has_rank_model": True,
                "train_metrics": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = GBMPredictor.load(tmp_path)

    assert loaded.is_fitted()
    assert loaded.rank_model_ppl is None
    assert loaded.rank_model_composite is None
    assert loaded.legacy_mixed_rank_model_loaded is True
    assert loaded.predict_rank({"feat_a": 1.0}) > 1e5
