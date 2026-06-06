"""Tier-2 value-predictor consumer: dormant fallback + deployed activation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import component_fab.proposer.tier2_features as TF
import component_fab.state.tier2_predictor as TP
from component_fab.improver.cross_anchor import enumerate_frontier_core_specs
from component_fab.proposer.quality import _tier2_win_probability

_SPEC = enumerate_frontier_core_specs()[0]
_AFF = SimpleNamespace(affinity=0.0, confidence=0.0)


def test_predictor_dormant_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(TP, "MODEL_PATH", tmp_path / "absent.joblib")
    TP._CACHE.clear()
    assert TP.predictor_available() is False
    assert TP.predict_mean_delta(_SPEC) is None


def test_predictor_predicts_when_deployed(monkeypatch):
    class _Stub:
        def predict(self, X):
            assert X.shape == (1, 13)
            return np.array([0.123])

    monkeypatch.setattr(TP, "_load_model", lambda: _Stub())
    monkeypatch.setattr(TP, "_extractor", lambda: object())
    monkeypatch.setattr(TF, "features_for_spec", lambda spec, extractor: [0.0] * 13)
    assert abs(TP.predict_mean_delta(_SPEC) - 0.123) < 1e-6

    # unmeasurable candidate -> None, never 0.0
    monkeypatch.setattr(TF, "features_for_spec", lambda spec, extractor: None)
    assert TP.predict_mean_delta(_SPEC) is None


def test_winprob_uses_predicted_delta_over_heuristic():
    base, has, _ = _tier2_win_probability(
        None, binding=0.2, composite=0.2, affinity=_AFF
    )
    assert has is False
    hi, _, reasons = _tier2_win_probability(
        None, binding=0.2, composite=0.2, affinity=_AFF, predicted_delta=0.3
    )
    lo, _, _ = _tier2_win_probability(
        None, binding=0.2, composite=0.2, affinity=_AFF, predicted_delta=-0.3
    )
    # a positive predicted delta raises the estimate above the heuristic, and
    # above what a negative prediction yields
    assert hi > base
    assert hi > lo
    assert any("value predictor" in r for r in reasons)


def test_winprob_with_tier2_evidence_ignores_predictor():
    # Real Tier-2 evidence is authoritative; predicted_delta must not override it.
    from component_fab.proposer.tier2_feedback import Tier2Feedback

    t2 = Tier2Feedback(
        proposal_id="p",
        name="p",
        pass_count=5,
        n_tasks=6,
        tier2_passed=True,
        tier2_passed_niche=False,
        mean_delta=0.05,
        wins=(),
        failures=(),
        signatures=(),
        task_results=(),
    )
    prob, has, _ = _tier2_win_probability(
        t2, binding=0.0, composite=0.0, affinity=_AFF, predicted_delta=-0.9
    )
    assert has is True
    assert prob >= 0.70  # driven by measured Tier-2 pass, not the predictor
