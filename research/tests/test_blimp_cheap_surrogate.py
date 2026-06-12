from __future__ import annotations

import numpy as np

from research.tools import blimp_cheap_surrogate as surrogate


class _Predictor:
    def __init__(self, values: list[float]) -> None:
        self.values = np.array(values, dtype=float)

    def predict(self, _x):
        return self.values


def test_score_outputs_advisory_priority_not_skip(monkeypatch):
    def fake_rows(*_args, **_kwargs):
        return np.zeros((3, 1), dtype=float), ["low", "high", "mid"]

    monkeypatch.setattr(surrogate, "_load_score_rows", fake_rows)
    model = {
        "features": ["f"],
        "target": "blimp_overall_accuracy",
        "median": _Predictor([0.10, 0.90, 0.55]),
        "upper": _Predictor([0.20, 0.95, 0.60]),
        "train_pred_sorted": np.array([0.0, 0.5, 0.8], dtype=float),
    }

    rows = surrogate.score_and_gate(
        "unused.db",
        model,
        keep_frac=0.30,
        keep_threshold=None,
        only_missing=True,
        require_stage1=False,
        require_screened=True,
        limit=10,
    )

    assert [row["result_id"] for row in rows] == ["high", "mid", "low"]
    assert [row["priority"] for row in rows] == ["high", "medium", "low"]
    assert [row["recommended_for_eval"] for row in rows] == [True, False, False]
    assert all(row["hard_gate"] is False for row in rows)
    assert all(row["advisory_only"] is True for row in rows)
    assert all("gate" not in row for row in rows)
    assert all(row["priority"] != "skip" for row in rows)


def test_budget_marks_top_n_high_without_discarding_tail(monkeypatch):
    def fake_rows(*_args, **_kwargs):
        return np.zeros((3, 1), dtype=float), ["a", "b", "c"]

    monkeypatch.setattr(surrogate, "_load_score_rows", fake_rows)
    model = {
        "features": ["f"],
        "target": "blimp_overall_accuracy",
        "median": _Predictor([0.30, 0.20, 0.10]),
        "upper": _Predictor([0.31, 0.21, 0.11]),
        "train_pred_sorted": np.array([0.0, 0.5, 1.0], dtype=float),
    }

    rows = surrogate.score_and_gate(
        "unused.db",
        model,
        keep_frac=0.0,
        keep_threshold=None,
        only_missing=True,
        require_stage1=False,
        require_screened=True,
        limit=10,
        budget=1,
    )

    assert [row["priority"] for row in rows] == ["high", "low", "low"]
    assert [row["recommended_for_eval"] for row in rows] == [True, False, False]
    assert all("skip" not in row.values() for row in rows)
