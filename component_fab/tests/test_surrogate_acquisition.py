"""Tests for WS-3: ledger surrogate + acquisition selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from component_fab.proposer.acquisition import score_specs, select_by_acquisition
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state._stats import spearman
from component_fab.state.surrogate import (
    MeanFieldApproximant,
    _marginal_scores,
    _recall_at_k,
    compute_surrogate_report,
    features_for_spec,
    write_surrogate_report,
)
from component_fab.tests.conftest import (
    grade_record,
    make_spec,
    promote_record,
    write_ledger_jsonl,
)


# --------------------------------------------------------------------------- #
# Pure metric helpers
# --------------------------------------------------------------------------- #
def test_spearman_monotone():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == 1.0
    assert spearman(a, a[::-1]) == -1.0


def test_spearman_degenerate_inputs_zero():
    # constant input and n < 2 are defined degenerate cases -> 0.0, no warning
    assert spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert spearman(np.array([1.0]), np.array([2.0])) == 0.0


def test_recall_at_k_counts_positives_in_topk():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    promoted = np.array([0, 1, 1, 0, 0])
    assert _recall_at_k(scores, promoted, 2) == 1.0  # both positives in top-2
    assert _recall_at_k(scores, promoted, 1) == 0.5  # one positive in top-1


def test_marginal_scores_is_additive_mean():
    # feature 0 active rows have mean y 1.0; feature 1 active rows mean 0.0
    X_train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y_train = np.array([1.0, 1.0, 0.0, 0.0])
    X_eval = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = _marginal_scores(X_train, y_train, X_eval)
    assert out[0] == 1.0  # only feature 0
    assert out[1] == 0.0  # only feature 1
    assert out[2] == 0.5  # mean of both feature means


# --------------------------------------------------------------------------- #
# Feature encoding
# --------------------------------------------------------------------------- #
def _spec(pid: str, axes: dict) -> ProposalSpec:
    return make_spec(axes, pid, anchor_witness_op="x", rationale="t")


def test_features_for_spec_encodes_axes_and_knobs():
    feat = features_for_spec(
        _spec("s", {"op_algebraic_space": "tropical", "op_math_knobs": "a+b"})
    )
    assert feat["op_algebraic_space=tropical"] == 1.0
    assert feat["synthesis_kind=novel_hybrid"] == 1.0
    assert feat["knob=a"] == 1.0 and feat["knob=b"] == 1.0


# --------------------------------------------------------------------------- #
# Acquisition ranking (stub surrogate — deterministic, no training)
# --------------------------------------------------------------------------- #
class _StubMeanFieldApproximant:
    def predict(self, features: dict) -> tuple[float, float]:
        base = 0.6 if features.get("op_algebraic_space=tropical") else 0.2
        return base, base + 0.3


def test_score_specs_ranks_by_ucb():
    specs = [
        _spec("low", {"op_algebraic_space": "euclidean"}),
        _spec("high", {"op_algebraic_space": "tropical"}),
    ]
    scored = score_specs(specs, _StubMeanFieldApproximant(), beta=1.0)
    assert scored[0].spec.proposal_id == "high"
    assert scored[0].ucb > scored[1].ucb


def test_select_by_acquisition_budget_and_fallbacks():
    specs = [
        _spec("a", {"op_algebraic_space": "euclidean"}),
        _spec("b", {"op_algebraic_space": "tropical"}),
        _spec("c", {"op_algebraic_space": "euclidean"}),
    ]
    sel = select_by_acquisition(specs, _StubMeanFieldApproximant(), budget=1)
    assert [s.proposal_id for s in sel] == ["b"]  # highest UCB
    # budget 0 -> no cap, identity order
    assert len(select_by_acquisition(specs, _StubMeanFieldApproximant(), budget=0)) == 3
    # no surrogate -> identity prefix
    assert [s.proposal_id for s in select_by_acquisition(specs, None, budget=2)] == [
        "a",
        "b",
    ]


# --------------------------------------------------------------------------- #
# Report + fitted surrogate on a synthetic ledger
# --------------------------------------------------------------------------- #
def _grade(pid: str, axes: dict, composite: float) -> dict:
    return grade_record(
        pid,
        composite=composite,
        learned=composite > 0.5,
        math_axes=axes,
        synthesis_kind="novel_hybrid",
    )


def _synthetic_ledger(path: Path, n: int = 24) -> None:
    records: list[dict] = []
    for i in range(n):
        good = i % 2 == 0
        axes = {"op_algebraic_space": "tropical" if good else "euclidean"}
        records.append(_grade(f"p{i}", axes, 0.7 if good else 0.2))
        if good and i < 8:  # a handful of promotions, all "good"
            records.append(promote_record(f"p{i}"))
    write_ledger_jsonl(path, records)


def test_report_runs_and_writes(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _synthetic_ledger(ledger)
    report = compute_surrogate_report(ledger_path=ledger, n_folds=2)
    assert report.n_rows == 24
    assert report.n_promoted == 4
    assert set(report.recall_at_k) == {25, 50, 100}
    out = write_surrogate_report(report, output_path=tmp_path / "report.json")
    data = json.loads(out.read_text())
    assert "acceptance_passed" in data and "findings" in data


def test_report_insufficient_rows(tmp_path: Path):
    ledger = tmp_path / "tiny.jsonl"
    ledger.write_text(
        json.dumps(_grade("p0", {"op_algebraic_space": "tropical"}, 0.5)) + "\n"
    )
    report = compute_surrogate_report(ledger_path=ledger, n_folds=5)
    assert not report.acceptance_passed
    assert report.n_rows == 1


def test_surrogate_fit_predict(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _synthetic_ledger(ledger)
    surrogate = MeanFieldApproximant.fit(ledger_path=ledger)
    assert surrogate is not None
    median, upper = surrogate.predict(
        features_for_spec(_spec("x", {"op_algebraic_space": "tropical"}))
    )
    assert upper >= median


def test_surrogate_fit_none_on_tiny_ledger(tmp_path: Path):
    ledger = tmp_path / "tiny.jsonl"
    ledger.write_text(
        json.dumps(_grade("p0", {"op_algebraic_space": "tropical"}, 0.5)) + "\n"
    )
    assert MeanFieldApproximant.fit(ledger_path=ledger) is None
