"""Cheap mocked tests for trust-related cohort output shapes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from component_fab.harness.lm_eval import LMEvalResult, WikitextTrainTrace
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.tests.conftest import make_spec


def _spec() -> ProposalSpec:
    return make_spec(
        {"op_algebraic_space": "tropical"},
        "candidate_abc",
        name="candidate",
        category="lane",
        predicted_lift=0.0,
    )


def test_tier2_cohort_aggregates_seed_results(monkeypatch) -> None:
    from research.tools import run_tier2_binding_cohort as tier2

    monkeypatch.setattr(
        tier2, "_load_proposals_by_id", lambda: {"candidate_abc": _spec()}
    )
    monkeypatch.setattr(tier2, "default_hard_binding_tasks", lambda seed: ("task",))

    def fake_suite(*args, seed: int, **kwargs):  # noqa: ARG001
        delta = 0.1 + 0.1 * seed

        def row(label: str, acc: float):
            return SimpleNamespace(
                eval_accuracy=acc,
                mixer_label=label,
                chance_accuracy=0.125,
                n_params=10,
            )

        return {
            "long_gap_recall": [row("candidate", 0.5 + delta), row("softmax", 0.5)],
            "compositional_binding": [
                row("candidate", 0.5 + delta),
                row("softmax", 0.5),
            ],
            "multi_query_kv_recall": [
                row("candidate", 0.5 + delta),
                row("softmax", 0.5),
            ],
        }

    monkeypatch.setattr(tier2, "run_harder_binding_suite", fake_suite)
    summary = tier2.run_cohort(
        ["candidate_abc"],
        n_train_steps=1,
        seed_count=2,
        accumulate_labels=False,  # never pollute the real predictor training table
        quiet=True,
    )
    row = summary["results"]["candidate_abc"]
    assert row["seed_count"] == 2
    assert row["tier2_passed"] is True
    assert row["per_task"]["long_gap_recall"]["seed_deltas"] == pytest.approx(
        [0.1, 0.2]
    )


def test_blimp_cohort_aggregates_seed_results(monkeypatch) -> None:
    from research.tools import run_blimp_cohort as blimp

    monkeypatch.setattr(
        blimp, "_load_proposals_by_id", lambda: {"candidate_abc": _spec()}
    )

    def result(label: str, acc: float, ppl: float) -> LMEvalResult:
        return LMEvalResult(
            mixer_label=label,
            n_params=10,
            wikitext=WikitextTrainTrace(
                initial_loss=4.0,
                final_loss=3.0,
                pre_train_ppl=100.0,
                post_train_ppl=ppl,
                n_steps=1,
                converged=True,
            ),
            blimp_overall_accuracy=acc,
            blimp_by_subtask={"foo": acc},
            blimp_status="ok",
        )

    def fake_baseline(name, *, seed: int, **kwargs):  # noqa: ANN001, ARG001
        return result(name, 0.50 + 0.01 * seed, 30.0)

    def fake_candidate(spec, *, seed: int, **kwargs):  # noqa: ANN001, ARG001
        return result(spec.name, 0.56 + 0.01 * seed, 31.0)

    monkeypatch.setattr(blimp, "_baseline_result", fake_baseline)
    monkeypatch.setattr(blimp, "_evaluate_one", fake_candidate)
    summary = blimp.run_cohort(
        ["candidate_abc"],
        baseline_names=("softmax_attention",),
        n_train_steps=1,
        seed_count=2,
        quiet=True,
    )
    row = summary["results"]["candidate_abc"]
    assert summary["seed_count"] == 2
    assert row["seed_count"] == 2
    assert row["delta_vs_softmax_blimp"] == 0.06
    assert row["wikitext_post_ppl"] == 31.0
