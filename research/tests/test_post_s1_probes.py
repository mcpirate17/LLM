"""Tests for the shared post-S1 probe helper.

The helper centralises probe orchestration so the synthesis runner and the
under-observed exploration tool produce the same metric shape. Required
because `_enforce_s1_metric_completeness` (notebook/program_writes.py)
rejects any stage1_passed=True write missing the canonical 6 metrics.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research.eval.post_s1_probes import (
    _REQUIRED_S1_METRICS,
    _compute_binding_composite,
    missing_required_metrics,
    run_post_s1_probes,
)


def test_required_metrics_match_guardrail() -> None:
    """The helper's canonical list must match the writer-side guardrail.

    If a new metric is added to ``_S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL``
    in notebook/program_writes.py without also being added here, callers
    will keep producing writes that the guardrail rejects.
    """
    from research.scientist.notebook.program_writes import (
        _S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL,
    )

    assert set(_REQUIRED_S1_METRICS) == set(
        _S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL
    )


def test_missing_required_metrics_empty_dict() -> None:
    assert set(missing_required_metrics({})) == set(_REQUIRED_S1_METRICS)


def test_missing_required_metrics_partial() -> None:
    """Returns only the keys that are still ``None``."""
    partial = {
        "wikitext_perplexity": 100.0,
        "hellaswag_acc": 0.25,
        "induction_screening_auc": 0.4,
    }
    absent = set(missing_required_metrics(partial))
    assert absent == {
        "binding_screening_auc",
        "binding_screening_composite",
        "ar_legacy_auc",
    }


def test_missing_required_metrics_full() -> None:
    full = {k: 0.5 for k in _REQUIRED_S1_METRICS}
    assert missing_required_metrics(full) == []


def test_missing_required_metrics_treats_none_as_missing() -> None:
    """A column explicitly set to None still counts as missing.

    This matches the guardrail's `kwargs.get(c) is None` check — a row
    with the column present-but-None is not a complete S1 write.
    """
    sparse: dict[str, object] = {key: None for key in _REQUIRED_S1_METRICS}
    assert set(missing_required_metrics(sparse)) == set(_REQUIRED_S1_METRICS)


def test_compute_binding_composite_with_ar_gate() -> None:
    """Composite = 0.4 * ar_gate + 0.3 * induction + 0.3 * binding."""
    got = _compute_binding_composite(
        induction_auc=0.5, binding_auc=0.6, ar_gate_score=0.7
    )
    assert got == pytest.approx(0.4 * 0.7 + 0.3 * 0.5 + 0.3 * 0.6, abs=1e-4)


def test_compute_binding_composite_without_ar_gate() -> None:
    """When AR gate isn't available, fall back to 0.3 * induction + 0.3 * binding."""
    got = _compute_binding_composite(
        induction_auc=0.4, binding_auc=0.5, ar_gate_score=None
    )
    assert got == pytest.approx(0.3 * 0.4 + 0.3 * 0.5, abs=1e-4)


def test_compute_binding_composite_returns_none_when_missing_inputs() -> None:
    assert (
        _compute_binding_composite(
            induction_auc=None, binding_auc=0.5, ar_gate_score=0.6
        )
        is None
    )
    assert (
        _compute_binding_composite(
            induction_auc=0.5, binding_auc=None, ar_gate_score=0.6
        )
        is None
    )


def test_run_post_s1_probes_swallows_probe_failures() -> None:
    """A broken probe should not abort the rest of the suite.

    The contract is that ``run_post_s1_probes`` always returns a dict;
    callers use ``missing_required_metrics`` to decide whether to demote
    stage1_passed.
    """
    model = MagicMock(spec=["__call__"])
    with (
        patch(
            "research.eval.wikitext_eval.screening_wikitext_eval",
            side_effect=RuntimeError("simulated wt failure"),
        ),
        patch(
            "research.eval.hellaswag_eval.screening_hellaswag_eval",
            side_effect=RuntimeError("simulated hs failure"),
        ),
        patch(
            "research.eval.native_induction.induction_score_gold",
            side_effect=RuntimeError("simulated induction failure"),
        ),
        patch(
            "research.eval.binding_range.binding_range_profile",
            side_effect=RuntimeError("simulated binding failure"),
        ),
        patch(
            "research.eval.associative_recall.associative_recall_score",
            side_effect=RuntimeError("simulated ar failure"),
        ),
        patch(
            "research.eval.ar_gate.ar_gate",
            side_effect=RuntimeError("simulated ar-gate failure"),
        ),
    ):
        metrics = run_post_s1_probes(model, vocab_size=100, device="cpu")
    # All probes failed — none of the required metrics should be present.
    absent = missing_required_metrics(metrics)
    assert set(absent) == set(_REQUIRED_S1_METRICS)
    # And the helper should still have returned a dict, not raised.
    assert isinstance(metrics, dict)


def test_run_post_s1_probes_records_composite_when_inputs_present() -> None:
    """When induction and binding probes succeed, composite is computed."""
    model = MagicMock(spec=["__call__"])

    class _IndStub:
        auc = 0.5

    class _BindStub:
        auc = 0.6
        distance_accuracies = []
        elapsed_ms = 10.0

    class _ARStub:
        auc = 0.55
        final_acc = 0.6
        timed_out = False
        above_chance = True

    class _NaiStub:
        metric_version = "v1"
        in_dist_pair_acc = 0.7
        in_dist_class_acc = 0.5
        held_pair_acc = 0.4
        held_class_acc = 0.45
        status = "ok"
        elapsed_ms = 20.0
        finetune_steps_done = 100

    with (
        patch(
            "research.eval.wikitext_eval.screening_wikitext_eval",
            return_value={"wikitext_perplexity": 50.0, "wikitext_score": 0.8},
        ),
        patch(
            "research.eval.hellaswag_eval.screening_hellaswag_eval",
            return_value={
                "hellaswag_acc": 0.25,
                "hellaswag_total": 100,
                "elapsed_ms": 5.0,
            },
        ),
        patch(
            "research.eval.native_induction.induction_score_gold",
            return_value=_IndStub(),
        ),
        patch(
            "research.eval.native_induction.induction_result_metadata",
            return_value={"induction_screening_auc": 0.5},
        ),
        patch(
            "research.eval.binding_range.binding_range_profile",
            return_value=_BindStub(),
        ),
        patch(
            "research.eval.associative_recall.associative_recall_score",
            return_value=_ARStub(),
        ),
        patch(
            "research.eval.ar_gate.ar_gate",
            return_value=_NaiStub(),
        ),
    ):
        metrics = run_post_s1_probes(
            model,
            vocab_size=100,
            device="cpu",
            run_binding_curriculum=False,
        )

    # All 6 required metrics populated
    assert missing_required_metrics(metrics) == []
    # AR-gate composite formula: 0.4 * (0.6*0.7 + 0.4*0.45) + 0.3*0.5 + 0.3*0.6
    expected_ar_gate = 0.6 * 0.7 + 0.4 * 0.45
    expected_composite = 0.4 * expected_ar_gate + 0.3 * 0.5 + 0.3 * 0.6
    assert metrics["binding_screening_composite"] == pytest.approx(
        round(expected_composite, 4), abs=1e-4
    )
