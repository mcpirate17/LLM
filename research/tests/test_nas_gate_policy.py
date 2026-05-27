"""Unit tests for the explicit NAS gate policy (research/tools/nas_gate_policy.py)."""

from __future__ import annotations

import pytest

from research.tools.nas_gate_policy import (
    DEFAULT_THRESHOLDS,
    GatePolicyConfig,
    NanoBand,
    Stage,
    candidate_from_row,
    classify_nano,
    evaluate_candidate,
    evaluate_candidates,
)

pytestmark = pytest.mark.unit

THR = DEFAULT_THRESHOLDS
CFG = GatePolicyConfig()


def _row(**pred):
    """A bare NAS candidate row with predicted axes."""
    base = {
        "ar_gate": 1.0,
        "ar_curriculum": 0.6,
        "nano_induction_nearest": 0.3,
        "induction": 0.4,
    }
    base.update(pred)
    return {
        "fingerprint": "fp",
        "label_free_probe_predictions": base,
        "failure_risk": {},
    }


def test_all_gates_pass_is_exploit():
    d = evaluate_candidate(candidate_from_row(_row()), THR, CFG)
    assert d.accepted and d.stage is Stage.EXPLOIT
    assert not d.rejections


def test_ar_gate_below_threshold_rejects():
    d = evaluate_candidate(candidate_from_row(_row(ar_gate=0.5)), THR, CFG)
    assert not d.accepted and d.stage is Stage.REJECTED
    assert d.first_failed_gate == "ar_gate"


def test_ar_curriculum_below_threshold_rejects_unknown_graph():
    d = evaluate_candidate(candidate_from_row(_row(ar_curriculum=0.3)), THR, CFG)
    assert d.stage is Stage.REJECTED
    assert any(r.gate == "ar_curriculum" for r in d.rejections)


def test_known_good_rescued_past_predictor_rejection():
    row = _row(ar_gate=0.5, ar_curriculum=0.3)
    row["lit_match_type"] = "family"
    d = evaluate_candidate(candidate_from_row(row), THR, CFG)
    assert not d.accepted and d.stage is Stage.RESCUE
    # rescue records the predictor rejections it overrode.
    assert {r.gate for r in d.rejections} >= {"ar_gate", "ar_curriculum"}


def test_hard_failure_blocks_rescue_even_for_known_good():
    row = _row(ar_gate=0.5)
    row["lit_match_type"] = "exact"
    row["failure_risk"] = {"compile": 0.9}
    d = evaluate_candidate(candidate_from_row(row), THR, CFG)
    assert d.stage is Stage.REJECTED
    assert any(
        r.gate == "failure_risk.compile" and r.kind.value == "hard"
        for r in d.rejections
    )


def test_nano_bands():
    assert classify_nano(0.6, CFG) is NanoBand.STRONG
    assert classify_nano(0.3, CFG) is NanoBand.WEAK
    assert classify_nano(0.12, CFG) is NanoBand.FRONTIER
    assert classify_nano(0.04, CFG) is NanoBand.DEGENERATE
    assert classify_nano(None, CFG) is NanoBand.PROBE_ERROR
    assert classify_nano(0.0, CFG) is NanoBand.PROBE_ERROR


def test_frontier_nano_does_not_gate_or_rank_up():
    # nano in 0.08-0.20 is frontier-neutral: not a rejection, not a rank-up.
    d = evaluate_candidate(
        candidate_from_row(_row(nano_induction_nearest=0.12)), THR, CFG
    )
    assert d.accepted
    assert d.rank_signals["nano_band"] == NanoBand.FRONTIER.value
    assert d.rank_signals["nano_rank_up"] is False


def test_low_nano_is_never_a_sole_reject():
    # nano < 0.08 with everything else passing must NOT reject on its own.
    d = evaluate_candidate(
        candidate_from_row(_row(nano_induction_nearest=0.04)), THR, CFG
    )
    assert d.accepted
    assert not any("nano" in r.gate for r in d.rejections)


def test_degenerate_nano_contributes_only_with_another_failure():
    d = evaluate_candidate(
        candidate_from_row(_row(ar_gate=0.5, nano_induction_nearest=0.04)), THR, CFG
    )
    assert any(r.gate == "nano_degenerate" for r in d.rejections)


def test_nano_persistent_zero_is_execution_no_go():
    row = _row(nano_induction_nearest=0.0)
    row["lit_match_type"] = (
        "exact"  # even known-good cannot rescue an execution failure
    )
    d = evaluate_candidate(candidate_from_row(row), THR, CFG)
    assert d.stage is Stage.REJECTED
    assert any(
        r.gate == "nano_probe_health" and r.kind.value == "hard" for r in d.rejections
    )


def test_nb_recorded_not_folded():
    row = _row()
    row["cheap_actual"] = {"nb05": 0.8, "nb10": 0.95}
    d = evaluate_candidate(candidate_from_row(row), THR, CFG)
    assert d.rank_signals["nb05"] == 0.8 and d.rank_signals["nb10"] == 0.95
    assert d.rank_signals["cheap_evidence_rank_up"] is True  # ar_gate pass + nb10>=0.9


def test_missing_nb_handled():
    d = evaluate_candidate(candidate_from_row(_row()), THR, CFG)
    assert "nb05" not in d.rank_signals and "nb10" not in d.rank_signals


def test_rescue_quota_caps_admissions():
    rows = []
    for i in range(5):
        r = _row(ar_gate=0.5)  # all fail predicted ar_gate
        r["fingerprint"] = f"fp{i}"
        r["lit_match_type"] = "family"
        rows.append(r)
    decisions = evaluate_candidates(rows, THR, GatePolicyConfig(rescue_quota=2))
    assert sum(1 for d in decisions if d.stage is Stage.RESCUE) == 2
    over = [d for d in decisions if d.stage is Stage.REJECTED]
    assert all("rescue quota exhausted" in d.reason for d in over)


def test_ar_gate_advisory_admits_when_target_is_induction():
    # With ar_gate demoted to advisory, a graph failing only predicted ar_gate is admitted on merit.
    cfg = GatePolicyConfig(ar_gate_hard=False)
    d = evaluate_candidate(candidate_from_row(_row(ar_gate=0.3)), THR, cfg)
    assert d.accepted and d.stage is Stage.EXPLOIT
    assert not any(r.gate == "ar_gate" for r in d.rejections)


def test_published_hybrid_is_rescued():
    # The Transformer/Mamba hybrid fails predicted ar_gate but is published-family with no
    # hard failure: it must be admitted for measurement (its real S1 induction was 0.972).
    row = {
        "fingerprint": "b9dbd71c20b26f6a",  # pragma: allowlist secret
        "published_key": "jamba_style_hybrid",
        "predicted": {
            "ar_gate": 0.30,
            "ar_curriculum": 0.29,
            "nano_induction_nearest": 0.146,
            "induction": 0.01,
        },
        "s1_actual": {"s1_induction_auc": 0.972},
    }
    d = evaluate_candidate(candidate_from_row(row), THR, CFG)
    assert d.stage is Stage.RESCUE
