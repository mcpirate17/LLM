"""Unit tests for the shared breakthrough gate helper.

Real-world reference rows used: ``b0c38826`` (legitimate breakthrough,
should pass) and ``5a26e254`` (the d904 ``Gated-MLP`` false positive,
should fail).
"""

from __future__ import annotations

import pytest

from research.scientist.breakthrough_gates import (
    BREAKTHROUGH_CAPABILITY_FLOOR,
    BREAKTHROUGH_COMPOSITE_FLOOR,
    passes_breakthrough_from_row,
    passes_breakthrough_gates,
    passes_capability_floor,
)


def _real_breakthrough_row() -> dict:
    """Snapshot of b0c38826 (real breakthrough, post-rescore)."""
    return {
        "tier": "breakthrough",
        "composite_score": 515.46,
        "validation_baseline_ratio": 0.925,
        "induction_auc": 0.966,
        "binding_composite": 0.292,
        "induction_v2_investigation_auc": 0.994,
        "binding_v2_investigation_auc": 0.921,
    }


def _d904_false_breakthrough_row() -> dict:
    """Snapshot of 5a26e254 / d904 (false positive, after 3-seed rescreen)."""
    return {
        "tier": "breakthrough",
        "composite_score": 334.83,
        "validation_baseline_ratio": 0.939,
        "induction_auc": 0.016,
        "binding_composite": 0.008,
        "induction_v2_investigation_auc": 0.026,
        "binding_v2_investigation_auc": 0.091,
    }


class TestCapabilityFloor:
    def test_real_breakthrough_passes(self):
        row = _real_breakthrough_row()
        assert passes_capability_floor(
            induction_auc=row["induction_auc"],
            binding_composite=row["binding_composite"],
            induction_v2_investigation_auc=row["induction_v2_investigation_auc"],
            binding_v2_investigation_auc=row["binding_v2_investigation_auc"],
        )

    def test_d904_fails(self):
        row = _d904_false_breakthrough_row()
        assert not passes_capability_floor(
            induction_auc=row["induction_auc"],
            binding_composite=row["binding_composite"],
            induction_v2_investigation_auc=row["induction_v2_investigation_auc"],
            binding_v2_investigation_auc=row["binding_v2_investigation_auc"],
        )

    def test_all_none_fails(self):
        assert not passes_capability_floor()

    def test_single_signal_above_floor_passes(self):
        # Only one metric populated, exactly at the floor — passes.
        assert passes_capability_floor(induction_auc=BREAKTHROUGH_CAPABILITY_FLOOR)

    def test_string_values_coerced(self):
        # Some callers pass strings from JSON — must coerce safely.
        assert passes_capability_floor(induction_auc="0.5")
        assert not passes_capability_floor(induction_auc="nope")

    def test_floor_override(self):
        assert passes_capability_floor(induction_auc=0.05, floor=0.05)
        assert not passes_capability_floor(induction_auc=0.05, floor=0.06)


class TestBreakthroughGates:
    def test_real_breakthrough_passes(self):
        passed, reason = passes_breakthrough_from_row(_real_breakthrough_row())
        assert passed
        assert reason is None

    def test_d904_fails_on_composite(self):
        passed, reason = passes_breakthrough_from_row(_d904_false_breakthrough_row())
        assert not passed
        # composite=334.83 < 450 floor
        assert reason == "composite_below_floor"

    def test_d904_fails_on_capability_when_composite_inflated(self):
        # Even if rescore had not dropped composite, capability would block.
        row = _d904_false_breakthrough_row()
        passed, reason = passes_breakthrough_from_row(
            row, composite_score=BREAKTHROUGH_COMPOSITE_FLOOR + 100
        )
        assert not passed
        assert reason == "capability_signal_below_floor"

    def test_baseline_ratio_one_blocks(self):
        # validation_baseline_ratio >= 1.0 means no improvement vs reference.
        passed, reason = passes_breakthrough_gates(
            composite_score=600.0,
            val_baseline_ratio=1.0,
            induction_auc=0.9,
        )
        assert not passed
        assert reason == "no_baseline_improvement"

    def test_missing_composite_fails(self):
        passed, reason = passes_breakthrough_gates(induction_auc=0.9)
        assert not passed
        assert reason == "composite_below_floor"

    def test_composite_override_used(self):
        # Row has stale 499 composite, override with new lower value.
        row = _real_breakthrough_row()
        row["composite_score"] = 200.0  # stale row
        passed, _ = passes_breakthrough_from_row(row, composite_score=600.0)
        assert passed

    def test_floor_constants_round_trip(self):
        # If user later edits constants, gate semantics still hold at boundary.
        passed, _ = passes_breakthrough_gates(
            composite_score=BREAKTHROUGH_COMPOSITE_FLOOR,
            val_baseline_ratio=0.5,
            induction_auc=BREAKTHROUGH_CAPABILITY_FLOOR,
        )
        assert passed


@pytest.mark.parametrize(
    "composite,baseline,induction,binding,expected_passed",
    [
        (515.0, 0.92, 0.96, 0.30, True),  # b0c38826-like
        (335.0, 0.94, 0.02, 0.01, False),  # d904-like
        (500.0, 0.94, 0.02, 0.01, False),  # composite high but no capability
        (500.0, 0.94, 0.20, 0.01, True),  # composite + induction signal only
        (500.0, 0.94, None, 0.20, True),  # binding signal only
        (449.99, 0.5, 0.9, 0.5, False),  # composite just below floor
    ],
)
def test_gate_matrix(composite, baseline, induction, binding, expected_passed):
    passed, _ = passes_breakthrough_gates(
        composite_score=composite,
        val_baseline_ratio=baseline,
        induction_auc=induction,
        binding_composite=binding,
    )
    assert passed is expected_passed
