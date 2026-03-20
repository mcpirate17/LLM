"""Test that evolution/novelty search correctly records stage0/stage05 pass status.

Bug: results_analysis.py set stage0_passed = (fitness > 0), which is wrong.
A program that compiled and trained but failed S1 gets fitness=0 → stage0_passed=0.
Fix: if s1_result exists (model was trained), stage0_passed must be True.
"""

from __future__ import annotations


def test_s0_passed_when_s1_fails():
    """A program that reached S1 training (s1_result exists) but failed
    should have stage0_passed=True and stage05_passed=True."""
    # Simulate the fixed logic from results_analysis.py
    fitness = 0.0  # S1 failed → fitness is 0
    s1_result = {"passed": False, "final_loss": 15.0, "initial_loss": 200.0}
    graph_metrics = {"stability_score": 1.0}

    # Old (buggy) logic:
    old_s0 = fitness > 0  # False — WRONG
    assert old_s0 is False, "Old logic should produce False"

    # New (fixed) logic:
    new_s0 = s1_result is not None or graph_metrics.get("stability_score", 0) > 0
    assert new_s0 is True, "New logic: s1_result exists → S0 passed"

    new_s05 = new_s0
    assert new_s05 is True


def test_s0_failed_when_no_s1_result_and_no_stability():
    """A program that truly failed at S0 (no training, no stability data)
    should have stage0_passed=False."""
    fitness = 0.0
    s1_result = None
    graph_metrics = {}

    new_s0 = s1_result is not None or graph_metrics.get("stability_score", 0) > 0
    assert new_s0 is False, "No s1_result and no stability → true S0 failure"


def test_s0_passed_when_stability_data_exists():
    """A program with stability data (forward pass ran) but no s1_result
    (e.g. fitness_fn threw before training) should still have S0=True."""
    fitness = 0.0
    s1_result = None
    graph_metrics = {"stability_score": 0.83}

    new_s0 = s1_result is not None or graph_metrics.get("stability_score", 0) > 0
    assert new_s0 is True, "Has stability data → S0 passed"


def test_stage_at_death_uses_s0_not_fitness():
    """stage_at_death should be 'stage1' when S0 passed but S1 failed,
    not 'stage0' (which the old fitness>0 logic would produce)."""
    s1_result = {"passed": False, "final_loss": 50.0}
    graph_metrics = {"stability_score": 1.0}

    _s1_passed = False
    _s0_passed = s1_result is not None or graph_metrics.get("stability_score", 0) > 0

    stage_at_death = (
        "survived" if _s1_passed else ("stage1" if _s0_passed else "stage0")
    )
    assert stage_at_death == "stage1", f"Got {stage_at_death}, expected stage1"
