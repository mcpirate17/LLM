"""The SSM-fair state-tracking subscore + its additive composite bonus.

Locks in the 2026-06-07 wiring: non-QKV mechanisms that learn the SSM-favoured
probe tasks (state-tracking / copy / compression) earn composite credit the
binding-only composite ignored. Reads the per-task loss ratios already in the
in_context probe scorecard — no new compute.
"""

from __future__ import annotations

import pytest

from component_fab.improver.ranking import (
    _SSM_FAVOURED_TASKS,
    _STATE_TRACK_BONUS,
    composite_score,
    state_tracking_subscore,
)

_SMOKE_OK = {
    "forward_passed": True,
    "backward_passed": True,
    "output_finite": True,
    "param_grad_finite": True,
}


def _probe(ratio: float, tasks: tuple[str, ...] = _SSM_FAVOURED_TASKS) -> dict:
    return {
        "per_task": {
            t: {"loss_ratio_initial_over_final": ratio, "trained_successfully": True}
            for t in tasks
        },
        "aggregate_loss_ratio": ratio,
    }


def test_subscore_zero_without_scorecard_or_learning() -> None:
    assert state_tracking_subscore(None) == 0.0
    assert state_tracking_subscore({}) == 0.0
    # ratio <= 1.0 means no reduction → no credit.
    assert state_tracking_subscore(_probe(1.0)) == 0.0
    # untrained tasks are skipped.
    untrained = {
        "per_task": {
            "causal_max": {
                "loss_ratio_initial_over_final": 9.0,
                "trained_successfully": False,
            }
        }
    }
    assert state_tracking_subscore(untrained) == 0.0


def test_subscore_rewards_state_tracking_reduction() -> None:
    weak = state_tracking_subscore(_probe(1.5))
    strong = state_tracking_subscore(_probe(8.0))
    assert 0.0 < weak < strong <= 1.0


def test_subscore_ignores_recall_induction_axis() -> None:
    # causal_induction is the attention-favoured axis — it must NOT feed the
    # SSM-fair subscore even with a huge ratio.
    recall_only = {
        "per_task": {
            "causal_induction": {
                "loss_ratio_initial_over_final": 50.0,
                "trained_successfully": True,
            }
        }
    }
    assert state_tracking_subscore(recall_only) == 0.0


def test_composite_adds_state_tracking_bonus() -> None:
    solo = {"smoke": _SMOKE_OK, "property_cross_check": {}}
    # Hold aggregate_loss_ratio constant across both cards so learning_subscore is
    # identical and the composite delta isolates the state-tracking bonus.
    strong = _probe(6.0)
    strong["aggregate_loss_ratio"] = 1.0  # learning_subscore == 0 for both
    flat = _probe(1.0)
    flat["aggregate_loss_ratio"] = 1.0

    score, components = composite_score(solo, strong, None)
    assert "state_tracking" in components
    st = components["state_tracking"]
    assert st > 0.0
    flat_score = composite_score(solo, flat, None)[0]
    assert score - flat_score == pytest.approx(_STATE_TRACK_BONUS * st, abs=1e-9)
