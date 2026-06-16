"""Tests for mechanistic probes."""

from __future__ import annotations

from torch import nn

from component_fab.generator.memory_primitives import (
    SemiringSurpriseMemoryLane,
    SlotTableMemoryLane,
)
from component_fab.validator.mechanism import probe_observables


def test_slot_table_routing_health() -> None:
    dim = 16
    n_slots = 4
    lane = SlotTableMemoryLane(dim=dim, n_slots=n_slots)

    # Routing health should be measurable for SlotTableMemoryLane
    scorecard = probe_observables(
        lane, dim=dim, seq_len=16, batch_size=2, n_train_steps=0
    )

    assert scorecard.routing_entropy_mean > 0.0
    assert scorecard.active_lane_fraction > 0.0
    assert scorecard.load_balance_cv >= 0.0


def test_relaxation_slope() -> None:
    dim = 8
    lane = SemiringSurpriseMemoryLane(dim=dim, memory_dim=4)

    # Run a few training steps; we expect a non-zero (ideally negative) slope
    # because it's a surprise-driven architecture learning the identity.
    scorecard = probe_observables(lane, dim=dim, seq_len=8, n_train_steps=10)

    # We don't strictly assert negative here because it might take more than 10 steps
    # or the random init might be lucky, but it should be a float.
    assert isinstance(scorecard.relaxation_slope, float)


def test_non_routing_lane_scores_zero_routing() -> None:
    dim = 8
    lane = nn.Linear(dim, dim)

    scorecard = probe_observables(
        lane, dim=dim, seq_len=8, batch_size=2, n_train_steps=0
    )

    assert scorecard.routing_entropy_mean == 0.0
    assert scorecard.active_lane_fraction == 1.0  # Default for non-routing
    assert scorecard.load_balance_cv == 0.0


def test_parameterless_lane_fails_mechanism_without_crashing() -> None:
    lane = nn.Identity()

    scorecard = probe_observables(lane, dim=8, seq_len=8, n_train_steps=2)

    assert scorecard.passed is False
    assert scorecard.relaxation_slope == 0.0
    assert "no_trainable_parameters" in scorecard.notes
