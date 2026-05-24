"""Tests for the component-level distance-resolved binding probe."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.primitive_templates import TropicalAttention
from component_fab.harness.range_binding_probe import (
    RangeBindingResult,
    range_binding_gate,
)
from component_fab.inventor.mechanism_catalog import enumerate_invention_specs
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)


def test_range_probe_structure() -> None:
    torch.manual_seed(0)
    res = range_binding_gate(
        TropicalAttention(16), dim=16, distances=(4, 8), n_train_steps=60, seed=0
    )
    assert isinstance(res, RangeBindingResult)
    assert set(res.per_distance_accuracy) == {4, 8}
    assert all(0.0 <= a <= 1.0 for a in res.per_distance_accuracy.values())
    assert res.effective_distance in (0, 4, 8)
    assert res.random_baseline == 0.125  # n_classes=8 default


def test_range_probe_discriminates_global_vs_per_position() -> None:
    """A per-position ``nn.Linear`` structurally cannot bind across a gap (its
    output at the readout has no information about the key at distance d>0), so
    it must score at baseline while a global mixer binds at short range."""
    torch.manual_seed(0)
    glob = range_binding_gate(
        TropicalAttention(16), dim=16, distances=(4, 8), n_train_steps=200, seed=0
    )
    local = range_binding_gate(
        nn.Linear(16, 16), dim=16, distances=(4, 8), n_train_steps=200, seed=0
    )
    assert glob.aggregate_accuracy > local.aggregate_accuracy
    assert glob.effective_distance >= local.effective_distance
    assert local.effective_distance == 0  # per-position never binds across a gap


def test_validate_capabilities_range_probe_opt_in() -> None:
    spec = enumerate_invention_specs()[0]
    torch.manual_seed(0)
    card_on = validate_capabilities(
        spec,
        TropicalAttention(16),
        dim=16,
        seq_len=16,
        run_range_probe=True,
        range_train_steps=60,
        range_distances=(4, 8),
    )
    on = capability_scorecard_to_dict(card_on)
    assert on["range_ran"] is True
    assert set(on["range_per_distance"]) == {"4", "8"}

    card_off = validate_capabilities(spec, TropicalAttention(16), dim=16, seq_len=16)
    assert capability_scorecard_to_dict(card_off)["range_ran"] is False
