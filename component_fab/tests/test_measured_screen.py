"""Tests for the measured graph-property screen (real-module binding filter)."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.proposer.measured_screen import (  # noqa: F401
    LONG_RANGE_THRESHOLD,
    MeasuredScreen,
    _FabProbeAdapter,
    _capability_score,
    _unavailable,
    measured_screen_for_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec


def _spec(axes: dict, pid: str = "ms_cand") -> ProposalSpec:
    return ProposalSpec(
        proposal_id=pid,
        name="cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        math_axes=axes,
        anchor_witness_op="",
        anchor_witnesses_all=(),
        declared_property_row=dict(axes),
        predicted_lift=0.5,
        rationale="test",
    )


def test_adapter_exposes_probe_contract_and_unwraps_tuple() -> None:
    class TupleLane(nn.Module):
        def forward(self, x):
            return x * 2.0, "aux"

    adapter = _FabProbeAdapter(TupleLane(), dim=8)
    ids = torch.zeros(2, 5, dtype=torch.long)
    emb = adapter.embed(ids)
    assert emb.shape == (2, 5, 8)
    out = adapter._fingerprint_forward_from_embed(emb)
    assert torch.is_tensor(out)  # tuple unwrapped to its first element
    assert out.shape == emb.shape


def test_rank_rewards_backward_routing_penalizes_acausality() -> None:
    binder = {
        "long_range_reach": 0.8,
        "content_match_gating": 0.1,
        "content_dependence": 0.2,
        "causality_violation": 0.0,
    }
    acausal = {**binder, "causality_violation": 0.9}
    nonbinder = {k: 0.0 for k in binder}
    assert _capability_score(binder) > _capability_score(acausal)
    assert _capability_score(binder) > _capability_score(nonbinder)
    assert _capability_score(nonbinder) == 0.0


def test_unavailable_fails_open() -> None:
    ms = _unavailable("p1", "boom")
    assert ms.available is False
    assert ms.binds_likely is True  # fail open: never silently drop
    assert ms.reason == "boom"


def test_measured_screen_for_spec_returns_consistent_record() -> None:
    # A stateful global lane should build and probe; assert a well-formed record
    # regardless of environment (fail-open keeps binds_likely True if unprobeable).
    spec = _spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        }
    )
    ms = measured_screen_for_spec(spec, dim=16)
    assert isinstance(ms, MeasuredScreen)
    assert ms.proposal_id == spec.proposal_id
    if ms.available:
        assert ms.binds_likely == (ms.long_range_reach >= LONG_RANGE_THRESHOLD)
        assert ms.rank_score >= 0.0
        assert ms.descriptors is not None
    else:
        assert ms.binds_likely is True
