"""Tests for the measured graph-property screen (real-module binding filter)."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.proposer.measured_screen import (  # noqa: F401
    LONG_RANGE_THRESHOLD,
    MAX_CAUSALITY_VIOLATION,
    MeasuredScreen,
    _FabProbeAdapter,
    _capability_score,
    _unavailable,
    measured_screen_for_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.tests.conftest import make_spec


def _spec(axes: dict, pid: str = "ms_cand") -> ProposalSpec:
    return make_spec(axes, pid, name="cand", category="lane")


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


def test_rank_rewards_binding_and_is_causality_neutral() -> None:
    """The capability RANK rewards backward routing but is neutral to causality.

    ``causality_violation`` measured at random init is noise (nas_funnel_ood_eval
    ROC 0.49), so the shared, calibrated ``capability_score_from_descriptors`` —
    to which ``_capability_score`` is a pure delegate — deliberately zero-weights
    it (see ``research/tests/test_measured_capability_score.py``). Acausality is a
    *disqualifier*, not a rank term; it is handled by the hard gate, asserted
    separately below. Folding it into the rank would de-calibrate the oracle.
    """
    binder = {
        "long_range_reach": 0.8,
        "content_match_gating": 0.1,
        "content_dependence": 0.2,
        "causality_violation": 0.0,
    }
    acausal = {**binder, "causality_violation": 0.9}
    nonbinder = {k: 0.0 for k in binder}
    # Rank rewards binding ...
    assert _capability_score(binder) > _capability_score(nonbinder)
    assert _capability_score(nonbinder) == 0.0
    # ... but is neutral to causality (the delegate does not penalize it).
    assert _capability_score(binder) == _capability_score(acausal)


def test_acausality_is_disqualified_by_the_hard_gate_not_the_rank() -> None:
    """Acausal candidates are caught by ``downstream_gate_pass``, not the rank.

    This is the correct locus: ``causality_violation > MAX_CAUSALITY_VIOLATION``
    fails the gate, and a failed downstream gate cuts the NAS score multiplier
    (and the quality verdict, ``quality.py``). So acausal ops are demoted in the
    funnel even though the capability rank stays calibrated and causality-neutral.
    """
    from component_fab.proposer.nas_screen import (
        NasScreenResult,
        nas_score_multiplier,
    )

    def _result(causality_violation: float) -> NasScreenResult:
        return NasScreenResult(
            proposal_id="x",
            available=True,
            gate_pass=True,
            downstream_gate_pass=causality_violation <= MAX_CAUSALITY_VIOLATION,
            rank_score=1.0,
            source="measured_descriptors",
        )

    causal = _result(0.0)
    acausal = _result(0.9)
    assert causal.downstream_gate_pass
    assert not acausal.downstream_gate_pass
    # The hard gate demotes the acausal candidate via the score multiplier.
    assert nas_score_multiplier(acausal) < nas_score_multiplier(causal)


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
