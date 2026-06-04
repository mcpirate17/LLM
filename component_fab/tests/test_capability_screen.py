"""Tests for capability-screener scoring of fab specs (diagnostic signal).

The screeners predict induction/nano CAPABILITY cross-family; a 2026-06-03 audit
found they do NOT predict Tier-2 'beats baseline' on the fab population (induction
r~0, nano r~-0.28), so they are diagnostic-only — these tests pin the plumbing,
not a ranking claim.
"""

from __future__ import annotations

from typing import Any

from component_fab.proposer.capability_screen import (
    CapabilityScreen,
    _FAB_CLASS_TO_OP,
    capability_screen_for_spec,
    fab_op_multiset,
)
from component_fab.proposer.spec_generator import ProposalSpec


def _spec(axes: dict[str, Any], pid: str = "cap_cand") -> ProposalSpec:
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


def test_op_multiset_is_faithful_and_scaled() -> None:
    spec = _spec({"op_algebraic_space": "tropical"})
    ops = fab_op_multiset(spec, dim=16, n_blocks=2)
    # scaffold the harness wraps around the lane is always present
    assert "embedding_lookup" in ops
    assert ops.count("layernorm") == 3  # n_blocks + 1
    assert "linear_proj" in ops  # tied lm head (+ any lane linears)
    # all mapped ops are valid vocab entries (no stray class names leak through)
    assert all(
        op in set(_FAB_CLASS_TO_OP.values()) | {"embedding_lookup", "layernorm"}
        for op in ops
    )


def test_capability_screen_returns_record() -> None:
    spec = _spec(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 1,
            "op_geometric_receptive_field": "global",
        }
    )
    cs = capability_screen_for_spec(spec, dim=16)
    assert isinstance(cs, CapabilityScreen)
    assert cs.proposal_id == spec.proposal_id
    if cs.available:
        assert cs.op_count > 0
        assert cs.n_distinct_ops > 0
    else:
        # fail-open: a missing screener must not crash the caller
        assert cs.op_count == 0
