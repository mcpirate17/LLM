"""Combined dispatch + probe + ranking tests for sprint 4."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.primitive_templates import (
    CliffordAttention,
    PadicProjection,
    SpikingActivationGate,
    TropicalTopKStateSpace,
)
from component_fab.harness.probe_block import (
    WinnerLikeBlock,
    short_training_probe,
)
from component_fab.improver.ranking import (
    composite_score,
    cross_check_subscore,
    leaderboard_to_json,
    learning_subscore,
    rank_proposals,
    smoke_subscore,
)
from component_fab.validator.in_context import validate_in_context
from component_fab.proposer.property_miner import AxisLift, CandidateTuple
from component_fab.proposer.spec_generator import spec_from_candidate


def _spec(axes: dict):
    lifts = tuple(
        AxisLift(
            axis=k,
            value=v,
            n_ops=1,
            total_evals=1,
            total_s1_pass=0,
            pass_rate=0.5,
            representative_ops=(),
        )
        for k, v in axes.items()
    )
    candidate = CandidateTuple(
        tuple_values=tuple(axes.items()),
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=("anchor",),
    )
    return spec_from_candidate(candidate)


def test_dispatch_tropical_topk_with_state() -> None:
    m = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 1,
            "op_activation_sparsity_pattern": "top_k",
        },
        dim=16,
    )
    assert isinstance(m, TropicalTopKStateSpace)


def test_dispatch_tropical_takes_priority_over_topk() -> None:
    m = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_activation_sparsity_pattern": "top_k",
        },
        dim=16,
    )
    # tropical wins (TropicalAttention), not TopKLinear
    from component_fab.generator.primitive_templates import TropicalAttention

    assert isinstance(m, TropicalAttention)


def test_dispatch_clifford() -> None:
    m = generate_module({"op_algebraic_space": "clifford"}, dim=16)
    assert isinstance(m, CliffordAttention)


def test_dispatch_clifford_falls_back_when_dim_invalid() -> None:
    m = generate_module({"op_algebraic_space": "clifford"}, dim=15)
    assert isinstance(m, nn.Linear)


def test_dispatch_spiking() -> None:
    m = generate_module({"op_algebraic_space": "spiking"}, dim=16)
    assert isinstance(m, SpikingActivationGate)


def test_dispatch_padic() -> None:
    m = generate_module({"op_algebraic_space": "padic"}, dim=16)
    assert isinstance(m, PadicProjection)


def test_dispatch_padic_falls_back_when_dim_invalid() -> None:
    m = generate_module({"op_algebraic_space": "padic"}, dim=15)
    assert isinstance(m, nn.Linear)


def test_winner_like_block_preserves_shape() -> None:
    lane = nn.Linear(16, 16)
    block = WinnerLikeBlock(lane, dim=16)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_short_training_probe_linear_lane_learns() -> None:
    lane = nn.Linear(16, 16)
    result = short_training_probe(lane, dim=16, seq_len=16, n_steps=80, batch_size=4)
    assert result.trained_successfully
    assert result.loss_ratio_initial_over_final > 1.0


def test_validate_in_context_returns_scorecard() -> None:
    spec = _spec({"op_algebraic_space": "euclidean"})
    lane = nn.Linear(16, 16)
    card = validate_in_context(spec, lane, dim=16, seq_len=16, n_steps=40)
    assert card.proposal_id == spec.proposal_id
    assert card.per_task
    assert "running_mean" in card.per_task
    assert isinstance(card.aggregate_loss_ratio, float)
    assert isinstance(card.learned_signal, bool)


def test_composite_score_combines_three_subscores() -> None:
    solo = {
        "smoke": {
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
        },
        "property_cross_check": {
            "tropical_consistent": True,
            "state_consistent": True,
        },
    }
    probe = {
        "aggregate_loss_ratio": 10.0,
        "learned_signal": True,
    }
    score, comps = composite_score(solo, probe)
    assert comps["smoke"] == 1.0
    assert comps["cross_check"] == 1.0
    assert 0.0 < comps["learning"] <= 1.0
    assert 0.0 < score <= 1.0


def test_smoke_subscore_zero_when_any_check_fails() -> None:
    assert (
        smoke_subscore(
            {
                "forward_passed": True,
                "backward_passed": False,
                "output_finite": True,
                "param_grad_finite": True,
            }
        )
        == 0.0
    )


def test_cross_check_subscore_handles_no_consistent_keys() -> None:
    assert cross_check_subscore({"some_metric": 0.42}) == 1.0


def test_learning_subscore_zero_for_no_improvement() -> None:
    assert learning_subscore({"aggregate_loss_ratio": 1.0}) == 0.0
    assert learning_subscore({"aggregate_loss_ratio": 0.5}) == 0.0
    assert learning_subscore(None) == 0.0


def test_rank_proposals_orders_by_composite() -> None:
    high = {
        "proposal_id": "high",
        "name": "high",
        "category": "lane",
        "synthesis_kind": "semiring_swap",
        "promoted": True,
        "smoke": {
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
        },
        "property_cross_check": {"tropical_consistent": True},
    }
    low = {
        "proposal_id": "low",
        "name": "low",
        "category": "lane",
        "synthesis_kind": "semiring_swap",
        "promoted": False,
        "smoke": {"forward_passed": False},
        "property_cross_check": {},
    }
    ranked = rank_proposals([low, high])
    assert ranked[0].proposal_id == "high"
    assert ranked[1].proposal_id == "low"
    json_out = leaderboard_to_json(ranked)
    assert json_out[0]["rank"] == 1
    assert json_out[0]["proposal_id"] == "high"
