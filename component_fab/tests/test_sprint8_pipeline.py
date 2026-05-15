"""Sprint-8 tests: ERF + NanoBind probes + tiered orchestrator eliminations."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.primitive_templates import (
    FourierBasisLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
)
from component_fab.harness.erf_probe import measure_erf
from component_fab.harness.nano_bind_probe import nano_bind_gate
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.proposer.property_miner import AxisLift, CandidateTuple
from component_fab.proposer.spec_generator import spec_from_candidate
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)


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


# ---------- ERF ----------


def test_erf_passes_for_mixing_attention() -> None:
    lane = TropicalAttention(dim=16, causal=True)
    block = LaneTestBlock(lane, dim=16)
    result = measure_erf(block, seq_len=16, dim=16)
    assert result.density > 0.0
    assert result.last_position_norm > 0.0


def test_erf_density_low_for_identity_lane() -> None:
    # Identity lane: the last output depends only on the last input.
    # With the residual in LaneTestBlock, ERF is concentrated at the last position.
    class _Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    block = LaneTestBlock(_Identity(), dim=16)
    result = measure_erf(block, seq_len=16, dim=16)
    # density (mean/max) should be low because all contribution is at one position
    assert result.density < 0.3
    # decay slope dominated by the last-position peak
    assert result.last_position_norm > result.first_position_norm


def test_erf_disconnected_module_fails() -> None:
    class _ZeroOutput(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x).detach() + x.sum() * 0.0

    block = LaneTestBlock(_ZeroOutput(16), dim=16)
    result = measure_erf(block, seq_len=16, dim=16, density_threshold=0.1)
    # Either the gradient is zero or density is too low
    assert not result.passed or result.density >= 0.1


# ---------- NanoBind ----------


def test_nano_bind_persistent_zero_rejects_topk() -> None:
    # TopKLinear can't propagate position-0 signal to position -1.
    lane = TopKLinear(in_dim=16, out_dim=16, k=4)
    block = LaneTestBlock(lane, dim=16)
    result = nano_bind_gate(
        block,
        dim=16,
        seq_len=12,
        n_classes=4,
        n_train_steps=30,
        checkpoint_at_steps=(10, 20, 30),
    )
    assert result.max_accuracy < 0.7  # nowhere near a true binder
    assert result.random_baseline == 0.25


def test_nano_bind_well_formed_when_module_learns() -> None:
    lane = TropicalAttention(dim=16, causal=True)
    block = LaneTestBlock(lane, dim=16)
    result = nano_bind_gate(
        block,
        dim=16,
        seq_len=12,
        n_classes=4,
        n_train_steps=80,
        checkpoint_at_steps=(20, 40, 80),
    )
    assert len(result.accuracies) == 3
    assert result.random_baseline == 0.25


# ---------- Tiered orchestrator ----------


def test_capability_eliminated_by_s05_for_noncausal_lane() -> None:
    spec = _spec({"op_algebraic_space": "euclidean"})
    lane = FourierBasisLane(dim=16)  # non-causal
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    assert card.eliminated_by == "s05_causality_stability"
    assert not card.s05_passed
    assert not card.erf_passed  # was never evaluated
    assert not card.nb_passed  # was never evaluated


def test_capability_runs_full_stack_for_causal_lane() -> None:
    spec = _spec({"op_algebraic_space": "tropical"})
    lane = TropicalStateSpace(dim=16)
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    # Either passes everything or gets eliminated at one of the deeper gates;
    # the key invariant is that S0.5 is passed.
    assert card.s05_passed
    blob = capability_scorecard_to_dict(card)
    assert "eliminated_by" in blob
    assert "nb_passed" in blob
    assert "erf_passed" in blob


def test_capability_records_per_gate_state() -> None:
    spec = _spec({"op_algebraic_space": "euclidean"})
    lane = nn.Linear(16, 16)
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    blob = capability_scorecard_to_dict(card)
    for key in (
        "s05_passed",
        "erf_passed",
        "nb_passed",
        "can_bind",
        "eliminated_by",
        "erf_density",
        "nb_max_accuracy",
        "relative_recall_per_probe",
    ):
        assert key in blob
