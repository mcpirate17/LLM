# pyright: reportPrivateImportUsage=false
"""Tests for SparsemaxAttention + nano_induction_gate + capability wiring.

CPU-only. The probe sizes (dim=16, seq_len=24, 150 steps) keep each test
under ~5s on a single CPU core.
"""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.block_templates import ThreeLaneAdaptive
from component_fab.generator.primitive_templates import (
    MultiscaleWaveletLane,
    SparsemaxAttention,
    TropicalAttention,
)
from component_fab.harness.nano_induction_probe import (
    NanoInductionResult,
    nano_induction_gate,
)
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.harness.tiny_lm import SoftmaxCausalAttention
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
    cand = CandidateTuple(
        tuple_values=tuple(axes.items()),
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=("anchor",),
    )
    return spec_from_candidate(cand)


# ---------- SparsemaxAttention ----------


def test_sparsemax_attention_forward_shape_is_preserved() -> None:
    lane = SparsemaxAttention(dim=16, causal=True)
    x = torch.randn(2, 12, 16)
    out = lane(x)
    assert out.shape == x.shape


def test_sparsemax_weights_sum_to_one_per_query() -> None:
    """End-to-end: a SparsemaxAttention with V=identity returns the sparse-weighted
    sum of inputs, so the row-sum on V's output equals 1 only when inputs were
    1s. Use the internal sparsemax via the lane's affinity-only path."""
    from component_fab.generator.primitive_templates import _causal_sparsemax

    logits = torch.randn(3, 7, 7)
    causal = torch.triu(torch.full((7, 7), -1e4), diagonal=1)
    weights = _causal_sparsemax(logits + causal)
    assert (weights >= 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_sparsemax_is_causal() -> None:
    """Perturbing a future position must not change current-position output."""
    torch.manual_seed(0)
    lane = SparsemaxAttention(dim=16, causal=True).eval()
    x = torch.randn(1, 8, 16)
    y = lane(x).detach().clone()
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] = torch.randn(1, 16)  # change last position only
    y2 = lane(x_perturbed).detach()
    # First 7 positions unchanged.
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)


# ---------- nano_induction_gate ----------


def test_nano_induction_returns_well_formed_result() -> None:
    """Tropical alone (1-block wrapper) is expected to score near baseline.
    We only assert the result object is well-formed, not that it passes."""
    lane = TropicalAttention(dim=16, causal=True)
    block = LaneTestBlock(lane, dim=16)
    result = nano_induction_gate(
        block,
        dim=16,
        seq_len=16,
        n_classes=4,
        n_train_steps=30,
        checkpoint_at_steps=(10, 20, 30),
    )
    assert isinstance(result, NanoInductionResult)
    assert len(result.accuracies) == 3
    assert result.random_baseline == 0.25
    assert 0.0 <= result.max_accuracy <= 1.0


def test_nano_induction_softmax_two_stack_can_exceed_baseline() -> None:
    """2 stacked LaneTestBlocks over softmax attention should learn induction
    above baseline in 150 steps. This is the canonical positive control."""
    torch.manual_seed(0)
    lane = SoftmaxCausalAttention(dim=16)
    stacked = nn.Sequential(LaneTestBlock(lane, 16), LaneTestBlock(lane, 16))
    result = nano_induction_gate(
        stacked,
        dim=16,
        seq_len=24,
        n_classes=8,
        n_train_steps=150,
        checkpoint_at_steps=(50, 100, 150),
        seed=0,
    )
    assert result.max_accuracy > result.random_baseline, (
        f"softmax 2-stack failed to clear baseline: max={result.max_accuracy:.3f} "
        f"baseline={result.random_baseline:.3f}"
    )


def test_nano_induction_records_notes_on_failure() -> None:
    """A broken block (raises) yields a soft-fail result with notes, not an
    exception escaping the harness."""

    class _BrokenLane(nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            raise RuntimeError("intentional")

    result = nano_induction_gate(
        _BrokenLane(), dim=16, seq_len=16, n_train_steps=5, checkpoint_at_steps=(5,)
    )
    assert result.above_baseline is False
    assert any("RuntimeError" in note for note in result.notes)


# ---------- validate_capabilities wiring ----------


def test_validate_capabilities_runs_induction_when_nb_passes() -> None:
    """Softmax attention should clear S0.5+ERF+NB, so induction must run
    and populate the new fields in the scorecard."""
    spec = _spec({"op_algebraic_space": "euclidean"})
    lane = SoftmaxCausalAttention(dim=16)
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    blob = capability_scorecard_to_dict(card)
    if card.nb_passed:
        assert blob["ind_ran"] is True
        assert "ind_max_accuracy" in blob
        assert "ind_above_baseline" in blob
        assert 0.0 <= blob["ind_max_accuracy"] <= 1.0


def test_validate_capabilities_never_eliminates_on_induction() -> None:
    """Induction is a soft signal — eliminated_by must never be 'nano_induction'."""
    for axes in (
        {"op_algebraic_space": "tropical"},
        {"op_algebraic_space": "euclidean"},
    ):
        spec = _spec(axes)
        lane = (
            TropicalAttention(dim=16)
            if axes["op_algebraic_space"] == "tropical"
            else SoftmaxCausalAttention(dim=16)
        )
        card = validate_capabilities(spec, lane, dim=16, seq_len=16)
        assert card.eliminated_by != "nano_induction"


# ---------- 3-lane composite ----------


def test_three_lane_tropical_sparsemax_wavelet_runs() -> None:
    """The proposed 3-lane gate must forward + backward without error."""
    torch.manual_seed(0)
    block = ThreeLaneAdaptive(
        lambda d: TropicalAttention(d),
        lambda d: SparsemaxAttention(d),
        lambda d: MultiscaleWaveletLane(d),
        16,
    )
    x = torch.randn(2, 12, 16, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
