# pyright: reportPrivateImportUsage=false
"""Tests for the learning-dynamics trajectory now captured by the capability
probes and wired through the scorecard.

Covers the user objective "rapid induction and binding and ace associated
recall in **minimum steps**" — the trajectory ``train_lane_head`` already
computed but the old ``train_and_score`` discarded after reading only
``final_loss``.

NOTE on the contrast metric: at this tiny scale (dim 16, 60 steps) the
``LaneTestBlock`` readout head has enough capacity to memorize *some* recall
even over a no-mixing backbone, and ``relative_recall = 1 - final/init``
flatters a lane with a terrible init (identity starts at MSE ~83 and the head
improves it 30%; a real mixer starts near-optimal and improves less in relative
terms). So the honest, scale-robust discriminator is **absolute final masked
MSE** — a real mixer routes the key->value association to the query position
and reaches a strictly lower absolute loss than an identity backbone whose head
must work from the raw query token alone. The relative_recall artifact itself
is recorded in the robustness audit note.
"""

from __future__ import annotations

import torch
from torch import nn

from component_fab.harness.capability_probes import make_ar_probe, train_and_score
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.harness.tiny_lm import SoftmaxCausalAttention
from component_fab.tests.conftest import make_candidate_spec
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)

DIM = 16
SEQ = 16


class _IdentityLane(nn.Module):
    """No mixing at all — output == input; only the readout head can learn."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _score(lane: nn.Module, *, seed: int = 0):
    probe = make_ar_probe(
        n_pairs=2, name="ar_unit", pass_threshold=0.3, n_train_steps=60
    )
    return train_and_score(
        LaneTestBlock(lane, DIM).train(), probe, seq_len=SEQ, dim=DIM, seed=seed
    )


def test_learning_curve_is_captured_and_nonincreasing_for_a_mixer() -> None:
    """A mixer's masked-MSE trajectory must be recorded and not get worse."""
    result = _score(SoftmaxCausalAttention(DIM))
    assert len(result.learning_curve) >= 2
    assert result.learning_curve[-1] <= result.learning_curve[0]
    # steps_to_threshold is either a positive checkpoint step or None (never reached).
    assert result.steps_to_threshold is None or result.steps_to_threshold > 0


def test_mixer_achieves_lower_absolute_recall_loss_than_identity() -> None:
    """Honest contrast: a real mixer reaches strictly lower absolute query MSE."""
    mixer = _score(SoftmaxCausalAttention(DIM), seed=1)
    identity = _score(_IdentityLane(), seed=1)
    assert mixer.learning_curve[-1] < identity.learning_curve[-1]


def test_validate_capabilities_surfaces_learning_dynamics_and_mixing() -> None:
    """The scorecard + dict must carry the trajectory and mixing fields."""
    spec = make_candidate_spec({"op_algebraic_space": "euclidean"})
    card = validate_capabilities(
        spec, SoftmaxCausalAttention(DIM), dim=DIM, seq_len=SEQ
    )
    blob = capability_scorecard_to_dict(card)

    for key in (
        "learning_curves_per_probe",
        "steps_to_threshold_per_probe",
        "mean_steps_to_threshold",
        "mixing_subscore",
        "mixing_reach_subscore",
        "mixing_breadth_subscore",
        "mixing_offdiag_mass_fraction",
        "mixing_effective_rank",
    ):
        assert key in blob

    # Softmax attention clears the full hard-gate stack (see
    # test_nano_induction_and_sparsemax), so the soft signals must have run.
    if card.nb_passed:
        assert card.learning_curves_per_probe  # at least one probe's curve
        for curve in card.learning_curves_per_probe.values():
            assert len(curve) >= 1
        # A global mixer must score above zero on the mixing axis.
        assert card.mixing_subscore > 0.0
        assert card.mixing_offdiag_mass_fraction > 0.0
