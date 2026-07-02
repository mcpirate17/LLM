"""Tests for NM-C20 — mixture-of-subspaces mixer.

Pins the spec (compaction-lanes note, Lever 6):
- O(r·d) params for total subspace width r = m·s ≪ d — never O(d²); the cut
  GROWS with d at fixed r.
- Identity-at-init (ReZero scale = 0 ⟹ the module is exactly identity).
- The forward == the assembled operator ``Σ_j U_j M_j P_j diag(mask_j)``
  applied densely; the learned partition is a REAL partition (each channel
  read by exactly one subspace).
- Assignment is hard argmax with a Lorentzian bounded-reciprocal STE backward —
  NON-softmax, and there is no token router anywhere (not MoE).
- **Anti-collapse gates (the note's "subspace orthogonality + effective
  rank"):** ``assignment_balance`` / differentiable ``assignment_balance_loss``
  (exactly 1 at total pile-up = the single-subspace LoRA degeneracy);
  ``subspace_overlap`` on the orthonormalized write spans (1 when identical);
  ``assembled_rank`` (≤ m·s, collapsing toward s under pile-up).
- Pointwise per token ⟹ passes the NM-11 softmax-twin detector and is
  NM-10-measurable.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.subspace_mixture_mix import (
    SubspaceMixtureMix,
    subspace_mixture_param_count,
)


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = SubspaceMixtureMix(dim=32, n_subspaces=4, subspace_dim=4)
    with torch.no_grad():
        mix.scale.fill_(0.5)  # open the path
    x = torch.randn(4, 10, 32)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d,m,s", [(16, 2, 2), (32, 4, 4), (64, 4, 8)])
def test_identity_at_init(d: int, m: int, s: int) -> None:
    """ReZero ``scale=0`` ⟹ the module is exactly the identity."""
    mix = SubspaceMixtureMix(dim=d, n_subspaces=m, subspace_dim=s)
    x = torch.randn(3, 7, d)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-7)


@pytest.mark.parametrize("d,m,s", [(16, 2, 2), (32, 4, 4), (64, 4, 8)])
def test_param_count_matches_helper_and_numel(d: int, m: int, s: int) -> None:
    mix = SubspaceMixtureMix(dim=d, n_subspaces=m, subspace_dim=s)
    assert mix.num_parameters == subspace_mixture_param_count(d, m, s)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters
    assert mix.num_parameters == d * m + m * s * d + m * s * s + m * d * s + 1


def test_params_compact_and_cut_grows_with_dim() -> None:
    """The compaction claim: O(r·d) beats the dense d² and the advantage GROWS
    with d at fixed total subspace width r = m·s."""
    m, s = 4, 4
    small = SubspaceMixtureMix(dim=64, n_subspaces=m, subspace_dim=s)
    large = SubspaceMixtureMix(dim=256, n_subspaces=m, subspace_dim=s)
    assert small.num_parameters < 64 * 64
    assert large.num_parameters < 256 * 256
    cut_small = (64 * 64) / small.num_parameters
    cut_large = (256 * 256) / large.num_parameters
    assert cut_large > cut_small  # the O(r·d) vs O(d²) scaling gap widens


def test_forward_matches_assembled_operator() -> None:
    """The masked down→mix→up forward == the assembled dense operator:
    ``out = x + scale · (W x)`` exactly."""
    torch.manual_seed(0)
    mix = SubspaceMixtureMix(dim=16, n_subspaces=2, subspace_dim=3)
    with torch.no_grad():
        mix.scale.fill_(0.7)
    x = torch.randn(3, 6, 16)
    w = mix.assemble_operator()
    expected = x + 0.7 * torch.einsum("de,...e->...d", w, x)
    assert torch.allclose(mix(x), expected, atol=1e-5)


def test_partition_is_exclusive() -> None:
    """The learned grouping is a real PARTITION: every channel is read by
    exactly one subspace (hard one-hot rows), and a subspace's masked
    down-projection is zero on channels it does not own."""
    torch.manual_seed(0)
    mix = SubspaceMixtureMix(dim=16, n_subspaces=4, subspace_dim=2)
    hard = mix._ste_assignment().detach()
    assert torch.allclose(hard.sum(dim=-1), torch.ones(mix.dim))
    assert ((hard == 0) | (hard == 1)).all()
    owner = mix.assignment()
    masked_down = (mix.down * hard.t().unsqueeze(1)).detach()
    for c in range(mix.dim):
        for j in range(mix.n_subspaces):
            if j != int(owner[c]):
                assert (masked_down[j, :, c] == 0).all()


def test_balance_gates_detect_single_subspace_pileup() -> None:
    """The collapse mode for this lane: all channels piled onto ONE subspace ⟹
    the operator degenerates to a single rank-s LoRA. ``assignment_balance``
    hits 0 (dead subspaces), the differentiable loss hits ≈1, and the assembled
    rank collapses to ≤ s; a balanced partition keeps rank near m·s."""
    torch.manual_seed(0)
    d, m, s = 64, 4, 4
    mix = SubspaceMixtureMix(dim=d, n_subspaces=m, subspace_dim=s)
    # Balanced: channels striped evenly across subspaces. (Gap 30: the
    # Lorentzian backward is polynomial-tailed by design — not softmax — so a
    # decisive soft assignment needs a wide logit gap.)
    with torch.no_grad():
        mix.assign_logits.zero_()
        for c in range(d):
            mix.assign_logits[c, c % m] = 30.0
    assert mix.assignment_balance() == 1.0
    assert float(mix.assignment_balance_loss().detach()) < 0.05
    rank_balanced = mix.assembled_rank()
    assert rank_balanced > s
    assert rank_balanced <= m * s
    # Pile-up: every channel on subspace 0.
    with torch.no_grad():
        mix.assign_logits.zero_()
        mix.assign_logits[:, 0] = 30.0
    assert mix.assignment_balance() == 0.0
    assert float(mix.assignment_balance_loss().detach()) > 0.9
    assert mix.assembled_rank() <= s
    assert mix.assembled_rank() < rank_balanced


def test_subspace_overlap_gate_is_load_bearing() -> None:
    """The note's orthogonality gate: independent random write spans overlap
    ≈ s/d; identical ``U_j`` (the recombination degeneracy) score 1."""
    torch.manual_seed(0)
    mix = SubspaceMixtureMix(dim=64, n_subspaces=4, subspace_dim=4)
    overlap_random = mix.subspace_overlap()
    assert overlap_random < 0.3  # random s-dim spans in R^d ⟹ ≈ s/d = 0.0625
    with torch.no_grad():
        mix.up.copy_(mix.up[0:1].expand_as(mix.up).clone())
    overlap_degen = mix.subspace_overlap()
    assert overlap_degen == pytest.approx(1.0, abs=1e-5)
    assert overlap_degen > overlap_random


def test_selection_is_not_softmax() -> None:
    """The soft assignment surrogate is the Lorentzian bounded-reciprocal form
    ``1/(1+gap²/γ)`` normalized — NOT exp(gap). Pins the non-softmax law."""
    mix = SubspaceMixtureMix(dim=16, n_subspaces=4, subspace_dim=2, lorentz_gamma=1.0)
    with torch.no_grad():
        mix.assign_logits.zero_()
        mix.assign_logits[0] = torch.tensor([3.0, 1.0, 0.0, 1.0])
    soft = mix._soft_assignment()[0]
    raw = torch.tensor([1.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 5.0])
    assert torch.allclose(soft, raw / raw.sum(), atol=1e-6)


def test_backward_reaches_assignment_bottlenecks_and_scale() -> None:
    """With the path open, gradient reaches the assignment logits (through the
    Lorentzian STE), all three bottleneck tensors, and the ReZero scale — and
    the balance loss injects gradient into the assignment (the anti-pile-up
    training signal)."""
    torch.manual_seed(0)
    mix = SubspaceMixtureMix(dim=16, n_subspaces=4, subspace_dim=2)
    with torch.no_grad():
        mix.scale.fill_(0.5)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = mix(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    loss = out.square().mean() + 0.1 * mix.assignment_balance_loss()
    loss.backward()
    for name in ("assign_logits", "down", "mix", "up", "scale"):
        p = getattr(mix, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the mixer is pointwise (channels partitioned
    and mixed, tokens never mixed ⟹ ``cross_token_mixing ≈ 0``) with no softmax
    anywhere — a distinct compaction mechanism, not attention in disguise."""
    mix = SubspaceMixtureMix(dim=32, n_subspaces=4, subspace_dim=4)
    with torch.no_grad():  # open the path for an active measurement
        mix.scale.fill_(0.5)
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mix)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f})"
    )
    assert props.cross_token_mixing < 0.1  # pointwise, not attention


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the mixer exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside the other NM-C primitives."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = SubspaceMixtureMix(dim=16, n_subspaces=2, subspace_dim=2)
    with torch.no_grad():  # open the path for a non-trivial fingerprint
        mix.scale.fill_(0.5)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"


def test_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        SubspaceMixtureMix(dim=0)
    with pytest.raises(ValueError):
        SubspaceMixtureMix(dim=16, n_subspaces=0)
    with pytest.raises(ValueError):
        SubspaceMixtureMix(dim=16, subspace_dim=0)
    with pytest.raises(ValueError):
        SubspaceMixtureMix(dim=16, n_subspaces=4, subspace_dim=4)  # m·s >= dim
    with pytest.raises(ValueError):
        SubspaceMixtureMix(dim=16, n_subspaces=2, subspace_dim=2, lorentz_gamma=0.0)
    with pytest.raises(ValueError):
        subspace_mixture_param_count(16, 4, 4)
