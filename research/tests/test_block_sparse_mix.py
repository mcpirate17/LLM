"""Tests for NM-C11 — native block-sparse mixer.

Pins the spec (compaction-lanes note, Lever 4):
- Sparsity IS the mechanism: the ONLY weight storage is ``n_blocks`` dense
  ``b×b`` blocks + their learned bipartite grid addresses — params/VRAM/FLOP
  ∝ ``n_blocks·b²`` ≪ d². DISTINCT from the repo's ``block_sparse_linear``
  (full dense ``d×d`` weight + magnitude mask = post-hoc pruning, the baseline).
- Identity-at-init (ReZero scale = 0 ⟹ the module is exactly identity).
- The forward's gather/mix/scatter is EXACTLY the assembled sparse weight
  applied densely (including blocks colliding on one address, which sum).
- Placement is hard argmax with a Lorentzian bounded-reciprocal STE backward —
  NON-softmax by construction (no exponential anywhere in selection).
- **Anti-pile-up gates:** ``placement_utilization`` reports unique addresses /
  n_blocks (1/n_blocks at total pile-up); ``placement_overlap_loss`` is
  differentiable, ≈1 when all blocks share one address and low when spread;
  ``assembled_rank`` collapses under pile-up.
- Pointwise per token ⟹ passes the NM-11 softmax-twin detector
  (``cross_token_mixing ≈ 0``) and is NM-10-measurable.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.block_sparse_mix import (
    BlockSparseMix,
    block_sparse_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = BlockSparseMix(dim=32, block_size=8, n_blocks=6)
    with torch.no_grad():
        mix.scale.fill_(0.5)  # open the path
    x = torch.randn(4, 10, 32)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d,b,k", [(8, 4, 2), (16, 4, 5), (32, 8, 8)])
def test_identity_at_init(d: int, b: int, k: int) -> None:
    """ReZero ``scale=0`` ⟹ the module is exactly the identity."""
    mix = BlockSparseMix(dim=d, block_size=b, n_blocks=k)
    x = torch.randn(3, 7, d)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-7)


@pytest.mark.parametrize("d,b,k", [(8, 4, 2), (16, 4, 5), (64, 8, 8)])
def test_param_count_matches_helper_and_numel(d: int, b: int, k: int) -> None:
    mix = BlockSparseMix(dim=d, block_size=b, n_blocks=k)
    assert mix.num_parameters == block_sparse_param_count(d, b, k)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters
    assert mix.num_parameters == k * b * b + 2 * k * (d // b) + 1


def test_params_compact_vs_dense_and_vs_pruning_baseline() -> None:
    """The compaction claim: storage ∝ K·b², NOT d². The repo's
    ``block_sparse_linear`` pruning baseline stores the FULL d² weight and only
    masks it — this mechanism must be strictly smaller than d² and shrink as K
    shrinks (density is a real storage knob, not a mask)."""
    d, b = 64, 8
    dense = d * d
    sparse8 = BlockSparseMix(dim=d, block_size=b, n_blocks=8)
    sparse4 = BlockSparseMix(dim=d, block_size=b, n_blocks=4)
    assert sparse8.num_parameters < dense
    assert dense / sparse8.num_parameters > 6  # 12.5% block density
    assert sparse4.num_parameters < sparse8.num_parameters  # K is a storage knob


def test_forward_matches_assembled_dense_weight() -> None:
    """The gather/mix/scatter forward == the assembled sparse weight applied
    densely: ``out = x + scale · (W x)`` exactly."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=16, block_size=4, n_blocks=5)
    with torch.no_grad():
        mix.scale.fill_(0.7)
    x = torch.randn(3, 6, 16)
    w = mix.assemble_weight()
    expected = x + 0.7 * torch.einsum("ij,...j->...i", w, x)
    assert torch.allclose(mix(x), expected, atol=1e-5)


def test_colliding_blocks_sum_consistently() -> None:
    """Blocks forced onto the SAME grid address sum — identically in the
    assembled weight and in the forward's scatter-add."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=16, block_size=4, n_blocks=3)
    with torch.no_grad():
        mix.scale.fill_(1.0)
        # Pin every block to grid cell (1, 2).
        mix.row_logits.zero_()
        mix.row_logits[:, 1] = 5.0
        mix.col_logits.zero_()
        mix.col_logits[:, 2] = 5.0
    w = mix.assemble_weight()
    summed = mix.block_values.sum(dim=0)
    assert torch.allclose(w[4:8, 8:12], summed, atol=1e-6)
    outside = torch.ones_like(w, dtype=torch.bool)
    outside[4:8, 8:12] = False
    assert (w.detach()[outside] == 0).all()  # everything else is exactly zero
    x = torch.randn(2, 5, 16)
    expected = x + torch.einsum("ij,...j->...i", w, x)
    assert torch.allclose(mix(x), expected, atol=1e-5)


def test_weight_is_structurally_sparse() -> None:
    """The assembled weight has nonzeros ONLY inside the K placed blocks —
    at most ``n_blocks·b²`` nonzero entries, on block-grid boundaries."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=32, block_size=8, n_blocks=3)
    w = mix.assemble_weight()
    nonzero = (w != 0).sum().item()
    assert nonzero <= mix.n_blocks * mix.block_size**2
    row_idx, col_idx = mix.placements()
    mask = torch.zeros_like(w, dtype=torch.bool)
    b = mix.block_size
    for k in range(mix.n_blocks):
        r, c = int(row_idx[k]) * b, int(col_idx[k]) * b
        mask[r : r + b, c : c + b] = True
    assert (w[~mask] == 0).all()


def test_selection_is_not_softmax() -> None:
    """The soft address surrogate is the Lorentzian bounded-reciprocal form:
    weight 1 at the argmax, ``1/(1+gap²/γ)`` elsewhere (pre-normalization) —
    NOT exp(gap). Pins the non-softmax selection law."""
    mix = BlockSparseMix(dim=16, block_size=4, n_blocks=1, lorentz_gamma=1.0)
    logits = torch.tensor([[3.0, 1.0, 0.0, 3.0 - 2.0]])
    soft = mix._soft_address(logits)
    raw = torch.tensor([[1.0, 1.0 / (1.0 + 4.0), 1.0 / (1.0 + 9.0), 1.0 / (1.0 + 4.0)]])
    assert torch.allclose(soft, raw / raw.sum(), atol=1e-6)


def test_placement_utilization_and_rank_detect_pileup() -> None:
    """The collapse mode for this lane: all blocks on ONE grid cell ⟹
    utilization = 1/K and the assembled rank collapses to ≤ b. Healthy spread
    placement keeps utilization high and rank well above one block."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=64, block_size=8, n_blocks=8)
    # Spread: pin each block to its own diagonal cell.
    with torch.no_grad():
        mix.row_logits.zero_()
        mix.col_logits.zero_()
        for k in range(mix.n_blocks):
            mix.row_logits[k, k] = 5.0
            mix.col_logits[k, k] = 5.0
    assert mix.placement_utilization() == 1.0
    rank_spread = mix.assembled_rank()
    assert rank_spread > mix.block_size
    # Pile-up: every block on cell (0, 0).
    with torch.no_grad():
        mix.row_logits.zero_()
        mix.row_logits[:, 0] = 5.0
        mix.col_logits.zero_()
        mix.col_logits[:, 0] = 5.0
    assert mix.placement_utilization() == pytest.approx(1.0 / mix.n_blocks)
    assert mix.assembled_rank() <= mix.block_size
    assert mix.assembled_rank() < rank_spread


def test_placement_overlap_loss_is_load_bearing() -> None:
    """The differentiable anti-pile-up guard: identical soft addresses score
    ≈1; blocks pinned to distinct cells score ≈0."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=64, block_size=8, n_blocks=6)
    with torch.no_grad():
        mix.row_logits.zero_()
        mix.col_logits.zero_()
        for k in range(mix.n_blocks):
            mix.row_logits[k, k] = 8.0
            mix.col_logits[k, k] = 8.0
    loss_spread = float(mix.placement_overlap_loss().detach())
    with torch.no_grad():
        mix.row_logits.zero_()
        mix.row_logits[:, 0] = 8.0
        mix.col_logits.zero_()
        mix.col_logits[:, 0] = 8.0
    loss_pileup = float(mix.placement_overlap_loss().detach())
    assert loss_pileup > 0.95
    assert loss_spread < 0.5
    assert loss_pileup > loss_spread


def test_backward_reaches_blocks_addresses_and_scale() -> None:
    """With the path open, gradient reaches the block values, BOTH address
    logit tables (through the Lorentzian STE), and the ReZero scale — and the
    overlap loss injects gradient into the addresses (the anti-pile-up
    training signal)."""
    torch.manual_seed(0)
    mix = BlockSparseMix(dim=16, block_size=4, n_blocks=4)
    with torch.no_grad():
        mix.scale.fill_(0.5)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = mix(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    loss = out.square().mean() + 0.1 * mix.placement_overlap_loss()
    loss.backward()
    for name in ("block_values", "row_logits", "col_logits", "scale"):
        p = getattr(mix, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the mixer is pointwise (channels mixed via the
    sparse weight, never tokens ⟹ ``cross_token_mixing ≈ 0``) and its selection
    has no softmax — a distinct compaction mechanism, not attention in
    disguise."""
    mix = BlockSparseMix(dim=32, block_size=8, n_blocks=8)
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
    mix = BlockSparseMix(dim=16, block_size=4, n_blocks=4)
    with torch.no_grad():  # open the path for a non-trivial fingerprint
        mix.scale.fill_(0.5)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"


def test_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        BlockSparseMix(dim=0)
    with pytest.raises(ValueError):
        BlockSparseMix(dim=8, block_size=0)
    with pytest.raises(ValueError):
        BlockSparseMix(dim=8, block_size=4, n_blocks=0)
    with pytest.raises(ValueError):
        BlockSparseMix(dim=10, block_size=4)  # dim not divisible by block_size
    with pytest.raises(ValueError):
        BlockSparseMix(dim=8, block_size=4, lorentz_gamma=0.0)
    with pytest.raises(ValueError):
        block_sparse_param_count(10, 4, 2)
