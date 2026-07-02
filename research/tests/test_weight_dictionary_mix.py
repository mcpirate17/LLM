"""Tests for NM-C8 — shared weight-dictionary virtual-depth mixer.

Pins the spec:
- Collapses ``n_layers``: a SHARED bank of ``n_basis`` basis-matrices is read by
  every layer; each layer MATERIALIZES ``W_l = Σ_b c_{l,b} M_b``. Params =
  ``n_basis·d² + n_layers·n_basis + n_layers`` ≪ the ``n_layers·d²`` of
  independent layers (the compaction claim — cut grows with depth).
- Identity-at-init (per-layer ReZero scales = 0 ⟹ the whole stack is identity).
- Compositionality: ``materialize_weight(l) == Σ_b c_{l,b} M_b`` exactly.
- Per-layer weights are DISTINCT (the anti-ALBERT structural property — distinct
  materialized weights per layer, not one shared matrix).
- **Anti-collapse guard (the novel hook vs ALBERT):** ``basis_diversity_loss``
  is low for an independent basis and ≈1 for a degenerate (all-equal) bank;
  ``coeff_rank`` reports whether the per-layer coefficients actually span the
  basis (rank 1 ⟹ the weight-tying degeneracy).
- Pointwise per token ⟹ passes the NM-11 softmax-twin detector
  (``cross_token_mixing ≈ 0``) and is NM-10-measurable.
- DISTINCT from NM-C7 (one ``W`` applied ``k`` times) and from codex's
  ``shared_basis_proj`` (rank-k vector basis for ONE projection).
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.weight_dictionary_mix import (
    WeightDictionaryMix,
    weight_dict_param_count,
)


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = WeightDictionaryMix(dim=32, n_layers=4, n_basis=2)
    with torch.no_grad():
        mix.layer_scales.fill_(0.5)  # open the path
    x = torch.randn(4, 10, 32)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d,nl,nb", [(8, 4, 2), (16, 6, 3), (32, 8, 2)])
def test_identity_at_init(d: int, nl: int, nb: int) -> None:
    """ReZero ``layer_scales=0`` ⟹ every layer is x ← x + 0 ⟹ stack = identity."""
    mix = WeightDictionaryMix(dim=d, n_layers=nl, n_basis=nb)
    x = torch.randn(3, 7, d)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-7)


@pytest.mark.parametrize("d,nl,nb", [(8, 4, 2), (16, 5, 3)])
def test_param_count_matches_helper_and_numel(d: int, nl: int, nb: int) -> None:
    mix = WeightDictionaryMix(dim=d, n_layers=nl, n_basis=nb)
    assert mix.num_parameters == weight_dict_param_count(d, nl, nb)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters
    assert mix.num_parameters == nb * d * d + nl * nb + nl


@pytest.mark.parametrize("nl,nb", [(12, 2), (16, 3), (24, 4)])
def test_params_compact_vs_independent_layers(nl: int, nb: int) -> None:
    """The compaction claim: shared bank ≪ ``n_layers`` independent ``d×d`` layers,
    and the cut GROWS with depth (the collapse-n_layers lever)."""
    d = 64
    mix = WeightDictionaryMix(dim=d, n_layers=nl, n_basis=nb)
    baseline = nl * d * d  # n_layers independent dense layers
    assert mix.num_parameters < baseline
    # cut ratio = baseline / ours grows as n_layers grows (at fixed n_basis)
    ratio = baseline / mix.num_parameters
    assert ratio > 1.0


def test_param_cut_grows_with_depth() -> None:
    """Holding the basis fixed, adding layers costs only ``n_basis + 1`` params
    (a coefficient row + a ReZero scale) — NOT another ``d²`` matrix. So the cut
    vs independent layers widens with depth."""
    d, nb = 64, 2
    shallow = WeightDictionaryMix(dim=d, n_layers=4, n_basis=nb)
    deep = WeightDictionaryMix(dim=d, n_layers=32, n_basis=nb)
    shallow_cut = (4 * d * d) / shallow.num_parameters
    deep_cut = (32 * d * d) / deep.num_parameters
    assert deep_cut > shallow_cut  # compaction improves with depth


def test_materialize_weight_is_compositional() -> None:
    """``W_l == Σ_b c_{l,b} M_b`` exactly (verifies the einsum contraction over the
    shared basis index — guards the 'k/b' label bug that outer-products instead)."""
    mix = WeightDictionaryMix(dim=12, n_layers=3, n_basis=4)
    for layer in range(mix.n_layers):
        w = mix.materialize_weight(layer)
        ref = sum(mix.coeffs[layer, b] * mix.basis[b] for b in range(mix.n_basis))
        assert torch.allclose(w, ref, atol=1e-6), f"layer {layer} mismatch"
    # vectorized form matches the per-layer form
    all_w = mix.materialize_weights()
    assert all_w.shape == (mix.n_layers, mix.dim, mix.dim)
    for layer in range(mix.n_layers):
        assert torch.allclose(all_w[layer], mix.materialize_weight(layer), atol=1e-6)


def test_per_layer_weights_are_distinct() -> None:
    """The anti-ALBERT structural property: distinct layers materialize DISTINCT
    weights (a product ∏(I+α_l W_l) of different matrices, not a power of one)."""
    torch.manual_seed(0)
    mix = WeightDictionaryMix(dim=16, n_layers=5, n_basis=3)
    w = mix.materialize_weights()
    for i in range(mix.n_layers):
        for j in range(i + 1, mix.n_layers):
            assert not torch.allclose(w[i], w[j], atol=1e-6)


def test_basis_shared_across_layers_counted_once() -> None:
    """The param-sharing claim: ONE basis bank is read by every layer. The basis
    is a single (n_basis, d, d) parameter — its cost is counted exactly once
    regardless of n_layers."""
    d, nb = 32, 2
    mix4 = WeightDictionaryMix(dim=d, n_layers=4, n_basis=nb)
    mix32 = WeightDictionaryMix(dim=d, n_layers=32, n_basis=nb)
    assert tuple(mix4.basis.shape) == (nb, d, d)
    assert mix4.basis.numel() == mix32.basis.numel() == nb * d * d  # bank counted once
    # only the per-layer coefficient + scale rows grow with depth
    delta = mix32.num_parameters - mix4.num_parameters
    assert delta == (32 - 4) * (nb + 1)


def test_basis_diversity_loss_is_load_bearing() -> None:
    """The novel anti-collapse hook vs ALBERT: an independent basis scores ≈0; a
    degenerate bank (all basis matrices equal) scores ≈1. The fab adds this to
    the training loss so the dictionary cannot silently collapse to one matrix."""
    torch.manual_seed(0)
    mix = WeightDictionaryMix(dim=16, n_layers=4, n_basis=3)
    loss_random = float(mix.basis_diversity_loss().detach())
    assert loss_random < 0.05  # random independent basis ≈ uncorrelated
    # degenerate: all basis matrices identical (the ALBERT regime)
    with torch.no_grad():
        collapse = mix.basis[0:1].expand_as(mix.basis).clone()
        mix.basis.copy_(collapse)
    loss_degen = float(mix.basis_diversity_loss().detach())
    assert loss_degen > 0.9
    assert loss_degen > 10 * loss_random


def test_coeff_rank_reports_basis_usage() -> None:
    """Full-rank coefficients ⟹ layers use distinct basis combinations (healthy);
    rank 1 ⟹ all layers are scalar multiples of one basis vector (the weight-tying
    degeneracy the anti-collapse guard targets)."""
    mix = WeightDictionaryMix(dim=16, n_layers=4, n_basis=3)
    assert mix.coeff_rank() == 3  # random coeffs span the basis
    # force rank-1: every layer uses the same basis direction
    with torch.no_grad():
        mix.coeffs.copy_(torch.zeros_like(mix.coeffs))
        mix.coeffs[:, 0] = 1.0
    assert mix.coeff_rank() == 1


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the mixer is pointwise (each token mixed only by
    its own ``W_l`` ⟹ ``cross_token_mixing ≈ 0``), never a softmax over tokens.
    Confirmed not a softmax-attention twin — the structural guarantee that this
    is a distinct compaction mechanism, not attention in disguise."""
    mix = WeightDictionaryMix(dim=32, n_layers=4, n_basis=2)
    with torch.no_grad():  # open the path for an active measurement
        mix.layer_scales.fill_(0.5)
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mix)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f})"
    )
    assert props.cross_token_mixing < 0.1  # pointwise, not attention


def test_backward_reaches_basis_coeffs_and_scales() -> None:
    """With the path open, gradient reaches the shared basis bank, the per-layer
    coefficients, and the ReZero scales — and the diversity loss injects gradient
    into the basis (the anti-collapse training signal)."""
    mix = WeightDictionaryMix(dim=16, n_layers=4, n_basis=2)
    with torch.no_grad():
        mix.layer_scales.fill_(0.5)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = mix(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    loss = out.square().mean() + 0.1 * mix.basis_diversity_loss()
    loss.backward()
    for name in ("basis", "coeffs", "layer_scales"):
        p = getattr(mix, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the mixer exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside the other NM-C primitives."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = WeightDictionaryMix(dim=16, n_layers=4, n_basis=2)
    with torch.no_grad():  # open the path for a non-trivial fingerprint
        mix.layer_scales.fill_(0.5)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"


def test_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        WeightDictionaryMix(dim=0)
    with pytest.raises(ValueError):
        WeightDictionaryMix(dim=8, n_layers=0)
    with pytest.raises(ValueError):
        WeightDictionaryMix(dim=8, n_basis=0)
    with pytest.raises(ValueError):
        weight_dict_param_count(8, 0, 2)
    mix = WeightDictionaryMix(dim=8, n_layers=3, n_basis=2)
    with pytest.raises(IndexError):
        mix.materialize_weight(3)
