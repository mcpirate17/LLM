"""Tests for NM-C9 — hypernetwork-generated per-layer weights.

Pins the spec (compaction-lanes note, Lever 3):
- Collapses ``n_layers``: a tiny SHARED hypernet generates ``W_{l,ρ}`` from
  composed (layer, role, chunk) embeddings — NO free per-layer d²-scale
  storage; the generator is CONSTANT in depth (adding a layer costs
  ``embed_dim + 1`` params, not ``n_roles·d²``), so the cut GROWS with depth.
- Identity-at-init (per-layer ReZero scales = 0 ⟹ the whole stack is identity).
- Role+layer conditioning is load-bearing: distinct layers AND distinct roles
  get DISTINCT generated weights (the anti-ALBERT structural property).
- **Capacity gate (the spec's risk):** ``generated_rank`` /
  ``min_rank_fraction`` report whether generated weights span the required
  rank — full at healthy init, collapsing when the generator's output
  projection degenerates.
- **Anti-tying guard:** ``layer_tying_loss`` is exactly 1 when the hypernet
  ignores the layer embedding (silent ALBERT weight tying) and well below at
  healthy layer-conditioned init.
- Pointwise per token ⟹ passes the NM-11 softmax-twin detector
  (``cross_token_mixing ≈ 0``) and is NM-10-measurable.
- DISTINCT from NM-C8 (explicit basis bank + free linear coefficients) and
  NM-C7 (one ``W`` applied ``k`` times).
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.hypernet_layer_mix import (
    HyperLayerMix,
    hyper_layer_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = HyperLayerMix(dim=32, n_layers=4)
    with torch.no_grad():
        mix.layer_scales.fill_(0.5)  # open the path
    x = torch.randn(4, 10, 32)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d,nl,nr", [(8, 4, 1), (16, 6, 2), (32, 8, 3)])
def test_identity_at_init(d: int, nl: int, nr: int) -> None:
    """ReZero ``layer_scales=0`` ⟹ every layer is x ← x + 0 ⟹ stack = identity."""
    mix = HyperLayerMix(dim=d, n_layers=nl, n_roles=nr)
    x = torch.randn(3, 7, d)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-7)


@pytest.mark.parametrize("d,nl,nr", [(8, 4, 1), (16, 5, 2)])
def test_param_count_matches_helper_and_numel(d: int, nl: int, nr: int) -> None:
    mix = HyperLayerMix(dim=d, n_layers=nl, n_roles=nr)
    assert mix.num_parameters == hyper_layer_param_count(d, nl, n_roles=nr)
    assert sum(p.numel() for p in mix.parameters()) == mix.num_parameters


@pytest.mark.parametrize("nl", [12, 16, 24])
def test_params_compact_vs_independent_layers(nl: int) -> None:
    """The compaction claim: tiny shared generator ≪ ``n_layers·n_roles`` independent
    ``d×d`` weights (the collapse-n_layers lever)."""
    d = 64
    mix = HyperLayerMix(dim=d, n_layers=nl)
    baseline = nl * mix.n_roles * d * d  # independent dense weights per site
    assert mix.num_parameters < baseline
    assert baseline / mix.num_parameters > 10  # an order-of-magnitude cut


def test_param_cut_grows_with_depth() -> None:
    """The generator is CONSTANT in depth: adding a layer costs only
    ``embed_dim + 1`` params (one conditioning row + one ReZero scale) — NOT
    another ``n_roles·d²``. So the cut vs independent layers widens with depth."""
    d = 64
    shallow = HyperLayerMix(dim=d, n_layers=4)
    deep = HyperLayerMix(dim=d, n_layers=32)
    delta = deep.num_parameters - shallow.num_parameters
    assert delta == (32 - 4) * (shallow.embed_dim + 1)
    shallow_cut = (4 * shallow.n_roles * d * d) / shallow.num_parameters
    deep_cut = (32 * deep.n_roles * d * d) / deep.num_parameters
    assert deep_cut > shallow_cut  # compaction improves with depth


def test_no_per_layer_dsquared_storage() -> None:
    """The structural difference from NM-C8: NO parameter tensor scales with
    ``n_layers·d²`` — the largest tensor is the chunk-shared output projection
    ``hidden_dim·(d/n_chunks)·d``, independent of depth."""
    d = 64
    mix = HyperLayerMix(dim=d, n_layers=16)
    largest = max(p.numel() for p in mix.parameters())
    assert largest == mix.w_out.numel()
    assert largest == mix.hidden_dim * (d // mix.n_chunks) * d
    assert largest <= d * d  # at defaults: hidden_dim=8 == n_chunks=8 ⟹ exactly d²


def test_materialize_weight_matches_vectorized_and_reference() -> None:
    """Per-site materialization == the vectorized pass == an explicit per-chunk
    reference computation (guards the chunk-stacking reshape)."""
    mix = HyperLayerMix(dim=16, n_layers=3, n_roles=2, n_chunks=4)
    all_w = mix.materialize_weights()
    assert all_w.shape == (3, 2, 16, 16)
    for layer in range(mix.n_layers):
        for role in range(mix.n_roles):
            w = mix.materialize_weight(layer, role)
            assert torch.allclose(all_w[layer, role], w, atol=1e-6)
            chunks = []
            for c in range(mix.n_chunks):
                z = mix.layer_embed[layer] + mix.role_embed[role] + mix.chunk_embed[c]
                h = torch.tanh(z @ mix.w_in + mix.b_in)
                rows = h @ mix.w_out + mix.b_out
                chunks.append(rows.reshape(mix.rows_per_chunk, mix.dim))
            ref = torch.cat(chunks, dim=0)
            assert torch.allclose(w, ref, atol=1e-6), f"site ({layer},{role}) mismatch"


def test_layer_and_role_conditioning_are_load_bearing() -> None:
    """The anti-ALBERT structural property: distinct (layer, role) sites get
    DISTINCT generated weights — the hypernet actually reads its conditioning."""
    torch.manual_seed(0)
    mix = HyperLayerMix(dim=16, n_layers=5, n_roles=2, n_chunks=4)
    w = mix.materialize_weights()
    flat = w.reshape(mix.n_layers * mix.n_roles, -1)
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            assert not torch.allclose(flat[i], flat[j], atol=1e-6)


def test_min_rank_fraction_gate_is_load_bearing() -> None:
    """The spec's hypernet-capacity gate: healthy init generates FULL-rank
    weights; degenerating the shared output projection to a rank-1 row pattern
    (every generated row ∝ one vector) collapses every ``W_{l,ρ}`` to rank 1 —
    and the gate reports it."""
    torch.manual_seed(0)
    mix = HyperLayerMix(dim=16, n_layers=4, n_chunks=4)
    assert mix.min_rank_fraction() == 1.0
    assert mix.generated_rank(0, 0) == mix.dim
    with torch.no_grad():
        v = torch.randn(mix.dim)
        gains = torch.randn(mix.hidden_dim, mix.rows_per_chunk)
        # w_out[k] reshaped is outer(gains[k], v): all generated rows ∝ v.
        mix.w_out.copy_((gains[:, :, None] * v[None, None, :]).reshape_as(mix.w_out))
        mix.b_out.zero_()
    assert mix.min_rank_fraction() == pytest.approx(1.0 / mix.dim)
    assert mix.generated_rank(0, 0) == 1


def test_layer_tying_loss_detects_albert_degeneracy() -> None:
    """The anti-tying guard: zeroing the layer embedding makes the hypernet
    ignore the layer ⟹ every layer emits IDENTICAL weights (silent ALBERT
    tying) ⟹ loss = 1 exactly. Healthy layer-conditioned init sits well below."""
    torch.manual_seed(0)
    mix = HyperLayerMix(dim=16, n_layers=6, n_chunks=4)
    loss_healthy = float(mix.layer_tying_loss().detach())
    assert loss_healthy < 0.9  # compositional conditioning ⟹ moderate floor, not 1
    with torch.no_grad():
        mix.layer_embed.zero_()  # generator can no longer distinguish layers
    loss_tied = float(mix.layer_tying_loss().detach())
    assert loss_tied == pytest.approx(1.0, abs=1e-5)
    assert loss_tied > loss_healthy


def test_backward_reaches_embeddings_generator_and_scales() -> None:
    """With the path open, gradient reaches ALL conditioning embeddings, the
    shared generator MLP, and the ReZero scales — and the tying loss injects
    gradient into the layer embedding (the anti-collapse training signal)."""
    mix = HyperLayerMix(dim=16, n_layers=4, n_chunks=4)
    with torch.no_grad():
        mix.layer_scales.fill_(0.5)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = mix(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    loss = out.square().mean() + 0.1 * mix.layer_tying_loss()
    loss.backward()
    for name in (
        "layer_embed",
        "role_embed",
        "chunk_embed",
        "w_in",
        "w_out",
        "layer_scales",
    ):
        p = getattr(mix, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: the mixer is pointwise (each token mixed only by
    its layer's generated weights ⟹ ``cross_token_mixing ≈ 0``), never a softmax
    over tokens — a distinct compaction mechanism, not attention in disguise."""
    mix = HyperLayerMix(dim=32, n_layers=4)
    with torch.no_grad():  # open the path for an active measurement
        mix.layer_scales.fill_(0.5)
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
    mix = HyperLayerMix(dim=16, n_layers=4, n_chunks=4)
    with torch.no_grad():  # open the path for a non-trivial fingerprint
        mix.layer_scales.fill_(0.5)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"


def test_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        HyperLayerMix(dim=0)
    with pytest.raises(ValueError):
        HyperLayerMix(dim=8, n_layers=0)
    with pytest.raises(ValueError):
        HyperLayerMix(dim=8, n_roles=0)
    with pytest.raises(ValueError):
        HyperLayerMix(dim=10, n_chunks=4)  # dim not divisible by n_chunks
    with pytest.raises(ValueError):
        hyper_layer_param_count(8, 0)
    mix = HyperLayerMix(dim=8, n_layers=3, n_chunks=4)
    with pytest.raises(IndexError):
        mix.materialize_weight(3)
    with pytest.raises(IndexError):
        mix.materialize_weight(0, mix.n_roles)
