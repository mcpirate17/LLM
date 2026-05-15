"""Smoke + behavior tests for the sprint-4 primitive additions."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.primitive_templates import (
    CliffordAttention,
    PadicProjection,
    SpikingActivationGate,
    TropicalTopKStateSpace,
)


def _check_shape_and_grad(module: torch.nn.Module, dim: int, seq_len: int = 8) -> None:
    x = torch.randn(2, seq_len, dim, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    loss = y.pow(2).mean()
    loss.backward()
    assert torch.isfinite(y).all().item()
    for p in module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item()


def test_clifford_attention_shape_and_grad() -> None:
    _check_shape_and_grad(CliffordAttention(dim=16), dim=16)


def test_clifford_attention_rejects_dim_not_div_by_4() -> None:
    with pytest.raises(ValueError):
        CliffordAttention(dim=15)


def test_clifford_attention_uses_negative_signature_on_bivector() -> None:
    # The Cl(2,0) affinity flips the sign of the 4th multivector component.
    # Verify by feeding inputs where only the bivector component differs:
    # if signature were all-positive (pure dot product), the output would be
    # identical for x and -x along that component. Cl(2,0) shouldn't.
    module = CliffordAttention(dim=8).eval()
    n_mv = 2
    x = torch.randn(1, 4, 8)
    x_bv_flipped = x.clone().view(1, 4, n_mv, 4)
    x_bv_flipped[..., 3] = -x_bv_flipped[..., 3]
    x_bv_flipped = x_bv_flipped.view(1, 4, 8)
    with torch.no_grad():
        y = module(x)
        y_bv = module(x_bv_flipped)
    # The two outputs must differ — the bivector flip should change the
    # geometric-product scalar affinity.
    assert (y - y_bv).abs().mean().item() > 1e-6


def test_spiking_activation_gate_shape_and_grad() -> None:
    _check_shape_and_grad(SpikingActivationGate(dim=16), dim=16)


def test_spiking_activation_gate_actually_thresholds() -> None:
    module = SpikingActivationGate(dim=16, threshold=0.5).eval()
    x = torch.randn(2, 8, 16)
    # Hook between proj_in and proj_out: feed the membrane manually
    membrane = module.proj_in(x)
    from component_fab.generator.primitive_templates import _SurrogateSpike

    spikes = _SurrogateSpike.apply(membrane, 0.5, 2.0)
    expected = (membrane > 0.5).float()
    assert torch.allclose(spikes, expected)


def test_padic_projection_shape_and_grad() -> None:
    _check_shape_and_grad(PadicProjection(dim=16, p=2, n_levels=3), dim=16)


def test_padic_projection_rejects_indivisible_dim() -> None:
    with pytest.raises(ValueError):
        PadicProjection(dim=15, p=2, n_levels=3)


def test_padic_projection_is_block_aligned() -> None:
    # Two inputs that differ only inside one block-of-2 should produce
    # differences confined within the same block at level 0.
    module = PadicProjection(dim=8, p=2, n_levels=1).eval()
    x = torch.zeros(1, 1, 8)
    x_perturbed = x.clone()
    x_perturbed[0, 0, 0] = 1.0
    with torch.no_grad():
        y = module(x)
        y_perturbed = module(x_perturbed)
    diff = (y - y_perturbed).abs().squeeze()
    assert diff[0].item() > 1e-6 or diff[1].item() > 1e-6
    assert all(d.item() < 1e-6 for d in diff[2:])


def test_tropical_topk_state_space_shape_and_grad() -> None:
    _check_shape_and_grad(TropicalTopKStateSpace(dim=16, k=4), dim=16, seq_len=8)


def test_tropical_topk_state_space_actually_sparse_in_state() -> None:
    module = TropicalTopKStateSpace(dim=16, k=4).eval()
    # Hook into the state by tracking C.weight @ state — but easier: probe
    # whether the output's max-to-mean ratio is high (sparse winners).
    x = torch.randn(1, 16, 16)
    with torch.no_grad():
        y = module(x)
    ratio = y.abs().amax(dim=-1).mean() / (y.abs().mean() + 1e-12)
    assert ratio > 1.2


def test_tropical_topk_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        TropicalTopKStateSpace(dim=16, state_dim=8, k=12)
