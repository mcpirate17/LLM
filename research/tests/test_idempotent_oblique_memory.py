# pyright: reportPrivateImportUsage=false
"""Tests for NM-F2 idempotent oblique-projection memory.

Pins the actual mechanism: Householder-generated orthonormal overwrite frames,
``P^2=P`` projector algebra, exact same-key overwrite instead of additive
blending, identity-at-init, gradient flow through the causal scan, O(rD) core
parameter accounting, and NM-10 measurability.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.idempotent_oblique_memory import (
    IdempotentObliqueMemory,
    householder_frame,
    idempotent_oblique_core_param_count,
    idempotent_oblique_param_count,
    idempotent_oblique_update,
    left_project,
    read_state,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mem = IdempotentObliqueMemory(dim=16, rank=4)
    x = torch.randn(2, 10, 16)
    y = mem(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d,rank", [(1, 1), (4, 1), (8, 3), (16, 8)])
def test_identity_at_init(d: int, rank: int) -> None:
    """Zero-init output lift makes the mixer an exact no-op drop-in."""
    mem = IdempotentObliqueMemory(dim=d, rank=rank)
    x = torch.randn(3, 6, d)
    assert torch.allclose(mem(x), x, atol=1e-6), f"dim={d}, rank={rank}"


def test_householder_frame_is_orthonormal_and_token_dependent() -> None:
    mem = IdempotentObliqueMemory(dim=8, rank=3)
    x = torch.zeros(2, 5, 8)
    x[1].fill_(5.0)
    frame = mem.overwrite_frame(x)
    gram = torch.einsum("...dr,...ds->...rs", frame, frame)
    eye = torch.eye(3).expand_as(gram)
    assert torch.allclose(gram, eye, atol=1e-5)
    assert not torch.allclose(frame[0], frame[1], atol=1e-4)


def test_projector_is_idempotent() -> None:
    torch.manual_seed(0)
    vectors = torch.randn(4, 2, 6)
    frame = householder_frame(vectors, dim=6, rank=2)
    matrix = torch.randn(4, 6, 6)
    once = left_project(matrix, frame)
    twice = left_project(once, frame)
    assert torch.allclose(twice, once, atol=1e-5)


def test_overwrite_update_replaces_selected_subspace_exactly() -> None:
    """For gate=1, Q^T S is replaced by Q^T(vk^T) and the orthogonal rows stay fixed."""
    state = torch.randn(1, 4, 4)
    frame = torch.eye(4, 2).unsqueeze(0)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[3.0, -2.0, 0.0, 0.0]])
    target = value.unsqueeze(-1) * key.unsqueeze(-2)

    updated = idempotent_oblique_update(state, key, value, frame)
    assert torch.allclose(
        left_project(updated, frame), left_project(target, frame), atol=1e-6
    )
    residual_before = state - left_project(state, frame)
    residual_after = updated - left_project(updated, frame)
    assert torch.allclose(residual_after, residual_before, atol=1e-6)

    again = idempotent_oblique_update(updated, key, value, frame)
    assert torch.allclose(again, updated, atol=1e-6)


def test_same_key_overwrite_is_not_an_additive_blend() -> None:
    """Full-rank F2 update returns the latest value for a key; additive memory would leak old value."""
    frame = torch.eye(4, 4).unsqueeze(0)
    state = torch.zeros(1, 4, 4)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    old_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    new_value = torch.tensor([[0.0, 0.0, 2.0, 0.0]])

    state = idempotent_oblique_update(state, key, old_value, frame)
    state = idempotent_oblique_update(state, key, new_value, frame)
    read = read_state(state, key)
    assert torch.allclose(read, new_value, atol=1e-6)

    additive = old_value.unsqueeze(-1) * key.unsqueeze(-2)
    additive = additive + new_value.unsqueeze(-1) * key.unsqueeze(-2)
    additive_read = read_state(additive, key)
    assert additive_read[0, 1] > 0.0
    assert not torch.allclose(additive_read, new_value, atol=1e-6)


def test_scan_is_causal_exclusive_by_default() -> None:
    mem = IdempotentObliqueMemory(dim=4, rank=4)
    with torch.no_grad():
        mem.gate_bias.fill_(20.0)
    x = torch.eye(4).unsqueeze(0)
    reads, states = mem.scan_memory(x)
    assert torch.allclose(reads[:, 0], torch.zeros_like(reads[:, 0]), atol=1e-6)
    assert states.shape == (1, 4, 4, 4)
    post_reads, _ = mem.scan_memory(x, read_before_write=False)
    assert post_reads[:, 0].abs().sum() > 0.0


def test_backward_flows_through_scan_and_controller() -> None:
    mem = IdempotentObliqueMemory(dim=8, rank=3)
    with torch.no_grad():
        mem.out_lift.weight.add_(0.3 * torch.randn_like(mem.out_lift.weight))
        mem.gate_bias.fill_(2.0)
    x = torch.randn(2, 7, 8, requires_grad=True)
    mem(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mem.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert mem.reflector_gain.grad.abs().sum() > 0
    assert mem.key_lift.weight.grad.abs().sum() > 0
    assert mem.value_lift.weight.grad.abs().sum() > 0


def test_param_counts_and_low_rank_core() -> None:
    d, rank = 32, 3
    mem = IdempotentObliqueMemory(dim=d, rank=rank)
    assert idempotent_oblique_core_param_count(d, rank) == rank * d + d + 1
    assert idempotent_oblique_param_count(d, rank) == rank * d + d + 1 + 4 * d * d
    assert mem.core_parameters == rank * d + d + 1
    assert mem.num_parameters == sum(p.numel() for p in mem.parameters())
    assert mem.core_parameters < d * d


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        IdempotentObliqueMemory(dim=0)
    with pytest.raises(ValueError):
        IdempotentObliqueMemory(dim=8, rank=0)
    with pytest.raises(ValueError):
        IdempotentObliqueMemory(dim=8, rank=9)


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint for geometric-novelty scoring."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mem = IdempotentObliqueMemory(dim=16, rank=4)
    with torch.no_grad():
        mem.out_lift.weight.add_(0.4 * torch.randn_like(mem.out_lift.weight))
        mem.gate_bias.fill_(1.0)
    desc = probe.describe_operator(mem)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
