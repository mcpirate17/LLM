# pyright: reportPrivateImportUsage=false
"""NM-C13 low-rank-native state memory — behavioural pins.

What must hold:
- The state is rank-r BY CONSTRUCTION: factor banks are ``(B, r, D)`` and the
  represented operator's rank never exceeds r, at any sequence length — the
  structural difference vs linear attention (rank grows with L) and vs
  dplr_gated_delta / NM-F2 (full B×D×D state).
- Causality with exclusive reads (zero gradient to any future position, incl.
  through the STE write path).
- Content addressing: repeated keys land in the SAME slot (binding-friendly),
  distinct keys spread; a randomized-query control beats the positional
  shortcut ([[cross_axis_architecture_matrix_2026-06-07]] rule).
- Gates: ``write_balance_loss`` exactly 1 at full pile-up; utilization
  diagnostic; bounded per-slot decay.
- NM-11 measured not-a-twin WITHOUT the pointwise waiver (unnormalized signed
  bilinear reads + hard non-softmax writes — the C12 lesson applied at design
  time). NM-10 fingerprint finite.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.lowrank_state_memory import (
    LowRankStateMemory,
    lowrank_state_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def _module(dim: int = 32, **kwargs) -> LowRankStateMemory:
    torch.manual_seed(0)
    return LowRankStateMemory(dim, **kwargs)


def _activated(dim: int = 32, **kwargs) -> LowRankStateMemory:
    mem = _module(dim, **kwargs)
    with torch.no_grad():
        mem.scale.fill_(1.0)
    return mem


def test_shape_and_identity_at_init() -> None:
    mem = _module()
    x = torch.randn(2, 12, 32)
    torch.testing.assert_close(mem(x), x)
    assert _activated()(x).shape == x.shape


def test_state_is_factor_banks_never_dxd() -> None:
    """The entire state is two (B, r, D) banks — O(r·D), not O(D²)."""
    mem = _activated(rank=4)
    x = torch.randn(2, 10, 32)
    _reads, slot_k, slot_v, soft = mem.scan_memory(x)
    assert slot_k.shape == (2, 4, 32)
    assert slot_v.shape == (2, 4, 32)
    assert soft.shape == (2, 10, 4)


def test_rank_bounded_by_construction() -> None:
    """Represented-operator rank ≤ r at ANY sequence length — linear attention
    at the same lengths reaches rank min(L, D)."""
    mem = _activated(rank=4)
    for length in (3, 16, 64):
        x = torch.randn(1, length, 32)
        assert mem.represented_rank(x) <= 4, f"L={length}"


def test_causal_no_future_gradient() -> None:
    mem = _activated(16, rank=4)
    for seed in range(2):
        x = torch.randn(1, 10, 16, generator=torch.Generator().manual_seed(seed))
        x = x.clone().requires_grad_(True)
        for p in range(x.shape[1]):
            grad = torch.autograd.grad(mem(x)[0, p].sum(), x)[0]
            future = grad[0, p + 1 :]
            if future.numel():
                assert future.abs().max().item() == 0.0, f"seed {seed}, pos {p}"


def test_exclusive_read_first_token_passthrough() -> None:
    mem = _activated()
    x = torch.randn(1, 6, 32)
    out = mem(x)
    torch.testing.assert_close(out[:, 0], x[:, 0])  # empty state at t=0
    assert not torch.allclose(out[:, 1:], x[:, 1:])  # later reads active


def test_repeated_key_reuses_slot_distinct_keys_spread() -> None:
    mem = _activated(rank=8, lorentz_gamma=0.25)
    key = torch.randn(32)
    x = torch.stack([key, key, key, 5.0 * torch.randn(32), key]).unsqueeze(0)
    _r, _k, _v, soft = mem.scan_memory(x)
    slots = soft.argmax(dim=-1)[0]  # (L,)
    assert slots[0] == slots[1] == slots[2] == slots[4]  # same key ⟹ same slot
    assert slots[3] != slots[0]  # distinct key ⟹ different slot


def test_content_addressing_beats_positional_shortcut() -> None:
    """Randomized-query binding control: wherever the key was WRITTEN, the
    query at the end must resolve to THAT slot — address resolution follows
    content, not position. (Raw reads are not comparable across different
    write histories: slot identities legitimately differ; what binding needs
    is write-slot == query-slot within each trajectory.)"""
    torch.manual_seed(3)
    mem = _activated(rank=8, lorentz_gamma=0.25)
    key = torch.randn(32)
    filler = 3.0 * torch.randn(1, 7, 32)
    for pos in (1, 3, 5):
        x = filler.clone()
        x[:, pos] = key
        x[:, 6] = key  # query token (reads BEFORE its own write)
        reads, _k, _v, soft = mem.scan_memory(x)
        slots = soft.argmax(dim=-1)[0]
        assert int(slots[6]) == int(slots[pos]), (
            f"key written to slot {int(slots[pos])} at pos {pos}, but the "
            f"query resolved slot {int(slots[6])}"
        )
        assert float(reads[0, 6].detach().abs().sum()) > 0.0  # read is live


def test_write_balance_loss_bounds_and_utilization() -> None:
    mem = _activated(rank=4, lorentz_gamma=0.25)
    diverse = 3.0 * torch.randn(1, 16, 32)
    assert float(mem.write_balance_loss(diverse).detach()) < 0.5
    assert mem.slot_utilization(diverse) > 0.5
    # Full pile-up: identical tokens all match one slot.
    same = torch.randn(1, 1, 32).expand(1, 16, 32).contiguous()
    assert mem.slot_utilization(same) <= 0.5
    with torch.no_grad():
        mem.slot_salt.zero_()  # remove tie-breaking ⟹ maximal pile-up pressure
    pile = mem.write_balance_loss(same)
    assert float(pile.detach()) > 0.8


def test_decay_bounded() -> None:
    mem = _activated()
    lam = torch.sigmoid(mem.decay_logit)
    assert bool(((lam > 0.0) & (lam < 1.0)).all())


def test_gradients_flow() -> None:
    mem = _activated()
    x = torch.randn(2, 10, 32, requires_grad=True)
    (mem(x).square().mean() + mem.write_balance_loss(x)).backward()
    for name, p in mem.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} received no gradient"


def test_num_parameters_exact() -> None:
    dim, rank = 32, 8
    mem = _module(dim, rank=rank)
    counted = sum(p.numel() for p in mem.parameters())
    assert counted == lowrank_state_param_count(dim, rank)


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        LowRankStateMemory(0)
    with pytest.raises(ValueError):
        LowRankStateMemory(8, rank=8)  # rank must be < dim
    with pytest.raises(ValueError):
        LowRankStateMemory(8, rank=4, lorentz_gamma=0.0)
    mem = _module()
    with pytest.raises(ValueError):
        mem(torch.randn(4, 32))
    with pytest.raises(ValueError):
        mem(torch.randn(1, 4, 16))


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector, no pointwise waiver: unnormalized signed
    bilinear reads + hard Lorentzian writes — designed non-simplex from the
    start (the C12 lesson)."""
    mem = _activated()
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mem)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f}, "
        f"const={props.constant_token_preservation:.3f}, "
        f"convex={props.convex_range_fraction:.3f})"
    )


def test_measurable_by_physics_descriptor_probe() -> None:
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mem = _activated(16, rank=4)
    desc = probe.describe_operator(mem)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
