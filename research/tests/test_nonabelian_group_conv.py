"""NM-F7 nonabelian group conv — mechanism contract.

Pins the algebraic structure that makes the op non-QKV and falsifiable: the fixed
buffer is a genuine left-regular representation of a nonabelian dihedral group
(``P[i]@P[j] == P[i*j]``), it is noncommutative, and the scrambled control breaks
exactly that closure. Plus the op-level contract shared by every NM-F mixer:
shape-preserving, identity-at-init, causal, fail-fast, grad to all params,
measurable by the physics descriptor probe.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.nonabelian_group_conv import (
    NonabelianGroupConv,
    _dihedral_mult_table,
    _regular_representation,
    _scrambled_representation,
)

_DIM = 32


@pytest.mark.parametrize("order", [6, 8, 10, 12])
def test_regular_representation_is_a_group_homomorphism(order: int) -> None:
    """P(g_i) P(g_j) = P(g_i g_j) — the exact group law, and each P is a
    permutation (doubly stochastic). This is the structure a scrambled control
    destroys; it is the whole reason the op is more than a random linear mix."""
    reps = _regular_representation(order)
    table = _dihedral_mult_table(order)
    for i in range(order):
        assert torch.allclose(reps[i].sum(0), torch.ones(order))
        assert torch.allclose(reps[i].sum(1), torch.ones(order))
        for j in range(order):
            assert torch.allclose(reps[i] @ reps[j], reps[table[i][j]])


def test_group_is_nonabelian() -> None:
    """Some pair must not commute — otherwise the ordered-product claim (and the
    whole mechanism) is vacuous."""
    reps = _regular_representation(8)
    assert any(
        not torch.allclose(reps[i] @ reps[j], reps[j] @ reps[i])
        for i in range(8)
        for j in range(8)
    )


def test_scramble_destroys_the_group_structure() -> None:
    """The falsification control: same shapes, identity kept at index 0 (so
    identity-at-init survives), but closure is broken."""
    scr = _scrambled_representation(8, seed=0)
    table = _dihedral_mult_table(8)
    assert torch.allclose(scr[0], torch.eye(8))
    for i in range(8):  # still permutations
        assert torch.allclose(scr[i].sum(0), torch.ones(8))
    broke = any(
        not torch.allclose(scr[i] @ scr[j], scr[table[i][j]])
        for i in range(8)
        for j in range(8)
    )
    assert broke, "scrambled representation unexpectedly stayed a homomorphism"


@pytest.mark.parametrize("dim", [1, 2, 8, 16, 33, 64])
def test_shape_and_finiteness(dim: int) -> None:
    op = NonabelianGroupConv(dim, group_order=8)
    x = torch.randn(2, 12, dim)
    y = op(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("dim", [1, 2, 8, 16, 33, 64])
def test_identity_at_init(dim: int) -> None:
    op = NonabelianGroupConv(dim, group_order=8)
    x = torch.randn(3, 10, dim)
    assert torch.allclose(op(x), x, atol=1e-6)


def test_causal() -> None:
    op = NonabelianGroupConv(48, group_order=8).eval()
    x = torch.randn(1, 20, 48)
    y1 = op(x)
    x2 = x.clone()
    x2[:, 12:] = torch.randn(1, 8, 48)  # perturb the future only
    y2 = op(x2)
    assert torch.allclose(y1[:, :12], y2[:, :12], atol=1e-5)


def test_order_sensitive() -> None:
    """After training away from the identity init, reordering tokens must change
    the output — the op mixes across the sequence, and does so noncommutatively."""
    op = NonabelianGroupConv(16, group_order=8)
    torch.nn.init.normal_(op.select.weight, std=1.0)
    torch.nn.init.normal_(op.readout.weight, std=0.5)
    x = torch.randn(1, 6, 16)
    y = op(x)
    y_rev = op(x.flip(1))
    assert not torch.allclose(y, y_rev.flip(1), atol=1e-4)


def test_param_count_matches_formula() -> None:
    op = NonabelianGroupConv(_DIM, group_order=8, state_width=4)
    actual = sum(p.numel() for p in op.parameters())
    assert actual == op.num_parameters
    assert actual == 72 * _DIM + 8  # matches the registry param_formula


@pytest.mark.parametrize(
    "kwargs", [{"group_order": 7}, {"group_order": 4}, {"state_width": 0}]
)
def test_invalid_configs_fail_fast(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        NonabelianGroupConv(64, **kwargs)


def test_backward_flows_to_all_parameters() -> None:
    op = NonabelianGroupConv(_DIM, group_order=8)
    torch.nn.init.normal_(op.readout.weight, std=0.1)  # break the zero-init no-op
    x = torch.randn(2, 8, _DIM, requires_grad=True)
    op(x).pow(2).mean().backward()
    assert x.grad is not None
    for name, p in op.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_measurable_by_physics_descriptor_probe() -> None:
    from research.synthesis.physics_descriptors import PhysicsDescriptorProbe

    op = NonabelianGroupConv(_DIM, group_order=8)
    torch.nn.init.normal_(op.select.weight, std=0.5)
    torch.nn.init.normal_(op.readout.weight, std=0.3)
    desc = PhysicsDescriptorProbe(dim=_DIM).describe_operator(op)
    assert all(isinstance(v, float) and v == v for v in desc.values())
