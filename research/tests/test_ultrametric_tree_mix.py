"""Tests for the ultrametric_tree_mix novel content-addressed sequence mixer (NM-5).

Covers: registry wiring, fwd/bwd finiteness, the causal look-back property, and the two structural
claims that make this NOT a softmax twin — (1) the product-over-scales kernel zeros a pair's mass on
a single scale disagreement (a property softmax-over-dot-product provably cannot match when additive
scores tie), and (2) the learned scale threshold controls selectivity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.synthesis.compiled_op_params import CompiledOpParamInitMixin
from research.synthesis.compiler_ops_routing import OP_IMPLS
from research.synthesis.primitives import get_primitive


class _Host(nn.Module, CompiledOpParamInitMixin):
    """Minimal param host exercising the real init method + forward together."""

    model_dim = 16


def _build(dim: int = 16) -> _Host:
    host = _Host()
    host._init_ultrametric_tree_mix(dim)
    return host


def _row_entropy(w: torch.Tensor) -> torch.Tensor:
    p = w.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def test_primitive_registered():
    op = get_primitive("ultrametric_tree_mix")
    assert op.name == "ultrametric_tree_mix"
    assert op.has_params
    assert op.binding_range_class == "full"
    assert "ultrametric_tree_mix" in OP_IMPLS


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16)
    out = OP_IMPLS["ultrametric_tree_mix"](host, [x], {})
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_grads_finite():
    torch.manual_seed(1)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = OP_IMPLS["ultrametric_tree_mix"](host, [x], {})
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name in (
        "ut_q_proj",
        "ut_k_proj",
        "ut_v_proj",
        "ut_o_proj",
        "ut_scale_dirs",
        "ut_scale_bias",
        "ut_scale_log_temp",
    ):
        p = getattr(host, name)
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_causal_mask():
    """Content-addressed look-back: query i can read only keys j <= i (future keys weight 0)."""
    torch.manual_seed(2)
    host = _build(dim=16)
    x = torch.randn(2, 6, 16)
    OP_IMPLS["ultrametric_tree_mix"](host, [x], {})
    w = host._last_ut_weights
    S = w.shape[-1]
    for i in range(S):
        for j in range(i + 1, S):
            assert w[:, i, j].abs().max().item() < 1e-9, (i, j)


def test_product_kernel_zeros_on_single_scale_disagreement():
    """The ultrametric kernel is a PRODUCT over scales: one scale disagreement drives pair mass to 0.

    Construct two keys with EQUAL additive score against the query (both dot products are 0 because
    the query is the zero vector). Softmax-over-dot-product must give them EQUAL mass; the ultrametric
    product still separates them by scale structure (the pair differing on scale 0 is killed). This is
    the structural property that makes the mixer not a softmax twin.
    """
    torch.manual_seed(3)
    host = _build(dim=16)
    D = 16
    with torch.no_grad():
        # Identity projections so q = k = x (full control over content codes).
        eye = torch.eye(D)
        host.ut_q_proj.copy_(eye)
        host.ut_k_proj.copy_(eye)
        host.ut_v_proj.copy_(eye)
        host.ut_o_proj.copy_(eye)
        # Scale l projects onto coordinate l (standard basis); strong agreement threshold.
        host.ut_scale_dirs.copy_(torch.eye(D)[:8])
        host.ut_scale_bias.fill_(10.0)
    # token0 = 0 (A); token1 = e0*100 (B: differs from A only on scale 0); token2 = 0 (query Q = A).
    x = torch.zeros(1, 3, D)
    x[0, 1, 0] = 100.0
    OP_IMPLS["ultrametric_tree_mix"](host, [x], {})
    w = host._last_ut_weights[0]  # (S, S)
    # Query 2 looks back: key0 (identical, full agreement) vs key1 (disagrees on scale 0).
    # Softmax(softmax([0,0,0])) would give key0 and key1 equal mass; the ultrametric product kills key1.
    assert w[2, 0].item() > 0.4, w[2, 0].item()
    assert w[2, 1].item() < 1e-6, w[2, 1].item()
    # Sanity: the two keys really do tie under a dot-product score (so softmax could not separate them).
    q = x[0, 2]
    assert torch.allclose((q @ x[0, 0]).abs(), torch.tensor(0.0))
    assert torch.allclose((q @ x[0, 1]).abs(), torch.tensor(0.0))


def test_scale_threshold_controls_selectivity():
    """A soft (large-positive) scale threshold -> broad, near-uniform retrieval (high row entropy);
    a hard (negative) threshold -> concentrated retrieval on near-zero-difference keys (low entropy)."""
    torch.manual_seed(4)
    x = torch.randn(2, 8, 16)
    soft = _build()
    hard = _build()
    with torch.no_grad():
        soft.ut_scale_bias.fill_(5.0)
        hard.ut_scale_bias.fill_(-5.0)
    OP_IMPLS["ultrametric_tree_mix"](soft, [x], {})
    OP_IMPLS["ultrametric_tree_mix"](hard, [x], {})
    soft_ent = _row_entropy(soft._last_ut_weights).mean().item()
    hard_ent = _row_entropy(hard._last_ut_weights).mean().item()
    assert soft_ent > hard_ent, (soft_ent, hard_ent)
