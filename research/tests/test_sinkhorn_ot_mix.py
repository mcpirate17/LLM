"""Tests for the sinkhorn_ot_mix novel optimal-transport mixer (NM-4).

Covers: registry wiring, fwd/bwd finiteness, the doubly-stochastic balanced-marginal property
that makes OT structurally distinct from softmax (row-stochastic) attention, the non-softmax-twin
column-uniformity assertion, and the learned-epsilon sharpness control.
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
    host._init_sinkhorn_ot_mix(dim)
    return host


def _plan_entropy(plan: torch.Tensor) -> torch.Tensor:
    p = plan.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=(1, 2))


def test_primitive_registered():
    op = get_primitive("sinkhorn_ot_mix")
    assert op.name == "sinkhorn_ot_mix"
    assert op.has_params
    assert op.binding_range_class == "full"
    assert "sinkhorn_ot_mix" in OP_IMPLS


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16)
    out = OP_IMPLS["sinkhorn_ot_mix"](host, [x], {"sinkhorn_iters": 20})
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_grads_finite():
    torch.manual_seed(1)
    host = _build(dim=16)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = OP_IMPLS["sinkhorn_ot_mix"](host, [x], {"sinkhorn_iters": 20})
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name in (
        "ot_q_proj",
        "ot_k_proj",
        "ot_v_proj",
        "ot_o_proj",
        "sinkhorn_log_eps",
    ):
        p = getattr(host, name)
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_plan_is_doubly_stochastic():
    """Balanced OT: every row AND every column sums to 1/S — the structural property
    softmax attention (row-stochastic only) cannot satisfy."""
    torch.manual_seed(2)
    host = _build(dim=16)
    x = torch.randn(3, 8, 16)
    OP_IMPLS["sinkhorn_ot_mix"](host, [x], {"sinkhorn_iters": 50})
    T = host._last_ot_plan
    s = T.shape[-1]
    target = 1.0 / s
    assert torch.allclose(
        T.sum(dim=2), torch.full_like(T.sum(dim=2), target), atol=1e-3
    )
    assert torch.allclose(
        T.sum(dim=1), torch.full_like(T.sum(dim=1), target), atol=1e-3
    )


def test_not_softmax_twin():
    """A softmax-over-keys twin concentrates mass unevenly across keys (non-uniform column
    sums). Balanced OT forces uniform column sums — the anti-collapse / novelty assertion."""
    torch.manual_seed(3)
    host = _build(dim=16)
    x = torch.randn(3, 8, 16)
    OP_IMPLS["sinkhorn_ot_mix"](host, [x], {"sinkhorn_iters": 50})
    col_sums = host._last_ot_plan.sum(dim=1)
    assert col_sums.std(dim=-1).max().item() < 1e-3


def test_learned_eps_controls_sharpness():
    """Small eps -> sharp (low-entropy) plan; large eps -> near-uniform (high-entropy) plan."""
    torch.manual_seed(4)
    x = torch.randn(2, 8, 16)
    sharp = _build()
    soft = _build()
    with torch.no_grad():
        sharp.sinkhorn_log_eps.fill_(-6.0)
        soft.sinkhorn_log_eps.fill_(6.0)
    OP_IMPLS["sinkhorn_ot_mix"](sharp, [x], {"sinkhorn_iters": 50})
    OP_IMPLS["sinkhorn_ot_mix"](soft, [x], {"sinkhorn_iters": 50})
    assert torch.all(
        _plan_entropy(sharp._last_ot_plan) < _plan_entropy(soft._last_ot_plan)
    )
