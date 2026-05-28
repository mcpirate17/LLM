"""Tests for learnable-semiring attention (`mathspaces.semiring`).

Validates the core mathematical contract (β→0 == softmax attention; the fast
kernel matches the naive weighted-log-sum-exp definition; β-gradients are finite
including at β=0), plus end-to-end compile + backward through the registered op
and screening template.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from research.mathspaces.semiring import (
    semiring_attention,
    semiring_value_aggregate,
)

pytestmark = pytest.mark.unit


def _causal_logw(scores: torch.Tensor) -> torch.Tensor:
    s = scores.shape[-1]
    mask = torch.triu(torch.ones(scores.shape[-2], s, dtype=torch.bool), diagonal=1)
    return F.log_softmax(scores.masked_fill(mask, float("-inf")), dim=-1)


def test_beta_zero_equals_softmax_attention():
    """β→0 must recover standard causal softmax attention exactly."""
    torch.manual_seed(0)
    B, H, S, hd = 2, 3, 16, 8
    q, k, v = (torch.randn(B, H, S, hd) for _ in range(3))
    scale = hd**-0.5

    out = semiring_attention(q, k, v, torch.full((H,), 1e-6), scale)
    ref = _causal_logw((q @ k.transpose(-2, -1)) * scale).exp() @ v
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("beta_val", [-3.0, -0.5, 0.5, 2.0, 5.0])
def test_fast_kernel_matches_naive_definition(beta_val):
    """Fast (w @ exp(βv)) path equals the naive (1/β)·logsumexp_j(logw+βv) form."""
    torch.manual_seed(1)
    B, H, Sq, Sk, hd = 2, 3, 10, 10, 8
    v = torch.randn(B, H, Sk, hd)
    logw = _causal_logw(torch.randn(B, H, Sq, Sk))

    fast = semiring_value_aggregate(logw, v, torch.full((H,), beta_val))
    b = torch.full((H,), beta_val).view(1, -1, 1, 1)
    stacked = logw.unsqueeze(-1) + b.unsqueeze(-1) * v.unsqueeze(2)
    naive = torch.logsumexp(stacked, dim=3) / b
    assert torch.allclose(fast, naive, atol=1e-5)


def test_beta_gradient_finite_including_zero():
    """β must receive a finite gradient for every head, even a head sitting at β=0."""
    torch.manual_seed(0)
    B, H, S, hd = 2, 3, 12, 8
    q, k, v = (torch.randn(B, H, S, hd) for _ in range(3))
    beta = torch.tensor([-0.6, 0.0, 0.6], requires_grad=True)

    semiring_attention(q, k, v, beta, hd**-0.5).sum().backward()
    assert beta.grad is not None and torch.isfinite(beta.grad).all()


def test_large_beta_is_finite_and_stable():
    """Extreme β stays finite (clamp + log-sum-exp stabilisation)."""
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 2, 32, 8) for _ in range(3))
    out = semiring_attention(q, k, v, torch.full((2,), 1e3), 8**-0.5)
    assert torch.isfinite(out).all()


def test_op_registered_and_compiles_end_to_end():
    """The op is in the registry and compiles + backprops with a trainable β."""
    from research.synthesis.compiler import compile_model
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.primitives import PRIMITIVE_REGISTRY

    assert "learnable_semiring_attention" in PRIMITIVE_REGISTRY

    g = ComputationGraph(128)
    inp = g.add_input()
    ln = g.add_op("rmsnorm", [inp])
    at = g.add_op("learnable_semiring_attention", [ln])
    g.set_output(g.add_op("add", [inp, at]))

    model = compile_model([g] * 2, vocab_size=256, max_seq_len=32)
    y = model(torch.randint(0, 256, (2, 16)))
    assert tuple(y.shape) == (2, 16, 256)
    y.float().pow(2).mean().backward()

    betas = [p for n, p in model.named_parameters() if "semiring_beta" in n]
    assert betas, "semiring_beta parameter missing"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in betas)


def test_screening_template_instantiates_and_compiles():
    """The screening template builds a valid graph that compiles + runs."""
    from research.synthesis._template_attention_manifest import (
        tpl_learnable_semiring_attention_block,
    )
    from research.synthesis.compiler import compile_model
    from research.synthesis.graph import ComputationGraph

    g = ComputationGraph(128)
    g.set_output(
        tpl_learnable_semiring_attention_block(g, g.add_input(), random.Random(0))
    )
    model = compile_model([g] * 2, vocab_size=256, max_seq_len=32)
    y = model(torch.randint(0, 256, (2, 16)))
    assert tuple(y.shape) == (2, 16, 256)
