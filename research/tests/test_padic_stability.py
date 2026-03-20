"""Tests for the padic_expand smooth expansion + ReZero fix.

Verifies:
  - padic_expansion uses smooth sin/cos decomposition (differentiable)
  - execute_padic_expand uses ReZero residual_scale (starts at 0)
  - 20-step dynamics probe passes (CV < 0.25)
  - No NaN/Inf in forward or backward pass
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.mathspaces.padic import padic_expansion, execute_padic_expand


def test_padic_expansion_is_smooth():
    """Verify the expansion is differentiable (no hard remainder)."""
    x = torch.randn(2, 8, 16, requires_grad=True)
    expanded = padic_expansion(x, n_digits=1)

    # n_digits=1 → sin+cos pair → D*2
    assert expanded.shape == (2, 8, 32), f"Expected (2,8,32), got {expanded.shape}"

    # Verify gradients flow (hard remainder has zero gradients almost everywhere)
    loss = expanded.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow through smooth expansion"
    assert (x.grad != 0).any(), (
        "Gradients should be non-zero (smooth, not hard remainder)"
    )


def test_padic_expansion_multi_digit():
    """n_digits=2 → 2 sin/cos pairs → D*4."""
    x = torch.randn(2, 8, 16)
    expanded = padic_expansion(x, n_digits=2)
    assert expanded.shape == (2, 8, 64), f"Expected (2,8,64), got {expanded.shape}"


def test_execute_padic_expand_rezero():
    """ReZero: with residual_scale=0, output should equal input."""
    D = 32

    class MockModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(D, D * 2) * 0.02)
            self.residual_scale = nn.Parameter(torch.zeros(1))

    module = MockModule()
    x = torch.randn(2, 8, D)
    out = execute_padic_expand(module, x)

    # With scale=0, output = x + projected * 0 = x
    assert torch.allclose(out, x, atol=1e-6), (
        "ReZero: output should equal input when scale=0"
    )


def test_padic_expand_dynamics_stable():
    """20-step dynamics probe should pass (CV < 0.25)."""
    from research.synthesis.compiler import compile_model
    from research.synthesis.graph import ComputationGraph

    D = 64
    g = ComputationGraph(D)
    inp = g.add_input()
    rn = g.add_op("rmsnorm", [inp])
    pe = g.add_op("padic_expand", [rn])
    proj = g.add_op("linear_proj", [pe], config={"out_dim": D})
    res = g.add_op("add", [inp, proj])
    g.set_output(res)

    model = compile_model([g] * 2, vocab_size=500, max_seq_len=32)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        ids = torch.randint(0, 500, (2, 16))
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        assert not torch.isnan(loss), "Loss should not be NaN"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    mean_l = sum(losses) / len(losses)
    var_l = sum((x - mean_l) ** 2 for x in losses) / len(losses)
    cv = (var_l**0.5) / mean_l if mean_l > 0 else 0

    assert cv < 0.25, f"Dynamics probe CV={cv:.3f} > 0.25 — training unstable"
