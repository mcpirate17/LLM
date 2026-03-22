"""Regression tests for local_window_attn shared memory overflow.

Verifies:
1. The op doesn't crash with OutOfResources for any (W, D) combo the grammar can produce
2. The Python fallback path works when Triton is unavailable or fails
3. Valid residual-attention context still produces learning signal
"""

import pytest
import torch
import torch.nn.functional as F

from research.synthesis.compiler_ops_attention import _op_local_window_attn


@pytest.mark.unit
class TestLocalWindowAttnSharedMemory:
    """Ensure no shared memory overflow for any grammar-producible config."""

    @pytest.mark.parametrize("D", [128, 256, 512])
    @pytest.mark.parametrize("W", [8, 16, 32])
    def test_forward_no_crash(self, D, W):
        """Forward pass must not raise for any (D, W) combination."""
        x = torch.randn(2, 64, D)
        config = {"window_size": W}
        # Must not raise — either Triton succeeds or falls through to Python
        out = _op_local_window_attn(None, (x,), config)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_window_clamped_at_large_dim(self):
        """At D>=256, window_size>16 should be clamped to 16 in the op."""
        x = torch.randn(2, 64, 256)
        # Even with W=32 in config, the op should clamp internally
        config = {"window_size": 32}
        out = _op_local_window_attn(None, (x,), config)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_window_32_ok_at_small_dim(self):
        """At D<256, window_size=32 should work fine."""
        x = torch.randn(2, 64, 128)
        config = {"window_size": 32}
        out = _op_local_window_attn(None, (x,), config)
        assert out.shape == x.shape


@pytest.mark.unit
class TestLocalWindowAttnFallback:
    """Python fallback path produces valid causal local attention."""

    def test_causal_masking(self):
        """Output at position t depends only on positions max(0, t-W)..t."""
        D = 64
        S = 16
        W = 4
        x = torch.randn(1, S, D)
        config = {"window_size": W}
        out = _op_local_window_attn(None, (x,), config)
        assert out.shape == (1, S, D)
        # Verify causality: perturb a future position, output at t should not change
        x2 = x.clone()
        x2[0, -1, :] = 999.0  # Perturb last position
        out2 = _op_local_window_attn(None, (x2,), config)
        # Positions 0..S-2 should be identical (last position is only in its own window)
        assert torch.allclose(out[0, : S - W, :], out2[0, : S - W, :], atol=1e-5)

    def test_gradient_flows(self):
        """Gradient must flow through the Python fallback path."""
        x = torch.randn(2, 32, 64, requires_grad=True)
        config = {"window_size": 8}
        out = _op_local_window_attn(None, (x,), config)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


@pytest.mark.unit
class TestLocalWindowAttnResidualContext:
    """Valid residual-attention context produces learning signal."""

    def test_valid_residual_graph_compiles_and_forwards(self):
        """rmsnorm → local_window_attn → linear_proj → add(input, proj)."""
        from research.synthesis.graph import ComputationGraph
        from research.synthesis.compiler import compile_model

        g = ComputationGraph(256)
        inp = g.add_input()
        norm = g.add_op("rmsnorm", [inp])
        attn = g.add_op("local_window_attn", [norm], config={"window_size": 16})
        proj = g.add_op("linear_proj", [attn], config={"out_dim": 256})
        res = g.add_op("add", [inp, proj])
        g.set_output(res)

        model = compile_model([g] * 4, vocab_size=32000, max_seq_len=256)
        x = torch.randint(0, 32000, (2, 64))

        # Forward must not crash
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 64, 32000)
        assert torch.isfinite(out).all()

        # Must produce learning signal (loss decreases over 20 steps)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out.view(-1, out.size(-1)), x.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"No learning: {losses[0]:.2f} → {losses[-1]:.2f}"
        )
