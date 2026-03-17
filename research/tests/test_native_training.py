"""End-to-end native training tests.

Verifies that forward + backward + parameter update works through the native
C kernels via the torch.autograd.Function subclasses in native_autograd.py.

Tests:
1. Simple parameter optimization (relu + matmul, MSE loss, optimizer step)
2. Multi-step training convergence (2-layer network, 50 SGD steps)
3. Gradient accumulation (forward/backward twice without zeroing)
4. Mixed native/PyTorch ops (native relu chained with PyTorch linear)
5. Training matches PyTorch reference (same init, same data, 10 steps)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import native autograd; skip all tests if Cython bridge unavailable.
# ---------------------------------------------------------------------------

try:
    from research.scientist.native_autograd import (
        NativeRelu,
        NativeMatmul,
        NativeSigmoid,
        native_autograd_dispatch,
    )
    from research.scientist.native_runner import dispatch_op_native

    # Smoke-test the Cython bridge.
    dispatch_op_native("relu", np.array([1.0], dtype=np.float32))
    _HAS_NATIVE = True
except Exception:
    _HAS_NATIVE = False

pytestmark = [
    pytest.mark.native,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_NATIVE, reason="Native Cython bridge unavailable"),
]


# ---------------------------------------------------------------------------
# Test 1: Simple parameter optimization
# ---------------------------------------------------------------------------


class TestSimpleParameterOptimization:
    """Run a parameter through native relu + native matmul, compute MSE loss,
    call backward, verify grad exists, take an optimizer step."""

    def test_forward_backward_parameter_update(self):
        torch.manual_seed(42)

        # Learnable parameter: a 4x8 weight matrix
        W = torch.randn(4, 8, requires_grad=True)
        W_init = W.clone().detach()

        # Input and target
        x = torch.randn(4, 4)  # (4, 4)
        target = torch.randn(4, 8)

        # Forward: x -> matmul(W) -> relu -> output
        # NativeMatmul expects (M, K) @ (K, N) -> (M, N)
        h = NativeMatmul.apply(x, W)  # (4, 8)
        out = NativeRelu.apply(h)  # (4, 8)

        # MSE loss
        loss = F.mse_loss(out, target)

        # Backward
        loss.backward()

        # Verify gradient exists and has correct shape
        assert W.grad is not None, "W.grad should not be None after backward"
        assert W.grad.shape == W.shape, (
            f"Grad shape {W.grad.shape} != param shape {W.shape}"
        )
        assert torch.isfinite(W.grad).all(), "Gradients should be finite"
        assert W.grad.abs().sum() > 0, "Gradients should be non-zero"

        # Optimizer step
        optimizer = torch.optim.SGD([W], lr=0.01)
        optimizer.step()

        # Parameter should have changed
        assert not torch.allclose(W.data, W_init), (
            "Parameter should change after optimizer step"
        )


# ---------------------------------------------------------------------------
# Test 2: Multi-step training convergence
# ---------------------------------------------------------------------------


class TestMultiStepTrainingConvergence:
    """Build a 2-layer network using native ops: x -> matmul(W1) -> relu ->
    matmul(W2) -> output. Train for 50 steps with SGD, verify loss decreases."""

    def test_two_layer_convergence(self):
        torch.manual_seed(123)

        in_dim, hidden_dim, out_dim = 4, 8, 4

        # Learnable parameters (multiply before setting requires_grad to keep leaf)
        W1 = (torch.randn(in_dim, hidden_dim) * 0.1).requires_grad_(True)
        W2 = (torch.randn(hidden_dim, out_dim) * 0.1).requires_grad_(True)

        # Fixed data
        x = torch.randn(8, in_dim)
        target = torch.randn(8, out_dim)

        optimizer = torch.optim.SGD([W1, W2], lr=0.1)

        losses = []
        for step in range(100):
            optimizer.zero_grad()

            # Forward: x -> matmul(W1) -> relu -> matmul(W2)
            h = NativeMatmul.apply(x, W1)  # (8, hidden_dim)
            h = NativeRelu.apply(h)  # (8, hidden_dim)
            out = NativeMatmul.apply(h, W2)  # (8, out_dim)

            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify convergence: final loss should be meaningfully less than initial
        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f} final={losses[-1]:.4f}"
        )
        # Check that loss decreased by at least 20%
        assert losses[-1] < losses[0] * 0.8, (
            f"Loss should decrease significantly: initial={losses[0]:.4f} final={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Gradient accumulation
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    """Run forward/backward twice without zeroing grads, verify gradients
    accumulated (approximately doubled)."""

    def test_gradient_accumulation(self):
        torch.manual_seed(77)

        W = torch.randn(4, 4, requires_grad=True)
        x = torch.randn(4, 4)
        target = torch.randn(4, 4)

        # First forward/backward
        out1 = NativeRelu.apply(NativeMatmul.apply(x, W))
        loss1 = F.mse_loss(out1, target)
        loss1.backward()

        grad_after_one = W.grad.clone()

        # Second forward/backward WITHOUT zeroing grads
        out2 = NativeRelu.apply(NativeMatmul.apply(x, W))
        loss2 = F.mse_loss(out2, target)
        loss2.backward()

        grad_after_two = W.grad.clone()

        # Gradients should have accumulated (doubled, since same data)
        torch.testing.assert_close(
            grad_after_two,
            grad_after_one * 2,
            atol=1e-5,
            rtol=1e-4,
            msg="Gradients should accumulate (double) after two backward passes",
        )


# ---------------------------------------------------------------------------
# Test 4: Mixed native/PyTorch ops
# ---------------------------------------------------------------------------


class TestMixedNativePyTorchOps:
    """Chain a native relu with a PyTorch nn.Linear, verify gradients flow
    through the boundary correctly."""

    def test_native_relu_into_pytorch_linear(self):
        torch.manual_seed(99)

        linear = nn.Linear(8, 4)
        x = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 4)

        # Forward: x -> native relu -> PyTorch linear
        h = NativeRelu.apply(x)
        out = linear(h)

        loss = F.mse_loss(out, target)
        loss.backward()

        # Gradients should flow back through both ops
        assert x.grad is not None, "Input grad should not be None"
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

        # Linear layer params should also have grads
        assert linear.weight.grad is not None
        assert linear.bias.grad is not None

    def test_pytorch_linear_into_native_sigmoid(self):
        torch.manual_seed(99)

        linear = nn.Linear(4, 8)
        x = torch.randn(4, 4, requires_grad=True)
        target = torch.randn(4, 8)

        # Forward: x -> PyTorch linear -> native sigmoid
        h = linear(x)
        out = NativeSigmoid.apply(h)

        loss = F.mse_loss(out, target)
        loss.backward()

        # Gradients should flow back through both ops
        assert x.grad is not None, "Input grad should not be None"
        assert torch.isfinite(x.grad).all()
        assert linear.weight.grad is not None

    def test_sandwich_native_pytorch_native(self):
        """native relu -> pytorch linear -> native relu -> loss"""
        torch.manual_seed(42)

        linear = nn.Linear(8, 8)
        x = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)

        h = NativeRelu.apply(x)
        h = linear(h)
        out = NativeRelu.apply(h)

        loss = F.mse_loss(out, target)
        loss.backward()

        assert x.grad is not None
        assert linear.weight.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(linear.weight.grad).all()


# ---------------------------------------------------------------------------
# Test 5: Training matches PyTorch reference
# ---------------------------------------------------------------------------


class TestTrainingMatchesPyTorchReference:
    """Same network architecture, same init, same data. Run 10 steps with
    native ops vs PyTorch ops. Verify final losses are close."""

    def test_native_vs_pytorch_training_parity(self):
        n_steps = 10
        lr = 0.01
        in_dim, hidden_dim, out_dim = 4, 8, 4

        # Fixed data
        torch.manual_seed(0)
        x = torch.randn(8, in_dim)
        target = torch.randn(8, out_dim)

        # --- Native path ---
        torch.manual_seed(1)
        W1_native = (torch.randn(in_dim, hidden_dim) * 0.1).requires_grad_(True)
        W2_native = (torch.randn(hidden_dim, out_dim) * 0.1).requires_grad_(True)
        opt_native = torch.optim.SGD([W1_native, W2_native], lr=lr)

        native_losses = []
        for _ in range(n_steps):
            opt_native.zero_grad()
            h = NativeMatmul.apply(x, W1_native)
            h = NativeRelu.apply(h)
            out = NativeMatmul.apply(h, W2_native)
            loss = F.mse_loss(out, target)
            loss.backward()
            opt_native.step()
            native_losses.append(loss.item())

        # --- PyTorch reference path ---
        torch.manual_seed(1)
        W1_ref = (torch.randn(in_dim, hidden_dim) * 0.1).requires_grad_(True)
        W2_ref = (torch.randn(hidden_dim, out_dim) * 0.1).requires_grad_(True)
        opt_ref = torch.optim.SGD([W1_ref, W2_ref], lr=lr)

        ref_losses = []
        for _ in range(n_steps):
            opt_ref.zero_grad()
            h = x @ W1_ref
            h = F.relu(h)
            out = h @ W2_ref
            loss = F.mse_loss(out, target)
            loss.backward()
            opt_ref.step()
            ref_losses.append(loss.item())

        # Verify losses match at each step (float32 tolerance)
        for step, (nl, rl) in enumerate(zip(native_losses, ref_losses)):
            assert abs(nl - rl) < 1e-3, (
                f"Step {step}: native loss {nl:.6f} != ref loss {rl:.6f} "
                f"(diff={abs(nl - rl):.2e})"
            )

        # Final parameters should be close
        torch.testing.assert_close(
            W1_native.data,
            W1_ref.data,
            atol=1e-3,
            rtol=1e-3,
            msg="W1 parameters diverged between native and PyTorch",
        )
        torch.testing.assert_close(
            W2_native.data,
            W2_ref.data,
            atol=1e-3,
            rtol=1e-3,
            msg="W2 parameters diverged between native and PyTorch",
        )

    def test_native_vs_pytorch_add_mul_chain(self):
        """Verify native add/mul in a training loop match PyTorch."""
        n_steps = 10
        lr = 0.01

        torch.manual_seed(0)
        x = torch.randn(4, 8)
        target = torch.randn(4, 8)

        # --- Native path ---
        torch.manual_seed(2)
        scale = (torch.randn(8) * 0.1).requires_grad_(True)
        bias = (torch.randn(8) * 0.1).requires_grad_(True)
        opt_native = torch.optim.SGD([scale, bias], lr=lr)

        native_final_loss = None
        for _ in range(n_steps):
            opt_native.zero_grad()
            # x * scale + bias using native ops
            h = native_autograd_dispatch("mul", x, scale.unsqueeze(0).expand_as(x))
            out = native_autograd_dispatch("add", h, bias.unsqueeze(0).expand_as(h))
            loss = F.mse_loss(out, target)
            loss.backward()
            opt_native.step()
            native_final_loss = loss.item()

        # --- PyTorch path ---
        torch.manual_seed(2)
        scale_ref = (torch.randn(8) * 0.1).requires_grad_(True)
        bias_ref = (torch.randn(8) * 0.1).requires_grad_(True)
        opt_ref = torch.optim.SGD([scale_ref, bias_ref], lr=lr)

        ref_final_loss = None
        for _ in range(n_steps):
            opt_ref.zero_grad()
            h = x * scale_ref.unsqueeze(0).expand_as(x)
            out = h + bias_ref.unsqueeze(0).expand_as(h)
            loss = F.mse_loss(out, target)
            loss.backward()
            opt_ref.step()
            ref_final_loss = loss.item()

        assert abs(native_final_loss - ref_final_loss) < 1e-3, (
            f"Native final loss {native_final_loss:.6f} != ref {ref_final_loss:.6f}"
        )
