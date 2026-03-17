"""Tests for native C kernel autograd integration (scientist/native_autograd.py).

Verifies:
1. Forward correctness — native autograd output matches PyTorch reference.
2. Backward correctness — ``torch.autograd.gradcheck`` passes for each op.
3. Gradient chain — gradients flow correctly through multi-op compositions.
4. NativeForwardWrapper autograd routing — wrapper uses autograd path when
   inputs require grad.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attempt to import the native autograd module.  All tests are skipped when
# the Cython bridge (aria_bridge) is unavailable.
# ---------------------------------------------------------------------------

try:
    from research.scientist.native_autograd import (
        NATIVE_AUTOGRAD_SUPPORTED_OPS,
        NativeAdd,
        NativeGelu,
        NativeMatmul,
        NativeMul,
        NativeRelu,
        NativeSigmoid,
        NativeSilu,
        NativeSub,
        NativeTanh,
        native_autograd_dispatch,
    )
    from research.scientist.native_runner import (
        NativeForwardWrapper,
        dispatch_op_native,
    )

    # Smoke-test that the Cython bridge is loadable.
    dispatch_op_native("relu", np.array([1.0], dtype=np.float32))
    _HAS_NATIVE = True
except Exception:
    _HAS_NATIVE = False

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(not _HAS_NATIVE, reason="Native Cython bridge unavailable"),
]


# ---------------------------------------------------------------------------
# Forward correctness tests
# ---------------------------------------------------------------------------


class TestForwardCorrectness:
    """Forward output of native autograd ops must match PyTorch reference."""

    def test_relu_forward(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = NativeRelu.apply(x)
        expected = F.relu(x)
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_sigmoid_forward(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = NativeSigmoid.apply(x)
        expected = torch.sigmoid(x)
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_tanh_forward(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = NativeTanh.apply(x)
        expected = torch.tanh(x)
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_gelu_forward(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = NativeGelu.apply(x)
        expected = F.gelu(x)
        # C kernel may use tanh approximation; allow wider tolerance.
        torch.testing.assert_close(result, expected.detach(), atol=5e-4, rtol=5e-3)

    def test_silu_forward(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = NativeSilu.apply(x)
        expected = F.silu(x)
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_add_forward(self):
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        result = NativeAdd.apply(a, b)
        expected = a + b
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_mul_forward(self):
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result = NativeMul.apply(a, b)
        expected = a * b
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_sub_forward(self):
        a = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = NativeSub.apply(a, b)
        expected = a - b
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_matmul_forward(self):
        A = torch.randn(4, 3, requires_grad=True)
        B = torch.randn(3, 5, requires_grad=True)
        result = NativeMatmul.apply(A, B)
        expected = A @ B
        torch.testing.assert_close(result, expected.detach(), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Backward correctness tests (gradcheck)
# ---------------------------------------------------------------------------


class TestBackwardGradcheck:
    """Each native autograd Function must pass torch.autograd.gradcheck."""

    def test_relu_gradcheck(self):
        # Avoid exact zero where relu is non-differentiable.
        x = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeRelu.apply, (x.float(),), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_sigmoid_gradcheck(self):
        x = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeSigmoid.apply, (x.float(),), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_tanh_gradcheck(self):
        x = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeTanh.apply, (x.float(),), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_gelu_gradcheck(self):
        x = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeGelu.apply, (x.float(),), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_silu_gradcheck(self):
        x = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeSilu.apply, (x.float(),), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_add_gradcheck(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeAdd.apply, (a.float(), b.float()), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_mul_gradcheck(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeMul.apply, (a.float(), b.float()), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_sub_gradcheck(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeSub.apply, (a.float(), b.float()), eps=1e-3, atol=1e-3, rtol=1e-2
        )

    def test_matmul_gradcheck(self):
        A = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        B = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            NativeMatmul.apply, (A.float(), B.float()), eps=1e-3, atol=1e-3, rtol=1e-2
        )


# ---------------------------------------------------------------------------
# Manual backward tests (verify gradient values)
# ---------------------------------------------------------------------------


class TestBackwardValues:
    """Verify that backward produces the correct gradient values."""

    def test_relu_backward_values(self):
        x = torch.tensor([-1.0, 0.5, 2.0, -0.5], requires_grad=True)
        y = NativeRelu.apply(x)
        y.sum().backward()
        # relu grad: 1 where x > 0, 0 elsewhere
        expected_grad = torch.tensor([0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-5)

    def test_add_backward_values(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        y = NativeAdd.apply(a, b)
        y.sum().backward()
        # add grad: 1 for both inputs
        torch.testing.assert_close(a.grad, torch.ones(2), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(b.grad, torch.ones(2), atol=1e-6, rtol=1e-5)

    def test_mul_backward_values(self):
        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([5.0, 7.0], requires_grad=True)
        y = NativeMul.apply(a, b)
        y.sum().backward()
        # mul grad: grad_a = b, grad_b = a
        torch.testing.assert_close(
            a.grad, torch.tensor([5.0, 7.0]), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            b.grad, torch.tensor([2.0, 3.0]), atol=1e-6, rtol=1e-5
        )

    def test_sub_backward_values(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        y = NativeSub.apply(a, b)
        y.sum().backward()
        # sub grad: grad_a = 1, grad_b = -1
        torch.testing.assert_close(a.grad, torch.ones(2), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(b.grad, -torch.ones(2), atol=1e-6, rtol=1e-5)

    def test_matmul_backward_values(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = torch.tensor([[5.0], [6.0]], requires_grad=True)
        C = NativeMatmul.apply(A, B)
        C.sum().backward()
        # grad_A = grad_out @ B^T, grad_B = A^T @ grad_out
        # grad_out is [[1],[1]], B^T = [[5,6]]
        # grad_A = [[5,6],[5,6]]
        expected_grad_A = torch.tensor([[5.0, 6.0], [5.0, 6.0]])
        # A^T = [[1,3],[2,4]], grad_out = [[1],[1]]
        # grad_B = [[4],[6]]
        expected_grad_B = torch.tensor([[4.0], [6.0]])
        torch.testing.assert_close(A.grad, expected_grad_A, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(B.grad, expected_grad_B, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Multi-op chain gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientChain:
    """Gradients must flow correctly through chains of native autograd ops."""

    def test_relu_add_chain(self):
        """relu(a) + b should propagate gradients to both a and b."""
        a = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        b = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = NativeAdd.apply(NativeRelu.apply(a), b)
        loss = y.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        # relu(a) = [1, 0, 2], grad of relu: [1, 0, 1]
        # add passes grad 1 to both
        expected_a_grad = torch.tensor([1.0, 0.0, 1.0])
        torch.testing.assert_close(a.grad, expected_a_grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(b.grad, torch.ones(3), atol=1e-6, rtol=1e-5)

    def test_matmul_sigmoid_chain(self):
        """sigmoid(A @ B) should propagate gradients to both A and B."""
        A = torch.randn(3, 4, requires_grad=True)
        B = torch.randn(4, 2, requires_grad=True)
        C = NativeMatmul.apply(A, B)
        y = NativeSigmoid.apply(C.view(-1))
        loss = y.sum()
        loss.backward()
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape

    def test_mul_tanh_sub_chain(self):
        """tanh(a * b) - c should propagate to a, b, c."""
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([0.5, 0.3], requires_grad=True)
        c = torch.tensor([0.1, 0.2], requires_grad=True)
        prod = NativeMul.apply(a, b)
        activated = NativeTanh.apply(prod)
        result = NativeSub.apply(activated, c)
        loss = result.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None

    def test_three_op_chain_relu_matmul_sigmoid(self):
        """relu -> matmul -> sigmoid pipeline."""
        x = torch.randn(4, requires_grad=True)
        W = torch.randn(4, 3, requires_grad=True)

        h = NativeRelu.apply(x)
        # matmul expects 2D
        out = NativeMatmul.apply(h.unsqueeze(0), W)
        y = NativeSigmoid.apply(out.view(-1))
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert W.grad is not None
        assert x.grad.shape == x.shape
        assert W.grad.shape == W.shape


# ---------------------------------------------------------------------------
# Dispatch registry tests
# ---------------------------------------------------------------------------


class TestDispatchRegistry:
    """native_autograd_dispatch should select the correct Function class."""

    def test_dispatch_relu(self):
        x = torch.tensor([1.0, -1.0], requires_grad=True)
        result = native_autograd_dispatch("relu", x)
        assert result.grad_fn is not None  # has autograd history
        expected = F.relu(x)
        torch.testing.assert_close(result, expected.detach(), atol=1e-6, rtol=1e-5)

    def test_dispatch_add(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = native_autograd_dispatch("add", a, b)
        assert result.grad_fn is not None
        torch.testing.assert_close(result, (a + b).detach(), atol=1e-6, rtol=1e-5)

    def test_dispatch_unknown_raises(self):
        x = torch.tensor([1.0])
        with pytest.raises(ValueError, match="No native autograd Function"):
            native_autograd_dispatch("unknown_op", x)

    def test_supported_ops_set(self):
        expected = {
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            "silu",
            "add",
            "mul",
            "sub",
            "matmul",
            "softmax",
            "layernorm",
            "rmsnorm",
            "minimum",
            "maximum",
            "div_safe",
            "sign_ste",
        }
        assert NATIVE_AUTOGRAD_SUPPORTED_OPS == expected


# ---------------------------------------------------------------------------
# NativeForwardWrapper autograd routing test
# ---------------------------------------------------------------------------


class TestWrapperAutogradRouting:
    """NativeForwardWrapper.dispatch should use autograd path when grad needed."""

    def test_wrapper_routes_through_autograd_when_grad_required(self):
        """When input requires_grad, wrapper should return a tensor with grad_fn."""
        wrapper = NativeForwardWrapper(MagicMock(), {"relu", "add", "mul"})
        x = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        result = wrapper.dispatch("relu", x)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        # Should have a grad_fn since we went through autograd
        assert result.grad_fn is not None
        # Gradient should flow
        result.sum().backward()
        assert x.grad is not None

    def test_wrapper_no_autograd_when_no_grad(self):
        """When input does not require grad, wrapper should use the non-autograd path."""
        wrapper = NativeForwardWrapper(MagicMock(), {"relu"})
        x = torch.tensor([1.0, -1.0, 2.0], requires_grad=False)
        result = wrapper.dispatch("relu", x)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        # Should NOT have a grad_fn
        assert result.grad_fn is None
