"""Tests for NativeSubgraphFunction — full-graph native forward + backward.

Verifies that:
1. Forward through subgraph matches per-op path.
2. Backward through subgraph produces gradients.
3. Gradient correctness: compare against per-op autograd path.
4. Multi-step training using subgraph dispatch.

The key value proposition being tested: instead of N per-op Python->C
roundtrips in forward AND backward, the subgraph path does 1 Rust call
for forward + 1 Rust call for backward.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure the research package is importable.
_root = str(Path(__file__).resolve().parents[1].parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.synthesis.graph import ComputationGraph, OpNode, ShapeInfo
from research.scientist.native_runner import (
    NativeSubgraphFunction,
    SubgraphDispatcher,
    dispatch_graph_forward_native_saved,
    dispatch_graph_backward_native,
)


# ---------------------------------------------------------------------------
# Detect whether the Rust scheduler is available (skip tests if not).
# ---------------------------------------------------------------------------

try:
    from research.scientist.native_runner import _try_import_rust_scheduler
    _rust = _try_import_rust_scheduler()
    if _rust is None:
        raise ImportError("Rust scheduler not available")
    # Smoke test
    from research.scientist.native_runner import dispatch_op_native
    dispatch_op_native("relu", np.array([1.0], dtype=np.float32))
    _HAS_NATIVE = True
except Exception:
    _HAS_NATIVE = False

pytestmark = pytest.mark.skipif(
    not _HAS_NATIVE, reason="Native Rust scheduler or Cython bridge unavailable"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_relu_graph(model_dim: int = 4) -> ComputationGraph:
    """input -> relu -> output"""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    r = g.add_op("relu", [inp])
    g.set_output(r)
    return g


def _make_add_graph(model_dim: int = 4) -> ComputationGraph:
    """input -> add(input, input) -> output  (doubles the input)"""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    a = g.add_op("add", [inp, inp])
    g.set_output(a)
    return g


def _make_chain_graph(model_dim: int = 4) -> ComputationGraph:
    """input -> relu -> add(relu_out, input) -> output"""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    r = g.add_op("relu", [inp])
    a = g.add_op("add", [r, inp])
    g.set_output(a)
    return g


def _make_diamond_graph(model_dim: int = 4) -> ComputationGraph:
    """input -> relu, input -> gelu, relu+gelu -> add -> output"""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    relu_id = g.add_op("relu", [inp])
    gelu_id = g.add_op("gelu", [inp])
    add_id = g.add_op("add", [relu_id, gelu_id])
    g.set_output(add_id)
    return g


# ---------------------------------------------------------------------------
# Test 1: Forward through subgraph matches per-op path
# ---------------------------------------------------------------------------

class TestForwardMatches:
    """Forward output from NativeSubgraphFunction should match per-op execution."""

    def test_relu_forward_matches(self):
        g = _make_relu_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8)
        result = fn_cls.apply(x)
        expected = F.relu(x)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_add_forward_matches(self):
        g = _make_add_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8)
        result = fn_cls.apply(x)
        expected = x + x

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_chain_forward_matches(self):
        g = _make_chain_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8)
        result = fn_cls.apply(x)
        expected = F.relu(x) + x

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_diamond_forward_matches(self):
        g = _make_diamond_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8)
        result = fn_cls.apply(x)
        expected = F.relu(x) + F.gelu(x)

        # gelu has slight numerical differences between C and PyTorch implementations
        torch.testing.assert_close(result, expected, atol=5e-4, rtol=5e-4)


# ---------------------------------------------------------------------------
# Test 2: Backward through subgraph produces gradients
# ---------------------------------------------------------------------------

class TestBackwardProducesGradients:
    """NativeSubgraphFunction.backward should populate input.grad."""

    def test_relu_backward_produces_grad(self):
        g = _make_relu_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = fn_cls.apply(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None, "x.grad should not be None after backward"
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_add_backward_produces_grad(self):
        g = _make_add_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = fn_cls.apply(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_chain_backward_produces_grad(self):
        g = _make_chain_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = fn_cls.apply(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_diamond_backward_produces_grad(self):
        g = _make_diamond_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = fn_cls.apply(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Test 3: Gradient correctness — compare against per-op autograd path
# ---------------------------------------------------------------------------

class TestGradientCorrectness:
    """Compare subgraph backward gradients against PyTorch reference."""

    def test_relu_gradient_matches_pytorch(self):
        """Subgraph relu gradient should match torch.relu gradient."""
        g = _make_relu_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)
        torch.manual_seed(42)

        # Subgraph path
        x_sub = torch.randn(1, 2, 8, requires_grad=True)
        out_sub = fn_cls.apply(x_sub)
        loss_sub = out_sub.sum()
        loss_sub.backward()

        # PyTorch reference
        x_ref = x_sub.data.clone().requires_grad_(True)
        out_ref = F.relu(x_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        torch.testing.assert_close(
            x_sub.grad, x_ref.grad, atol=1e-5, rtol=1e-5,
            msg="Relu gradient mismatch between subgraph and PyTorch",
        )

    def test_add_gradient_matches_pytorch(self):
        """Subgraph add(x,x) gradient should be 2*ones (since d(x+x)/dx = 2)."""
        g = _make_add_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x_sub = torch.randn(1, 2, 8, requires_grad=True)
        out_sub = fn_cls.apply(x_sub)
        loss_sub = out_sub.sum()
        loss_sub.backward()

        # PyTorch reference: x + x
        x_ref = x_sub.data.clone().requires_grad_(True)
        out_ref = x_ref + x_ref
        loss_ref = out_ref.sum()
        loss_ref.backward()

        torch.testing.assert_close(
            x_sub.grad, x_ref.grad, atol=1e-5, rtol=1e-5,
            msg="Add gradient mismatch between subgraph and PyTorch",
        )

    def test_chain_gradient_matches_pytorch(self):
        """input -> relu -> add(relu_out, input) gradient should match PyTorch."""
        g = _make_chain_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x_sub = torch.randn(1, 2, 8, requires_grad=True)
        out_sub = fn_cls.apply(x_sub)
        loss_sub = out_sub.sum()
        loss_sub.backward()

        x_ref = x_sub.data.clone().requires_grad_(True)
        out_ref = F.relu(x_ref) + x_ref
        loss_ref = out_ref.sum()
        loss_ref.backward()

        torch.testing.assert_close(
            x_sub.grad, x_ref.grad, atol=1e-4, rtol=1e-4,
            msg="Chain gradient mismatch between subgraph and PyTorch",
        )

    def test_diamond_gradient_matches_pytorch(self):
        """Diamond: relu(x) + gelu(x) gradient should match PyTorch."""
        g = _make_diamond_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x_sub = torch.randn(1, 2, 8, requires_grad=True)
        out_sub = fn_cls.apply(x_sub)
        loss_sub = out_sub.sum()
        loss_sub.backward()

        x_ref = x_sub.data.clone().requires_grad_(True)
        out_ref = F.relu(x_ref) + F.gelu(x_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # gelu backward has slight numerical differences between C and PyTorch
        torch.testing.assert_close(
            x_sub.grad, x_ref.grad, atol=5e-4, rtol=5e-4,
            msg="Diamond gradient mismatch between subgraph and PyTorch",
        )


# ---------------------------------------------------------------------------
# Test 4: SubgraphDispatcher routes to autograd when requires_grad=True
# ---------------------------------------------------------------------------

class TestSubgraphDispatcherAutograd:
    """SubgraphDispatcher.try_dispatch should use NativeSubgraphFunction
    when the input requires grad, and the inference path otherwise."""

    def test_dispatch_with_grad_returns_tensor_with_grad_fn(self):
        g = _make_relu_graph(model_dim=8)
        supported = {"relu"}
        dispatcher = SubgraphDispatcher(g, supported)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = dispatcher.try_dispatch(x)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.grad_fn is not None, "Result should have grad_fn for autograd"

    def test_dispatch_with_grad_backward_works(self):
        g = _make_relu_graph(model_dim=8)
        supported = {"relu"}
        dispatcher = SubgraphDispatcher(g, supported)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = dispatcher.try_dispatch(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @patch("research.scientist.native_runner.dispatch_graph_native_cached")
    def test_dispatch_without_grad_uses_inference_path(self, mock_cached):
        """When requires_grad=False, should use the numpy inference path."""
        g = _make_relu_graph(model_dim=8)
        supported = {"relu"}
        dispatcher = SubgraphDispatcher(g, supported)

        mock_cached.return_value = np.zeros((1, 2, 8), dtype=np.float32)
        x = torch.randn(1, 2, 8, requires_grad=False)
        result = dispatcher.try_dispatch(x)

        assert result is not None
        mock_cached.assert_called_once()

    def test_dispatch_stats_tracked_for_grad_path(self):
        g = _make_relu_graph(model_dim=8)
        supported = {"relu"}
        dispatcher = SubgraphDispatcher(g, supported)

        x = torch.randn(1, 2, 8, requires_grad=True)
        dispatcher.try_dispatch(x)
        dispatcher.try_dispatch(x)

        assert dispatcher.stats["subgraph_dispatches"] == 2


# ---------------------------------------------------------------------------
# Test 5: Multi-step training using subgraph dispatch
# ---------------------------------------------------------------------------

class TestMultiStepTraining:
    """Train a simple model using SubgraphDispatcher for forward/backward."""

    def test_training_loss_decreases(self):
        """Use a relu subgraph in a training loop; verify loss decreases."""
        g = _make_relu_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        torch.manual_seed(42)
        W = (torch.randn(8, 8) * 0.1).requires_grad_(True)
        x = torch.randn(4, 2, 8)
        target = torch.randn(4, 2, 8)

        optimizer = torch.optim.SGD([W], lr=0.01)

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            # x @ W -> relu (via subgraph)
            h = x @ W
            out = fn_cls.apply(h)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f} final={losses[-1]:.4f}"
        )

    def test_training_via_dispatcher(self):
        """End-to-end: SubgraphDispatcher in a training loop."""
        g = _make_chain_graph(model_dim=8)
        supported = {"relu", "add"}
        dispatcher = SubgraphDispatcher(g, supported)

        torch.manual_seed(42)
        W = (torch.randn(8, 8) * 0.1).requires_grad_(True)
        x = torch.randn(4, 2, 8)
        target = torch.randn(4, 2, 8)

        optimizer = torch.optim.SGD([W], lr=0.01)

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            h = x @ W
            h.requires_grad_(True)
            out = dispatcher.try_dispatch(h)
            assert out is not None, f"Dispatcher returned None at step {step}"
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f} final={losses[-1]:.4f}"
        )
        assert dispatcher.stats["subgraph_dispatches"] == 30

    def test_gradient_accumulation_via_subgraph(self):
        """Two forward/backward passes without zeroing should accumulate grads."""
        g = _make_relu_graph(model_dim=8)
        fn_cls = NativeSubgraphFunction.make(g)

        x = torch.randn(1, 2, 8, requires_grad=True)

        # First forward/backward
        out1 = fn_cls.apply(x)
        loss1 = out1.sum()
        loss1.backward()
        grad_after_one = x.grad.clone()

        # Second forward/backward WITHOUT zeroing
        out2 = fn_cls.apply(x)
        loss2 = out2.sum()
        loss2.backward()
        grad_after_two = x.grad.clone()

        # Gradients should be accumulated (doubled for same data)
        torch.testing.assert_close(
            grad_after_two, grad_after_one * 2,
            atol=1e-5, rtol=1e-4,
            msg="Gradients should accumulate after two backward passes",
        )


# ---------------------------------------------------------------------------
# Test 6: NativeSubgraphFunction with pre-cached IR JSON
# ---------------------------------------------------------------------------

class TestIRCaching:
    """Verify NativeSubgraphFunction works with pre-serialized IR JSON."""

    def test_with_explicit_ir_json(self):
        from research.synthesis.native_ir_converter import graph_to_native_ir_json

        g = _make_relu_graph(model_dim=8)
        ir_json = graph_to_native_ir_json(g)
        fn_cls = NativeSubgraphFunction.make(g, ir_json=ir_json)

        x = torch.randn(1, 2, 8, requires_grad=True)
        result = fn_cls.apply(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        expected = F.relu(x.data).sign()
        # relu gradient: 1 where x > 0, 0 elsewhere
        torch.testing.assert_close(
            x.grad, (x.data > 0).float(), atol=1e-5, rtol=1e-5,
        )
