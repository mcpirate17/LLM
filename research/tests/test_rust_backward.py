"""Tests for Rust scheduler backward graph execution.

These tests verify that the backward pass through the Rust scheduler
produces correct gradients for simple chains of operations.
"""

import json
import math
import pytest
import numpy as np
from research.scientist.native.core import _try_import_rust_scheduler

aria_scheduler = _try_import_rust_scheduler()
HAS_RUST_SCHEDULER = aria_scheduler is not None

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        not HAS_RUST_SCHEDULER,
        reason="aria_scheduler Rust module not available",
    ),
]


def _make_graph_json(nodes, edges, output_node_id):
    """Helper to build a native IR JSON string."""
    return json.dumps(
        {
            "schema_version": "0.1",
            "model_dim": 4,
            "nodes": nodes,
            "edges": edges,
            "output_node_id": output_node_id,
            "metadata": None,
        }
    )


# ── Graph builders ──────────────────────────────────────────────────


def _relu_chain_graph():
    """input(0) -> relu(1) -> output(2)"""
    nodes = [
        {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        },
        {
            "id": 1,
            "op_name": "relu",
            "input_ids": [0],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 2,
            "op_name": "output",
            "input_ids": [1],
            "config": {},
            "is_input": False,
            "is_output": True,
        },
    ]
    edges = [
        {"source": 0, "target": 1, "source_port": None, "target_port": None},
        {"source": 1, "target": 2, "source_port": None, "target_port": None},
    ]
    return _make_graph_json(nodes, edges, 2)


def _add_graph():
    """input_a(0) + input_b(1) -> add(2) -> output(3)

    Note: for the Rust scheduler, we use a single input node and the add
    op has two references to the same input.  This tests gradient accumulation.
    """
    nodes = [
        {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        },
        {
            "id": 1,
            "op_name": "add",
            "input_ids": [0, 0],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 2,
            "op_name": "output",
            "input_ids": [1],
            "config": {},
            "is_input": False,
            "is_output": True,
        },
    ]
    edges = [
        {"source": 0, "target": 1, "source_port": None, "target_port": None},
        {"source": 1, "target": 2, "source_port": None, "target_port": None},
    ]
    return _make_graph_json(nodes, edges, 2)


def _sigmoid_chain_graph():
    """input(0) -> sigmoid(1) -> output(2)"""
    nodes = [
        {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        },
        {
            "id": 1,
            "op_name": "sigmoid",
            "input_ids": [0],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 2,
            "op_name": "output",
            "input_ids": [1],
            "config": {},
            "is_input": False,
            "is_output": True,
        },
    ]
    edges = [
        {"source": 0, "target": 1, "source_port": None, "target_port": None},
        {"source": 1, "target": 2, "source_port": None, "target_port": None},
    ]
    return _make_graph_json(nodes, edges, 2)


def _gelu_chain_graph():
    """input(0) -> gelu(1) -> output(2)"""
    nodes = [
        {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        },
        {
            "id": 1,
            "op_name": "gelu",
            "input_ids": [0],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 2,
            "op_name": "output",
            "input_ids": [1],
            "config": {},
            "is_input": False,
            "is_output": True,
        },
    ]
    edges = [
        {"source": 0, "target": 1, "source_port": None, "target_port": None},
        {"source": 1, "target": 2, "source_port": None, "target_port": None},
    ]
    return _make_graph_json(nodes, edges, 2)


def _relu_relu_chain_graph():
    """input(0) -> relu(1) -> relu(2) -> output(3)"""
    nodes = [
        {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        },
        {
            "id": 1,
            "op_name": "relu",
            "input_ids": [0],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 2,
            "op_name": "relu",
            "input_ids": [1],
            "config": {},
            "is_input": False,
            "is_output": False,
        },
        {
            "id": 3,
            "op_name": "output",
            "input_ids": [2],
            "config": {},
            "is_input": False,
            "is_output": True,
        },
    ]
    edges = [
        {"source": 0, "target": 1, "source_port": None, "target_port": None},
        {"source": 1, "target": 2, "source_port": None, "target_port": None},
        {"source": 2, "target": 3, "source_port": None, "target_port": None},
    ]
    return _make_graph_json(nodes, edges, 3)


# ── Tests ───────────────────────────────────────────────────────────


class TestForwardSaved:
    """Test execute_graph_forward_saved returns activations."""

    def test_relu_forward_saved(self):
        graph_json = _relu_chain_graph()
        x = [-1.0, 0.0, 2.0, 3.5]
        result = aria_scheduler.execute_graph_forward_saved(graph_json, x)

        assert "output" in result
        assert "saved_activations" in result

        output = result["output"]
        saved = result["saved_activations"]

        # ReLU output: [0, 0, 2, 3.5]
        np.testing.assert_allclose(output, [0.0, 0.0, 2.0, 3.5], atol=1e-6)

        # Should have activations for all 3 nodes.
        assert 0 in saved  # input node
        assert 1 in saved  # relu node
        assert 2 in saved  # output node

        # Input activation should match input.
        np.testing.assert_allclose(saved[0], x, atol=1e-6)

    def test_sigmoid_forward_saved(self):
        graph_json = _sigmoid_chain_graph()
        x = [0.0, 1.0, -1.0, 2.0]
        result = aria_scheduler.execute_graph_forward_saved(graph_json, x)

        output = result["output"]
        expected = [1.0 / (1.0 + math.exp(-v)) for v in x]
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_relu_forward_saved_handle_round_trip(self):
        graph_json = _relu_chain_graph()
        x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)
        result = aria_scheduler.execute_graph_forward_saved_arrays_handle(graph_json, x)

        assert "output" in result
        assert "saved_state" in result

        grad_out = np.ones_like(x)
        bwd = aria_scheduler.execute_graph_backward_arrays_handle(
            graph_json,
            grad_out,
            result["saved_state"],
        )
        input_grad = np.array(bwd["grads"][0])
        assert input_grad[0] == 0.0
        assert input_grad[2] == 1.0
        assert input_grad[3] == 1.0


class TestBackwardRelu:
    """Test backward pass for relu chains."""

    def test_relu_backward_ones_grad(self):
        graph_json = _relu_chain_graph()
        x = [-1.0, 0.0, 2.0, 3.5]

        # Forward pass with saved activations.
        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        # Backward with all-ones gradient.
        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        # Input node (0) should have gradient: relu'(x) * grad_out
        # relu'(-1) = 0, relu'(0) = 0 (or 1 depending on impl), relu'(2) = 1, relu'(3.5) = 1
        input_grad = np.array(grads[0])
        assert input_grad[0] == 0.0, "grad for x=-1 should be 0 (relu kills it)"
        assert input_grad[2] == 1.0, "grad for x=2 should be 1"
        assert input_grad[3] == 1.0, "grad for x=3.5 should be 1"

    def test_relu_backward_scaled_grad(self):
        graph_json = _relu_chain_graph()
        x = [-1.0, 0.5, 2.0, 3.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        grad_out = [2.0, 3.0, 0.5, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        input_grad = np.array(grads[0])
        # x=-1 -> relu'=0 -> grad=0
        assert input_grad[0] == 0.0
        # x=0.5 -> relu'=1 -> grad=3.0
        np.testing.assert_allclose(input_grad[1], 3.0, atol=1e-6)
        # x=2 -> relu'=1 -> grad=0.5
        np.testing.assert_allclose(input_grad[2], 0.5, atol=1e-6)

    def test_relu_relu_chain_backward(self):
        """Two consecutive relu ops."""
        graph_json = _relu_relu_chain_graph()
        x = [-2.0, -0.5, 1.0, 3.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        input_grad = np.array(grads[0])
        # relu(relu(x)): grad = relu'(relu(x)) * relu'(x)
        # x=-2: relu(-2)=0, relu'(-2)=0 -> grad=0
        # x=-0.5: relu(-0.5)=0, relu'(-0.5)=0 -> grad=0
        # x=1: relu(1)=1, relu'(1)=1, relu'(1)=1 -> grad=1
        # x=3: relu(3)=3, relu'(3)=1, relu'(3)=1 -> grad=1
        np.testing.assert_allclose(input_grad, [0.0, 0.0, 1.0, 1.0], atol=1e-6)


class TestBackwardSigmoid:
    """Test backward pass for sigmoid."""

    def test_sigmoid_backward(self):
        graph_json = _sigmoid_chain_graph()
        x = [0.0, 1.0, -1.0, 2.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        input_grad = np.array(grads[0])

        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        for i, xi in enumerate(x):
            s = 1.0 / (1.0 + math.exp(-xi))
            expected_grad = s * (1.0 - s)
            np.testing.assert_allclose(
                input_grad[i],
                expected_grad,
                atol=1e-5,
                err_msg=f"sigmoid backward mismatch at index {i}",
            )


class TestBackwardAdd:
    """Test backward pass for add op."""

    def test_add_self_backward(self):
        """Add with same input on both ports: grad should accumulate."""
        graph_json = _add_graph()
        x = [1.0, 2.0, 3.0, 4.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        # add(x, x) = 2x, so output should be [2, 4, 6, 8]
        np.testing.assert_allclose(fwd["output"], [2.0, 4.0, 6.0, 8.0], atol=1e-6)

        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        input_grad = np.array(grads[0])
        # d(x+x)/dx = 2, so grad should be [2, 2, 2, 2]
        np.testing.assert_allclose(input_grad, [2.0, 2.0, 2.0, 2.0], atol=1e-6)


class TestBackwardGelu:
    """Test backward pass for gelu."""

    def test_gelu_backward_ones(self):
        graph_json = _gelu_chain_graph()
        x = [0.0, 1.0, -1.0, 2.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        grads = bwd["grads"]
        input_grad = np.array(grads[0])

        # GELU'(0) = 0.5 (the derivative at x=0 for standard GELU)
        np.testing.assert_allclose(input_grad[0], 0.5, atol=0.05)

        # GELU'(x) > 0 for x > 0
        assert input_grad[1] > 0, "GELU gradient for x=1 should be positive"
        assert input_grad[3] > 0, "GELU gradient for x=2 should be positive"


class TestBackwardResult:
    """Test backward result structure."""

    def test_backward_result_has_all_node_grads(self):
        graph_json = _relu_chain_graph()
        x = [1.0, 2.0, 3.0, 4.0]

        fwd = aria_scheduler.execute_graph_forward_saved(graph_json, x)
        saved = fwd["saved_activations"]

        grad_out = [1.0, 1.0, 1.0, 1.0]
        bwd = aria_scheduler.execute_graph_backward(graph_json, grad_out, saved)

        assert "grads" in bwd
        assert "arena_bytes_used" in bwd
        # Should have gradient for at least the input node.
        assert 0 in bwd["grads"]
