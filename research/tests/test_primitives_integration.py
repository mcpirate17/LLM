"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


try:
    from research.scientist.notebook import LabNotebook, ExperimentEntry
    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    from research.scientist.persona import Aria
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    import research.scientist.llm.prompts as _prompts_mod  # noqa: F401
    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    import research.scientist.llm.context as _context_mod  # noqa: F401
    HAS_CONTEXT = True
except Exception as e:
    HAS_CONTEXT = False
    print(f"Context import failed: {e}")


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestCompoundMathSpaceOps(unittest.TestCase):
    """Test compound cross-space math primitives."""

    def test_hyperbolic_norm_shape(self):
        """hyperbolic_norm preserves shape."""
        import torch
        from research.mathspaces.hyperbolic import execute_hyperbolic_norm
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16) * 0.1
        out = execute_hyperbolic_norm(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_tropical_gate_shape(self):
        """tropical_gate preserves shape."""
        import torch
        from research.mathspaces.tropical import execute_tropical_gate
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_tropical_gate(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_clifford_attention_shape(self):
        """clifford_attention preserves shape (D must be multiple of 8)."""
        import torch
        from research.mathspaces.clifford import execute_clifford_attention
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_clifford_attention(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_clifford_attention_padding(self):
        """clifford_attention handles D not divisible by 8."""
        import torch
        from research.mathspaces.clifford import execute_clifford_attention
        module = torch.nn.Module()
        x = torch.randn(2, 4, 12)
        out = execute_clifford_attention(module, x)
        self.assertEqual(out.shape, (2, 4, 12))

    def test_padic_residual_shape(self):
        """padic_residual preserves shape."""
        import torch
        from research.mathspaces.padic import execute_padic_residual
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_padic_residual(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_compound_ops_registered(self):
        """All 4 compound ops appear in the registry after registration."""
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import list_primitives, OpCategory
        register_all_mathspaces()
        math_ops = {op.name for op in list_primitives(OpCategory.MATH_SPACE)}
        for name in ["hyperbolic_norm", "tropical_gate",
                     "clifford_attention", "padic_residual"]:
            self.assertIn(name, math_ops, f"Compound op {name} not registered")

    def test_compound_ops_have_params(self):
        """Compound ops are registered with has_params=True."""
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import list_primitives, OpCategory
        register_all_mathspaces()
        math_ops = {op.name: op for op in list_primitives(OpCategory.MATH_SPACE)}
        for name in ["hyperbolic_norm", "tropical_gate",
                     "clifford_attention", "padic_residual"]:
            self.assertTrue(math_ops[name].has_params,
                            f"{name} should have has_params=True")


class TestAlternativeLearningRules(unittest.TestCase):
    """Test that all alternative learning rule optimizers work correctly."""

    def _make_simple_model(self):
        """Create a simple model for optimizer testing."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        return model

    def _run_optimizer_steps(self, optimizer_name, n_steps=5):
        """Run a few optimization steps and verify parameters change."""
        import torch
        from research.training.optimizer_synthesis import SynthesizedOptimizer

        model = self._make_simple_model()
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        opt = SynthesizedOptimizer(
            name=optimizer_name,
            components=[optimizer_name],
            lr=1e-3,
            weight_decay=0.01,
        ).create(model.parameters())

        x = torch.randn(4, 16)
        for _ in range(n_steps):
            out = model(x)
            loss = out.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Verify parameters changed
        changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, initial_params[name], atol=1e-7):
                changed = True
                break
        self.assertTrue(changed, f"{optimizer_name} did not update parameters")

    def test_hebbian_optimizer(self):
        self._run_optimizer_steps("hebbian")

    def test_forward_forward_optimizer(self):
        self._run_optimizer_steps("forward_forward")

    def test_perturbation_optimizer(self):
        self._run_optimizer_steps("perturbation")

    def test_contrastive_local_optimizer(self):
        self._run_optimizer_steps("contrastive_local")

    def test_all_recipes_instantiate(self):
        """Verify all OPTIMIZER_RECIPES can be instantiated."""
        from research.training.optimizer_synthesis import OPTIMIZER_RECIPES, SynthesizedOptimizer

        model = self._make_simple_model()
        for name, components, desc in OPTIMIZER_RECIPES:
            opt = SynthesizedOptimizer(
                name=name, components=components, lr=1e-3, weight_decay=0.01,
            ).create(model.parameters())
            self.assertIsNotNone(opt, f"Failed to create optimizer: {name}")

    def test_synthesize_optimizer_includes_new_recipes(self):
        """Verify new recipes appear in random synthesis."""
        from research.training.optimizer_synthesis import OPTIMIZER_RECIPES
        recipe_names = [r[0] for r in OPTIMIZER_RECIPES]
        for expected in ["hebbian", "forward_forward", "perturbation", "contrastive_local"]:
            self.assertIn(expected, recipe_names,
                          f"Missing recipe: {expected}")



@unittest.skipUnless(HAS_TORCH, "torch required")
class TestSpikingPrimitives(unittest.TestCase):
    """Tests for spiking/event-driven math space primitives."""

    def setUp(self):
        self.B, self.S, self.D = 2, 16, 32
        self.x = torch.randn(self.B, self.S, self.D, requires_grad=True)

    def _run_op(self, fn):
        # Harmonized signature: fn(module, *inputs)
        return fn(None, self.x)

    # Shape preservation
    def test_lif_shape(self):
        from research.mathspaces.spiking import execute_lif
        out = self._run_op(execute_lif)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_spike_rate_code_shape(self):
        from research.mathspaces.spiking import execute_spike_rate_code
        out = self._run_op(execute_spike_rate_code)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_stdp_attention_shape(self):
        from research.mathspaces.spiking import execute_stdp_attention
        out = self._run_op(execute_stdp_attention)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_sparse_threshold_shape(self):
        from research.mathspaces.spiking import execute_sparse_threshold
        out = self._run_op(execute_sparse_threshold)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    # Gradient flow
    def test_lif_gradient(self):
        from research.mathspaces.spiking import execute_lif
        out = self._run_op(execute_lif)
        out.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.abs().sum().item(), 0)

    def test_spike_rate_code_gradient(self):
        from research.mathspaces.spiking import execute_spike_rate_code
        out = self._run_op(execute_spike_rate_code)
        out.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.abs().sum().item(), 0)

    def test_sparse_threshold_gradient(self):
        from research.mathspaces.spiking import execute_sparse_threshold
        out = self._run_op(execute_sparse_threshold)
        out.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.abs().sum().item(), 0)

    # LIF output bounded
    def test_lif_output_bounded(self):
        from research.mathspaces.spiking import execute_lif
        out = self._run_op(execute_lif)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    # STDP causality: changing future tokens should not affect past output
    def test_stdp_causality(self):
        from research.mathspaces.spiking import execute_stdp_attention
        x_base = torch.randn(1, 8, 16)
        x_mod = x_base.clone().detach()
        x_mod[:, 6:, :] = torch.randn(1, 2, 16)  # Change last 2 tokens
        out1 = execute_stdp_attention(None, x_base)
        out2 = execute_stdp_attention(None, x_mod)
        # First 6 positions (0-5) attend only to themselves and earlier,
        # so they should be unaffected by changes at positions 6-7
        torch.testing.assert_close(out1[:, :6, :], out2[:, :6, :])

    # Sparse threshold promotes sparsity
    def test_sparse_threshold_sparsity(self):
        from research.mathspaces.spiking import execute_sparse_threshold
        x = torch.randn(4, 32, 64)
        out = execute_sparse_threshold(None, x)
        # At least 20% near-zero (threshold targets ~50%)
        near_zero = (out.abs() < 1e-6).float().mean().item()
        self.assertGreater(near_zero, 0.2)

    # Registry integration
    def test_spiking_ops_registered(self):
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        for name in ["lif_neuron", "spike_rate_code", "stdp_attention",
                      "sparse_threshold"]:
            self.assertIn(name, PRIMITIVE_REGISTRY,
                          f"Spiking op '{name}' not in PRIMITIVE_REGISTRY")

    def test_spiking_ops_identity_shape_rule(self):
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        for name in ["lif_neuron", "spike_rate_code", "stdp_attention",
                      "sparse_threshold"]:
            op = PRIMITIVE_REGISTRY[name]
            self.assertEqual(op.shape_rule, "identity")
        # stdp_attention has a learnable temporal decay parameter
        for name in ["lif_neuron", "spike_rate_code", "sparse_threshold"]:
            op = PRIMITIVE_REGISTRY[name]
            self.assertFalse(op.has_params, f"{name} should be parameter-free")

    def test_stdp_attention_gradient(self):
        from research.mathspaces.spiking import execute_stdp_attention
        out = self._run_op(execute_stdp_attention)
        out.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.abs().sum().item(), 0)



@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestCompressionPrimitives(unittest.TestCase):
    """Tests for weight compression math space primitives."""

    def setUp(self):
        self.B, self.S, self.D = 2, 16, 32
        self.x = torch.randn(self.B, self.S, self.D)

    # ── Shape preservation ──

    def test_low_rank_proj_shape(self):
        from research.mathspaces.compression import execute_low_rank_proj
        import torch.nn as nn
        module = nn.Module()
        r = self.D // 4
        module.U = nn.Parameter(torch.randn(self.D, r))
        module.V = nn.Parameter(torch.randn(r, self.D))
        out = execute_low_rank_proj(module, self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_grouped_linear_shape(self):
        from research.mathspaces.compression import execute_grouped_linear
        import torch.nn as nn
        module = nn.Module()
        g = 4
        gd = self.D // g
        module.weight = nn.Parameter(torch.randn(g, gd, gd))
        module.n_groups = g
        out = execute_grouped_linear(module, self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_bottleneck_proj_shape(self):
        from research.mathspaces.compression import execute_bottleneck_proj
        import torch.nn as nn
        module = nn.Module()
        r = self.D // 4
        module.down = nn.Parameter(torch.randn(r, self.D))
        module.up = nn.Parameter(torch.randn(self.D, r))
        out = execute_bottleneck_proj(module, self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_shared_basis_proj_shape(self):
        from research.mathspaces.compression import execute_shared_basis_proj
        import torch.nn as nn
        module = nn.Module()
        k = 8
        module.mixing = nn.Parameter(torch.randn(self.D, k))
        module.basis = nn.Parameter(torch.randn(k, self.D))
        out = execute_shared_basis_proj(module, self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    # ── Gradient flow ──

    def test_low_rank_proj_gradient(self):
        from research.mathspaces.compression import execute_low_rank_proj
        import torch.nn as nn
        x = torch.randn(self.B, self.S, self.D, requires_grad=True)
        module = nn.Module()
        r = self.D // 4
        module.U = nn.Parameter(torch.randn(self.D, r))
        module.V = nn.Parameter(torch.randn(r, self.D))
        out = execute_low_rank_proj(module, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_grouped_linear_gradient(self):
        from research.mathspaces.compression import execute_grouped_linear
        import torch.nn as nn
        x = torch.randn(self.B, self.S, self.D, requires_grad=True)
        module = nn.Module()
        g = 4
        gd = self.D // g
        module.weight = nn.Parameter(torch.randn(g, gd, gd))
        module.n_groups = g
        out = execute_grouped_linear(module, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_bottleneck_proj_gradient(self):
        from research.mathspaces.compression import execute_bottleneck_proj
        import torch.nn as nn
        x = torch.randn(self.B, self.S, self.D, requires_grad=True)
        module = nn.Module()
        r = self.D // 4
        module.down = nn.Parameter(torch.randn(r, self.D))
        module.up = nn.Parameter(torch.randn(self.D, r))
        out = execute_bottleneck_proj(module, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_shared_basis_proj_gradient(self):
        from research.mathspaces.compression import execute_shared_basis_proj
        import torch.nn as nn
        x = torch.randn(self.B, self.S, self.D, requires_grad=True)
        module = nn.Module()
        k = 8
        module.mixing = nn.Parameter(torch.randn(self.D, k))
        module.basis = nn.Parameter(torch.randn(k, self.D))
        out = execute_shared_basis_proj(module, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    # ── Registry integration ──

    def test_compression_ops_registered(self):
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        for name in ["low_rank_proj", "grouped_linear", "bottleneck_proj",
                      "shared_basis_proj"]:
            self.assertIn(name, PRIMITIVE_REGISTRY,
                          f"Compression op '{name}' not in PRIMITIVE_REGISTRY")
            op = PRIMITIVE_REGISTRY[name]
            self.assertTrue(op.has_params, f"'{name}' should have has_params=True")
            self.assertEqual(op.shape_rule, "identity")

    # ── Parameter count verification ──

    def test_low_rank_proj_param_count(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 64
        cop = CompiledOp("low_rank_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        n_params = sum(p.numel() for p in cop.parameters())
        expected = 2 * D * (D // 4)  # D²/2
        self.assertEqual(n_params, expected)
        self.assertLess(n_params, D * D)

    def test_grouped_linear_param_count(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 64
        cop = CompiledOp("grouped_linear", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        n_params = sum(p.numel() for p in cop.parameters())
        expected = 4 * (D // 4) ** 2  # D²/4
        self.assertEqual(n_params, expected)
        self.assertLess(n_params, D * D)

    def test_bottleneck_proj_param_count(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 64
        cop = CompiledOp("bottleneck_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        n_params = sum(p.numel() for p in cop.parameters())
        expected = 2 * D * (D // 4)  # D²/2
        self.assertEqual(n_params, expected)
        self.assertLess(n_params, D * D)

    def test_shared_basis_proj_param_count(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 64
        cop = CompiledOp("shared_basis_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        n_params = sum(p.numel() for p in cop.parameters())
        expected = 2 * 8 * D  # 16D
        self.assertEqual(n_params, expected)
        self.assertLess(n_params, D * D)

    # ── Compiler integration (end-to-end forward) ──

    def test_compiler_low_rank_forward(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 32
        cop = CompiledOp("low_rank_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)

    def test_compiler_grouped_forward(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 32
        cop = CompiledOp("grouped_linear", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)

    def test_compiler_bottleneck_forward(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 32
        cop = CompiledOp("bottleneck_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)

    def test_compiler_shared_basis_forward(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.mathspaces.registry import register_all_mathspaces
        register_all_mathspaces()
        D = 32
        cop = CompiledOp("shared_basis_proj", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestSparsePrimitives(unittest.TestCase):
    """Tests for sparse linear primitive families and sparse constraints."""

    def test_sparse_primitives_registered(self):
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
        for name in ["nm_sparse_linear", "block_sparse_linear", "semi_structured_2_4_linear"]:
            self.assertIn(name, PRIMITIVE_REGISTRY)
            op = PRIMITIVE_REGISTRY[name]
            self.assertTrue(op.has_params)

    def test_compiler_nm_sparse_linear_shape_and_grad(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        D = 32
        cop = CompiledOp("nm_sparse_linear", {"n": 2, "m": 4}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D, requires_grad=True)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_compiler_block_sparse_linear_shape_and_grad(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        D = 64
        cop = CompiledOp(
            "block_sparse_linear",
            {"block_size": 8, "block_density": 0.25},
            ShapeInfo(dim=D),
            ShapeInfo(dim=D),
            D,
        )
        x = torch.randn(2, 8, D, requires_grad=True)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_semi_structured_telemetry_records_kernel_fallback_on_cpu(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        D = 32
        cop = CompiledOp("semi_structured_2_4_linear", {}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)
        telemetry = getattr(cop, "sparse_telemetry", {})
        self.assertIn("semi_structured_2_4_linear", telemetry)
        stats = telemetry["semi_structured_2_4_linear"]
        self.assertGreaterEqual(stats.get("fallback_calls", 0), 1)

    def test_nm_sparse_invalid_config_falls_back_dense(self):
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        D = 32
        cop = CompiledOp("nm_sparse_linear", {"n": 5, "m": 4}, ShapeInfo(dim=D), ShapeInfo(dim=D), D)
        x = torch.randn(2, 8, D)
        out = cop(x)
        self.assertEqual(out.shape, x.shape)
        telemetry = getattr(cop, "sparse_telemetry", {})
        stats = telemetry.get("nm_sparse_linear", {})
        self.assertGreaterEqual(stats.get("fallback_calls", 0), 1)

    def test_sparse_weight_storage_constraints(self):
        from research.morphological_box import ArchSpec, is_valid_spec
        base = {
            "token_representation": "dense_float",
            "weight_storage": "structured_sparse",
            "token_mixing": "softmax_attention",
            "channel_mixing": "swiglu_mlp",
            "compute_routing": "uniform",
            "topology": "sequential",
            "normalization": "rmsnorm_pre",
            "positional_encoding": "rope",
        }
        valid, reason = is_valid_spec(ArchSpec(choices=base, seed=1))
        self.assertTrue(valid, reason)

        bad_dense_net = dict(base)
        bad_dense_net["topology"] = "dense_net"
        valid, reason = is_valid_spec(ArchSpec(choices=bad_dense_net, seed=2))
        self.assertFalse(valid)
        self.assertIn("dense_net", reason)

        bad_no_norm = dict(base)
        bad_no_norm["weight_storage"] = "block_sparse"
        bad_no_norm["normalization"] = "no_norm"
        valid, reason = is_valid_spec(ArchSpec(choices=bad_no_norm, seed=3))
        self.assertFalse(valid)
        self.assertIn("block-sparse", reason)

        bad_token = dict(base)
        bad_token["weight_storage"] = "semi_structured_2_4"
        bad_token["token_representation"] = "binary_hash"
        valid, reason = is_valid_spec(ArchSpec(choices=bad_token, seed=4))
        self.assertFalse(valid)
        self.assertIn("dense_float", reason)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestSparseTelemetryPersistence(unittest.TestCase):
    """Sparse telemetry extraction and notebook persistence schema tests."""

    def test_runner_sparse_telemetry_aggregation(self):
        from research.scientist.runner import ExperimentRunner

        class DummyOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(16, 16))
                self.sparse_telemetry = {
                    "nm_sparse_linear": {
                        "calls": 4,
                        "fallback_calls": 1,
                        "density_sum": 2.0,
                        "last_density": 0.5,
                        "last_fallback_reason": "invalid_nm_configuration",
                    },
                    "semi_structured_2_4_linear": {
                        "calls": 2,
                        "fallback_calls": 2,
                        "density_sum": 2.0,
                        "last_density": 1.0,
                        "last_fallback_reason": "kernel_unavailable",
                    },
                }

        class DummyLayer:
            def __init__(self):
                self.ops = {"1": DummyOp()}

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [DummyLayer()]

        runner = ExperimentRunner.__new__(ExperimentRunner)
        metrics = runner._extract_architecture_telemetry(DummyModel())
        self.assertIn("sparse_density_mean", metrics)
        self.assertIn("sparse_fallback_calls", metrics)
        self.assertEqual(metrics["sparse_fallback_calls"], 3)
        self.assertEqual(metrics["sparse_kernel_fallback_calls"], 2)
        self.assertIn("sparse_telemetry_json", metrics)
        self.assertGreater(len(json.loads(metrics["sparse_telemetry_json"])), 0)

    def test_notebook_schema_has_sparse_telemetry_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sparse_metrics.db")
            nb = LabNotebook(db_path)
            try:
                cols = {
                    row[1]
                    for row in nb.conn.execute("PRAGMA table_info(program_results)").fetchall()
                }
                for col in [
                    "sparse_density_mean",
                    "sparse_density_last",
                    "sparse_fallback_calls",
                    "sparse_kernel_fallback_calls",
                    "sparse_nm_compliance",
                    "sparse_active_params_estimate",
                    "sparse_telemetry_json",
                    "pruning_method",
                    "pruning_target_sparsity",
                    "pruning_actual_sparsity",
                    "pruning_n_params_total",
                    "pruning_n_params_pruned",
                    "pruning_dense_eval_loss",
                    "pruning_pruned_eval_loss",
                    "pruning_quality_retention",
                    "pruning_active_params_estimate",
                    "pruning_error",
                ]:
                    self.assertIn(col, cols)
            finally:
                nb.close()


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestOneShotPruningBaseline(unittest.TestCase):
    def test_apply_one_shot_pruning_hits_target_range(self):
        from research.eval.pruning import apply_one_shot_pruning

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
        )
        result = apply_one_shot_pruning(model, target_sparsity=0.5, method="wanda")
        self.assertGreater(result.n_params_total, 0)
        self.assertGreater(result.n_params_pruned, 0)
        self.assertGreater(result.actual_sparsity, 0.35)
        self.assertLess(result.actual_sparsity, 0.65)

    def test_micro_train_emits_pruning_metrics_when_enabled(self):
        from research.scientist.runner import ExperimentRunner, RunConfig

        class TinyLM(torch.nn.Module):
            def __init__(self, vocab_size=64, dim=32):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, dim)
                self.proj = torch.nn.Linear(dim, vocab_size, bias=False)

            def forward(self, input_ids):
                return self.proj(self.embed(input_ids))

        runner = ExperimentRunner.__new__(ExperimentRunner)

        class _Stop:
            def is_set(self):
                return False

        runner._stop_event = _Stop()
        runner._corpus_batcher = None
        runner._corpus_signature = None
        runner._corpus_warned_unavailable = False
        runner._hydra_loader = None

        cfg = RunConfig(
            vocab_size=64,
            stage1_steps=3,
            stage1_batch_size=2,
            max_seq_len=32,
            one_shot_pruning_baseline=True,
            one_shot_pruning_sparsity=0.5,
            one_shot_pruning_eval_batches=2,
            one_shot_pruning_batch_size=2,
        )
        model = TinyLM(vocab_size=64, dim=32)
        dev = torch.device("cpu")
        out = runner._micro_train(model, cfg, dev, seed=123)
        self.assertIn("pruning_method", out)
        self.assertIn("pruning_actual_sparsity", out)
        self.assertIn("pruning_quality_retention", out)
        self.assertGreaterEqual(out.get("pruning_actual_sparsity", 0.0), 0.0)


class TestQuantizationUtils(unittest.TestCase):
    """Tests for fake-quantization and sparse+quant co-design utilities."""

    def test_fake_quantize_tensor_int8(self):
        import torch
        from research.eval.quantization import fake_quantize_tensor

        t = torch.randn(16, 16)
        q = fake_quantize_tensor(t, bits=8)
        self.assertEqual(q.shape, t.shape)
        # Quantized values should be close but not identical
        self.assertFalse(torch.equal(t, q))
        # Error should be small for INT8
        self.assertLess((t - q).abs().max().item(), t.abs().max().item() * 0.02)

    def test_fake_quantize_tensor_int4(self):
        import torch
        from research.eval.quantization import fake_quantize_tensor

        t = torch.randn(16, 16)
        q4 = fake_quantize_tensor(t, bits=4)
        q8 = fake_quantize_tensor(t, bits=8)
        # INT4 should have larger quantization error than INT8
        err4 = (t - q4).abs().mean().item()
        err8 = (t - q8).abs().mean().item()
        self.assertGreater(err4, err8)

    def test_fake_quantize_tensor_fp16_passthrough(self):
        import torch
        from research.eval.quantization import fake_quantize_tensor

        t = torch.randn(8, 8)
        q = fake_quantize_tensor(t, bits=16)
        self.assertTrue(torch.equal(t, q))

    def test_fake_quantize_zero_tensor(self):
        import torch
        from research.eval.quantization import fake_quantize_tensor

        t = torch.zeros(4, 4)
        q = fake_quantize_tensor(t, bits=8)
        self.assertTrue(torch.equal(t, q))

    def test_apply_fake_quantization(self):
        import torch
        import torch.nn as nn
        from research.eval.quantization import apply_fake_quantization

        model = nn.Linear(32, 32)
        original_weight = model.weight.data.clone()
        result = apply_fake_quantization(model, bits=8)
        self.assertEqual(result.bits, 8)
        self.assertGreater(result.n_params_total, 0)
        self.assertEqual(result.n_params_quantized, result.n_params_total)
        # Weight should have changed
        self.assertFalse(torch.equal(original_weight, model.weight.data))

    def test_apply_fake_quantization_preserves_zeros(self):
        """Fake quant should not revive pruned (zero) weights."""
        import torch
        import torch.nn as nn
        from research.eval.quantization import apply_fake_quantization

        model = nn.Linear(32, 32, bias=False)
        # Prune half the weights
        with torch.no_grad():
            mask = torch.ones_like(model.weight)
            mask[:16, :] = 0.0
            model.weight.mul_(mask)
        zeros_before = (model.weight.data == 0).sum().item()

        result = apply_fake_quantization(model, bits=8)
        # Zeros stay zero; quantization may also round small values to zero
        zeros_after = (model.weight.data == 0).sum().item()
        self.assertGreaterEqual(zeros_after, zeros_before)
        self.assertGreater(result.actual_sparsity, 0.0)

    def test_fake_quant_result_to_dict(self):
        from research.eval.quantization import FakeQuantResult

        r = FakeQuantResult(
            bits=8, target_sparsity=0.5, actual_sparsity=0.5,
            n_params_total=1000, n_params_quantized=1000,
            bytes_per_param_original=4.0, bytes_per_param_effective=0.5,
        )
        d = r.to_dict()
        self.assertIn("bits", d)
        self.assertIn("bytes_per_param_effective", d)

    def test_sparse_quant_codesign_summary_empty(self):
        """Analytics method should return empty summary when no sparse/quant data."""
        from research.scientist.analytics import ExperimentAnalytics
        from research.scientist.notebook import LabNotebook

        nb = LabNotebook(db_path=":memory:")
        analytics = ExperimentAnalytics(nb)
        result = analytics.sparse_quant_codesign_summary()
        self.assertEqual(result["n_programs"], 0)
        self.assertEqual(result["programs"], [])




if __name__ == '__main__':
    unittest.main()
