"""
Package wiring and import structure tests.

Verifies that package __init__.py files export expected modules,
math-space registrations work, and dead-code audit tooling runs.
"""

import json
import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.unit

# Detect available dependencies -- lazy import to reduce memory in parallel runs
try:
    import torch; HAS_TORCH = True; del torch  # noqa: E702
except ImportError:
    HAS_TORCH = False


class TestPackageWiring(unittest.TestCase):
    """Ensure explicitly connected package modules remain importable."""

    def test_mathspaces_exports_modules(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "mathspaces", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import clifford, compression, hyperbolic, padic, spiking, tropical", content)
        self.assertIn("from .registry import register_all_mathspaces", content)
        self.assertIn('"hyperbolic"', content)
        self.assertIn('"tropical"', content)
        self.assertIn('"padic"', content)
        self.assertIn('"clifford"', content)
        self.assertIn('"spiking"', content)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_registry_includes_hyp_distance(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        self.assertIn("hyp_distance", PRIMITIVE_REGISTRY)
        op = PRIMITIVE_REGISTRY["hyp_distance"]
        self.assertEqual(op.category.value, "math_space")
        self.assertEqual(op.n_inputs, 2)
        self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_external_op_nonfinite_sanitization_and_telemetry(self):
        import torch
        from research.synthesis.compiler import _execute_op
        from research.synthesis.primitives import PrimitiveOp, OpCategory, PRIMITIVE_REGISTRY, register_external_primitive

        op_name = "test_nonfinite_mathspace_op"
        op = PrimitiveOp(
            name=op_name,
            category=OpCategory.MATH_SPACE,
            n_inputs=1,
            shape_rule="identity",
            description="test external op",
        )

        def _execute_fn(module, x):
            return x / 0.0

        object.__setattr__(op, "execute_fn", _execute_fn)
        register_external_primitive(op)
        module = torch.nn.Module()
        x = torch.ones(2, 3, 4)
        try:
            out = _execute_op(module, op_name, (x,), {})
            self.assertTrue(torch.isfinite(out).all())
            telemetry = getattr(module, "mathspace_telemetry", {})
            self.assertIn(op_name, telemetry)
            self.assertGreaterEqual(telemetry[op_name]["calls"], 1)
            self.assertGreater(telemetry[op_name]["nonfinite_elements"], 0)
            self.assertGreaterEqual(telemetry[op_name]["sanitized_calls"], 1)
        finally:
            PRIMITIVE_REGISTRY.pop(op_name, None)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_registered(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            self.assertIn(op_name, PRIMITIVE_REGISTRY)
            op = PRIMITIVE_REGISTRY[op_name]
            self.assertEqual(op.category.value, "math_space")
            self.assertEqual(op.n_inputs, 1)
            self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_execute_shape_and_finite(self):
        import torch
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        x = torch.randn(2, 5, 16)
        module = torch.nn.Module()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            op = PRIMITIVE_REGISTRY[op_name]
            out = op.execute_fn(module, x)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(torch.isfinite(out).all(), f"{op_name} produced non-finite values")

    def test_llm_package_exports_context_and_prompts(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "scientist", "llm", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import context, prompts", content)
        self.assertIn("from .backend import", content)
        self.assertIn('"context"', content)
        self.assertIn('"prompts"', content)


if __name__ == "__main__":
    unittest.main()
