"""
Package wiring and import structure tests.

Verifies that package __init__.py files export expected modules,
math-space registrations work, and dead-code audit tooling runs.
"""

import unittest

import pytest

pytestmark = pytest.mark.unit

# Detect available dependencies -- lazy import to reduce memory in parallel runs
try:
    import torch

    HAS_TORCH = True
    del torch  # noqa: E702
except ImportError:
    HAS_TORCH = False


class TestPackageWiring(unittest.TestCase):
    """Ensure explicitly connected package modules remain importable."""

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
        from research.synthesis.compiler import CompiledOp
        from research.synthesis.graph import ShapeInfo
        from research.synthesis.primitives import (
            PrimitiveOp,
            OpCategory,
            PRIMITIVE_REGISTRY,
            register_external_primitive,
        )

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
        x = torch.ones(2, 3, 4)
        try:
            compiled_op = CompiledOp(op_name, {}, ShapeInfo(dim=4), ShapeInfo(dim=4), 4)
            compiled_op.collect_telemetry = True
            out = compiled_op(x)
            self.assertTrue(torch.isfinite(out).all())
            telemetry = getattr(compiled_op, "mathspace_telemetry", {})
            self.assertIn(op_name, telemetry)
            self.assertGreaterEqual(telemetry[op_name]["calls"], 1)
            self.assertGreater(telemetry[op_name]["nonfinite"], 0)
        finally:
            PRIMITIVE_REGISTRY.pop(op_name, None)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_registered(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        for op_name in (
            "hyp_tangent_nonlinear",
            "tropical_center",
            "padic_gate",
            "grade_mix",
        ):
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
        for op_name in (
            "hyp_tangent_nonlinear",
            "tropical_center",
            "padic_gate",
            "grade_mix",
        ):
            op = PRIMITIVE_REGISTRY[op_name]
            out = op.execute_fn(module, x)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(
                torch.isfinite(out).all(), f"{op_name} produced non-finite values"
            )


if __name__ == "__main__":
    unittest.main()
