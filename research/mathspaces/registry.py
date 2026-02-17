"""
Math Space Registry

Registers all mathematical space operations as primitives
available to the synthesis engine.
"""

from __future__ import annotations

from ..synthesis.primitives import (
    PrimitiveOp, OpCategory, register_external_primitive,
)
from . import hyperbolic, tropical, padic, clifford


def register_all_mathspaces():
    """Register all math space primitives with the synthesis engine."""

    # ── Hyperbolic ──
    op = PrimitiveOp(
        name="poincare_add",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D",
        description="Mobius addition with learnable bias in Poincare ball",
    )
    op = _with_execute(op, hyperbolic.execute_poincare_add)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="exp_map",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Exponential map: Euclidean -> Poincare ball",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_exp_map)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="log_map",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Logarithmic map: Poincare ball -> Euclidean",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_log_map)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="hyp_linear",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D",
        description="Linear transformation in hyperbolic space",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_hyp_linear)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="hyp_distance",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="reduce_last",
        description="Hyperbolic distance between two points in the Poincare ball",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_hyp_distance)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="hyp_tangent_nonlinear",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Tangent-space nonlinearity with exp/log manifold mapping",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_hyp_tangent_nonlinear)
    register_external_primitive(op)

    # ── Tropical ──
    op = PrimitiveOp(
        name="tropical_matmul",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="binary_broadcast",
        description="Tropical (min-plus) matrix multiply — shortest path distances",
    )
    op = _with_execute(op, tropical.execute_tropical_matmul)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_add",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="binary_broadcast",
        description="Tropical addition (element-wise minimum)",
    )
    op = _with_execute(op, tropical.execute_tropical_add)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D",
        description="Self-attention using tropical geometry (shortest-path)",
    )
    op = _with_execute(op, tropical.execute_tropical_attention)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_center",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Subtract sequence-wise tropical baseline (min) from features",
    )
    op = _with_execute(op, tropical.execute_tropical_center)
    register_external_primitive(op)

    # ── p-adic ──
    op = PrimitiveOp(
        name="padic_expand",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*2*D",
        description="Multi-scale p-adic expansion and projection",
    )
    op = _with_execute(op, padic.execute_padic_expand)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="ultrametric_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Attention using ultrametric (p-adic) distance",
    )
    op = _with_execute(op, padic.execute_ultrametric_attn)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="padic_gate",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Gate activations by smooth p-adic valuation strength",
    )
    op = _with_execute(op, padic.execute_padic_gate)
    register_external_primitive(op)

    # ── Clifford ──
    op = PrimitiveOp(
        name="geometric_product",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="binary_broadcast",
        description="Clifford geometric product (dot + wedge)",
    )
    op = _with_execute(op, clifford.execute_geometric_product)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="rotor_transform",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="8",
        description="Clifford rotor transformation (efficient rotation)",
    )
    op = _with_execute(op, clifford.execute_rotor_transform)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="grade_select",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Select vector grade from Clifford multivector",
    )
    op = _with_execute(op, clifford.execute_grade_select)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="grade_mix",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Blend vector and bivector grades in Clifford multivectors",
    )
    op = _with_execute(op, clifford.execute_grade_mix)
    register_external_primitive(op)


def _with_execute(op: PrimitiveOp, fn) -> PrimitiveOp:
    """Attach an execution function to a PrimitiveOp.

    Since PrimitiveOp is frozen, we store it as an attribute on the object.
    """
    # Use object.__setattr__ to bypass frozen dataclass
    object.__setattr__(op, 'execute_fn', fn)
    return op
