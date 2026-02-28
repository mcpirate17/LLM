"""
Math Space Registry

Registers all mathematical space operations as primitives
available to the synthesis engine.
"""

from __future__ import annotations

from ..synthesis.primitives import (
    PrimitiveOp, OpCategory, register_external_primitive,
)
from . import hyperbolic, tropical, padic, clifford, compression, spiking
from . import tropical_routing


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

    # ── Compound Cross-Space Primitives ──

    op = PrimitiveOp(
        name="hyperbolic_norm",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D+D",
        description="Manifold-aware normalization: log-map → LayerNorm → exp-map",
        numerically_risky=True,
    )
    op = _with_execute(op, hyperbolic.execute_hyperbolic_norm)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_gate",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D",
        description="Shortest-path tropical distances as a gating mechanism",
    )
    op = _with_execute(op, tropical.execute_tropical_gate)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_router",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D//4",
        description="Tropical (shortest-path) routing as a gating signal",
    )
    op = _with_execute(op, tropical_routing.execute_tropical_router)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_moe",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D*4",
        description="Full Mixture-of-Experts with tropical (shortest-path) routing",
    )
    op = _with_execute(op, tropical_routing.execute_tropical_moe)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="clifford_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D",
        description="Attention via geometric product (dot + wedge) for richer token scores",
    )
    op = _with_execute(op, clifford.execute_clifford_attention)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="padic_residual",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*2*D",
        description="Multi-resolution p-adic expansion with per-scale transform + residual",
    )
    op = _with_execute(op, padic.execute_padic_residual)
    register_external_primitive(op)

    # ── Weight Compression ──
    op = PrimitiveOp(
        name="low_rank_proj",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D//2",
        description="Low-rank factored linear (rank=D/4)",
    )
    op = _with_execute(op, compression.execute_low_rank_proj)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="grouped_linear",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D//4",
        description="Block-diagonal linear (4 groups)",
    )
    op = _with_execute(op, compression.execute_grouped_linear)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="bottleneck_proj",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D//2",
        description="Squeeze-expand bottleneck (D→D/4→D)",
    )
    op = _with_execute(op, compression.execute_bottleneck_proj)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="shared_basis_proj",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*16",
        description="Shared-basis projection (8 basis vectors)",
    )
    op = _with_execute(op, compression.execute_shared_basis_proj)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tied_proj",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D//4",
        description="Tied down/up projection (shared transposed weights, rank=D/4)",
    )
    op = _with_execute(op, compression.execute_tied_proj)
    register_external_primitive(op)

    # ── Spiking / Event-Driven ──
    op = PrimitiveOp(
        name="lif_neuron",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Leaky Integrate-and-Fire neuron with surrogate gradient",
    )
    op = _with_execute(op, spiking.execute_lif)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="spike_rate_code",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Continuous-to-spike-to-continuous rate coding with STE",
    )
    op = _with_execute(op, spiking.execute_spike_rate_code)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="stdp_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="STDP-inspired causal attention with temporal decay kernel",
    )
    op = _with_execute(op, spiking.execute_stdp_attention)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="sparse_threshold",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Adaptive median-based threshold gate (~50% sparsity)",
    )
    op = _with_execute(op, spiking.execute_sparse_threshold)
    register_external_primitive(op)


def _with_execute(op: PrimitiveOp, fn) -> PrimitiveOp:
    """Attach an execution function to a PrimitiveOp.

    Since PrimitiveOp is frozen, we store it as an attribute on the object.
    """
    # Use object.__setattr__ to bypass frozen dataclass
    object.__setattr__(op, 'execute_fn', fn)
    return op
