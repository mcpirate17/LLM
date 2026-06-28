"""
Math Space Registry

Registers all mathematical space operations as primitives
available to the synthesis engine.

Each op is a ``(execute_fn, kwargs)`` spec in ``_MATH_SPACE_SPECS``;
``register_all_mathspaces`` builds a ``PrimitiveOp`` (category MATH_SPACE) per
spec, attaches its execute fn, and registers it. Add an op by adding a spec —
not by repeating the build/attach/register boilerplate.
"""

from __future__ import annotations

from typing import Any, Callable

from ..synthesis.primitives import (
    PrimitiveOp,
    OpCategory,
    register_external_primitive,
)
from . import (
    hyperbolic,
    tropical,
    padic,
    clifford,
    compression,
    spiking,
    projective,
    cawn,
)
from . import tropical_routing
from . import tree_mix as _tree_mix_mod
from . import mla as _mla_mod
from . import pq_embedding as _pq_emb_mod
from . import mlstm as _mlstm_mod

# (execute_fn, PrimitiveOp kwargs). ``category=OpCategory.MATH_SPACE`` is applied
# uniformly in register_all_mathspaces and omitted here.
_MATH_SPACE_SPECS: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
    # ── Hyperbolic ──
    (
        hyperbolic.execute_poincare_add,
        dict(
            name="poincare_add",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D",
            description="Mobius addition with learnable bias in Poincare ball",
            algebraic_space="poincare",
        ),
    ),
    (
        hyperbolic.execute_exp_map,
        dict(
            name="exp_map",
            n_inputs=1,
            shape_rule="identity",
            description="Exponential map: Euclidean -> Poincare ball",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    (
        hyperbolic.execute_log_map,
        dict(
            name="log_map",
            n_inputs=1,
            shape_rule="identity",
            description="Logarithmic map: Poincare ball -> Euclidean",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    (
        hyperbolic.execute_hyp_linear,
        dict(
            name="hyp_linear",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D",
            description="Linear transformation in hyperbolic space",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    (
        hyperbolic.execute_hyp_distance,
        dict(
            name="hyp_distance",
            n_inputs=2,
            shape_rule="reduce_last",
            description="Hyperbolic distance between two points in the Poincare ball",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    (
        hyperbolic.execute_hyp_tangent_nonlinear,
        dict(
            name="hyp_tangent_nonlinear",
            n_inputs=1,
            shape_rule="identity",
            description="Tangent-space nonlinearity with exp/log manifold mapping",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    # ── Tropical ──
    (
        tropical.execute_tropical_matmul,
        dict(
            name="tropical_matmul",
            n_inputs=2,
            shape_rule="binary_broadcast",
            description="Tropical (min-plus) matrix multiply — shortest path distances",
            algebraic_space="tropical",
        ),
    ),
    (
        tropical.execute_tropical_add,
        dict(
            name="tropical_add",
            n_inputs=2,
            shape_rule="binary_broadcast",
            description="Tropical addition (element-wise minimum)",
            algebraic_space="tropical",
        ),
    ),
    (
        tropical.execute_tropical_attention,
        dict(
            name="tropical_attention",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D",
            description="Self-attention using tropical geometry (shortest-path)",
            algebraic_space="tropical",
            binding_range_class="full",
        ),
    ),
    (
        tropical.execute_tropical_center,
        dict(
            name="tropical_center",
            n_inputs=1,
            shape_rule="identity",
            description="Subtract sequence-wise tropical baseline (min) from features",
            algebraic_space="tropical",
        ),
    ),
    # Phase 5b-bis (2026-05-10): temperature-scaled softmin (LogSumExp over -x/tau)
    # — gradient-friendly replacement for hard tropical max; grammar can swap it in.
    (
        tropical.execute_tropical_softmax,
        dict(
            name="tropical_softmax",
            n_inputs=1,
            shape_rule="identity",
            description="Gradient-friendly softmin (softmax over -x/tau) — drop-in for vanilla softmax in tropical/shortest-path contexts",
            algebraic_space="tropical",
        ),
    ),
    # Phase 5c (2026-05-11): binary-tree feature mixer (research §2.1 leafed layers).
    (
        _tree_mix_mod.execute_tree_mix,
        dict(
            name="tree_mix",
            n_inputs=2,
            shape_rule="identity",
            has_params=True,
            param_formula="D",
            description="Atomic binary mixer node: z = sigmoid(W) * x + (1 - sigmoid(W)) * y. Templates compose 2^K - 1 nodes to build a balanced binary tree of depth K (research §2.1 leafed layers).",
        ),
    ),
    # Phase 5c (2026-05-11): Multi-Head Latent Attention (research §1.1). K/V share a
    # low-rank latent down-proj, distinct up-projs; ~93% KV-cache cut at d_latent=D/8.
    (
        _mla_mod.execute_mla_attention,
        dict(
            name="mla_attention",
            n_inputs=2,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D*3/8",
            description="Multi-head latent attention: K and V share a low-rank latent path (d_latent=D/8 by default), reconstructed via distinct up-projections, then standard softmax attention on Q. KV-cache compression.",
            binding_range_class="full",
        ),
    ),
    # Phase 5d (2026-05-11): Product-Quantized Embedding (research §2.3).
    (
        _pq_emb_mod.execute_pq_embedding,
        dict(
            name="pq_embedding",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="M*K*D/M",
            description="Product-quantized embedding: split D into M subspaces of D/M, each with K learnable codebook centroids; per-subspace softmax assignment then weighted reconstruction. Compression-style learned quantization.",
        ),
    ),
    # Phase 5e (2026-05-11): Matrix-memory LSTM cell (xLSTM/mLSTM, research §1.5).
    (
        _mlstm_mod.execute_mlstm_cell,
        dict(
            name="mlstm_cell",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="4*D*D + 2*D",
            description="Matrix-memory LSTM cell: recurrent state is a (D, D) outer-product accumulator addressed by per-token queries. Per-token compute is (4*D + 2)·D multiply-add; backward via autograd.",
            binding_range_class="full",
        ),
    ),
    # ── p-adic ──
    (
        padic.execute_padic_expand,
        dict(
            name="padic_expand",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*2*D",
            description="Multi-scale p-adic expansion and projection",
            algebraic_space="padic",
        ),
    ),
    (
        padic.execute_ultrametric_attn,
        dict(
            name="ultrametric_attention",
            n_inputs=1,
            shape_rule="identity",
            description="Attention using ultrametric (p-adic) distance",
            algebraic_space="padic",
            binding_range_class="full",
        ),
    ),
    (
        padic.execute_padic_gate,
        dict(
            name="padic_gate",
            n_inputs=1,
            shape_rule="identity",
            description="Gate activations by smooth p-adic valuation strength",
            algebraic_space="padic",
        ),
    ),
    # ── Clifford ──
    (
        clifford.execute_geometric_product,
        dict(
            name="geometric_product",
            n_inputs=2,
            shape_rule="binary_broadcast",
            description="Clifford geometric product (dot + wedge)",
            algebraic_space="clifford",
        ),
    ),
    (
        clifford.execute_rotor_transform,
        dict(
            name="rotor_transform",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="8",
            description="Clifford rotor transformation (efficient rotation)",
            algebraic_space="clifford",
        ),
    ),
    (
        clifford.execute_grade_select,
        dict(
            name="grade_select",
            n_inputs=1,
            shape_rule="identity",
            description="Select vector grade from Clifford multivector",
            algebraic_space="clifford",
        ),
    ),
    (
        clifford.execute_grade_mix,
        dict(
            name="grade_mix",
            n_inputs=1,
            shape_rule="identity",
            description="Blend vector and bivector grades in Clifford multivectors",
            algebraic_space="clifford",
        ),
    ),
    # Phase 5 V2 (2026-05-04): Clifford companion ops (novel_math_ops_proposal §3.1).
    (
        clifford.execute_clifford_inverse,
        dict(
            name="clifford_inverse",
            n_inputs=1,
            shape_rule="identity",
            description="Multiplicative inverse of a Cl(3,0) multivector "
            "(~mv / ||mv||²); exact for versors, regularized fallback otherwise",
            algebraic_space="clifford",
        ),
    ),
    (
        clifford.execute_versor_apply,
        dict(
            name="versor_apply",
            n_inputs=2,
            shape_rule="binary_broadcast",
            description="Versor sandwich product v · mv · v⁻¹ — canonical "
            "Clifford rotation; generalizes rotor_transform to non-unit versors",
            algebraic_space="clifford",
        ),
    ),
    # ── Compound Cross-Space Primitives ──
    (
        hyperbolic.execute_hyperbolic_norm,
        dict(
            name="hyperbolic_norm",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D+D",
            description="Manifold-aware normalization: log-map → LayerNorm → exp-map",
            numerically_risky=True,
            algebraic_space="poincare",
        ),
    ),
    (
        tropical.execute_tropical_gate,
        dict(
            name="tropical_gate",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D",
            description="Shortest-path tropical distances as a gating mechanism",
            algebraic_space="tropical",
        ),
    ),
    (
        tropical_routing.execute_tropical_router,
        dict(
            name="tropical_router",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D//4",
            description="Tropical (shortest-path) routing as a gating signal",
            algebraic_space="tropical",
        ),
    ),
    (
        tropical_routing.execute_tropical_moe,
        dict(
            name="tropical_moe",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D*4",
            description="Full Mixture-of-Experts with tropical (shortest-path) routing",
            algebraic_space="tropical",
        ),
    ),
    (
        clifford.execute_clifford_attention,
        dict(
            name="clifford_attention",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D",
            description="Attention via geometric product (dot + wedge) for richer token scores",
            algebraic_space="clifford",
            binding_range_class="full",
        ),
    ),
    (
        padic.execute_padic_residual,
        dict(
            name="padic_residual",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*2*D",
            description="Multi-resolution p-adic expansion with per-scale transform + residual",
            algebraic_space="padic",
        ),
    ),
    # ── Weight Compression ──
    (
        compression.execute_low_rank_proj,
        dict(
            name="low_rank_proj",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D//2",
            description="Low-rank factored linear (rank=D/4)",
        ),
    ),
    (
        compression.execute_grouped_linear,
        dict(
            name="grouped_linear",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D//4",
            description="Block-diagonal linear (4 groups)",
        ),
    ),
    (
        compression.execute_bottleneck_proj,
        dict(
            name="bottleneck_proj",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D//2",
            description="Squeeze-expand bottleneck (D→D/4→D)",
        ),
    ),
    (
        compression.execute_shared_basis_proj,
        dict(
            name="shared_basis_proj",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*16",
            description="Shared-basis projection (8 basis vectors)",
        ),
    ),
    (
        compression.execute_tied_proj,
        dict(
            name="tied_proj",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="D*D//4",
            description="Tied down/up projection (shared transposed weights, rank=D/4)",
        ),
    ),
    # ── Spiking / Event-Driven ──
    (
        spiking.execute_lif,
        dict(
            name="lif_neuron",
            n_inputs=1,
            shape_rule="identity",
            description="Leaky Integrate-and-Fire neuron with surrogate gradient",
            algebraic_space="spiking",
        ),
    ),
    (
        spiking.execute_spike_rate_code,
        dict(
            name="spike_rate_code",
            n_inputs=1,
            shape_rule="identity",
            description="Continuous-to-spike-to-continuous rate coding with STE",
            algebraic_space="spiking",
        ),
    ),
    (
        spiking.execute_stdp_attention,
        dict(
            name="stdp_attention",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="1",
            description="STDP-inspired causal attention with learnable temporal decay",
            algebraic_space="spiking",
            binding_range_class="full",
        ),
    ),
    (
        spiking.execute_sparse_threshold,
        dict(
            name="sparse_threshold",
            n_inputs=1,
            shape_rule="identity",
            description="Adaptive median-based threshold gate (~50% sparsity)",
            algebraic_space="spiking",
        ),
    ),
    # ── Projective Geometry ──
    (
        projective.execute_projective_linear,
        dict(
            name="projective_linear",
            n_inputs=1,
            shape_rule="identity",
            has_params=True,
            param_formula="(D+1)*(D+1)",
            description="Linear transformation (homography) in projective space",
            algebraic_space="projective",
        ),
    ),
    (
        projective.execute_projective_attention,
        dict(
            name="projective_attention",
            n_inputs=1,
            shape_rule="identity",
            description="Self-attention using projective angular (cosine) distances",
            algebraic_space="projective",
            binding_range_class="full",
        ),
    ),
    # ── Continuous Acoustic Wave Network (CAWN) ──
    (
        cawn.execute_cawn_mixer,
        dict(
            name="cawn_mixer",
            n_inputs=1,
            shape_rule="identity",
            description="Continuous complex-domain phase accumulation sequence mixer",
            algebraic_space="complex",
            binding_range_class="full",
        ),
    ),
)


def _with_execute(op: PrimitiveOp, fn) -> PrimitiveOp:
    """Attach an execution function to a frozen PrimitiveOp (bypass frozen-ness)."""
    object.__setattr__(op, "execute_fn", fn)
    return op


def register_all_mathspaces():
    """Register all math space primitives with the synthesis engine."""
    for execute_fn, kwargs in _MATH_SPACE_SPECS:
        op = PrimitiveOp(category=OpCategory.MATH_SPACE, **kwargs)
        register_external_primitive(_with_execute(op, execute_fn))
