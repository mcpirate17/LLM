"""
Math Space Registry

Registers all mathematical space operations as primitives
available to the synthesis engine.
"""

from __future__ import annotations

from ..synthesis.primitives import (
    PrimitiveOp,
    OpCategory,
    register_external_primitive,
)
from . import hyperbolic, tropical, padic, clifford, compression, spiking
from . import tropical_routing
from . import tree_mix as _tree_mix_mod  # noqa: F401 — used in register_all_mathspaces
from . import mla as _mla_mod  # noqa: F401 — used in register_all_mathspaces
from . import pq_embedding as _pq_emb_mod  # noqa: F401 — used in register_all_mathspaces
from . import mlstm as _mlstm_mod  # noqa: F401 — used in register_all_mathspaces


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
        algebraic_space="poincare",
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
        algebraic_space="poincare",
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
        algebraic_space="poincare",
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
        algebraic_space="poincare",
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
        algebraic_space="poincare",
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
        algebraic_space="poincare",
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
        algebraic_space="tropical",
    )
    op = _with_execute(op, tropical.execute_tropical_matmul)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_add",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="binary_broadcast",
        description="Tropical addition (element-wise minimum)",
        algebraic_space="tropical",
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
        algebraic_space="tropical",
    )
    op = _with_execute(op, tropical.execute_tropical_attention)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="tropical_center",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Subtract sequence-wise tropical baseline (min) from features",
        algebraic_space="tropical",
    )
    op = _with_execute(op, tropical.execute_tropical_center)
    register_external_primitive(op)

    # Phase 5b-bis (2026-05-10) — softmin primitive per
    # external_research_2026-05-10.md §3.5: temperature-scaled softmin
    # (LogSumExp over -x/tau) is the gradient-friendly replacement for
    # hard tropical max and lets the grammar swap it in as a softmax variant.
    op = PrimitiveOp(
        name="tropical_softmax",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Gradient-friendly softmin (softmax over -x/tau) — drop-in for vanilla softmax in tropical/shortest-path contexts",
        algebraic_space="tropical",
    )
    op = _with_execute(op, tropical.execute_tropical_softmax)
    register_external_primitive(op)

    # Phase 5c (2026-05-11) — binary-tree feature mixer per
    # external_research_2026-05-10.md §2.1 ("leafed layers"). Reshapes
    # the feature dim into 2^depth leaves and gate-blends pairs at each
    # level. Adds a structural axis the grammar previously couldn't express.
    op = PrimitiveOp(
        name="tree_mix",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="identity",
        has_params=True,
        param_formula="D",  # one learned gate vector of size D
        description="Atomic binary mixer node: z = sigmoid(W) * x + (1 - sigmoid(W)) * y. Templates compose 2^K - 1 nodes to build a balanced binary tree of depth K (research §2.1 leafed layers).",
    )
    op = _with_execute(op, _tree_mix_mod.execute_tree_mix)
    register_external_primitive(op)

    # Phase 5c (2026-05-11) — Multi-Head Latent Attention per research §1.1.
    # DeepSeek V2/V3's asymmetric KV compression: K and V share a low-rank
    # latent down-projection but use distinct up-projections to reconstruct
    # K and V at attention time. Cuts KV cache by ~93% at d_latent = D/8
    # while retaining standard attention semantics.
    op = PrimitiveOp(
        name="mla_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="identity",
        has_params=True,
        param_formula="D*D*3/8",  # 3 * D * (D/8) — down + 2 up-projs
        description="Multi-head latent attention: K and V share a low-rank latent path (d_latent=D/8 by default), reconstructed via distinct up-projections, then standard softmax attention on Q. KV-cache compression.",
    )
    op = _with_execute(op, _mla_mod.execute_mla_attention)
    register_external_primitive(op)

    # Phase 5d (2026-05-11) — Product-Quantized Embedding per research §2.3.
    # Splits feature dim D into M subspaces, each with K learnable codebook
    # centroids. Token slices are replaced by softmax-weighted combinations
    # of codebook entries. Maps to the compression-dominant convergence
    # observed in the 2026-05-11 MLA-run leaderboard (latent_attn_* +
    # sparse_ffn templates ran at 100% S1).
    op = PrimitiveOp(
        name="pq_embedding",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="M*K*D/M",  # M codebooks × K centroids × D/M sub_dim = K*D
        description="Product-quantized embedding: split D into M subspaces of D/M, each with K learnable codebook centroids; per-subspace softmax assignment then weighted reconstruction. Compression-style learned quantization.",
    )
    op = _with_execute(op, _pq_emb_mod.execute_pq_embedding)
    register_external_primitive(op)

    # Phase 5e (2026-05-11) — Matrix-memory LSTM cell (xLSTM/mLSTM, research §1.5).
    # Recurrent state is a (D, D) matrix updated by a key-value outer product.
    # Novel state form: complements diagonal SSMs (Mamba/S5) and cross-token
    # attention. Used by tpl_mlstm_block / tpl_mlstm_sparse_ffn_block.
    op = PrimitiveOp(
        name="mlstm_cell",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="4*D*D + 2*D",  # W_q, W_k, W_v, W_o + scalar gate vectors
        description="Matrix-memory LSTM cell: recurrent state is a (D, D) outer-product accumulator addressed by per-token queries. Per-token compute is (4*D + 2)·D multiply-add; backward via autograd.",
    )
    op = _with_execute(op, _mlstm_mod.execute_mlstm_cell)
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
        algebraic_space="padic",
    )
    op = _with_execute(op, padic.execute_padic_expand)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="ultrametric_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Attention using ultrametric (p-adic) distance",
        algebraic_space="padic",
    )
    op = _with_execute(op, padic.execute_ultrametric_attn)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="padic_gate",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Gate activations by smooth p-adic valuation strength",
        algebraic_space="padic",
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
        algebraic_space="clifford",
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
        algebraic_space="clifford",
    )
    op = _with_execute(op, clifford.execute_rotor_transform)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="grade_select",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Select vector grade from Clifford multivector",
        algebraic_space="clifford",
    )
    op = _with_execute(op, clifford.execute_grade_select)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="grade_mix",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Blend vector and bivector grades in Clifford multivectors",
        algebraic_space="clifford",
    )
    op = _with_execute(op, clifford.execute_grade_mix)
    register_external_primitive(op)

    # Phase 5 V2 (2026-05-04) — Clifford companion ops per
    # research/reports/novel_math_ops_proposal_20260504.md §3.1
    op = PrimitiveOp(
        name="clifford_inverse",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Multiplicative inverse of a Cl(3,0) multivector "
        "(~mv / ||mv||²); exact for versors, regularized fallback otherwise",
        algebraic_space="clifford",
    )
    op = _with_execute(op, clifford.execute_clifford_inverse)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="versor_apply",
        category=OpCategory.MATH_SPACE,
        n_inputs=2,
        shape_rule="binary_broadcast",
        description="Versor sandwich product v · mv · v⁻¹ — canonical "
        "Clifford rotation; generalizes rotor_transform to non-unit versors",
        algebraic_space="clifford",
    )
    op = _with_execute(op, clifford.execute_versor_apply)
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
        algebraic_space="poincare",
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
        algebraic_space="tropical",
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
        algebraic_space="tropical",
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
        algebraic_space="tropical",
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
        algebraic_space="clifford",
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
        algebraic_space="padic",
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
        algebraic_space="spiking",
    )
    op = _with_execute(op, spiking.execute_lif)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="spike_rate_code",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Continuous-to-spike-to-continuous rate coding with STE",
        algebraic_space="spiking",
    )
    op = _with_execute(op, spiking.execute_spike_rate_code)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="stdp_attention",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="1",
        description="STDP-inspired causal attention with learnable temporal decay",
        algebraic_space="spiking",
    )
    op = _with_execute(op, spiking.execute_stdp_attention)
    register_external_primitive(op)

    op = PrimitiveOp(
        name="sparse_threshold",
        category=OpCategory.MATH_SPACE,
        n_inputs=1,
        shape_rule="identity",
        description="Adaptive median-based threshold gate (~50% sparsity)",
        algebraic_space="spiking",
    )
    op = _with_execute(op, spiking.execute_sparse_threshold)
    register_external_primitive(op)


def _with_execute(op: PrimitiveOp, fn) -> PrimitiveOp:
    """Attach an execution function to a PrimitiveOp.

    Since PrimitiveOp is frozen, we store it as an attribute on the object.
    """
    # Use object.__setattr__ to bypass frozen dataclass
    object.__setattr__(op, "execute_fn", fn)
    return op
