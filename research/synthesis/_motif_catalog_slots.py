"""Slot motifs: guarded activations, routing/control, position, reduction, math-space.

Template-managed ops, guarded activation chains, routing control motifs,
position+attention combos, reduction motifs, and math-space algebraic bridges.
"""

from __future__ import annotations

from typing import Tuple

from ._motif_types import (
    Motif,
    MotifStep,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_REDUCE,
)
from .op_roles import OpRole

# ── Template-managed ops (intentionally excluded from standalone motifs) ──
# These ops require specific wiring context provided by dedicated templates:
#   - routing_conditioned_compression: needs token_type_classifier input
#   - compression_mixture_experts: needs token_type_classifier input
#   - token_type_classifier: signal producer only, wired by templates
#   - div_safe: UNSAFE role, needs template context for safety
#   - adaptive_lane_mixer: 2-input routing, wired by templates

SLOT_MOTIFS: Tuple[Motif, ...] = (
    # ── B. Guarded Activation motifs (safe predecessor context) ───
    Motif(
        name="act_exp_normed",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("exp", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → exp → proj (norm bounds input to safe range)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="act_log_sigmoid",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("sigmoid", OpRole.ACTIVATE),
            MotifStep("log", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="sigmoid → log → proj (sigmoid guarantees x > 0)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_log_exp",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("exp", OpRole.ACTIVATE),
            MotifStep("log", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="exp → log → proj (exp guarantees x > 0, log inverts)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_log_safe",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("log", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="log (softplus-guarded internally) → proj",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="act_sqrt_square",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("square", OpRole.ACTIVATE),
            MotifStep("sqrt", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="square → sqrt → proj (square guarantees x >= 0)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="act_sqrt_abs",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("abs", OpRole.ACTIVATE),
            MotifStep("sqrt", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="abs → sqrt → proj (abs guarantees x >= 0)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="act_square_proj",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("square", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → square → proj (norm bounds gradient amp)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_abs_proj",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("abs", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="abs → proj (always defined, gradient +/-1)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_neg_proj",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("neg", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="neg → proj (trivially safe, gradient = -1)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_sign_ste",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("sign_ste", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="sign_ste → proj (STE passes gradient through)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_reciprocal_safe",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("reciprocal", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → reciprocal → proj (reciprocal impl is 1/(1+sigmoid(x)), always [0.5,1.0])",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="act_sin_proj",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("sin", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="sin → proj (bounded [-1,1])",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="act_cos_proj",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("cos", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="cos → proj (bounded [-1,1])",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    # ── C. Gate/Routing motifs ────────────────────────────────────
    Motif(
        name="gate_scale",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("learnable_scale", OpRole.GATE),),
        description="Standalone learnable scale gate",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="gate_bias_act",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("learnable_bias", OpRole.GATE),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Learnable bias → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="gate_entropy",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("token_entropy", OpRole.GATE),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="entropy_score (->dim=1) → linear_proj_up (restores dim)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="gate_progressive",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("adaptive_rank_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Progressive compression gate → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    # NOTE: token_type_classifier has wiring constraints (output only valid for
    # entropy_score, compression_mixture_experts, etc.) — not suitable for
    # generic motif chains. Handled by dedicated templates instead.
    Motif(
        name="route_identity",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("identity", OpRole.RESIDUAL),),
        description="Identity pass-through (for ablation/skip)",
        support=0,
        avg_loss_ratio=0.0,
        lift=0.5,
    ),
    # ── D. Routing/Control motifs ─────────────────────────────────
    # cascade and early_exit require residual bypass (REQUIRES_RESIDUAL_BYPASS).
    # _instantiate_motif auto-wraps these with add(input, gated) to satisfy
    # the bypass constraint, so they are safe in any template slot.
    Motif(
        name="route_early_exit",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("confidence_token_gate", OpRole.ROUTE, config={"threshold": 0.5}),
        ),
        description="Early-exit confidence gate (auto-bypassed)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="route_cascade",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("learned_token_gate", OpRole.ROUTE, config={"threshold": 0.5}),
        ),
        description="Cascade difficulty gate (auto-bypassed)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="route_adaptive_recursion",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("depth_weighted_proj", OpRole.ROUTE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Adaptive recursion → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    # NOTE: no standalone `route_hybrid_token_gate` motif. `hybrid_token_gate`
    # carries a mandatory residual bypass (REQUIRES_RESIDUAL_BYPASS in
    # _template_helpers._instantiate_motif), so an `add` is always inserted
    # immediately after it. That breaks `hybrid_sparse_router`'s "immediate
    # hybrid_token_gate predecessor" rule, making any sampled single/two-step
    # variant un-instantiable as a valid slot fill. The hybrid-routing chain is
    # only valid as the explicit branch construction in
    # tpl_intelligent_multilane_router, so it is not offered as a sampled motif.
    Motif(
        name="route_sparse_triplet",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep(
                "sparse_span_builder",
                OpRole.ROUTE,
                config={"span_width": 3, "fallback_behavior": "default_path"},
            ),
            MotifStep(
                "hybrid_sparse_router",
                OpRole.ROUTE,
                config={"span_width": 3, "lane_count": 3, "confidence_threshold": 0.45},
            ),
        ),
        description="Sparse triplet span builder feeding hybrid lane router",
        support=0,
        avg_loss_ratio=0.0,
        lift=3.5,
    ),
    # ── E. Position + Attention motifs ────────────────────────────
    Motif(
        name="attn_rope",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("rope_rotate", OpRole.POSITION),
            MotifStep("softmax_attention", OpRole.MIX),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="RoPE → softmax attention → norm → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=0.3,  # Demoted: rope_attention_block has 4.9% S1 — broken combo
    ),
    Motif(
        name="attn_causal_mask",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("softmax_attention", OpRole.MIX),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Softmax attention → norm → projection (causality handled internally)",
        support=0,
        avg_loss_ratio=0.0,
        lift=0.05,
    ),
    Motif(
        name="attn_sliding_window",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("sliding_window_mask", OpRole.POSITION),
            MotifStep("linear_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Sliding window mask → linear attention → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=3.0,
    ),
    # ── F. Reduction motifs ───────────────────────────────────────
    Motif(
        name="reduce_sum",
        motif_class=MOTIF_CLASS_REDUCE,
        steps=(
            MotifStep("sum_last", OpRole.REDUCE),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="sum_last → linear_proj_up (restore collapsed dim)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="reduce_mean",
        motif_class=MOTIF_CLASS_REDUCE,
        steps=(
            MotifStep("mean_last", OpRole.REDUCE),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="mean_last → linear_proj_up (restore collapsed dim)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="reduce_max",
        motif_class=MOTIF_CLASS_REDUCE,
        steps=(
            MotifStep("max_last", OpRole.REDUCE),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="max_last → linear_proj_up (restore collapsed dim)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="reduce_norm",
        motif_class=MOTIF_CLASS_REDUCE,
        steps=(
            MotifStep("norm_last", OpRole.REDUCE),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="norm_last → linear_proj_up (restore collapsed dim)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="reduce_cumsum",
        motif_class=MOTIF_CLASS_REDUCE,
        steps=(
            MotifStep("cumsum", OpRole.REDUCE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="cumsum → projection (running sum along sequence)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── G. Math-Space motifs (algebraic bridges) ──────────────────
    Motif(
        name="tropical_moe_block",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("tropical_moe", OpRole.ROUTE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="tropical_moe → linear_proj (back to euclidean)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="tropical_router_block",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("tropical_router", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="tropical_router → linear_proj (routing scores)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="clifford_rotor_grade",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("rotor_transform", OpRole.MIX),
            MotifStep("grade_select", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rotor_transform → grade_select → proj (Clifford bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="spiking_lif_rate",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("lif_neuron", OpRole.ACTIVATE),
            MotifStep("spike_rate_code", OpRole.ACTIVATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="lif_neuron → spike_rate_code → proj (spiking bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=3.0,
    ),
    Motif(
        name="spiking_threshold_stdp",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("lif_neuron", OpRole.ACTIVATE),
            MotifStep("sparse_threshold", OpRole.ACTIVATE),
            MotifStep("stdp_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="lif_neuron → sparse_threshold → stdp_attention → proj (spiking attn)",
        support=0,
        avg_loss_ratio=0.0,
        lift=3.0,
    ),
    # ── Spiking + tropical routing (proven lr=0.007 pattern) ────────
    Motif(
        name="spiking_tropical_gate",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("lif_neuron", OpRole.ACTIVATE),
            MotifStep("tropical_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="lif_neuron → tropical_gate → proj (spike encoding + tropical routing, lr=0.007)",
        support=2,
        avg_loss_ratio=0.007,
        lift=5.0,
    ),
    Motif(
        name="spiking_rate_tropical_gate",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("spike_rate_code", OpRole.ACTIVATE),
            MotifStep("tropical_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="spike_rate_code → tropical_gate → proj (rate coding + tropical routing, lr=0.007)",
        support=2,
        avg_loss_ratio=0.007,
        lift=5.0,
    ),
    Motif(
        name="spiking_threshold_tropical_gate",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("lif_neuron", OpRole.ACTIVATE),
            MotifStep("sparse_threshold", OpRole.ACTIVATE),
            MotifStep("tropical_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="lif_neuron → sparse_threshold → tropical_gate → proj (spiking + sparsification + routing)",
        support=0,
        avg_loss_ratio=0.0,
        lift=4.0,
    ),
    Motif(
        name="padic_gate_proj",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("padic_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="padic_gate → linear_proj (p-adic hierarchy bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="poincare_norm_bridge",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("hyperbolic_norm", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="hyperbolic_norm → linear_proj (Poincare bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── H. Channel/Mix motifs ─────────────────────────────────────
    Motif(
        name="mix_multi_head",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("multi_head_mix", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="multi_head_mix → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    # ── I. Cumprod motif (sigmoid guards decay) ───────────────────
    Motif(
        name="decay_cumprod",
        motif_class=MOTIF_CLASS_GUARDED_ACT,
        steps=(
            MotifStep("sigmoid", OpRole.ACTIVATE),
            MotifStep("cumprod_safe", OpRole.REDUCE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="sigmoid → cumprod_safe → proj (sigmoid in (0,1) => decays)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
)
