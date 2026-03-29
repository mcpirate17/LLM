"""
Validated Motif Library for Compositional Grammar

A motif is a 2-4 op chain empirically validated to:
  (a) produce gradients, (b) learn, (c) be numerically stable.

Motifs are the atoms of the new grammar — known-good component
combinations that templates compose into full architectures.

Mined from 734 top performers out of 4,959 candidates.
See: research/docs/motif_mining_report.md
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .op_roles import OpRole


@dataclass(slots=True, frozen=True)
class MotifStep:
    """A single step in a motif's op sequence."""

    op_name: str
    role: OpRole
    config: Dict = field(default_factory=dict)
    # If True, this step's op can be substituted with any op of the same role.
    substitutable: bool = False


@dataclass(slots=True, frozen=True)
class Motif:
    """A validated functional unit — 2-4 ops that compose correctly."""

    name: str
    motif_class: str  # e.g., "ffn_core", "attention_core", "ssm_core"
    steps: Tuple[MotifStep, ...]
    description: str = ""
    # Statistical evidence from mining
    support: int = 0  # Number of top performers containing this pattern
    avg_loss_ratio: float = 0.0
    lift: float = 1.0  # Enrichment in winners vs general population


# ── Motif Class Constants ───────────────────────────────────────────

MOTIF_CLASS_FFN = "ffn_core"
MOTIF_CLASS_ATTENTION = "attention_core"
MOTIF_CLASS_SSM = "ssm_core"
MOTIF_CLASS_CONV = "conv_core"
MOTIF_CLASS_GATE = "gate_core"
MOTIF_CLASS_NORM = "norm_wrap"
MOTIF_CLASS_SPARSE = "sparse_core"
MOTIF_CLASS_MOE = "moe_core"
MOTIF_CLASS_CHANNEL = "channel_core"
MOTIF_CLASS_EFFICIENT_PROJ = "efficient_proj"
MOTIF_CLASS_REDUCE = "reduce_core"
MOTIF_CLASS_GUARDED_ACT = "guarded_act"
MOTIF_CLASS_MATH_SPACE = "math_space"

ALL_MOTIF_CLASSES: FrozenSet[str] = frozenset(
    {
        MOTIF_CLASS_FFN,
        MOTIF_CLASS_ATTENTION,
        MOTIF_CLASS_SSM,
        MOTIF_CLASS_CONV,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_NORM,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_MOE,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_REDUCE,
        MOTIF_CLASS_GUARDED_ACT,
        MOTIF_CLASS_MATH_SPACE,
    }
)


# ── Validated Motifs ────────────────────────────────────────────────
# Derived from motif_mining_report.md findings.
# Each motif has statistical backing from the top-performer pool.

_MOTIF_LIST: Tuple[Motif, ...] = (
    # ── FFN cores (Cluster 4/5 pattern, 21% of top performers) ──────
    Motif(
        name="ffn_expand_contract",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("linear_proj_up", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
            MotifStep("linear_proj_down", OpRole.PROJECT),
        ),
        description="Standard FFN: expand → activate → contract",
        support=157,
        avg_loss_ratio=0.063,
        lift=1.75,
    ),
    Motif(
        name="ffn_bottleneck",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("linear_proj_down", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="Bottleneck FFN: contract → activate → expand",
        support=168,
        avg_loss_ratio=0.064,
        lift=1.61,
    ),
    Motif(
        name="ffn_fused_gelu",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("fused_linear_gelu", OpRole.PROJECT),
            MotifStep("linear_proj_down", OpRole.PROJECT),
        ),
        description="Fused Linear+GELU → contract (Triton-accelerated)",
        support=40,
        avg_loss_ratio=0.089,
        lift=2.5,
    ),
    # ── Attention cores (2.2-2.4x lift) ────────────────────────────
    Motif(
        name="attn_softmax",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("softmax_attention", OpRole.MIX),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Softmax self-attention → norm → projection",
        support=13,
        avg_loss_ratio=0.142,
        lift=2.37,
    ),
    Motif(
        name="attn_linear",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("linear_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Linear attention → projection",
        support=9,
        avg_loss_ratio=0.128,
        lift=2.34,
    ),
    Motif(
        name="attn_graph",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("graph_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Graph attention with learned adjacency → projection",
        support=12,
        avg_loss_ratio=0.120,
        lift=3.5,
    ),
    Motif(
        name="attn_local_window",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("local_window_attn", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Local windowed causal attention → projection",
        support=15,
        avg_loss_ratio=0.062,
        lift=3.0,
    ),
    Motif(
        name="attn_latent_compress",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("latent_attention_compressor", OpRole.MIX),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="MLA-style KV compression → expand (best pair LR 0.040)",
        support=20,
        avg_loss_ratio=0.040,
        lift=1.8,
    ),
    # ── SSM / state-space cores ─────────────────────────────────────
    Motif(
        name="ssm_selective_scan",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("selective_scan", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Mamba-style selective scan → projection",
        support=36,
        avg_loss_ratio=0.161,
        lift=1.47,
    ),
    Motif(
        name="ssm_state_space",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("state_space", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="State-space block: rmsnorm bounds input → state_space → projection",
        support=20,
        avg_loss_ratio=0.180,
        lift=3.0,
    ),
    Motif(
        name="ssm_ternary_scan",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("selective_scan", OpRole.MIX),
            MotifStep("ternary_projection", OpRole.PROJECT),
        ),
        description="Scan → ternary projection (6.75x trigram lift)",
        support=4,
        avg_loss_ratio=0.090,
        lift=6.75,
    ),
    # ── Conv cores ──────────────────────────────────────────────────
    Motif(
        name="conv_mamba_like",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE),
            MotifStep("selective_scan", OpRole.MIX),
        ),
        description="Conv → SiLU → scan (Mamba block pattern)",
        support=15,
        avg_loss_ratio=0.071,
        lift=2.0,
    ),
    Motif(
        name="conv_gelu_proj",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Local conv → activation → projection",
        support=48,
        avg_loss_ratio=0.194,
        lift=1.75,
    ),
    Motif(
        name="conv_swiglu",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("swiglu_mlp", OpRole.GATE),
        ),
        description="Conv → SwiGLU (6.75x trigram lift, avg LR 0.071)",
        support=15,
        avg_loss_ratio=0.071,
        lift=6.75,
    ),
    # ── Gate cores ──────────────────────────────────────────────────
    Motif(
        name="gate_swiglu",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("swiglu_mlp", OpRole.GATE),),
        description="SwiGLU (LLaMA FFN pattern, 2.49x lift)",
        support=41,
        avg_loss_ratio=0.166,
        lift=2.49,
    ),
    Motif(
        name="gate_linear",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("gated_linear", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Fused gate → projection (1.91x lift)",
        support=30,
        avg_loss_ratio=0.147,
        lift=1.91,
    ),
    Motif(
        name="gate_relu_routing",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("relu_gated_moe", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="ReLU gate routing → projection",
        support=15,
        avg_loss_ratio=0.120,
        lift=1.5,
    ),
    # ── Normalization wrappers ──────────────────────────────────────
    Motif(
        name="norm_rms",
        motif_class=MOTIF_CLASS_NORM,
        steps=(MotifStep("rmsnorm", OpRole.NORMALIZE),),
        description="RMSNorm (pre-norm wrapper)",
        support=200,
        avg_loss_ratio=0.150,
        lift=1.4,
    ),
    Motif(
        name="norm_layer",
        motif_class=MOTIF_CLASS_NORM,
        steps=(MotifStep("layernorm", OpRole.NORMALIZE),),
        description="LayerNorm (pre-norm wrapper)",
        support=140,
        avg_loss_ratio=0.210,
        lift=1.41,
    ),
    # ── Sparse linear cores (2.0-2.2x lift) ────────────────────────
    Motif(
        name="sparse_nm",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("nm_sparse_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="N:M sparse linear → activation (2.15x lift, LR 0.094)",
        support=36,
        avg_loss_ratio=0.094,
        lift=2.15,
    ),
    Motif(
        name="sparse_block",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("block_sparse_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Block-sparse linear → activation (2.09x lift)",
        support=67,
        avg_loss_ratio=0.144,
        lift=2.09,
    ),
    Motif(
        name="sparse_ternary",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("ternary_projection", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="1.58-bit ternary projection → activation (2.08x lift)",
        support=53,
        avg_loss_ratio=0.111,
        lift=2.08,
    ),
    Motif(
        name="sparse_semi_structured",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("semi_structured_2_4_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="2:4 semi-structured sparse → activation (1.97x lift)",
        support=40,
        avg_loss_ratio=0.152,
        lift=1.97,
    ),
    # ── MoE cores (3.0-3.4x lift) ──────────────────────────────────
    Motif(
        name="moe_topk",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("moe_topk", OpRole.ROUTE),),
        description="Sparse top-k MoE (3.09x lift)",
        support=27,
        avg_loss_ratio=0.115,
        lift=3.09,
    ),
    Motif(
        name="moe_2expert",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("moe_2expert", OpRole.ROUTE),),
        description="Lightweight 2-expert MoE (2.59x lift)",
        support=63,
        avg_loss_ratio=0.114,
        lift=2.59,
    ),
    Motif(
        name="moe_proj_block",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("linear_proj", OpRole.PROJECT),
            MotifStep("moe_topk", OpRole.ROUTE),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="Proj → MoE → norm (sparse MoE block)",
        support=15,
        avg_loss_ratio=0.100,
        lift=2.5,
    ),
    # ── Functional / neural-field motifs ────────────────────────────
    Motif(
        name="mix_integral_kernel",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("integral_kernel", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Integral kernel mixing → projection",
        support=10,
        avg_loss_ratio=0.200,
        lift=2.5,
    ),
    Motif(
        name="mix_fixed_point",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("fixed_point_iter", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Fixed-point iteration → projection",
        support=5,
        avg_loss_ratio=0.250,
        lift=2.0,
    ),
    Motif(
        name="mix_basis_expansion",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("basis_expansion", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Basis expansion → activation (sinusoidal features)",
        support=8,
        avg_loss_ratio=0.200,
        lift=1.0,
    ),
    Motif(
        name="hyperbolic_residual_bridge",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("exp_map", OpRole.MIX),
            MotifStep("hyp_linear", OpRole.PROJECT),
            MotifStep("hyp_tangent_nonlinear", OpRole.ACTIVATE),
            MotifStep("log_map", OpRole.MIX),
        ),
        description="Leaderboard-seeded hyperbolic bridge block: exp_map → hyp_linear → tangent nonlinearity → log_map",
        support=2,
        avg_loss_ratio=0.010,
        lift=3.0,
    ),
    Motif(
        name="tropical_attention_gate",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("tropical_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Tropical block: rmsnorm bounds input → tropical attention → learnable proj",
        support=3,
        avg_loss_ratio=0.009,
        lift=2.1,
    ),
    Motif(
        name="clifford_attention_mix",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("clifford_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Clifford block: rmsnorm bounds input → clifford attention → learnable proj",
        support=3,
        avg_loss_ratio=0.008,
        lift=2.3,
    ),
    Motif(
        name="padic_hierarchy_block",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("padic_expand", OpRole.PROJECT),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="P-adic block: rmsnorm bounds input → padic expansion → learnable proj back",
        support=4,
        avg_loss_ratio=0.009,
        lift=2.4,
    ),
    # ── Channel mixing cores ────────────────────────────────────────
    Motif(
        name="channel_rwkv",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(MotifStep("rwkv_channel", OpRole.MIX),),
        description="RWKV channel mixing (2.41x lift, LR 0.103)",
        support=51,
        avg_loss_ratio=0.103,
        lift=2.41,
    ),
    Motif(
        name="channel_rwkv_time",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("rwkv_time_mixing", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="RWKV time mixing → projection",
        support=20,
        avg_loss_ratio=0.130,
        lift=1.5,
    ),
    # ── Spectral / frequency-domain mixing ───────────────────────────
    Motif(
        name="spectral_filter_mix",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("spectral_filter", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → spectral_filter → projection (FFT-based per-position filtering)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.5,
    ),
    # ── Compound Efficiency motifs ───────────────────────────────────
    Motif(
        name="sparse_moe_proj",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("nm_sparse_linear", OpRole.PROJECT),
            MotifStep("moe_topk", OpRole.ROUTE),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="Sparse linear → MoE routing → norm (compound efficiency)",
        support=10,
        avg_loss_ratio=0.095,
        lift=3.5,
    ),
    Motif(
        name="bottleneck_sparse",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("linear_proj_down", OpRole.PROJECT),
            MotifStep("nm_sparse_linear", OpRole.PROJECT),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="Bottleneck D/4 → sparse linear → expand (compound savings)",
        support=8,
        avg_loss_ratio=0.110,
        lift=2.5,
    ),
    Motif(
        name="routed_ternary",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("ternary_projection", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Ternary projection → activation (4x efficiency, routing via template)",
        support=5,
        avg_loss_ratio=0.088,
        lift=4.0,
    ),
    Motif(
        name="merge_scan",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("adjacent_token_merge", OpRole.MIX),
            MotifStep("swiglu_mlp", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Token merge → FFN → proj (sequence compression with safe successor)",
        support=6,
        avg_loss_ratio=0.105,
        lift=2.5,
    ),
    Motif(
        name="sparse_gated_ffn",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("block_sparse_linear", OpRole.PROJECT),
            MotifStep("swiglu_mlp", OpRole.GATE),
            MotifStep("linear_proj_down", OpRole.PROJECT),
        ),
        description="Block-sparse → SwiGLU → contract (sparse FFN)",
        support=12,
        avg_loss_ratio=0.098,
        lift=3.0,
    ),
    Motif(
        name="conditional_skip",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("gated_linear", OpRole.GATE),),
        description="Gated linear conditional compute (routing via template)",
        support=8,
        avg_loss_ratio=0.130,
        lift=2.0,
    ),
    # ── Missing mixing ops (catalog byte_safe, no motif) ─────────────
    Motif(
        name="attn_diff",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("diff_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Differential attention (dual softmax subtraction) → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="attn_gated_delta",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("gated_delta", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Gated delta rule recurrence → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="conv_only_block",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("conv_only", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Pure conv stack (local + dilated) → projection",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    # ── Missing routing ops (catalog byte_safe, no motif) ────────────
    Motif(
        name="route_mod_topk",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("depth_token_mask", OpRole.ROUTE),),
        description="Mixture-of-Depths top-k token routing",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="route_speculative",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("cheap_verify_blend", OpRole.ROUTE),),
        description="Speculative dual-path blend (cheap + verify gate)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="route_topk_gate",
        motif_class=MOTIF_CLASS_GATE,
        steps=(MotifStep("topk_gate", OpRole.GATE),),
        description="Sparse top-k gating over feature halves",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="route_lanes_block",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("gated_lane_blend", OpRole.ROUTE),),
        description="Multi-lane dispatch with learned lane scorer",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="route_recursion_block",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("depth_gated_transform", OpRole.ROUTE),),
        description="Adaptive recursion depth per token",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="route_topk_sparse",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(MotifStep("feature_sparsity", OpRole.ROUTE),),
        description="Hard top-k token selection with STE",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    # ── A. Efficient Projection motifs (unconditionally safe) ─────
    Motif(
        name="proj_bottleneck",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("bottleneck_proj", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Bottleneck projection → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="proj_low_rank",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("low_rank_proj", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Low-rank projection → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="proj_grouped",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("grouped_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Grouped linear → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="proj_shared_basis",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("shared_basis_proj", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Shared basis projection → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="proj_tied",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("tied_proj", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Tied projection → activation",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    # ── Recently-added ops: motifs for grammar reachability ─────
    # These ops have dispatch handlers but no motifs, making them unreachable.
    Motif(
        name="codebook_proj",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("embedding_lookup", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Soft codebook projection → activation (learnable VQ discretization)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="kronecker_proj",
        motif_class=MOTIF_CLASS_EFFICIENT_PROJ,
        steps=(
            MotifStep("kronecker_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Kronecker-factored projection → activation (128x param compression)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="chebyshev_spectral",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("chebyshev_spectral_mix", OpRole.MIX),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="Chebyshev spectral mix → normalize",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="n_way_routing",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("sparse_bottleneck_moe", OpRole.ROUTE),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="N-way sparse routing → normalize",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    # ── True routing — heterogeneous expert dispatch ─────────────────
    Motif(
        name="true_hetero_moe",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("hetero_moe", OpRole.ROUTE),),
        description="True heterogeneous MoE: routes tokens to attention, conv, or SSM experts",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="true_arch_router",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("arch_router", OpRole.ROUTE),),
        description="Architecture router: tokens choose transformer, mamba, or MLP style",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="true_compute_budget",
        motif_class=MOTIF_CLASS_MOE,
        steps=(MotifStep("compute_budget_router", OpRole.ROUTE),),
        description="Compute budget router: easy→cheap linear, medium→conv, hard→attention",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.5,
    ),
    Motif(
        name="spectral_filter_block",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("spectral_filter", OpRole.MIX),
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="spectral_filter → rmsnorm (satisfies must_precede constraint)",
        support=0,
        avg_loss_ratio=0.0,
        lift=1.0,
    ),
    Motif(
        name="tropical_matmul_block",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("tropical_matmul", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="tropical_matmul → linear_proj (back to euclidean)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── Tropical ops (real→real, no space conflict) ─────────────────
    Motif(
        name="tropical_gate_block",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("tropical_gate", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="tropical_gate → linear_proj (shortest-path gating)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    Motif(
        name="tropical_center_norm",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("tropical_center", OpRole.NORMALIZE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="tropical_center → linear_proj (subtract tropical baseline)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── Grade mix (must follow clifford op per MATH_SPACE_RULES) ──
    Motif(
        name="clifford_attention_grade",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("clifford_attention", OpRole.MIX),
            MotifStep("grade_mix", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → clifford_attention → grade_mix → proj (full Clifford block)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── Padic residual bridge (padic→real output) ─────────────────
    Motif(
        name="padic_residual_bridge",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("padic_expand", OpRole.PROJECT),
            MotifStep("padic_residual", OpRole.RESIDUAL),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → padic_expand → padic_residual → proj (multi-res p-adic bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── Space-bridging motifs (euclidean → non-euclidean → euclidean) ──
    Motif(
        name="poincare_add_bridge",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("exp_map", OpRole.MIX),
            MotifStep("poincare_add", OpRole.RESIDUAL),
            MotifStep("log_map", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="exp_map → poincare_add → log_map → proj (Möbius addition bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # hyp_distance: binary op, handled by tpl_hyp_distance_scoring template
    Motif(
        name="ultrametric_attention_bridge",
        motif_class=MOTIF_CLASS_MATH_SPACE,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
            MotifStep("padic_expand", OpRole.PROJECT),
            MotifStep("ultrametric_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="rmsnorm → padic_expand → ultrametric_attention → proj (p-adic attention bridge)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
    # ── Template-managed ops (intentionally excluded from standalone motifs) ──
    # These ops require specific wiring context provided by dedicated templates:
    #   - routing_conditioned_compression: needs token_type_classifier input
    #   - compression_mixture_experts: needs token_type_classifier input
    #   - token_type_classifier: signal producer only, wired by templates
    #   - div_safe: UNSAFE role, needs template context for safety
    #   - adaptive_lane_mixer: 2-input routing, wired by templates
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
        description="square → sqrt → proj (square guarantees x ≥ 0)",
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
        description="abs → sqrt → proj (abs guarantees x ≥ 0)",
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
        description="abs → proj (always defined, gradient ±1)",
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
        description="entropy_score (→dim=1) → linear_proj_up (restores dim)",
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
        lift=1.5,
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
        description="hyperbolic_norm → linear_proj (Poincaré bridge)",
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
    Motif(
        name="mix_transpose",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("transpose_sd", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="transpose_sd → projection (seq↔dim swap)",
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
        description="sigmoid → cumprod_safe → proj (sigmoid ∈ (0,1) ⇒ decays)",
        support=0,
        avg_loss_ratio=0.0,
        lift=2.0,
    ),
)


# ── Index structures for O(1) lookup ────────────────────────────────

VALIDATED_MOTIFS: Dict[str, Motif] = {m.name: m for m in _MOTIF_LIST}

MOTIFS_BY_CLASS: Dict[str, List[Motif]] = {}
for _m in _MOTIF_LIST:
    MOTIFS_BY_CLASS.setdefault(_m.motif_class, []).append(_m)

ALL_MOTIFS: Tuple[Motif, ...] = tuple(_MOTIF_LIST)


# ── Activation substitution pool ────────────────────────────────────
# When a motif step is marked substitutable=True and has role ACTIVATE,
# these are the valid replacements.
ACTIVATION_POOL: Tuple[str, ...] = (
    "gelu",
    "silu",
    "relu",
    "tanh",
    "sigmoid",
    "sin",
    "cos",
    "abs",
    "neg",
    "square",
    "softmax_last",
    "reciprocal",
)

# ── Context-aware activation placement rules ───────────────────────
# Each activation maps to:
#   "after": set of predecessor OpRoles or specific op names where valid.
#             None means any predecessor is fine.
#   "before": set of successor op names where valid.
#              None means any successor is fine.
# Derived from empirical analysis of 203 training runs.

ACTIVATION_RULES: Dict[str, Dict] = {
    # Universal: safe everywhere
    "gelu": {"after": None, "before": None},
    "silu": {"after": None, "before": None},
    "relu": {"after": None, "before": None},
    # Risky: exp amplifies, must only follow bounded output (norm/sigmoid/tanh)
    "exp": {
        "after": {OpRole.NORMALIZE, "tanh", "sigmoid", "rmsnorm", "layernorm"},
        "before": None,
    },
    # Bounded [-1,1]: safe after up-projections, but kills signal after
    # down-projections (0% S1 with linear_proj_down across 60 entries).
    # Restrict: must feed into a projection or merge, not stand alone.
    "tanh": {"after": None, "before": {"linear_proj", "linear_proj_up", "mul", "add"}},
    # Gating: sigmoid → [0,1], useful before multiplicative/decay ops and residual add
    "sigmoid": {
        "after": None,
        "before": {
            "mul",
            "outer_product",
            "matmul",
            "cosine_similarity",
            "cumprod_safe",
            "add",
            "reciprocal",
            "log",
            "moe_topk",
            "moe_2expert",
        },
    },
    # Periodic: learnable frequency features, best after projections/norms.
    # Must feed into projection/merge — periodic output in compressed space
    # is noise without a learned transform to interpret it (0% S1 after down-proj).
    "sin": {
        "after": {OpRole.PROJECT, OpRole.NORMALIZE},
        "before": {"linear_proj", "linear_proj_up", "mul", "add"},
    },
    "cos": {
        "after": {OpRole.PROJECT, OpRole.NORMALIZE},
        "before": {"linear_proj", "linear_proj_up", "mul", "add"},
    },
    # Magnitude: loses sign info, must feed into gating/scaling
    "abs": {
        "after": {OpRole.PROJECT, OpRole.MIX},
        "before": {"mul", "learnable_scale", "topk_gate", "add"},
    },
    # Inversion: -x, works broadly but not after sign-flipping ops
    "neg": {
        "after": {OpRole.PROJECT, OpRole.MIX, OpRole.ACTIVATE},
        "before": None,
    },
    # Quadratic: magnitude-expanding, needs bounded input
    "square": {
        "after": {"ternary_projection", "tanh", "sigmoid", OpRole.NORMALIZE},
        "before": None,
    },
    # Attention-norm: only before matmul-like ops that use weights
    "softmax_last": {
        "after": {OpRole.PROJECT, OpRole.MIX},
        "before": {
            "matmul",
            "outer_product",
            "mul",
            "cosine_similarity",
            "div_safe",
            "add",
        },
    },
    # Reciprocal: bounded [0.5, 1.0] via 1/(1+sigmoid(x)), safe after norm/bounded
    "reciprocal": {
        "after": {OpRole.NORMALIZE, "sigmoid", "tanh", "rmsnorm", "layernorm"},
        "before": None,
    },
}


def _get_valid_activations(
    prev_op: Optional[str] = None,
    next_op: Optional[str] = None,
) -> List[str]:
    """Filter ACTIVATION_POOL by context rules.

    Args:
        prev_op: Name of the preceding op (checked against "after" rules).
        next_op: Name of the following op (checked against "before" rules).

    Returns:
        List of activation names valid in this context. Falls back to
        universally-safe activations if nothing matches.
    """
    from .op_roles import get_role

    prev_role = get_role(prev_op) if prev_op else None
    candidates = []
    for act in ACTIVATION_POOL:
        rules = ACTIVATION_RULES.get(act)
        if rules is None:
            candidates.append(act)
            continue
        # Check "after" constraint
        after = rules.get("after")
        if after is not None and prev_op is not None:
            if prev_op not in after and (prev_role is None or prev_role not in after):
                continue
        # Check "before" constraint
        before = rules.get("before")
        if before is not None and next_op is not None:
            if next_op not in before:
                continue
        candidates.append(act)
    # Fallback: always allow the universally safe ones
    if not candidates:
        candidates = ["gelu", "silu", "relu"]
    return candidates


# ── Math-space composition rules ─────────────────────────────────
# Tells the grammar which ops MUST be preceded by a normalizer for
# numerical stability.  Checked in templates._instantiate_motif().
#   "must_precede": predecessor must be one of these ops/roles
#   "must_follow_with": a successor from this set must appear after the op

MATH_SPACE_RULES: Dict[str, Dict] = {
    # Tropical ops: input must be bounded (they use min/max/softmax internally).
    # tropical_gate: min-based routing produces sparse gradients. Must be
    # preceded by norm, tropical_attention, or spiking ops (which normalize
    # internally via firing rate encoding). The proven chains:
    #   layernorm → tropical_attention → tropical_gate → tropical_center (8/11 S1, lr=0.079)
    #   spike_rate_code → tropical_gate → linear_proj (2 S1, lr=0.007)
    "tropical_gate": {
        "must_precede": {
            "rmsnorm",
            "layernorm",
            "tropical_attention",
            "lif_neuron",
            "spike_rate_code",
            "sparse_threshold",
        },
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "tropical_center",
        },
    },
    # tropical_attention: sequential min ops compound gradient sparsity.
    # Must be followed by a projection, tropical_center, or tropical_gate.
    # The proven architecture (8/11 S1 passes, best lr=0.079) chains:
    #   tropical_attention → tropical_gate → tropical_center → linear_proj
    # (Updated 2026-03-22: added tropical_gate as valid successor per DB evidence.)
    "tropical_attention": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "tropical_center",
            "tropical_gate",
        },
    },
    # tropical_center: structural centerer inside tropical motifs. Must follow
    # a tropical mixer (tropical_attention or tropical_gate) and must be
    # followed by a projection to return gradient density.
    # (Volta audit 2026-03-21: 0% S1 in freeform context, passes in
    # valid tropical_core context with loss_ratio=0.52.)
    "tropical_center": {
        "must_follow": {"tropical_attention", "tropical_gate"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "tropical_gate"},
    },
    # tropical_matmul: binary min-based matmul. Needs gradient re-densification
    # via projection after the matmul. The norm requirement is satisfied by the
    # template structure (norm → proj → tropical_matmul), not by must_precede
    # which only checks direct parents (binary ops take projections as inputs).
    "tropical_matmul": {
        "must_follow_with": {"linear_proj", "linear_proj_down"},
    },
    # Clifford ops: input must be bounded (geometric product can amplify)
    "clifford_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "grade_mix": {
        "must_follow": {"clifford_attention", "rotor_transform", "grade_select"}
    },
    # P-adic ops: expansion doubles dim, must project back
    "padic_expand": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "padic_residual",
            "ultrametric_attention",
        },
    },
    # State space: bound input to prevent scan explosion
    "state_space": {"must_precede": {"rmsnorm", "layernorm"}},
    # conv_only: local-only mixing is insufficient as a sole mixer. Must be
    # preceded by normalization and followed by a projection that can learn
    # non-local patterns. (Diagnosis 2026-03-20: 0% S1 rate across 40 attempts,
    # all unstable_dynamics. The conv itself is fine — it just can't carry a
    # language model alone.)
    "conv_only": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "fused_linear_gelu",
        },
    },
    # Spectral filter: always inside residual (handled by grammar.py fix)
    "spectral_filter": {"must_precede": {"rmsnorm", "layernorm"}},
    # Spiking ops: AlgebraicType("spiking", "real", "real") is compatible
    # with euclidean, so algebraic_types_compatible() does NOT prevent
    # placement in non-spiking contexts. These rules enforce spiking-only
    # predecessor chains.
    # (Volta audit 2026-03-21: sparse_threshold and stdp_attention both have
    # 8% compile rate when placed in non-spiking contexts. 92% failure from
    # forward_error and nan_forward.)
    "lif_neuron": {
        "must_follow_with": {
            "spike_rate_code",
            "sparse_threshold",
            "stdp_attention",
            "tropical_gate",
        },
    },
    "sparse_threshold": {
        "must_follow": {"lif_neuron", "spike_rate_code"},
    },
    "stdp_attention": {
        "must_follow": {"sparse_threshold", "spike_rate_code", "lif_neuron"},
    },
    # Hyperbolic ops: require exp_map → op → log_map bridge for Poincaré
    # ball operations. Algebraic type provides partial protection but does
    # not enforce the bridge chain.
    # (Volta audit 2026-03-21: hyp_linear valid context loss_ratio=0.165,
    # default context loss_ratio=0.43. 18/74 init_poisoned, 43/74 s1_fail.)
    "hyp_linear": {
        "must_follow": {"exp_map"},
    },
    "hyp_tangent_nonlinear": {
        "must_follow": {"hyp_linear"},
        "must_follow_with": {"log_map", "linear_proj"},
    },
    # Numerically risky ops: must be preceded by norm to bound activations
    "cosine_similarity": {"must_precede": {"rmsnorm", "layernorm"}},
    "cumprod_safe": {"must_precede": {"rmsnorm", "layernorm"}},
    "div_safe": {"must_precede": {"rmsnorm", "layernorm"}},
    "exp": {"must_precede": {"rmsnorm", "layernorm"}},
    "hyperbolic_norm": {"must_precede": {"rmsnorm", "layernorm"}},
    "log": {"must_precede": {"rmsnorm", "layernorm"}},
    "reciprocal": {"must_precede": {"rmsnorm", "layernorm"}},
    "sqrt": {"must_precede": {"rmsnorm", "layernorm"}},
    # Ops with domain constraints
    "spike_rate_code": {"must_precede": {"rmsnorm", "layernorm", "lif_neuron"}},
    "padic_gate": {"must_precede": {"rmsnorm", "layernorm"}},
    "chebyshev_spectral_mix": {"must_precede": {"rmsnorm", "layernorm"}},
    "kronecker_linear": {"must_precede": {"rmsnorm", "layernorm"}},
    "sparse_bottleneck_moe": {"must_precede": {"rmsnorm", "layernorm"}},
    "integral_kernel": {"must_precede": {"rmsnorm", "layernorm"}},
    "basis_expansion": {"must_precede": {"rmsnorm", "layernorm"}},
    "rotor_transform": {"must_precede": {"rmsnorm", "layernorm"}},
    # Tropical routing ops: tropical algebraic type amplifies via min/max,
    # must be preceded by norm and followed by projection to re-densify gradients.
    "tropical_moe": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "gated_linear"},
    },
    "tropical_router": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "gated_linear"},
    },
    # Sequence-length-altering ops: SSM/recurrent ops assume fixed sequence
    # length and will crash or produce garbage after these ops.
    "adjacent_token_merge": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "gated_linear",
            "fused_linear_gelu",
            "gelu",
            "silu",
            "relu",
            # Data-mined: top-3 token_merge graphs (loss_ratio 0.006–0.043)
            # use conv1d_seq/swiglu_mlp after merge.  Post-merge rmsnorm
            # satisfies conv1d_seq.must_precede when conv follows merge.
            "conv1d_seq",
            "swiglu_mlp",
            "rmsnorm",
            "layernorm",
        },
    },
    "depth_token_mask": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "gated_linear",
            "fused_linear_gelu",
        },
    },
    # Ops that were in the original audit's must_precede list
    "exp_map": {"must_precede": {"rmsnorm", "layernorm", "linear_proj"}},
    "log_map": {"must_follow": {"exp_map", "poincare_add", "hyp_linear"}},
    "fixed_point_iter": {"must_precede": {"rmsnorm", "layernorm"}},
    "hyp_distance": {"must_precede": {"exp_map"}},
    "rwkv_time_mixing": {"must_precede": {"rmsnorm", "layernorm"}},
    "selective_scan": {"must_precede": {"rmsnorm", "layernorm"}},
    "n_way_sparse_router": {"must_precede": {"rmsnorm", "layernorm"}},
    # Mixing ops: 3.3% pass when pred=input, 1% when pred=add, 57% when pred=norm.
    # Every sequence-mixing op must be preceded by normalization.
    # Data: 3000 experiments, mixing ops after add → 1% S0 pass rate.
    "softmax_attention": {
        "must_precede": {
            "rmsnorm",
            "layernorm",
            "rope_rotate",
            "causal_mask",
            "sliding_window_mask",
        }
    },
    "linear_attention": {
        "must_precede": {"rmsnorm", "layernorm", "sliding_window_mask"}
    },
    "diff_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "graph_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "local_window_attn": {
        "must_precede": {"rmsnorm", "layernorm", "sliding_window_mask"}
    },
    "multi_head_mix": {"must_precede": {"rmsnorm", "layernorm"}},
    "gated_delta": {"must_precede": {"rmsnorm", "layernorm"}},
    "conv1d_seq": {"must_precede": {"rmsnorm", "layernorm"}},
    "latent_attention_compressor": {"must_precede": {"rmsnorm", "layernorm"}},
}


def pick_motif(
    rng: random.Random,
    motif_class: str,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Motif]:
    """Pick a random motif from the given class, weighted by lift or custom weights."""
    candidates = MOTIFS_BY_CLASS.get(motif_class)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    w = [weights.get(m.name, m.lift) if weights else m.lift for m in candidates]
    return rng.choices(candidates, weights=w, k=1)[0]


def pick_motif_from_classes(
    rng: random.Random,
    classes: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Motif]:
    """Pick a motif from any of the given classes."""
    pool: List[Motif] = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, []))
    if not pool:
        return None
    w = [weights.get(m.name, m.lift) if weights else m.lift for m in pool]
    return rng.choices(pool, weights=w, k=1)[0]


def resolve_step(
    step: MotifStep,
    rng: random.Random,
    *,
    prev_op: Optional[str] = None,
    next_op: Optional[str] = None,
    op_weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict]:
    """Resolve a motif step to a concrete (op_name, config) pair.

    Handles context-aware activation substitution for substitutable steps.
    When op_weights is provided, biases selection toward higher-weighted ops.
    """
    if step.substitutable and step.role == OpRole.ACTIVATE:
        candidates = _get_valid_activations(prev_op=prev_op, next_op=next_op)
        if op_weights and len(candidates) > 1:
            weights = [op_weights.get(c, 1.0) for c in candidates]
            op_name = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            op_name = rng.choice(candidates)
    else:
        op_name = step.op_name
    return op_name, dict(step.config)
