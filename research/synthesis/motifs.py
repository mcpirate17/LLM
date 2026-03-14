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
    motif_class: str        # e.g., "ffn_core", "attention_core", "ssm_core"
    steps: Tuple[MotifStep, ...]
    description: str = ""
    # Statistical evidence from mining
    support: int = 0        # Number of top performers containing this pattern
    avg_loss_ratio: float = 0.0
    lift: float = 1.0       # Enrichment in winners vs general population


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

ALL_MOTIF_CLASSES: FrozenSet[str] = frozenset({
    MOTIF_CLASS_FFN, MOTIF_CLASS_ATTENTION, MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV, MOTIF_CLASS_GATE, MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE, MOTIF_CLASS_MOE, MOTIF_CLASS_CHANNEL,
})


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
        support=157, avg_loss_ratio=0.063, lift=1.75,
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
        support=168, avg_loss_ratio=0.064, lift=1.61,
    ),
    Motif(
        name="ffn_fused_gelu",
        motif_class=MOTIF_CLASS_FFN,
        steps=(
            MotifStep("fused_linear_gelu", OpRole.PROJECT),
            MotifStep("linear_proj_down", OpRole.PROJECT),
        ),
        description="Fused Linear+GELU → contract (Triton-accelerated)",
        support=40, avg_loss_ratio=0.089, lift=1.2,
    ),

    # ── Attention cores (2.2-2.4x lift) ────────────────────────────
    Motif(
        name="attn_softmax",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("softmax_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Standard softmax self-attention → projection",
        support=13, avg_loss_ratio=0.142, lift=2.37,
    ),
    Motif(
        name="attn_linear",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("linear_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Linear attention → projection",
        support=9, avg_loss_ratio=0.128, lift=2.34,
    ),
    Motif(
        name="attn_graph",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("graph_attention", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Graph attention with learned adjacency → projection",
        support=12, avg_loss_ratio=0.120, lift=2.19,
    ),
    Motif(
        name="attn_local_window",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("local_window_attn", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Local windowed causal attention → projection",
        support=15, avg_loss_ratio=0.062, lift=1.5,
    ),
    Motif(
        name="attn_latent_compress",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("latent_attention_compressor", OpRole.MIX),
            MotifStep("linear_proj_up", OpRole.PROJECT),
        ),
        description="MLA-style KV compression → expand (best pair LR 0.040)",
        support=20, avg_loss_ratio=0.040, lift=1.8,
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
        support=36, avg_loss_ratio=0.161, lift=1.47,
    ),
    Motif(
        name="ssm_state_space",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("state_space", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="State-space mixer → projection",
        support=20, avg_loss_ratio=0.180, lift=1.3,
    ),
    Motif(
        name="ssm_ternary_scan",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("selective_scan", OpRole.MIX),
            MotifStep("ternary_projection", OpRole.PROJECT),
        ),
        description="Scan → ternary projection (6.75x trigram lift)",
        support=4, avg_loss_ratio=0.090, lift=6.75,
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
        support=15, avg_loss_ratio=0.071, lift=2.0,
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
        support=48, avg_loss_ratio=0.194, lift=1.75,
    ),
    Motif(
        name="conv_swiglu",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("swiglu_mlp", OpRole.GATE),
        ),
        description="Conv → SwiGLU (6.75x trigram lift, avg LR 0.071)",
        support=15, avg_loss_ratio=0.071, lift=6.75,
    ),

    # ── Gate cores ──────────────────────────────────────────────────
    Motif(
        name="gate_swiglu",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("swiglu_mlp", OpRole.GATE),
        ),
        description="SwiGLU (LLaMA FFN pattern, 2.49x lift)",
        support=41, avg_loss_ratio=0.166, lift=2.49,
    ),
    Motif(
        name="gate_linear",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("gated_linear", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Fused gate → projection (1.91x lift)",
        support=30, avg_loss_ratio=0.147, lift=1.91,
    ),
    Motif(
        name="gate_relu_routing",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("relu_gate_routing", OpRole.GATE),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="ReLU gate routing → projection",
        support=15, avg_loss_ratio=0.120, lift=1.5,
    ),

    # ── Normalization wrappers ──────────────────────────────────────
    Motif(
        name="norm_rms",
        motif_class=MOTIF_CLASS_NORM,
        steps=(
            MotifStep("rmsnorm", OpRole.NORMALIZE),
        ),
        description="RMSNorm (pre-norm wrapper)",
        support=200, avg_loss_ratio=0.150, lift=1.4,
    ),
    Motif(
        name="norm_layer",
        motif_class=MOTIF_CLASS_NORM,
        steps=(
            MotifStep("layernorm", OpRole.NORMALIZE),
        ),
        description="LayerNorm (pre-norm wrapper)",
        support=140, avg_loss_ratio=0.210, lift=1.41,
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
        support=36, avg_loss_ratio=0.094, lift=2.15,
    ),
    Motif(
        name="sparse_block",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("block_sparse_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Block-sparse linear → activation (2.09x lift)",
        support=67, avg_loss_ratio=0.144, lift=2.09,
    ),
    Motif(
        name="sparse_ternary",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("ternary_projection", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="1.58-bit ternary projection → activation (2.08x lift)",
        support=53, avg_loss_ratio=0.111, lift=2.08,
    ),
    Motif(
        name="sparse_semi_structured",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("semi_structured_2_4_linear", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="2:4 semi-structured sparse → activation (1.97x lift)",
        support=40, avg_loss_ratio=0.152, lift=1.97,
    ),

    # ── MoE cores (3.0-3.4x lift) ──────────────────────────────────
    Motif(
        name="moe_topk",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("moe_topk", OpRole.ROUTE),
        ),
        description="Sparse top-k MoE (3.09x lift)",
        support=27, avg_loss_ratio=0.115, lift=3.09,
    ),
    Motif(
        name="moe_2expert",
        motif_class=MOTIF_CLASS_MOE,
        steps=(
            MotifStep("moe_2expert", OpRole.ROUTE),
        ),
        description="Lightweight 2-expert MoE (2.59x lift)",
        support=63, avg_loss_ratio=0.114, lift=2.59,
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
        support=15, avg_loss_ratio=0.100, lift=2.5,
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
        support=10, avg_loss_ratio=0.200, lift=1.0,
    ),
    Motif(
        name="mix_fixed_point",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("fixed_point_iter", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Fixed-point iteration → projection",
        support=5, avg_loss_ratio=0.250, lift=0.8,
    ),
    Motif(
        name="mix_basis_expansion",
        motif_class=MOTIF_CLASS_CONV,
        steps=(
            MotifStep("basis_expansion", OpRole.PROJECT),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Basis expansion → activation (sinusoidal features)",
        support=8, avg_loss_ratio=0.200, lift=1.0,
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
        support=2, avg_loss_ratio=0.010, lift=1.9,
    ),
    Motif(
        name="tropical_attention_gate",
        motif_class=MOTIF_CLASS_ATTENTION,
        steps=(
            MotifStep("tropical_attention", OpRole.MIX),
            MotifStep("tropical_gate", OpRole.GATE),
            MotifStep("tropical_center", OpRole.NORMALIZE),
        ),
        description="Leaderboard-seeded tropical block: attention → gate → tropical centering",
        support=3, avg_loss_ratio=0.009, lift=2.1,
    ),
    Motif(
        name="clifford_attention_mix",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("clifford_attention", OpRole.MIX),
            MotifStep("grade_mix", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Leaderboard-seeded Clifford block: multivector attention → grade mixing → projection",
        support=3, avg_loss_ratio=0.008, lift=2.3,
    ),
    Motif(
        name="padic_hierarchy_block",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("padic_expand", OpRole.PROJECT),
            MotifStep("ultrametric_attention", OpRole.MIX),
            MotifStep("padic_residual", OpRole.RESIDUAL),
        ),
        description="Leaderboard-seeded p-adic hierarchy block: expansion → ultrametric attention → residual merge",
        support=4, avg_loss_ratio=0.009, lift=2.4,
    ),

    # ── Channel mixing cores ────────────────────────────────────────
    Motif(
        name="channel_rwkv",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("rwkv_channel", OpRole.MIX),
        ),
        description="RWKV channel mixing (2.41x lift, LR 0.103)",
        support=51, avg_loss_ratio=0.103, lift=2.41,
    ),
    Motif(
        name="channel_rwkv_time",
        motif_class=MOTIF_CLASS_CHANNEL,
        steps=(
            MotifStep("rwkv_time_mixing", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="RWKV time mixing → projection",
        support=20, avg_loss_ratio=0.130, lift=1.5,
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
        support=10, avg_loss_ratio=0.095, lift=3.5,
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
        support=8, avg_loss_ratio=0.110, lift=2.5,
    ),
    Motif(
        name="routed_ternary",
        motif_class=MOTIF_CLASS_SPARSE,
        steps=(
            MotifStep("ternary_projection", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Ternary projection → activation (4x efficiency, routing via template)",
        support=5, avg_loss_ratio=0.088, lift=4.0,
    ),
    Motif(
        name="merge_scan",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("token_merging", OpRole.MIX),
            MotifStep("selective_scan", OpRole.MIX),
            MotifStep("linear_proj", OpRole.PROJECT),
        ),
        description="Token merge → selective scan → proj (sequence compression)",
        support=6, avg_loss_ratio=0.105, lift=2.5,
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
        support=12, avg_loss_ratio=0.098, lift=3.0,
    ),
    Motif(
        name="conditional_skip",
        motif_class=MOTIF_CLASS_GATE,
        steps=(
            MotifStep("gated_linear", OpRole.GATE),
        ),
        description="Gated linear conditional compute (routing via template)",
        support=8, avg_loss_ratio=0.130, lift=2.0,
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
ACTIVATION_POOL: Tuple[str, ...] = ("gelu", "silu", "relu", "tanh", "sigmoid")


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


def resolve_step(step: MotifStep, rng: random.Random) -> Tuple[str, Dict]:
    """Resolve a motif step to a concrete (op_name, config) pair.

    Handles activation substitution for substitutable steps.
    """
    if step.substitutable and step.role == OpRole.ACTIVATE:
        op_name = rng.choice(ACTIVATION_POOL)
    else:
        op_name = step.op_name
    return op_name, dict(step.config)
