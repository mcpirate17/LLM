"""Core validated motifs: FFN, attention, SSM, conv, gate, norm, channel.

Derived from motif_mining_report.md findings.
Each motif has statistical backing from the top-performer pool.
"""

from __future__ import annotations

from typing import Tuple

from ._motif_types import (
    Motif,
    MotifStep,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
)
from .op_roles import OpRole

CORE_MOTIFS: Tuple[Motif, ...] = (
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
        lift=0.5,  # Demoted: 2.9% S1 rate in production — broken
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
        lift=4.0,  # Boosted: 27.5% S1 rate — second-best attention variant
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
        lift=4.0,  # Boosted: 30.2% S1 rate — best attention variant
    ),
    # ── SSM / state-space cores ─────────────────────────────────────
    Motif(
        name="ssm_selective_scan",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE),
            MotifStep("selective_scan", OpRole.MIX),
        ),
        description="Mamba-style conv preconditioning → SiLU → selective scan",
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
        name="ssm_scan_gelu",
        motif_class=MOTIF_CLASS_SSM,
        steps=(
            MotifStep("conv1d_seq", OpRole.PROJECT),
            MotifStep("silu", OpRole.ACTIVATE),
            MotifStep("selective_scan", OpRole.MIX),
            MotifStep("gelu", OpRole.ACTIVATE, substitutable=True),
        ),
        description="Conv → SiLU → scan → activation (safe adaptive-SSM refinement)",
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
)
