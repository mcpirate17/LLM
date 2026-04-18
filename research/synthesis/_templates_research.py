"""Research templates — 0%% S1 fixes, zero-coverage ops, reference architectures."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE,
    MotifWeights,
    TemplateBuildError,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _shuffle_wrap,
    _tpl_norm_op_motif_residual,
    _tpl_norm_op_residual,
    template_add_op as _add,
    template_add_residual as _residual,
)


# ── 0% S1 fix: dedicated templates for ops that work in isolation ────
# but fail in production search. Each template provides the specific
# architectural context where these ops contribute to learning.


def tpl_cumulative_sequence(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → cumsum → rmsnorm → proj → residual_add."""
    return _tpl_norm_op_residual(
        graph, input_id, rng, weights, op_name="cumsum", post_norm=True
    )


def tpl_sqrt_gated_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_up → sqrt → gate(sigmoid) → proj_down → residual_add.

    Sqrt as a bounded activation: compresses positive values (via abs→sqrt)
    while sigmoid provides a learned gate. The combination acts as a
    soft-magnitude attention over features.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    up = _add(
        graph,
        "linear_proj_up",
        [normed],
        {"out_dim": D * 2},
        context="sqrt_gated_ffn.up",
    )
    abs_val = _add(graph, "abs", [up], context="sqrt_gated_ffn.abs")
    sqrted = _add(graph, "sqrt", [abs_val], context="sqrt_gated_ffn.sqrt")
    gate = _add(graph, "sigmoid", [up], context="sqrt_gated_ffn.gate")
    gated = _add(graph, "mul", [sqrted, gate], context="sqrt_gated_ffn.mul")
    down = _add(
        graph,
        "linear_proj_down",
        [gated],
        {"out_dim": D},
        context="sqrt_gated_ffn.down",
    )
    return _residual(graph, input_id, down, context="sqrt_gated_ffn.output")


def tpl_reduce_attend(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → FFN_path ⊗ reduce_gate → proj → residual_add.

    Squeeze-and-excite style: reduce ops compute per-token feature summary
    as a side-channel gate, applied to a proper FFN path that carries the
    actual learning. Without the FFN, the reduce-gate alone has insufficient
    capacity (scalar → broadcast creates uniform per-feature scaling).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    reduce_op = rng.choice(["norm_last", "mean_last", "max_last", "sum_last"])

    up = _add(
        graph,
        "linear_proj_up",
        [normed],
        {"out_dim": D * 4},
        context="reduce_attend.up",
    )
    activated = _add(graph, "gelu", [up], context="reduce_attend.activated")
    down = _add(
        graph,
        "linear_proj_down",
        [activated],
        {"out_dim": D},
        context="reduce_attend.down",
    )
    reduced = _add(graph, reduce_op, [normed], context="reduce_attend.reduced")
    gate_proj = _add(
        graph,
        "linear_proj_up",
        [reduced],
        {"out_dim": D},
        context="reduce_attend.gate_proj",
    )
    gate = _add(graph, "sigmoid", [gate_proj], context="reduce_attend.gate")
    gated = _add(graph, "mul", [down, gate], context="reduce_attend.mul")
    return _residual(graph, input_id, gated, context="reduce_attend.output")


# ── 0% S1 fix round 2: attention, SSM, activation, and structural ops ──


def tpl_fused_gelu_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → fused_linear_gelu ⊗ sigmoid(linear_proj) → proj_down → residual_add.

    Gated FFN: fused_linear_gelu provides GELU-activated up-projection,
    a parallel linear_proj with sigmoid provides the gate. The element-wise
    product lets the network selectively suppress features (SwiGLU-style).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    fused = _add(
        graph,
        "fused_linear_gelu",
        [normed],
        {"out_dim": D},
        context="fused_gelu_ffn.fused",
    )
    gate = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="fused_gelu_ffn.gate_proj",
    )
    gate = _add(graph, "sigmoid", [gate], context="fused_gelu_ffn.gate")
    gated = _add(graph, "mul", [fused, gate], context="fused_gelu_ffn.mul")
    down = _add(
        graph,
        "linear_proj_down",
        [gated],
        {"out_dim": D},
        context="fused_gelu_ffn.down",
    )
    return _residual(graph, input_id, down, context="fused_gelu_ffn.output")


def tpl_exp_gated_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → exp → rmsnorm → sigmoid_gate → proj → residual_add.

    Exp as a soft attention mechanism: exp amplifies positive features,
    rmsnorm after exp controls magnitude (prevents explosion during training),
    sigmoid gate selects which amplified features pass through.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    exped = _add(graph, "exp", [normed], context="exp_gated_residual.exp")
    stabilized = _add(
        graph, "rmsnorm", [exped], context="exp_gated_residual.stabilized"
    )
    gate = _add(graph, "sigmoid", [normed], context="exp_gated_residual.gate")
    gated = _add(graph, "mul", [stabilized, gate], context="exp_gated_residual.mul")
    projected = _add(
        graph,
        "linear_proj",
        [gated],
        {"out_dim": D},
        context="exp_gated_residual.projected",
    )
    return _residual(graph, input_id, projected, context="exp_gated_residual.output")


def tpl_integral_kernel_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {integral_kernel | attention} → proj → [FFN motif] → residual_add.

    40% chance of adding an attention path parallel to the integral kernel.
    """
    from ._template_helpers import MOTIF_CLASS_ATTENTION

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    kernel = _add(
        graph, "integral_kernel", [normed], context="integral_kernel_block.kernel"
    )
    kernel = _add(
        graph,
        "linear_proj",
        [kernel],
        {"out_dim": D},
        context="integral_kernel_block.project",
    )

    # 40% chance: attention parallel path
    if rng.random() < 0.4:
        attn = _pick_compatible_motif(
            graph, normed, rng, MOTIF_CLASS_ATTENTION, weights
        )
        if attn:
            path_attn = _instantiate_motif(graph, normed, attn, rng)
            path_attn = _fix_dim(graph, path_attn)
            kernel = _residual(
                graph, kernel, path_attn, context="integral_kernel_block.attn_merge"
            )

    ffn = _pick_compatible_motif_from_classes(
        graph, kernel, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, kernel, ffn, rng) if ffn else kernel
    processed = _fix_dim(graph, processed)

    return _residual(graph, input_id, processed, context="integral_kernel_block.output")


def tpl_windowed_attention(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sliding_window_mask → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph,
        input_id,
        rng,
        weights,
        op_name="sliding_window_mask",
        op_config={"window_size": rng.choice([8, 16, 32])},
    )


def tpl_local_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → proj → [FFN motif] → residual_add."""
    D = graph.model_dim
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    return _tpl_norm_op_motif_residual(
        graph,
        input_id,
        rng,
        weights,
        op_name="local_window_attn",
        op_config={"window_size": rng.choice(choices)},
        motif_classes=(
            MOTIF_CLASS_SPARSE,
            MOTIF_CLASS_EFFICIENT_PROJ,
            MOTIF_CLASS_GUARDED_ACT,
        ),
    )


def tpl_state_space_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → state_space → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph, input_id, rng, weights, op_name="state_space"
    )


def tpl_rwkv_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """layernorm → rwkv_channel → add(input) → layernorm → [FFN motif] → add(mid).

    Double-norm RWKV pattern from 4-gram mining (n=11,447 programs):
      norm → rwkv_channel → add → norm → FFN → add
    The post-residual norm is the key structural insight — graphs with this
    pattern achieve loss_ratio ~0.05 vs ~0.6 without it.
    """
    normed = _add(graph, "layernorm", [input_id], context="rwkv_block.norm")
    mixed = _add(
        graph,
        "rwkv_channel",
        [normed],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="rwkv_block.mixed",
    )
    mid = _residual(graph, input_id, mixed, context="rwkv_block.mid")
    norm2 = _add(graph, "layernorm", [mid], context="rwkv_block.norm2")

    ffn = _pick_compatible_motif_from_classes(
        graph, norm2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, norm2, ffn, rng) if ffn else norm2
    processed = _fix_dim(graph, processed)

    return _residual(graph, mid, processed, context="rwkv_block.output")


def tpl_rwkv_double_norm(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """layernorm → rwkv_channel → add(input) → layernorm → swiglu_mlp → add(mid).

    Fixed high-confidence recipe — zero motif slots. Encodes the exact
    4-gram winner: layernorm → rwkv_channel → add → layernorm with swiglu_mlp
    as the FFN (best pairing from data mining).
    """
    normed = _add(graph, "layernorm", [input_id], context="rwkv_double_norm.norm")
    mixed = _add(
        graph,
        "rwkv_channel",
        [normed],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="rwkv_double_norm.mixed",
    )
    mid = _residual(graph, input_id, mixed, context="rwkv_double_norm.mid")
    norm2 = _add(graph, "layernorm", [mid], context="rwkv_double_norm.norm2")
    ffn_out = _add(graph, "swiglu_mlp", [norm2], context="rwkv_double_norm.ffn")
    ffn_out = _fix_dim(graph, ffn_out)
    return _residual(graph, mid, ffn_out, context="rwkv_double_norm.output")


def tpl_rwkv_sparse_chain(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → rwkv_channel → add(input) → rmsnorm → [sparse motif] → add(mid).

    Encodes the Var H architecture family. Uses rmsnorm (data shows rmsnorm
    also works well here at loss_ratio ~0.049) with a sparse motif slot for
    the FFN stage — keeps variety while locking in the RWKV backbone.
    """
    normed = _add(graph, "rmsnorm", [input_id], context="rwkv_sparse_chain.norm")
    mixed = _add(
        graph,
        "rwkv_channel",
        [normed],
        {"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        context="rwkv_sparse_chain.mixed",
    )
    mid = _residual(graph, input_id, mixed, context="rwkv_sparse_chain.mid")
    norm2 = _add(graph, "rmsnorm", [mid], context="rwkv_sparse_chain.norm2")

    sparse = _pick_compatible_motif_from_classes(
        graph, norm2, rng, [MOTIF_CLASS_SPARSE], weights
    )
    processed = _instantiate_motif(graph, norm2, sparse, rng) if sparse else norm2
    if graph.nodes[processed].op_name == "feature_sparsity":
        processed = _add(
            graph,
            "linear_proj",
            [processed],
            {"out_dim": graph.model_dim},
            context="rwkv_sparse_chain.sparse_bridge",
        )
    processed = _fix_dim(graph, processed)

    return _residual(graph, mid, processed, context="rwkv_sparse_chain.output")


def tpl_reciprocal_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sigmoid → reciprocal → gate(mul) → proj → residual_add.

    Reciprocal as inverse-attention: 1/(1+exp(-x)) → 1/sigmoid(x) maps
    confident features to ~1.0 and uncertain features to ~2.0, inverting
    the attention distribution. Sigmoid predecessor bounds reciprocal input.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    sig = _add(graph, "sigmoid", [normed], context="reciprocal_gated.sig")
    recip = _add(graph, "reciprocal", [sig], context="reciprocal_gated.recip")
    gated = _add(graph, "mul", [normed, recip], context="reciprocal_gated.gated")
    projected = _add(
        graph,
        "linear_proj",
        [gated],
        {"out_dim": D},
        context="reciprocal_gated.projected",
    )
    return _residual(graph, input_id, projected, context="reciprocal_gated.output")


def tpl_log_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sigmoid → log → gate(mul) → proj → residual_add.

    Log-compression: sigmoid guarantees positive input (range (0,1)),
    log compresses to (-inf, 0). Gate controls which log-compressed
    features pass through, preventing unbounded negative values from
    dominating the representation.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    bounded = _add(graph, "sigmoid", [normed], context="log_gated.bounded")
    logged = _add(graph, "log", [bounded], context="log_gated.logged")
    gate_dim = graph.nodes[logged].output_shape.dim
    gate_proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": gate_dim},
        context="log_gated.gate_proj",
    )
    gate = _add(graph, "sigmoid", [gate_proj], context="log_gated.gate")
    gated = _add(graph, "mul", [logged, gate], context="log_gated.gated")
    projected = _add(
        graph,
        "linear_proj",
        [gated],
        {"out_dim": D},
        context="log_gated.projected",
    )
    return _residual(graph, input_id, projected, context="log_gated.output")


def tpl_sign_ste_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → sign_ste → gate(sigmoid) → mul → proj → residual_add.

    Binary quantization via STE: sign binarizes activations while
    straight-through estimator passes gradients. Sigmoid gate controls
    which binarized features contribute, preventing sign from zeroing
    gradient signal entirely.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    projected = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="sign_ste_gated.projected",
    )
    signed = _add(graph, "sign_ste", [projected], context="sign_ste_gated.signed")
    gate = _add(graph, "sigmoid", [projected], context="sign_ste_gated.gate")
    gated = _add(graph, "mul", [signed, gate], context="sign_ste_gated.gated")
    out = _add(
        graph,
        "linear_proj",
        [gated],
        {"out_dim": D},
        context="sign_ste_gated.out",
    )
    return _residual(graph, input_id, out, context="sign_ste_gated.output")


def tpl_ultrametric_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → ultrametric_attention → moe_2expert → [FFN motif] → residual.

    Ultrametric (p-adic) attention achieves lr=0.010 — lowest loss of any
    attention variant. The proven pattern pairs it with moe_2expert for
    capacity. 34 of 76 historical attempts died from unstable_dynamics,
    so the norm predecessor is critical for input bounding.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attended = _add(
        graph,
        "ultrametric_attention",
        [normed],
        context="ultrametric_attention_block.attended",
    )
    expert = _add(
        graph, "moe_2expert", [attended], context="ultrametric_attention_block.expert"
    )

    ffn = _pick_compatible_motif_from_classes(
        graph, expert, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, expert, ffn, rng) if ffn else expert
    processed = _fix_dim(graph, processed)
    return _residual(
        graph, input_id, processed, context="ultrametric_attention_block.output"
    )


def tpl_diff_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → diff_attention → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph, input_id, rng, weights, op_name="diff_attention"
    )


def tpl_graph_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → graph_attention → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph, input_id, rng, weights, op_name="graph_attention"
    )


# ── Phase 3: Zero-coverage + reference templates ───────────────────


def tpl_chebyshev_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → chebyshev_spectral_mix → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph, input_id, rng, weights, op_name="chebyshev_spectral_mix"
    )


def tpl_kronecker_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → kronecker_linear → proj → [FFN motif] → residual_add."""
    D = graph.model_dim
    return _tpl_norm_op_motif_residual(
        graph,
        input_id,
        rng,
        weights,
        op_name="kronecker_linear",
        op_config={"out_dim": D},
    )


def tpl_multi_head_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → multi_head_mix → [mixer motif] → proj → residual_add.

    Multi-head feature rearrangement before a mixer — splits features into
    heads and applies learned mixing within each head. Best avg_loss (0.30)
    of any structural op. The head structure gives downstream attention/SSM
    ops pre-organized feature groups to work with.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    n_heads = rng.choice([2, 4, 8])
    mixed = _add(
        graph,
        "multi_head_mix",
        [normed],
        {"n_heads": n_heads},
        context="multi_head_mix_block.mixed",
    )

    # Downstream mixer benefits from head-organized features
    mixer = _pick_compatible_motif_from_classes(
        graph, mixed, rng, _MIXER_CLASSES, weights
    )
    if mixer:
        processed = _instantiate_motif(graph, mixed, mixer, rng)
    else:
        processed = mixed

    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="multi_head_mix_block.output")


def tpl_spiking_stdp_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → lif_neuron → stdp_attention → proj → residual_add.

    Full spiking pipeline: LIF encoding → STDP temporal attention → projection.
    STDP learns temporal correlations via spike-timing-dependent plasticity.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    spiked = _add(graph, "lif_neuron", [normed], context="spiking_stdp_block.spiked")
    attended = _add(
        graph, "stdp_attention", [spiked], context="spiking_stdp_block.attended"
    )

    attended = _fix_dim(graph, attended)
    return _residual(graph, input_id, attended, context="spiking_stdp_block.output")


def tpl_rope_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → rope_rotate → softmax_attention → proj → [FFN motif] → residual_add.

    RoPE (Rotary Position Embedding) + attention — standard in modern LLMs.
    rope_rotate injects relative position info before attention.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    rotated = _add(
        graph, "rope_rotate", [normed], context="rope_attention_block.rotated"
    )
    attended = _add(
        graph, "softmax_attention", [rotated], context="rope_attention_block.attended"
    )
    post_norm = _add(
        graph, "rmsnorm", [attended], context="rope_attention_block.post_norm"
    )
    projected = _add(
        graph,
        "linear_proj",
        [post_norm],
        {"out_dim": D},
        context="rope_attention_block.projected",
    )

    # Data: sparse/routing motifs dramatically outperform random FFN here.
    # routed_ternary 25% S1, bottleneck_sparse 17% vs 10% baseline.
    _ROPE_FFN_CLASSES = ("sparse_core", "efficient_proj", "gate_core")
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_ROPE_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="rope_attention_block.output")


def tpl_gpt2_reference(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → softmax_attention → norm → proj → add → norm → swiglu_mlp → add.

    Canonical GPT-2 transformer block: causal self-attention + MLP.
    softmax_attention handles causality internally (is_causal=True).
    This is the baseline floor — every novel architecture should beat this.
    """
    D = graph.model_dim
    # Attention sub-block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    attended = _add(
        graph, "softmax_attention", [normed1], context="gpt2_reference.attended"
    )
    post_attn_norm = _add(
        graph, "rmsnorm", [attended], context="gpt2_reference.post_attn_norm"
    )
    proj1 = _add(
        graph,
        "linear_proj",
        [post_attn_norm],
        {"out_dim": D},
        context="gpt2_reference.proj1",
    )
    mid = _residual(graph, input_id, proj1, context="gpt2_reference.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffn = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 4.0])},
        context="gpt2_reference.ffn",
    )

    ffn = _fix_dim(graph, ffn)
    return _residual(graph, mid, ffn, context="gpt2_reference.output")


def tpl_mamba_reference(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv1d_seq → selective_scan → proj → add → norm → swiglu_mlp → add.

    Canonical Mamba block: depthwise conv → selective state-space → MLP.
    SSM baseline — linear-time sequence mixing vs quadratic attention.
    """
    D = graph.model_dim
    # SSM sub-block
    normed1 = _add(graph, "rmsnorm", [input_id], context="mamba_reference.norm1")

    convolved = _add(
        graph, "conv1d_seq", [normed1], context="mamba_reference.convolved"
    )
    conv_norm = _add(graph, "rmsnorm", [convolved], context="mamba_reference.conv_norm")
    scanned = _add(
        graph, "selective_scan", [conv_norm], context="mamba_reference.scanned"
    )
    mid = _residual(graph, input_id, scanned, context="mamba_reference.mid")

    # FFN sub-block
    normed2 = _add(graph, "rmsnorm", [mid], context="mamba_reference.norm2")

    ffn = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": rng.choice([2.0, 4.0])},
        context="mamba_reference.ffn",
    )

    ffn = _fix_dim(graph, ffn)
    return _residual(graph, mid, ffn, context="mamba_reference.output")


# ── True Routing Templates ───────────────────────────────────────────


def tpl_hetero_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → hetero_moe → norm → [FFN motif] → residual add.

    Heterogeneous MoE with attention+conv+SSM experts.
    Pre-projection provides learnable routing features;
    post-FFN adds capacity to interpret routed output.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="hetero_moe_block.proj",
    )
    routed = _add(graph, "hetero_moe", [proj], context="hetero_moe_block.routed")

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="hetero_moe_block.output")


def tpl_arch_router_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → arch_router → norm → [FFN motif] → residual add.

    Architecture router selects transformer/mamba/MLP styles per token.
    Pre-projection learns routing features; post-FFN interprets output.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="arch_router_block.proj",
    )
    routed = _add(graph, "arch_router", [proj], context="arch_router_block.routed")

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="arch_router_block.output")


def tpl_compute_budget_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → compute_budget_router → norm → [FFN motif] → residual add.

    Compute budget router assigns tokens to cheap/medium/expensive tiers.
    Pre-projection provides routing signal; post-FFN processes routed output.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="compute_budget_block.proj",
    )
    routed = _add(
        graph,
        "compute_budget_router",
        [proj],
        context="compute_budget_block.routed",
    )

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    return _residual(graph, input_id, processed, context="compute_budget_block.output")


# ── Cross-Dimension Templates (transpose_sd channel interleave) ─────


def tpl_cross_dim_mixer(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → transpose_sd → [mixer motif] → transpose_sd → proj → residual.

    Channel interleave before a sequence mixer lets it see a permuted
    feature ordering, then a second interleave restores the original
    feature semantics before projection. Analogous to ShuffleNet's
    channel shuffle between group convolutions.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    proj_in = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": D},
        context="cross_dim_mixer.proj_in",
    )

    # Channel-shuffle → mixer → channel-unshuffle.
    # Data: SSM/conv motifs handle transposed features best (selective_scan 33%,
    # gated_linear 100% on small n). Broad _MIXER_CLASSES picks random attention
    # which fails on transposed layout.
    _CROSS_DIM_MIXER_CLASSES = ("ssm_core", "conv_core", "gate_core")
    mixed = _shuffle_wrap(graph, proj_in, rng, _CROSS_DIM_MIXER_CLASSES, weights)

    proj_out = _add(
        graph,
        "linear_proj",
        [mixed],
        {"out_dim": D},
        context="cross_dim_mixer.proj_out",
    )
    return _residual(graph, input_id, proj_out, context="cross_dim_mixer.output")


def tpl_dual_axis_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → split2 → {mixer_a | transpose_sd → FFN → transpose_sd} → concat → proj → residual.

    Dual-axis: path_a applies a sequence mixer on the original feature
    order, path_b applies an FFN on channel-shuffled features. The two
    perspectives are concatenated and projected back to model_dim.
    """
    if graph.nodes[input_id].output_shape.dim < 16:
        raise TemplateBuildError("dual_axis_block: requires input dim >= 16")

    normed = _add(graph, "rmsnorm", [input_id], context="dual_axis_block.norm")

    split_a = _add(
        graph, "split2", [normed], {"part": 0}, context="dual_axis_block.split_a"
    )
    split_b = _add(
        graph, "split2", [normed], {"part": 1}, context="dual_axis_block.split_b"
    )
    path_a = _add(graph, "rmsnorm", [split_a], context="dual_axis_block.path_a_norm")
    path_a = _add(graph, "conv1d_seq", [path_a], context="dual_axis_block.path_a")
    path_b = _add(
        graph,
        "linear_proj",
        [split_b],
        {"out_dim": graph.nodes[split_b].output_shape.dim},
        context="dual_axis_block.path_b",
    )
    path_b = _add(graph, "gelu", [path_b], context="dual_axis_block.path_b_act")

    merged = _add(graph, "concat", [path_a, path_b], context="dual_axis_block.merged")

    merged = _fix_dim(graph, merged)
    return _residual(graph, input_id, merged, context="dual_axis_block.output")
