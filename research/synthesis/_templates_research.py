"""Research templates — 0%% S1 fixes, zero-coverage ops, reference architectures."""

from __future__ import annotations

import random

from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SPARSE,
    MotifWeights,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _shuffle_wrap,
    _tpl_norm_op_motif_residual,
    _tpl_norm_op_residual,
)
from ._templates_core import tpl_residual_block, tpl_transformer_block


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

    try:
        up = graph.add_op("linear_proj_up", [normed], config={"out_dim": D * 2})
        # abs ensures non-negative input for sqrt
        abs_val = graph.add_op("abs", [up])
        sqrted = graph.add_op("sqrt", [abs_val])
        # Gate branch
        gate = graph.add_op("sigmoid", [up])
        gated = graph.add_op("mul", [sqrted, gate])
        down = graph.add_op("linear_proj_down", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, down])
    except ValueError:
        return down


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

    try:
        # Primary path: FFN with learned transform (the actual learner)
        up = graph.add_op("linear_proj_up", [normed], config={"out_dim": D * 4})
        activated = graph.add_op("gelu", [up])
        down = graph.add_op("linear_proj_down", [activated], config={"out_dim": D})

        # Side channel: reduce → expand → sigmoid gate
        reduced = graph.add_op(reduce_op, [normed])  # (B, S, 1)
        gate_proj = graph.add_op("linear_proj_up", [reduced], config={"out_dim": D})
        gate = graph.add_op("sigmoid", [gate_proj])

        # Gate modulates FFN output: features scaled by their summary
        gated = graph.add_op("mul", [down, gate])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, gated])
    except ValueError:
        return gated


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

    try:
        fused = graph.add_op("fused_linear_gelu", [normed], config={"out_dim": D})
        gate = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("sigmoid", [gate])
        gated = graph.add_op("mul", [fused, gate])
        down = graph.add_op("linear_proj_down", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, down])
    except ValueError:
        return down


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

    try:
        exped = graph.add_op("exp", [normed])
        # rmsnorm after exp prevents magnitude explosion
        stabilized = graph.add_op("rmsnorm", [exped])
        gate = graph.add_op("sigmoid", [normed])
        gated = graph.add_op("mul", [stabilized, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_integral_kernel_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → integral_kernel → proj → [FFN motif] → residual_add."""
    return _tpl_norm_op_motif_residual(
        graph, input_id, rng, weights, op_name="integral_kernel"
    )


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
    try:
        normed = graph.add_op("layernorm", [input_id])
        mixed = graph.add_op(
            "rwkv_channel",
            [normed],
            config={"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        )
        mid = graph.add_op("add", [input_id, mixed])
        norm2 = graph.add_op("layernorm", [mid])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, norm2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, norm2, ffn, rng) if ffn else norm2
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [mid, processed])
    except ValueError:
        return processed


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
    try:
        normed = graph.add_op("layernorm", [input_id])
        mixed = graph.add_op(
            "rwkv_channel",
            [normed],
            config={"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        )
        mid = graph.add_op("add", [input_id, mixed])
        norm2 = graph.add_op("layernorm", [mid])
        ffn_out = graph.add_op("swiglu_mlp", [norm2])
        ffn_out = _fix_dim(graph, ffn_out)
        return graph.add_op("add", [mid, ffn_out])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)


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
    try:
        normed = graph.add_op("rmsnorm", [input_id])
        mixed = graph.add_op(
            "rwkv_channel",
            [normed],
            config={"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        )
        mid = graph.add_op("add", [input_id, mixed])
        norm2 = graph.add_op("rmsnorm", [mid])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    sparse = _pick_compatible_motif_from_classes(
        graph, norm2, rng, [MOTIF_CLASS_SPARSE], weights
    )
    processed = _instantiate_motif(graph, norm2, sparse, rng) if sparse else norm2
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [mid, processed])
    except ValueError:
        return processed


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

    try:
        sig = graph.add_op("sigmoid", [normed])
        recip = graph.add_op("reciprocal", [sig])
        gated = graph.add_op("mul", [normed, recip])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


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

    try:
        # Log path: sigmoid bounds (0,1), log compresses to (-inf, 0)
        bounded = graph.add_op("sigmoid", [normed])
        logged = graph.add_op("log", [bounded])
        # Gate path: separate projection so gate learns different features
        gate_proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("sigmoid", [gate_proj])
        gated = graph.add_op("mul", [logged, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


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

    try:
        projected = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        signed = graph.add_op("sign_ste", [projected])
        gate = graph.add_op("sigmoid", [projected])
        gated = graph.add_op("mul", [signed, gate])
        out = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


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

    try:
        attended = graph.add_op("ultrametric_attention", [normed])
        expert = graph.add_op("moe_2expert", [attended])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, expert, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, expert, ffn, rng) if ffn else expert
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    try:
        mixed = graph.add_op("multi_head_mix", [normed], config={"n_heads": n_heads})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Downstream mixer benefits from head-organized features
    mixer = _pick_compatible_motif_from_classes(
        graph, mixed, rng, _MIXER_CLASSES, weights
    )
    if mixer:
        processed = _instantiate_motif(graph, mixed, mixer, rng)
    else:
        processed = mixed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        spiked = graph.add_op("lif_neuron", [normed])
        attended = graph.add_op("stdp_attention", [spiked])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    attended = _fix_dim(graph, attended)
    try:
        return graph.add_op("add", [input_id, attended])
    except ValueError:
        return attended


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

    try:
        rotated = graph.add_op("rope_rotate", [normed])
        attended = graph.add_op("softmax_attention", [rotated])
        post_norm = graph.add_op("rmsnorm", [attended])
        projected = graph.add_op("linear_proj", [post_norm], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

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
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        attended = graph.add_op("softmax_attention", [normed1])
        post_attn_norm = graph.add_op("rmsnorm", [attended])
        proj1 = graph.add_op("linear_proj", [post_attn_norm], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_transformer_block(graph, input_id, rng, weights)

    try:
        mid = graph.add_op("add", [input_id, proj1])
    except ValueError:
        mid = proj1

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    try:
        ffn = graph.add_op(
            "swiglu_mlp", [normed2], config={"mlp_ratio": rng.choice([2.0, 4.0])}
        )
    except (ValueError, KeyError):
        ffn = normed2

    ffn = _fix_dim(graph, ffn)
    try:
        return graph.add_op("add", [mid, ffn])
    except ValueError:
        return ffn


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
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    try:
        convolved = graph.add_op("conv1d_seq", [normed1])
        conv_norm = graph.add_op("rmsnorm", [convolved])
        scanned = graph.add_op("selective_scan", [conv_norm])
        proj1 = graph.add_op("linear_proj", [scanned], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_state_space_block(graph, input_id, rng, weights)

    try:
        mid = graph.add_op("add", [input_id, proj1])
    except ValueError:
        mid = proj1

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    try:
        ffn = graph.add_op(
            "swiglu_mlp", [normed2], config={"mlp_ratio": rng.choice([2.0, 4.0])}
        )
    except (ValueError, KeyError):
        ffn = normed2

    ffn = _fix_dim(graph, ffn)
    try:
        return graph.add_op("add", [mid, ffn])
    except ValueError:
        return ffn


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

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        routed = graph.add_op("hetero_moe", [proj])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        routed = graph.add_op("arch_router", [proj])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        routed = graph.add_op("compute_budget_router", [proj])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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

    try:
        proj_in = graph.add_op("linear_proj", [normed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Channel-shuffle → mixer → channel-unshuffle.
    # Data: SSM/conv motifs handle transposed features best (selective_scan 33%,
    # gated_linear 100% on small n). Broad _MIXER_CLASSES picks random attention
    # which fails on transposed layout.
    _CROSS_DIM_MIXER_CLASSES = ("ssm_core", "conv_core", "gate_core")
    mixed = _shuffle_wrap(graph, proj_in, rng, _CROSS_DIM_MIXER_CLASSES, weights)

    try:
        proj_out = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError):
        proj_out = _fix_dim(graph, mixed)

    try:
        return graph.add_op("add", [input_id, proj_out])
    except ValueError:
        return proj_out


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
        return tpl_residual_block(graph, input_id, rng, weights)

    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        split_a = graph.add_op("split2", [normed], config={"part": 0})
        split_b = graph.add_op("split2", [normed], config={"part": 1})
    except ValueError:
        return tpl_residual_block(graph, input_id, rng, weights)

    # Path A: sequence mixer on original feature order (half dim from split2).
    # Must use _BOTTLENECK_CLASSES — ops that adapt to input dim, not model_dim.
    from ._template_helpers import _BOTTLENECK_CLASSES

    mixer = _pick_compatible_motif_from_classes(
        graph, split_a, rng, list(_BOTTLENECK_CLASSES), weights
    )
    path_a = _instantiate_motif(graph, split_a, mixer, rng) if mixer else split_a

    # Path B: channel shuffle → FFN → channel unshuffle (half dim from split2).
    path_b = _shuffle_wrap(graph, split_b, rng, _BOTTLENECK_CLASSES, weights)

    try:
        merged = graph.add_op("concat", [path_a, path_b])
    except ValueError:
        return path_a

    merged = _fix_dim(graph, merged)
    try:
        return graph.add_op("add", [input_id, merged])
    except ValueError:
        return merged
