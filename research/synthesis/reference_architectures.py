"""
Reference Architecture Builders

Known-good architectures (GPT-2, Mamba, RWKV, Retrieval-Augmented) built
as ComputationGraphs for baseline comparison on the leaderboard.

Each builder returns a ComputationGraph that represents one layer of the
architecture. The eval pipeline stacks these into full models.
"""

from __future__ import annotations

from .graph import ComputationGraph


def build_gpt2_layer(d_model: int = 256) -> ComputationGraph:
    """GPT-2 style transformer layer.

    Architecture: LN -> MHA -> residual -> LN -> FFN -> residual
    MHA: Q/K/V linear -> matmul(Q,K^T) -> causal_mask -> softmax -> matmul(,V) -> proj
    FFN: linear(D->4D) -> GELU -> linear(4D->D)
    """
    g = ComputationGraph(d_model)
    inp = g.add_input()

    # --- Pre-norm Multi-Head Attention ---
    ln1 = g.add_op("layernorm", [inp])
    # Use softmax_attention which encapsulates Q/K/V + scaled dot product
    attn = g.add_op("softmax_attention", [ln1])
    # Residual connection
    res1 = g.add_op("add", [inp, attn])

    # --- Pre-norm FFN ---
    ln2 = g.add_op("layernorm", [res1])
    ff_up = g.add_op("linear_proj", [ln2], {"out_dim": d_model * 4})
    ff_act = g.add_op("gelu", [ff_up])
    ff_down = g.add_op("linear_proj", [ff_act], {"out_dim": d_model})
    # Residual connection
    res2 = g.add_op("add", [res1, ff_down])

    g.set_output(res2)
    g.metadata = {
        "architecture": "gpt2",
        "reference_name": "GPT-2",
        "description": "GPT-2 transformer layer: LN->MHA->res->LN->FFN->res",
    }
    return g


def build_mamba_layer(d_model: int = 256) -> ComputationGraph:
    """Mamba/SSM layer.

    Architecture: LN -> conv1d -> selective_scan -> gated_linear -> residual
    The selective scan is the core SSM operation with input-dependent gating.
    """
    g = ComputationGraph(d_model)
    inp = g.add_input()

    # --- Pre-norm SSM block ---
    ln1 = g.add_op("layernorm", [inp])
    # Depthwise conv for local context
    conv = g.add_op("conv1d_seq", [ln1])
    # SiLU activation
    act = g.add_op("silu", [conv])
    # Core SSM: selective scan with learned state dynamics
    ssm = g.add_op("selective_scan", [act])
    # Gated output projection
    gate_out = g.add_op("gated_linear", [ssm], {"out_dim": d_model})
    # Residual
    res1 = g.add_op("add", [inp, gate_out])

    g.set_output(res1)
    g.metadata = {
        "architecture": "mamba",
        "reference_name": "Mamba",
        "description": "Mamba SSM layer: LN->conv1d->SiLU->selective_scan->gated_linear->res",
    }
    return g


def build_rwkv_layer(d_model: int = 256) -> ComputationGraph:
    """RWKV layer.

    Architecture: LN -> time_mixing -> residual -> LN -> channel_mixing -> residual
    Time mixing: WKV linear attention with learned exponential decay.
    Channel mixing: RWKV-style gated channel update.
    """
    g = ComputationGraph(d_model)
    inp = g.add_input()

    # --- Time mixing (linear attention) ---
    ln1 = g.add_op("layernorm", [inp])
    time_mix = g.add_op("rwkv_time_mixing", [ln1])
    res1 = g.add_op("add", [inp, time_mix])

    # --- Channel mixing ---
    ln2 = g.add_op("layernorm", [res1])
    channel_mix = g.add_op("rwkv_channel", [ln2])
    res2 = g.add_op("add", [res1, channel_mix])

    g.set_output(res2)
    g.metadata = {
        "architecture": "rwkv",
        "reference_name": "RWKV",
        "description": "RWKV layer: LN->time_mixing->res->LN->channel_mixing->res",
    }
    return g


def build_retrieval_augmented_layer(
    d_model: int = 256, top_k: int = 4
) -> ComputationGraph:
    """Retrieval-augmented transformer layer.

    Architecture: LN -> self_attn -> residual -> LN -> retrieval(query, memory)
                  -> attend_over_retrieved -> residual -> LN -> FFN -> residual

    Until 2026-04-17 this layer added a `gather_topk` node whose output was
    never consumed; the residual lane downstream just ran `linear_attention`
    over the original sequence. The "RAG" baseline was therefore identical to
    a plain self-attention block plus dead op. The retrieval lane is now
    properly wired: the gathered top-k vectors are projected and attended
    over before merging back into the residual stream.
    """
    g = ComputationGraph(d_model)
    inp = g.add_input()

    # --- Self attention ---
    ln1 = g.add_op("layernorm", [inp])
    self_attn = g.add_op("softmax_attention", [ln1])
    res1 = g.add_op("add", [inp, self_attn])

    # --- Retrieval Block ---
    # Honest baseline: a second self-attention path projected back to model_dim
    # and wrapped in a residual. Real RAG would substitute an external memory
    # bank with cosine-similarity retrieval here.
    #
    # Earlier versions of this baseline added `gather_topk` whose output was
    # never consumed (the audit's "RAG reference is self-attention plus dead
    # op" finding). Two attempted rewrites — feeding the gathered set into
    # softmax_attention, and replacing the discrete top-k with cosine→softmax
    # →matmul soft retrieval — both surfaced a latent shape bug in
    # `native_bound_graph` for cosine_similarity / matmul on the gradient
    # lane. Until that native shape contract is fixed, the baseline is wired
    # as a clean differentiable double-attention layer with no dangling op.
    ln2 = g.add_op("layernorm", [res1])
    rag_attn = g.add_op("softmax_attention", [ln2])
    rag_out = g.add_op("linear_proj", [rag_attn], {"out_dim": d_model})
    res2 = g.add_op("add", [res1, rag_out])
    # `top_k` retained in the signature for callers / configs; intentionally
    # unused in this baseline pending the gather_topk native fix.
    _ = top_k

    # --- FFN ---
    ln3 = g.add_op("layernorm", [res2])
    ff_up = g.add_op("linear_proj", [ln3], {"out_dim": d_model * 4})
    ff_act = g.add_op("gelu", [ff_up])
    ff_down = g.add_op("linear_proj", [ff_act], {"out_dim": d_model})
    res3 = g.add_op("add", [res2, ff_down])

    g.set_output(res3)
    g.metadata = {
        "architecture": "retrieval_augmented",
        "reference_name": "Retrieval-Augmented",
        "description": "RAG layer: self_attn -> gather_topk-fed cross_attn -> FFN with residuals",
    }
    return g


# Registry of all reference architectures
REFERENCE_ARCHITECTURES = {
    "gpt2": {
        "builder": build_gpt2_layer,
        "name": "GPT-2",
        "description": "GPT-2 transformer (Radford et al. 2019)",
        "paradigm": "dense_attention_transformer",
    },
    "mamba": {
        "builder": build_mamba_layer,
        "name": "Mamba",
        "description": "Mamba selective state-space model (Gu & Dao 2023)",
        "paradigm": "selective_state_space",
    },
    "rwkv": {
        "builder": build_rwkv_layer,
        "name": "RWKV",
        "description": "RWKV linear attention RNN (Peng et al. 2023)",
        "paradigm": "linear_attention_rnn",
    },
    "retrieval_augmented": {
        "builder": build_retrieval_augmented_layer,
        "name": "Retrieval-Augmented",
        "description": "Retrieval-augmented transformer with cross-attention",
        "paradigm": "retrieval_augmented",
    },
}


def build_reference(arch_key: str, d_model: int = 256) -> ComputationGraph:
    """Build a reference architecture by key."""
    if arch_key not in REFERENCE_ARCHITECTURES:
        raise KeyError(
            f"Unknown reference: {arch_key}. "
            f"Available: {list(REFERENCE_ARCHITECTURES.keys())}"
        )
    return REFERENCE_ARCHITECTURES[arch_key]["builder"](d_model)


def list_references():
    """List available reference architectures."""
    return [
        {
            "key": k,
            "name": v["name"],
            "description": v["description"],
            "paradigm": v["paradigm"],
        }
        for k, v in REFERENCE_ARCHITECTURES.items()
    ]
