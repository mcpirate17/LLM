"""
Matrix-memory LSTM cell (mLSTM, research §1.5).

Per external_research_2026-05-10.md §1.5, the xLSTM family replaces the
LSTM's scalar cell state with a matrix C ∈ R^{D×D} updated by a
key-value outer product: C_t = f_t · C_{t-1} + i_t · v_t k_t^⊤.
Retrieval at step t reads ``o_t · C_t @ q_t / max(n_t^⊤ q_t, 1)`` where
n_t is a normaliser tracked alongside C. This gives a recurrent
*key-addressable* state — a novel state form versus the diagonal SSM
matrices (Mamba, S5) and the cross-token attention scores already in the
substrate.

Forward shape contract:
    input  (B, S, D) → output (B, S, D)

Per-token computation:
    q_t, k_t, v_t = W_q x_t,  W_k x_t,  W_v x_t          # (D,)
    i_t = sigmoid(w_i · x_t + b_i)                       # scalar gate
    f_t = sigmoid(w_f · x_t + b_f)                       # scalar gate
    o_t = sigmoid(W_o x_t + b_o)                         # (D,) output gate
    C_t = f_t · C_{t-1} + i_t · v_t k_t^⊤                # (D, D) matrix state
    n_t = f_t · n_{t-1} + i_t · k_t                      # (D,) normaliser
    h_t = o_t * (C_t @ q_t) / max(|n_t^⊤ q_t|, 1)        # (D,)

Param count: 4·D² + 2·D (W_q, W_k, W_v, W_o + w_i, w_f). For D=128 that's
~66 K params, in line with other attention-class primitives.

Hot path is pure torch primitives (matmul, sigmoid, outer product) which
dispatch to native C++/CUDA. A fused chunkwise-recurrence kernel à la
flash-mlstm would be a future optimisation against the existing aria_core
ABI; not warranted at synthesis-tier granularity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def execute_mlstm_cell(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the mLSTM cell over a (B, S, D) tensor.

    Identity passthrough if the module hasn't been parameterised yet
    (matches the convention of the other mathspace ops).
    """
    if not hasattr(module, "W_q"):
        return x

    B, S, D = x.shape
    dtype = x.dtype
    device = x.device

    W_q = module.W_q.to(dtype)
    W_k = module.W_k.to(dtype)
    W_v = module.W_v.to(dtype)
    W_o = module.W_o.to(dtype)
    w_i = module.w_i.to(dtype)
    w_f = module.w_f.to(dtype)
    b_i = module.b_i.to(dtype)
    b_f = module.b_f.to(dtype)
    b_o = module.b_o.to(dtype)

    q = x @ W_q  # (B, S, D)
    k = x @ W_k
    v = x @ W_v
    o = torch.sigmoid(x @ W_o + b_o)  # (B, S, D)
    # Scalar gates per token.
    i_gate_raw = x @ w_i + b_i  # (B, S)
    f_gate_raw = x @ w_f + b_f
    i_gate = torch.sigmoid(i_gate_raw)
    f_gate = torch.sigmoid(f_gate_raw)

    C = torch.zeros(B, D, D, device=device, dtype=dtype)
    n = torch.zeros(B, D, device=device, dtype=dtype)
    outputs = []
    for t in range(S):
        q_t = q[:, t, :]  # (B, D)
        k_t = k[:, t, :]
        v_t = v[:, t, :]
        i_t = i_gate[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1) for broadcasting
        f_t = f_gate[:, t].unsqueeze(-1).unsqueeze(-1)
        # Matrix recurrence: outer product of v and k.
        vk = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(-2))  # (B, D, D)
        C = f_t * C + i_t * vk
        # Normaliser recurrence.
        n = f_gate[:, t].unsqueeze(-1) * n + i_gate[:, t].unsqueeze(-1) * k_t
        # Retrieve: C_t @ q_t, normalised.
        cq = torch.bmm(C, q_t.unsqueeze(-1)).squeeze(-1)  # (B, D)
        denom = (n * q_t).sum(dim=-1, keepdim=True).abs()
        denom = torch.clamp(denom, min=1.0)
        h_t = o[:, t, :] * (cq / denom)
        outputs.append(h_t)

    return torch.stack(outputs, dim=1)  # (B, S, D)
