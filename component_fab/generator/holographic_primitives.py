"""Holographic (VSA/HRR) binding lane — non-QKV associative binding (W1 wall).

Compositional multi-slot binding is the field-wide wall: nearly every non-QKV
mechanism collapses to chance at 2+ key/value triples. This lane attacks it with
Holographic Reduced Representations (Plate 1995) / Vector-Symbolic Algebra: bind
a key and value by **circular convolution** and superpose the bound pairs into a
causal associative memory; retrieve by **circular correlation** (approximate
unbind) with a query. Binding is algebraic — an FFT product, not a softmax score —
so there is no convex token average and no query/key attention matrix.

    bind(k, v)      = irfft( rfft(k) · rfft(v) )        (circular convolution)
    unbind(q, m)    = irfft( conj(rfft(q)) · rfft(m) )  (circular correlation)

For (near-)unit random vectors, ``unbind(k, bind(k, v)) ≈ v`` up to crosstalk
noise that grows with the number of superposed pairs (VSA capacity ≈ D / #pairs).
The causal memory ``m_t = Σ_{s≤t} ρ**(t-s) bind(k_s, v_s)`` reuses the shared
chunked decay scan, so training is O(L·chunk) and inference streams with O(1)
state (no growing KV cache).

Efficiency angle: binding/unbinding are O(D log D) FFTs, and the mechanism keeps
only ``O(D)`` state — a param- and memory-light alternative to a dense attention
matrix.
"""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator._causal_scan import causal_decay_context


def circular_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two vectors by circular convolution over the last dim (FFT, O(D log D))."""
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(fa * fb, n=a.shape[-1], dim=-1)


def circular_unbind(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """Approximately unbind ``query`` from ``memory`` by circular correlation."""
    fq = torch.fft.rfft(query, dim=-1)
    fm = torch.fft.rfft(memory, dim=-1)
    return torch.fft.irfft(torch.conj(fq) * fm, n=query.shape[-1], dim=-1)


class HolographicBindingLane(nn.Module):
    """Causal VSA/HRR associative-memory mixer (non-QKV binding).

    Projects each token to a key/value/query, binds key⊛value, superposes into a
    causal decayed memory, and unbinds the query to read associated values. This
    is not attention: there is no score normalization and no convex token
    averaging — retrieval is algebraic (FFT correlation). Anti-softmax-twin by
    construction (circular convolution does not preserve token-constant inputs).
    Finite forward/backward at init; the gated residual starts at ``tanh(0.5)``
    so the lane is non-degenerate.
    """

    def __init__(self, dim: int, decay_init: float = 0.9) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("HolographicBindingLane requires dim >= 2")
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        self.dim = dim
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        logit = torch.logit(torch.tensor(decay_init))
        self.log_decay = nn.Parameter(torch.full((dim,), float(logit)))
        self.mix_gate = nn.Parameter(torch.full((dim,), 0.5))

    def _keys_values_queries(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unit-normalize keys/queries: HRR retrieval is cleanest when the binding
        # vectors are ~unit norm (so the correlation inverse is well-conditioned).
        k = torch.nn.functional.normalize(self.k_proj(x), dim=-1)
        q = torch.nn.functional.normalize(self.q_proj(x), dim=-1)
        v = self.v_proj(x)
        return k, v, q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, v, q = self._keys_values_queries(x)
        bound = circular_bind(k, v)  # [B, L, D] key⊛value
        decay = torch.sigmoid(self.log_decay).clamp(1e-4, 1 - 1e-4)
        memory = causal_decay_context(bound, decay)  # causal superposition
        read = circular_unbind(q, memory)  # retrieve associated values
        return x + torch.tanh(self.mix_gate) * self.out_proj(read)
