"""Parallel scan for a constant-matrix linear recurrence.

Removes the per-step Python loop from constant-transition state recurrences
(NM-F5 port-Hamiltonian is the first consumer; NM-F §5 native-kernel work). For

    h_t = P h_{t-1} + b_t ,   P in R^{D x D} constant across t,

the closed form is ``h_t = sum_{k<=t} P^{t-k} b_k``. Because ``P`` is *constant*,
a Kogge-Stone prefix scan needs only the fixed powers ``P^(2^j)`` — precomputed by
repeated squaring (``log2 L`` matmuls) — so each of the ``log2 L`` scan steps is a
single batched ``D x D`` matmul over the whole sequence: ``O(L log L D^2)`` work,
no ``D^3`` per element and no eigendecomposition. It is pure ``torch`` (matmul +
cat), so gradients flow for free and it is bit-comparable to the reference loop
(a native/CUDA kernel can replace it later behind the same signature).

Contrast with the repo's diagonal associative scan (``compiler_ops_sequence.
_parallel_associative_scan``): that combines per-channel scalar decays; this
combines a shared dense transition, the case a diagonal SSM cannot express.
"""

from __future__ import annotations

import torch


def constant_matrix_scan(
    transition: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    """``h_t = transition @ h_{t-1} + inputs_t`` for all ``t``, in parallel.

    Args:
        transition: ``(D, D)`` constant transition matrix ``P``.
        inputs: ``(B, L, D)`` per-step injections ``b_t`` (already ``h_t``'s row
            convention: ``h_new = h_prev @ P.T + b`` == ``P @ h_prev + b``).

    Returns:
        ``(B, L, D)`` states ``h_t``.
    """
    if transition.dim() != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError(
            f"transition must be square (D, D), got {tuple(transition.shape)}"
        )
    if inputs.dim() != 3 or inputs.shape[-1] != transition.shape[0]:
        raise ValueError(
            f"inputs must be (B, L, {transition.shape[0]}), got {tuple(inputs.shape)}"
        )

    seq_len = inputs.shape[1]
    h = inputs
    power = transition  # P^(stride)
    stride = 1
    while stride < seq_len:
        # Bridge every gap of `stride`: h_t += P^stride @ h_{t-stride} for t>=stride.
        # `h[:, : seq_len - stride]` are the current (already partially-scanned)
        # h_{t-stride}; matmul by power.T applies P^stride row-wise.
        contrib = h[:, : seq_len - stride] @ power.T
        h = torch.cat([h[:, :stride], h[:, stride:] + contrib], dim=1)
        power = power @ power  # P^(2*stride)
        stride *= 2
    return h
