"""NM-C7 — Recurrent-depth weight-shared refinement with a collapse-proof
non-softmax (p-adic Lorentzian) loop gate.

A ``[B, L, D] -> [B, L, D]`` feature-refinement layer that collapses ``n_layers``
of stacked dense blocks into ONE weight-shared block iterated to effective depth
``max_depth``::

    h_0 = x ;  h_k = W h_{k-1}            (shared block W, identity-at-init)
    out  = sum_{k=1..T} w_k(x) * h_k

The per-token depth weights ``w_k`` are NOT a softmax depth router. The
``recursive_depth_router`` / ``depth_weighted_proj`` lane learned depth with
``F.softmax(depth_logits)`` and 6/8 routers collapsed at scale (2026-06-29,
[[rdr_padic_scale_result_2026-06-30]]). Here ``w_k`` is the VALIDATED
collapse-proof gate from ``_op_padic_depth_route``: a bounded reciprocal
(Cauchy/Lorentzian) proximity between the token's INTRINSIC p-adic valuation
``v_p(x)`` (an ultrametric / tree-distance signal) and learnable depth anchors::

    val  = standardize(mean_D v_p(x))
    w_k  = lorentzian(val, anchor_k) / sum_j lorentzian(val, anchor_j)
    lorentzian(v, a) = 1 / (1 + (sharp * |v - a|)^2)      (inverse-distance, in [0,1])

— inverse-distance, not exponentiated dot-product, so the weights track the real
ultrametric spread of token hierarchy and structurally resist single-depth
collapse. Because the gate signal is the token's fixed ultrametric structure
(plus a learnable sharpness + anchors) rather than a free linear gate, it is the
p-adic analogue the softmax router cannot mimic.

Parameter cost (compaction = amplifier, not throttle): ONE shared block ``W``
(``D^2``) + ``max_depth`` anchors + 1 sharpness. Effective depth ``T`` comes from
recursion, not stacked weights, so per-layer weight VRAM scales **÷ n_layers** —
depth/width a novel non-QKV mechanism can spend to reach the scale where it
beats softmax ([[feedback_active_params_mean_non_embedding]]). The softmax
depth-router collapse that killed ``recursive_depth_router`` does not recur
because the gate is not a softmax.

Identity-at-init: ``W = I`` ⟹ ``h_k = x`` for all ``k`` ⟹ ``out = x`` for any
gate weights — a safe drop-in for any ``D``. Self-contained (imports only
``torch`` + ``math``) so it is measurable by
``PhysicsDescriptorProbe.describe_operator`` (NM-10). The p-adic valuation is a
faithful in-line replica of ``research.mathspaces.padic.padic_valuation``
(smooth ``v_p``), inlined to keep the module hermetic and NM-10-measurable;
registry wiring (``_init_recurrent_depth_refine`` + ``OP_DISPATCH`` +
``estimate_op_params``) is deferred until codex's in-flight
factorization/embedding work commits — see NM-C3/C5.

NM-C7 lane: ``research/notes/component_fab_compaction_lanes_2026-07-01.md``.
Composes with NM-C10 (recurse a memory-augmented block). Plan mirror:
``tasks/fab_novel_math_expansion_plan.md`` Tier D.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_PADIC_EPS = 1e-6
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_P = 2


def recurrent_depth_param_count(dim: int, max_depth: int = _DEFAULT_MAX_DEPTH) -> int:
    """Trainable params: one shared ``D x D`` block + ``max_depth`` anchors + 1 sharpness."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    return dim * dim + max_depth + 1


def _padic_valuation(x: torch.Tensor, p: int = _DEFAULT_P) -> torch.Tensor:
    """Smooth p-adic valuation ``v_p(x) = -log_p(|x|)``.

    Faithful replica of ``research.mathspaces.padic.padic_valuation``:
    ``smooth_abs = sqrt(x^2 + eps^2)``. Inlined (not imported) so this synthesis
    module stays self-contained on ``torch``+``math`` only and remains measurable
    by ``PhysicsDescriptorProbe`` without pulling the native/compiler stack.
    """
    log_p = math.log(p)
    smooth_abs = torch.sqrt(x * x + _PADIC_EPS * _PADIC_EPS)
    return -(torch.log(smooth_abs.clamp_min(_PADIC_EPS)) / log_p)


def _lorentzian_weights(
    val: torch.Tensor, anchors: torch.Tensor, log_sharp: torch.Tensor
) -> torch.Tensor:
    """Bounded-reciprocal (Cauchy/Lorentzian) proximity of ``val`` to each anchor.

    ``val``: ``(...,)`` standardized token valuations. ``anchors``: ``(k,)``.
    Returns ``(..., k)`` weights summing to 1 — inverse-distance, NOT softmax
    (gradient-safe as ``dist -> 0``, unlike a ``dist**-sharp`` pole). This is the
    exact collapse-proof gate of ``_op_padic_depth_route``.
    """
    sharp = torch.nn.functional.softplus(log_sharp.to(val.dtype)) + 0.5
    dist = (val.unsqueeze(-1) - anchors.to(val.dtype)).abs()  # (..., k)
    inv = 1.0 / (1.0 + (dist * sharp).pow(2))  # bounded reciprocal in [0, 1]
    return inv / inv.sum(dim=-1, keepdim=True)


class RecurrentDepthRefine(nn.Module):
    """Recurrent-depth weight-shared refinement; collapse-proof p-adic loop gate."""

    def __init__(
        self,
        dim: int,
        *,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        p: int = _DEFAULT_P,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if p < 2:
            raise ValueError(f"p must be a prime >= 2, got {p}")
        self.d = dim
        self.max_depth = int(max_depth)
        self.p = int(p)
        # Shared block W, identity-at-init ⟹ h_k = x ∀k ⟹ out = x (any gate).
        self.W = nn.Parameter(torch.eye(dim))
        # Learnable ultrametric depth anchors + sharpness (the p-adic gate knobs).
        self.depth_anchors = nn.Parameter(torch.arange(max_depth, dtype=torch.float32))
        self.route_log_sharpness = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return recurrent_depth_param_count(self.d, self.max_depth)

    def depth_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token p-adic Lorentzian weights over the ``max_depth`` iterations.

        The gate signal is the token's fixed ultrametric structure; only the
        anchors/sharpness are learned (so the gate cannot collapse to a single
        learned expert the way a softmax depth-router does).
        """
        val = _padic_valuation(x.float(), self.p).mean(dim=-1)  # (B, S)
        val = (val - val.mean()) / (val.std() + 1e-5)  # standardized
        return _lorentzian_weights(val, self.depth_anchors, self.route_log_sharpness)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., D) -> (..., D)``: shared-block recursion to depth ``max_depth``,
        p-adic-Lorentzian-weighted over depth (NOT a softmax depth router)."""
        # Recursive compositions h_k = W^k x (shared block ⟹ true weight-sharing).
        states: list[torch.Tensor] = []
        h = x
        for _ in range(self.max_depth):
            h = torch.einsum("ij,...j->...i", self.W, h)
            states.append(h)
        stacked = torch.stack(states, dim=-2)  # (..., max_depth, D)
        w = self.depth_weights(x).unsqueeze(-1)  # (..., max_depth, 1)
        return (w * stacked).sum(dim=-2)

    # ── verification helpers (tests only; never the hot path) ──────────────────

    def dense_weight(self) -> torch.Tensor:
        """Effective single-pass ``D x D`` equivalent when the gate collapses to
        one depth (symmetry with Monarch/Butterfly ``dense_matrix``); the real
        operator is depth-mixed, so this is the per-iteration block ``W``."""
        return self.W

    def shape(self) -> tuple[int, int]:
        return (self.d, self.d)
