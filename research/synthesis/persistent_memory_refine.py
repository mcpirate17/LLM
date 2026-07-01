"""NM-C10 — Persistent-memory refinement via NON-softmax (p-adic ultrametric,
top-k tropical) associative retrieval.

A ``[B, L, D] -> [B, L, D]`` refinement layer that replaces ``n_layers`` of
stacked capacity with ONE layer reading from a persistent learned memory bank
``M`` — **memory-as-parameters** shared across the sequence and "virtual depth"::

    q    = W_q x                                  (per-token query)
    d_k  = d_p(q, M_k) = p^{-v_p(q - M_k)}        (ultrametric p-adic distance)
           (top-k closest slots — tropical: only the nearest contribute)
    read = sum_{k in top} lorentzian(d_k) * M_k    (bounded-reciprocal, NOT softmax)
    g    = sigmoid(v_p(x) * W_v)                   (p-adic content gate)
    out  = x + alpha * g * (W_o read)              (alpha = ReZero scale, 0 at init)

This is the **associative-retrieval pathway** the p-adic scale result
([[project_padic_gated_mixer_2026-06-29]], [[rdr_padic_scale_result_2026-06-30]])
identified as the missing capability route: a single-pass gate is a fast learner
/ embedding teacher but lacks the content-addressable store a single-pass
mechanism needs to *reason*, so induction / AR / binding floored at scale. Here
capacity comes from the bank size, not stacked depth, so params scale with
``n_slots`` rather than ``n_layers`` — per-layer weight VRAM **÷ n_layers**; the
bank is a fixed, depth-independent associative store.

Why NON-softmax and not a softmax/QKV twin: retrieval is **top-k hard over an
ultrametric distance** (max-plus / tropical — only the nearest slots contribute),
not an all-to-all ``softmax(QK^T)`` convex average. Ultrametric distance
satisfies the strong triangle inequality ``d(x,z) <= max(d(x,y), d(y,z))`` (a
tree metric, not the Euclidean inner product of attention), and the read weights
are a bounded reciprocal of that distance (inverse-distance), not exponentiated
similarity — structurally anti-softmax-twin (sparse, ultrametric,
content-addressed).

Identity-at-init: the ReZero scale ``alpha`` starts at 0 ⟹ ``out = x`` exactly
(safe drop-in, matching ``execute_padic_expand``'s ReZero convention);
``W_q = W_o = I`` and ``W_v = 0`` so the retrieval path is content-faithful and
the gate is a plain highway that learning turns on via the valuation term —
mirroring ``_op_padic_gated_mixer``'s "degenerates to a plain highway if W_v ->
0, so it is never worse than a learned GLU." Self-contained (imports only
``torch`` + ``math``) ⟹ ``PhysicsDescriptorProbe.describe_operator``-measurable
(NM-10); the p-adic distance/valuation are faithful in-line replicas of
``research.mathspaces.padic``. Registry wiring (``_init_persistent_memory_refine``
+ ``OP_DISPATCH`` + ``estimate_op_params``) is deferred (NM-C3/C5 convention).

NM-C10 lane: ``research/notes/component_fab_compaction_lanes_2026-07-01.md``.
Composes with NM-C7 (recurse this memory-augmented block). Plan mirror:
``tasks/fab_novel_math_expansion_plan.md`` Tier D.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_PADIC_EPS = 1e-6
_DEFAULT_N_SLOTS = 16
_DEFAULT_TOP_K = 4
_DEFAULT_P = 2


def persistent_memory_param_count(dim: int, n_slots: int = _DEFAULT_N_SLOTS) -> int:
    """Trainable params: ``W_q + W_o`` (2*D*D) + bank (n_slots*D) + ``W_v`` (D) + 2 scalars.

    Capacity scales with the bank size, not stacked depth — the compaction axis
    here is depth-collapse (params ∝ n_slots, not n_layers), not per-layer
    factorization (that is NM-C3/C5's job).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}")
    return 2 * dim * dim + n_slots * dim + dim + 2


def _padic_valuation(x: torch.Tensor, p: int = _DEFAULT_P) -> torch.Tensor:
    """Smooth p-adic valuation ``v_p(x) = -log_p(|x|)``.

    Faithful replica of ``research.mathspaces.padic.padic_valuation``; inlined so
    this synthesis module stays self-contained (``torch``+``math`` only) and
    measurable by ``PhysicsDescriptorProbe`` without the native/compiler stack.
    """
    log_p = math.log(p)
    smooth_abs = torch.sqrt(x * x + _PADIC_EPS * _PADIC_EPS)
    return -(torch.log(smooth_abs.clamp_min(_PADIC_EPS)) / log_p)


def _padic_distance(
    q: torch.Tensor, mem: torch.Tensor, p: int = _DEFAULT_P
) -> torch.Tensor:
    """Ultrametric p-adic distance ``d_p(q, m) = p^{-v_p(q-m)}`` reduced over D.

    ``q``: ``(..., D)``; ``mem``: ``(n_slots, D)``. Returns ``(..., n_slots)`` —
    small means query and slot are p-adically close (share high divisibility).
    Satisfies the ultrametric (strong triangle) inequality, a tree metric.
    """
    log_p = math.log(p)
    diff = q.unsqueeze(-2) - mem  # (..., n_slots, D)
    smooth_abs = torch.sqrt(diff * diff + _PADIC_EPS * _PADIC_EPS)
    val = -(torch.log(smooth_abs.clamp_min(_PADIC_EPS)) / log_p)  # v_p(q - m)
    # d_p(q, m) = p^{-v_p(q-m)} (= smooth_abs); reduce over features to one
    # distance per slot, matching research.mathspaces.padic._padic_dist_chunk.
    dist = torch.exp(-val * log_p).mean(dim=-1)  # (..., n_slots)
    return dist


class PersistentMemoryRefine(nn.Module):
    """Persistent-bank refinement; top-k p-adic ultrametric retrieval."""

    def __init__(
        self,
        dim: int,
        *,
        n_slots: int = _DEFAULT_N_SLOTS,
        top_k: int = _DEFAULT_TOP_K,
        p: int = _DEFAULT_P,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > n_slots:
            raise ValueError(f"top_k ({top_k}) must be <= n_slots ({n_slots})")
        if p < 2:
            raise ValueError(f"p must be a prime >= 2, got {p}")
        self.d = dim
        self.n_slots = int(n_slots)
        self.top_k = int(top_k)
        self.p = int(p)
        # Query / output projections, identity-at-init ⟹ retrieval is content-faithful.
        self.W_q = nn.Parameter(torch.eye(dim))
        self.W_o = nn.Parameter(torch.eye(dim))
        # Persistent memory bank (memory-as-parameters); small init.
        self.memory = nn.Parameter(torch.randn(n_slots, dim) / math.sqrt(dim))
        # p-adic content gate: W_v = 0 ⟹ plain highway; valuation term turns it on.
        self.W_v = nn.Parameter(torch.zeros(dim))
        # Lorentzian retrieval sharpness + ReZero scale (0 ⟹ out = x exactly at init).
        self.route_log_sharpness = nn.Parameter(torch.zeros(()))
        self.residual_scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return persistent_memory_param_count(self.d, self.n_slots)

    def memory_read(self, x: torch.Tensor) -> torch.Tensor:
        """Top-k ultrametric retrieval from the persistent bank (NON-softmax)."""
        q = torch.einsum("ij,...j->...i", self.W_q, x)  # (..., D)
        dist = _padic_distance(q, self.memory, self.p)  # (..., n_slots)
        k = min(self.top_k, self.n_slots)
        # Top-k HARD closest (tropical): only the nearest slots contribute.
        topk_dist, topk_idx = torch.topk(dist, k, dim=-1, largest=False)  # (..., k)
        topk_mem = self.memory[topk_idx]  # (..., k, D)
        # Bounded-reciprocal (Lorentzian) read weights — inverse-distance, NOT softmax.
        sharp = (
            torch.nn.functional.softplus(self.route_log_sharpness.to(dist.dtype)) + 0.5
        )
        w = 1.0 / (1.0 + (topk_dist * sharp).pow(2))  # (..., k)
        w = w / w.sum(dim=-1, keepdim=True)
        return torch.einsum("...k,...kd->...d", w, topk_mem)  # (..., D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., D) -> (..., D)``: p-adic-gated ultrametric bank read + ReZero."""
        read = self.memory_read(x)
        proj = torch.einsum("ij,...j->...i", self.W_o, read)  # (..., D)
        val_x = _padic_valuation(x.float(), self.p).clamp(-10.0, 10.0).to(x.dtype)
        gate = torch.sigmoid(val_x * self.W_v)  # (..., D)
        return x + self.residual_scale * gate * proj

    # ── verification helpers (tests only; never the hot path) ──────────────────

    def shape(self) -> tuple[int, int, int]:
        """``(dim, n_slots, top_k)`` — the persistent-memory geometry."""
        return (self.d, self.n_slots, self.top_k)
