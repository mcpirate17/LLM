"""NM-C2 — compositional feature-hash (Bloom) token embedding.

Replaces the V×d lookup table with a vocab-size-INDEPENDENT compositional
multi-hash representation:

    emb(t) = ⊕_{k=1..K} W[k, f_k(t)]   (+ optional dense anchor fallback)

The K complementary hash functions ``f_k(t) = mix(t ⊕ seed_k) mod h`` are FIXED
(registered buffers, never learned — the hash IS the compositional structure),
where ``mix`` is a two-round xorshift-multiply finalizer that destroys ALL
token-id structure (linearity AND periodicity) so the bucket assignment is
~uniform for ANY token id — including small contiguous ids. This matters: a
naive ``(a·t+b) mod P`` universal hash is only uniform over ``t ∈ [0, P)`` and is
``h``-periodic under ``mod h``, so contiguous low ids collapse to one bucket and
tokens ``t``, ``t+h`` collide in every hash (defeating Bloom distinctness). The
finalizer removes both failure modes: ``bucket(t) ≠ bucket(t+h)`` for all ``t``.

The SHARED learned codebook ``W ∈ R^{K×h×D}`` holds the only learned parameters.
A token's embedding is the SUM of the K code vectors its K independent hashes
select. Two tokens produce the SAME embedding only when all K hashes land on the
same buckets (probability ≈ 1/h^K for the K independent hashes, vs ≈ 1/h for a
single hash) — this is the compositional multi-hash (Bloom) generalization of the
single-hash ``ModuloHashEmbedding`` control shipped in
``component_fab/generator/ecc_codeword_embedding.py``: collision resistance
scales EXPONENTIALLY in K rather than linearly in h. A tiny learned dense
``anchor`` table (``n_anchors`` special tokens, ids < ``n_anchors``) is the
hybrid fallback for collision undercoverage (default 0 = fully table-free).

Parameters = ``K·h·D`` (+ ``n_anchors·D``) — INDEPENDENT of vocab size ``V``.
At cl100k (V=100277, D=384): ``K=8, h=256`` → 786 432 params vs the 38.5 M
tied V×d table (~49× cut) with essentially perfect distinctness.

NOTE — measurability: this is a TOKEN EMBEDDING (long ids → D vectors), not a
``[B,L,D]→[B,L,D]`` feature mixer, so the ``PhysicsDescriptorProbe`` (NM-10) and
``AlgebraicPropertyProbe`` softmax-twin (NM-11) feature-mixer probes do not
apply. It is measured by param-independence-from-V + distinctness +
collision-resistance (the embedding analog of a distinctive physics fingerprint).

Self-contained (imports only ``torch`` + ``math``); registry wiring DEFERRED
(NM-C3/C5/C7/C10/C15 convention — ship the mechanism, wire once codex's NM-1
lands).
"""

from __future__ import annotations

import math

import torch
from torch import nn

# K distinct odd seeds (< 2^31) for the K independent mixing hashes. Distinct
# seeds ⟹ the K hashes are genuinely independent (the structure that gives
# Bloom-style distinctness). Tabulated (not generated) so construction is
# deterministic across runs — these are STRUCTURE, not learned.
_HASH_SEEDS = (
    0x27D4EB2F,
    0x164F7BB5,
    0x4F6A50E9,
    0x3F1C4D39,
    0x1B873593,
    0x38CC3C1D,
    0x5BD1E9A7,
    0x6D2B36F1,
    0x1F6B4A8D,
    0x3A7C5E91,
    0x52D4B83F,
    0x6E1A09C7,
    0x2C8F73B5,
    0x49E5C6AD,
    0x5A2F381D,
    0x7C3D9E51,
)
# Murmur3-style finalizer multiplier (odd, < 2^31). Kept in positive int64
# throughout (intermediates < 2^31 · multiplier < 2^62), so the int64 multiply
# wraps cleanly modulo 2^64 with no sign/overflow ambiguity.
_MIX_MULT = 73244443
_MASK31 = 0x7FFFFFFF


def _int_mix(x: torch.Tensor) -> torch.Tensor:
    """Two-round xorshift-multiply finalizer — destroys all input structure.

    Input ``x`` is a non-negative int64 tensor (shape arbitrary); the output is a
    pseudo-random int64 tensor of the same shape with values in ``[0, 2^31)``.
    Every step is masked to 31 bits so the multiply stays in positive int64
    (no signed-shift / overflow ambiguity). ``mix(t)`` and ``mix(t+1)`` are
    uncorrelated (avalanche) ⟹ contiguous token ids hash ~uniformly.
    """
    x = (x ^ (x >> 16)) & _MASK31
    x = (x * _MIX_MULT) & _MASK31
    x = (x ^ (x >> 16)) & _MASK31
    x = (x * _MIX_MULT) & _MASK31
    x = (x ^ (x >> 16)) & _MASK31
    return x


def _hash_seeds(n_hashes: int) -> torch.Tensor:
    """The first ``n_hashes`` fixed seeds as an int64 tensor of shape ``(K,)``."""
    if n_hashes < 1:
        raise ValueError(f"n_hashes must be >= 1, got {n_hashes}")
    if n_hashes > len(_HASH_SEEDS):
        raise ValueError(
            f"n_hashes={n_hashes} exceeds the {len(_HASH_SEEDS)} tabulated seeds; "
            f"extend _HASH_SEEDS to support more."
        )
    return torch.tensor(_HASH_SEEDS[:n_hashes], dtype=torch.int64)


def feature_hash_param_count(
    dim: int, n_hashes: int, n_buckets: int, n_anchors: int = 0
) -> int:
    """Exact trainable parameter count (independent of vocab size V)."""
    if dim < 1 or n_hashes < 1 or n_buckets < 1 or n_anchors < 0:
        raise ValueError(
            f"need dim>=1, n_hashes>=1, n_buckets>=1, n_anchors>=0; got "
            f"{dim=}, {n_hashes=}, {n_buckets=}, {n_anchors=}"
        )
    return n_hashes * n_buckets * dim + n_anchors * dim


class FeatureHashEmbedding(nn.Module):
    """Compositional feature-hash (Bloom) token embedding — zero V×d table.

    ``forward(token_ids)`` accepts a long tensor of arbitrary shape ``(...,)`` and
    returns embeddings ``(..., dim)``. Parameter count is independent of
    ``vocab_size`` (that field is informational: it bounds the anchored-token
    range and documents the intended vocabulary; it does not allocate a table).
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        *,
        n_hashes: int = 8,
        n_buckets: int = 256,
        n_anchors: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.n_hashes = int(n_hashes)
        self.n_buckets = int(n_buckets)
        self.n_anchors = int(n_anchors)
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {self.n_buckets}")
        if self.n_anchors < 0 or self.n_anchors > self.vocab_size:
            raise ValueError(
                f"n_anchors must be in [0, vocab_size={self.vocab_size}], got {self.n_anchors}"
            )

        # Fixed hash seeds (STRUCTURE — not learned).
        self.register_buffer("hash_seeds", _hash_seeds(self.n_hashes))

        # Shared learned codebook — the ONLY vocab-independent learned params.
        # Small init so the embedding does not destabilize when swapped in for a
        # pretrained dense table (the embedding analog of identity-at-init).
        scale = 1.0 / math.sqrt(self.dim)
        self.codebook = nn.Parameter(
            torch.randn(self.n_hashes, self.n_buckets, self.dim) * scale
        )

        # Hybrid fallback: tiny dense anchor table for ``n_anchors`` special tokens
        # (ids < n_anchors). Zero-init so it is inert unless trained — collision
        # undercoverage insurance, NOT a hidden full table.
        if self.n_anchors > 0:
            self.anchors = nn.Parameter(torch.zeros(self.n_anchors, self.dim))
        else:
            self.anchors = None

    @property
    def num_parameters(self) -> int:
        return feature_hash_param_count(
            self.dim, self.n_hashes, self.n_buckets, self.n_anchors
        )

    def bucket_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """K independent hash bucket ids for each token: shape ``(..., K)``.

        ``bucket_k(t) = mix(t ⊕ seed_k) mod h``. The finalizer makes the bucket
        assignment ~uniform and aperiodic for every token id, so K independent
        seeds give Bloom-style distinctness (collisions ≈ 1/h^K).
        """
        t = (token_ids.to(torch.int64) & _MASK31).unsqueeze(-1)  # (..., 1)
        mixed = _int_mix(t ^ self.hash_seeds)  # (..., K) in [0, 2^31)
        return mixed % self.n_buckets  # (..., K) in [0, h-1]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        buckets = self.bucket_ids(token_ids)  # (..., K)
        k_idx = torch.arange(self.n_hashes, device=buckets.device)  # (K,)
        # Select the K chosen code vectors (advanced indexing): (..., K, D)
        sel = self.codebook[k_idx, buckets]
        emb = sel.sum(dim=-2)  # ⊕_k  → (..., D)
        if self.anchors is not None:
            mask = token_ids < self.n_anchors  # (...,)
            if bool(mask.any()):
                anchor_lookup = self.anchors[token_ids.clamp_max(self.n_anchors - 1)]
                emb = torch.where(mask.unsqueeze(-1), emb + anchor_lookup, emb)
        return emb
