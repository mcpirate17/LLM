"""Tests for NM-C2 compositional feature-hash (Bloom) token embedding.

Pins the spec:
- ZERO V×d table; trainable params = ``K·h·D`` (+ ``n_anchors·D``), INDEPENDENT
  of vocab size ``V`` (the compaction claim — at cl100k this is the ~75% param
  block [[feedback_active_params_mean_non_embedding]]).
- Compositionality: ``emb(t) = ⊕_k W[k, f_k(t)]`` exactly (sum of K code vectors).
- The hash is a bit-mixing finalizer ``mix(t ⊕ seed_k) mod h`` that is APERIODIC
  and ~uniform over CONTIGUOUS ids — pins the design fix (a naive
  ``(a·t+b) mod P mod h`` is h-periodic and collapses contiguous low ids to one
  bucket; ``t`` and ``t+h`` would collide in every hash and defeat Bloom).
- Collision resistance scales EXPONENTIALLY in K: a single hash collapses ``V>h``
  tokens to ≤ ``h`` distinct addresses (pigeonhole — the
  ``ModuloHashEmbedding`` control regime), while ``K=8`` independent hashes make
  the K-tuple address space ``h^K ≫ V`` ⟹ near-unique.
- Hybrid anchor fallback for collision undercoverage (default off = table-free).
- Gradient reaches the codebook (and anchors when present).

NOTE: this is a token EMBEDDING (long ids → D), not a ``[B,L,D]→[B,L,D]`` mixer,
so the PhysicsDescriptorProbe (NM-10) / softmax-twin (NM-11) feature-mixer probes
do not apply — it is measured by param-independence-from-V + distinctness +
collision-resistance instead.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.feature_hash_embedding import (
    FeatureHashEmbedding,
    feature_hash_param_count,
)


def test_forward_preserves_shape_and_is_finite() -> None:
    emb = FeatureHashEmbedding(dim=32, vocab_size=1000, n_hashes=8, n_buckets=64)
    ids = torch.randint(0, 1000, (4, 10))
    out = emb(ids)
    assert out.shape == (4, 10, 32)
    assert torch.isfinite(out).all()


def test_accepts_arbitrary_id_shape() -> None:
    """``forward`` maps an id tensor of shape ``(...)`` to embeddings ``(..., dim)``."""
    emb = FeatureHashEmbedding(dim=16, vocab_size=512, n_hashes=4, n_buckets=32)
    assert emb(torch.tensor(7)).shape == (16,)
    assert emb(torch.tensor([1, 2, 3])).shape == (3, 16)
    assert emb(torch.randint(0, 512, (2, 3, 5))).shape == (2, 3, 5, 16)


@pytest.mark.parametrize("V", [256, 10_000, 100_277])
def test_param_count_independent_of_vocab_size(V: int) -> None:
    """Core compaction claim: params do NOT scale with vocab size V."""
    emb = FeatureHashEmbedding(dim=64, vocab_size=V, n_hashes=8, n_buckets=128)
    assert emb.num_parameters == feature_hash_param_count(64, 8, 128)
    assert sum(p.numel() for p in emb.parameters()) == emb.num_parameters
    # identical params across wildly different vocab sizes
    e_small = FeatureHashEmbedding(dim=64, vocab_size=256, n_hashes=8, n_buckets=128)
    e_big = FeatureHashEmbedding(dim=64, vocab_size=100_277, n_hashes=8, n_buckets=128)
    n_small = sum(p.numel() for p in e_small.parameters())
    n_big = sum(p.numel() for p in e_big.parameters())
    assert n_small == n_big == 8 * 128 * 64


def test_no_full_vocab_table_allocated() -> None:
    """No parameter has a vocab-sized dimension; the codebook is (K, h, D)."""
    V = 100_277
    emb = FeatureHashEmbedding(dim=64, vocab_size=V, n_hashes=8, n_buckets=128)
    assert tuple(emb.codebook.shape) == (8, 128, 64)
    for p in emb.parameters():
        assert V not in p.shape, f"vocab-sized parameter {tuple(p.shape)} allocated"
    # seeds are a fixed STRUCTURE buffer, not a learned parameter
    assert tuple(emb.hash_seeds.shape) == (8,)
    assert "hash_seeds" not in dict(emb.named_parameters())


def test_compositionality_is_sum_of_k_code_vectors() -> None:
    """``emb(t) == Σ_k codebook[k, bucket_k(t)]`` exactly (verifies the ⊕_k gather)."""
    torch.manual_seed(1)
    emb = FeatureHashEmbedding(dim=24, vocab_size=2000, n_hashes=6, n_buckets=48)
    ids = torch.randint(0, 2000, (3, 7))
    out = emb(ids)
    buckets = emb.bucket_ids(ids)  # (3, 7, 6)
    contribs = torch.stack(
        [emb.codebook[k, buckets[..., k]] for k in range(emb.n_hashes)], dim=-2
    )  # (3, 7, 6, 24)
    ref = contribs.sum(dim=-2)  # (3, 7, 24)
    assert torch.allclose(out, ref, atol=1e-6)


def test_deterministic_same_id_same_embedding() -> None:
    emb = FeatureHashEmbedding(dim=16, vocab_size=512, n_hashes=4, n_buckets=32)
    ids = torch.randint(0, 512, (3, 6))
    with torch.no_grad():
        assert torch.equal(emb(ids), emb(ids))  # repeated forward is identical
    # a repeated token id yields identical embedding rows
    rep = torch.tensor([[5, 5, 5], [9, 9, 9]])
    with torch.no_grad():
        out = emb(rep)
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 1], out[:, 2])


def test_distinct_tokens_get_distinct_embeddings() -> None:
    torch.manual_seed(0)
    emb = FeatureHashEmbedding(dim=32, vocab_size=100_000, n_hashes=8, n_buckets=64)
    ids = torch.randint(0, 100_000, (64,))
    with torch.no_grad():
        embs = emb(ids)
    dist = torch.cdist(embs, embs)
    dist.fill_diagonal_(float("inf"))
    assert float(dist.min()) > 0.0  # no two distinct tokens collapsed


def test_single_hash_is_aperiodic_and_uniform_over_contiguous_ids() -> None:
    """Pins the design fix. A naive ``(a·t+b) mod P mod h`` is h-periodic:
    ``bucket(t) == bucket(t+h)`` for EVERY t (collision fraction 1.0), so ``t``
    and ``t+h`` collide in every hash and Bloom distinctness is destroyed. The
    mixing finalizer must instead (1) fill ALL ``h`` buckets over contiguous ids
    (no low-id collapse) and (2) collide at only the random ~1/h rate, NOT 1.0."""
    emb = FeatureHashEmbedding(dim=8, vocab_size=2048, n_hashes=1, n_buckets=8)
    buckets = emb.bucket_ids(torch.arange(2048)).squeeze(-1)  # (2048,)
    # (1) uniform fill: all buckets used over contiguous ids
    assert int(torch.unique(buckets).numel()) == 8
    # (2) NO systematic m·h periodicity: bucket(t) vs bucket(t + m·h) collide only
    #     at the random 1/h rate (~0.125 for h=8), never ~1.0 (the naive regime).
    for stride in (1, 2, 4):
        m = stride * emb.n_buckets
        frac = (buckets[:-m] == buckets[m:]).float().mean().item()
        assert frac < 0.35, (
            f"systematic {m}-periodicity: {frac:.2f} of tokens collide with their "
            f"+{m} neighbour (random≈{1 / emb.n_buckets:.3f}, systematic=1.0)"
        )


def test_collision_resistance_grows_with_n_hashes() -> None:
    """The Bloom payoff vs the single-hash ``ModuloHashEmbedding`` control: a
    single hash collapses ``V > h`` tokens to ≤ ``h`` distinct addresses
    (pigeonhole); K=8 independent hashes make the K-tuple address space
    ``h^K ≫ V`` ⟹ near-unique — collision resistance scales exponentially in K."""
    ids = torch.arange(64)
    e1 = FeatureHashEmbedding(dim=16, vocab_size=512, n_hashes=1, n_buckets=8)
    e8 = FeatureHashEmbedding(dim=16, vocab_size=512, n_hashes=8, n_buckets=8)
    nd1 = int(torch.unique(e1.bucket_ids(ids), dim=0).shape[0])
    nd8 = int(torch.unique(e8.bucket_ids(ids), dim=0).shape[0])
    assert nd1 <= 8  # single hash: ≤ h distinct addresses
    assert nd8 >= 50  # multi-hash: near-unique over 64 tokens
    assert nd8 > 5 * nd1  # ≥5× the distinctness of a single hash
    # at the EMBEDDING level too, K=1 collapses to ≤ h distinct vectors
    with torch.no_grad():
        embs1 = e1(ids)
    assert int(torch.unique(embs1.round(decimals=3), dim=0).shape[0]) <= 8


def test_hashes_independent_across_k() -> None:
    """The K hash columns are not identical — distinct seeds ⟹ distinct, hence
    independent, bucket assignments (the independence Bloom distinctness needs)."""
    emb = FeatureHashEmbedding(dim=8, vocab_size=512, n_hashes=8, n_buckets=16)
    buckets = emb.bucket_ids(torch.arange(100))  # (100, 8)
    for i in range(emb.n_hashes):
        for j in range(i + 1, emb.n_hashes):
            assert not torch.equal(buckets[..., i], buckets[..., j]), (
                f"hashes {i} and {j} are identical (seeds not independent)"
            )


def test_default_is_fully_table_free() -> None:
    """Default ``n_anchors=0`` ⟹ no dense table at all, params exactly ``K·h·D``."""
    emb = FeatureHashEmbedding(dim=32, vocab_size=100_277)  # defaults K=8, h=256
    assert emb.anchors is None
    assert emb.num_parameters == 8 * 256 * 32
    assert sum(p.numel() for p in emb.parameters()) == emb.num_parameters


def test_hybrid_anchor_fallback() -> None:
    """``n_anchors > 0`` adds a dense anchor vector to special tokens (ids <
    n_anchors) only; at init anchors==0 (inert), and non-anchor tokens are
    untouched — the collision-undercoverage insurance the spec demands."""
    emb = FeatureHashEmbedding(
        dim=16, vocab_size=512, n_hashes=4, n_buckets=32, n_anchors=4
    )
    assert emb.anchors is not None and tuple(emb.anchors.shape) == (4, 16)
    probe = torch.tensor([0, 1, 2, 3, 10])  # 4 anchors + 1 non-anchor

    # at init anchors==0 ⟹ identical to a no-anchor module with the same codebook
    no_anchor = FeatureHashEmbedding(
        dim=16, vocab_size=512, n_hashes=4, n_buckets=32, n_anchors=0
    )
    with torch.no_grad():
        no_anchor.codebook.copy_(emb.codebook)
        base = emb(probe)
    assert torch.allclose(base, no_anchor(probe), atol=1e-6)

    # perturbing anchors shifts ONLY the anchor tokens
    with torch.no_grad():
        emb.anchors.add_(1.0)
        shifted = emb(probe)
    assert torch.allclose(
        shifted[:4] - base[:4], torch.full((4, 16), 1.0), atol=1e-6
    ), "anchor tokens must shift by the perturbation"
    assert torch.allclose(shifted[4], base[4], atol=1e-6), (
        "non-anchor token must be unaffected by the anchor table"
    )


def test_backward_flows_to_codebook_and_anchors() -> None:
    emb = FeatureHashEmbedding(
        dim=32, vocab_size=10_000, n_hashes=8, n_buckets=64, n_anchors=4
    )
    ids = torch.randint(0, 10_000, (2, 8))
    ids[0, 0] = 0  # force an anchor token so the anchor table is in the path
    ids[1, 3] = 3
    emb(ids).square().mean().backward()
    assert emb.codebook.grad is not None and torch.isfinite(emb.codebook.grad).all()
    assert float(emb.codebook.grad.abs().sum()) > 0
    assert emb.anchors.grad is not None, "anchor table must receive gradient"
    assert torch.isfinite(emb.anchors.grad).all()
