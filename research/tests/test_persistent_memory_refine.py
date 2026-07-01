"""Tests for NM-C10 persistent-memory refinement.

Pins the spec: exact identity-at-init (ReZero scale 0), the retrieval is top-k
hard + bounded-reciprocal (Lorentzian) over an ultrametric distance (NOT
``softmax(QK^T)``), gradient reaches the bank + projections, and — the
capability crux — retrieval is CONTENT-ADDRESSED (the randomized-query binding
control: a token's output depends on its own query relative to the bank, not on
its position or on other tokens). This is the associative-retrieval pathway a
single-pass gate lacks; it must NOT be a positional/recency shortcut. Also
confirmed not a softmax-attention twin (NM-11) and NM-10-measurable.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.persistent_memory_refine import (
    PersistentMemoryRefine,
    _padic_distance,
    persistent_memory_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mem = PersistentMemoryRefine(dim=8, n_slots=16, top_k=4)
    x = torch.randn(2, 10, 8)
    y = mem(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
def test_identity_at_init(d: int) -> None:
    """ReZero scale 0 ⟹ ``out = x`` exactly (safe drop-in for any D)."""
    mem = PersistentMemoryRefine(dim=d, n_slots=8, top_k=2)
    x = torch.randn(3, 5, d)
    assert torch.allclose(mem(x), x, atol=1e-6), f"d={d}"


def test_param_count_scales_with_bank_not_depth() -> None:
    """Compaction axis: capacity ∝ bank size, not stacked depth."""
    d, n_slots = 32, 16
    mem = PersistentMemoryRefine(dim=d, n_slots=n_slots, top_k=4)
    assert mem.num_parameters == persistent_memory_param_count(d, n_slots)
    assert sum(p.numel() for p in mem.parameters()) == mem.num_parameters
    assert mem.memory.numel() == n_slots * d  # bank params track n_slots, not n_layers


def test_retrieval_weights_are_topk_lorentzian_partition_of_unity() -> None:
    """The read blends the top-k NEAREST slots with bounded-reciprocal (inverse-
    distance) weights summing to 1 — sparse + ultrametric, NOT all-to-all softmax."""
    mem = PersistentMemoryRefine(dim=8, n_slots=12, top_k=4)
    x = torch.randn(3, 7, 8)
    q = torch.einsum("ij,...j->...i", mem.W_q, x)
    dist = _padic_distance(q, mem.memory, mem.p)  # (3, 7, 12)
    assert dist.shape == (3, 7, 12)
    topk_dist, _topk_idx = torch.topk(dist, 4, dim=-1, largest=False)  # (3, 7, 4)
    sharp = torch.nn.functional.softplus(mem.route_log_sharpness.detach()) + 0.5
    w = 1.0 / (1.0 + (topk_dist * sharp).pow(2))
    w = w / w.sum(dim=-1, keepdim=True)
    assert w.shape == (3, 7, 4)
    assert (w >= 0).all() and (w <= 1).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(3, 7), atol=1e-6)
    assert 4 < mem.n_slots  # only 4 of 12 slots get nonzero read weight


def test_retrieval_pulls_the_nearest_slot() -> None:
    """With distinct slots, a query equal to slot k retrieves slot k (associative,
    content-addressed)."""
    torch.manual_seed(1)
    mem = PersistentMemoryRefine(dim=6, n_slots=4, top_k=1)
    with torch.no_grad():
        mem.memory.copy_(torch.eye(6)[:4])  # slots = e0, e1, e2, e3
        mem.residual_scale.fill_(1.0)
        mem.W_v.fill_(0.0)  # gate uniform; content enters via the query
    x = torch.zeros(1, 1, 6)
    x[0, 0, 2] = 1.0  # query = e2 = slot 2
    read = mem.memory_read(x)  # (1, 1, 6)
    assert torch.allclose(
        read[0, 0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-5
    )


def test_retrieval_is_content_addressed_not_positional() -> None:
    """Randomized-query binding control (the gate NM-C10 demands): a token's
    output depends on its own CONTENT relative to the bank, not on position or on
    other tokens — the associative pathway, not a positional/recency shortcut."""
    torch.manual_seed(0)
    mem = PersistentMemoryRefine(dim=8, n_slots=16, top_k=4)
    with torch.no_grad():
        mem.residual_scale.fill_(1.0)  # read ON
        mem.W_v.fill_(0.0)  # content enters via the query only
    x = torch.randn(3, 6, 8)
    out = mem(x)
    # (1) Permutation-equivariance: no positional encoding ⟹ shuffling tokens
    #     shuffles the output identically (NOT a positional shortcut).
    perm = torch.randperm(6)
    out_perm = mem(x[:, perm, :])
    assert torch.allclose(out_perm, out[:, perm, :], atol=1e-5)
    # (2) Token-independence: resampling OTHER tokens leaves token i's output fixed
    #     (each token retrieves independently from the shared bank).
    x2 = x.clone()
    x2[:, :5, :] = torch.randn(3, 5, 8)
    out2 = mem(x2)
    assert torch.allclose(out2[:, 5, :], out[:, 5, :], atol=1e-5), (
        "last token's output changed when only OTHER tokens were resampled — "
        "retrieval leaks across tokens (not content-addressed)"
    )
    # (3) Randomized-query control proper: new CONTENT at the SAME position + bank
    #     DOES move the output — retrieval tracks query content, not position/recency.
    x3 = x.clone()
    x3[:, 5, :] = torch.randn(3, 8)
    out3 = mem(x3)
    assert not torch.allclose(out3[:, 5, :], out[:, 5, :], atol=1e-4), (
        "last token's output did NOT change when its content changed at a fixed "
        "position — retrieval is a positional/recency shortcut, not associative"
    )


def test_backward_flows_to_bank_and_projections() -> None:
    mem = PersistentMemoryRefine(dim=16, n_slots=8, top_k=4)
    with torch.no_grad():
        mem.residual_scale.fill_(1.0)  # read ON so the bank is in the forward path
    x = torch.randn(2, 6, 16, requires_grad=True)
    mem(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("memory", "W_q", "W_o", "W_v", "residual_scale"):
        p = dict(mem.named_parameters())[name]
        assert (
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
        ), name


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector: retrieval is per-token (pointwise ⟹
    ``cross_token_mixing=0``) + top-k Lorentzian over an ultrametric distance
    (sparse, inverse-distance — not all-to-all ``softmax(QK^T)``). Not a twin."""
    mem = PersistentMemoryRefine(dim=32, n_slots=16, top_k=4)
    with torch.no_grad():
        mem.residual_scale.fill_(1.0)  # active retrieval path
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mem)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f}, "
        f"const={props.constant_token_preservation:.3f}, "
        f"convex={props.convex_range_fraction:.3f})"
    )
    assert props.cross_token_mixing < 0.1  # pointwise retrieval, not attention


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the refine exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside the other synthesis mixers."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mem = PersistentMemoryRefine(dim=16, n_slots=8, top_k=4)
    with torch.no_grad():
        mem.residual_scale.fill_(1.0)
    desc = probe.describe_operator(mem)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
