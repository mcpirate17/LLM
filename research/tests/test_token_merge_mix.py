# pyright: reportPrivateImportUsage=false
"""NM-C12 sheaf-consistency token-merge mixer — behavioural pins.

What must hold:

- Identity at init (ReZero) and shape preservation.
- CAUSALITY: output[p] has zero gradient to x[q > p] even through the STE
  score path — the content-aware fold decision for token p uses x_{p+1}, so
  the exclusive read (nearest completed cluster STRICTLY before p) is the
  structural fix for the 2026-05-23 restore-hold leak class
  (test_adjacent_token_merge_causality.py documents the incident).
- Content-awareness — the structural difference vs the content-blind stride
  baseline ``_op_adjacent_token_merge``: agreeing duplicates fold, disagreeing
  tokens survive BIT-EXACT in the compressed stream, and the compression ratio
  responds to input content.
- Gates: ``merge_collapse_loss`` is exactly 1 at full (or spurious ρ≡0)
  agreement; ``merge_budget_loss`` carries gradient to the threshold τ.
- NM-11: cross-token BY DESIGN (no pointwise waiver) yet measured not-a-twin.
- NM-10: finite physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.token_merge_mix import (
    TokenMergeMix,
    token_merge_param_count,
)


def _module(dim: int = 32, **kwargs) -> TokenMergeMix:
    torch.manual_seed(0)
    return TokenMergeMix(dim, **kwargs)


def _activated(dim: int = 32, **kwargs) -> TokenMergeMix:
    mix = _module(dim, **kwargs)
    with torch.no_grad():
        mix.scale.fill_(1.0)
    return mix


def test_shape_preserved() -> None:
    mix = _activated()
    x = torch.randn(3, 17, 32)
    assert mix(x).shape == x.shape


def test_identity_at_init() -> None:
    mix = _module()
    x = torch.randn(2, 9, 32)
    torch.testing.assert_close(mix(x), x)


def test_single_token_sequence_is_safe() -> None:
    mix = _activated()
    x = torch.randn(2, 1, 32)
    torch.testing.assert_close(mix(x), x)  # no completed cluster to read


def test_causal_no_future_gradient() -> None:
    """output[p] must not depend on x[q>p] — including through the STE score
    weights and the fold decisions (the leak class this design rules out)."""
    torch.manual_seed(1)
    mix = _activated(16)
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        base = torch.randn(1, 12, 16, generator=gen)
        # Duplicate some adjacent tokens so real merges participate.
        base[:, 3] = base[:, 2]
        base[:, 7] = base[:, 6]
        x = base.clone().requires_grad_(True)
        for p in range(x.shape[1]):
            grad = torch.autograd.grad(
                mix(x)[0, p].sum(), x, retain_graph=False, allow_unused=False
            )[0]
            future = grad[0, p + 1 :]
            if future.numel() == 0:
                continue
            assert future.abs().max().item() == 0.0, (
                f"seed {seed}: output[{p}] leaks gradient from positions > {p}"
            )


def test_duplicate_adjacent_tokens_fold() -> None:
    mix = _activated()
    x = torch.randn(1, 8, 32)
    x[:, 4] = x[:, 3]  # exact agreement ⟹ δ=0 ⟹ w=1 > τ
    structure = mix.merge_structure(x)
    assert bool(structure.fold[0, 3])
    assert not bool(structure.fold[0, 7])  # last position never folds


def test_distinct_tokens_do_not_fold() -> None:
    mix = _activated()
    x = 3.0 * torch.randn(2, 10, 32)  # well-separated ⟹ δ large ⟹ w small
    structure = mix.merge_structure(x)
    assert not structure.fold.any()
    assert mix.compression_ratio(x) == 1.0


def test_compression_responds_to_content() -> None:
    """The stride baseline compresses every input identically; here redundant
    input compresses and distinct input does not — same module, same params."""
    mix = _activated()
    token = torch.randn(1, 1, 32)
    redundant = token.expand(1, 12, 32).contiguous()
    distinct = 3.0 * torch.randn(1, 12, 32)
    assert mix.merge_rate(redundant) > 0.5
    assert mix.merge_rate(distinct) == 0.0


def test_chain_merge_forms_single_cluster_with_mean_value() -> None:
    mix = _activated()
    x = 3.0 * torch.randn(1, 9, 32)
    x[:, 2] = x[:, 1]
    x[:, 3] = x[:, 1]
    x[:, 4] = x[:, 1]  # run of 4 identical tokens: positions 1..4
    structure = mix.merge_structure(x)
    assert structure.fold[0, 1:4].all() and not bool(structure.fold[0, 4])
    assert int(structure.head_of[0, 1]) == 4
    # Pure superposition of 4 equal members: 4·x — amplitude IS cluster mass;
    # a convex mean here would be the softmax simplex signature.
    torch.testing.assert_close(structure.merged[0, 4], 4.0 * x[0, 1])


def test_merged_value_is_cluster_superposition() -> None:
    mix = _activated(8, overlap_dim=4, lorentz_gamma=1e6)  # γ→∞ ⟹ all agree
    x = torch.randn(1, 4, 8)
    structure = mix.merge_structure(x)
    assert structure.fold[0, :3].all()
    torch.testing.assert_close(structure.merged[0, 3], x[0].sum(dim=0))


def test_salient_token_survives_bit_exact() -> None:
    """The lane's information-preservation claim vs the content-blind stride:
    a disagreeing token becomes a singleton cluster and its compressed-stream
    value is EXACTLY its input — never averaged into filler."""
    mix = _activated()
    filler = torch.randn(32)
    x = filler.repeat(1, 11, 1)
    salient = filler + 8.0 * torch.randn(32)
    x[:, 5] = salient
    merged, is_head = mix.compressed_stream(x)
    assert bool(is_head[0, 5])  # disagrees with successor ⟹ head
    torch.testing.assert_close(merged[0, 5], salient)  # singleton ⟹ bit-exact
    assert mix.merge_rate(x) > 0.5  # while the filler still compresses


def test_exclusive_read_uses_strict_past() -> None:
    """Positions read only clusters COMPLETED strictly before them: with two
    disagreeing blocks [a a a | b b b], the a-block (its own cluster still
    open, no completed past) passes through untouched, while the b-block
    actively reads the completed a-cluster."""
    mix = _activated()
    a = torch.randn(32)
    b = a + 8.0 * torch.randn(32)
    x = torch.stack([a, a, a, b, b, b]).unsqueeze(0)
    out = mix(x)
    torch.testing.assert_close(out[:, :3], x[:, :3])  # no completed past
    assert not torch.allclose(out[:, 3:], x[:, 3:])  # reads the a-cluster


def test_cluster_span_is_bounded_and_kills_identity_degeneracy() -> None:
    """The bounded gluing cover: a constant sequence may not chain into one
    forever-open cluster (which would silently reduce the op to the identity —
    the softmax partition-of-unity signature). Heads are forced every
    ``max_cluster`` positions and completed summaries actively perturb the
    output."""
    mix = _activated(max_cluster=4)
    x = torch.randn(1, 1, 32).expand(1, 20, 32).contiguous()
    structure = mix.merge_structure(x)
    heads = structure.is_head[0].nonzero().flatten().tolist()
    assert heads == [3, 7, 11, 15, 19]
    spans = torch.diff(torch.tensor([-1] + heads))
    assert int(spans.max()) <= 4
    out = mix(x)
    torch.testing.assert_close(out[:, :4], x[:, :4])  # before the first head
    assert not torch.allclose(out[:, 4:], x[:, 4:])  # constants NOT preserved


def test_gradients_flow_to_mechanism_parameters() -> None:
    mix = _activated()
    x = torch.randn(2, 12, 32, requires_grad=True)
    x = x.clone()
    x.retain_grad()
    with torch.no_grad():
        x[:, 4] = x[:, 3]  # ensure at least one fold so STE weights engage
    (mix(x).square().mean() + mix.merge_budget_loss(x)).backward()
    for name in ("rho_left", "rho_right", "out_lift", "scale", "threshold_logit"):
        grad = dict(mix.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} received no gradient"


def test_merge_collapse_loss_bounds() -> None:
    mix = _activated()
    x = 3.0 * torch.randn(2, 10, 32)
    assert float(mix.merge_collapse_loss(x).detach()) < 0.3  # disagreement ⟹ low
    with torch.no_grad():
        mix.rho_left.zero_()
        mix.rho_right.zero_()  # spurious-agreement degeneracy: δ ≡ 0 ⟹ w ≡ 1
    torch.testing.assert_close(
        mix.merge_collapse_loss(x), torch.ones(()), atol=1e-6, rtol=0.0
    )


def test_merge_budget_loss_penalizes_both_extremes() -> None:
    mix = _activated()
    no_merge = 3.0 * torch.randn(1, 10, 32)
    all_merge = torch.randn(1, 1, 32).expand(1, 10, 32).contiguous()
    assert float(mix.merge_budget_loss(no_merge, target_rate=0.5).detach()) > 0.05
    assert float(mix.merge_budget_loss(all_merge, target_rate=0.5).detach()) > 0.05
    with pytest.raises(ValueError):
        mix.merge_budget_loss(no_merge, target_rate=1.5)


def test_num_parameters_exact() -> None:
    dim, k = 32, 8
    mix = _module(dim, overlap_dim=k)
    counted = sum(p.numel() for p in mix.parameters())
    assert counted == token_merge_param_count(dim, k) == 2 * k * dim + dim * dim + 2


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        TokenMergeMix(0)
    with pytest.raises(ValueError):
        TokenMergeMix(8, overlap_dim=8)  # overlap must be a PROPER restriction
    with pytest.raises(ValueError):
        TokenMergeMix(8, overlap_dim=4, lorentz_gamma=0.0)
    with pytest.raises(ValueError):
        TokenMergeMix(8, overlap_dim=4, gate_temp=0.0)
    with pytest.raises(ValueError):
        TokenMergeMix(8, overlap_dim=4, max_cluster=1)
    mix = _module()
    with pytest.raises(ValueError):
        mix(torch.randn(4, 32))
    with pytest.raises(ValueError):
        mix(torch.randn(1, 4, 16))


def test_not_a_softmax_attention_twin() -> None:
    """NM-11 measured detector — WITHOUT the pointwise waiver: this mixer is
    cross-token by design (sequence compression), so the not-a-twin verdict
    must come from the mechanism itself. The detector DROVE the design: a
    convex cluster mean measured 0.898 (twin — the simplex signature), a
    √n-normalized superposition rode the 0.7 boundary (0.68–0.72 across
    seeds), and only the shipped form — unnormalized superposition glue +
    bilinear (1+x) read coupling + bounded cover — measures clear (0.647 max
    over 5 init seeds)."""
    mix = _activated()
    probe = AlgebraicPropertyProbe(batch=4, seq_len=16, dim=32, n_seeds=3)
    props = probe.measure(mix)
    assert not props.is_softmax_twin(), (
        f"softmax_twin_score={props.softmax_twin_score:.3f} "
        f"(xmix={props.cross_token_mixing:.3f}, "
        f"const={props.constant_token_preservation:.3f}, "
        f"convex={props.convex_range_fraction:.3f})"
    )


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint ⟹ scoreable on the geometric-novelty
    axis alongside the other synthesis mixers."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = _activated(16)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
