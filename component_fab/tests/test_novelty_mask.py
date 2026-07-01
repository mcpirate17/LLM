"""Tests for the fingerprint-guided lottery-ticket mask (M2, W5 + W1).

The decisive checks: the density is exact (a knob), the straight-through
estimator actually learns which channels to keep, and — the novel part —
certification accepts a mask only when the masked lane stays out of the softmax
basin (a softmax lane fails, a genuinely-novel lane passes).
"""

from __future__ import annotations

import pytest
import torch

from component_fab.proposer.novelty_mask import (
    NOVELTY_MASK_TWIN_THRESHOLD,
    NoveltyConstrainedMask,
    NoveltyMaskedLane,
)


def _softmax_mixer(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    gen = torch.Generator().manual_seed(1)
    wq = torch.randn(d, d, generator=gen) * d**-0.5
    wk = torch.randn(d, d, generator=gen) * d**-0.5
    return torch.softmax((x @ wq) @ (x @ wk).transpose(1, 2) / d**0.5, dim=-1) @ x


def test_density_is_exact_and_forward_is_binary() -> None:
    mask = NoveltyConstrainedMask(64, target_density=0.3)
    assert mask.density() == pytest.approx(round(0.3 * 64) / 64)
    hard = mask.hard_mask().detach()
    assert set(torch.unique(hard).tolist()) <= {0.0, 1.0}
    assert int((hard > 0.5).sum()) == mask.k


def test_mask_param_cost_is_dim_logits() -> None:
    mask = NoveltyConstrainedMask(48, target_density=0.5)
    n_params = sum(p.numel() for p in mask.parameters())
    assert n_params == 48  # D importance logits, nothing else


def test_straight_through_gradient_flows() -> None:
    mask = NoveltyConstrainedMask(32, target_density=0.5)
    x = torch.randn(2, 8, 32, requires_grad=True)
    mask(x).sum().backward()
    assert mask.logits.grad is not None
    assert torch.isfinite(mask.logits.grad).all()
    assert torch.isfinite(x.grad).all()


def test_ste_learns_which_channels_to_keep() -> None:
    torch.manual_seed(0)
    dim, k = 32, 8
    informative = list(range(0, dim, dim // k))
    keep = torch.zeros(dim)
    keep[informative] = 1.0
    mask = NoveltyConstrainedMask(dim, target_density=k / dim)
    opt = torch.optim.Adam(mask.parameters(), lr=0.05)
    for _ in range(400):
        x = torch.randn(64, 4, dim)
        opt.zero_grad()
        loss = ((mask(x) * keep - x * keep) ** 2).mean()
        loss.backward()
        opt.step()
    final = mask._hard_topk() > 0.5
    assert sum(int(final[c]) for c in informative) == k  # recovers all informative


def test_certification_rejects_softmax_lane() -> None:
    mask = NoveltyConstrainedMask(48, target_density=0.5)
    cert = mask.certify(_softmax_mixer, dim=48)
    assert cert.softmax_twin_score >= NOVELTY_MASK_TWIN_THRESHOLD
    assert not cert.certified


def test_certification_accepts_novel_lane() -> None:
    def novel_gate(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    mask = NoveltyConstrainedMask(48, target_density=0.5)
    cert = mask.certify(novel_gate, dim=48)
    assert cert.softmax_twin_score < NOVELTY_MASK_TWIN_THRESHOLD
    assert cert.certified
    assert cert.density == pytest.approx(0.5)


def test_masked_lane_prunes_output_channels() -> None:
    torch.manual_seed(0)
    base = torch.nn.Linear(24, 24)
    lane = NoveltyMaskedLane(base, 24, target_density=0.25)
    x = torch.randn(2, 6, 24)
    y = lane(x)
    assert y.shape == x.shape
    # Exactly the pruned channels are zero across all positions.
    zero_channels = (y.detach().abs().sum(dim=(0, 1)) == 0).sum().item()
    assert zero_channels == 24 - lane.mask.k
    assert lane.density() == pytest.approx(round(0.25 * 24) / 24)


def test_rejects_wrong_dim() -> None:
    mask = NoveltyConstrainedMask(16, target_density=0.5)
    with pytest.raises(ValueError):
        mask(torch.randn(2, 4, 8))


def test_rejects_bad_density() -> None:
    with pytest.raises(ValueError):
        NoveltyConstrainedMask(16, target_density=0.0)
    with pytest.raises(ValueError):
        NoveltyConstrainedMask(16, target_density=1.5)
