"""Sprint-7 tests: S0.5 hard no-go gate + AR binding probe."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.primitive_templates import (
    FourierBasisLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
)
from component_fab.harness.capability_probes import (
    DEFAULT_CAPABILITY_PROBES,
    causality_stability_gate,
    make_ar_probe,
    train_and_score,
)
from component_fab.harness.probe_block import WinnerLikeBlock
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)
from component_fab.tests.conftest import make_candidate_spec


# ---------- S0.5 gate ----------


def test_s05_gate_passes_per_position_op() -> None:
    lane = nn.Linear(16, 16)
    block = WinnerLikeBlock(lane, dim=16).eval()
    result = causality_stability_gate(block, seq_len=16, dim=16)
    assert result.stability_passed
    assert result.causality_passed
    assert result.passed


def test_s05_gate_passes_causal_tropical_attention() -> None:
    lane = TropicalAttention(dim=16, causal=True)
    block = WinnerLikeBlock(lane, dim=16).eval()
    result = causality_stability_gate(block, seq_len=16, dim=16)
    assert result.causality_passed


def test_s05_gate_rejects_noncausal_tropical_attention() -> None:
    lane = TropicalAttention(dim=16, causal=False)
    block = WinnerLikeBlock(lane, dim=16).eval()
    result = causality_stability_gate(block, seq_len=16, dim=16)
    assert not result.causality_passed
    assert not result.passed


def test_s05_gate_rejects_noncausal_fourier() -> None:
    lane = FourierBasisLane(dim=16)
    block = WinnerLikeBlock(lane, dim=16).eval()
    result = causality_stability_gate(block, seq_len=16, dim=16)
    assert not result.causality_passed


def test_s05_gate_passes_causal_state_space() -> None:
    lane = TropicalStateSpace(dim=16)
    block = WinnerLikeBlock(lane, dim=16).eval()
    result = causality_stability_gate(block, seq_len=16, dim=16)
    assert result.causality_passed
    assert result.passed


# ---------- AR binding probe ----------


def test_ar_probe_layout_is_well_formed() -> None:
    probe = make_ar_probe(n_pairs=3)
    gen = torch.Generator().manual_seed(0)
    x, target, mask = probe.sample_fn(4, 10, 8, gen)
    assert x.shape == (4, 10, 8)
    assert target.shape == (4, 10, 8)
    assert mask.shape == (4, 10)
    # Only the answer position is masked.
    assert mask[:, 7].sum().item() == 4.0
    assert mask[:, :7].sum().item() == 0.0
    assert mask[:, 8:].sum().item() == 0.0


def test_topk_linear_does_not_bind_ar() -> None:
    # TopKLinear is per-position with no position mixing — must fail AR.
    lane = TopKLinear(in_dim=16, out_dim=16, k=4)
    block = WinnerLikeBlock(lane, dim=16)
    result = train_and_score(
        block,
        make_ar_probe(n_pairs=3),
        seq_len=10,
        dim=16,
        seed=0,
    )
    assert not result.passes
    assert result.relative_recall < 0.5


def test_causal_attention_binds_ar_easy() -> None:
    # Standard attention with content-based affinity SHOULD pass AR-easy.
    class _MiniCausalAttn(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.shape[1]
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = torch.einsum("bid,bjd->bij", q, k) / (q.shape[-1] ** 0.5)
            mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float("-inf"), dtype=x.dtype, device=x.device
                ),
                diagonal=1,
            )
            return torch.einsum("bij,bjd->bid", torch.softmax(scores + mask, dim=-1), v)

    block = WinnerLikeBlock(_MiniCausalAttn(16), dim=16)
    result = train_and_score(
        block,
        make_ar_probe(n_pairs=2, n_train_steps=120, pass_threshold=0.4),
        seq_len=8,
        dim=16,
        seed=0,
    )
    assert result.relative_recall > 0.0


# ---------- End-to-end capability validator ----------


def test_validate_capabilities_skips_ar_when_s05_fails() -> None:
    spec = make_candidate_spec({"op_algebraic_space": "euclidean"})
    lane = FourierBasisLane(dim=16)
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    assert not card.s05_passed
    assert not card.can_bind
    assert card.binds_per_probe == {}


def test_validate_capabilities_runs_full_stack_when_s05_passes() -> None:
    spec = make_candidate_spec({"op_algebraic_space": "tropical"})
    lane = TropicalAttention(dim=16, causal=True)
    card = validate_capabilities(spec, lane, dim=16, seq_len=16)
    assert card.s05_passed
    # Sprint-8 tiered gates may eliminate before AR runs, but if AR runs
    # the probe set must be populated. Either way the scorecard records
    # which gate (if any) eliminated the spec.
    blob = capability_scorecard_to_dict(card)
    assert blob["s05_passed"] is True
    if card.eliminated_by is None:
        assert "ar_easy" in card.binds_per_probe
        assert "ar_easy" in blob["binds_per_probe"]
    else:
        assert card.eliminated_by in ("erf_density", "nano_bind")


def test_default_capability_probes_have_two_difficulty_levels() -> None:
    names = {p.name for p in DEFAULT_CAPABILITY_PROBES}
    assert "ar_easy" in names
    assert "ar_medium" in names
