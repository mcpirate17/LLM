"""P0 tests for the parametric op-synthesis substrate.

The load-bearing guarantee: EVERY StageSpec is plain softmax attention at init
(so any sampled mechanism is stable, finite, and gradient-carrying before
training). Plus fwd/bwd finiteness, finite knob grads, and a 2-block stack.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from research.synthesis.parametric_ops import (
    AGGREGATE_FAMILIES,
    ADDRESS_FAMILIES,
    SCORE_NORM_FAMILIES,
    ParametricMix,
    StageSpec,
    all_stage_specs,
)

pytestmark = pytest.mark.unit

_DIM = 16
_B, _S = 2, 12


def _x(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(_B, _S, _DIM, generator=g)


def _share_projections(src: ParametricMix, dst: ParametricMix) -> None:
    """Copy q/k/v/o weights so two specs differ ONLY in their stage math."""
    for name in ("q", "k", "v", "o"):
        getattr(dst, name).load_state_dict(getattr(src, name).state_dict())


def test_enumerate_specs_distinct_and_complete() -> None:
    specs = all_stage_specs()
    assert len(specs) == len(ADDRESS_FAMILIES) * len(SCORE_NORM_FAMILIES) * len(
        AGGREGATE_FAMILIES
    )
    assert len({s.key for s in specs}) == len(specs)


def test_every_spec_is_softmax_attention_at_init() -> None:
    """The identity-at-init contract: any StageSpec == default (dot/softmax/mean)."""
    torch.manual_seed(0)
    ref = ParametricMix(_DIM, StageSpec())  # plain softmax attention
    x = _x()
    ref_out = ref(x)
    for spec in all_stage_specs():
        mix = ParametricMix(_DIM, spec)
        _share_projections(ref, mix)  # isolate the stage math from random init
        out = mix(x)
        assert torch.isfinite(out).all(), f"{spec.key} produced non-finite output"
        assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-4), (
            f"{spec.key} is not softmax-attention at init "
            f"(max abs diff {(out - ref_out).abs().max().item():.2e})"
        )


def test_invalid_family_rejected() -> None:
    with pytest.raises(ValueError):
        StageSpec(address="quantum")
    with pytest.raises(ValueError):
        StageSpec(score_norm="nope")
    with pytest.raises(ValueError):
        StageSpec(aggregate="bogus")


def test_forward_backward_finite_and_knob_grads() -> None:
    """fwd+bwd finite for every spec; the stage's active knobs get finite grads."""
    active = {
        "reciprocal": "reciprocal_logit_scale",
        "cosine": "cosine_gate",
        "sharpen": "log_tau",
        "semiring": "semiring_beta",
    }
    for spec in all_stage_specs():
        mix = ParametricMix(_DIM, spec)
        mix.zero_grad()
        out = mix(_x(1))
        assert torch.isfinite(out).all()
        out.pow(2).mean().backward()
        # Projections always train.
        for name in ("q", "k", "v", "o"):
            g = getattr(mix, name).weight.grad
            assert g is not None and torch.isfinite(g).all(), f"{spec.key}:{name}"
        # The spec's active alternative knobs must receive a finite gradient.
        for fam in (spec.address, spec.score_norm, spec.aggregate):
            knob = active.get(fam)
            if knob is None:
                continue
            g = getattr(mix, knob).grad
            assert g is not None and torch.isfinite(g).all(), f"{spec.key}:{knob}"


def test_knobs_move_the_mechanism_off_softmax() -> None:
    """Sanity: away from init, an alternative knob actually changes the output
    (so the families are real, not inert)."""
    torch.manual_seed(0)
    ref = ParametricMix(_DIM, StageSpec(aggregate="semiring"))
    x = _x(2)
    base = ref(x)
    with torch.no_grad():
        ref.semiring_beta.fill_(4.0)  # slide toward max-pool
    moved = ref(x)
    assert not torch.allclose(base, moved, atol=1e-4)


class _Stack(nn.Module):
    """Minimal 2-block residual stack to confirm specs compose in a network."""

    def __init__(self, dim: int, spec: StageSpec, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_blocks))
        self.mixers = nn.ModuleList(ParametricMix(dim, spec) for _ in range(n_blocks))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for norm, mix in zip(self.norms, self.mixers):
            h = h + mix(norm(h))
        return self.head(h)


def test_two_block_stack_trains_a_step() -> None:
    spec = StageSpec(address="reciprocal", score_norm="sharpen", aggregate="semiring")
    stack = _Stack(_DIM, spec)
    opt = torch.optim.Adam(stack.parameters(), lr=1e-3)
    x, y = _x(3), _x(4)
    loss0 = None
    for _ in range(3):
        opt.zero_grad()
        loss = (stack(x) - y).pow(2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        loss0 = loss0 if loss0 is not None else float(loss.detach())
    assert loss0 is not None
