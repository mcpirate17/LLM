"""Unit tests for in-loop monster-surprisal routing (Workstream D inc. 5).

Duck-typed: a routing block is any module with a settable ``route_prior``. These
tests use minimal fakes so research/ stays free of a component_fab import; the
real ``LossMonsterPairedBlock`` end-to-end lives in component_fab/tests.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from research.synthesis.data_pipeline_grammar import DataRouteSpec
from research.training.surprisal_router import (
    clear_route_prior,
    set_route_prior_from_surprisal,
    surprisal_routed_logits,
    token_surprisal,
)


class _UniformMonster(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, l = ids.shape
        return torch.zeros(b, l, self.vocab)


class _PeakyMonster(nn.Module):
    """Confidently predicts token 0 everywhere (large logit on class 0)."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, l = ids.shape
        logits = torch.zeros(b, l, self.vocab)
        logits[..., 0] = 20.0
        return logits


class _RoutingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route_prior: torch.Tensor | None = None
        self.saw_prior = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.saw_prior = self.route_prior is not None
        return x


class _RoutingCarrier(nn.Module):
    def __init__(self, vocab: int, dim: int, n_blocks: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([_RoutingBlock() for _ in range(n_blocks)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


def test_token_surprisal_uniform_equals_log2_vocab() -> None:
    vocab = 16
    ids = torch.randint(0, vocab, (2, 8))
    sur = token_surprisal(_UniformMonster(vocab), ids, ids)
    assert sur.shape == ids.shape
    assert torch.allclose(sur, torch.full_like(sur, math.log2(vocab)), atol=1e-4)
    # nats variant is ln(vocab)
    nats = token_surprisal(_UniformMonster(vocab), ids, ids, in_bits=False)
    assert torch.allclose(nats, torch.full_like(nats, math.log(vocab)), atol=1e-4)


def test_token_surprisal_low_when_target_predicted() -> None:
    vocab = 16
    ids = torch.randint(0, vocab, (1, 6))
    monster = _PeakyMonster(vocab)
    easy = token_surprisal(monster, ids, torch.zeros_like(ids))  # target 0 = predicted
    hard = token_surprisal(monster, ids, torch.ones_like(ids))  # target 1 = surprising
    assert (easy < hard).all()


def test_token_surprisal_rejects_non_3d_logits() -> None:
    class _BadMonster(nn.Module):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return ids.float()

    with pytest.raises(ValueError, match=r"\[B, L, V\]"):
        token_surprisal(
            _BadMonster(),
            torch.zeros(2, 4, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
        )


def test_set_route_prior_updates_all_blocks() -> None:
    carrier = _RoutingCarrier(16, 8, n_blocks=3)
    surprisal = torch.rand(2, 10)
    spec = DataRouteSpec(route="surprisal_split", carrier_fraction=0.3)
    n = set_route_prior_from_surprisal(carrier, surprisal, spec, strength=4.0)
    assert n == 3
    for block in carrier.blocks:
        assert block.route_prior is not None
        assert block.route_prior.shape == (2, 10, 1)


def test_set_route_prior_requires_surprisal_split() -> None:
    carrier = _RoutingCarrier(16, 8)
    with pytest.raises(ValueError, match="surprisal_split"):
        set_route_prior_from_surprisal(carrier, torch.rand(2, 10), DataRouteSpec())


def test_clear_route_prior_resets() -> None:
    carrier = _RoutingCarrier(16, 8)
    set_route_prior_from_surprisal(
        carrier, torch.rand(2, 10), DataRouteSpec(route="surprisal_split")
    )
    clear_route_prior(carrier)
    assert all(b.route_prior is None for b in carrier.blocks)


def test_routed_logits_sets_then_clears_around_forward() -> None:
    vocab = 16
    carrier = _RoutingCarrier(vocab, 8, n_blocks=2)
    monster = _UniformMonster(vocab)
    x = torch.randint(0, vocab, (2, 12))
    y = torch.randint(0, vocab, (2, 12))
    spec = DataRouteSpec(route="surprisal_split", carrier_fraction=0.3)
    logits = surprisal_routed_logits(carrier, monster, x, y, spec, strength=6.0)
    assert logits.shape == (2, 12, vocab)
    # the prior was live DURING the forward, and cleared AFTER it
    assert all(b.saw_prior for b in carrier.blocks)
    assert all(b.route_prior is None for b in carrier.blocks)


def test_routed_logits_fails_loud_without_consumer() -> None:
    class _PlainCarrier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.emb(ids))

    x = torch.randint(0, 16, (2, 8))
    with pytest.raises(ValueError, match="no carrier block consumes"):
        surprisal_routed_logits(
            _PlainCarrier(),
            _UniformMonster(16),
            x,
            x,
            DataRouteSpec(route="surprisal_split"),
        )
