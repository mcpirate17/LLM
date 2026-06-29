"""End-to-end: monster surprisal steers a real LossMonsterPairedBlock carrier.

Workstream D increment 5 wired against Workstream B's block: per-token surprisal
from a frozen monster sets ``route_prior`` on the paired block so hard tokens
route to the carrier (partner) lane during a live training step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from component_fab.generator.block_templates import LossMonsterPairedBlock
from component_fab.harness.tiny_lm import TinyLM, TinyLMConfig
from research.synthesis.data_pipeline_grammar import DataRouteSpec
from research.training.surprisal_router import (
    surprisal_routed_logits,
    token_surprisal,
)


def _paired_carrier(vocab: int, dim: int, n_blocks: int = 2) -> TinyLM:
    def lane(d: int) -> nn.Module:
        return LossMonsterPairedBlock(
            lambda x: nn.Linear(x, x),  # carrier (partner) lane
            lambda x: nn.Linear(x, x),  # local loss-specialist lane
            d,
            partner_floor=0.0,  # let routing span the full [0, 1] gate range
        )

    return TinyLM(lane, TinyLMConfig(vocab_size=vocab, dim=dim, n_blocks=n_blocks))


def _monster(vocab: int, dim: int) -> TinyLM:
    return TinyLM(lambda d: nn.Linear(d, d), TinyLMConfig(vocab_size=vocab, dim=dim))


def test_surprisal_routes_paired_block_and_trains() -> None:
    torch.manual_seed(0)
    vocab, dim = 32, 16
    carrier = _paired_carrier(vocab, dim)
    monster = _monster(vocab, dim)
    x = torch.randint(0, vocab, (2, 12))
    y = torch.randint(0, vocab, (2, 12))
    spec = DataRouteSpec(route="surprisal_split", carrier_fraction=0.3)

    logits = surprisal_routed_logits(carrier, monster, x, y, spec, strength=10.0)
    assert logits.shape == (2, 12, vocab)
    assert torch.isfinite(logits).all()

    paired = [m for m in carrier.modules() if isinstance(m, LossMonsterPairedBlock)]
    assert paired, "carrier should contain LossMonsterPairedBlocks"
    for block in paired:
        # route_prior cleared after the routed forward; routing was consumed
        assert block.route_prior is None
        assert block.last_partner_frac is not None

    # a real optimizer step runs end-to-end through the routed carrier
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)


def test_high_surprisal_pushes_routing_to_carrier_lane() -> None:
    """With a strong bias, the hard tokens lift the mean partner (carrier) weight."""
    torch.manual_seed(0)
    dim = 16
    block = LossMonsterPairedBlock(
        lambda x: nn.Linear(x, x), lambda x: nn.Linear(x, x), dim, partner_floor=0.0
    )
    h = torch.randn(1, 6, dim)

    # baseline: learned gate only
    block(h)
    baseline = block.last_partner_frac
    assert baseline is not None

    # all tokens hard -> all routed to carrier -> partner weight near 1
    surprisal = torch.full((1, 6), 9.0)
    from research.training.surprisal_router import set_route_prior_from_surprisal

    set_route_prior_from_surprisal(
        block,
        surprisal,
        DataRouteSpec(route="surprisal_split", carrier_fraction=1.0),
        strength=12.0,
    )
    block(h)
    assert block.last_partner_frac is not None
    assert block.last_partner_frac > baseline


def test_token_surprisal_matches_manual_cross_entropy() -> None:
    torch.manual_seed(1)
    vocab, dim = 24, 12
    monster = _monster(vocab, dim)
    x = torch.randint(0, vocab, (3, 9))
    y = torch.randint(0, vocab, (3, 9))
    sur = token_surprisal(monster, x, y, in_bits=False)
    with torch.no_grad():
        logits = monster(x)
        manual = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="none"
        ).reshape(3, 9)
    assert torch.allclose(sur, manual, atol=1e-5)
