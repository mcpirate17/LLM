"""In-loop monster-surprisal routing (Workstream D, increment 5).

Closes the surprisal -> route -> carrier loop *during training*:

1. A frozen loss-monster scores per-token surprisal on the live batch (the
   in-loop version of ``research/tools/loss_monster_surprisal.py``).
2. The data-pipeline grammar turns that surprisal into route segments + a gate
   bias (``route_segment_ids`` / ``gate_bias_from_segments``).
3. The bias is pushed onto every routing block in the carrier so the hardest
   tokens route to the long-range carrier lane and the predictable rest stay on
   the cheap local (loss-monster) lane, this step.

Decoupled from ``component_fab`` by duck-typing: any submodule exposing a
settable ``route_prior`` attribute is updated. ``LossMonsterPairedBlock`` is the
canonical consumer, but anything with the same hook works — research/ does not
import component_fab.

Graded on capability at a fixed token budget, never loss.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from research.synthesis.data_pipeline_grammar import (
    DataRouteSpec,
    gate_bias_from_segments,
    route_segment_ids,
)

_LN2 = math.log(2.0)


@torch.no_grad()
def token_surprisal(
    monster: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    in_bits: bool = True,
) -> torch.Tensor:
    """Per-token surprisal ``[B, L]`` = ``-log p(target | context)`` under ``monster``.

    Detached (the monster is a fixed scorer; no gradient flows to the carrier).
    The monster is left in whatever mode the caller set — loss-monster halt/depth
    graphs are scored in train mode (W0 note), so this does not toggle ``eval()``.
    """
    logits = monster(input_ids)
    if logits.ndim != 3:
        raise ValueError(
            f"monster must output [B, L, V] logits, got {tuple(logits.shape)}"
        )
    vocab = logits.shape[-1]
    sur = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    ).reshape(input_ids.shape)
    return sur / _LN2 if in_bits else sur


def set_route_prior_from_surprisal(
    carrier: torch.nn.Module,
    surprisal: torch.Tensor,
    spec: DataRouteSpec,
    *,
    strength: float = 4.0,
) -> int:
    """Push a surprisal-derived gate bias onto every routing block in ``carrier``.

    Returns the number of blocks updated. Zero means nothing in the carrier
    consumes routing (fail loud at the call site if a route was expected to fire).
    """
    if spec.route != "surprisal_split":
        raise ValueError(
            f"surprisal routing needs route='surprisal_split', got {spec.route!r}"
        )
    segments = route_segment_ids(spec, surprisal=surprisal)
    bias = gate_bias_from_segments(segments, strength=strength)
    updated = 0
    for module in carrier.modules():
        if hasattr(module, "route_prior"):
            module.route_prior = bias
            updated += 1
    return updated


def clear_route_prior(carrier: torch.nn.Module) -> None:
    """Reset every block's ``route_prior`` to ``None`` (avoid a stale batch-shaped
    bias leaking into the next forward, e.g. eval with a different batch size)."""
    for module in carrier.modules():
        if hasattr(module, "route_prior"):
            block: Any = module
            block.route_prior = None


def surprisal_routed_logits(
    carrier: torch.nn.Module,
    monster: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    spec: DataRouteSpec,
    *,
    strength: float = 4.0,
    require_consumer: bool = True,
) -> torch.Tensor:
    """One routed forward: score surprisal, set the prior, run the carrier, clear.

    The drop-in training-step body for a paired carrier — replaces a plain
    ``carrier(input_ids)`` so the loss-monster surprisal steers routing live.
    """
    surprisal = token_surprisal(monster, input_ids, targets)
    n = set_route_prior_from_surprisal(carrier, surprisal, spec, strength=strength)
    if require_consumer and n == 0:
        raise ValueError(
            "surprisal route set but no carrier block consumes route_prior; "
            "expected a LossMonsterPairedBlock (or a route_prior hook) in the carrier"
        )
    try:
        return carrier(input_ids)
    finally:
        clear_route_prior(carrier)
