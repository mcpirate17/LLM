"""Motif selection, index structures, and step resolution."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ._motif_catalog_core import CORE_MOTIFS
from ._motif_catalog_extended import EXTENDED_MOTIFS
from ._motif_catalog_slots import SLOT_MOTIFS
from ._motif_rules import _get_valid_activations
from ._motif_types import Motif, MotifStep
from ._selection_utils import context_pair_allowed
from .op_roles import OpRole


# ── Assembled motif list from all catalogs ─────────────────────────

_MOTIF_LIST: Tuple[Motif, ...] = CORE_MOTIFS + EXTENDED_MOTIFS + SLOT_MOTIFS

# ── Index structures for O(1) lookup ────────────────────────────────

VALIDATED_MOTIFS: Dict[str, Motif] = {m.name: m for m in _MOTIF_LIST}

MOTIFS_BY_CLASS: Dict[str, List[Motif]] = {}
for _m in _MOTIF_LIST:
    MOTIFS_BY_CLASS.setdefault(_m.motif_class, []).append(_m)

ALL_MOTIFS: Tuple[Motif, ...] = tuple(_MOTIF_LIST)

# Auto-register untapped-pair motifs from the pair_proposer pipeline.
# No-op unless ARIA_ENABLE_MINED_MOTIFS is set; mined motifs land in a
# dedicated 'mined_pair' class disjoint from the validated catalog so they
# only appear when a template explicitly requests that slot class.
from ._motifs_mined import register_mined_motifs as _register_mined_motifs  # noqa: E402

_register_mined_motifs(VALIDATED_MOTIFS, MOTIFS_BY_CLASS)

# Cross-pollinate component_fab inventions as grammar motifs (diversity generator
# M2). No-op unless ARIA_ENABLE_FAB_MOTIFS is set; lands in the dedicated
# 'fab_invention' class, reachable via the wildcard exploration path.
from ._motifs_fab import register_fab_motifs as _register_fab_motifs  # noqa: E402

_register_fab_motifs(VALIDATED_MOTIFS, MOTIFS_BY_CLASS)

# Alias for backward compatibility (used by execution_candidates.py)
MOTIF_LIBRARY: Dict[str, Motif] = VALIDATED_MOTIFS


# Audit fix 2026-04-17 (P1.7): motifs with lift < this floor are excluded
# from sampling unless an explicit ``weights`` override provides a positive
# weight for them. Set to 0.5 to filter only the actively-broken motifs
# (4 motifs with lift < 0.5, including the demoted attn_softmax) while
# keeping borderline-positive routing motifs that templates require under
# routing_mandatory=True. Override per-call via ``min_lift``; 0.0 disables.
MIN_MOTIF_LIFT: float = 0.5


def _filter_by_lift(
    candidates: List[Motif],
    weights: Optional[Dict[str, float]],
    min_lift: float,
) -> List[Motif]:
    """Drop motifs whose lift is below ``min_lift`` AND have no override weight.

    A motif explicitly weighted by the caller is always kept — overrides exist
    precisely so a learned policy can pull a low-lift motif back into the
    pool when it sees an opportunity.
    """
    if min_lift <= 0:
        return candidates
    if weights:
        return [
            m for m in candidates if m.lift >= min_lift or weights.get(m.name, 0) > 0
        ]
    return [m for m in candidates if m.lift >= min_lift]


def pick_motif(
    rng: random.Random,
    motif_class: str,
    weights: Optional[Dict[str, float]] = None,
    *,
    min_lift: float = MIN_MOTIF_LIFT,
) -> Optional[Motif]:
    """Pick a random motif from the given class, weighted by lift or custom weights.

    Motifs with lift below ``min_lift`` are excluded unless ``weights`` provides
    a positive override — see module docstring for ``MIN_MOTIF_LIFT``.
    """
    candidates = MOTIFS_BY_CLASS.get(motif_class)
    if not candidates:
        return None
    candidates = _filter_by_lift(list(candidates), weights, min_lift)
    if not candidates:
        # Floor wiped the pool: relax to original list rather than emit None,
        # since downstream may not have a graceful fallback.
        candidates = list(MOTIFS_BY_CLASS.get(motif_class, ()))
        if not candidates:
            return None
    if len(candidates) == 1:
        return candidates[0]
    w = [weights.get(m.name, m.lift) if weights else m.lift for m in candidates]
    return rng.choices(candidates, weights=w, k=1)[0]


def pick_motif_from_classes(
    rng: random.Random,
    classes: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
    *,
    min_lift: float = MIN_MOTIF_LIFT,
) -> Optional[Motif]:
    """Pick a motif from any of the given classes (lift-floor applies)."""
    pool: List[Motif] = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, []))
    if not pool:
        return None
    pool = _filter_by_lift(pool, weights, min_lift)
    if not pool:
        # Same fallback as pick_motif — never empty if any source class had ops.
        for cls in classes:
            pool.extend(MOTIFS_BY_CLASS.get(cls, []))
        if not pool:
            return None
    w = [weights.get(m.name, m.lift) if weights else m.lift for m in pool]
    return rng.choices(pool, weights=w, k=1)[0]


def resolve_step(
    step: MotifStep,
    rng: random.Random,
    *,
    prev_op: Optional[str] = None,
    next_op: Optional[str] = None,
    op_weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict]:
    """Resolve a motif step to a concrete (op_name, config) pair.

    Handles context-aware activation substitution for substitutable steps.
    When op_weights is provided, biases selection toward higher-weighted ops.
    """
    if step.substitutable and step.role == OpRole.ACTIVATE:
        candidates = _get_valid_activations(prev_op=prev_op, next_op=next_op)
        candidates = [
            candidate
            for candidate in candidates
            if context_pair_allowed(prev_op, candidate)
            and context_pair_allowed(candidate, next_op)
        ]
        if not candidates:
            # Ultimate fallback: still honor context rules. Only emit a safe
            # default if it doesn't violate prev_op/next_op forbidden pairs.
            # If even gelu/silu/relu are forbidden, return gelu and let the
            # graph validator reject the resulting chain explicitly.
            safe_defaults = ["gelu", "silu", "relu"]
            candidates = [
                c
                for c in safe_defaults
                if context_pair_allowed(prev_op, c) and context_pair_allowed(c, next_op)
            ]
            if not candidates:
                candidates = ["gelu"]
        if op_weights and len(candidates) > 1:
            weights = [op_weights.get(c, 1.0) for c in candidates]
            op_name = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            op_name = rng.choice(candidates)
    else:
        op_name = step.op_name
    return op_name, dict(step.config)
