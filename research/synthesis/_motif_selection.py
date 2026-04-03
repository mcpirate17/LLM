"""Motif selection, index structures, and step resolution."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ._motif_catalog_core import CORE_MOTIFS
from ._motif_catalog_extended import EXTENDED_MOTIFS
from ._motif_catalog_slots import SLOT_MOTIFS
from ._motif_rules import _get_valid_activations
from ._motif_types import Motif, MotifStep
from .op_roles import OpRole

# ── Assembled motif list from all catalogs ─────────────────────────

_MOTIF_LIST: Tuple[Motif, ...] = CORE_MOTIFS + EXTENDED_MOTIFS + SLOT_MOTIFS

# ── Index structures for O(1) lookup ────────────────────────────────

VALIDATED_MOTIFS: Dict[str, Motif] = {m.name: m for m in _MOTIF_LIST}

MOTIFS_BY_CLASS: Dict[str, List[Motif]] = {}
for _m in _MOTIF_LIST:
    MOTIFS_BY_CLASS.setdefault(_m.motif_class, []).append(_m)

ALL_MOTIFS: Tuple[Motif, ...] = tuple(_MOTIF_LIST)

# Alias for backward compatibility (used by execution_candidates.py)
MOTIF_LIBRARY: Dict[str, Motif] = VALIDATED_MOTIFS


def pick_motif(
    rng: random.Random,
    motif_class: str,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Motif]:
    """Pick a random motif from the given class, weighted by lift or custom weights."""
    candidates = MOTIFS_BY_CLASS.get(motif_class)
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
) -> Optional[Motif]:
    """Pick a motif from any of the given classes."""
    pool: List[Motif] = []
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
        if op_weights and len(candidates) > 1:
            weights = [op_weights.get(c, 1.0) for c in candidates]
            op_name = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            op_name = rng.choice(candidates)
    else:
        op_name = step.op_name
    return op_name, dict(step.config)
