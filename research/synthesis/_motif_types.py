"""Motif dataclasses and class constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

from .op_roles import OpRole


@dataclass(slots=True, frozen=True)
class MotifStep:
    """A single step in a motif's op sequence."""

    op_name: str
    role: OpRole
    config: Dict = field(default_factory=dict)
    # If True, this step's op can be substituted with any op of the same role.
    substitutable: bool = False


@dataclass(slots=True, frozen=True)
class Motif:
    """A validated functional unit — 2-4 ops that compose correctly."""

    name: str
    motif_class: str  # e.g., "ffn_core", "attention_core", "ssm_core"
    steps: Tuple[MotifStep, ...]
    description: str = ""
    # Statistical evidence from mining
    support: int = 0  # Number of top performers containing this pattern
    avg_loss_ratio: float = 0.0
    lift: float = 1.0  # Enrichment in winners vs general population


# ── Motif Class Constants ───────────────────────────────────────────

MOTIF_CLASS_FFN = "ffn_core"
MOTIF_CLASS_ATTENTION = "attention_core"
MOTIF_CLASS_SSM = "ssm_core"
MOTIF_CLASS_CONV = "conv_core"
MOTIF_CLASS_GATE = "gate_core"
MOTIF_CLASS_NORM = "norm_wrap"
MOTIF_CLASS_SPARSE = "sparse_core"
MOTIF_CLASS_MOE = "moe_core"
MOTIF_CLASS_CHANNEL = "channel_core"
MOTIF_CLASS_EFFICIENT_PROJ = "efficient_proj"
MOTIF_CLASS_REDUCE = "reduce_core"
MOTIF_CLASS_GUARDED_ACT = "guarded_act"
MOTIF_CLASS_MATH_SPACE = "math_space"
# Dynamic provenance class for cross-pollinated component_fab inventions. Kept
# OUT of ALL_MOTIF_CLASSES (like ``mined_pair``) so it never silently replaces a
# validated motif; reachable only via the wildcard exploration path (it is added
# to ``_template_helpers._ALL_CLASSES``) and explicit ``exploration_targets``.
MOTIF_CLASS_FAB = "fab_invention"

ALL_MOTIF_CLASSES: FrozenSet[str] = frozenset(
    {
        MOTIF_CLASS_FFN,
        MOTIF_CLASS_ATTENTION,
        MOTIF_CLASS_SSM,
        MOTIF_CLASS_CONV,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_NORM,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_MOE,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_REDUCE,
        MOTIF_CLASS_GUARDED_ACT,
        MOTIF_CLASS_MATH_SPACE,
    }
)
