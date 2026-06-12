"""
Validated Motif Library for Compositional Grammar

A motif is a 2-4 op chain empirically validated to:
  (a) produce gradients, (b) learn, (c) be numerically stable.

Motifs are the atoms of the new grammar — known-good component
combinations that templates compose into full architectures.

Mined from 734 top performers out of 4,959 candidates.
See: research/docs/motif_mining_report.md

Split into submodules by responsibility:
  _motif_types.py       — MotifStep, Motif dataclasses, class constants
  _motif_catalog_core.py    — FFN, attention, SSM, conv, gate, norm, channel motifs
  _motif_catalog_extended.py — sparse, MoE, functional, exotic motifs
  _motif_catalog_slots.py   — guarded acts, routing, position, reduction, math-space
  _motif_rules.py       — ACTIVATION_POOL, ACTIVATION_RULES, MATH_SPACE_RULES
  _motif_selection.py   — pick_motif, resolve_step, index structures
"""

# Re-export everything so `from .motifs import X` continues to work.

from ._motif_rules import (  # noqa: F401
    ACTIVATION_POOL,
    ACTIVATION_RULES,
    MATH_SPACE_RULES,
    _get_valid_activations,
)
from ._motif_selection import (  # noqa: F401
    ALL_MOTIFS,
    MOTIF_LIBRARY,
    MOTIFS_BY_CLASS,
    VALIDATED_MOTIFS,
    _MOTIF_LIST,
    pick_motif,
    pick_motif_from_classes,
    resolve_step,
)
from ._motif_types import (  # noqa: F401
    ALL_MOTIF_CLASSES,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_FAB,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_SSM,
    Motif,
    MotifStep,
)
