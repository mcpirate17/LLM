"""Improvement track — turn underperforming-novel anchors into axis-variants.

For each goal-(b) anchor op (``tropical_attention``, ``clifford_attention``,
``padic_gate``, etc.), enumerate axis variants (add state, swap basis,
add top-k routing), generate runnable modules via the code generator,
grade via the solo validator, and rank.
"""

from .axis_variants import (
    AnchorAxes,
    AxisVariant,
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    anchor_axes_for_op,
    enumerate_axis_variants,
    spec_for_variant,
)
from .ranking import (
    RankedEntry,
    composite_score,
    cross_check_subscore,
    leaderboard_to_json,
    learning_subscore,
    rank_proposals,
    smoke_subscore,
)
from .math_knob_catalog import (
    DEFAULT_MATH_KNOBS,
    KnobStackScore,
    MathKnob,
    enumerate_adaptive_math_knob_compositions,
    enumerate_math_knob_compositions,
    score_knob_stack,
)

__all__ = [
    "AnchorAxes",
    "AxisVariant",
    "DEFAULT_AXIS_VARIANT_TEMPLATES",
    "DEFAULT_MATH_KNOBS",
    "KnobStackScore",
    "MathKnob",
    "RankedEntry",
    "anchor_axes_for_op",
    "composite_score",
    "cross_check_subscore",
    "enumerate_adaptive_math_knob_compositions",
    "enumerate_axis_variants",
    "enumerate_math_knob_compositions",
    "leaderboard_to_json",
    "learning_subscore",
    "rank_proposals",
    "score_knob_stack",
    "smoke_subscore",
    "spec_for_variant",
]
