"""Compatibility exports for canonical math-knob definitions.

The implementation lives in ``component_fab.math_knobs`` so generator and
validator imports do not load the full improver package during initialization.
"""

from __future__ import annotations

from component_fab.math_knobs import (
    AUTO_DEEPENING_MATH_KNOBS,
    DEFAULT_MATH_KNOBS,
    KNOB_ID_BY_FAMILY,
    MathKnob,
    math_knobs_from_axes,
)

__all__ = [
    "AUTO_DEEPENING_MATH_KNOBS",
    "DEFAULT_MATH_KNOBS",
    "KNOB_ID_BY_FAMILY",
    "MathKnob",
    "math_knobs_from_axes",
]
