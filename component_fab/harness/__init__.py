"""Standard test harnesses — canonical wrappers for grading components.

A new lane is plugged into one of these wrappers and graded under
identical conditions. The wrappers stay decoupled from research/synthesis
so the fab can evolve independently.
"""

from .probe_block import ProbeResult, WinnerLikeBlock, short_training_probe
from .standard_block import LaneTestBlock, make_lane_test_block

__all__ = [
    "LaneTestBlock",
    "ProbeResult",
    "WinnerLikeBlock",
    "make_lane_test_block",
    "short_training_probe",
]
