"""Standard test harnesses — canonical wrappers for grading components.

A new lane / router / compressor is plugged into one of these wrappers
and graded under identical conditions. The wrappers stay decoupled from
research/synthesis so the fab can evolve independently.
"""

from .probe_block import ProbeResult, WinnerLikeBlock, short_training_probe
from .standard_block import (
    CanonicalLaneEnvironment,
    CompressionTestBlock,
    LaneTestBlock,
    RoutingTestBlock,
    make_canonical_lanes,
    make_compression_test_block,
    make_lane_test_block,
    make_routing_test_block,
)

__all__ = [
    "CanonicalLaneEnvironment",
    "CompressionTestBlock",
    "LaneTestBlock",
    "ProbeResult",
    "RoutingTestBlock",
    "WinnerLikeBlock",
    "make_canonical_lanes",
    "make_compression_test_block",
    "make_lane_test_block",
    "make_routing_test_block",
    "short_training_probe",
]
