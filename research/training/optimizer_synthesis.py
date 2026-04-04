"""Public optimizer synthesis API.

This module stays as the stable import surface while the implementation is split
into focused modules. The supported set is intentionally small: stock AdamW
variants, Muon, and RigL. The previous speculative Python optimizer zoo has
been removed.
"""

from ._optimizer_factory import build_optimizer as build_optimizer
from ._optimizer_muon import MuonOptimizer as MuonOptimizer
from ._optimizer_synth import (
    OPTIMIZER_RECIPES as OPTIMIZER_RECIPES,
    SynthesizedOptimizer as SynthesizedOptimizer,
    synthesize_optimizer as synthesize_optimizer,
)

__all__ = [
    "OPTIMIZER_RECIPES",
    "SynthesizedOptimizer",
    "synthesize_optimizer",
    "build_optimizer",
    "MuonOptimizer",
]
