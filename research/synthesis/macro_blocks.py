"""Macro-block helpers for grammar action `macro_block`.

This module intentionally keeps behavior simple: delegate to template
selection so grammar-level macro-block actions remain available even when
specialized macro libraries are absent.
"""

from __future__ import annotations

import random

from .graph import ComputationGraph
from .templates import apply_random_template


def apply_random_macro_block(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Apply a random template as a macro-block expansion."""
    return apply_random_template(graph, node_id, rng)

