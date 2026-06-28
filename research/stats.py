"""Small shared statistical helpers."""

from __future__ import annotations

import math


def wilson_score_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = float(k) / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / float(n)
    centre = (p + z2 / (2.0 * float(n))) / denom
    half = (
        float(z)
        * math.sqrt(p * (1.0 - p) / float(n) + z2 / (4.0 * float(n) * float(n)))
    ) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
