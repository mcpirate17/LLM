from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from research.frontier_kernel import pareto_frontier_mask


def pareto_mask(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    minimize: Iterable[bool] | None = None,
) -> np.ndarray:
    """Return a boolean mask for the non-dominated rows in ``points``."""
    costs = np.asarray(points, dtype=np.float32)
    if costs.ndim != 2:
        raise ValueError("pareto_mask expects a 2D objective matrix")
    if minimize is not None:
        minimize_directions = tuple(minimize)
    else:
        minimize_directions = (True,) * costs.shape[1]
    return pareto_frontier_mask(costs, minimize=minimize_directions)
