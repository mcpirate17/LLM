"""Shared scipy-backed statistics helpers for the state analyzers."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as _scipy_stats


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; 0.0 for degenerate input (n < 2 or zero variance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return 0.0
    with warnings.catch_warnings():
        # Constant input is a defined degenerate case here (-> 0.0), not noise
        # worth a per-call warning in analyzer loops.
        warnings.simplefilter("ignore", _scipy_stats.ConstantInputWarning)
        rho = _scipy_stats.spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else 0.0
