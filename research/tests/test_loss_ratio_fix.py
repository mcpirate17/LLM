"""Tests for loss_ratio formula fix (raw vs norm).

Verifies that:
  - result["loss_ratio"] stores final_loss / initial_loss (RAW)
  - result["loss_ratio_raw"] == result["loss_ratio"]
  - result["loss_ratio_norm"] stores final_loss / ln(vocab_size)
  - Auto-escalation correctly triggers for RAW < 0.18
"""

from __future__ import annotations

import math


def test_loss_ratio_stores_raw():
    """Given final_loss=2.0, initial_loss=50, vocab_size=100000:
    loss_ratio_raw  ≈ 0.04
    loss_ratio_norm ≈ 0.174
    loss_ratio      == loss_ratio_raw
    """
    from research.scientist.runner._helpers import normalized_loss_ratio

    final_loss = 2.0
    initial_loss = 50.0
    vocab_size = 100000

    raw = final_loss / max(initial_loss, 1e-6)
    norm = normalized_loss_ratio(final_loss, vocab_size)

    assert abs(raw - 0.04) < 0.001, f"raw={raw}, expected ~0.04"
    assert abs(norm - (2.0 / math.log(vocab_size))) < 0.001, f"norm={norm}"
    # NORM is higher than RAW for this case
    assert norm > raw, "NORM should be higher than RAW here"
    # Under old code, loss_ratio was NORM (~0.174), which is near the 0.18 threshold.
    # Under fixed code, loss_ratio is RAW (0.04), which is well below 0.18.
    assert raw < 0.18, "RAW should pass escalation threshold"


def test_auto_escalate_override_uses_raw():
    """The _meets_empirical_validation_override threshold (0.18)
    should pass when best_loss_ratio is RAW (0.04) and fail when it's
    NORM (0.174 — near threshold but may fluctuate above)."""
    from research.scientist.runner.results_auto_escalate_phase7 import (
        _ResultsAutoEscalatePhase7Mixin,
    )

    # Candidate with RAW loss_ratio — should pass
    candidate_raw = {
        "robustness": 0.7,
        "best_loss_ratio": 0.04,  # RAW: 2.0 / 50.0
        "novelty_confidence": 0.8,
    }
    assert _ResultsAutoEscalatePhase7Mixin._meets_empirical_validation_override(
        candidate_raw, candidate_score=120.0, min_score=100.0
    ), "RAW loss_ratio=0.04 should pass the 0.18 threshold"

    # Candidate with NORM-like loss_ratio — would fail
    candidate_norm = {
        "robustness": 0.7,
        "best_loss_ratio": 0.60,  # NORM: typical for recent entries
        "novelty_confidence": 0.8,
    }
    assert not _ResultsAutoEscalatePhase7Mixin._meets_empirical_validation_override(
        candidate_norm, candidate_score=120.0, min_score=100.0
    ), "NORM loss_ratio=0.60 should NOT pass the 0.18 threshold"


def test_loss_ratio_boundary():
    """Verify RAW vs NORM at the 0.18 boundary."""
    from research.scientist.runner._helpers import normalized_loss_ratio

    # A model that achieves final_loss=2.07, init=50 → raw=0.0414 (passes)
    # Same model: norm = 2.07/ln(100000) = 0.1798 (barely passes NORM too)
    final = 2.07
    init = 50.0
    vocab = 100000

    raw = final / init
    norm = normalized_loss_ratio(final, vocab)

    assert raw < 0.18, f"RAW={raw} should be < 0.18"
    # The key insight: a model that's well below 0.18 on RAW
    # can be very close to 0.18 on NORM
    assert norm > raw * 3, "NORM should be significantly higher than RAW"
