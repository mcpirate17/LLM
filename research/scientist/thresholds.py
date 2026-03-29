"""Centralized threshold constants for scoring and escalation gates.

Every hardcoded numeric threshold used in promotion decisions lives here.
Each constant includes: value, calibration date, and rationale.

loss_ratio convention: all thresholds compare against RAW loss_ratio
(final_loss / initial_loss), NOT NORM (final_loss / ln(vocab_size)).
See results_auto_escalate_phase7.py header for the full distinction.
"""

from __future__ import annotations

from typing import NewType

# Type alias enforcing that thresholds compare RAW loss_ratio only.
# RAW = final_loss / initial_loss  (range 0-1+, lower is better)
# NORM = final_loss / ln(vocab_size) (range 0-1+, different scale)
RawLossRatio = NewType("RawLossRatio", float)


# ---------------------------------------------------------------------------
# Screening -> Investigation gate
# ---------------------------------------------------------------------------

# 90% of frontier avg composite score (69.67) at screening.
# Calibrated 2026-03-23 against wiki103 4-ref frontier.
V7_SCREENING_THRESHOLD: float = 62.7

# ---------------------------------------------------------------------------
# Investigation -> Validation gate
# ---------------------------------------------------------------------------

# 90% of frontier avg composite score at investigation tier.
# Calibrated 2026-03-23 against wiki103 4-ref frontier.
V7_INVESTIGATION_THRESHOLD: float = 121.2

# ---------------------------------------------------------------------------
# Empirical override gate (investigation -> validation bypass)
# ---------------------------------------------------------------------------

# Best RAW loss_ratio must be below this for empirical override.
# Calibrated against RAW formula; under NORM 0.174 is nearly unreachable.
EMPIRICAL_OVERRIDE_BEST_LR: float = 0.18

# Baseline loss_ratio must be below this for empirical override.
# Prevents override when baseline comparison is inconclusive.
EMPIRICAL_OVERRIDE_BASELINE_LR: float = 0.80

# Robustness floor for empirical override pathway.
# Candidate must have robustness >= this to qualify.
EMPIRICAL_OVERRIDE_ROBUSTNESS: float = 0.50

# Score multiplier: candidate_score must be >= min_score * this.
# Ensures empirical override candidates are close to the threshold.
EMPIRICAL_OVERRIDE_SCORE_MULT: float = 1.15

# ---------------------------------------------------------------------------
# Validation hard gate (loss_ratio)
# ---------------------------------------------------------------------------

# Maximum RAW loss_ratio for validation promotion.
# Calibrated 2026-03-23; prevents weak learners from reaching validation.
VALIDATION_BEST_LR_HARD: float = 0.25

# ---------------------------------------------------------------------------
# Investigation early pass thresholds
# ---------------------------------------------------------------------------

# RAW loss_ratio below which investigation is considered an early pass.
# Calibrated by judgment; no formal sweep performed.
INVESTIGATION_EARLY_PASS_LR: float = 0.50

# RAW loss_ratio below which brittle_risk is overridden.
# Even brittle models pass if they learn this well.
INVESTIGATION_BRITTLE_OVERRIDE_LR: float = 0.30

# ---------------------------------------------------------------------------
# Insufficient learning cap
# ---------------------------------------------------------------------------

# RAW loss_ratio above which a model is considered to have not learned.
# Triggers a hard score cap in composite scoring.
INSUFFICIENT_LEARNING_LR: float = 0.95

# ---------------------------------------------------------------------------
# Structural novelty floor
# ---------------------------------------------------------------------------

# Minimum structural novelty to pass lightning gate.
# Below this, the graph is too similar to existing population.
STRUCTURAL_NOVELTY_FLOOR: float = 0.10

# ---------------------------------------------------------------------------
# Novelty scoring caps
# ---------------------------------------------------------------------------

# Maximum composite novelty points when novelty_valid_for_promotion is False.
# Prevents structural-only novelty from dominating the score.
STRUCTURAL_ONLY_NOVELTY_CAP: float = 15.0

# ---------------------------------------------------------------------------
# Spectral norm floor
# ---------------------------------------------------------------------------

# Below this spectral_norm, the model is considered degenerate (collapsed).
# Used in multiple composite scoring versions.
SPECTRAL_NORM_FLOOR: float = 0.01
