"""Centralized thresholds for scoring and escalation gates.

loss_ratio convention: all thresholds compare against RAW loss_ratio
(final_loss / initial_loss), NOT NORM (final_loss / ln(vocab_size)).
See results_auto_escalate_phase7.py header for the full distinction.
"""

from __future__ import annotations

from typing import NewType

RawLossRatio = NewType("RawLossRatio", float)


# 90% of frontier avg composite score (69.67) at screening.
# Calibrated 2026-03-23 against wiki103 4-ref frontier.
V7_SCREENING_THRESHOLD: float = 62.7

# 90% of frontier avg composite score at investigation tier.
# Calibrated 2026-03-23 against wiki103 4-ref frontier.
V7_INVESTIGATION_THRESHOLD: float = 121.2

# Empirical override gate. Calibrated against RAW formula; under NORM 0.174 is nearly unreachable.
EMPIRICAL_OVERRIDE_BEST_LR: float = 0.18

# Prevents override when baseline comparison is inconclusive.
EMPIRICAL_OVERRIDE_BASELINE_LR: float = 0.80

EMPIRICAL_OVERRIDE_ROBUSTNESS: float = 0.50

EMPIRICAL_OVERRIDE_SCORE_MULT: float = 1.15

# Maximum RAW loss_ratio for validation promotion.
# Calibrated 2026-03-23; prevents weak learners from reaching validation.
VALIDATION_BEST_LR_HARD: float = 0.25

# RAW loss_ratio below which investigation is considered an early pass.
# Calibrated by judgment; no formal sweep performed.
INVESTIGATION_EARLY_PASS_LR: float = 0.50

# RAW loss_ratio below which brittle_risk is overridden.
# Even brittle models pass if they learn this well.
INVESTIGATION_BRITTLE_OVERRIDE_LR: float = 0.30

# RAW loss_ratio above which a model is considered to have not learned.
# Triggers a hard score cap in composite scoring.
INSUFFICIENT_LEARNING_LR: float = 0.95

# Minimum structural novelty to pass lightning gate.
# Below this, the graph is too similar to existing population.
STRUCTURAL_NOVELTY_FLOOR: float = 0.10

# Prevents structural-only novelty from dominating the score when novelty is invalid for promotion.
STRUCTURAL_ONLY_NOVELTY_CAP: float = 15.0

# Below this spectral_norm, the model is considered degenerate (collapsed).
# Used in multiple composite scoring versions.
SPECTRAL_NORM_FLOOR: float = 0.01

GPT2_REF = {
    "loss_ratio": 0.2646,
    "param_count": 9_767_424,
    "flops_forward": 19_534_848,
    "throughput_tok_s": 1_200_845,
    "peak_memory_mb": 115.0,
    "forward_time_ms": 0.43,
}

# WikiText reference score floor and perplexity ceiling.
# Calibrated 2026-03-23 against wiki103 4-ref frontier.
WIKITEXT_REF_SCORE_FLOOR: float = 0.5868
WIKITEXT_REF_PPL_CEILING: float = 72.68

# Accuracy at or below this = noise floor → hard score cap, and the
# screening_understanding_filter treats it as "probe measured but near-random".
# 4-choice random is 25%; binomial std at n=50 is ~6%, so 0.30 is roughly
# 1σ above random — tighter than the old 0.28 (~0.5σ) but still below GPT-2
# Small's ~0.31 so the noise floor doesn't start rejecting GPT-2-class models.
# Hard promotion gate lives in UNDERSTANDING_MIN_HELLASWAG (0.40).
# Calibrated 2026-04-17.
HELLASWAG_RANDOM_CHANCE_GATE: float = 0.30

# Score cap applied when hellaswag_acc <= HELLASWAG_RANDOM_CHANCE_GATE.
# Prevents perplexity-only optimizers from dominating the leaderboard.
HELLASWAG_RANDOM_CHANCE_CAP: float = 25.0

# GPT-2 Small HellaSwag acc_norm (~31%) — frontier anchor for S-curve.
# Measured on d_model=768, 12-layer, 124M param reference. Retained as the
# empirical reference for scoring; the capability-first promotion gate is
# decoupled (UNDERSTANDING_MIN_HELLASWAG = 0.40).
HELLASWAG_FRONTIER_ACC: float = 0.31

# Investigation hard gate: 100 examples → lower variance than screening.
# Models still at noise-floor with 100 examples → investigation_failed.
# Calibrated 2026-04-17 (moved with RANDOM_CHANCE_GATE). Slightly below the
# screening gate because more examples reduce noise (binomial std ~4.3% at
# n=100 vs ~6% at n=50), so the same true accuracy reads tighter here.
HELLASWAG_INVESTIGATION_GATE: float = 0.29

# Validation hard gate: 200 examples → even lower variance.
# Blocks investigation→validation promotion for noise-floor models.
# Calibrated 2026-04-17.
HELLASWAG_VALIDATION_GATE: float = 0.29

# Soft gate at ALL tiers: fires only when ALL measured binding signals are
# near zero simultaneously (3-signal AND). This is the pure conv-3 case.
#
# CRITICAL: The induction probe measures exact token retrieval across a gap.
# Only full attention reliably passes it. Mamba/SSM/RWKV score ~0 on
# induction but their failure mechanism is fundamentally different from
# conv-3 — state compression cannot do exact retrieval, but these models
# still have real non-local perplexity capability. Do NOT penalize them.
#
# Expected scores at nano scale (d_model=256, 1000 training steps):
#   Causal transformer (attention): induction=0.5-0.6, binding_screening_auc>0.3, ar>0.01
#   Mamba/SSM/recurrent:            induction≈0.0-0.05, binding_screening_auc>0.1, ar≈0.0-0.05
#   RWKV:                           induction≈0.0-0.05, binding_screening_auc>0.1, ar≈0.0-0.05
#   Champion c9c7075e (conv-3):     induction≈0.003, binding_screening_auc<0.01, ar<0.01
#
# The penalty fires for conv-3 (all three below) but NOT for Mamba/RWKV
# (binding_screening_auc above threshold).
BINDING_AR_SOFT_GATE: float = 0.05
BINDING_INDUCTION_SOFT_GATE: float = 0.05
BINDING_BINDING_AUC_SOFT_GATE: float = 0.10
BINDING_LOCAL_ONLY_PENALTY: float = 0.80  # multiply composite by this

# Calibrated 2026-04-17 (capability-first): require ≥ UNDERSTANDING_MIN_SIGNALS
# of {diagnostic, binding_screening_composite, hellaswag} above the corresponding strict
# threshold. The legacy soft thresholds (diagnostic=0.15 etc.) are retained
# below for diagnostic logging only — they are NOT the promotion criterion.
UNDERSTANDING_MIN_DIAGNOSTIC: float = 0.30  # ~2× above 4-task random (~12.5%)
UNDERSTANDING_MIN_BINDING: float = 0.10  # 2× the soft AND-gate floor
UNDERSTANDING_MIN_HELLASWAG: float = 0.40  # ~15% above 4-way random
UNDERSTANDING_MIN_SIGNALS: int = 2  # of 3, must clear the gate above

# Soft diagnostic-only thresholds, used for screening filtering and logging:
# the screening→investigation filter only blocks candidates whose probes have
# been measured AND are all below these soft floors (i.e., demonstrably weak).
UNDERSTANDING_SOFT_DIAGNOSTIC: float = 0.15
UNDERSTANDING_SOFT_BINDING: float = 0.05
# UNDERSTANDING_SOFT_HELLASWAG uses HELLASWAG_RANDOM_CHANCE_GATE (0.28)

# v8 scoring thresholds (placeholder — calibrate after the current scoring backfill)
# These should be re-estimated from frontier references before enforcing them.
V8_SCREENING_THRESHOLD: float = 50.0  # placeholder, ~90% of v8 frontier avg
V8_INVESTIGATION_THRESHOLD: float = 95.0  # placeholder, ~90% of v8 frontier avg

TIER_RANK = {
    "screened_out": 0,
    "screening": 1,
    "investigation_failed": 1,
    "investigation_fingerprint_incomplete": 1,
    "investigation": 2,
    "capability_ranking": 2.5,
    "validation": 3,
    "breakthrough": 4,
}
