"""Leaderboard column constants — single source of truth.

Scoring and fingerprint-aggregation column maps used by
``leaderboard_scoring`` and ``notebook_leaderboard._LeaderboardMixin``.
"""
from __future__ import annotations

# Column→parameter mapping for ``compute_composite_score``.
# Keys are leaderboard/program_results column names; values are the kwarg
# names accepted by ``compute_composite_score``.
SCORE_COLUMN_MAP: dict[str, str] = {
    "screening_loss_ratio": "screening_lr",
    "screening_novelty": "screening_nov",
    "investigation_loss_ratio": "inv_lr",
    "investigation_robustness": "inv_robust",
    "validation_loss_ratio": "val_lr",
    "validation_baseline_ratio": "val_baseline",
    "validation_multi_seed_std": "val_std",
    "validation_robustness_score": "robustness_score",
    "validation_is_unstable": "is_unstable",
    "routing_savings_ratio": "routing_savings",
    "compression_ratio": "compression_ratio",
    "discovery_loss_ratio": "discovery_lr",
    "fp_jacobian_spectral_norm": "spectral_norm",
    "robustness_noise_score": "robustness_noise",
    "quant_int8_retention": "quant_retention",
    "robustness_long_ctx_score": "long_ctx_score",
    "init_sensitivity_std": "init_std",
    "quant_quality_per_byte": "quant_quality_per_byte",
    "ncd_score": "ncd_score",
    "recursion_savings_ratio": "recursion_savings",
    "depth_savings_ratio": "depth_savings",
    "activation_sparsity_score": "activation_sparsity",
    "max_viable_seq_len": "max_viable_seq_len",
    "robustness_long_ctx_scaling_score": "long_ctx_scaling",
    "robustness_long_ctx_passkey_score": "long_ctx_passkey",
    "robustness_long_ctx_multi_hop_score": "long_ctx_multi_hop",
    "robustness_long_ctx_assoc_score": "long_ctx_assoc",
    "routing_expert_count": "routing_expert_count",
    "routing_confidence_mean": "routing_confidence_mean",
    "routing_drop_rate": "routing_drop_rate",
    "wikitext_perplexity": "wikitext_perplexity",
    "wikitext_score": "wikitext_score",
    "peak_ppl": "peak_ppl",
    "ppl_500": "ppl_500",
    "steps_to_divergence": "steps_to_divergence",
    "investigation_passed": "investigation_passed",
    "validation_passed": "validation_passed",
}

# Columns used in _sync_fingerprint_leaderboard for best-of-run aggregation.
FINGERPRINT_MIN_COLS: tuple[str, ...] = (
    "screening_loss_ratio", "investigation_loss_ratio",
    "validation_loss_ratio", "validation_baseline_ratio",
    "validation_multi_seed_std", "discovery_loss_ratio",
    "compression_ratio", "routing_drop_rate",
    "robustness_noise_score", "wikitext_perplexity",
    "tinystories_perplexity", "ncd_score",
)

FINGERPRINT_MAX_COLS: tuple[str, ...] = (
    "screening_novelty", "investigation_robustness",
    "normalized_baseline_ratio", "param_efficiency",
    "quant_int8_retention", "quant_quality_per_byte",
    "robustness_long_ctx_score", "init_sensitivity_std",
    "scaling_param_efficiency", "scaling_flop_efficiency",
    "scaling_d512_param_efficiency", "routing_savings_ratio",
    "activation_sparsity_score", "depth_savings_ratio",
    "recursion_savings_ratio", "routing_expert_count",
    "routing_confidence_mean", "efficiency_multiple",
    "wikitext_score", "tinystories_score",
    "cross_task_score", "efficiency_wall_score",
    "max_viable_seq_len",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "loss_improvement_rate",
)

FINGERPRINT_BOOL_COLS: tuple[str, ...] = (
    "screening_passed", "investigation_passed",
    "validation_passed", "scaling_gate_passed",
)

# Columns written back to all fingerprint-sibling rows after aggregation.
FINGERPRINT_UPDATE_COLS: tuple[str, ...] = (
    "tier", "composite_score",
    *FINGERPRINT_MIN_COLS,
    *[c for c in FINGERPRINT_MAX_COLS if c not in ("loss_improvement_rate",)],
    "loss_improvement_rate",
    *FINGERPRINT_BOOL_COLS,
    "ncd_score", "efficiency_multiple", "timestamp",
)
# Deduplicate while preserving order.
_seen: set[str] = set()
_deduped: list[str] = []
for _c in FINGERPRINT_UPDATE_COLS:
    if _c not in _seen:
        _seen.add(_c)
        _deduped.append(_c)
FINGERPRINT_UPDATE_COLS = tuple(_deduped)
del _seen, _deduped, _c
