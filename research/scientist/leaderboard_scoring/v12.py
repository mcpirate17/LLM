"""v12 — loss-budget rebalance + champion eligibility gate (2026-04-29).

Builds on v11 by:
  1. Shrinking the loss tier (perf_short/medium/long, param_eff, learn_eff,
     speed, early_convergence) toward a 175pt budget.
  2. Adding a validation-tier AR Validation/easy25 rank-order signal.
  3. Gating high-scoring rows that lack induction/AR-validation/SSM-bypass
     evidence — prevents efficiency-only candidates from breaking 360.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ._config import (
    _LOSS_TIER_BD_KEYS,
    _V11_TRUST_BLIMP_FLOOR,
    _V11_TRUST_HELLASWAG_FLOOR,
    _V11_TRUST_PPL_FLOOR,
    _V12_CHAMPION_ELIGIBILITY_CEILING,
    _V14_CONFIG,
)
from .ar_validation import (
    _ar_validation_validation_signal,
    _score_ar_validation_validation_tier,
)
from .champion_tiny import _apply_champion_tiny_model_hard_failure_gate
from .v11 import compute_composite_v11


_V12_LOSS_COMPONENT_FACTORS = {
    "perf_short": 30.0 / 35.0,
    "perf_medium": 40.0 / 50.0,
    "perf_long": 55.0 / 65.0,
    "param_efficiency": 20.0 / 30.0,
    "learning_efficiency": 10.0 / 15.0,
    "speed": 15.0 / 25.0,
    "early_convergence": 5.0 / 10.0,
}
_V12_LOSS_BUDGET_MAX = 175.0
_V12_INDUCTION_QUALIFIED = 0.05
_V12_STRONG_INDUCTION = 0.30
_V12_BINDING_QUALIFIED = 0.20
_V12_STRONG_BINDING = 0.50
_V12_AR_VALIDATION_QUALIFIED = 0.55


def _v12_effective_signal(
    kw: Dict[str, Any],
    preferred: str,
    fallback: str,
) -> Optional[float]:
    value = kw.get(preferred)
    if value is None:
        value = kw.get(fallback)
    if value is None:
        return None
    return float(value)


def _v12_is_non_attention_exception_family(kw: Dict[str, Any]) -> bool:
    if bool(kw.get("non_attention_model")):
        return True
    haystack = " ".join(
        str(kw.get(key) or "").lower()
        for key in (
            "architecture_family",
            "model_family",
            "mechanism",
            "model_source",
            "op_names",
            "graph_ops",
            # template_name is the most populated identifier in practice —
            # the others are typically None unless the caller specifically
            # set them. Templates like `latent_attn_ssm_hybrid`,
            # `local_attn_ssm_hybrid`, `codex_ssm_*`, `spiking_*` all carry
            # the family signal in their name.
            "template_name",
        )
    )
    return any(
        token in haystack
        for token in (
            "mamba",
            "ssm",
            "state_space",
            "selective_scan",
            "rwkv",
            "recurrent",
        )
    )


def _v12_has_reproduced_bpe_loss(kw: Dict[str, Any]) -> bool:
    metric_version = (
        str(kw.get("screening_wikitext_metric_version") or "").strip().lower()
    )
    if metric_version != "bpe_eval_v1":
        return False
    return any(
        p is not None and float(p) > 0.0 and float(p) <= _V11_TRUST_PPL_FLOOR
        for p in (
            kw.get("ppl_validation"),
            kw.get("ppl_investigation"),
            kw.get("ppl_screening"),
        )
    )


# Per-signal bypass thresholds. AUC-style signals (passkey, multi_hop,
# scaling, combined, selective_copy) keep the 0.20 floor because they
# live on a [0,1] scale where chance is near 0 — 0.20 means "20pts above
# noise". ICLD is on a different scale: it's a loss-decline rate in
# nats/step. Cohort distribution (n=18,644 program_results rows) of
# |fp_icld_velocity|: p50=0.016, p75=0.026, p90=0.034, p99=0.043,
# max=0.160. The original 0.20 threshold qualified zero rows in the
# entire database. Setting ICLD threshold to 0.030 (≈p85 of the
# cohort) keeps it stringent — only the top ~15% of learners trigger
# the bypass signal — while making it physically reachable.
_V12_BYPASS_SIGNAL_THRESHOLDS: Dict[str, float] = {
    "selective_copy": 0.20,
    "long_ctx_passkey": 0.20,
    "long_ctx_multi_hop": 0.20,
    "long_ctx_scaling": 0.20,
    "long_ctx_combined": 0.20,
    "icld": 0.030,
}


def _v12_non_loss_sequence_signal_count(kw: Dict[str, Any]) -> int:
    thr = _V12_BYPASS_SIGNAL_THRESHOLDS
    pairs = (
        ("selective_copy", kw.get("selective_copy_score")),
        (
            "long_ctx_passkey",
            kw.get("long_ctx_passkey_score")
            or kw.get("robustness_long_ctx_passkey_score"),
        ),
        (
            "long_ctx_multi_hop",
            kw.get("long_ctx_multi_hop_score")
            or kw.get("robustness_long_ctx_multi_hop_score"),
        ),
        (
            "long_ctx_scaling",
            kw.get("long_ctx_scaling_score")
            or kw.get("robustness_long_ctx_scaling_score"),
        ),
        (
            "long_ctx_combined",
            kw.get("long_ctx_combined_score")
            or kw.get("robustness_long_ctx_combined_score"),
        ),
        ("icld", kw.get("icld_score") or kw.get("trajectory_learning_score")),
    )
    count = sum(
        1 for name, value in pairs if value is not None and float(value) >= thr[name]
    )

    hellaswag = max(
        float(v)
        for v in (
            kw.get("hellaswag_acc_validation") or 0.0,
            kw.get("hellaswag_acc_investigation") or 0.0,
            kw.get("hellaswag_acc_screening") or 0.0,
        )
    )
    if hellaswag >= _V11_TRUST_HELLASWAG_FLOOR:
        count += 1

    blimp = kw.get("blimp_accuracy")
    if blimp is not None and float(blimp) >= _V11_TRUST_BLIMP_FLOOR:
        count += 1
    return count


def _v12_champion_eligibility_gate(
    score: float,
    bd: Dict[str, Any],
    kw: Dict[str, Any],
) -> float:
    tier = str(kw.get("tier") or "").strip().lower()
    if tier not in {"investigation", "validation", "breakthrough"}:
        return score

    champion_gate_score = _apply_champion_tiny_model_hard_failure_gate(score, bd, kw)
    if champion_gate_score is not None:
        return champion_gate_score

    eff_ind = _v12_effective_signal(
        kw, "induction_intermediate_inv_auc", "induction_screening_auc"
    )
    eff_bind = _v12_effective_signal(
        kw, "binding_intermediate_inv_auc", "binding_screening_auc"
    )
    induction_qualified = eff_ind is not None and eff_ind >= _V12_INDUCTION_QUALIFIED
    strong_induction = eff_ind is not None and eff_ind >= _V12_STRONG_INDUCTION
    binding_qualified = eff_bind is not None and eff_bind >= _V12_BINDING_QUALIFIED
    strong_binding = eff_bind is not None and eff_bind >= _V12_STRONG_BINDING
    ar_validation_signal = _ar_validation_validation_signal(kw)
    ar_validation_qualified = (
        ar_validation_signal is not None
        and ar_validation_signal >= _V12_AR_VALIDATION_QUALIFIED
    )
    sequence_signal_count = _v12_non_loss_sequence_signal_count(kw)
    exception_allowed = (
        _v12_is_non_attention_exception_family(kw)
        and _v12_has_reproduced_bpe_loss(kw)
        and sequence_signal_count >= 2
    )

    bd["_v12_champion_induction_qualified"] = induction_qualified
    bd["_v12_champion_binding_qualified"] = binding_qualified
    bd["_v12_champion_exception_allowed"] = exception_allowed
    bd["_v12_champion_sequence_signal_count"] = sequence_signal_count
    bd["_v12_champion_strong_induction"] = strong_induction
    bd["_v12_champion_strong_binding"] = strong_binding
    bd["_v12_champion_ar_validation_qualified"] = ar_validation_qualified

    if score <= _V12_CHAMPION_ELIGIBILITY_CEILING or bool(kw.get("is_reference")):
        return score

    if induction_qualified or ar_validation_qualified or exception_allowed:
        return score

    bd["_v12_champion_eligibility_ceiling"] = _V12_CHAMPION_ELIGIBILITY_CEILING
    return _V12_CHAMPION_ELIGIBILITY_CEILING


def compute_composite_v12(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v12 dry-run: v11 plus loss-budget rebalance and champion gate."""
    result = compute_composite_v11(decompose=True, **kw)
    score = float(result["composite_score"])
    bd = result.get("breakdown") or {}

    loss_before = sum(float(bd.get(key, 0.0) or 0.0) for key in _LOSS_TIER_BD_KEYS)
    for key, factor in _V12_LOSS_COMPONENT_FACTORS.items():
        if key not in bd or not bd[key]:
            continue
        old_value = float(bd[key])
        new_value = old_value * factor
        bd[key] = new_value
        score += new_value - old_value
    loss_after = sum(float(bd.get(key, 0.0) or 0.0) for key in _LOSS_TIER_BD_KEYS)
    bd["_v12_loss_budget_before"] = loss_before
    bd["_v12_loss_budget_after"] = loss_after
    bd["_v12_loss_budget_max"] = _V12_LOSS_BUDGET_MAX
    if "_v10_base_v8style_total" in bd:
        bd["_v10_base_v8style_total"] = max(
            0.0,
            float(bd.get("_v10_base_v8style_total") or 0.0)
            + (loss_after - loss_before),
        )

    tier = str(kw.get("tier") or "").strip().lower()
    ar_validation_pts, ar_validation_bd = _score_ar_validation_validation_tier(
        _V14_CONFIG,
        inv_failed=tier in ("investigation_failed", "screened_out"),
        is_validation=tier in ("validation", "breakthrough")
        if tier
        else kw.get("ppl_validation") is not None,
        kw=kw,
    )
    score += ar_validation_pts
    bd.update(ar_validation_bd)

    score = _v12_champion_eligibility_gate(score, bd, kw)

    if decompose:
        result["composite_score"] = score
        result["breakdown"] = bd
        return result
    return score
