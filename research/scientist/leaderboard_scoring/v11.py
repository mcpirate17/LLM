"""v11 — Breakthrough alignment (2026-04-26).

Structural changes from v10:
  1. High-ceiling capability anchors: cap_binding_anchor (0.004 -> 0.500)
     and cap_induction_anchor (0.006 -> 0.300).
  2. Higher capability weights: cap_binding and cap_induction (25 -> 50).
  3. Breakthrough multiplier: 1.2x boost to Understanding tier if
     binding_screening_auc > 0.8 AND induction_screening_auc > 0.3.
  4. Tokenizer Integrity: 0.1x total multiplier if tokenizer_mode != 'tiktoken'.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ._config import (
    _UND_TIER_BD_KEYS,
    _V10_CONFIG,
    _V11_CONFIG,
    _V11_TRUST_BINDING_FLOOR,
    _V11_TRUST_BLIMP_FLOOR,
    _V11_TRUST_CEILING,
    _V11_TRUST_HELLASWAG_FLOOR,
    _V11_TRUST_INDUCTION_FLOOR,
    _V11_TRUST_PPL_FLOOR,
)
from .champion_tiny import _apply_champion_tiny_model_hard_failure_gate
from .v10 import compute_composite_v10


_V11_CAP_INDUCTION_MULTIPLIER = (
    _V11_CONFIG["w_cap_induction"] / _V10_CONFIG["w_cap_induction"]
)
_V11_CAP_BINDING_MULTIPLIER = (
    _V11_CONFIG["w_cap_binding"] / _V10_CONFIG["w_cap_binding"]
)


# v11 tokenizer-integrity penalty (graded).  ``screening_wikitext_metric_version``
# is the reliable signal that PPL is in BPE units; ``tokenizer_mode`` is
# unreliable (was set to 'tiktoken' on some byte-era rows AND missing on
# many legitimate ones).
#
#   * 'bpe_eval_v1'         → 1.00  (good, full credit)
#   * 'screening_wikitext_v1' → 0.10  (definitively byte-era, hammer)
#   * NULL / empty / other  → 0.70  (unknown — soft uncertainty discount)
_V11_TOKENIZER_PENALTY_BPE = 1.0
_V11_TOKENIZER_PENALTY_BYTE = 0.1
_V11_TOKENIZER_PENALTY_UNKNOWN = 0.7


def _v11_trust_ceiling(
    score: float,
    bd: Dict[str, Any],
    kw: Dict[str, Any],
) -> float:
    """Cap untrusted validation-tier candidates below champion range."""
    if bool(kw.get("is_reference")):
        return score

    champion_gate_score = _apply_champion_tiny_model_hard_failure_gate(score, bd, kw)
    if champion_gate_score is not None:
        return champion_gate_score

    if score <= _V11_TRUST_CEILING:
        return score

    tier = str(kw.get("tier") or "").strip().lower()
    if tier not in {"investigation", "validation", "breakthrough"}:
        return score

    has_reproduced_low_loss = any(
        p is not None and float(p) > 0.0 and float(p) <= _V11_TRUST_PPL_FLOOR
        for p in (
            kw.get("ppl_validation"),
            kw.get("ppl_investigation"),
            kw.get("ppl_screening"),
        )
    )
    has_understanding = any(
        v is not None and float(v) > _V11_TRUST_HELLASWAG_FLOOR
        for v in (
            kw.get("hellaswag_acc_validation"),
            kw.get("hellaswag_acc_investigation"),
            kw.get("hellaswag_acc_screening"),
        )
    )
    blimp = kw.get("blimp_accuracy")
    if blimp is not None and float(blimp) >= _V11_TRUST_BLIMP_FLOOR:
        has_understanding = True

    eff_ind = (
        kw.get("induction_intermediate_inv_auc")
        if kw.get("induction_intermediate_inv_auc") is not None
        else kw.get("induction_screening_auc")
    )
    eff_bind = (
        kw.get("binding_intermediate_inv_auc")
        if kw.get("binding_intermediate_inv_auc") is not None
        else kw.get("binding_screening_auc")
    )
    has_nonlocal_binding = (
        eff_ind is not None
        and eff_bind is not None
        and float(eff_ind) >= _V11_TRUST_INDUCTION_FLOOR
        and float(eff_bind) >= _V11_TRUST_BINDING_FLOOR
    )

    if has_reproduced_low_loss or has_understanding or has_nonlocal_binding:
        return score

    bd["_v11_trust_ceiling"] = _V11_TRUST_CEILING
    bd["_v11_trust_low_loss"] = has_reproduced_low_loss
    bd["_v11_trust_understanding"] = has_understanding
    bd["_v11_trust_nonlocal_binding"] = has_nonlocal_binding
    return _V11_TRUST_CEILING


def _v11_tokenizer_penalty(metric_version: Optional[str]) -> float:
    if metric_version is None:
        return _V11_TOKENIZER_PENALTY_UNKNOWN
    mv = str(metric_version).strip().lower()
    if mv == "bpe_eval_v1":
        return _V11_TOKENIZER_PENALTY_BPE
    if mv == "screening_wikitext_v1":
        return _V11_TOKENIZER_PENALTY_BYTE
    return _V11_TOKENIZER_PENALTY_UNKNOWN


def compute_composite_v11(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v11 — Breakthrough-first alignment.

    Builds on v10 with high-ceiling capability weights for binding /
    induction, a breakthrough multiplier for logic-probes, and a graded
    tokenizer-integrity penalty.

    Implementation note: ``compute_composite_v10`` hardcodes
    ``_V10_CONFIG`` internally, so ``_V11_CONFIG``'s cap-tier weight
    bumps cannot take effect through the v10 call directly.  We
    recompose them here by multiplying ``cap_induction`` /
    ``cap_binding`` in the breakdown by the v11/v10 weight ratios.
    Anchors are NOT rescaled here — that would require re-evaluating
    ``_score_capability_tier_v10``, and the larger weights already
    deliver the intended "frontier-archs win bigger" effect on top of
    v10's anchors.
    """
    # Use v10 as the structural base.  Inside v10 the CV penalty (if any)
    # has already been applied to loss/und/cap subscores per tier.
    result = compute_composite_v10(decompose=True, **kw)
    score = float(result["composite_score"])
    bd = result.get("breakdown") or {}

    # 1. v11 capability rescale: lift cap_induction / cap_binding to the
    #    v11 weights (50pts each) by multiplying the v10 contribution.
    cap_ind_old = float(bd.get("cap_induction", 0.0) or 0.0)
    cap_bind_old = float(bd.get("cap_binding", 0.0) or 0.0)
    cap_ind_new = cap_ind_old * _V11_CAP_INDUCTION_MULTIPLIER
    cap_bind_new = cap_bind_old * _V11_CAP_BINDING_MULTIPLIER
    score += (cap_ind_new - cap_ind_old) + (cap_bind_new - cap_bind_old)
    bd["cap_induction"] = cap_ind_new
    bd["cap_binding"] = cap_bind_new

    # 2. Breakthrough multiplier — 1.2× understanding tier when both
    #    induction and binding clear their gates.  Uses v2 probes when
    #    populated, falls back to v1.
    eff_ind = (
        kw.get("induction_intermediate_inv_auc")
        if kw.get("induction_intermediate_inv_auc") is not None
        else kw.get("induction_screening_auc")
    )
    eff_bind = (
        kw.get("binding_intermediate_inv_auc")
        if kw.get("binding_intermediate_inv_auc") is not None
        else kw.get("binding_screening_auc")
    )
    is_breakthrough = (
        eff_ind is not None
        and eff_ind > 0.3
        and eff_bind is not None
        and eff_bind > 0.8
    )
    if is_breakthrough:
        und_sum = sum(float(bd.get(k, 0.0) or 0.0) for k in _UND_TIER_BD_KEYS)
        boost = und_sum * 0.2
        score += boost
        bd["_v11_breakthrough_boost"] = boost
        for k in _UND_TIER_BD_KEYS:
            if k in bd and bd[k]:
                bd[k] = float(bd[k]) * 1.2

    # 3. Tokenizer integrity penalty (graded by metric_version).
    metric_version = kw.get("screening_wikitext_metric_version")
    tok_pen = _v11_tokenizer_penalty(metric_version)
    if tok_pen < 1.0:
        score *= tok_pen
        bd["_v11_tokenizer_penalty"] = tok_pen
        bd["_v11_tokenizer_penalty_metric_version"] = metric_version

    # 4. Trust ceiling — a candidate cannot be a champion on efficiency and
    # median-relative side metrics alone.
    score = _v11_trust_ceiling(score, bd, kw)

    if decompose:
        result["composite_score"] = score
        result["breakdown"] = bd
        return result
    return score
