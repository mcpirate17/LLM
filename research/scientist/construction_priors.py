"""Multi-metric construction priors derived from causal ablation evidence.

The original `causal_generation_adjustments` weights rules by `effect_size`
and `confidence` from `causal_rule_evidence`, both of which collapse to loss.
This module computes priors from the per-metric child↔parent deltas in
program_results, so a rule that improves induction/binding/AR/BLiMP/HellaSwag
gets credited even when its loss delta is small or noisy.

Snapshots are versioned and immutable. Activating a new snapshot demotes the
previous active one. The grammar/screening path can call
`get_active_construction_prior(nb)` to read whatever's currently active and
bias generation accordingly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# Per-metric weight in the multi-metric verdict score. Sum to 1.0.
# Reasoning probes (induction/binding/ar) are weighted higher than scalar
# accuracy benchmarks because they measure mechanism, not just average ability.
METRIC_WEIGHTS: Dict[str, float] = {
    "induction_v2": 0.18,
    "binding_v2": 0.18,
    "induction": 0.12,
    "binding": 0.12,
    "ar": 0.08,
    "blimp": 0.10,
    "hellaswag": 0.08,
    "ppl_pct": 0.08,
    "loss": 0.06,
}

# Per-metric scale: a Δ of `scale` saturates the per-metric contribution.
# Tuned to typical observed magnitudes; protects against one outlier metric.
METRIC_SCALE: Dict[str, float] = {
    "induction_v2": 0.25,
    "binding_v2": 0.25,
    "induction": 0.20,
    "binding": 0.15,
    "ar": 0.10,
    "blimp": 0.05,
    "hellaswag": 0.05,
    "ppl_pct": 0.30,
    "loss": 0.20,
}

# Verdict thresholds on the composite score (range ≈ [-1, 1]).
USE_THRESHOLD = 0.10
AVOID_THRESHOLD = -0.10
DENYLIST_THRESHOLD = -0.30  # only confidently bad rules go to grammar denylist

# Multiplier clamp. The grammar applies these as op_weight multipliers.
MULTIPLIER_CLAMP = (0.4, 1.8)

# Activation filters keep noisy ablation findings as reviewable evidence
# without letting them steer graph generation globally.
DEFAULT_ACTIVATION_MIN_CONTEXTS = 3
DEFAULT_ACTIVATION_MAX_RISK_RATIO = 0.25
DEFAULT_ACTIVATION_MIN_WEIGHT_USED = 0.30


def _scale_metric(metric: str, value: Optional[float]) -> Optional[float]:
    """Map a raw Δ to [-1, 1] using the per-metric scale, or None if missing."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    scale = max(1e-6, METRIC_SCALE.get(metric, 0.1))
    if v >= 0:
        return min(v / scale, 1.0)
    return max(v / scale, -1.0)


def _composite_score(per_metric: Dict[str, Optional[float]]) -> Tuple[float, float]:
    """Return (score, total_weight_used). Score in [-1, 1]."""
    score_num = 0.0
    weight_total = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        scaled = _scale_metric(metric, per_metric.get(metric))
        if scaled is None:
            continue
        score_num += weight * scaled
        weight_total += weight
    if weight_total <= 0:
        return 0.0, 0.0
    return score_num / weight_total, weight_total


def _classify(score: float) -> str:
    if score >= USE_THRESHOLD:
        return "use"
    if score <= AVOID_THRESHOLD:
        return "avoid"
    return "mixed"


def assess_local_edit_prior(
    active_prior: Optional[Dict[str, Any]],
    *,
    rule_type: str,
    rule_key: str,
) -> Dict[str, Any]:
    """Return the active prior's verdict for one local edit, or neutral.

    Ablation drivers attach this to provenance so later analysis can tell
    whether an edit was encouraged, discouraged, or unevaluated. The helper is
    deliberately read-only and never activates priors.
    """
    if not active_prior:
        return {"verdict": "none", "multiplier": 1.0, "source": "no_active_prior"}
    payload = active_prior.get("payload") if isinstance(active_prior, dict) else None
    rules = active_prior.get("rules") or (
        payload.get("rules") if isinstance(payload, dict) else None
    )
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if rule.get("rule_type") == rule_type and rule.get("rule_key") == rule_key:
                return {
                    "verdict": str(rule.get("verdict") or "mixed"),
                    "multiplier": float(rule.get("multiplier") or 1.0),
                    "score": rule.get("score"),
                    "source": str(active_prior.get("snapshot_id") or "active_prior"),
                }
    return {"verdict": "none", "multiplier": 1.0, "source": "no_matching_rule"}


_CHILD_PARENT_DELTA_SQL = """
WITH metric_rows AS (
    SELECT obs.rule_type,
           obs.rule_key,
           obs.parent_result_id,
           'child_observation' AS evidence_source,
           COALESCE(cp.stage1_passed, obs.stage1_passed) AS child_stage1_passed,
           cp.loss_ratio AS child_loss,
           pp.loss_ratio AS parent_loss,
           cp.induction_auc AS child_induction,
           pp.induction_auc AS parent_induction,
           cp.binding_composite AS child_binding,
           pp.binding_composite AS parent_binding,
           cp.ar_auc AS child_ar,
           pp.ar_auc AS parent_ar,
           cp.hellaswag_acc AS child_hellaswag,
           pp.hellaswag_acc AS parent_hellaswag,
           cp.blimp_overall_accuracy AS child_blimp,
           pp.blimp_overall_accuracy AS parent_blimp,
           cp.wikitext_perplexity AS child_ppl,
           pp.wikitext_perplexity AS parent_ppl,
           cp.induction_v2_investigation_auc AS child_induction_v2,
           pp.induction_v2_investigation_auc AS parent_induction_v2,
           cp.induction_v2_investigation_status AS child_induction_v2_status,
           cp.binding_v2_investigation_auc AS child_binding_v2,
           pp.binding_v2_investigation_auc AS parent_binding_v2,
           cp.binding_v2_investigation_status AS child_binding_v2_status
    FROM causal_ablation_child_observations obs
    LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
    LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id

    UNION ALL

    SELECT ev.rule_type,
           ev.rule_key,
           ev.parent_result_id,
           CASE WHEN ev.rule_type = 'node_delete_investigation'
                THEN 'knockout_investigation' ELSE 'knockout_s1' END
                AS evidence_source,
           COALESCE(
               cp.stage1_passed,
               json_extract(ev.evidence_json, '$.child_stage1_passed')
           ) AS child_stage1_passed,
           COALESCE(
               cp.loss_ratio,
               json_extract(ev.evidence_json, '$.child_metrics.loss_ratio')
           ) AS child_loss,
           COALESCE(
               pp.loss_ratio,
               json_extract(ev.evidence_json, '$.parent_metrics.loss_ratio')
           ) AS parent_loss,
           COALESCE(
               cp.induction_auc,
               json_extract(ev.evidence_json, '$.child_metrics.induction_auc')
           ) AS child_induction,
           COALESCE(
               pp.induction_auc,
               json_extract(ev.evidence_json, '$.parent_metrics.induction_auc')
           ) AS parent_induction,
           COALESCE(
               cp.binding_composite,
               json_extract(ev.evidence_json, '$.child_metrics.binding_composite')
           ) AS child_binding,
           COALESCE(
               pp.binding_composite,
               json_extract(ev.evidence_json, '$.parent_metrics.binding_composite')
           ) AS parent_binding,
           COALESCE(
               cp.ar_auc,
               json_extract(ev.evidence_json, '$.child_metrics.ar_auc')
           ) AS child_ar,
           COALESCE(
               pp.ar_auc,
               json_extract(ev.evidence_json, '$.parent_metrics.ar_auc')
           ) AS parent_ar,
           COALESCE(
               cp.hellaswag_acc,
               json_extract(ev.evidence_json, '$.child_metrics.hellaswag_acc')
           ) AS child_hellaswag,
           COALESCE(
               pp.hellaswag_acc,
               json_extract(ev.evidence_json, '$.parent_metrics.hellaswag_acc')
           ) AS parent_hellaswag,
           COALESCE(
               cp.blimp_overall_accuracy,
               json_extract(ev.evidence_json, '$.child_metrics.blimp_overall_accuracy')
           ) AS child_blimp,
           COALESCE(
               pp.blimp_overall_accuracy,
               json_extract(ev.evidence_json, '$.parent_metrics.blimp_overall_accuracy')
           ) AS parent_blimp,
           COALESCE(
               cp.wikitext_perplexity,
               json_extract(ev.evidence_json, '$.child_metrics.wikitext_perplexity')
           ) AS child_ppl,
           COALESCE(
               pp.wikitext_perplexity,
               json_extract(ev.evidence_json, '$.parent_metrics.wikitext_perplexity')
           ) AS parent_ppl,
           COALESCE(
               cp.induction_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.induction_v2_investigation_auc'
               )
           ) AS child_induction_v2,
           COALESCE(
               pp.induction_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.parent_metrics.induction_v2_investigation_auc'
               )
           ) AS parent_induction_v2,
           COALESCE(
               cp.induction_v2_investigation_status,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.induction_v2_investigation_status'
               )
           ) AS child_induction_v2_status,
           COALESCE(
               cp.binding_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.binding_v2_investigation_auc'
               )
           ) AS child_binding_v2,
           COALESCE(
               pp.binding_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.parent_metrics.binding_v2_investigation_auc'
               )
           ) AS parent_binding_v2,
           COALESCE(
               cp.binding_v2_investigation_status,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.binding_v2_investigation_status'
               )
           ) AS child_binding_v2_status
    FROM causal_rule_evidence ev
    LEFT JOIN program_results cp
      ON cp.result_id = json_extract(ev.evidence_json, '$.child_result_id')
    LEFT JOIN program_results pp ON pp.result_id = ev.parent_result_id
    WHERE ev.rule_type IN ('node_delete_s1', 'node_delete_investigation')
      AND json_extract(ev.evidence_json, '$.child.fingerprint') IS NOT NULL
),
clean_rows AS (
    SELECT *,
           CASE WHEN COALESCE(child_stage1_passed, 0) = 1
                     AND LOWER(COALESCE(child_induction_v2_status, 'ok'))
                         NOT IN ('diverged', 'failed', 'error')
                     AND LOWER(COALESCE(child_binding_v2_status, 'ok'))
                         NOT IN ('diverged', 'failed', 'error')
                THEN 1 ELSE 0 END AS clean_comparable,
           CASE WHEN COALESCE(child_stage1_passed, 0) = 0
                     OR LOWER(COALESCE(child_induction_v2_status, 'ok'))
                         IN ('diverged', 'failed', 'error')
                     OR LOWER(COALESCE(child_binding_v2_status, 'ok'))
                         IN ('diverged', 'failed', 'error')
                THEN 1 ELSE 0 END AS risk_row
    FROM metric_rows
),
deltas AS (
    SELECT rule_type,
           rule_key,
           parent_result_id,
           evidence_source,
           risk_row,
           CASE WHEN clean_comparable = 1
                     AND child_loss IS NOT NULL AND parent_loss IS NOT NULL
                THEN child_loss - parent_loss END AS d_loss,
           CASE WHEN clean_comparable = 1
                     AND child_induction_v2 IS NOT NULL
                     AND parent_induction_v2 IS NOT NULL
                THEN parent_induction_v2 - child_induction_v2
           END AS d_induction_v2,
           CASE WHEN clean_comparable = 1
                     AND child_binding_v2 IS NOT NULL
                     AND parent_binding_v2 IS NOT NULL
                THEN parent_binding_v2 - child_binding_v2
           END AS d_binding_v2,
           CASE WHEN clean_comparable = 1
                     AND child_induction IS NOT NULL AND parent_induction IS NOT NULL
                THEN parent_induction - child_induction END AS d_induction,
           CASE WHEN clean_comparable = 1
                     AND child_binding IS NOT NULL AND parent_binding IS NOT NULL
                THEN parent_binding - child_binding END AS d_binding,
           CASE WHEN clean_comparable = 1
                     AND child_ar IS NOT NULL AND parent_ar IS NOT NULL
                THEN parent_ar - child_ar END AS d_ar,
           CASE WHEN clean_comparable = 1
                     AND child_hellaswag IS NOT NULL
                     AND parent_hellaswag IS NOT NULL
                THEN parent_hellaswag - child_hellaswag END AS d_hellaswag,
           CASE WHEN clean_comparable = 1
                     AND child_blimp IS NOT NULL AND parent_blimp IS NOT NULL
                THEN parent_blimp - child_blimp END AS d_blimp,
           CASE WHEN clean_comparable = 1
                     AND child_ppl IS NOT NULL
                     AND parent_ppl IS NOT NULL
                     AND parent_ppl > 0
                THEN (child_ppl - parent_ppl) / parent_ppl END AS d_ppl_pct,
           CASE WHEN clean_comparable = 1
                     AND child_hellaswag IS NOT NULL
                     AND child_blimp IS NOT NULL
                     AND child_induction IS NOT NULL
                     AND child_binding IS NOT NULL
                     AND child_ar IS NOT NULL
                     AND child_ppl IS NOT NULL
                THEN 1 ELSE 0 END AS metric_complete,
           CASE WHEN clean_comparable = 1
                     AND (child_induction_v2 IS NOT NULL
                          OR child_binding_v2 IS NOT NULL)
                THEN 1 ELSE 0 END AS v2_observation,
           CASE WHEN evidence_source LIKE 'knockout_%' THEN 1 ELSE 0 END
                AS knockout_observation
    FROM clean_rows
)
SELECT rule_type, rule_key,
       COUNT(*) AS n,
       COUNT(DISTINCT parent_result_id) AS contexts,
       SUM(metric_complete) AS metric_complete_count,
       SUM(v2_observation) AS v2_observation_count,
       SUM(knockout_observation) AS knockout_observation_count,
       SUM(risk_row) AS risk_row_count,
       AVG(d_loss) AS avg_d_loss,
       AVG(d_induction_v2) AS avg_d_induction_v2,
       AVG(d_binding_v2) AS avg_d_binding_v2,
       AVG(d_induction) AS avg_d_induction,
       AVG(d_binding) AS avg_d_binding,
       AVG(d_ar) AS avg_d_ar,
       AVG(d_hellaswag) AS avg_d_hellaswag,
       AVG(d_blimp) AS avg_d_blimp,
       AVG(d_ppl_pct) AS avg_d_ppl_pct
FROM deltas
GROUP BY rule_type, rule_key
HAVING COUNT(*) >= ? AND (
    SUM(metric_complete) >= ? OR SUM(v2_observation) >= ?
    OR SUM(risk_row) >= ?
)
ORDER BY n DESC
LIMIT ?
"""


_OP_PAIR_SCAFFOLD_OPS = frozenset(
    {
        "add",
        "mul",
        "linear_proj",
        "rmsnorm",
        "layernorm",
        "identity",
        "gelu",
        "silu",
    }
)

_CONTEXT_SENSITIVE_NODE_DELETE_OPS = frozenset({"rmsnorm", "layernorm"})


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _node_delete_op_name(rule_key: str) -> str:
    raw = str(rule_key or "")
    return raw.split(":", 1)[1] if ":" in raw else raw


def _risk_ratio(rule: Dict[str, Any]) -> float:
    n = max(0, int(rule.get("n") or 0))
    if n <= 0:
        return 0.0
    return max(0.0, min(1.0, float(rule.get("risk_row_count") or 0) / float(n)))


def _rule_filter_reasons(
    rule: Dict[str, Any],
    *,
    min_contexts: int = DEFAULT_ACTIVATION_MIN_CONTEXTS,
    max_risk_ratio: float = DEFAULT_ACTIVATION_MAX_RISK_RATIO,
    min_weight_used: float = DEFAULT_ACTIVATION_MIN_WEIGHT_USED,
) -> List[str]:
    reasons: List[str] = []
    verdict = str(rule.get("verdict") or "mixed")
    if verdict == "mixed":
        reasons.append("mixed_verdict")
    if float(rule.get("weight_used") or 0.0) < float(min_weight_used):
        reasons.append("low_metric_weight")
    if int(rule.get("contexts") or 0) < int(min_contexts):
        reasons.append("low_context_count")
    if _risk_ratio(rule) > float(max_risk_ratio):
        reasons.append("high_risk_ratio")
    rule_type = str(rule.get("rule_type") or "")
    if rule_type in {"node_delete", "node_delete_s1", "node_delete_investigation"}:
        op_name = _node_delete_op_name(str(rule.get("rule_key") or ""))
        if op_name in _CONTEXT_SENSITIVE_NODE_DELETE_OPS:
            reasons.append("context_sensitive_node_delete")
    return reasons


def audit_construction_prior_payload(
    prior_or_payload: Optional[Dict[str, Any]],
    *,
    min_contexts: int = DEFAULT_ACTIVATION_MIN_CONTEXTS,
    max_risk_ratio: float = DEFAULT_ACTIVATION_MAX_RISK_RATIO,
    min_weight_used: float = DEFAULT_ACTIVATION_MIN_WEIGHT_USED,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Return a risk-aware report for a construction-prior payload.

    The report is advisory: it explains which rules are clean enough for
    generation-wide activation and which should remain review-only hints.
    """
    if not prior_or_payload:
        return {
            "eligible_rules": 0,
            "blocked_rules": 0,
            "reason_counts": {},
            "top_use": [],
            "top_avoid": [],
            "top_risky": [],
            "top_blocked": [],
        }
    payload = (
        prior_or_payload.get("payload")
        if isinstance(prior_or_payload.get("payload"), dict)
        else prior_or_payload
    )
    rules = [r for r in (payload.get("rules") or []) if isinstance(r, dict)]
    reason_counts: Dict[str, int] = {}
    eligible: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for rule in rules:
        reasons = _rule_filter_reasons(
            rule,
            min_contexts=min_contexts,
            max_risk_ratio=max_risk_ratio,
            min_weight_used=min_weight_used,
        )
        annotated = dict(rule)
        annotated["risk_ratio"] = round(_risk_ratio(rule), 4)
        annotated["activation_filter_reasons"] = reasons
        if reasons:
            blocked.append(annotated)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            eligible.append(annotated)

    def _trim(rule: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "rule_type",
            "rule_key",
            "verdict",
            "score",
            "n",
            "contexts",
            "metric_complete_count",
            "v2_observation_count",
            "risk_row_count",
            "risk_ratio",
            "weight_used",
            "multiplier",
            "activation_filter_reasons",
        )
        return {k: rule.get(k) for k in keys if k in rule}

    top_use = sorted(
        [r for r in eligible if r.get("verdict") == "use"],
        key=lambda r: (float(r.get("score") or 0.0), int(r.get("contexts") or 0)),
        reverse=True,
    )
    top_avoid = sorted(
        [r for r in eligible if r.get("verdict") == "avoid"],
        key=lambda r: (float(r.get("score") or 0.0), -int(r.get("contexts") or 0)),
    )
    top_risky = sorted(
        [dict(r, risk_ratio=round(_risk_ratio(r), 4)) for r in rules],
        key=lambda r: (
            float(r.get("risk_ratio") or 0.0),
            int(r.get("risk_row_count") or 0),
            abs(float(r.get("score") or 0.0)),
        ),
        reverse=True,
    )
    top_blocked = sorted(
        blocked,
        key=lambda r: (
            len(r.get("activation_filter_reasons") or []),
            abs(float(r.get("score") or 0.0)),
            int(r.get("n") or 0),
        ),
        reverse=True,
    )
    return {
        "version": payload.get("version"),
        "thresholds": {
            "min_contexts": int(min_contexts),
            "max_risk_ratio": float(max_risk_ratio),
            "min_weight_used": float(min_weight_used),
        },
        "total_rules": len(rules),
        "eligible_rules": len(eligible),
        "blocked_rules": len(blocked),
        "reason_counts": dict(sorted(reason_counts.items())),
        "top_use": [_trim(r) for r in top_use[: int(top_n)]],
        "top_avoid": [_trim(r) for r in top_avoid[: int(top_n)]],
        "top_risky": [_trim(r) for r in top_risky[: int(top_n)]],
        "top_blocked": [_trim(r) for r in top_blocked[: int(top_n)]],
    }


def filter_construction_prior_payload_for_activation(
    prior_or_payload: Dict[str, Any],
    *,
    min_contexts: int = DEFAULT_ACTIVATION_MIN_CONTEXTS,
    max_risk_ratio: float = DEFAULT_ACTIVATION_MAX_RISK_RATIO,
    min_weight_used: float = DEFAULT_ACTIVATION_MIN_WEIGHT_USED,
) -> Dict[str, Any]:
    """Return a payload whose grammar adjustments use only eligible rules.

    Blocked rules are preserved under ``candidate_hints`` for analysis, but
    they do not contribute to op weights, slot multipliers, or denylists.
    """
    payload = (
        prior_or_payload.get("payload")
        if isinstance(prior_or_payload.get("payload"), dict)
        else prior_or_payload
    )
    filtered_payload = dict(payload)
    rules = [r for r in (payload.get("rules") or []) if isinstance(r, dict)]
    eligible_rules: List[Dict[str, Any]] = []
    blocked_rules: List[Dict[str, Any]] = []
    op_weights: Dict[str, float] = {}
    slot_motif_multipliers: Dict[str, Dict[str, float]] = {}
    slot_motif_denylist: Dict[str, set] = {}
    for rule in rules:
        reasons = _rule_filter_reasons(
            rule,
            min_contexts=min_contexts,
            max_risk_ratio=max_risk_ratio,
            min_weight_used=min_weight_used,
        )
        annotated = dict(rule)
        annotated["risk_ratio"] = round(_risk_ratio(rule), 4)
        annotated["activation_filter_reasons"] = reasons
        if reasons:
            blocked_rules.append(annotated)
            continue
        eligible_rules.append(annotated)
        _accumulate_grammar_adjustments(
            annotated,
            op_weights=op_weights,
            slot_motif_multipliers=slot_motif_multipliers,
            slot_motif_denylist=slot_motif_denylist,
        )

    source_counts = dict(payload.get("source_counts") or {})
    source_counts.update(
        {
            "activation_eligible_rules": len(eligible_rules),
            "activation_blocked_rules": len(blocked_rules),
        }
    )
    filtered_payload.update(
        {
            "rules": eligible_rules,
            "op_weights": op_weights,
            "slot_motif_multipliers": slot_motif_multipliers,
            "slot_motif_denylist": {
                k: sorted(v) for k, v in slot_motif_denylist.items()
            },
            "candidate_hints": blocked_rules,
            "source_counts": source_counts,
            "activation_filter": {
                "min_contexts": int(min_contexts),
                "max_risk_ratio": float(max_risk_ratio),
                "min_weight_used": float(min_weight_used),
                "eligible_rules": len(eligible_rules),
                "blocked_rules": len(blocked_rules),
            },
        }
    )
    return filtered_payload


def _classify_row(row: Any) -> Dict[str, Any]:
    """Score one aggregated row and return its rule dict (verdict, multiplier, per-metric)."""
    per_metric = {
        "induction_v2": _row_value(row, "avg_d_induction_v2"),
        "binding_v2": _row_value(row, "avg_d_binding_v2"),
        "induction": _row_value(row, "avg_d_induction"),
        "binding": _row_value(row, "avg_d_binding"),
        "ar": _row_value(row, "avg_d_ar"),
        "blimp": _row_value(row, "avg_d_blimp"),
        "hellaswag": _row_value(row, "avg_d_hellaswag"),
        "ppl_pct": _row_value(row, "avg_d_ppl_pct"),
        "loss": _row_value(row, "avg_d_loss"),
    }
    score, weight_used = _composite_score(per_metric)
    verdict = _classify(score) if weight_used >= 0.30 else "mixed"
    raw_mult = 1.0 + score
    multiplier = max(MULTIPLIER_CLAMP[0], min(MULTIPLIER_CLAMP[1], raw_mult))
    return {
        "rule_type": row["rule_type"],
        "rule_key": row["rule_key"],
        "verdict": verdict,
        "n": int(row["n"] or 0),
        "contexts": int(row["contexts"] or 0),
        "metric_complete_count": int(row["metric_complete_count"] or 0),
        "v2_observation_count": int(_row_value(row, "v2_observation_count", 0) or 0),
        "knockout_observation_count": int(
            _row_value(row, "knockout_observation_count", 0) or 0
        ),
        "risk_row_count": int(_row_value(row, "risk_row_count", 0) or 0),
        "score": round(score, 4),
        "weight_used": round(weight_used, 3),
        "multiplier": round(multiplier, 3),
        "per_metric": {
            k: (None if v is None else round(float(v), 4))
            for k, v in per_metric.items()
        },
    }


def _accumulate_grammar_adjustments(
    rule: Dict[str, Any],
    *,
    op_weights: Dict[str, float],
    slot_motif_multipliers: Dict[str, Dict[str, float]],
    slot_motif_denylist: Dict[str, set],
) -> None:
    """Fold one classified rule into the grammar adjustment maps. In-place."""
    verdict = rule["verdict"]
    if verdict == "mixed":
        return
    multiplier = rule["multiplier"]
    rule_type = rule["rule_type"]
    rule_key = rule["rule_key"]
    if rule_type == "op":
        cur = op_weights.get(rule_key, 1.0)
        op_weights[rule_key] = (
            max(cur, multiplier) if verdict == "use" else min(cur, multiplier)
        )
    elif rule_type == "op_pair":
        half = multiplier**0.5
        for op_name in str(rule_key).split("->", 1):
            if op_name and op_name not in _OP_PAIR_SCAFFOLD_OPS:
                cur = op_weights.get(op_name, 1.0)
                op_weights[op_name] = (
                    max(cur, half) if verdict == "use" else min(cur, half)
                )
    elif rule_type == "slot_motif" and ":" in rule_key:
        slot_key, motif_name = rule_key.rsplit(":", 1)
        if not slot_key or not motif_name:
            return
        if verdict == "avoid" and rule["score"] <= DENYLIST_THRESHOLD:
            slot_motif_denylist.setdefault(slot_key, set()).add(motif_name)
        else:
            slot_motif_multipliers.setdefault(slot_key, {})
            cur = float(slot_motif_multipliers[slot_key].get(motif_name, 1.0))
            slot_motif_multipliers[slot_key][motif_name] = (
                max(cur, multiplier) if verdict == "use" else min(cur, multiplier)
            )
    elif rule_type in {"node_delete", "node_delete_s1", "node_delete_investigation"}:
        op_name = _node_delete_op_name(rule_key)
        if not op_name or op_name in _CONTEXT_SENSITIVE_NODE_DELETE_OPS:
            return
        cur = op_weights.get(op_name, 1.0)
        op_weights[op_name] = (
            max(cur, multiplier) if verdict == "use" else min(cur, multiplier)
        )


def compute_construction_prior(
    nb: Any,
    *,
    min_n: int = 4,
    min_metric_complete: int = 3,
    local_min_n: int = 4,
    limit: int = 600,
    local_limit: int = 5000,
) -> Dict[str, Any]:
    """Compute a fresh construction prior from current evidence.

    Returns a dict ready to be persisted via `record_construction_prior_snapshot`.
    Caller decides whether to activate it.
    """
    min_count = max(1, min(int(min_n), int(local_min_n)))
    rows = nb.conn.execute(
        _CHILD_PARENT_DELTA_SQL,
        (
            min_count,
            int(min_metric_complete),
            int(min_metric_complete),
            min_count,
            max(int(limit), int(local_limit)),
        ),
    ).fetchall()

    classified_rules: List[Dict[str, Any]] = []
    op_weights: Dict[str, float] = {}
    slot_motif_multipliers: Dict[str, Dict[str, float]] = {}
    slot_motif_denylist: Dict[str, set] = {}
    counts = {"use": 0, "avoid": 0, "mixed": 0}
    local_edit_observations = 0
    v2_observations = 0
    risk_rows = 0

    for row in rows:
        rule_type = str(row["rule_type"] or "")
        min_required = (
            int(local_min_n) if rule_type.startswith("node_delete") else int(min_n)
        )
        if int(row["n"] or 0) < max(1, min_required):
            continue
        rule = _classify_row(row)
        counts[rule["verdict"]] = counts.get(rule["verdict"], 0) + 1
        classified_rules.append(rule)
        local_edit_observations += int(rule.get("knockout_observation_count") or 0)
        v2_observations += int(rule.get("v2_observation_count") or 0)
        risk_rows += int(rule.get("risk_row_count") or 0)
        _accumulate_grammar_adjustments(
            rule,
            op_weights=op_weights,
            slot_motif_multipliers=slot_motif_multipliers,
            slot_motif_denylist=slot_motif_denylist,
        )

    version = f"v{time.strftime('%Y%m%d-%H%M%S')}"
    payload = {
        "version": version,
        "computed_at": time.time(),
        "rules": classified_rules,
        "op_weights": op_weights,
        "slot_motif_multipliers": slot_motif_multipliers,
        "slot_motif_denylist": {k: sorted(v) for k, v in slot_motif_denylist.items()},
        "thresholds": {
            "use": USE_THRESHOLD,
            "avoid": AVOID_THRESHOLD,
            "denylist": DENYLIST_THRESHOLD,
            "multiplier_clamp": list(MULTIPLIER_CLAMP),
        },
        "metric_weights": METRIC_WEIGHTS,
        "source_counts": {
            "local_edit_observations": local_edit_observations,
            "v2_observations": v2_observations,
            "risk_rows": risk_rows,
        },
    }
    summary = {
        "version": version,
        "n_rules": len(classified_rules),
        "n_use": counts["use"],
        "n_avoid": counts["avoid"],
        "n_mixed": counts["mixed"],
        "n_local_edit_observations": local_edit_observations,
        "n_v2_observations": v2_observations,
        "n_risk_rows": risk_rows,
        "n_op_weights": len(op_weights),
        "n_slot_motif_multipliers": sum(
            len(v) for v in slot_motif_multipliers.values()
        ),
        "n_slot_motif_denylist": sum(len(v) for v in slot_motif_denylist.values()),
    }
    return {"payload": payload, "summary": summary}


def record_construction_prior_snapshot(
    nb: Any,
    prior: Dict[str, Any],
    *,
    activate: bool = True,
    notes: str = "",
) -> str:
    """Persist a new snapshot. If activate=True, mark as active and demote prior."""
    payload = prior["payload"]
    summary = prior.get("summary", {})
    version = str(payload.get("version") or f"v{int(time.time())}")
    if activate:
        nb.conn.execute(
            "UPDATE construction_prior_snapshots SET is_active = 0 WHERE is_active = 1"
        )
    nb.conn.execute(
        """
        INSERT INTO construction_prior_snapshots
            (version, created_at, is_active, payload_json, summary_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            float(payload.get("computed_at") or time.time()),
            1 if activate else 0,
            json.dumps(payload, sort_keys=True),
            json.dumps(summary, sort_keys=True),
            notes,
        ),
    )
    nb.conn.commit()
    return version


def get_active_construction_prior(nb: Any) -> Optional[Dict[str, Any]]:
    """Return the currently active snapshot payload, or None."""
    row = nb.conn.execute(
        """
        SELECT version, created_at, payload_json, summary_json, notes
        FROM construction_prior_snapshots
        WHERE is_active = 1
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    summary = {}
    try:
        summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
    except (json.JSONDecodeError, TypeError):
        summary = {}
    return {
        "version": row["version"],
        "created_at": row["created_at"],
        "payload": payload,
        "summary": summary,
        "notes": row["notes"] or "",
    }


def list_construction_prior_snapshots(nb: Any, limit: int = 20) -> List[Dict[str, Any]]:
    rows = nb.conn.execute(
        """
        SELECT id, version, created_at, is_active, summary_json, notes
        FROM construction_prior_snapshots
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        summary = {}
        try:
            summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
        except (json.JSONDecodeError, TypeError):
            summary = {}
        out.append(
            {
                "id": row["id"],
                "version": row["version"],
                "created_at": row["created_at"],
                "is_active": bool(row["is_active"]),
                "summary": summary,
                "notes": row["notes"] or "",
            }
        )
    return out


def construction_prior_as_grammar_adjustments(
    prior: Optional[Dict[str, Any]],
    *,
    apply_activation_filter: bool = True,
) -> Dict[str, Any]:
    """Convert an active prior payload to the dict shape consumed by the
    screening grammar feedback loop (matches `causal_generation_adjustments`).
    """
    if not prior:
        return {
            "op_weights": {},
            "slot_motif_multipliers": {},
            "slot_motif_denylist": {},
            "version": None,
        }
    payload = prior.get("payload") or {}
    if apply_activation_filter:
        payload = filter_construction_prior_payload_for_activation(payload)
    return {
        "op_weights": dict(payload.get("op_weights") or {}),
        "slot_motif_multipliers": {
            k: dict(v) for k, v in (payload.get("slot_motif_multipliers") or {}).items()
        },
        "slot_motif_denylist": {
            k: frozenset(v)
            for k, v in (payload.get("slot_motif_denylist") or {}).items()
        },
        "version": payload.get("version"),
    }
