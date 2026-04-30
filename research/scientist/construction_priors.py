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
    "induction": 0.20,
    "binding": 0.20,
    "ar": 0.10,
    "blimp": 0.15,
    "hellaswag": 0.10,
    "ppl_pct": 0.15,
    "loss": 0.10,
}

# Per-metric scale: a Δ of `scale` saturates the per-metric contribution.
# Tuned to typical observed magnitudes; protects against one outlier metric.
METRIC_SCALE: Dict[str, float] = {
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


_CHILD_PARENT_DELTA_SQL = """
WITH metric_rows AS (
    SELECT obs.rule_type,
           obs.rule_key,
           obs.parent_result_id,
           CASE WHEN cp.loss_ratio IS NOT NULL AND pp.loss_ratio IS NOT NULL
                THEN cp.loss_ratio - pp.loss_ratio END AS d_loss,
           CASE WHEN cp.induction_auc IS NOT NULL AND pp.induction_auc IS NOT NULL
                THEN pp.induction_auc - cp.induction_auc END AS d_induction,
           CASE WHEN cp.binding_composite IS NOT NULL AND pp.binding_composite IS NOT NULL
                THEN pp.binding_composite - cp.binding_composite END AS d_binding,
           CASE WHEN cp.ar_auc IS NOT NULL AND pp.ar_auc IS NOT NULL
                THEN pp.ar_auc - cp.ar_auc END AS d_ar,
           CASE WHEN cp.hellaswag_acc IS NOT NULL AND pp.hellaswag_acc IS NOT NULL
                THEN pp.hellaswag_acc - cp.hellaswag_acc END AS d_hellaswag,
           CASE WHEN cp.blimp_overall_accuracy IS NOT NULL
                     AND pp.blimp_overall_accuracy IS NOT NULL
                THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy END AS d_blimp,
           CASE WHEN cp.wikitext_perplexity IS NOT NULL
                     AND pp.wikitext_perplexity IS NOT NULL
                     AND pp.wikitext_perplexity > 0
                THEN (cp.wikitext_perplexity - pp.wikitext_perplexity)
                     / pp.wikitext_perplexity END AS d_ppl_pct,
           CASE WHEN cp.hellaswag_acc IS NOT NULL
                     AND cp.blimp_overall_accuracy IS NOT NULL
                     AND cp.induction_auc IS NOT NULL
                     AND cp.binding_composite IS NOT NULL
                     AND cp.ar_auc IS NOT NULL
                     AND cp.wikitext_perplexity IS NOT NULL
                THEN 1 ELSE 0 END AS metric_complete
    FROM causal_ablation_child_observations obs
    LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
    LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
)
SELECT rule_type, rule_key,
       COUNT(*) AS n,
       COUNT(DISTINCT parent_result_id) AS contexts,
       SUM(metric_complete) AS metric_complete_count,
       AVG(d_loss) AS avg_d_loss,
       AVG(d_induction) AS avg_d_induction,
       AVG(d_binding) AS avg_d_binding,
       AVG(d_ar) AS avg_d_ar,
       AVG(d_hellaswag) AS avg_d_hellaswag,
       AVG(d_blimp) AS avg_d_blimp,
       AVG(d_ppl_pct) AS avg_d_ppl_pct
FROM metric_rows
GROUP BY rule_type, rule_key
HAVING COUNT(*) >= ? AND SUM(metric_complete) >= ?
ORDER BY n DESC
LIMIT ?
"""


_OP_PAIR_SCAFFOLD_OPS = frozenset({
    "add", "mul", "linear_proj", "rmsnorm",
    "layernorm", "identity", "gelu", "silu",
})


def _classify_row(row: Any) -> Dict[str, Any]:
    """Score one aggregated row and return its rule dict (verdict, multiplier, per-metric)."""
    per_metric = {
        "induction": row["avg_d_induction"],
        "binding":   row["avg_d_binding"],
        "ar":        row["avg_d_ar"],
        "blimp":     row["avg_d_blimp"],
        "hellaswag": row["avg_d_hellaswag"],
        "ppl_pct":   row["avg_d_ppl_pct"],
        "loss":      row["avg_d_loss"],
    }
    score, weight_used = _composite_score(per_metric)
    verdict = _classify(score) if weight_used >= 0.30 else "mixed"
    raw_mult = 1.0 + score
    multiplier = max(MULTIPLIER_CLAMP[0], min(MULTIPLIER_CLAMP[1], raw_mult))
    return {
        "rule_type": row["rule_type"],
        "rule_key":  row["rule_key"],
        "verdict":   verdict,
        "n":         int(row["n"] or 0),
        "contexts":  int(row["contexts"] or 0),
        "metric_complete_count": int(row["metric_complete_count"] or 0),
        "score":     round(score, 4),
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
        op_weights[rule_key] = max(cur, multiplier) if verdict == "use" else min(cur, multiplier)
    elif rule_type == "op_pair":
        half = multiplier ** 0.5
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


def compute_construction_prior(
    nb: Any,
    *,
    min_n: int = 4,
    min_metric_complete: int = 3,
    limit: int = 600,
) -> Dict[str, Any]:
    """Compute a fresh construction prior from current evidence.

    Returns a dict ready to be persisted via `record_construction_prior_snapshot`.
    Caller decides whether to activate it.
    """
    rows = nb.conn.execute(
        _CHILD_PARENT_DELTA_SQL,
        (int(min_n), int(min_metric_complete), int(limit)),
    ).fetchall()

    classified_rules: List[Dict[str, Any]] = []
    op_weights: Dict[str, float] = {}
    slot_motif_multipliers: Dict[str, Dict[str, float]] = {}
    slot_motif_denylist: Dict[str, set] = {}
    counts = {"use": 0, "avoid": 0, "mixed": 0}

    for row in rows:
        rule = _classify_row(row)
        counts[rule["verdict"]] = counts.get(rule["verdict"], 0) + 1
        classified_rules.append(rule)
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
    }
    summary = {
        "version": version,
        "n_rules": len(classified_rules),
        "n_use": counts["use"],
        "n_avoid": counts["avoid"],
        "n_mixed": counts["mixed"],
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
