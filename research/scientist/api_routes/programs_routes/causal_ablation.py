"""Causal ablation API surface: per-program evidence + bulk start + diagnostics rollups + construction prior."""

from __future__ import annotations

import logging
import time
from flask import jsonify, request

from ...runner._types import RunConfig
from .._helpers import get_runner
from ...json_utils import json_safe

logger = logging.getLogger(__name__)


_ABLATION_OBSERVATION_METRICS_CTE = """
WITH ablation_observation_metrics AS (
    SELECT obs.parent_result_id,
           obs.parent_fingerprint,
           obs.child_result_id,
           obs.child_fingerprint,
           obs.source,
           obs.rule_type,
           obs.rule_key,
           obs.timestamp,
           cp.stage1_passed AS child_stage1_passed,
           cp.loss_ratio AS child_loss_ratio,
           cp.wikitext_perplexity AS child_ppl,
           cp.hellaswag_acc AS child_hellaswag,
           cp.blimp_overall_accuracy AS child_blimp,
           cp.induction_auc AS child_induction,
           cp.binding_composite AS child_binding,
           cp.ar_auc AS child_ar,
           cp.fp_jacobian_erf_density AS child_erf_density,
           cp.fp_icld_delta_loss AS child_icld_delta,
           cp.trust_label AS child_trust_label,
           cp.comparability_label AS child_comparability_label,
           cp.induction_v2_investigation_auc AS child_induction_v2,
           cp.induction_v2_investigation_status AS child_induction_v2_status,
           cp.binding_v2_investigation_auc AS child_binding_v2,
           cp.binding_v2_investigation_status AS child_binding_v2_status,
           pp.loss_ratio AS parent_loss_ratio,
           pp.wikitext_perplexity AS parent_ppl,
           pp.hellaswag_acc AS parent_hellaswag,
           pp.blimp_overall_accuracy AS parent_blimp,
           pp.induction_auc AS parent_induction,
           pp.binding_composite AS parent_binding,
           pp.ar_auc AS parent_ar,
           pp.fp_jacobian_erf_density AS parent_erf_density,
           pp.induction_v2_investigation_auc AS parent_induction_v2,
           pp.binding_v2_investigation_auc AS parent_binding_v2
    FROM causal_ablation_child_observations obs
    LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
    LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id

    UNION ALL

    SELECT ev.parent_result_id,
           ev.parent_fingerprint,
           json_extract(ev.evidence_json, '$.child_result_id') AS child_result_id,
           json_extract(ev.evidence_json, '$.child.fingerprint') AS child_fingerprint,
           CASE
               WHEN ev.rule_type = 'node_delete_investigation'
               THEN 'knockout_investigation'
               ELSE 'knockout_s1'
           END AS source,
           ev.rule_type,
           ev.rule_key,
           ev.timestamp,
           COALESCE(
               cp.stage1_passed,
               json_extract(ev.evidence_json, '$.child_stage1_passed')
           ) AS child_stage1_passed,
           COALESCE(
               cp.loss_ratio,
               json_extract(ev.evidence_json, '$.child_metrics.loss_ratio')
           ) AS child_loss_ratio,
           COALESCE(
               cp.wikitext_perplexity,
               json_extract(ev.evidence_json, '$.child_metrics.wikitext_perplexity')
           ) AS child_ppl,
           COALESCE(
               cp.hellaswag_acc,
               json_extract(ev.evidence_json, '$.child_metrics.hellaswag_acc')
           ) AS child_hellaswag,
           COALESCE(
               cp.blimp_overall_accuracy,
               json_extract(ev.evidence_json, '$.child_metrics.blimp_overall_accuracy')
           ) AS child_blimp,
           COALESCE(
               cp.induction_auc,
               json_extract(ev.evidence_json, '$.child_metrics.induction_auc')
           ) AS child_induction,
           COALESCE(
               cp.binding_composite,
               json_extract(ev.evidence_json, '$.child_metrics.binding_composite')
           ) AS child_binding,
           COALESCE(
               cp.ar_auc,
               json_extract(ev.evidence_json, '$.child_metrics.ar_auc')
           ) AS child_ar,
           COALESCE(
               cp.fp_jacobian_erf_density,
               json_extract(ev.evidence_json, '$.child_metrics.fp_jacobian_erf_density')
           ) AS child_erf_density,
           COALESCE(
               cp.fp_icld_delta_loss,
               json_extract(ev.evidence_json, '$.child_metrics.fp_icld_delta_loss')
           ) AS child_icld_delta,
           cp.trust_label AS child_trust_label,
           cp.comparability_label AS child_comparability_label,
           COALESCE(
               cp.induction_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.induction_v2_investigation_auc'
               )
           ) AS child_induction_v2,
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
               cp.binding_v2_investigation_status,
               json_extract(
                   ev.evidence_json,
                   '$.child_metrics.binding_v2_investigation_status'
               )
           ) AS child_binding_v2_status,
           COALESCE(
               pp.loss_ratio,
               json_extract(ev.evidence_json, '$.parent_metrics.loss_ratio')
           ) AS parent_loss_ratio,
           COALESCE(
               pp.wikitext_perplexity,
               json_extract(ev.evidence_json, '$.parent_metrics.wikitext_perplexity')
           ) AS parent_ppl,
           COALESCE(
               pp.hellaswag_acc,
               json_extract(ev.evidence_json, '$.parent_metrics.hellaswag_acc')
           ) AS parent_hellaswag,
           COALESCE(
               pp.blimp_overall_accuracy,
               json_extract(ev.evidence_json, '$.parent_metrics.blimp_overall_accuracy')
           ) AS parent_blimp,
           COALESCE(
               pp.induction_auc,
               json_extract(ev.evidence_json, '$.parent_metrics.induction_auc')
           ) AS parent_induction,
           COALESCE(
               pp.binding_composite,
               json_extract(ev.evidence_json, '$.parent_metrics.binding_composite')
           ) AS parent_binding,
           COALESCE(
               pp.ar_auc,
               json_extract(ev.evidence_json, '$.parent_metrics.ar_auc')
           ) AS parent_ar,
           COALESCE(
               pp.fp_jacobian_erf_density,
               json_extract(ev.evidence_json, '$.parent_metrics.fp_jacobian_erf_density')
           ) AS parent_erf_density,
           COALESCE(
               pp.induction_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.parent_metrics.induction_v2_investigation_auc'
               )
           ) AS parent_induction_v2,
           COALESCE(
               pp.binding_v2_investigation_auc,
               json_extract(
                   ev.evidence_json,
                   '$.parent_metrics.binding_v2_investigation_auc'
               )
           ) AS parent_binding_v2
    FROM causal_rule_evidence ev
    LEFT JOIN program_results cp
      ON cp.result_id = json_extract(ev.evidence_json, '$.child_result_id')
    LEFT JOIN program_results pp ON pp.result_id = ev.parent_result_id
    WHERE ev.rule_type IN ('node_delete_s1', 'node_delete_investigation')
      AND json_extract(ev.evidence_json, '$.child.fingerprint') IS NOT NULL
)
"""


def _api_program_causal_evidence(result_id, nb=None):
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    rows = nb.get_causal_rule_evidence(result_id=result_id, limit=50)
    for item in rows:
        evidence_id = item.get("evidence_id")
        if evidence_id:
            item["child_observations"] = nb.get_causal_ablation_child_observations(
                evidence_id=evidence_id,
                limit=200,
            )
    return jsonify(json_safe({"result_id": result_id, "evidence": rows}))


def _api_program_causal_ablation(notebook_path: str, result_id, nb=None):
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    if nb.get_program_detail(result_id) is None:
        return jsonify({"error": "Program not found"}), 404
    body = request.get_json(silent=True) or {}
    config = RunConfig.from_dict(body if isinstance(body, dict) else {})
    config.enable_causal_ablation = True
    config.causal_ablation_top_k = max(
        1, int(body.get("top_k", body.get("causal_ablation_top_k", 1)) or 1)
    )
    config.causal_ablation_max_signals = max(
        1,
        int(body.get("max_signals", body.get("causal_ablation_max_signals", 2)) or 2),
    )
    config.causal_ablation_max_graphs = max(
        1, int(body.get("max_graphs", body.get("causal_ablation_max_graphs", 4)) or 4)
    )
    runner = get_runner(notebook_path, start_projector=True)
    try:
        run_id = runner.start_causal_ablation(result_id, config)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"status": "started", "run_id": run_id, "result_id": result_id})


def _api_bulk_causal_ablation_start(notebook_path: str, nb=None):
    body = request.get_json(silent=True) or {}
    config = RunConfig.from_dict(body if isinstance(body, dict) else {})
    config.continuous = True
    config.enable_causal_ablation = True
    config.causal_ablation_interval = max(
        1, int(body.get("interval", body.get("causal_ablation_interval", 3)) or 3)
    )
    config.causal_ablation_top_k = max(
        1, int(body.get("top_k", body.get("causal_ablation_top_k", 1)) or 1)
    )
    config.causal_ablation_max_signals = max(
        1,
        int(body.get("max_signals", body.get("causal_ablation_max_signals", 2)) or 2),
    )
    config.causal_ablation_max_graphs = max(
        1, int(body.get("max_graphs", body.get("causal_ablation_max_graphs", 4)) or 4)
    )
    if config.max_experiments <= 0:
        config.max_experiments = max(
            1, int(body.get("max_experiments", body.get("n_cycles", 5)) or 5)
        )
    if config.n_programs <= 0:
        config.n_programs = 40
    runner = get_runner(notebook_path, start_projector=True)
    try:
        run_id = runner.start_continuous(config)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(
        {
            "status": "started",
            "run_id": run_id,
            "mode": "continuous",
            "causal_ablation": {
                "interval": config.causal_ablation_interval,
                "top_k": config.causal_ablation_top_k,
                "max_signals": config.causal_ablation_max_signals,
                "max_graphs": config.causal_ablation_max_graphs,
            },
        }
    )


_CAUSAL_ABLATION_SUMMARY_SQL = f"""
        {_ABLATION_OBSERVATION_METRICS_CTE},
        evidence AS (
            SELECT rule_type,
                   rule_key,
                   COUNT(*) AS evidence_count,
                   SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                       AS supported_count,
                   SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                       AS refuted_count,
                   SUM(CASE WHEN outcome = 'inconclusive' THEN 1 ELSE 0 END)
                       AS inconclusive_count,
                   AVG(confidence) AS avg_confidence,
                   AVG(effect_size) AS avg_effect_size,
                   MIN(effect_size) AS min_effect_size,
                   MAX(effect_size) AS max_effect_size
            FROM causal_rule_evidence
            GROUP BY rule_type, rule_key
        ),
        children AS (
            SELECT rule_type,
                   rule_key,
                   COUNT(DISTINCT child_result_id) AS child_result_count,
                   COUNT(DISTINCT child_fingerprint) AS child_fingerprint_count,
                   GROUP_CONCAT(DISTINCT source) AS child_sources,
                   COUNT(*) AS child_observation_count
            FROM ablation_observation_metrics
            GROUP BY rule_type, rule_key
        ),
        metric_rows AS (
            SELECT rule_type,
                   rule_key,
                   child_result_id,
                   CASE
                       WHEN child_hellaswag IS NOT NULL
                        AND child_blimp IS NOT NULL
                        AND child_induction IS NOT NULL
                        AND child_binding IS NOT NULL
                        AND child_ar IS NOT NULL
                        AND child_ppl IS NOT NULL
                       THEN 1 ELSE 0
                   END AS metric_complete,
                   CASE WHEN child_loss_ratio IS NOT NULL
                              AND parent_loss_ratio IS NOT NULL
                        THEN child_loss_ratio - parent_loss_ratio END
                       AS loss_support_effect,
                   CASE WHEN child_hellaswag IS NOT NULL
                              AND parent_hellaswag IS NOT NULL
                        THEN parent_hellaswag - child_hellaswag END
                       AS hellaswag_support_effect,
                   CASE WHEN child_blimp IS NOT NULL
                              AND parent_blimp IS NOT NULL
                        THEN parent_blimp - child_blimp END
                       AS blimp_support_effect,
                   CASE WHEN child_induction IS NOT NULL
                              AND parent_induction IS NOT NULL
                        THEN parent_induction - child_induction END
                       AS induction_support_effect,
                   CASE WHEN child_binding IS NOT NULL
                              AND parent_binding IS NOT NULL
                        THEN parent_binding - child_binding END
                       AS binding_support_effect,
                   CASE WHEN child_ar IS NOT NULL AND parent_ar IS NOT NULL
                        THEN parent_ar - child_ar END
                       AS ar_support_effect,
                   CASE WHEN child_ppl IS NOT NULL
                              AND parent_ppl IS NOT NULL
                              AND parent_ppl > 0
                        THEN (child_ppl - parent_ppl) / parent_ppl END
                       AS wikitext_support_effect,
                   CASE WHEN child_induction_v2 IS NOT NULL
                              AND parent_induction_v2 IS NOT NULL
                        THEN parent_induction_v2 - child_induction_v2 END
                       AS induction_v2_support_effect,
                   CASE WHEN child_binding_v2 IS NOT NULL
                              AND parent_binding_v2 IS NOT NULL
                        THEN parent_binding_v2 - child_binding_v2 END
                       AS binding_v2_support_effect
            FROM ablation_observation_metrics
        ),
        metric_scored AS (
            SELECT *,
                   (
                       CASE WHEN loss_support_effect IS NOT NULL THEN 0.30 ELSE 0 END
                     + CASE WHEN hellaswag_support_effect IS NOT NULL THEN 0.08 ELSE 0 END
                     + CASE WHEN blimp_support_effect IS NOT NULL THEN 0.08 ELSE 0 END
                     + CASE WHEN induction_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                     + CASE WHEN binding_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                     + CASE WHEN ar_support_effect IS NOT NULL THEN 0.04 ELSE 0 END
                     + CASE WHEN wikitext_support_effect IS NOT NULL THEN 0.10 ELSE 0 END
                     + CASE WHEN induction_v2_support_effect IS NOT NULL THEN 0.12 ELSE 0 END
                     + CASE WHEN binding_v2_support_effect IS NOT NULL THEN 0.08 ELSE 0 END
                   ) AS metric_weight,
                   (
                       COALESCE(0.30 * loss_support_effect, 0.0)
                     + COALESCE(0.08 * hellaswag_support_effect, 0.0)
                     + COALESCE(0.08 * blimp_support_effect, 0.0)
                     + COALESCE(0.10 * induction_support_effect, 0.0)
                     + COALESCE(0.10 * binding_support_effect, 0.0)
                     + COALESCE(0.04 * ar_support_effect, 0.0)
                     + COALESCE(0.10 * wikitext_support_effect, 0.0)
                     + COALESCE(0.12 * induction_v2_support_effect, 0.0)
                     + COALESCE(0.08 * binding_v2_support_effect, 0.0)
                   ) AS metric_effect_numerator
            FROM metric_rows
        ),
        metrics AS (
            SELECT rule_type,
                   rule_key,
                   COUNT(child_result_id) AS metric_observation_count,
                   SUM(metric_complete) AS metric_complete_count,
                   SUM(CASE WHEN metric_weight > 0 THEN 1 ELSE 0 END)
                       AS metric_comparable_count,
                   AVG(loss_support_effect) AS avg_loss_support_effect,
                   AVG(hellaswag_support_effect) AS avg_hellaswag_support_effect,
                   AVG(blimp_support_effect) AS avg_blimp_support_effect,
                   AVG(induction_support_effect) AS avg_induction_support_effect,
                   AVG(binding_support_effect) AS avg_binding_support_effect,
                   AVG(ar_support_effect) AS avg_ar_support_effect,
                   AVG(wikitext_support_effect) AS avg_wikitext_support_effect,
                   AVG(induction_v2_support_effect)
                       AS avg_induction_v2_support_effect,
                   AVG(binding_v2_support_effect)
                       AS avg_binding_v2_support_effect,
                   AVG(CASE WHEN metric_weight > 0
                            THEN metric_effect_numerator / metric_weight END)
                       AS composite_support_effect
            FROM metric_scored
            GROUP BY rule_type, rule_key
        )
        SELECT evidence.*,
               COALESCE(children.child_result_count, 0) AS child_result_count,
               COALESCE(children.child_fingerprint_count, 0)
                   AS child_fingerprint_count,
               CASE WHEN evidence.evidence_count > 0
                    THEN CAST(evidence.supported_count AS REAL)
                         / evidence.evidence_count
                    ELSE 0.0 END AS support_rate,
               CASE WHEN evidence.evidence_count > 0
                    THEN CAST(evidence.refuted_count AS REAL)
                         / evidence.evidence_count
                    ELSE 0.0 END AS refute_rate,
               (evidence.supported_count - evidence.refuted_count) AS net_support,
               (
                   ABS(COALESCE(evidence.avg_effect_size, 0.0))
                   * SQRT(CAST(evidence.evidence_count AS REAL))
                   * COALESCE(evidence.avg_confidence, 0.0)
               ) AS stability_score,
               COALESCE(metrics.metric_observation_count, 0)
                   AS metric_observation_count,
               COALESCE(metrics.metric_complete_count, 0)
                   AS metric_complete_count,
               COALESCE(metrics.metric_comparable_count, 0)
                   AS metric_comparable_count,
               CASE WHEN COALESCE(metrics.metric_observation_count, 0) > 0
                    THEN CAST(metrics.metric_complete_count AS REAL)
                         / metrics.metric_observation_count
                    ELSE 0.0 END AS metric_complete_rate,
               metrics.avg_loss_support_effect,
               metrics.avg_hellaswag_support_effect,
               metrics.avg_blimp_support_effect,
               metrics.avg_induction_support_effect,
               metrics.avg_binding_support_effect,
               metrics.avg_ar_support_effect,
               metrics.avg_wikitext_support_effect,
               metrics.avg_induction_v2_support_effect,
               metrics.avg_binding_v2_support_effect,
               metrics.composite_support_effect,
               children.child_sources
        FROM evidence
        LEFT JOIN children
          ON children.rule_type = evidence.rule_type
         AND children.rule_key = evidence.rule_key
        LEFT JOIN metrics
          ON metrics.rule_type = evidence.rule_type
         AND metrics.rule_key = evidence.rule_key
        ORDER BY
                 CASE
                     WHEN evidence.evidence_count >= 3
                      AND COALESCE(children.child_fingerprint_count, 0) >= 3
                      AND COALESCE(metrics.metric_complete_count, 0) >= 3
                     THEN 0 ELSE 1
                 END,
                 ABS(COALESCE(metrics.composite_support_effect, 0.0)) DESC,
                 stability_score DESC,
                 evidence.evidence_count DESC,
                 ABS(COALESCE(evidence.avg_effect_size, 0.0)) DESC
        LIMIT ?

"""


def _api_causal_ablation_summary(nb=None):
    limit = request.args.get("limit", 50, type=int)
    capped_limit = max(1, min(int(limit or 50), 500))
    rows = nb.conn.execute(
        _CAUSAL_ABLATION_SUMMARY_SQL,
        (capped_limit,),
    ).fetchall()
    rows = [dict(row) for row in rows]
    evidence_total = nb.conn.execute(
        "SELECT COUNT(*) AS n FROM causal_rule_evidence"
    ).fetchone()
    observation_total = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE}
        SELECT COUNT(*) AS n FROM ablation_observation_metrics
        """
    ).fetchone()
    outcome_rows = nb.conn.execute(
        """
        SELECT outcome, COUNT(*) AS n, AVG(confidence) AS avg_confidence,
               AVG(effect_size) AS avg_effect_size
        FROM causal_rule_evidence
        GROUP BY outcome
        ORDER BY n DESC
        """
    ).fetchall()
    source_rows = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE}
        SELECT source, COUNT(*) AS n,
               SUM(CASE WHEN COALESCE(child_stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS stage1_count,
               AVG(child_loss_ratio) AS avg_loss_ratio
        FROM ablation_observation_metrics
        GROUP BY source
        ORDER BY n DESC
        """
    ).fetchall()
    backfill_gap = nb.conn.execute(
        """
        SELECT COUNT(*) AS total_ablation_rows,
               SUM(CASE WHEN COALESCE(pr.stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS s1_ablation_rows,
               SUM(CASE WHEN COALESCE(pr.stage1_passed, 0) = 1
                         AND (
                            pr.hellaswag_acc IS NULL
                            OR pr.blimp_overall_accuracy IS NULL
                            OR pr.induction_auc IS NULL
                            OR pr.binding_auc IS NULL
                            OR pr.binding_composite IS NULL
                            OR pr.ar_auc IS NULL
                            OR pr.wikitext_perplexity IS NULL
                         )
                        THEN 1 ELSE 0 END) AS s1_missing_core_metrics
        FROM program_results pr
        WHERE pr.model_source = 'ablation'
        """
    ).fetchone()
    recent_24h = nb.conn.execute(
        """
        SELECT COUNT(*) AS evidence_count,
               COUNT(DISTINCT ablation_experiment_id) AS experiments,
               SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                   AS supported_count,
               SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                   AS refuted_count
        FROM causal_rule_evidence
        WHERE timestamp >= ?
        """,
        (time.time() - 86400,),
    ).fetchone()
    totals = {
        "evidence_count": int(evidence_total["n"] or 0) if evidence_total else 0,
        "observation_count": (
            int(observation_total["n"] or 0) if observation_total else 0
        ),
        "recent_24h": dict(recent_24h) if recent_24h else {},
        "outcomes": [dict(row) for row in outcome_rows],
        "sources": [dict(row) for row in source_rows],
        "backfill_gap": dict(backfill_gap) if backfill_gap else {},
    }
    return jsonify(json_safe({"summary": rows, "totals": totals}))


def _api_causal_ablation_champions(nb=None):
    """Per-champion ablation rollup: how many children, support/refute counts,
    per-metric mean Δ, and metric coverage. Powers the 'By Champion' tab.
    """
    limit = max(1, min(int(request.args.get("limit", 50, type=int) or 50), 500))
    rows = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE},
        children AS (
            SELECT parent_result_id,
                   parent_fingerprint,
                   COUNT(*) AS evidence_count,
                   COUNT(DISTINCT child_fingerprint) AS child_fingerprint_count,
                   SUM(CASE WHEN child_stage1_passed = 1 THEN 1 ELSE 0 END)
                       AS s1_pass_count,
                   SUM(CASE WHEN
                       child_hellaswag IS NOT NULL
                       AND child_blimp IS NOT NULL
                       AND child_induction IS NOT NULL
                       AND child_binding IS NOT NULL
                       AND child_ar IS NOT NULL
                       AND child_ppl IS NOT NULL
                       THEN 1 ELSE 0 END) AS metric_complete_count,
                   SUM(CASE WHEN child_induction_v2 IS NOT NULL
                              OR child_induction_v2_status IS NOT NULL
                       THEN 1 ELSE 0 END) AS induction_v2_count,
                   SUM(CASE WHEN child_binding_v2 IS NOT NULL
                              OR child_binding_v2_status IS NOT NULL
                       THEN 1 ELSE 0 END) AS binding_v2_count,
                   AVG(CASE WHEN child_loss_ratio IS NOT NULL
                                  AND parent_loss_ratio IS NOT NULL
                            THEN child_loss_ratio - parent_loss_ratio END)
                       AS avg_loss_delta,
                   AVG(CASE WHEN child_induction IS NOT NULL
                                  AND parent_induction IS NOT NULL
                            THEN parent_induction - child_induction END)
                       AS avg_induction_drop,
                   AVG(CASE WHEN child_binding IS NOT NULL
                                  AND parent_binding IS NOT NULL
                            THEN parent_binding - child_binding END)
                       AS avg_binding_drop,
                   AVG(CASE WHEN child_ar IS NOT NULL AND parent_ar IS NOT NULL
                            THEN parent_ar - child_ar END)
                       AS avg_ar_drop,
                   AVG(CASE WHEN child_hellaswag IS NOT NULL
                                  AND parent_hellaswag IS NOT NULL
                            THEN parent_hellaswag - child_hellaswag END)
                       AS avg_hellaswag_drop,
                   AVG(CASE WHEN child_blimp IS NOT NULL
                                  AND parent_blimp IS NOT NULL
                            THEN parent_blimp - child_blimp END)
                       AS avg_blimp_drop,
                   AVG(CASE WHEN child_ppl IS NOT NULL
                                  AND parent_ppl IS NOT NULL
                                  AND parent_ppl > 0
                            THEN (child_ppl - parent_ppl) / parent_ppl END)
                       AS avg_ppl_pct_change,
                   AVG(CASE WHEN child_induction_v2 IS NOT NULL
                                  AND parent_induction_v2 IS NOT NULL
                            THEN parent_induction_v2 - child_induction_v2 END)
                       AS avg_induction_v2_drop,
                   AVG(CASE WHEN child_binding_v2 IS NOT NULL
                                  AND parent_binding_v2 IS NOT NULL
                            THEN parent_binding_v2 - child_binding_v2 END)
                       AS avg_binding_v2_drop
            FROM ablation_observation_metrics
            GROUP BY parent_result_id, parent_fingerprint
        ),
        rules AS (
            SELECT parent_result_id,
                   COUNT(*) AS rule_count,
                   SUM(CASE WHEN outcome = 'supported' THEN 1 ELSE 0 END)
                       AS supported_count,
                   SUM(CASE WHEN outcome LIKE 'refuted%' THEN 1 ELSE 0 END)
                       AS refuted_count
            FROM causal_rule_evidence
            GROUP BY parent_result_id
        )
        SELECT children.parent_result_id   AS result_id,
               children.parent_fingerprint AS graph_fingerprint,
               children.evidence_count,
               children.child_fingerprint_count,
               children.s1_pass_count,
               children.metric_complete_count,
               CASE WHEN children.evidence_count > 0
                    THEN CAST(children.metric_complete_count AS REAL)
                         / children.evidence_count
                    ELSE 0.0 END AS metric_complete_rate,
               children.induction_v2_count,
               children.binding_v2_count,
               COALESCE(rules.rule_count, 0) AS rule_count,
               COALESCE(rules.supported_count, 0) AS supported_count,
               COALESCE(rules.refuted_count, 0) AS refuted_count,
               children.avg_loss_delta,
               children.avg_induction_drop,
               children.avg_binding_drop,
               children.avg_ar_drop,
               children.avg_hellaswag_drop,
               children.avg_blimp_drop,
               children.avg_ppl_pct_change,
               children.avg_induction_v2_drop,
               children.avg_binding_v2_drop,
               l.composite_score,
               l.tier,
               pp.loss_ratio AS parent_loss_ratio,
               pp.wikitext_perplexity AS parent_wikitext_perplexity,
               pp.induction_auc AS parent_induction_auc,
               pp.binding_composite AS parent_binding_composite,
               pp.hellaswag_acc AS parent_hellaswag_acc,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.ar_auc AS parent_ar_auc,
               pp.induction_v2_investigation_auc AS parent_induction_v2,
               pp.binding_v2_investigation_auc AS parent_binding_v2
        FROM children
        LEFT JOIN rules ON rules.parent_result_id = children.parent_result_id
        LEFT JOIN program_results pp ON pp.result_id = children.parent_result_id
        LEFT JOIN leaderboard l ON l.result_id = children.parent_result_id
        ORDER BY children.evidence_count DESC, children.child_fingerprint_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return jsonify(json_safe({"champions": [dict(r) for r in rows]}))


def _api_causal_ablation_components(nb=None):
    """Per-component (op / op_pair / motif) summary across all champions.
    Surfaces what an op does in different contexts. Powers 'By Component'.
    """
    limit = max(1, min(int(request.args.get("limit", 200, type=int) or 200), 1000))
    rule_type = request.args.get("rule_type", "")
    where = ""
    params: list = []
    if rule_type:
        where = "WHERE rule_type = ?"
        params.append(rule_type)
    rows = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE},
        metric_rows AS (
            SELECT rule_type,
                   rule_key,
                   parent_result_id,
                   CASE WHEN child_loss_ratio IS NOT NULL
                              AND parent_loss_ratio IS NOT NULL
                        THEN child_loss_ratio - parent_loss_ratio END AS d_loss,
                   CASE WHEN child_induction IS NOT NULL
                              AND parent_induction IS NOT NULL
                        THEN parent_induction - child_induction END AS d_induction,
                   CASE WHEN child_binding IS NOT NULL AND parent_binding IS NOT NULL
                        THEN parent_binding - child_binding END AS d_binding,
                   CASE WHEN child_ar IS NOT NULL AND parent_ar IS NOT NULL
                        THEN parent_ar - child_ar END AS d_ar,
                   CASE WHEN child_hellaswag IS NOT NULL
                              AND parent_hellaswag IS NOT NULL
                        THEN parent_hellaswag - child_hellaswag END AS d_hellaswag,
                   CASE WHEN child_blimp IS NOT NULL AND parent_blimp IS NOT NULL
                        THEN parent_blimp - child_blimp END AS d_blimp,
                   CASE WHEN child_ppl IS NOT NULL
                              AND parent_ppl IS NOT NULL
                              AND parent_ppl > 0
                        THEN (child_ppl - parent_ppl) / parent_ppl END AS d_ppl_pct,
                   CASE WHEN child_induction_v2 IS NOT NULL
                              AND parent_induction_v2 IS NOT NULL
                        THEN parent_induction_v2 - child_induction_v2
                   END AS d_induction_v2,
                   CASE WHEN child_binding_v2 IS NOT NULL
                              AND parent_binding_v2 IS NOT NULL
                        THEN parent_binding_v2 - child_binding_v2
                   END AS d_binding_v2
            FROM ablation_observation_metrics
        )
        SELECT rule_type,
               rule_key,
               COUNT(*) AS observation_count,
               COUNT(DISTINCT parent_result_id) AS parent_count,
               AVG(d_loss) AS avg_d_loss,
               AVG(d_induction) AS avg_d_induction,
               AVG(d_binding) AS avg_d_binding,
               AVG(d_ar) AS avg_d_ar,
               AVG(d_hellaswag) AS avg_d_hellaswag,
               AVG(d_blimp) AS avg_d_blimp,
               AVG(d_ppl_pct) AS avg_d_ppl_pct,
               AVG(d_induction_v2) AS avg_d_induction_v2,
               AVG(d_binding_v2) AS avg_d_binding_v2,
               SUM(CASE WHEN d_loss IS NOT NULL THEN 1 ELSE 0 END) AS n_loss,
               SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
               SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
               SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
               SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
               SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
               SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl,
               SUM(CASE WHEN d_induction_v2 IS NOT NULL THEN 1 ELSE 0 END)
                   AS n_induction_v2,
               SUM(CASE WHEN d_binding_v2 IS NOT NULL THEN 1 ELSE 0 END)
                   AS n_binding_v2
        FROM metric_rows
        {where}
        GROUP BY rule_type, rule_key
        ORDER BY observation_count DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return jsonify(json_safe({"components": [dict(r) for r in rows]}))


def _api_causal_ablation_recommendations(nb=None):
    """Construction recommendations: ✓ USE / ✗ AVOID / ⚠ MIXED rules with
    n, contexts, and per-metric average impact. The 'do this' surface.
    """
    limit = max(1, min(int(request.args.get("limit", 80, type=int) or 80), 400))
    min_n = max(2, int(request.args.get("min_n", 4, type=int) or 4))
    rows = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE},
        metric_rows AS (
            SELECT rule_type,
                   rule_key,
                   parent_result_id,
                   CASE WHEN child_loss_ratio IS NOT NULL
                              AND parent_loss_ratio IS NOT NULL
                        THEN child_loss_ratio - parent_loss_ratio END AS d_loss,
                   CASE WHEN child_induction IS NOT NULL
                              AND parent_induction IS NOT NULL
                        THEN parent_induction - child_induction END AS d_induction,
                   CASE WHEN child_binding IS NOT NULL AND parent_binding IS NOT NULL
                        THEN parent_binding - child_binding END AS d_binding,
                   CASE WHEN child_ar IS NOT NULL AND parent_ar IS NOT NULL
                        THEN parent_ar - child_ar END AS d_ar,
                   CASE WHEN child_hellaswag IS NOT NULL
                              AND parent_hellaswag IS NOT NULL
                        THEN parent_hellaswag - child_hellaswag END AS d_hellaswag,
                   CASE WHEN child_blimp IS NOT NULL AND parent_blimp IS NOT NULL
                        THEN parent_blimp - child_blimp END AS d_blimp,
                   CASE WHEN child_ppl IS NOT NULL
                              AND parent_ppl IS NOT NULL
                              AND parent_ppl > 0
                        THEN (child_ppl - parent_ppl) / parent_ppl END AS d_ppl_pct,
                   CASE WHEN child_induction_v2 IS NOT NULL
                              AND parent_induction_v2 IS NOT NULL
                        THEN parent_induction_v2 - child_induction_v2
                   END AS d_induction_v2,
                   CASE WHEN child_binding_v2 IS NOT NULL
                              AND parent_binding_v2 IS NOT NULL
                        THEN parent_binding_v2 - child_binding_v2
                   END AS d_binding_v2,
                   CASE WHEN child_hellaswag IS NOT NULL
                             AND child_blimp IS NOT NULL
                             AND child_induction IS NOT NULL
                             AND child_binding IS NOT NULL
                             AND child_ar IS NOT NULL
                             AND child_ppl IS NOT NULL
                        THEN 1 ELSE 0 END AS metric_complete
            FROM ablation_observation_metrics
        ),
        agg AS (
            SELECT rule_type,
                   rule_key,
                   COUNT(*) AS n,
                   COUNT(DISTINCT parent_result_id) AS contexts,
                   SUM(metric_complete) AS metric_complete_count,
                   AVG(d_loss) AS avg_d_loss,
                   AVG(d_induction) AS avg_d_induction,
                   AVG(d_binding) AS avg_d_binding,
                   AVG(d_ar) AS avg_d_ar,
                   AVG(d_hellaswag) AS avg_d_hellaswag,
                   AVG(d_blimp) AS avg_d_blimp,
                   AVG(d_ppl_pct) AS avg_d_ppl_pct,
                   AVG(d_induction_v2) AS avg_d_induction_v2,
                   AVG(d_binding_v2) AS avg_d_binding_v2,
                   SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
                   SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
                   SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
                   SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
                   SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
                   SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl,
                   SUM(CASE WHEN d_induction_v2 IS NOT NULL THEN 1 ELSE 0 END)
                       AS n_induction_v2,
                   SUM(CASE WHEN d_binding_v2 IS NOT NULL THEN 1 ELSE 0 END)
                       AS n_binding_v2
            FROM metric_rows
            GROUP BY rule_type, rule_key
            HAVING COUNT(*) >= {int(min_n)} AND SUM(metric_complete) >= 3
        )
        SELECT * FROM agg
        ORDER BY n DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return jsonify(json_safe({"recommendations": [dict(r) for r in rows]}))


def _api_causal_ablation_children_for_rule(nb=None):
    """Drill-down: child observations for a given rule_type/rule_key with
    full per-metric numbers. Used by the rule detail drawer."""
    rule_type = request.args.get("rule_type", "")
    rule_key = request.args.get("rule_key", "")
    parent_result_id = request.args.get("parent_result_id", "")
    if (not rule_type or not rule_key) and not parent_result_id:
        return jsonify(
            {"error": "rule_type/rule_key or parent_result_id required"}
        ), 400
    limit = max(1, min(int(request.args.get("limit", 100, type=int) or 100), 500))
    clauses: list[str] = []
    params: list = []
    if rule_type:
        clauses.append("rule_type = ?")
        params.append(rule_type)
    if rule_key:
        clauses.append("rule_key = ?")
        params.append(rule_key)
    if parent_result_id:
        clauses.append("parent_result_id = ?")
        params.append(parent_result_id)
    where = " AND ".join(clauses)
    rows = nb.conn.execute(
        f"""
        {_ABLATION_OBSERVATION_METRICS_CTE}
        SELECT parent_result_id,
               parent_fingerprint,
               child_result_id,
               child_fingerprint,
               source,
               child_loss_ratio,
               child_ppl,
               child_hellaswag,
               child_blimp,
               child_induction,
               child_binding,
               child_ar,
               child_erf_density,
               child_icld_delta,
               child_trust_label,
               child_comparability_label,
               child_induction_v2,
               child_induction_v2_status,
               child_binding_v2,
               child_binding_v2_status,
               parent_loss_ratio,
               parent_ppl,
               parent_hellaswag,
               parent_blimp,
               parent_induction,
               parent_binding,
               parent_ar,
               parent_erf_density,
               parent_induction_v2,
               parent_binding_v2
        FROM ablation_observation_metrics
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return jsonify(json_safe({"children": [dict(r) for r in rows]}))


def _api_construction_prior_active(nb=None):
    """Return the currently active construction prior snapshot."""
    from research.scientist.construction_priors import (
        get_active_construction_prior,
        list_construction_prior_snapshots,
    )

    active = get_active_construction_prior(nb)
    snapshots = list_construction_prior_snapshots(nb, limit=20)
    return jsonify(json_safe({"active": active, "snapshots": snapshots}))


def _api_construction_prior_refresh(nb=None):
    """Compute a fresh prior from current evidence and activate it."""
    from research.scientist.construction_priors import (
        compute_construction_prior,
        record_construction_prior_snapshot,
    )

    body = (request.is_json and request.json) or {}
    min_n = max(2, int(body.get("min_n", 4) or 4))
    notes = str(body.get("notes") or "")
    prior = compute_construction_prior(nb, min_n=min_n)
    if not prior["payload"]["rules"]:
        return jsonify({"error": "no rules met threshold; nothing to snapshot"}), 400
    version = record_construction_prior_snapshot(nb, prior, activate=True, notes=notes)
    return jsonify(
        json_safe(
            {
                "status": "activated",
                "version": version,
                "summary": prior["summary"],
            }
        )
    )
