"""Causal ablation API surface: per-program evidence + bulk start + diagnostics rollups + construction prior."""

from __future__ import annotations

import logging
import time
from flask import jsonify, request

from .._helpers import get_runner
from ...json_utils import json_safe

logger = logging.getLogger(__name__)


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


def _api_causal_ablation_summary(nb=None):
    limit = request.args.get("limit", 50, type=int)
    rows = nb.get_causal_component_interaction_summary(limit=limit)
    evidence_total = nb.conn.execute(
        "SELECT COUNT(*) AS n FROM causal_rule_evidence"
    ).fetchone()
    observation_total = nb.conn.execute(
        "SELECT COUNT(*) AS n FROM causal_ablation_child_observations"
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
        """
        SELECT source, COUNT(*) AS n,
               SUM(CASE WHEN COALESCE(stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS stage1_count,
               AVG(loss_ratio) AS avg_loss_ratio
        FROM causal_ablation_child_observations
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
                            OR pr.wikitext_score IS NULL
                            OR pr.fp_jacobian_erf_density IS NULL
                            OR pr.fp_icld_delta_loss IS NULL
                            OR pr.fp_logit_margin_delta IS NULL
                         )
                        THEN 1 ELSE 0 END) AS s1_missing_core_metrics
        FROM program_results pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE e.experiment_type = 'ablation'
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
        """
        WITH children AS (
            SELECT obs.parent_result_id,
                   obs.parent_fingerprint,
                   COUNT(*) AS evidence_count,
                   COUNT(DISTINCT obs.child_fingerprint) AS child_fingerprint_count,
                   SUM(CASE WHEN cp.stage1_passed = 1 THEN 1 ELSE 0 END)
                       AS s1_pass_count,
                   SUM(CASE WHEN
                       cp.hellaswag_acc IS NOT NULL
                       AND cp.blimp_overall_accuracy IS NOT NULL
                       AND cp.induction_auc IS NOT NULL
                       AND cp.binding_auc IS NOT NULL
                       AND cp.binding_composite IS NOT NULL
                       AND cp.ar_auc IS NOT NULL
                       AND cp.wikitext_perplexity IS NOT NULL
                       THEN 1 ELSE 0 END) AS metric_complete_count,
                   AVG(CASE WHEN cp.loss_ratio IS NOT NULL AND pp.loss_ratio IS NOT NULL
                            THEN cp.loss_ratio - pp.loss_ratio END)
                       AS avg_loss_delta,
                   AVG(CASE WHEN cp.induction_auc IS NOT NULL AND pp.induction_auc IS NOT NULL
                            THEN pp.induction_auc - cp.induction_auc END)
                       AS avg_induction_drop,
                   AVG(CASE WHEN cp.binding_composite IS NOT NULL AND pp.binding_composite IS NOT NULL
                            THEN pp.binding_composite - cp.binding_composite END)
                       AS avg_binding_drop,
                   AVG(CASE WHEN cp.ar_auc IS NOT NULL AND pp.ar_auc IS NOT NULL
                            THEN pp.ar_auc - cp.ar_auc END)
                       AS avg_ar_drop,
                   AVG(CASE WHEN cp.hellaswag_acc IS NOT NULL AND pp.hellaswag_acc IS NOT NULL
                            THEN pp.hellaswag_acc - cp.hellaswag_acc END)
                       AS avg_hellaswag_drop,
                   AVG(CASE WHEN cp.blimp_overall_accuracy IS NOT NULL
                                 AND pp.blimp_overall_accuracy IS NOT NULL
                            THEN pp.blimp_overall_accuracy - cp.blimp_overall_accuracy END)
                       AS avg_blimp_drop,
                   AVG(CASE WHEN cp.wikitext_perplexity IS NOT NULL
                                 AND pp.wikitext_perplexity IS NOT NULL
                                 AND pp.wikitext_perplexity > 0
                            THEN (cp.wikitext_perplexity - pp.wikitext_perplexity)
                                 / pp.wikitext_perplexity END)
                       AS avg_ppl_pct_change
            FROM causal_ablation_child_observations obs
            LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
            LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
            GROUP BY obs.parent_result_id, obs.parent_fingerprint
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
               l.composite_score,
               l.tier,
               pp.loss_ratio AS parent_loss_ratio,
               pp.wikitext_perplexity AS parent_wikitext_perplexity,
               pp.induction_auc AS parent_induction_auc,
               pp.binding_composite AS parent_binding_composite,
               pp.hellaswag_acc AS parent_hellaswag_acc,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.ar_auc AS parent_ar_auc
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
                             / pp.wikitext_perplexity END AS d_ppl_pct
            FROM causal_ablation_child_observations obs
            LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
            LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
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
               SUM(CASE WHEN d_loss IS NOT NULL THEN 1 ELSE 0 END) AS n_loss,
               SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
               SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
               SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
               SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
               SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
               SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl
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
                   SUM(CASE WHEN d_induction IS NOT NULL THEN 1 ELSE 0 END) AS n_induction,
                   SUM(CASE WHEN d_binding IS NOT NULL THEN 1 ELSE 0 END) AS n_binding,
                   SUM(CASE WHEN d_blimp IS NOT NULL THEN 1 ELSE 0 END) AS n_blimp,
                   SUM(CASE WHEN d_hellaswag IS NOT NULL THEN 1 ELSE 0 END) AS n_hellaswag,
                   SUM(CASE WHEN d_ar IS NOT NULL THEN 1 ELSE 0 END) AS n_ar,
                   SUM(CASE WHEN d_ppl_pct IS NOT NULL THEN 1 ELSE 0 END) AS n_ppl
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
    if not rule_type or not rule_key:
        return jsonify({"error": "rule_type and rule_key required"}), 400
    limit = max(1, min(int(request.args.get("limit", 100, type=int) or 100), 500))
    where = "obs.rule_type = ? AND obs.rule_key = ?"
    params: list = [rule_type, rule_key]
    if parent_result_id:
        where += " AND obs.parent_result_id = ?"
        params.append(parent_result_id)
    rows = nb.conn.execute(
        f"""
        SELECT obs.parent_result_id,
               obs.parent_fingerprint,
               obs.child_result_id,
               obs.child_fingerprint,
               obs.source,
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
               pp.loss_ratio AS parent_loss_ratio,
               pp.wikitext_perplexity AS parent_ppl,
               pp.hellaswag_acc AS parent_hellaswag,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.induction_auc AS parent_induction,
               pp.binding_composite AS parent_binding,
               pp.ar_auc AS parent_ar,
               pp.fp_jacobian_erf_density AS parent_erf_density
        FROM causal_ablation_child_observations obs
        LEFT JOIN program_results cp ON cp.result_id = obs.child_result_id
        LEFT JOIN program_results pp ON pp.result_id = obs.parent_result_id
        WHERE {where}
        ORDER BY obs.timestamp DESC
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
