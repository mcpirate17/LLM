"""Component metric overlay SQL for observability health views."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_WINDOW_SECONDS: Dict[str, Optional[int]] = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
    "all": None,
}

_OVERLAY_FIELDS = (
    "avg_loss_ratio",
    "avg_validation_loss_ratio",
    "avg_induction_screening_auc",
    "avg_binding_screening_auc",
    "avg_binding_screening_composite",
    "avg_ar_legacy_auc",
    "avg_induction_intermediate_auc",
    "avg_binding_intermediate_auc",
    "avg_ar_curriculum_auc_pair_final",
    "avg_ar_curriculum_s0_retention",
    "avg_ar_curriculum_max_passing_stage",
    "n_ar_curriculum",
    "avg_hellaswag_acc",
    "avg_blimp_overall_accuracy",
    "avg_language_control_s05_sentence_assoc_score",
    "avg_language_control_s05_binding_order_acc",
    "avg_language_control_s05_binding_score",
    "avg_language_control_s10_sentence_assoc_score",
    "avg_language_control_s10_binding_order_acc",
    "avg_language_control_s10_binding_score",
    "avg_language_control_investigation_sentence_assoc_score",
    "avg_language_control_investigation_binding_order_acc",
    "avg_language_control_investigation_binding_score",
    "avg_composite_score",
    "avg_erf_density",
    "avg_id_collapse_rate",
    "avg_id_collapse_rate_normalized",
    "avg_erf_decay_slope",
    "avg_erf_first_norm",
    "avg_erf_last_norm",
    "avg_logit_margin_velocity",
    "avg_logit_margin_delta",
    "avg_erf_variance_log",
    "avg_spec_norm_log",
    "avg_icld_velocity",
    "avg_icld_delta_loss",
    "avg_jacobian_effective_rank",
    "avg_sensitivity_uniformity",
)

_METRIC_OVERLAY_SQL = """
WITH op_rows AS (
    SELECT DISTINCT
        pr.result_id AS result_id,
        gpo.op_name AS op_name,
        pr.loss_ratio AS loss_ratio,
        pr.validation_loss_ratio AS validation_loss_ratio,
        pr.induction_screening_auc AS induction_screening_auc,
        pr.induction_intermediate_auc AS induction_intermediate_auc,
        COALESCE(pr.binding_curriculum_auc, pr.binding_screening_auc) AS binding_screening_auc,
        {binding_screening_composite} AS binding_screening_composite,
        pr.binding_intermediate_auc AS binding_intermediate_auc,
        {ar_legacy_auc} AS ar_legacy_auc,
        {ar_curriculum_auc_pair_final} AS ar_curriculum_auc_pair_final,
        {ar_curriculum_s0_retention} AS ar_curriculum_s0_retention,
        {ar_curriculum_max_passing_stage} AS ar_curriculum_max_passing_stage,
        COALESCE(
            pr.hellaswag_acc,
            CASE
                WHEN pr.screening_hellaswag_total > 0
                THEN CAST(pr.screening_hellaswag_correct AS REAL)
                     / pr.screening_hellaswag_total
                ELSE NULL
            END
        ) AS hellaswag_acc,
        pr.blimp_overall_accuracy AS blimp_overall_accuracy,
        pr.language_control_s05_sentence_assoc_score AS language_control_s05_sentence_assoc_score,
        pr.language_control_s05_binding_order_acc AS language_control_s05_binding_order_acc,
        pr.language_control_s05_binding_score AS language_control_s05_binding_score,
        pr.language_control_s10_sentence_assoc_score AS language_control_s10_sentence_assoc_score,
        pr.language_control_s10_binding_order_acc AS language_control_s10_binding_order_acc,
        pr.language_control_s10_binding_score AS language_control_s10_binding_score,
        pr.language_control_investigation_sentence_assoc_score AS language_control_investigation_sentence_assoc_score,
        pr.language_control_investigation_binding_order_acc AS language_control_investigation_binding_order_acc,
        pr.language_control_investigation_binding_score AS language_control_investigation_binding_score,
        l.composite_score AS composite_score,
        pr.fp_jacobian_effective_rank AS jacobian_effective_rank,
        pr.fp_sensitivity_uniformity AS sensitivity_uniformity,
        pr.fp_jacobian_erf_density AS erf_density,
        pr.fp_id_collapse_rate AS id_collapse_rate,
        pr.fp_id_collapse_rate_normalized AS id_collapse_rate_normalized,
        pr.fp_jacobian_erf_decay_slope AS erf_decay_slope,
        pr.fp_jacobian_erf_first_norm AS erf_first_norm,
        pr.fp_jacobian_erf_last_norm AS erf_last_norm,
        pr.fp_logit_margin_velocity AS logit_margin_velocity,
        pr.fp_logit_margin_delta AS logit_margin_delta,
        CASE WHEN pr.fp_jacobian_erf_variance IS NOT NULL
             THEN log(abs(pr.fp_jacobian_erf_variance) + 0.000000001)
             ELSE NULL
        END AS erf_variance_log,
        CASE WHEN pr.fp_jacobian_spectral_norm IS NOT NULL
             THEN log(abs(pr.fp_jacobian_spectral_norm) + 0.000000001)
             ELSE NULL
        END AS spec_norm_log,
        pr.fp_icld_velocity AS icld_velocity,
        pr.fp_icld_delta_loss AS icld_delta_loss
    FROM program_results pr
    JOIN program_graph_ops gpo ON gpo.result_id = pr.result_id
    LEFT JOIN (
        SELECT result_id, AVG(composite_score) AS composite_score
        FROM leaderboard
        GROUP BY result_id
    ) l ON l.result_id = pr.result_id
    WHERE {where}
)
SELECT
    op_name,
    AVG(loss_ratio) AS avg_loss_ratio,
    AVG(validation_loss_ratio) AS avg_validation_loss_ratio,
    AVG(induction_screening_auc) AS avg_induction_screening_auc,
    AVG(binding_screening_auc) AS avg_binding_screening_auc,
    AVG(binding_screening_composite) AS avg_binding_screening_composite,
    AVG(ar_legacy_auc) AS avg_ar_legacy_auc,
    AVG(induction_intermediate_auc) AS avg_induction_intermediate_auc,
    AVG(binding_intermediate_auc) AS avg_binding_intermediate_auc,
    AVG(ar_curriculum_auc_pair_final) AS avg_ar_curriculum_auc_pair_final,
    AVG(ar_curriculum_s0_retention) AS avg_ar_curriculum_s0_retention,
    AVG(ar_curriculum_max_passing_stage) AS avg_ar_curriculum_max_passing_stage,
    COUNT(ar_curriculum_auc_pair_final) AS n_ar_curriculum,
    AVG(hellaswag_acc) AS avg_hellaswag_acc,
    AVG(blimp_overall_accuracy) AS avg_blimp_overall_accuracy,
    AVG(language_control_s05_sentence_assoc_score) AS avg_language_control_s05_sentence_assoc_score,
    AVG(language_control_s05_binding_order_acc) AS avg_language_control_s05_binding_order_acc,
    AVG(language_control_s05_binding_score) AS avg_language_control_s05_binding_score,
    AVG(language_control_s10_sentence_assoc_score) AS avg_language_control_s10_sentence_assoc_score,
    AVG(language_control_s10_binding_order_acc) AS avg_language_control_s10_binding_order_acc,
    AVG(language_control_s10_binding_score) AS avg_language_control_s10_binding_score,
    AVG(language_control_investigation_sentence_assoc_score) AS avg_language_control_investigation_sentence_assoc_score,
    AVG(language_control_investigation_binding_order_acc) AS avg_language_control_investigation_binding_order_acc,
    AVG(language_control_investigation_binding_score) AS avg_language_control_investigation_binding_score,
    AVG(composite_score) AS avg_composite_score,
    AVG(erf_density) AS avg_erf_density,
    AVG(id_collapse_rate) AS avg_id_collapse_rate,
    AVG(id_collapse_rate_normalized) AS avg_id_collapse_rate_normalized,
    AVG(erf_decay_slope) AS avg_erf_decay_slope,
    AVG(erf_first_norm) AS avg_erf_first_norm,
    AVG(erf_last_norm) AS avg_erf_last_norm,
    AVG(logit_margin_velocity) AS avg_logit_margin_velocity,
    AVG(logit_margin_delta) AS avg_logit_margin_delta,
    AVG(erf_variance_log) AS avg_erf_variance_log,
    AVG(spec_norm_log) AS avg_spec_norm_log,
    AVG(icld_velocity) AS avg_icld_velocity,
    AVG(icld_delta_loss) AS avg_icld_delta_loss,
    AVG(jacobian_effective_rank) AS avg_jacobian_effective_rank,
    AVG(sensitivity_uniformity) AS avg_sensitivity_uniformity
FROM op_rows
GROUP BY op_name
"""

_FAILURE_REASON_SQL = """
WITH reason_rows AS (
    SELECT DISTINCT
        pr.result_id AS result_id,
        gpo.op_name AS op_name,
        COALESCE(NULLIF(pr.error_type, ''), NULLIF(pr.stage_at_death, '')) AS reason
    FROM program_results pr
    JOIN program_graph_ops gpo ON gpo.result_id = pr.result_id
    WHERE {where} AND pr.stage1_passed = 0
),
reason_counts AS (
    SELECT op_name, reason, COUNT(*) AS n
    FROM reason_rows
    WHERE reason IS NOT NULL AND reason <> ''
    GROUP BY op_name, reason
),
ranked AS (
    SELECT
        op_name,
        reason,
        ROW_NUMBER() OVER (
            PARTITION BY op_name
            ORDER BY n DESC, reason ASC
        ) AS rn
    FROM reason_counts
)
SELECT op_name, reason
FROM ranked
WHERE rn = 1
"""


def _metric_where(window: str) -> tuple[str, tuple[Any, ...]]:
    where = "gpo.op_name IS NOT NULL AND gpo.op_name <> '' AND gpo.op_name <> 'input'"
    window_seconds = _WINDOW_SECONDS.get(window)
    if window_seconds is None:
        return where, ()
    return f"{where} AND pr.timestamp > ?", (time.time() - window_seconds,)


def _program_result_expr(program_result_columns: set[str], column: str) -> str:
    return f"pr.{column}" if column in program_result_columns else "NULL"


def _metric_sql(where: str, program_result_columns: set[str]) -> str:
    return _METRIC_OVERLAY_SQL.format(
        where=where,
        binding_screening_composite=_program_result_expr(
            program_result_columns, "binding_screening_composite"
        ),
        ar_legacy_auc=_program_result_expr(program_result_columns, "ar_legacy_auc"),
        ar_curriculum_auc_pair_final=_program_result_expr(
            program_result_columns, "ar_curriculum_auc_pair_final"
        ),
        ar_curriculum_s0_retention=_program_result_expr(
            program_result_columns, "ar_curriculum_s0_retention"
        ),
        ar_curriculum_max_passing_stage=_program_result_expr(
            program_result_columns, "ar_curriculum_max_passing_stage"
        ),
    )


def load_component_metric_overlays(nb, window: str) -> Dict[str, Dict[str, Any]]:
    where, params = _metric_where(window)
    overlays: Dict[str, Dict[str, Any]] = {}
    try:
        program_result_columns = {
            str(row[1])
            for row in nb.conn.execute("PRAGMA table_info(program_results)").fetchall()
        }
        rows = nb.conn.execute(
            _metric_sql(where, program_result_columns), params
        ).fetchall()
        for row in rows:
            overlays[row["op_name"]] = {field: row[field] for field in _OVERLAY_FIELDS}

        reason_rows = nb.conn.execute(
            _FAILURE_REASON_SQL.format(where=where), params
        ).fetchall()
        for row in reason_rows:
            overlays.setdefault(row["op_name"], {})["top_failure_reason"] = row[
                "reason"
            ]
    except sqlite3.OperationalError as exc:
        logger.debug("component metric overlay query failed: %s", exc)
    return overlays
