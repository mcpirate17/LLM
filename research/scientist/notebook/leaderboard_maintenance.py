from __future__ import annotations

"""Focused leaderboard maintenance helpers."""

import time
from typing import Any, Dict, List, Optional

from ..leaderboard_scoring import build_score_kwargs, compute_composite
from ..thresholds import TIER_RANK


def _timestamp_key(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(time.mktime(time.strptime(str(value), "%Y-%m-%dT%H:%M:%S")))
        except (TypeError, ValueError):
            return 0.0


FINGERPRINT_PROGRAM_RESULT_COLUMNS = (
    "result_id",
    "experiment_id",
    "loss_ratio",
    "novelty_score",
    "novelty_confidence",
    "fp_jacobian_spectral_norm",
    "loss_improvement_rate",
    "discovery_loss_ratio",
    "validation_loss_ratio",
    "baseline_loss_ratio",
    "validation_multi_seed_std",
    "validation_robustness_score",
    "validation_is_unstable",
    "validation_passed",
    "normalized_baseline_ratio",
    "param_efficiency",
    "efficiency_multiple",
    "max_viable_seq_len",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "robustness_noise_score",
    "activation_sparsity_score",
    "depth_savings_ratio",
    "recursion_savings_ratio",
    "routing_expert_count",
    "routing_confidence_mean",
    "routing_drop_rate",
    "wikitext_perplexity",
    "wikitext_score",
    "tinystories_perplexity",
    "tinystories_score",
    "cross_task_score",
    "efficiency_wall_score",
    "param_count",
    "throughput_tok_s",
    "forward_time_ms",
    "n_train_steps",
)

FINGERPRINT_MIN_COLUMNS = (
    "screening_loss_ratio",
    "investigation_loss_ratio",
    "validation_loss_ratio",
    "validation_baseline_ratio",
    "validation_multi_seed_std",
    "discovery_loss_ratio",
    "fp_jacobian_spectral_norm",
    "compression_ratio",
    "routing_drop_rate",
    "robustness_noise_score",
    "wikitext_perplexity",
    "tinystories_perplexity",
    "ncd_score",
)

FINGERPRINT_MAX_COLUMNS = (
    "screening_novelty",
    "investigation_robustness",
    "normalized_baseline_ratio",
    "param_efficiency",
    "quant_int8_retention",
    "quant_quality_per_byte",
    "robustness_long_ctx_score",
    "init_sensitivity_std",
    "scaling_param_efficiency",
    "scaling_flop_efficiency",
    "scaling_d512_param_efficiency",
    "routing_savings_ratio",
    "activation_sparsity_score",
    "depth_savings_ratio",
    "recursion_savings_ratio",
    "routing_expert_count",
    "routing_confidence_mean",
    "efficiency_multiple",
    "wikitext_score",
    "tinystories_score",
    "cross_task_score",
    "efficiency_wall_score",
    "max_viable_seq_len",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "loss_improvement_rate",
)

FINGERPRINT_BOOL_COLUMNS = (
    "screening_passed",
    "investigation_passed",
    "validation_passed",
    "scaling_gate_passed",
)

FINGERPRINT_UPDATE_COLUMNS = (
    "tier",
    "composite_score",
    "screening_loss_ratio",
    "screening_novelty",
    "screening_passed",
    "investigation_loss_ratio",
    "investigation_robustness",
    "investigation_passed",
    "validation_loss_ratio",
    "validation_baseline_ratio",
    "validation_multi_seed_std",
    "validation_passed",
    "discovery_loss_ratio",
    "loss_improvement_rate",
    "normalized_baseline_ratio",
    "param_efficiency",
    "quant_int8_retention",
    "quant_quality_per_byte",
    "robustness_long_ctx_score",
    "robustness_noise_score",
    "fp_jacobian_spectral_norm",
    "init_sensitivity_std",
    "scaling_param_efficiency",
    "scaling_flop_efficiency",
    "scaling_gate_passed",
    "scaling_d512_param_efficiency",
    "routing_savings_ratio",
    "compression_ratio",
    "activation_sparsity_score",
    "wikitext_perplexity",
    "wikitext_score",
    "tinystories_perplexity",
    "tinystories_score",
    "cross_task_score",
    "efficiency_wall_score",
    "max_viable_seq_len",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "depth_savings_ratio",
    "recursion_savings_ratio",
    "routing_expert_count",
    "routing_confidence_mean",
    "routing_drop_rate",
    "ncd_score",
    "efficiency_multiple",
    "timestamp",
)


def _best_min(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    try:
        return float(min(values))
    except (TypeError, ValueError):
        return None


def _best_max(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    try:
        return float(max(values))
    except (TypeError, ValueError):
        return None


def _best_bool(rows: List[Dict[str, Any]], key: str) -> Optional[int]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return int(any(bool(value) for value in values))


def _highest_tier(rows: List[Dict[str, Any]]) -> Optional[str]:
    tiers = [str(row.get("tier") or "").lower() for row in rows if row.get("tier")]
    if not tiers:
        return None
    return max(tiers, key=lambda tier: TIER_RANK.get(tier, -1))


def _tier_key(row: Dict[str, Any]) -> int:
    return int(TIER_RANK.get(str(row.get("tier") or "").lower(), -1))


def _tier_scope_rank(tier: str) -> int:
    tier_norm = str(tier or "").lower()
    if tier_norm.startswith("validation") or tier_norm == "breakthrough":
        return 3
    if tier_norm.startswith("investigation"):
        return 2
    return 1


def _has_stage_evidence(rows: List[Dict[str, Any]], columns: tuple[str, ...]) -> bool:
    for row in rows:
        for column in columns:
            if row.get(column) is not None:
                return True
    return False


def _has_real_validation_evidence(
    leaderboard_rows: List[Dict[str, Any]],
    program_rows: List[Dict[str, Any]],
) -> bool:
    if _has_stage_evidence(
        leaderboard_rows,
        (
            "validation_multi_seed_std",
            "validation_robustness_score",
            "validation_is_unstable",
            "normalized_baseline_ratio",
            "param_efficiency",
        ),
    ):
        return True
    if any(bool(row.get("validation_passed")) for row in leaderboard_rows):
        return True
    for row in program_rows:
        experiment_type = str(row.get("experiment_type") or "").lower()
        if experiment_type == "validation" and _has_stage_evidence(
            [row],
            (
                "validation_loss_ratio",
                "baseline_loss_ratio",
                "validation_multi_seed_std",
            ),
        ):
            return True
        if _has_stage_evidence(
            [row],
            (
                "validation_multi_seed_std",
                "validation_robustness_score",
                "validation_is_unstable",
                "normalized_baseline_ratio",
                "param_efficiency",
            ),
        ):
            return True
        if bool(row.get("validation_passed")):
            return True
    return False


def _has_real_investigation_evidence(
    leaderboard_rows: List[Dict[str, Any]],
    program_rows: List[Dict[str, Any]],
) -> bool:
    if _has_stage_evidence(
        leaderboard_rows,
        (
            "investigation_loss_ratio",
            "investigation_robustness",
        ),
    ):
        return True
    return any(
        str(row.get("experiment_type") or "").lower() == "investigation"
        and _has_stage_evidence([row], ("discovery_loss_ratio", "loss_ratio"))
        for row in program_rows
    )


def _effective_fingerprint_tier(
    leaderboard_rows: List[Dict[str, Any]],
    program_rows: List[Dict[str, Any]],
    merged: Dict[str, Any],
) -> str:
    tier = _highest_tier(leaderboard_rows) or "screening"
    if TIER_RANK.get(tier, -1) >= TIER_RANK.get("breakthrough", 4):
        # Recheck breakthrough gates against the merged row; rescreens that
        # add new program_results can drop composite below the floor or
        # reveal that capability signal never met the bar. Demote if so.
        from ..breakthrough_gates import passes_breakthrough_from_row

        passed, _ = passes_breakthrough_from_row(merged)
        if passed:
            return tier
        return "validation"

    has_validation = _has_real_validation_evidence(leaderboard_rows, program_rows)
    if has_validation:
        return "validation"

    has_investigation = _has_real_investigation_evidence(leaderboard_rows, program_rows)
    if has_investigation:
        if tier in {"investigation", "validation", "breakthrough"}:
            return tier
        if tier == "investigation_fingerprint_incomplete":
            return tier
        if bool(merged.get("investigation_passed")):
            return "investigation"
        robustness = merged.get("investigation_robustness")
        try:
            if robustness is not None and float(robustness) >= 0.5:
                return "investigation"
        except (TypeError, ValueError):
            pass
        return "investigation_failed"

    return "screening"


def _fingerprint_leaderboard_rows(nb, graph_fingerprint: str) -> List[Dict[str, Any]]:
    rows = nb.conn.execute(
        """
        SELECT l.*
        FROM leaderboard l
        JOIN program_results pr ON pr.result_id = l.result_id
        WHERE pr.graph_fingerprint = ?
        """,
        (graph_fingerprint,),
    ).fetchall()
    return [dict(row) for row in rows]


def _fingerprint_program_rows(nb, graph_fingerprint: str) -> List[Dict[str, Any]]:
    available_columns = nb._get_program_results_columns()
    select_columns = [
        column
        for column in FINGERPRINT_PROGRAM_RESULT_COLUMNS
        if column in available_columns
    ] or ["result_id"]
    rows = nb.conn.execute(
        f"""
        SELECT {", ".join(f"pr.{column}" for column in select_columns)},
               COALESCE(exp.experiment_type, '') AS experiment_type
        FROM program_results pr
        LEFT JOIN experiments exp ON exp.experiment_id = pr.experiment_id
        WHERE pr.graph_fingerprint = ?
        """,
        (graph_fingerprint,),
    ).fetchall()
    return [dict(row) for row in rows]


def sync_fingerprint_leaderboard(nb, result_id: str) -> None:
    fingerprint_row = nb.conn.execute(
        "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if not fingerprint_row or not fingerprint_row["graph_fingerprint"]:
        return
    graph_fingerprint = str(fingerprint_row["graph_fingerprint"])

    leaderboard_rows = _fingerprint_leaderboard_rows(nb, graph_fingerprint)
    if not leaderboard_rows:
        return
    program_rows = _fingerprint_program_rows(nb, graph_fingerprint)

    anchor = max(
        leaderboard_rows,
        key=lambda row: (
            _tier_key(row),
            _timestamp_key(row.get("timestamp")),
            float(row.get("composite_score") or -1e9),
        ),
    )
    merged = dict(anchor)
    combo_rows = leaderboard_rows + program_rows
    for column in FINGERPRINT_MIN_COLUMNS:
        value = _best_min(combo_rows, column)
        if value is not None:
            merged[column] = value
    for column in FINGERPRINT_MAX_COLUMNS:
        value = _best_max(combo_rows, column)
        if value is not None:
            merged[column] = value
    for column in FINGERPRINT_BOOL_COLUMNS:
        value = _best_bool(combo_rows, column)
        if value is not None:
            merged[column] = value

    screening_loss = _best_min(combo_rows, "screening_loss_ratio")
    if screening_loss is None:
        screening_loss = _best_min(program_rows, "loss_ratio")
    if screening_loss is not None:
        merged["screening_loss_ratio"] = screening_loss
    screening_novelty = _best_max(program_rows, "novelty_score")
    if screening_novelty is not None:
        merged["screening_novelty"] = screening_novelty

    tier = _effective_fingerprint_tier(leaderboard_rows, program_rows, merged)
    if tier:
        merged["tier"] = tier
        tier_rank = _tier_scope_rank(tier)
        if tier_rank >= 2 and merged.get("investigation_loss_ratio") is None:
            inv_rows = [
                row
                for row in program_rows
                if str(row.get("experiment_type") or "").lower() == "investigation"
            ]
            if inv_rows:
                merged["investigation_loss_ratio"] = _best_min(
                    inv_rows, "discovery_loss_ratio"
                )
                if merged.get("investigation_loss_ratio") is None:
                    try:
                        merged["investigation_loss_ratio"] = float(
                            min(
                                row.get("loss_ratio")
                                for row in inv_rows
                                if row.get("loss_ratio") is not None
                            )
                        )
                    except (TypeError, ValueError):
                        pass
        if tier_rank >= 3 and merged.get("validation_baseline_ratio") is None:
            val_rows = [
                row
                for row in program_rows
                if str(row.get("experiment_type") or "").lower() == "validation"
                or row.get("validation_loss_ratio") is not None
            ]
            if val_rows:
                merged["validation_baseline_ratio"] = _best_min(
                    val_rows, "baseline_loss_ratio"
                )
        if tier_rank < 3:
            merged["validation_multi_seed_std"] = None
            merged["validation_passed"] = 0
        if tier_rank < 2:
            merged["investigation_loss_ratio"] = None
            merged["investigation_robustness"] = None
            merged["investigation_passed"] = 0

    composite_score = compute_composite(
        **build_score_kwargs(
            nb.conn,
            nb,
            str(anchor.get("result_id") or result_id),
            merged,
            bool(merged.get("is_reference")),
        )
    )

    update_columns = [
        column
        for column in FINGERPRINT_UPDATE_COLUMNS
        if column in nb._get_leaderboard_columns()
    ]
    assignments = ", ".join(f"{column} = ?" for column in update_columns)
    update_template: List[Any] = []
    timestamp_now = time.time()
    for column in update_columns:
        if column == "composite_score":
            update_template.append(composite_score)
        elif column == "timestamp":
            update_template.append(timestamp_now)
        else:
            value = merged.get(column)
            update_template.append(int(value) if isinstance(value, bool) else value)

    for row in leaderboard_rows:
        nb.conn.execute(
            f"UPDATE leaderboard SET {assignments} WHERE entry_id = ?",
            [*update_template, row["entry_id"]],
        )


def leaderboard_consistency_report(nb) -> Dict[str, Any]:
    screening_modes = (
        "synthesis",
        "novelty",
        "evolution",
        "reference",
        "backfill",
        "forced_exploration",
        "ablation",
    )
    screening_placeholders = ",".join("?" for _ in screening_modes)

    def count_rows(sql: str, params: tuple[Any, ...] = ()) -> int:
        row = nb.conn.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0

    stage1_rows = nb.conn.execute(
        """
        SELECT
            p.result_id,
            COALESCE(e.experiment_type, 'unknown') AS experiment_type,
            EXISTS(SELECT 1 FROM leaderboard l WHERE l.result_id = p.result_id) AS has_direct_leaderboard,
            EXISTS(
                SELECT 1
                FROM leaderboard l
                JOIN program_results pr2 ON pr2.result_id = l.result_id
                WHERE pr2.graph_fingerprint = p.graph_fingerprint
            ) AS has_fingerprint_leaderboard
        FROM program_results p
        LEFT JOIN experiments e ON e.experiment_id = p.experiment_id
        WHERE p.stage1_passed = 1
        """
    ).fetchall()

    by_experiment_type: Dict[str, Dict[str, int]] = {}
    direct_covered = 0
    fingerprint_covered = 0
    descendant_only_ids: List[str] = []
    missing_screening_ids: List[str] = []
    missing_other_ids: List[str] = []

    for raw_row in stage1_rows:
        row = dict(raw_row)
        mode = str(row["experiment_type"] or "unknown")
        bucket = by_experiment_type.setdefault(
            mode,
            {
                "stage1_rows": 0,
                "direct_leaderboard_rows": 0,
                "fingerprint_covered_rows": 0,
                "uncovered_rows": 0,
            },
        )
        bucket["stage1_rows"] += 1

        has_direct = bool(row["has_direct_leaderboard"])
        has_fingerprint = bool(row["has_fingerprint_leaderboard"])
        if has_direct:
            direct_covered += 1
            bucket["direct_leaderboard_rows"] += 1
        if has_fingerprint:
            fingerprint_covered += 1
            bucket["fingerprint_covered_rows"] += 1

        if has_direct or has_fingerprint:
            if has_fingerprint and not has_direct:
                descendant_only_ids.append(str(row["result_id"]))
            continue

        bucket["uncovered_rows"] += 1
        if mode in screening_modes:
            missing_screening_ids.append(str(row["result_id"]))
        else:
            missing_other_ids.append(str(row["result_id"]))

    missing_screening_rows = count_rows(
        f"""
        SELECT COUNT(*)
        FROM program_results p
        JOIN experiments e ON e.experiment_id = p.experiment_id
        WHERE p.stage1_passed = 1
          AND e.experiment_type IN ({screening_placeholders})
          AND NOT EXISTS (SELECT 1 FROM leaderboard l WHERE l.result_id = p.result_id)
          AND NOT EXISTS (
                SELECT 1
                FROM leaderboard l
                JOIN program_results pr2 ON pr2.result_id = l.result_id
                WHERE pr2.graph_fingerprint = p.graph_fingerprint
          )
        """,
        screening_modes,
    )
    orphan_ids = [
        str(row["result_id"])
        for row in nb.conn.execute(
            """
            SELECT l.result_id
            FROM leaderboard l
            LEFT JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.result_id IS NULL
            ORDER BY l.timestamp DESC
            LIMIT 20
            """
        ).fetchall()
    ]

    return {
        "stage1_program_rows": count_rows(
            "SELECT COUNT(*) FROM program_results WHERE stage1_passed = 1"
        ),
        "leaderboard_rows": count_rows("SELECT COUNT(*) FROM leaderboard"),
        "direct_stage1_leaderboard_rows": direct_covered,
        "fingerprint_covered_stage1_rows": fingerprint_covered,
        "descendant_stage1_rows_without_direct_entry": len(descendant_only_ids),
        "missing_screening_leaderboard_rows": missing_screening_rows,
        "missing_non_screening_leaderboard_rows": len(missing_other_ids),
        "orphan_leaderboard_rows": count_rows(
            """
            SELECT COUNT(*)
            FROM leaderboard l
            LEFT JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.result_id IS NULL
            """
        ),
        "non_stage1_leaderboard_rows": count_rows(
            """
            SELECT COUNT(*)
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE COALESCE(pr.stage1_passed, 0) != 1
            """
        ),
        "by_experiment_type": by_experiment_type,
        "samples": {
            "missing_screening_result_ids": missing_screening_ids[:20],
            "missing_non_screening_result_ids": missing_other_ids[:20],
            "descendant_result_ids": descendant_only_ids[:20],
            "orphan_leaderboard_result_ids": orphan_ids,
        },
    }
