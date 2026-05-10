#!/usr/bin/env python3
"""Merge non-null probe metrics across cross-experiment fingerprint siblings.

For every ``graph_fingerprint`` that appears in 2+ unintentional experiments
(excluding ``exact_graph_replay``, ``validation``, ``reference``,
``reference_registration``, ``backfill`` — per the governance endpoint policy),
pick the canonical row using the same ``_best_rank`` ordering as
``ml_corpus.py::load_deduped_screening_predictor_rows`` and merge NULL fields
on the canonical row from non-NULL values on the siblings.

Pre-merge canonical state is written to ``program_results_cross_exp_merge_backup``
with an audit trail (``merged_from_result_ids``, ``merged_columns``,
``merged_at``). No rows are deleted.

Only probe metrics / evaluation outputs are merged. Per-run identity,
training-dynamics, novelty scoring, telemetry, and error columns are never
touched — those belong to the run that produced them.

Default mode is a dry-run that prints a group-level diff report. Use
``--apply`` to execute against the writer; the writer flock must be free.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from research.tools._db_maintenance import (
    connect_readonly,
    connect_writer,
    ensure_backup_table,
    quoted_columns,
    table_columns,
    table_row_count,
)

DEFAULT_DB = Path("research/runs.db")
BACKUP_TABLE = "program_results_cross_exp_merge_backup"

# Experiment types where re-evaluation of the same fingerprint is INTENTIONAL
# and the duplicate should be left alone. Matches the governance endpoint's
# exclusion list (api_routes/observability_bp.py).
INTENTIONAL_EXPERIMENT_TYPES = (
    "exact_graph_replay",
    "validation",
    "reference",
    "reference_registration",
    "backfill",
)

# Column families that are safe to merge (probe metrics, evaluation outputs
# where non-NULL measurements from any run of the same graph are equally valid
# to promote onto the canonical row). Keep this list explicit — adding a
# column requires deliberate review.
MERGE_COLUMNS_BY_FAMILY: Dict[str, Tuple[str, ...]] = {
    "v2_probes": (
        "induction_intermediate_auc",
        "induction_intermediate_max_gap_acc",
        "induction_intermediate_gap_accuracies_json",
        "induction_intermediate_steps_trained",
        "induction_intermediate_status",
        "induction_intermediate_elapsed_ms",
        "induction_intermediate_protocol_version",
        "binding_intermediate_auc",
        "binding_intermediate_max_distance_acc",
        "binding_intermediate_distance_accuracies_json",
        "binding_intermediate_gap_accuracies_json",
        "binding_intermediate_train_steps",
        "binding_intermediate_steps_trained",
        "binding_intermediate_status",
        "binding_intermediate_elapsed_ms",
        "binding_intermediate_protocol_version",
    ),
    "v1_probes": (
        "induction_screening_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "ar_legacy_auc",
        "ar_legacy_final_acc",
        "ar_legacy_above_chance",
        "ar_legacy_timed_out",
        "induction_screening_gap_accuracies_json",
        "induction_screening_train_steps",
        "induction_screening_eval_examples",
        "induction_screening_batch_size",
        "induction_screening_gaps_json",
        "induction_screening_elapsed_ms",
        "induction_screening_metric_version",
        "induction_screening_speed_mode",
        "induction_screening_pool_size",
        "binding_screening_distance_accuracies_json",
        "binding_screening_eval_examples",
        "binding_screening_distances_json",
        "binding_screening_elapsed_ms",
        "binding_curriculum_auc",
        "binding_curriculum_distance_accuracies_json",
        "binding_curriculum_steps",
        "binding_curriculum_elapsed_ms",
        "binding_curriculum_protocol_version",
    ),
    "language": (
        "hellaswag_acc",
        "hellaswag_status",
        "hellaswag_n_examples",
        "hellaswag_metric_version",
        "hellaswag_tokenizer_mode",
        "hellaswag_tiktoken_encoding",
        "screening_hellaswag_correct",
        "screening_hellaswag_total",
        "screening_hellaswag_elapsed_ms",
        "blimp_overall_accuracy",
        "blimp_subtask_accuracies_json",
        "blimp_n_subtasks",
        "blimp_status",
    ),
    "wikitext": (
        "wikitext_perplexity",
        "wikitext_score",
        "wikitext_ppl_200",
        "wikitext_ppl_500",
        "wikitext_pre_perplexity",
        "wikitext_ppl_improvement",
        "wikitext_improvement_ratio",
        "wikitext_eval_steps",
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
        "screening_wikitext_variant",
        "screening_wikitext_elapsed_ms",
    ),
    "validation": (
        "validation_loss",
        "validation_loss_ratio",
        "validation_baseline_ratio",
        "generalization_gap",
        "validation_robustness_score",
        "validation_is_unstable",
    ),
    "rapid_screening": (
        "rapid_screening_passed",
        "rapid_screening_elapsed_ms",
        "rapid_screening_steps_completed",
        "rapid_screening_max_steps",
        "rapid_screening_degraded",
        "rapid_screening_degraded_reasons_json",
        "rapid_screening_kill_reason",
        "rapid_screening_kill_step",
        "rapid_screening_kill_metric",
        "rapid_screening_gpu_minutes_saved",
        "rapid_screening_metrics_json",
    ),
    "robustness": (
        "robustness_long_ctx_scaling_score",
        "robustness_long_ctx_assoc_score",
        "robustness_long_ctx_multi_hop_score",
        "robustness_long_ctx_passkey_score",
        "robustness_long_ctx_retrieval_aggregate",
        "robustness_long_ctx_combined_score",
    ),
    "routing_fast_lane": (
        "routing_fast_lane_applied",
        "routing_fast_lane_status",
        "routing_fast_lane_metric_version",
        "routing_fast_lane_perplexity",
        "routing_fast_lane_score",
        "routing_fast_lane_pre_perplexity",
        "routing_fast_lane_ppl_improvement",
        "routing_fast_lane_elapsed_ms",
        "routing_fast_lane_budget_json",
        "routing_fast_lane_slope",
        "routing_fast_lane_slope_consistent",
        "routing_fast_lane_routing_ops_json",
    ),
    "modern_ar_binding": (
        "ar_gate_metric_version",
        "ar_gate_in_dist_pair_acc",
        "ar_gate_in_dist_class_acc",
        "ar_gate_held_pair_acc",
        "ar_gate_held_class_acc",
        "ar_gate_score",
        "ar_gate_status",
        "ar_gate_elapsed_ms",
        "ar_gate_train_steps_done",
        "ar_gate_no_go",
        "ar_intermediate_metric_version",
        "ar_intermediate_train_pair_acc",
        "ar_intermediate_held_pair_acc",
        "ar_intermediate_held_class_acc",
        "ar_intermediate_pair_chance_acc",
        "ar_intermediate_class_chance_acc",
        "ar_intermediate_held_pair_lift",
        "ar_intermediate_held_class_lift",
        "ar_intermediate_early_held_pair_acc",
        "ar_intermediate_final_held_pair_acc",
        "ar_intermediate_best_held_pair_acc",
        "ar_intermediate_improvement",
        "ar_intermediate_slope_per_100_steps",
        "ar_intermediate_auc",
        "ar_intermediate_auc_lift",
        "ar_intermediate_learning_curve_json",
        "ar_intermediate_steps_to_threshold",
        "ar_intermediate_diagnostic_score",
        "ar_intermediate_steps_trained",
        "ar_intermediate_status",
        "ar_intermediate_elapsed_ms",
        "ar_intermediate_error",
        "ar_curriculum_metric_version",
        "ar_curriculum_auc_pair_final",
        "ar_curriculum_auc_class_final",
        "ar_curriculum_s0_held_pair_acc",
        "ar_curriculum_s0_retention",
        "ar_curriculum_max_passing_stage",
        "ar_curriculum_per_stage_held_pair_acc",
        "ar_curriculum_per_stage_held_class_acc",
        "ar_curriculum_per_stage_lift_pair",
        "ar_curriculum_per_stage_z_score_pair",
        "ar_curriculum_per_stage_chance_pair",
        "ar_curriculum_learning_curve_json",
        "ar_curriculum_steps_trained",
        "ar_curriculum_n_eval_examples",
        "ar_curriculum_mode",
        "ar_curriculum_elapsed_ms",
        "ar_curriculum_status",
        "ar_curriculum_error",
        "binding_multislot_metric_version",
        "binding_multislot_train_slot_acc",
        "binding_multislot_held_entity_slot_acc",
        "binding_multislot_held_entity_class_acc",
        "binding_multislot_two_plus_slots_acc",
        "binding_multislot_all_slots_acc",
        "binding_multislot_mixed_query_acc",
        "binding_multislot_mixed_two_plus_slots_acc",
        "binding_multislot_mixed_all_slots_acc",
        "binding_multislot_slot_chance_acc",
        "binding_multislot_class_chance_acc",
        "binding_multislot_two_plus_slots_chance_acc",
        "binding_multislot_all_slots_chance_acc",
        "binding_multislot_held_slot_lift",
        "binding_multislot_held_class_lift",
        "binding_multislot_two_plus_slots_lift",
        "binding_multislot_all_slots_lift",
        "binding_multislot_mixed_query_lift",
        "binding_multislot_mixed_two_plus_slots_lift",
        "binding_multislot_mixed_all_slots_lift",
        "binding_multislot_early_slot_acc",
        "binding_multislot_final_slot_acc",
        "binding_multislot_best_slot_acc",
        "binding_multislot_improvement",
        "binding_multislot_slope_per_100_steps",
        "binding_multislot_auc",
        "binding_multislot_auc_lift",
        "binding_multislot_learning_curve_json",
        "binding_multislot_steps_to_threshold",
        "binding_multislot_diagnostic_score",
        "binding_multislot_steps_trained",
        "binding_multislot_status",
        "binding_multislot_elapsed_ms",
        "binding_multislot_error",
        "induction_validation_auc",
        "induction_validation_max_gap_acc",
        "induction_validation_gap_accuracy_cv",
        "induction_validation_gap_accuracies_json",
        "induction_validation_steps_trained",
        "induction_validation_status",
        "induction_validation_elapsed_ms",
        "induction_validation_protocol_version",
        "ar_validation_metric_version",
        "ar_validation_final_acc",
        "ar_validation_held_pair_acc",
        "ar_validation_held_class_acc",
        "ar_validation_learning_curve_json",
        "ar_validation_steps_to_floor",
        "ar_validation_rank_score",
        "ar_validation_status",
        "ar_validation_elapsed_ms",
        "champion_floor_protocol_version",
        "champion_steps_to_floor",
        "champion_floor_loss",
        "champion_floor_ppl",
        "champion_floor_loss_std",
        "champion_plateau_detected_step",
        "champion_plateau_window",
        "champion_baseline_result_id",
        "champion_baseline_layers",
        "champion_baseline_protocol_version",
        "champion_steps_to_floor_score",
        "champion_floor_quality_score",
        "champion_floor_stability_score",
        "champion_induction_validation_score",
        "champion_binding_long_context_score",
        "champion_ar_validation_score",
        "champion_tiny_model_score",
        "champion_tiny_model_protocol_version",
        "champion_hard_failure_reason",
    ),
    "diagnostics": (
        "diagnostic_tasks_json",
        "diagnostic_score",
        "tinystories_perplexity",
        "tinystories_score",
        "cross_task_score",
        "ncd_score",
        "ncd_description_length",
        "ncd_description_length_per_param",
        "judgment_score",
    ),
}

ALL_FAMILY_NAMES = tuple(MERGE_COLUMNS_BY_FAMILY.keys())


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    return (
        "program_results_compat"
        if _table_exists(conn, "program_results_compat")
        else "program_results"
    )


def _program_result_write_tables(conn: sqlite3.Connection) -> list[str]:
    tables = ["program_results"]
    if _table_exists(conn, "graph_runs"):
        tables.append("graph_runs")
    return tables


def _resolve_columns(family_filter: Optional[Iterable[str]]) -> List[str]:
    if family_filter:
        unknown = set(family_filter) - set(MERGE_COLUMNS_BY_FAMILY)
        if unknown:
            raise ValueError(
                f"Unknown column families: {sorted(unknown)}. "
                f"Valid: {sorted(MERGE_COLUMNS_BY_FAMILY)}"
            )
        families = family_filter
    else:
        families = ALL_FAMILY_NAMES
    cols: List[str] = []
    for fam in families:
        cols.extend(MERGE_COLUMNS_BY_FAMILY[fam])
    return cols


def _best_rank(row: sqlite3.Row) -> Tuple[Any, ...]:
    """Parity with ml_corpus.py:839-845.

    Lower tuple wins. Preference: trusted_positive > not-runtime-negative >
    has_loss_ratio > min(loss_ratio) > min(timestamp).
    """
    stage1_passed = bool(row["stage1_passed"])
    trust_label = row["trust_label"] or ""
    is_trusted_positive = stage1_passed and trust_label not in (
        "exploratory",
        "runtime_observation",
        "backfill_observation",
        "replay_observation",
    )
    is_runtime_negative = trust_label == "runtime_observation" and not stage1_passed

    # data_provenance_json can promote/demote based on screening_model_training_role
    raw_payload = row["data_provenance_json"]
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            payload = json.loads(raw_payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            screening_role = str(
                payload.get("screening_model_training_role") or ""
            ).strip()
            if screening_role == "negative":
                is_runtime_negative = True
            elif screening_role == "positive":
                is_trusted_positive = True

    loss_ratio = row["loss_ratio"]
    return (
        0 if is_trusted_positive else 1,
        0 if is_runtime_negative else 1,
        loss_ratio is None,
        float(loss_ratio) if loss_ratio is not None else float("inf"),
        float(row["timestamp"] or 0.0),
    )


def _fetch_unintended_groups(
    conn: sqlite3.Connection, fingerprint: Optional[str]
) -> Dict[str, List[sqlite3.Row]]:
    pr_table = _program_results_read_table(conn)
    placeholders = ",".join("?" for _ in INTENTIONAL_EXPERIMENT_TYPES)
    where_fp = "AND pr.graph_fingerprint = ?" if fingerprint else ""
    params: List[Any] = list(INTENTIONAL_EXPERIMENT_TYPES)
    if fingerprint:
        params.append(fingerprint)
    rows = conn.execute(
        f"""
        SELECT pr.*
        FROM {pr_table} pr
        JOIN experiments e ON pr.experiment_id = e.experiment_id
        WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
          AND e.experiment_type NOT IN ({placeholders})
          {where_fp}
          AND pr.graph_fingerprint IN (
            SELECT pr2.graph_fingerprint
            FROM {pr_table} pr2
            JOIN experiments e2 ON pr2.experiment_id = e2.experiment_id
            WHERE TRIM(COALESCE(pr2.graph_fingerprint, '')) <> ''
              AND e2.experiment_type NOT IN ({placeholders})
            GROUP BY pr2.graph_fingerprint
            HAVING COUNT(DISTINCT pr2.experiment_id) > 1
          )
        ORDER BY pr.graph_fingerprint, pr.timestamp DESC
        """,
        params + list(INTENTIONAL_EXPERIMENT_TYPES),
    ).fetchall()
    groups: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault(str(r["graph_fingerprint"]), []).append(r)
    return groups


def _plan_group(
    rows: List[sqlite3.Row], merge_columns: List[str]
) -> Optional[Dict[str, Any]]:
    """Return None when the group has a single row (no merge needed)."""
    if len(rows) < 2:
        return None
    ranked = sorted(rows, key=_best_rank)
    canonical = dict(ranked[0])
    siblings = [dict(r) for r in ranked[1:]]
    # siblings are sorted worst-rank-last within `ranked`; but we want the
    # most-recent-timestamp sibling among non-NULL. Sort siblings by timestamp DESC.
    siblings_by_ts = sorted(
        siblings, key=lambda r: float(r.get("timestamp") or 0.0), reverse=True
    )
    updates: Dict[str, Any] = {}
    contributing: Dict[str, str] = {}  # column → contributing result_id
    for col in merge_columns:
        if canonical.get(col) is not None:
            continue
        for sib in siblings_by_ts:
            val = sib.get(col)
            if val is not None:
                updates[col] = val
                contributing[col] = str(sib.get("result_id"))
                break
    if not updates:
        return None
    return {
        "graph_fingerprint": canonical.get("graph_fingerprint"),
        "canonical_result_id": str(canonical.get("result_id")),
        "canonical_experiment_id": str(canonical.get("experiment_id") or ""),
        "n_siblings": len(siblings),
        "merge_updates": updates,
        "contributing_result_ids": contributing,
    }


def _ensure_backup_table(conn: sqlite3.Connection) -> None:
    ensure_backup_table(
        conn,
        backup_table=BACKUP_TABLE,
        source_table="program_results",
        extra_columns=(
            ("merged_at", "REAL"),
            ("merged_from_result_ids", "TEXT"),
            ("merged_columns", "TEXT"),
        ),
        indexes=(
            (f"idx_{BACKUP_TABLE}_result_id", ("result_id",)),
            (f"idx_{BACKUP_TABLE}_merged_at", ("merged_at",)),
        ),
    )


def _apply_plans(
    conn: sqlite3.Connection, plans: List[Dict[str, Any]], merge_columns: List[str]
) -> None:
    col_set = set(merge_columns)
    now = time.time()
    _ensure_backup_table(conn)
    pr_table = _program_results_read_table(conn)
    pr_cols = table_columns(conn, "program_results")
    backup_cols = pr_cols + ["merged_at", "merged_from_result_ids", "merged_columns"]
    with conn:
        for plan in plans:
            canonical_id = plan["canonical_result_id"]
            updates = plan["merge_updates"]
            if not updates or not any(k in col_set for k in updates):
                continue
            current_row = conn.execute(
                f"SELECT * FROM {pr_table} WHERE result_id = ?", (canonical_id,)
            ).fetchone()
            if current_row is None:
                continue
            backup_values = [current_row[c] for c in pr_cols] + [
                now,
                json.dumps(sorted(set(plan["contributing_result_ids"].values()))),
                json.dumps(sorted(updates.keys())),
            ]
            placeholders = ",".join("?" for _ in backup_cols)
            quoted_cols = quoted_columns(backup_cols)
            conn.execute(
                f"INSERT INTO {BACKUP_TABLE} ({quoted_cols}) VALUES ({placeholders})",
                backup_values,
            )
            set_parts = ",".join(f'"{c}" = ?' for c in updates)
            vals = list(updates.values()) + [canonical_id]
            for table_name in _program_result_write_tables(conn):
                conn.execute(
                    f"UPDATE {table_name} SET {set_parts} WHERE result_id = ?", vals
                )


def run(
    db_path: Path,
    *,
    apply: bool,
    fingerprint: Optional[str],
    families: Optional[List[str]],
    limit_groups: Optional[int],
) -> int:
    merge_columns = _resolve_columns(families)
    read_conn = connect_readonly(db_path)
    try:
        groups = _fetch_unintended_groups(read_conn, fingerprint)
    finally:
        read_conn.close()

    plans: List[Dict[str, Any]] = []
    field_counter: Counter[str] = Counter()
    for fp, rows in groups.items():
        plan = _plan_group(rows, merge_columns)
        if plan is None:
            continue
        plans.append(plan)
        field_counter.update(plan["merge_updates"].keys())
    plans.sort(key=lambda p: -len(p["merge_updates"]))
    if limit_groups is not None:
        plans = plans[:limit_groups]

    total_groups = len(plans)
    total_field_values = sum(len(p["merge_updates"]) for p in plans)
    print(
        f"Plan: {total_groups} fingerprints, "
        f"{total_field_values} field-values to merge onto canonical rows."
    )
    if field_counter:
        print("\nTop merged columns:")
        for col, n in field_counter.most_common(25):
            print(f"  {col}: {n}")

    print("\nTop 10 fingerprints by merge volume:")
    for plan in plans[:10]:
        merged_keys = sorted(plan["merge_updates"].keys())[:6]
        tail = "..." if len(plan["merge_updates"]) > 6 else ""
        summary = ", ".join(merged_keys) + tail
        print(
            f"  fp={plan['graph_fingerprint'][:16]} "
            f"kept={plan['canonical_result_id'][:12]} "
            f"exp={plan['canonical_experiment_id'][:12]} "
            f"siblings={plan['n_siblings']} "
            f"merge={len(plan['merge_updates'])} ({summary})"
        )

    if not apply:
        print("\nDry-run only. Re-run with --apply to execute against the writer.")
        return 0

    from research.tools._db_maintenance import check_writer_lock

    check_writer_lock(Path(f"{db_path}.writer-lock"))
    print(f"\nApplying merges to {db_path}...")
    write_conn = connect_writer(db_path)
    try:
        _apply_plans(write_conn, plans, merge_columns)
        backup_count = table_row_count(write_conn, BACKUP_TABLE)
        print(
            f"Done. Merged {total_field_values} field-values across "
            f"{total_groups} fingerprints. Backup now has {backup_count} rows."
        )
    finally:
        write_conn.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="Execute merges")
    parser.add_argument(
        "--fingerprint",
        default=None,
        help="Scope to a single graph_fingerprint (audit one case)",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated column families to merge. Default: all. "
        f"Valid: {','.join(ALL_FAMILY_NAMES)}",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Cap to the top N merge-volume groups (for targeted passes)",
    )
    args = parser.parse_args(argv)

    families = (
        [f.strip() for f in args.columns.split(",") if f.strip()]
        if args.columns
        else None
    )
    try:
        return run(
            args.db,
            apply=args.apply,
            fingerprint=args.fingerprint,
            families=families,
            limit_groups=args.limit_groups,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
