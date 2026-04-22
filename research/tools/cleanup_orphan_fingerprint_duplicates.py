#!/usr/bin/env python3
"""Merge and delete duplicate fingerprint rows that never reached the leaderboard.

Scope:
- Only touches ``program_results`` groups with 2+ rows for the same
  ``graph_fingerprint``.
- Skips fingerprints that already have a leaderboard row, so promoted
  candidates stay under the existing fingerprint-governance path.
- Excludes reference-registration style experiment types that should remain
  isolated from runtime candidate cleanup.

For each eligible fingerprint:
1. Pick a canonical keeper row.
2. Merge the best available metric values onto the keeper.
3. Move auxiliary ``result_id``-keyed rows to the keeper.
4. Delete duplicate ``program_results`` siblings.
5. If the group looks like a failed runtime row polluted by post-screening
   metrics, relabel the surviving row as ``backfill`` provenance so the UI
   shows it as backfill rather than a screened candidate.

Default mode is dry-run. Use ``--apply`` to execute against the writer.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from research.tools._db_maintenance import (
    DEFAULT_WRITER_LOCK,
    check_writer_lock,
    connect_readonly,
    connect_writer,
    ensure_backup_table,
    quoted_columns,
    table_columns,
    table_row_count,
)
from research.scientist.shared_utils import coerce_finite_float as _safe_float

DEFAULT_DB = Path("research/lab_notebook.db")
BACKUP_TABLE = "program_results_orphan_fingerprint_cleanup_backup"

INTENTIONAL_EXPERIMENT_TYPES = (
    "reference",
    "reference_registration",
)

IDENTITY_COLUMNS = {
    "result_id",
    "experiment_id",
    "graph_fingerprint",
    "timestamp",
}

POST_SCREENING_SIGNAL_COLUMNS = (
    "rapid_screening_passed",
    "wikitext_perplexity",
    "wikitext_score",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "induction_v2_investigation_auc",
    "binding_v2_investigation_auc",
    "discovery_loss_ratio",
    "validation_loss_ratio",
)

HIGHER_BETTER_COLUMNS = {
    "novelty_score",
    "structural_novelty",
    "behavioral_novelty",
    "novelty_confidence",
    "throughput_tok_s",
    "stability_score",
    "loss_improvement_rate",
    "screening_slope",
    "activation_sparsity_score",
    "routing_confidence_mean",
    "routing_utilization_entropy",
    "routing_savings_ratio",
    "compression_ratio",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "ar_auc",
    "ar_final_acc",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "validation_robustness_score",
    "wikitext_score",
    "tinystories_score",
    "cross_task_score",
    "diagnostic_score",
    "judgment_score",
    "ncd_score",
    "efficiency_multiple",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "induction_v2_investigation_auc",
    "induction_v2_investigation_max_gap_acc",
    "binding_v2_investigation_auc",
    "binding_v2_investigation_max_distance_acc",
}

LOWER_BETTER_COLUMNS = {
    "loss_ratio",
    "final_loss",
    "discovery_loss",
    "discovery_loss_ratio",
    "validation_loss",
    "validation_loss_ratio",
    "generalization_gap",
    "baseline_loss_ratio",
    "wikitext_perplexity",
    "wikitext_pre_perplexity",
    "wikitext_ppl_200",
    "wikitext_ppl_500",
    "tinystories_perplexity",
    "ncd_description_length",
    "ncd_description_length_per_param",
    "fp_jacobian_spectral_norm",
    "peak_memory_mb",
    "compile_time_ms",
    "forward_time_ms",
    "backward_time_ms",
}

MAX_COLUMNS = {
    "stage0_passed",
    "stage05_passed",
    "stage1_passed",
    "rapid_screening_passed",
    "validation_passed",
    "validation_is_unstable",
    "extreme_input_passed",
    "random_input_passed",
    "has_zero_grad",
    "n_train_steps",
    "train_budget_steps",
    "rapid_screening_steps_completed",
    "rapid_screening_max_steps",
    "routing_tokens_total",
    "routing_tokens_processed",
    "routing_tokens_skipped",
    "routing_capacity_overflow_count",
    "routing_expert_count",
    "graph_n_ops",
    "graph_depth",
    "graph_n_edges",
    "graph_n_unique_ops",
    "max_viable_seq_len",
    "hellaswag_n_examples",
    "blimp_n_subtasks",
    "induction_probe_train_steps",
    "binding_probe_eval_examples",
    "screening_hellaswag_correct",
    "screening_hellaswag_total",
    "induction_v2_investigation_steps_trained",
    "binding_v2_investigation_train_steps",
}

RELABEL_COLUMNS = {
    "result_cohort": "backfill",
    "trust_label": "backfill_observation",
    "comparability_label": "reconstructed_init_variant",
    "evaluation_protocol_version": "backfill_replay_v1",
    "init_regime": "reconstructed_fresh_init",
}

AUX_RESULT_ID_TABLES = (
    "autonomous_actions",
    "induction_metrics_archive",
    "induction_metrics_v2",
    "program_graph_features",
    "program_graph_ops",
    "program_graph_pairs",
    "repair_log",
    "training_curves",
)


def _mode_label(mode: str) -> str:
    return "single-leaderboard" if mode == "single-lb" else "orphan"


def _has_post_screening_signal(row: Dict[str, Any]) -> bool:
    return any(row.get(col) is not None for col in POST_SCREENING_SIGNAL_COLUMNS)


def _metric_density(row: Dict[str, Any]) -> int:
    density_cols = ("loss_ratio", "novelty_score", *POST_SCREENING_SIGNAL_COLUMNS)
    return sum(1 for col in density_cols if row.get(col) is not None)


def _aux_ref_counts(
    conn: sqlite3.Connection, result_ids: Sequence[str]
) -> Dict[str, int]:
    counts = {rid: 0 for rid in result_ids}
    if not result_ids:
        return counts
    placeholders = ",".join("?" for _ in result_ids)
    for table_name in AUX_RESULT_ID_TABLES:
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if existing is None:
            continue
        rows = conn.execute(
            f"""
            SELECT result_id, COUNT(*) AS n
            FROM {table_name}
            WHERE result_id IN ({placeholders})
            GROUP BY result_id
            """,
            tuple(result_ids),
        ).fetchall()
        for row in rows:
            counts[str(row["result_id"])] += int(row["n"] or 0)
    return counts


def _keeper_key(row: Dict[str, Any], aux_counts: Dict[str, int]) -> tuple[Any, ...]:
    loss_ratio = _safe_float(row.get("loss_ratio"))
    return (
        0 if bool(row.get("stage1_passed")) else 1,
        0 if loss_ratio is not None else 1,
        loss_ratio if loss_ratio is not None else float("inf"),
        -float(row.get("timestamp") or 0.0),
        -_metric_density(row),
        0 if aux_counts.get(str(row.get("result_id") or ""), 0) > 0 else 1,
    )


def _pick_best_value(column: str, current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if column in HIGHER_BETTER_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f > current_f else current
    if column in LOWER_BETTER_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f < current_f else current
    if column in MAX_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f > current_f else current
    return current


def _fetch_eligible_groups(
    conn: sqlite3.Connection, fingerprint: Optional[str], mode: str
) -> Dict[str, List[Dict[str, Any]]]:
    placeholders = ",".join("?" for _ in INTENTIONAL_EXPERIMENT_TYPES)
    where_fp = "AND pr.graph_fingerprint = ?" if fingerprint else ""
    params: List[Any] = list(INTENTIONAL_EXPERIMENT_TYPES)
    if fingerprint:
        params.append(fingerprint)
    if mode == "single-lb":
        lb_having = "AND SUM(CASE WHEN l2.result_id IS NOT NULL THEN 1 ELSE 0 END) = 1"
    else:
        lb_having = "AND SUM(CASE WHEN l2.result_id IS NOT NULL THEN 1 ELSE 0 END) = 0"
    rows = conn.execute(
        f"""
        SELECT pr.*, e.experiment_type,
               CASE WHEN l.result_id IS NOT NULL THEN 1 ELSE 0 END AS has_leaderboard,
               l.tier AS leaderboard_tier
        FROM program_results pr
        LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
          AND (
            e.experiment_type IS NULL
            OR e.experiment_type NOT IN ({placeholders})
          )
          {where_fp}
          AND pr.graph_fingerprint IN (
            SELECT pr2.graph_fingerprint
            FROM program_results pr2
            LEFT JOIN experiments e2 ON e2.experiment_id = pr2.experiment_id
            LEFT JOIN leaderboard l2 ON l2.result_id = pr2.result_id
            WHERE TRIM(COALESCE(pr2.graph_fingerprint, '')) <> ''
              AND (
                e2.experiment_type IS NULL
                OR e2.experiment_type NOT IN ({placeholders})
              )
            GROUP BY pr2.graph_fingerprint
            HAVING COUNT(*) > 1
               {lb_having}
          )
        ORDER BY pr.graph_fingerprint, pr.timestamp DESC
        """,
        params + list(INTENTIONAL_EXPERIMENT_TYPES),
    ).fetchall()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["graph_fingerprint"]), []).append(dict(row))
    return groups


def _orphan_backfill_relabel_candidates(
    conn: sqlite3.Connection, fingerprint: Optional[str] = None
) -> List[sqlite3.Row]:
    where_fp = "AND pr.graph_fingerprint = ?" if fingerprint else ""
    params = (fingerprint,) if fingerprint else ()
    return conn.execute(
        f"""
        SELECT pr.*
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE l.result_id IS NULL
          {where_fp}
          AND COALESCE(pr.stage1_passed, 0) = 0
          AND (
            {" OR ".join(f"pr.{col} IS NOT NULL" for col in POST_SCREENING_SIGNAL_COLUMNS)}
          )
          AND (
            COALESCE(pr.result_cohort, '') <> 'backfill'
            OR COALESCE(pr.trust_label, '') <> 'backfill_observation'
            OR COALESCE(pr.comparability_label, '') <> 'reconstructed_init_variant'
          )
        """,
        params,
    ).fetchall()


def _plan_group(
    rows: List[Dict[str, Any]],
    pr_columns: Sequence[str],
    aux_counts: Dict[str, int],
    *,
    mode: str,
) -> Dict[str, Any]:
    keeper: Dict[str, Any]
    if mode == "single-lb":
        leaderboard_rows = [
            dict(row)
            for row in rows
            if row.get("leaderboard_tier") is not None
            or bool(row.get("has_leaderboard"))
        ]
        if not leaderboard_rows:
            keeper = dict(sorted(rows, key=lambda row: _keeper_key(row, aux_counts))[0])
        else:
            keeper = leaderboard_rows[0]
    else:
        keeper = dict(sorted(rows, key=lambda row: _keeper_key(row, aux_counts))[0])
    siblings = [
        dict(row)
        for row in rows
        if str(row.get("result_id")) != str(keeper.get("result_id"))
    ]
    mergeable = [col for col in pr_columns if col not in IDENTITY_COLUMNS]
    updates: Dict[str, Any] = {}
    for col in mergeable:
        best_value = keeper.get(col)
        for sib in siblings:
            best_value = _pick_best_value(col, best_value, sib.get(col))
        if best_value != keeper.get(col):
            updates[col] = best_value

    max_stage1 = max(int(bool(row.get("stage1_passed"))) for row in rows)
    relabel_as_backfill = mode != "single-lb" and (
        max_stage1 == 0 and any(_has_post_screening_signal(row) for row in rows)
    )
    if relabel_as_backfill:
        for col, value in RELABEL_COLUMNS.items():
            if keeper.get(col) != value:
                updates[col] = value

    return {
        "graph_fingerprint": str(keeper.get("graph_fingerprint") or ""),
        "keeper_result_id": str(keeper.get("result_id") or ""),
        "keeper_experiment_id": str(keeper.get("experiment_id") or ""),
        "deleted_result_ids": [str(row.get("result_id") or "") for row in siblings],
        "row_count": len(rows),
        "metric_updates": updates,
        "relabel_as_backfill": relabel_as_backfill,
    }


def _ensure_backup_table(conn: sqlite3.Connection) -> List[str]:
    return ensure_backup_table(
        conn,
        backup_table=BACKUP_TABLE,
        source_table="program_results",
        extra_columns=(
            ("backup_kind", "TEXT"),
            ("backup_timestamp", "REAL"),
            ("canonical_result_id", "TEXT"),
            ("cleanup_columns", "TEXT"),
        ),
        indexes=(
            (f"idx_{BACKUP_TABLE}_result_id", ("result_id",)),
            (f"idx_{BACKUP_TABLE}_canonical_result_id", ("canonical_result_id",)),
        ),
    )


def _backup_row(
    conn: sqlite3.Connection,
    backup_columns: Sequence[str],
    row: sqlite3.Row,
    *,
    backup_kind: str,
    backup_timestamp: float,
    canonical_result_id: str,
    cleanup_columns: Sequence[str],
) -> None:
    row_dict = dict(row)
    extras = {
        "backup_kind": backup_kind,
        "backup_timestamp": backup_timestamp,
        "canonical_result_id": canonical_result_id,
        "cleanup_columns": json.dumps(list(cleanup_columns)),
    }
    values = [extras.get(col, row_dict.get(col)) for col in backup_columns]
    quoted = quoted_columns(backup_columns)
    placeholders = ",".join("?" for _ in backup_columns)
    conn.execute(
        f"INSERT INTO {BACKUP_TABLE} ({quoted}) VALUES ({placeholders})",
        values,
    )


def _move_aux_rows(
    conn: sqlite3.Connection, old_result_id: str, new_result_id: str
) -> None:
    if old_result_id == new_result_id:
        return
    for table_name in AUX_RESULT_ID_TABLES:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if exists is None:
            continue
        cols = table_columns(conn, table_name)
        if "result_id" not in cols:
            continue
        quoted = quoted_columns(cols)
        select_exprs = ["?" if col == "result_id" else f'"{col}"' for col in cols]
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {table_name} ({quoted})
            SELECT {",".join(select_exprs)}
            FROM {table_name}
            WHERE result_id = ?
            """,
            (new_result_id, old_result_id),
        )
        conn.execute(f"DELETE FROM {table_name} WHERE result_id = ?", (old_result_id,))


def run(
    db_path: Path,
    *,
    apply: bool,
    fingerprint: Optional[str],
    limit_groups: Optional[int],
    mode: str = "orphan",
) -> int:
    read_conn = connect_readonly(db_path)
    try:
        groups = _fetch_eligible_groups(read_conn, fingerprint, mode)
        pr_columns = table_columns(read_conn, "program_results")
        all_result_ids = [
            str(row.get("result_id") or "")
            for rows in groups.values()
            for row in rows
            if row.get("result_id")
        ]
        aux_counts = _aux_ref_counts(read_conn, all_result_ids)
    finally:
        read_conn.close()

    plans: List[Dict[str, Any]] = []
    for rows in groups.values():
        plan = _plan_group(rows, pr_columns, aux_counts, mode=mode)
        plans.append(plan)

    plans.sort(
        key=lambda plan: (
            -len(plan["deleted_result_ids"]),
            -len(plan["metric_updates"]),
        )
    )
    if limit_groups is not None:
        plans = plans[:limit_groups]

    update_counter: Counter[str] = Counter()
    for plan in plans:
        update_counter.update(plan["metric_updates"].keys())

    relabel_count = sum(1 for plan in plans if plan["relabel_as_backfill"])
    orphan_relabel_count = 0
    if mode == "orphan":
        orphan_read_conn = connect_readonly(db_path)
        try:
            orphan_relabel_count = len(
                _orphan_backfill_relabel_candidates(orphan_read_conn, fingerprint)
            )
        finally:
            orphan_read_conn.close()
    total_groups = len(plans)
    total_deleted = sum(len(plan["deleted_result_ids"]) for plan in plans)
    total_updates = sum(len(plan["metric_updates"]) for plan in plans)

    print(
        f"Plan: {total_groups} {_mode_label(mode)} duplicate fingerprints, "
        f"{total_deleted} rows to delete, "
        f"{total_updates} keeper-field updates, "
        f"{relabel_count} groups relabeled as backfill, "
        f"{orphan_relabel_count} standalone orphan rows to relabel."
    )
    if update_counter:
        print("\nTop updated columns:")
        for col, count in update_counter.most_common(20):
            print(f"  {col}: {count}")

    print("\nTop 10 fingerprints by duplicate count:")
    for plan in plans[:10]:
        summary_cols = sorted(plan["metric_updates"].keys())[:6]
        tail = "..." if len(plan["metric_updates"]) > 6 else ""
        print(
            f"  fp={plan['graph_fingerprint'][:16]} "
            f"keep={plan['keeper_result_id'][:12]} "
            f"drop={len(plan['deleted_result_ids']):2d} "
            f"updates={len(plan['metric_updates']):2d} "
            f"relabel={plan['relabel_as_backfill']} "
            f"({', '.join(summary_cols)}{tail})"
        )

    if not apply:
        print("\nDry-run only. Re-run with --apply to execute against the writer.")
        return 0

    check_writer_lock(DEFAULT_WRITER_LOCK)
    write_conn = connect_writer(db_path)
    backup_columns = _ensure_backup_table(write_conn)
    now = time.time()
    try:
        with write_conn:
            for plan in plans:
                keeper_id = plan["keeper_result_id"]
                dup_ids = plan["deleted_result_ids"]
                if not dup_ids:
                    continue
                keeper_row = write_conn.execute(
                    "SELECT * FROM program_results WHERE result_id = ?",
                    (keeper_id,),
                ).fetchone()
                if keeper_row is None:
                    continue
                _backup_row(
                    write_conn,
                    backup_columns,
                    keeper_row,
                    backup_kind="keeper_premerge",
                    backup_timestamp=now,
                    canonical_result_id=keeper_id,
                    cleanup_columns=sorted(plan["metric_updates"].keys()),
                )
                placeholders = ",".join("?" for _ in dup_ids)
                dup_rows = write_conn.execute(
                    f"SELECT * FROM program_results WHERE result_id IN ({placeholders})",
                    tuple(dup_ids),
                ).fetchall()
                for dup_row in dup_rows:
                    _backup_row(
                        write_conn,
                        backup_columns,
                        dup_row,
                        backup_kind="deleted_duplicate",
                        backup_timestamp=now,
                        canonical_result_id=keeper_id,
                        cleanup_columns=sorted(plan["metric_updates"].keys()),
                    )
                if plan["metric_updates"]:
                    set_parts = ",".join(
                        f'"{col}" = ?' for col in plan["metric_updates"].keys()
                    )
                    values = list(plan["metric_updates"].values()) + [keeper_id]
                    write_conn.execute(
                        f"UPDATE program_results SET {set_parts} WHERE result_id = ?",
                        values,
                    )
                for dup_id in dup_ids:
                    _move_aux_rows(write_conn, dup_id, keeper_id)
                write_conn.execute(
                    f"DELETE FROM program_results WHERE result_id IN ({placeholders})",
                    tuple(dup_ids),
                )
            if mode == "orphan":
                orphan_rows = _orphan_backfill_relabel_candidates(
                    write_conn, fingerprint
                )
                for orphan_row in orphan_rows:
                    result_id = str(orphan_row["result_id"])
                    _backup_row(
                        write_conn,
                        backup_columns,
                        orphan_row,
                        backup_kind="orphan_backfill_relabel_preupdate",
                        backup_timestamp=now,
                        canonical_result_id=result_id,
                        cleanup_columns=sorted(RELABEL_COLUMNS.keys()),
                    )
                    set_parts = ",".join(f'"{col}" = ?' for col in RELABEL_COLUMNS)
                    write_conn.execute(
                        f"UPDATE program_results SET {set_parts} WHERE result_id = ?",
                        (*RELABEL_COLUMNS.values(), result_id),
                    )
        backup_count = table_row_count(write_conn, BACKUP_TABLE)
        print(
            f"Done. Deleted {total_deleted} rows, updated {total_updates} fields, "
            f"relabeled {orphan_relabel_count} standalone orphan rows, "
            f"backup table now has {backup_count} rows."
        )
    finally:
        write_conn.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="Execute cleanup")
    parser.add_argument(
        "--fingerprint",
        default=None,
        help="Restrict to one graph_fingerprint for audit/debug.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Only process the top N duplicate groups.",
    )
    parser.add_argument(
        "--mode",
        choices=("orphan", "single-lb"),
        default="orphan",
        help="Cleanup mode: orphan duplicates or groups with exactly one leaderboard row.",
    )
    args = parser.parse_args(argv)
    try:
        return run(
            args.db,
            apply=args.apply,
            fingerprint=args.fingerprint,
            limit_groups=args.limit_groups,
            mode=args.mode,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
