#!/usr/bin/env python3
"""Collapse leaderboard entries that share the same graph_fingerprint.

Leaderboard rows are keyed by ``entry_id``, with ``result_id`` → program_results.
Neither the schema nor the Python API blocks two leaderboard rows (different
``entry_id``/``result_id`` pairs) for the same ``graph_fingerprint``. The
governance work on ``program_results`` (slice 3/4) only enforces within-exp
and cross-exp INSERT rules there; the leaderboard layer has no equivalent.

This script cleans up the historical duplicates. Per fingerprint group:

1. **Select keeper** with this priority (ties break downward):
   a. Entry with non-NULL v2 probe data (`induction_v2_investigation_auc` or
      `binding_v2_investigation_auc`) — per user hint: v2-bearing rescores
      are the correct measurements.
   b. Higher tier ordering: validation > investigation > investigation_failed
      > screening > screened_out > other.
   c. Higher ``composite_score`` (NULLs treated as 0).
   d. More recent ``timestamp``.
2. **Merge non-null columns** from loser entries onto the keeper. Only probe
   metrics / evaluation outputs are merged — identity, tier, notes, rescore_*,
   pin, and reference fields are never touched.
3. **Backup** each deleted loser row plus the pre-merge keeper snapshot to
   ``leaderboard_dedup_backup`` (with audit trail columns).
4. **Delete** loser entries.

Dry-run by default; ``--apply`` executes. The writer flock must be free.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from research.tools._db_maintenance import (
    DEFAULT_WRITER_LOCK,
    connect_readonly,
    connect_writer,
    ensure_backup_table,
    quoted_columns,
    table_columns,
    table_row_count,
)

DEFAULT_DB = Path("research/lab_notebook.db")
BACKUP_TABLE = "leaderboard_dedup_backup"

# Tier rank: lower number = preferred keeper
_TIER_RANK = {
    "validation": 0,
    "breakthrough": 0,
    "investigation": 1,
    "investigation_failed": 2,
    "screening": 3,
    "screened_out": 4,
}
_TIER_RANK_DEFAULT = 5

# Columns to merge from loser rows onto keeper when keeper has NULL.
# Identity + rescore-audit + pin/reference fields are intentionally NOT here.
MERGE_COLUMNS: Tuple[str, ...] = (
    # Probe metrics — v2 investigation (the "correct" measurements)
    "induction_v2_investigation_auc",
    "induction_v2_investigation_max_gap_acc",
    "induction_v2_investigation_protocol_version",
    "binding_v2_investigation_auc",
    "binding_v2_investigation_max_distance_acc",
    "binding_v2_investigation_protocol_version",
    # v1 probes
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "ar_auc",
    # Language / commonsense
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "blimp_n_subtasks",
    "blimp_status",
    # Wikitext
    "wikitext_perplexity",
    "wikitext_score",
    "wikitext_pre_perplexity",
    "wikitext_ppl_improvement",
    "wikitext_ppl_improvement_ratio",
    "ppl_500",
    "peak_ppl",
    "peak_step",
    "steps_to_divergence",
    # Validation / investigation
    "investigation_loss_ratio",
    "investigation_robustness",
    "investigation_best_training",
    "investigation_passed",
    "validation_loss_ratio",
    "validation_baseline_ratio",
    "validation_multi_seed_std",
    "validation_passed",
    "normalized_baseline_ratio",
    # Robustness
    "robustness_long_ctx_score",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "robustness_noise_score",
    "robustness_grade",
    "init_sensitivity_std",
    # Architecture / quant / efficiency
    "param_efficiency",
    "param_count",
    "quant_int8_retention",
    "quant_quality_per_byte",
    "fp_jacobian_spectral_norm",
    "scaling_param_efficiency",
    "scaling_flop_efficiency",
    "scaling_gate_passed",
    "scaling_best_family",
    "scaling_d512_param_efficiency",
    "scaling_confidence",
    "scaling_regime",
    "efficiency_wall_score",
    "efficiency_multiple",
    "max_viable_seq_len",
    "routing_savings_ratio",
    "routing_expert_count",
    "routing_confidence_mean",
    "routing_drop_rate",
    "routing_collapse_score",
    "compression_ratio",
    "activation_sparsity_score",
    "dead_neuron_ratio",
    "n_routing_ops",
    "n_sparse_ops",
    "n_moe_ops",
    # Novelty + diagnostics
    "novelty_confidence",
    "loss_improvement_rate",
    "ncd_score",
    "ncd_description_length_per_param",
    "diagnostic_score",
    "cross_task_score",
    "tinystories_perplexity",
    "tinystories_score",
    "discovery_loss_ratio",
    "screening_novelty",
    "screening_loss_ratio",
    "screening_passed",
    "screening_metric_version",
    "screening_wikitext_status",
    "screening_wikitext_metric_version",
    "screening_wikitext_variant",
    "screening_wikitext_elapsed_ms",
    "screening_wikitext_budget_json",
    "tokenizer_mode",
    "corpus_path",
    "perplexity_tokenizer_penalty",
    "evaluation_protocol_version",
    "evaluation_stage",
    "eval_budget_steps",
    "capability_tier",
    "replication_n",
    "replication_loss_mean",
    "replication_loss_std",
    "replication_best_vs_mean_gap",
    # Scoring columns that get recomputed but we still preserve from loser
    # if keeper is NULL. composite_score itself is NOT merged because we
    # prefer a scoring recompute after the merge.
)

# Columns whose values belong to the specific entry and should never be merged:
# identity / primary keys / pin / notes / audit
_NEVER_MERGE = {
    "entry_id",
    "result_id",
    "timestamp",
    "model_source",
    "architecture_desc",
    "tier",  # keeper's tier wins (already picked by priority)
    "composite_score",  # re-derived; keep keeper's for now
    "tags",
    "notes",
    "is_reference",
    "reference_name",
    "is_pinned",
    "campaign_id",
    "reinvestigation_count",
    "rescore_status",
    "old_composite_score",
    "old_screening_loss_ratio",
    "rescore_timestamp",
    "rescore_reason",
    "pre_inv_score",
    "result_cohort",
    "trust_label",
    "comparability_label",
    "local_only",
    "needs_extended_training",
    "scoring_version",
}


def _fetch_dup_groups(
    conn: sqlite3.Connection,
) -> Dict[str, List[sqlite3.Row]]:
    rows = conn.execute(
        """
        SELECT l.*, pr.graph_fingerprint AS _graph_fingerprint
        FROM leaderboard l
        JOIN program_results pr ON l.result_id = pr.result_id
        WHERE pr.graph_fingerprint IN (
            SELECT pr2.graph_fingerprint FROM leaderboard l2
            JOIN program_results pr2 ON l2.result_id = pr2.result_id
            WHERE TRIM(COALESCE(pr2.graph_fingerprint, '')) <> ''
            GROUP BY pr2.graph_fingerprint HAVING COUNT(*) > 1
        )
        ORDER BY pr.graph_fingerprint, l.composite_score DESC NULLS LAST
        """
    ).fetchall()
    groups: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault(str(r["_graph_fingerprint"]), []).append(r)
    return groups


def _keeper_key(row: sqlite3.Row) -> Tuple[int, int, float, float]:
    """Lower tuple wins."""
    has_v2 = (
        row["induction_v2_investigation_auc"] is not None
        or row["binding_v2_investigation_auc"] is not None
    )
    tier_rank = _TIER_RANK.get(row["tier"] or "", _TIER_RANK_DEFAULT)
    score = float(row["composite_score"] or 0.0)
    ts = float(row["timestamp"] or 0.0)
    return (
        0 if has_v2 else 1,  # v2 first
        tier_rank,  # higher tier first
        -score,  # higher score first (negate for ascending)
        -ts,  # newer first
    )


def _plan_group(rows: List[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if len(rows) < 2:
        return None
    ranked = sorted(rows, key=_keeper_key)
    keeper = dict(ranked[0])
    losers = [dict(r) for r in ranked[1:]]
    # Merge order: per column, the first non-NULL loser (ranked order) wins ties
    updates: Dict[str, Any] = {}
    contributing: Dict[str, str] = {}
    for col in MERGE_COLUMNS:
        if col in _NEVER_MERGE:
            continue
        if keeper.get(col) is not None:
            continue
        for loser in losers:
            val = loser.get(col)
            if val is not None:
                updates[col] = val
                contributing[col] = str(loser.get("entry_id"))
                break
    return {
        "graph_fingerprint": keeper.get("_graph_fingerprint"),
        "keeper_entry_id": str(keeper.get("entry_id")),
        "keeper_result_id": str(keeper.get("result_id")),
        "keeper_tier": keeper.get("tier"),
        "keeper_composite": keeper.get("composite_score"),
        "keeper_has_v2": (
            keeper.get("induction_v2_investigation_auc") is not None
            or keeper.get("binding_v2_investigation_auc") is not None
        ),
        "loser_entry_ids": [str(l.get("entry_id")) for l in losers],
        "loser_summary": [
            {
                "entry_id": str(l.get("entry_id")),
                "tier": l.get("tier"),
                "composite": l.get("composite_score"),
                "has_v2": (
                    l.get("induction_v2_investigation_auc") is not None
                    or l.get("binding_v2_investigation_auc") is not None
                ),
            }
            for l in losers
        ],
        "merge_updates": updates,
        "contributing_entry_ids": contributing,
    }


def _ensure_backup_table(conn: sqlite3.Connection) -> None:
    ensure_backup_table(
        conn,
        backup_table=BACKUP_TABLE,
        source_table="leaderboard",
        extra_columns=(
            ("dedup_at", "REAL"),
            ("dedup_role", "TEXT"),
            ("dedup_keeper_entry_id", "TEXT"),
            ("dedup_merged_columns", "TEXT"),
        ),
        indexes=((f"idx_{BACKUP_TABLE}_entry_id", ("entry_id",)),),
    )


def _apply(conn: sqlite3.Connection, plans: List[Dict[str, Any]]) -> Tuple[int, int]:
    lb_cols = table_columns(conn, "leaderboard")
    backup_cols = lb_cols + [
        "dedup_at",
        "dedup_role",
        "dedup_keeper_entry_id",
        "dedup_merged_columns",
    ]
    now = time.time()
    _ensure_backup_table(conn)
    n_deleted = 0
    n_merged_fields = 0
    with conn:
        for plan in plans:
            keeper_id = plan["keeper_entry_id"]
            loser_ids = plan["loser_entry_ids"]
            updates = plan["merge_updates"]
            # Snapshot pre-merge keeper
            keeper_row = conn.execute(
                "SELECT * FROM leaderboard WHERE entry_id = ?", (keeper_id,)
            ).fetchone()
            if keeper_row is None:
                continue
            snapshot_vals = [keeper_row[c] for c in lb_cols] + [
                now,
                "keeper_pre_merge",
                keeper_id,
                json.dumps(sorted(updates.keys())),
            ]
            placeholders = ",".join("?" for _ in backup_cols)
            quoted = quoted_columns(backup_cols)
            conn.execute(
                f"INSERT INTO {BACKUP_TABLE} ({quoted}) VALUES ({placeholders})",
                snapshot_vals,
            )
            # Snapshot each loser
            for lid in loser_ids:
                loser_row = conn.execute(
                    "SELECT * FROM leaderboard WHERE entry_id = ?", (lid,)
                ).fetchone()
                if loser_row is None:
                    continue
                loser_vals = [loser_row[c] for c in lb_cols] + [
                    now,
                    "deleted_loser",
                    keeper_id,
                    json.dumps(sorted(updates.keys())),
                ]
                conn.execute(
                    f"INSERT INTO {BACKUP_TABLE} ({quoted}) VALUES ({placeholders})",
                    loser_vals,
                )
            # Merge non-null columns onto keeper
            if updates:
                set_parts = ",".join(f'"{c}" = ?' for c in updates)
                vals = list(updates.values()) + [keeper_id]
                conn.execute(
                    f"UPDATE leaderboard SET {set_parts} WHERE entry_id = ?", vals
                )
                n_merged_fields += len(updates)
            # Delete losers
            if loser_ids:
                qm = ",".join("?" for _ in loser_ids)
                conn.execute(
                    f"DELETE FROM leaderboard WHERE entry_id IN ({qm})",
                    tuple(loser_ids),
                )
                n_deleted += len(loser_ids)
    return n_deleted, n_merged_fields


def run(db: Path, *, apply: bool) -> int:
    read_conn = connect_readonly(db)
    try:
        groups = _fetch_dup_groups(read_conn)
    finally:
        read_conn.close()

    plans: List[Dict[str, Any]] = []
    col_counter: Counter[str] = Counter()
    v2_keeper_count = 0
    for fp, rows in groups.items():
        plan = _plan_group(rows)
        if plan is None:
            continue
        plans.append(plan)
        col_counter.update(plan["merge_updates"].keys())
        if plan["keeper_has_v2"]:
            v2_keeper_count += 1
    plans.sort(key=lambda p: -len(p["loser_entry_ids"]))

    n_groups = len(plans)
    n_losers = sum(len(p["loser_entry_ids"]) for p in plans)
    n_fields = sum(len(p["merge_updates"]) for p in plans)

    print(
        f"Plan: {n_groups} duplicate fingerprint groups, {n_losers} losers to delete, "
        f"{n_fields} field-values to merge onto keepers."
    )
    print(f"  keepers with v2 probe data: {v2_keeper_count}/{n_groups}")
    if col_counter:
        print("\nTop merged columns:")
        for col, n in col_counter.most_common(20):
            print(f"  {col}: {n}")
    print("\nTop 10 groups by loser count:")
    for plan in plans[:10]:
        fp = plan["graph_fingerprint"][:16]
        tier = plan["keeper_tier"] or "-"
        score = plan["keeper_composite"]
        score_s = f"{float(score):.1f}" if score is not None else "  -"
        v2 = "v2" if plan["keeper_has_v2"] else " -"
        print(
            f"  fp={fp}  keep={plan['keeper_entry_id'][:12]}  "
            f"tier={tier:<20}  score={score_s}  {v2}  "
            f"losers={len(plan['loser_entry_ids'])}  "
            f"merge={len(plan['merge_updates'])}"
        )

    if not apply:
        print("\nDry-run only. Re-run with --apply to execute against the writer.")
        return 0

    from research.tools._db_maintenance import check_writer_lock

    check_writer_lock(DEFAULT_WRITER_LOCK)
    print(f"\nApplying merges + deletes to {db}...")
    write_conn = connect_writer(db)
    try:
        n_del, n_mf = _apply(write_conn, plans)
        backup_n = table_row_count(write_conn, BACKUP_TABLE)
    finally:
        write_conn.close()
    print(
        f"Done. Deleted {n_del} loser rows; merged {n_mf} field-values. "
        f"Backup table now has {backup_n} rows (keepers + losers)."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--apply", action="store_true", help="Execute merges + deletes")
    args = ap.parse_args(argv)
    try:
        return run(args.db, apply=args.apply)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
