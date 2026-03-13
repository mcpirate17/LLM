"""Leaderboard fingerprint aggregation — syncs evidence across runs."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .leaderboard_schema import (
    FINGERPRINT_BOOL_COLS,
    FINGERPRINT_MAX_COLS,
    FINGERPRINT_MIN_COLS,
    FINGERPRINT_UPDATE_COLS,
    SCORE_COLUMN_MAP,
)
from .leaderboard_scoring import compute_composite_score


class LeaderboardFingerprintMixin:
    """Mixin providing fingerprint-level leaderboard aggregation."""

    __slots__ = ()

    def _sync_fingerprint_leaderboard(self, result_id: str) -> None:
        """Aggregate leaderboard evidence across all runs of a fingerprint.

        Ensures repeated training runs for the same architecture contribute
        to one coherent fingerprint-level score/tier rather than fragmenting
        across per-result rows.
        """
        fp_row = self.conn.execute(
            "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if not fp_row or not fp_row["graph_fingerprint"]:
            return
        graph_fingerprint = str(fp_row["graph_fingerprint"])

        lb_rows_raw = self.conn.execute(
            """
            SELECT l.*
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint = ?
            """,
            (graph_fingerprint,),
        ).fetchall()
        if not lb_rows_raw:
            return
        lb_rows = [dict(r) for r in lb_rows_raw]

        pr_cols_all = self.notebook._get_program_results_columns()
        wanted_pr_cols = [
            "result_id", "novelty_confidence", "loss_improvement_rate",
            "discovery_loss_ratio", "validation_loss_ratio", "efficiency_multiple",
            "max_viable_seq_len", "robustness_long_ctx_scaling_score",
            "robustness_long_ctx_assoc_score", "robustness_long_ctx_multi_hop_score",
            "robustness_long_ctx_passkey_score", "robustness_long_ctx_retrieval_aggregate",
            "robustness_long_ctx_combined_score", "robustness_noise_score",
            "activation_sparsity_score", "depth_savings_ratio",
            "recursion_savings_ratio", "routing_expert_count",
            "routing_confidence_mean", "routing_drop_rate",
            "wikitext_perplexity", "wikitext_score", "tinystories_perplexity",
            "tinystories_score", "cross_task_score", "efficiency_wall_score",
        ]
        pr_select_cols = [c for c in wanted_pr_cols if c in pr_cols_all]
        if not pr_select_cols:
            pr_select_cols = ["result_id"]
        pr_rows_raw = self.conn.execute(
            f"SELECT {', '.join(pr_select_cols)} FROM program_results WHERE graph_fingerprint = ?",
            (graph_fingerprint,),
        ).fetchall()
        pr_rows = [dict(r) for r in pr_rows_raw]

        # Use current best composite entry as anchor for stable metadata.
        anchor = max(
            lb_rows,
            key=lambda r: (
                float(r.get("composite_score") or -1e9),
                float(r.get("timestamp") or 0.0),
            ),
        )
        merged = dict(anchor)

        # Combine leaderboard + program rows where useful.
        combo_rows = lb_rows + pr_rows
        for col in FINGERPRINT_MIN_COLS:
            best = self.notebook._best_min(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in FINGERPRINT_MAX_COLS:
            best = self.notebook._best_max(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in FINGERPRINT_BOOL_COLS:
            best = self.notebook._best_bool(combo_rows, col)
            if best is not None:
                merged[col] = best

        # Tier is fingerprint-level progression.
        highest_tier = self.notebook._highest_tier(lb_rows)
        if highest_tier:
            merged["tier"] = highest_tier

        nov_conf = self.notebook._best_max(pr_rows, "novelty_confidence")
        n_routing = self.notebook._count_routing_ops(result_id)
        n_sparse = self.notebook._count_sparse_ops(result_id)
        n_moe = self.notebook._count_moe_ops(result_id)

        # Build kwargs from merged data using the column map.
        score_kw: Dict[str, Any] = {
            param: merged.get(col) for col, param in SCORE_COLUMN_MAP.items()
        }
        score_kw["novelty_confidence"] = nov_conf
        score_kw["scaling_param_efficiency"] = merged.get("scaling_param_efficiency")
        score_kw["is_reference"] = bool(merged.get("is_reference"))
        score_kw["loss_improvement_rate"] = merged.get("loss_improvement_rate")
        score_kw["n_routing_ops"] = n_routing
        score_kw["n_sparse_ops"] = n_sparse
        score_kw["n_moe_ops"] = n_moe

        composite = compute_composite_score(**score_kw)

        # Monotonic safeguard: aggregate should not score below historical best.
        prior_best = self.notebook._best_max(lb_rows, "composite_score")
        if prior_best is not None:
            composite = max(float(composite), float(prior_best))

        # Filter update cols to those that exist in the schema.
        existing_cols = self._get_leaderboard_columns()
        update_cols = [c for c in FINGERPRINT_UPDATE_COLS if c in existing_cols]
        sets = [f"{c} = ?" for c in update_cols]

        now_ts = time.time()
        params_template: List[Any] = []
        for col in update_cols:
            if col == "composite_score":
                params_template.append(composite)
            elif col == "timestamp":
                params_template.append(now_ts)
            else:
                val = merged.get(col)
                if isinstance(val, bool):
                    val = int(val)
                params_template.append(val)

        for row in lb_rows:
            params = list(params_template)
            params.append(row["entry_id"])
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )

    def backfill_fingerprint_aggregates(self) -> int:
        """Recompute fingerprint-level leaderboard aggregates for all entries."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT l.result_id
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint IS NOT NULL
            """
        ).fetchall()
        synced = 0
        seen_fp: set[str] = set()
        for row in rows:
            rid = row["result_id"]
            fp_row = self.conn.execute(
                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                (rid,),
            ).fetchone()
            fp = str(fp_row["graph_fingerprint"]) if fp_row and fp_row["graph_fingerprint"] else ""
            if not fp or fp in seen_fp:
                continue
            seen_fp.add(fp)
            self._sync_fingerprint_leaderboard(rid)
            synced += 1
        self.notebook._maybe_commit()
        return synced
