from __future__ import annotations
"""Auto-extracted mixin for LabNotebook."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ._shared import ExperimentEntry, LOGGER, sanitize_for_db


class _ProgramsMixin:
    """Programs operations for the Lab Notebook."""
    __slots__ = ()

    def purge_junk_programs(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Delete Stage 0 failure program results that carry no useful data.

        Targets results where stage0_passed = 0 or NULL, excluding any that
        somehow passed stage1 (safety guard).

        Returns dict with 'deleted' or 'would_delete' count and 'dry_run' flag.
        """
        junk_query = """
            SELECT result_id, experiment_id FROM program_results
            WHERE (stage0_passed = 0 OR stage0_passed IS NULL)
              AND (stage1_passed != 1 OR stage1_passed IS NULL)
        """
        junk_rows = self.conn.execute(junk_query).fetchall()
        count = len(junk_rows)

        if dry_run or count == 0:
            return {"would_delete": count, "dry_run": True}

        junk_ids = [r["result_id"] for r in junk_rows]
        affected_experiments = {r["experiment_id"] for r in junk_rows if r["experiment_id"]}

        # Cascade delete in foreign-key dependency order
        batch_size = 500
        for i in range(0, len(junk_ids), batch_size):
            batch = junk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            self.conn.execute(
                f"DELETE FROM training_curves WHERE result_id IN ({placeholders})", batch
            )
            self.conn.execute(
                f"DELETE FROM leaderboard WHERE result_id IN ({placeholders})", batch
            )
            self.conn.execute(
                f"DELETE FROM program_results WHERE result_id IN ({placeholders})", batch
            )

        self._maybe_commit()

        # Recalculate op success rates for affected experiments
        for exp_id in affected_experiments:
            try:
                self.update_op_success_rates(exp_id)
            except Exception:
                pass  # non-critical

        return {"deleted": count, "dry_run": False}


    # ── Entries ──

    def add_entry(self, entry: ExperimentEntry) -> str:
        """Add a notebook entry."""
        entry_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO entries
            (entry_id, experiment_id, timestamp, entry_type, title, content,
             metadata_json, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, entry.experiment_id, time.time(),
             entry.entry_type, entry.title, entry.content,
             json.dumps(entry.metadata), ",".join(entry.tags)),
        )
        self._maybe_commit()
        return entry_id


    # ── Program Results ──

    def has_fingerprint(self, graph_fingerprint: str) -> bool:
        """Check if a computation graph has already been evaluated."""
        if not graph_fingerprint:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM program_results WHERE graph_fingerprint = ? LIMIT 1",
            (graph_fingerprint,),
        ).fetchone()
        return row is not None


    def record_program_result(self, experiment_id: str,
                              graph_fingerprint: str, graph_json: str,
                              result_id: Optional[str] = None,
                              **kwargs) -> str:
        """Record results for a single synthesized program.

        Accepts all program_results columns as keyword arguments.
        Boolean fields (stage0_passed, etc.) are converted to int.

        Quality gate: rejects results that provide no learning signal —
        S0 failures, S1 failures with no loss data, and results with
        errors — to keep the database lean and focused.
        """
        # ── Quality gate: reject noise ──
        s0 = kwargs.get("stage0_passed")
        s1 = kwargs.get("stage1_passed")
        loss_ratio = kwargs.get("loss_ratio")
        err = kwargs.get("error_message")

        # Reject S0 failures that carry no error classification.
        # S0 failures WITH error_type inform compile-failure clustering.
        if s0 is not None and not s0:
            error_type = kwargs.get("error_type")
            if not error_type:
                LOGGER.debug("Quality gate: dropping S0 failure with no error_type (fp=%s)", graph_fingerprint)
                return ""

        # Reject S1 failures that carry no learning signal at all:
        # no loss data AND no error classification AND no novelty data.
        # Failures WITH loss_ratio inform grammar weights; failures WITH
        # error_type inform failure-pattern clustering; failures WITH
        # novelty data inform op success rates — all are valuable.
        if s0 and not s1:
            error_type = kwargs.get("error_type")
            novelty = kwargs.get("novelty_score") or kwargs.get("novelty_confidence")
            if loss_ratio is None and not error_type and not novelty:
                LOGGER.debug("Quality gate: dropping S1 failure with no signal (fp=%s)", graph_fingerprint)
                return ""

        if not result_id:
            result_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Convert booleans to int for SQLite
        bool_fields = {
            "stage0_passed", "stage05_passed", "stage1_passed",
            "extreme_input_passed", "random_input_passed",
            "has_nan_output", "has_inf_output", "has_nan_grad", "has_zero_grad",
            "graph_has_gradient_path", "graph_uses_math_spaces",
            "graph_uses_frequency_domain", "regression_gate_pass",
        }
        for f in bool_fields:
            if f in kwargs and kwargs[f] is not None:
                kwargs[f] = int(kwargs[f])

        # Sanitize numeric types (NumPy/Torch scalars) → native Python to prevent blob storage
        kwargs = sanitize_for_db(kwargs)

        # Handle legacy 'throughput' -> 'throughput_tok_s' alias
        if "throughput" in kwargs:
            kwargs.setdefault("throughput_tok_s", kwargs.pop("throughput"))
        valid_columns = self._get_program_results_columns()
        unknown_cols: List[str] = []
        filtered_kwargs: Dict[str, Any] = {}
        for col, val in kwargs.items():
            if col in valid_columns:
                filtered_kwargs[col] = val
            else:
                unknown_cols.append(col)
        if unknown_cols:
            LOGGER.debug(
                "Dropping unknown program_results columns: %s",
                ", ".join(sorted(unknown_cols)),
            )

        # Build column list dynamically from what's provided
        base_cols = ["result_id", "experiment_id", "timestamp",
                     "graph_fingerprint", "graph_json"]
        base_vals = [result_id, experiment_id, now,
                     graph_fingerprint, graph_json]

        extra_cols = []
        extra_vals = []
        for col, val in filtered_kwargs.items():
            extra_cols.append(col)
            extra_vals.append(val)

        all_cols = base_cols + extra_cols
        all_vals = base_vals + extra_vals
        placeholders = ", ".join(["?"] * len(all_cols))
        col_str = ", ".join(all_cols)

        self._submit_write(
            f"INSERT INTO program_results ({col_str}) VALUES ({placeholders})",
            all_vals,
        )
        return result_id


    def save_op_rehabilitation_result(self, op_name: str, compile_passed: bool,
                                       forward_passed: bool, error_message: Optional[str],
                                       model_dim: int) -> None:
        """Store a rehabilitation test result."""
        self.conn.execute(
            """INSERT INTO op_rehabilitation_cache
               (op_name, compile_passed, forward_passed, error_message, tested_at, model_dim)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(op_name) DO UPDATE SET
                compile_passed = excluded.compile_passed,
                forward_passed = excluded.forward_passed,
                error_message = excluded.error_message,
                tested_at = excluded.tested_at,
                model_dim = excluded.model_dim""",
            (op_name, int(compile_passed), int(forward_passed), error_message, time.time(), model_dim),
        )
        self._maybe_commit()


    def get_top_programs(self, n: int = 20,
                         sort_by: str = "novelty_score") -> List[Dict]:
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty",
                       "behavioral_novelty", "validation_loss_ratio",
                       "discovery_loss_ratio"}
        if sort_by not in valid_sorts:
            sort_by = "novelty_score"

        order = "DESC" if sort_by == "novelty_score" else "ASC"
        if sort_by in ("structural_novelty", "behavioral_novelty"):
            order = "DESC"

        rows = self.conn.execute(
            f"""SELECT * FROM program_results
                WHERE stage1_passed = 1
                ORDER BY {sort_by} {order} NULLS LAST
                LIMIT ?""",
            (n,)
        ).fetchall()
        rows_dicts = [dict(r) for r in rows]
        for d in rows_dicts:
            if not d.get("architecture_family"):
                d["architecture_family"] = self._classify_architecture_family(
                    graph_json=d.get("graph_json"),
                    routing_mode=d.get("routing_mode"),
                )
        return rows_dicts


    def get_report_top_programs_grouped_by_fingerprint(
        self,
        n: int = 20,
        sort_by: str = "loss_ratio",
    ) -> List[Dict]:
        """Get report ranking rows grouped by graph fingerprint.

        Returns one representative survivor per fingerprint, enriched with
        repeat-count and run-spread metadata across all stage1 survivors.
        """
        valid_sorts = {"novelty_score", "loss_ratio", "structural_novelty",
                       "behavioral_novelty", "validation_loss_ratio",
                       "discovery_loss_ratio"}
        if sort_by not in valid_sorts:
            sort_by = "loss_ratio"

        order = "DESC" if sort_by == "novelty_score" else "ASC"
        if sort_by in ("structural_novelty", "behavioral_novelty"):
            order = "DESC"

        # Pull enough candidates to fill n unique fingerprints.
        rows = self.conn.execute(
            f"""SELECT * FROM program_results
                WHERE stage1_passed = 1
                ORDER BY {sort_by} {order} NULLS LAST, timestamp DESC
                LIMIT ?""",
            (max(n * 12, 200),),
        ).fetchall()

        spread_rows = self.conn.execute(
            """SELECT
                   graph_fingerprint,
                   COUNT(*) AS repeat_count,
                   COUNT(DISTINCT experiment_id) AS repeat_experiment_span,
                   MIN(timestamp) AS repeat_first_seen_ts,
                   MAX(timestamp) AS repeat_last_seen_ts,
                   MIN(loss_ratio) AS repeat_loss_min,
                   MAX(loss_ratio) AS repeat_loss_max,
                   AVG(loss_ratio) AS repeat_loss_mean,
                   MIN(novelty_score) AS repeat_novelty_min,
                   MAX(novelty_score) AS repeat_novelty_max
               FROM program_results
               WHERE stage1_passed = 1
                 AND graph_fingerprint IS NOT NULL
                 AND TRIM(graph_fingerprint) != ''
               GROUP BY graph_fingerprint"""
        ).fetchall()
        spread_by_fp = {row["graph_fingerprint"]: dict(row) for row in spread_rows}

        grouped: List[Dict] = []
        seen_fingerprints = set()
        for row in rows:
            record = dict(row)
            fingerprint = record.get("graph_fingerprint")
            if not fingerprint or fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            spread = spread_by_fp.get(fingerprint, {})
            record["repeat_count"] = int(spread.get("repeat_count") or 1)
            record["repeat_experiment_span"] = int(spread.get("repeat_experiment_span") or 1)
            record["repeat_first_seen_ts"] = spread.get("repeat_first_seen_ts")
            record["repeat_last_seen_ts"] = spread.get("repeat_last_seen_ts")
            record["repeat_loss_min"] = spread.get("repeat_loss_min")
            record["repeat_loss_max"] = spread.get("repeat_loss_max")
            record["repeat_loss_mean"] = spread.get("repeat_loss_mean")
            record["repeat_novelty_min"] = spread.get("repeat_novelty_min")
            record["repeat_novelty_max"] = spread.get("repeat_novelty_max")
            grouped.append(record)

            if len(grouped) >= n:
                break

        return grouped


    def get_program_results(self, experiment_id: str,
                            limit: int = 500) -> List[Dict]:
        """Get ALL program results for an experiment (not just survivors)."""
        rows = self.conn.execute(
            """SELECT * FROM program_results
               WHERE experiment_id = ?
               ORDER BY novelty_score DESC NULLS LAST
               LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


    def get_program_detail(self, result_id: str) -> Optional[Dict]:
        """Get full detail for a single program result."""
        row = self.conn.execute(
            "SELECT * FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        return self._parse_program_json_fields(dict(row))


    def get_program_details(self, result_ids: List[str]) -> List[Dict]:
        """Batch fetch full details for multiple program results."""
        ids = [rid for rid in result_ids if rid]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM program_results WHERE result_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {}
        for row in rows:
            d = self._parse_program_json_fields(dict(row))
            by_id[d.get("result_id")] = d
        return [by_id.get(rid) for rid in ids]


    @staticmethod
    def _parse_program_json_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        """Parse known JSON fields for program results in-place."""
        json_fields = (
            "graph_json", "fingerprint_json", "training_program_json",
            "graph_category_histogram", "external_benchmarks_json",
            "perf_report_json", "kernel_timings_json", "starvation_report_json",
            "diagnostic_tasks_json", "sparsity_report_json"
        )
        for json_field in json_fields:
            val = d.get(json_field)
            if val and isinstance(val, str):
                try:
                    d[json_field + "_parsed"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


    def _sync_fingerprint_leaderboard(self, result_id: str) -> None:
        """Aggregate leaderboard evidence across all runs of a fingerprint.

        This ensures repeated training runs for the same architecture contribute
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

        pr_cols_all = self._get_program_results_columns()
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

        # Use current best composite entry as the anchor for stable metadata.
        anchor = max(
            lb_rows,
            key=lambda r: (float(r.get("composite_score") or -1e9), float(r.get("timestamp") or 0.0)),
        )
        merged = dict(anchor)

        # Best-of-run metrics used directly by scoring.
        min_cols = (
            "screening_loss_ratio",
            "investigation_loss_ratio",
            "validation_loss_ratio",
            "validation_baseline_ratio",
            "validation_multi_seed_std",
            "discovery_loss_ratio",
            "compression_ratio",
            "routing_drop_rate",
            "robustness_noise_score",
            "wikitext_perplexity",
            "tinystories_perplexity",
            "ncd_score",
        )
        max_cols = (
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
        bool_cols = (
            "screening_passed",
            "investigation_passed",
            "validation_passed",
            "scaling_gate_passed",
        )

        # Combine leaderboard + program rows where useful.
        combo_rows = lb_rows + pr_rows
        for col in min_cols:
            best = self._best_min(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in max_cols:
            best = self._best_max(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in bool_cols:
            best = self._best_bool(combo_rows, col)
            if best is not None:
                merged[col] = best

        # Tier is fingerprint-level progression.
        highest_tier = self._highest_tier(lb_rows)
        if highest_tier:
            merged["tier"] = highest_tier

        nov_conf = self._best_max(pr_rows, "novelty_confidence")
        n_routing = self._count_routing_ops(result_id)
        n_sparse = self._count_sparse_ops(result_id)
        n_moe = self._count_moe_ops(result_id)
        composite = self.compute_composite_score(
            screening_lr=merged.get("screening_loss_ratio"),
            screening_nov=merged.get("screening_novelty"),
            inv_lr=merged.get("investigation_loss_ratio"),
            inv_robust=merged.get("investigation_robustness"),
            val_lr=merged.get("validation_loss_ratio"),
            val_baseline=merged.get("validation_baseline_ratio"),
            val_std=merged.get("validation_multi_seed_std"),
            novelty_confidence=nov_conf,
            scaling_param_efficiency=merged.get("scaling_param_efficiency"),
            is_reference=bool(merged.get("is_reference")),
            routing_savings=merged.get("routing_savings_ratio"),
            compression_ratio=merged.get("compression_ratio"),
            discovery_lr=merged.get("discovery_loss_ratio"),
            spectral_norm=merged.get("fp_jacobian_spectral_norm"),
            robustness_noise=merged.get("robustness_noise_score"),
            quant_retention=merged.get("quant_int8_retention"),
            long_ctx_score=merged.get("robustness_long_ctx_score"),
            init_std=merged.get("init_sensitivity_std"),
            loss_improvement_rate=merged.get("loss_improvement_rate"),
            quant_quality_per_byte=merged.get("quant_quality_per_byte"),
            ncd_score=merged.get("ncd_score"),
            n_routing_ops=n_routing,
            n_sparse_ops=n_sparse,
            n_moe_ops=n_moe,
            recursion_savings=merged.get("recursion_savings_ratio"),
            depth_savings=merged.get("depth_savings_ratio"),
            activation_sparsity=merged.get("activation_sparsity_score"),
            max_viable_seq_len=merged.get("max_viable_seq_len"),
            long_ctx_scaling=merged.get("robustness_long_ctx_scaling_score"),
            long_ctx_passkey=merged.get("robustness_long_ctx_passkey_score"),
            long_ctx_multi_hop=merged.get("robustness_long_ctx_multi_hop_score"),
            long_ctx_assoc=merged.get("robustness_long_ctx_assoc_score"),
            routing_expert_count=merged.get("routing_expert_count"),
            routing_confidence_mean=merged.get("routing_confidence_mean"),
            routing_drop_rate=merged.get("routing_drop_rate"),
            wikitext_perplexity=merged.get("wikitext_perplexity"),
        )
        # Monotonic safeguard: fingerprint aggregate should not score below its
        # historical best leaderboard score when incorporating additional runs.
        prior_best = self._best_max(lb_rows, "composite_score")
        if prior_best is not None:
            composite = max(float(composite), float(prior_best))

        update_cols = [
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
        ]
        update_cols = [c for c in update_cols if c in self._get_leaderboard_columns()]
        sets = [f"{c} = ?" for c in update_cols]

        # Keep all rows for traceability but synchronize fingerprint-level evidence.
        now_ts = time.time()
        params_template = []
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
        self._maybe_commit()
        return synced


    def get_leaderboard_entry(self, result_id: str) -> Optional[Dict]:
        """Fetch a single leaderboard entry by result_id."""
        if not result_id:
            return None
        rows = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(rows) if rows else None


    def get_investigated_fingerprints(self) -> set:
        """Return fingerprints that have already been investigated or beyond.

        Checks both leaderboard tiers AND program_results from investigation/
        ablation experiments, so candidates tested in failed/interrupted
        investigations are not re-queued indefinitely.
        """
        fps = set()
        # Tier-based: candidates promoted in leaderboard
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM leaderboard l "
            "JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE l.tier IN ('investigation', 'validation', 'breakthrough')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        # History-based: fingerprints tested in investigation/ablation experiments
        # (catches failed/interrupted investigations that never reached leaderboard)
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM program_results pr "
            "JOIN experiments e ON e.experiment_id = pr.experiment_id "
            "WHERE e.experiment_type IN ('investigation', 'ablation')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        return fps


    def get_tiers_for_result_ids(self, result_ids: List[str]) -> Dict[str, str]:
        """Return {result_id: tier} for given result IDs that have leaderboard entries."""
        if not result_ids:
            return {}
        placeholders = ",".join("?" for _ in result_ids)
        rows = self.conn.execute(
            f"SELECT result_id, tier FROM leaderboard WHERE result_id IN ({placeholders})",
            result_ids,
        ).fetchall()
        return {r["result_id"]: r["tier"] for r in rows}

