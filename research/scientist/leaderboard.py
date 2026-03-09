import logging
import math
import uuid
import time
import json
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

class LeaderboardManager:
    """Manages leaderboard operations for the lab notebook."""

    def __init__(self, notebook: 'LabNotebook'):
        self.notebook = notebook

    @property
    def conn(self):
        return self.notebook.conn
        
    def _submit_write(self, sql: str, params: tuple = ()) -> None:
        self.notebook._submit_write(sql, params)

    def _maybe_commit(self) -> None:
        self.notebook._maybe_commit()


    def _get_leaderboard_columns(self) -> set[str]:
        """Return current leaderboard columns for defensive updates."""
        if self.notebook._leaderboard_columns is None:
            rows = self.conn.execute("PRAGMA table_info(leaderboard)").fetchall()
            self.notebook._leaderboard_columns = {str(row[1]) for row in rows}
        return self.notebook._leaderboard_columns


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
            best = self.notebook._best_min(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in max_cols:
            best = self.notebook._best_max(combo_rows, col)
            if best is not None:
                merged[col] = best
        for col in bool_cols:
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
        prior_best = self.notebook._best_max(lb_rows, "composite_score")
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
        self.notebook._maybe_commit()
        return synced

    @staticmethod
    def compute_efficiency_multiple(
        loss_ratio: Optional[float] = None,
        param_count: Optional[float] = None,
        flops_forward: Optional[float] = None,
        throughput_tok_s: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
        forward_time_ms: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Geometric mean of per-dimension ratios vs GPT-2.

        All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
        dimensions to return a result (graceful with missing data).
        Returns dict with per-dimension ratios and ``geomean``, or None.
        """
        ref = LabNotebook._GPT2_REF
        ratios: Dict[str, float] = {}

        # x_quality: ref_loss / cand_loss (lower loss = better)
        if loss_ratio is not None and loss_ratio > 0:
            ratios["x_quality"] = ref["loss_ratio"] / loss_ratio

        # x_params: ref_params / cand_params (fewer = better)
        if param_count is not None and param_count > 0:
            ratios["x_params"] = ref["param_count"] / param_count

        # x_flops: ref_flops / cand_flops (fewer = better)
        if flops_forward is not None and flops_forward > 0:
            ratios["x_flops"] = ref["flops_forward"] / flops_forward

        # x_throughput: cand_tput / ref_tput (higher = better)
        if throughput_tok_s is not None and throughput_tok_s > 0:
            ratios["x_throughput"] = throughput_tok_s / ref["throughput_tok_s"]

        # x_memory: ref_mem / cand_mem (less = better)
        if peak_memory_mb is not None and peak_memory_mb > 0:
            ratios["x_memory"] = ref["peak_memory_mb"] / peak_memory_mb

        # x_latency: ref_lat / cand_lat (lower = better)
        if forward_time_ms is not None and forward_time_ms > 0:
            ratios["x_latency"] = ref["forward_time_ms"] / forward_time_ms

        if len(ratios) < 3:
            return None

        geomean = 1.0
        for v in ratios.values():
            geomean *= v
        geomean = geomean ** (1.0 / len(ratios))
        ratios["geomean"] = geomean
        ratios["n_dimensions"] = float(len(ratios) - 1)  # exclude geomean itself
        return ratios

    @staticmethod
    def compute_composite_score(
        screening_lr: Optional[float] = None,
        screening_nov: Optional[float] = None,
        inv_lr: Optional[float] = None,
        inv_robust: Optional[float] = None,
        val_lr: Optional[float] = None,
        val_baseline: Optional[float] = None,
        val_std: Optional[float] = None,
        novelty_confidence: Optional[float] = None,
        scaling_param_efficiency: Optional[float] = None,
        is_reference: bool = False,
        routing_savings: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        entropy: Optional[float] = None,
        discovery_lr: Optional[float] = None,
        spectral_norm: Optional[float] = None,
        robustness_noise: Optional[float] = None,
        quant_retention: Optional[float] = None,
        long_ctx_score: Optional[float] = None,
        init_std: Optional[float] = None,
        loss_improvement_rate: Optional[float] = None,
        quant_quality_per_byte: Optional[float] = None,
        ncd_score: Optional[float] = None,
        n_routing_ops: Optional[int] = None,
        n_sparse_ops: Optional[int] = None,
        n_moe_ops: Optional[int] = None,
        recursion_savings: Optional[float] = None,
        depth_savings: Optional[float] = None,
        activation_sparsity: Optional[float] = None,
        max_viable_seq_len: Optional[int] = None,
        long_ctx_scaling: Optional[float] = None,
        long_ctx_passkey: Optional[float] = None,
        long_ctx_multi_hop: Optional[float] = None,
        long_ctx_assoc: Optional[float] = None,
        routing_expert_count: Optional[int] = None,
        routing_confidence_mean: Optional[float] = None,
        routing_drop_rate: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Compute "Total Scientific Utility" — an open-ended additive score.
        ...
        """
        score = 0.0

        # 1. Performance Utility (Primary)
        # Use validation_baseline_ratio if available, otherwise fallback.
        # Apply a confidence discount: screening-only metrics are less
        # trustworthy than investigation/validation-confirmed ones.
        if val_baseline is not None:
            perf_lr = val_baseline
            perf_confidence = 1.0
        elif val_lr is not None:
            perf_lr = val_lr
            perf_confidence = 1.0
        elif inv_lr is not None:
            perf_lr = inv_lr
            perf_confidence = 0.85
        elif screening_lr is not None:
            perf_lr = screening_lr
            perf_confidence = 0.65
        else:
            perf_lr = None
            perf_confidence = 0.0
        if perf_lr is not None:
            # Nonlinear curve (matches frontend): heavily reward strong loss,
            # suppress mediocre models that survive on novelty alone.
            perf_norm = max(0.0, min(1.0, 1.0 - perf_lr))
            score += 100.0 * (perf_norm ** 1.6) * perf_confidence

        # Discovery channel (random tokens)
        if discovery_lr is not None:
            score += 20.0 * max(0, 1.0 - discovery_lr)

        # Learning Efficiency: How fast did it learn?
        if loss_improvement_rate is not None:
            score += 20.0 * max(0, min(1.0, loss_improvement_rate))

        # 2. Novelty Utility — gated by performance quality so novelty
        # cannot dominate when loss evidence is weak (matches frontend).
        eff_nov = 1.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
        conf = 1.0 if is_reference else (novelty_confidence if novelty_confidence is not None else 1.0)
        novelty_gate = 1.0
        if perf_lr is not None:
            novelty_gate = max(0.0, min(1.0, (0.9 - perf_lr) / 0.6))
        score += 40.0 * eff_nov * conf * novelty_gate

        # 3. Efficiency & Scaling Utility
        # 5x TARGET: Accelerating reward curve for efficiency multiples.
        # 1x → 0pts, 2x → 15pts, 3x → 30pts, 5x → 60pts, 10x → 100pts.
        if scaling_param_efficiency is not None:
            eff_above_1 = max(0.0, scaling_param_efficiency - 1.0)
            # Superlinear reward: sqrt curve * 22 gives ~60pts at 5x
            score += 22.0 * math.sqrt(eff_above_1)
            # Milestone bonus at 5x+ (the explicit target)
            if scaling_param_efficiency >= 5.0:
                score += 25.0
        
        if routing_savings is not None:
            score += 50.0 * routing_savings
            
        if compression_ratio is not None:
            # Reward compression: 4x (0.25) -> 20 utility
            # Weight compression ratio + maintained quality
            comp_score = 20.0 * max(0, 1.0 - (compression_ratio / 1.0))
            if quant_quality_per_byte is not None:
                # Reward high quality per compressed byte
                comp_score += 10.0 * max(0, quant_quality_per_byte)
            score += comp_score

        # NCD: reward compact graph descriptions that explain training behavior
        if ncd_score is not None:
            # Low NCD = graph structure predicts training dynamics (good)
            # Max 15 points when NCD = 0
            score += 15.0 * max(0, 1.0 - ncd_score)

        # 3b. Structural complexity bonus: reward routing/branching architectures
        # Counterbalances MDL and NCD penalties for exotic architectures
        if n_routing_ops is not None and n_routing_ops > 0:
            # Up to 15 points (reduced from 25, replaced by MoE quality)
            score += min(15.0, n_routing_ops * 5.0)

        # 3c. Sparsity bonus (max 30pts: 20 structural + 10 activation)
        if n_sparse_ops is not None and n_sparse_ops > 0:
            score += min(20.0, n_sparse_ops * 6.0)
        if activation_sparsity is not None and activation_sparsity > 0.3:
            score += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)

        # 3d. MoE quality bonus (max ~25pts)
        if n_moe_ops is not None and n_moe_ops > 0:
            moe_base = min(10.0, n_moe_ops * 5.0)
            # Expert diversity multiplier: more experts = higher potential
            if routing_expert_count is not None and routing_expert_count > 1:
                expert_mult = min(1.5, 1.0 + math.log2(routing_expert_count) / 6.0)
                moe_base *= expert_mult
            # Confidence bonus: high confidence = routing is working
            if routing_confidence_mean is not None and routing_confidence_mean > 0.5:
                moe_base *= 1.0 + 0.3 * (routing_confidence_mean - 0.5)
            # Drop rate penalty: high drop = wasted compute
            if routing_drop_rate is not None and routing_drop_rate > 0.3:
                moe_base *= max(0.5, 1.0 - (routing_drop_rate - 0.3))
            score += moe_base

        # 3e. Adaptive computation bonus (max 25pts)
        if recursion_savings is not None and recursion_savings > 0:
            score += 15.0 * min(1.0, recursion_savings / 0.5)
        if depth_savings is not None and depth_savings > 0:
            score += 10.0 * min(1.0, depth_savings / 0.5)

        # 4. Robustness & Stability Utility
        if spectral_norm is not None:
            score += 10.0 * max(0, 1.0 - (spectral_norm / 20.0))

        if robustness_noise is not None:
            score += 15.0 * max(0, 1.0 - robustness_noise)

        if quant_retention is not None:
            score += 15.0 * max(0, quant_retention - 0.5) / 0.5

        # 4b. Expanded long-context scoring (total budget 50pts, up from 20)
        if long_ctx_score is not None:
            # Base combined score: 20pts (unchanged)
            score += 20.0 * long_ctx_score
            # Sub-score bonuses: reward specific long-context capabilities
            if long_ctx_passkey is not None:
                score += 10.0 * long_ctx_passkey
            if long_ctx_multi_hop is not None:
                score += 10.0 * long_ctx_multi_hop
            if long_ctx_scaling is not None:
                score += 5.0 * long_ctx_scaling
            if long_ctx_assoc is not None:
                score += 5.0 * long_ctx_assoc

        # Bonus for viable long sequences (log-scale, max 20pts)
        if max_viable_seq_len is not None and max_viable_seq_len > 512:
            seq_bonus = 5.0 * min(4.0, math.log2(max_viable_seq_len / 512))
            score += seq_bonus

        # 5. Generalization Utility (The "Anti-Cheat")
        # Wikitext/TinyStories scores are normalized 0-1, where 1 is good (low perplexity)
        # We also look at raw perplexity for severe penalties.
        # Note: These values might be in the future, but we add them now.
        
        # If we have raw perplexity data, apply severe penalties for "Zombie" models
        # We assume 10^6 is the cutoff for total failure to generalize.
        # We use wikitext_perplexity as the primary proxy.
        # This function signature might need updating or we use kwargs
        wikitext_perplexity = kwargs.get("wikitext_perplexity")
        if wikitext_perplexity is not None:
            if wikitext_perplexity > 1000000:
                return 0.0 # Instant disqualification for non-generalizing models
            if wikitext_perplexity > 1000:
                # Logarithmic penalty for high perplexity
                score -= 50.0 * math.log10(wikitext_perplexity / 1000.0)

        # 6. Numerical Integrity (Spectral Floor)
        if spectral_norm is not None and spectral_norm < 0.01:
            # Gradients are likely not propagating (numerical collapse)
            score -= 40.0

        # 7. Penalties
        if val_std is not None and val_std > 0.1:
            # High variance across seeds is a major red flag
            score -= 50.0 * min(2.0, val_std / 0.5)
            
        if entropy is not None and entropy > 0.95:
            # Only penalize truly unfocused routing, not healthy multi-lane distribution
            score -= 5.0 * (entropy - 0.95)

        # Scaling gate: stricter penalty for sub-baseline efficiency.
        # Models below 1x efficiency vs GPT-2 should be heavily suppressed —
        # they cannot plausibly beat GPT/Mamba by 5x.
        if scaling_param_efficiency is not None and scaling_param_efficiency < 1.0:
            # 0.5x → 50% score, 0.1x → 10% score
            score *= max(0.1, scaling_param_efficiency)

        # Sanity floor
        return max(0.0, score)

    @staticmethod
    def _reference_novelty_for_display(novelty: Optional[float]) -> Optional[float]:
        """Compress reference novelty values for dashboard display.

        Reference architectures are anchor points, so we intentionally present
        their novelty on a reduced scale to avoid implying they are frontier
        discoveries in the same sense as synthesized candidates.
        """
        if novelty is None:
            return None
        try:
            value = float(novelty)
        except (TypeError, ValueError):
            return None
        value = max(0.0, min(1.0, value))
        return min(0.35, value * 0.4)

    def upsert_leaderboard(
        self,
        result_id: str,
        model_source: str,
        architecture_desc: str = "",
        tier: str = "screening",
        tags: Optional[str] = None,
        notes: Optional[str] = None,
        is_reference: bool = False,
        reference_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Insert or update a leaderboard entry.

        Accepts all leaderboard columns as keyword arguments.
        Fields are only updated if provided and not None (prevents accidental NULLing).
        """
        # Check if entry exists for this result_id
        existing = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()

        # Combine kwargs with existing data for composite score recomputation
        d = dict(existing) if existing else {}
        # Sanitize all incoming values
        kwargs = self.notebook._sanitize_numeric(kwargs)
        
        d.update({k: v for k, v in kwargs.items() if v is not None})
        if tags is not None: d["tags"] = tags
        if notes is not None: d["notes"] = notes
        d["tier"] = tier
        d["model_source"] = model_source
        if architecture_desc: d["architecture_desc"] = architecture_desc
        d["is_reference"] = int(is_reference)
        if reference_name: d["reference_name"] = reference_name

        # Look up novelty_confidence from linked program_results
        nov_conf = d.get("novelty_confidence")
        if nov_conf is None:
            pr = self.conn.execute(
                "SELECT novelty_confidence FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if pr:
                nov_conf = pr["novelty_confidence"]

        composite = self.compute_composite_score(
            screening_lr=d.get("screening_loss_ratio"),
            screening_nov=d.get("screening_novelty"),
            inv_lr=d.get("investigation_loss_ratio"),
            inv_robust=d.get("investigation_robustness"),
            val_lr=d.get("validation_loss_ratio"),
            val_baseline=d.get("validation_baseline_ratio"),
            val_std=d.get("validation_multi_seed_std"),
            novelty_confidence=nov_conf,
            scaling_param_efficiency=d.get("scaling_param_efficiency"),
            is_reference=bool(is_reference),
            routing_savings=d.get("routing_savings_ratio"),
            compression_ratio=d.get("compression_ratio"),
            discovery_lr=d.get("discovery_loss_ratio"),
            spectral_norm=d.get("fp_jacobian_spectral_norm"),
            robustness_noise=d.get("robustness_noise_score"),
            quant_retention=d.get("quant_int8_retention"),
            long_ctx_score=d.get("robustness_long_ctx_score"),
            init_std=d.get("init_sensitivity_std"),
            loss_improvement_rate=d.get("loss_improvement_rate"),
            quant_quality_per_byte=d.get("quant_quality_per_byte"),
            ncd_score=d.get("ncd_score"),
            n_routing_ops=self.notebook._count_routing_ops(result_id),
            n_sparse_ops=self.notebook._count_sparse_ops(result_id),
            n_moe_ops=self.notebook._count_moe_ops(result_id),
            recursion_savings=d.get("recursion_savings_ratio"),
            depth_savings=d.get("depth_savings_ratio"),
            activation_sparsity=d.get("activation_sparsity_score"),
            max_viable_seq_len=d.get("max_viable_seq_len"),
            long_ctx_scaling=d.get("robustness_long_ctx_scaling_score"),
            long_ctx_passkey=d.get("robustness_long_ctx_passkey_score"),
            long_ctx_multi_hop=d.get("robustness_long_ctx_multi_hop_score"),
            long_ctx_assoc=d.get("robustness_long_ctx_assoc_score"),
            routing_expert_count=d.get("routing_expert_count"),
            routing_confidence_mean=d.get("routing_confidence_mean"),
            routing_drop_rate=d.get("routing_drop_rate"),
        )

        # Compute efficiency_multiple from program_results operational metrics
        eff_mult = kwargs.get("efficiency_multiple")
        if eff_mult is None:
            pr_row = self.conn.execute(
                "SELECT loss_ratio, param_count, flops_forward, "
                "throughput_tok_s, peak_memory_mb, forward_time_ms "
                "FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if pr_row:
                eff_result = self.compute_efficiency_multiple(
                    loss_ratio=pr_row["loss_ratio"],
                    param_count=pr_row["param_count"],
                    flops_forward=pr_row["flops_forward"],
                    throughput_tok_s=pr_row["throughput_tok_s"],
                    peak_memory_mb=pr_row["peak_memory_mb"],
                    forward_time_ms=pr_row["forward_time_ms"],
                )
                if eff_result is not None:
                    eff_mult = eff_result["geomean"]
        if eff_mult is not None:
            kwargs["efficiency_multiple"] = eff_mult

        if existing:
            entry_id = existing["entry_id"]
            sets = ["timestamp = ?", "model_source = ?", "tier = ?", "composite_score = ?", "is_reference = ?"]
            params = [time.time(), model_source, tier, composite, int(is_reference)]
            
            if architecture_desc:
                sets.append("architecture_desc = ?")
                params.append(architecture_desc)
            if tags is not None:
                sets.append("tags = ?")
                params.append(tags)
            if notes is not None:
                sets.append("notes = ?")
                params.append(notes)
            if reference_name is not None:
                sets.append("reference_name = ?")
                params.append(reference_name)

            # Whitelist for other columns from kwargs
            for col in ("screening_loss_ratio", "screening_novelty", "screening_passed",
                         "investigation_loss_ratio", "investigation_robustness",
                         "investigation_best_training", "investigation_passed",
                         "validation_loss_ratio", "validation_baseline_ratio",
                         "validation_multi_seed_std", "validation_passed",
                         "normalized_baseline_ratio", "param_efficiency",
                         "quant_int8_retention", "quant_quality_per_byte",
                         "robustness_long_ctx_score", "robustness_noise_score",
                         "init_sensitivity_std", "fp_jacobian_spectral_norm",
                         "scaling_param_efficiency", "scaling_flop_efficiency",
                         "scaling_gate_passed", "scaling_best_family",
                         "scaling_d512_param_efficiency", "scaling_confidence",
                         "routing_savings_ratio", "compression_ratio",
                         "discovery_loss_ratio", "ncd_score",
                         "robustness_long_ctx_scaling_score",
                         "robustness_long_ctx_assoc_score",
                         "robustness_long_ctx_multi_hop_score",
                         "robustness_long_ctx_passkey_score",
                         "robustness_long_ctx_retrieval_aggregate",
                         "robustness_long_ctx_combined_score",
                         "depth_savings_ratio", "recursion_savings_ratio",
                         "activation_sparsity_score", "routing_expert_count",
                         "routing_confidence_mean", "routing_drop_rate",
                         "efficiency_multiple",
                         "wikitext_perplexity", "wikitext_score",
                         "tinystories_perplexity", "tinystories_score"):
                if col in kwargs and kwargs[col] is not None:
                    sets.append(f"{col} = ?")
                    val = kwargs[col]
                    if isinstance(val, bool): val = int(val)
                    params.append(val)

            params.append(entry_id)
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )
        else:
            entry_id = str(uuid.uuid4())[:12]
            cols = ["entry_id", "result_id", "timestamp", "model_source", "architecture_desc",
                    "tier", "composite_score", "is_reference", "reference_name", "tags", "notes"]
            vals = [entry_id, result_id, time.time(), model_source, architecture_desc,
                    tier, composite, int(is_reference), reference_name, tags, notes]

            for col in ("screening_loss_ratio", "screening_novelty", "screening_passed",
                         "investigation_loss_ratio", "investigation_robustness",
                         "investigation_best_training", "investigation_passed",
                         "validation_loss_ratio", "validation_baseline_ratio",
                         "validation_multi_seed_std", "validation_passed",
                         "normalized_baseline_ratio", "param_efficiency",
                         "quant_int8_retention", "quant_quality_per_byte",
                         "robustness_long_ctx_score", "robustness_noise_score",
                         "init_sensitivity_std", "fp_jacobian_spectral_norm",
                         "scaling_param_efficiency", "scaling_flop_efficiency",
                         "scaling_gate_passed", "scaling_best_family",
                         "scaling_d512_param_efficiency", "scaling_confidence",
                         "routing_savings_ratio", "compression_ratio",
                         "discovery_loss_ratio", "ncd_score",
                         "robustness_long_ctx_scaling_score",
                         "robustness_long_ctx_assoc_score",
                         "robustness_long_ctx_multi_hop_score",
                         "robustness_long_ctx_passkey_score",
                         "robustness_long_ctx_retrieval_aggregate",
                         "robustness_long_ctx_combined_score",
                         "depth_savings_ratio", "recursion_savings_ratio",
                         "activation_sparsity_score", "routing_expert_count",
                         "routing_confidence_mean", "routing_drop_rate",
                         "efficiency_multiple"):
                if col in kwargs and kwargs[col] is not None:
                    cols.append(col)
                    val = kwargs[col]
                    if isinstance(val, bool): val = int(val)
                    vals.append(val)
            
            placeholders = ", ".join(["?"] * len(cols))
            self.conn.execute(
                f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )

        self.notebook._maybe_commit()
        return entry_id

    def get_leaderboard(self, tier: Optional[str] = None,
                        limit: int = 50,
                        sort_by: str = "composite_score",
                        include_family: bool = True,
                        include_references: bool = True) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {"composite_score", "screening_loss_ratio",
                       "investigation_loss_ratio", "validation_loss_ratio",
                       "screening_novelty", "timestamp",
                       "robustness_noise_score", "quant_int8_retention",
                       "robustness_long_ctx_score",
                       "discovery_loss_ratio", "generalization_gap",
                       "efficiency_multiple"}
        if sort_by not in valid_sorts:
            sort_by = "composite_score"

        query = (
            "SELECT l.*, pr.graph_json AS _graph_json, "
            "pr.routing_mode AS _routing_mode, "
            "pr.graph_fingerprint AS _graph_fingerprint, "
            "pr.arch_spec_json AS _arch_spec_json, "
            "pr.param_count AS _param_count, "
            "pr.graph_n_params_estimate AS _graph_n_params_estimate, "
            "pr.novelty_confidence AS _novelty_confidence, "
            "pr.cka_source AS _cka_source, "
            "pr.routing_confidence_mean AS _routing_confidence_mean, "
            "pr.fp_jacobian_spectral_norm AS jacobian_spectral_norm, "
            # Fields for client-side candidateScore computation
            "pr.loss_ratio AS loss_ratio, "
            "pr.discovery_loss AS discovery_loss, "
            "pr.discovery_loss_ratio AS _pr_discovery_loss_ratio, "
            "pr.validation_loss AS validation_loss, "
            "pr.validation_loss_ratio AS _pr_validation_loss_ratio, "
            "pr.generalization_gap AS generalization_gap, "
            "pr.novelty_score AS novelty_score, "
            "pr.final_loss AS final_loss, "
            "pr.throughput_tok_s AS throughput_tok_s, "
            "pr.peak_memory_mb AS peak_memory_mb, "
            "pr.loss_improvement_rate AS loss_improvement_rate, "
            "pr.forward_time_ms AS forward_time_ms, "
            "pr.flops_forward AS flops_forward, "
            "pr.flops_per_param AS flops_per_param, "
            "pr.sparsity_ratio AS sparsity_ratio, "
            "pr.baseline_loss_ratio AS baseline_loss_ratio, "
            "pr.routing_utilization_entropy AS routing_utilization_entropy, "
            "pr.routing_drop_rate AS routing_drop_rate, "
            "pr.routing_confidence_std AS routing_confidence_std, "
            "pr.routing_tokens_total AS routing_tokens_total, "
            "pr.routing_tokens_processed AS routing_tokens_processed, "
            "pr.routing_capacity_overflow_count AS routing_capacity_overflow_count, "
            "pr.depth_savings_ratio AS depth_savings_ratio, "
            "pr.effective_depth_ratio AS effective_depth_ratio, "
            "pr.recursion_savings_ratio AS recursion_savings_ratio, "
            "pr.recursion_depth_ratio AS recursion_depth_ratio, "
            "pr.activation_sparsity_score AS activation_sparsity_score, "
            "pr.routing_expert_count AS routing_expert_count, "
            "pr.routing_confidence_mean AS routing_confidence_mean, "
            "pr.max_viable_seq_len AS max_viable_seq_len, "
            "pr.robustness_long_ctx_scaling_score AS robustness_long_ctx_scaling_score, "
            "pr.robustness_long_ctx_assoc_score AS robustness_long_ctx_assoc_score, "
            "pr.robustness_long_ctx_multi_hop_score AS robustness_long_ctx_multi_hop_score, "
            "pr.robustness_long_ctx_passkey_score AS robustness_long_ctx_passkey_score, "
            "pr.efficiency_multiple AS _pr_efficiency_multiple "
            "FROM leaderboard l "
            "LEFT JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE 1=1"
        )
        params: List[Any] = []
        if tier:
            if include_references:
                query += " AND (l.tier = ? OR COALESCE(l.is_reference, 0) = 1)"
            else:
                query += " AND l.tier = ? AND COALESCE(l.is_reference, 0) = 0"
            params.append(tier)
        elif not include_references:
            query += " AND COALESCE(l.is_reference, 0) = 0"
        oversample = max(limit * 6, 200)
        # Fields sourced from program_results use the SELECT alias directly
        pr_sort_fields = {"discovery_loss_ratio", "generalization_gap"}
        sort_col = sort_by if sort_by in pr_sort_fields else f"l.{sort_by}"
        query += (
            f" ORDER BY COALESCE(l.is_pinned, 0) DESC, "
            f"COALESCE(l.is_reference, 0) DESC, "
            f"{sort_col} DESC NULLS LAST LIMIT ?"
        )
        params.append(oversample)

        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # Prefer leaderboard-curated phase metrics, but backfill from raw
            # program_results when leaderboard fields are absent.
            if d.get("discovery_loss_ratio") is None and d.get("_pr_discovery_loss_ratio") is not None:
                d["discovery_loss_ratio"] = d.get("_pr_discovery_loss_ratio")
            if d.get("validation_loss_ratio") is None and d.get("_pr_validation_loss_ratio") is not None:
                d["validation_loss_ratio"] = d.get("_pr_validation_loss_ratio")
            if include_family:
                d["architecture_family"] = self.notebook._classify_architecture_family(
                    graph_json=d.get("_graph_json"),
                    routing_mode=d.get("_routing_mode"),
                )
            d.pop("_graph_json", None)
            d["routing_mode"] = d.pop("_routing_mode", None)
            d["arch_spec_json"] = d.pop("_arch_spec_json", None)
            d["param_count"] = d.pop("_param_count", None)
            d["graph_n_params_estimate"] = d.pop("_graph_n_params_estimate", None)
            d["novelty_confidence"] = d.pop("_novelty_confidence", None)
            d["cka_source"] = d.pop("_cka_source", None)
            d["routing_confidence_mean"] = d.pop("_routing_confidence_mean", None)
            if d.get("efficiency_multiple") is None and d.get("_pr_efficiency_multiple") is not None:
                d["efficiency_multiple"] = d.get("_pr_efficiency_multiple")
            d.pop("_pr_discovery_loss_ratio", None)
            d.pop("_pr_validation_loss_ratio", None)
            d.pop("_pr_efficiency_multiple", None)
            
            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = json.loads(
                        d["investigation_best_training"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if d.get("is_reference"):
                d["screening_novelty"] = self.notebook._reference_novelty_for_display(
                    d.get("screening_novelty")
                )
                if d.get("novelty_score") is not None:
                    d["novelty_score"] = self.notebook._reference_novelty_for_display(
                        d.get("novelty_score")
                    )
            results.append(d)

        # Separate reference entries so they survive dedup and limit
        references = []
        non_references = []
        for entry in results:
            if include_references and entry.get("is_reference"):
                references.append(entry)
            else:
                non_references.append(entry)

        # Deduplicate references by graph fingerprint first
        seen_ref_fps: Dict[str, int] = {}
        deduped_refs = []
        for entry in references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                if fp in seen_ref_fps:
                    # Keep best reference for this fingerprint
                    existing_idx = seen_ref_fps[fp]
                    if (entry.get("composite_score") or 0) > (deduped_refs[existing_idx].get("composite_score") or 0):
                        deduped_refs[existing_idx] = entry
                    continue
                seen_ref_fps[fp] = len(deduped_refs)
            deduped_refs.append(entry)

        # Deduplicate non-references by graph fingerprint
        seen_fingerprints: Dict[str, int] = {}
        deduped = []
        for entry in non_references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                # If this fingerprint is already in references, skip it in non-references
                if fp in seen_ref_fps:
                    continue
                if fp in seen_fingerprints:
                    # Keep the one with higher composite_score
                    existing_idx = seen_fingerprints[fp]
                    existing_score = deduped[existing_idx].get("composite_score") or 0
                    new_score = entry.get("composite_score") or 0
                    if new_score > existing_score:
                        deduped[existing_idx] = entry
                    continue
                seen_fingerprints[fp] = len(deduped)
            deduped.append(entry)

        # Expose fingerprint as public field, drop internal alias
        for entry in deduped:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
        for entry in deduped_refs:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)

        # Always include reference entries regardless of limit
        merged = deduped[:limit]
        if include_references:
            ref_ids = {e.get("entry_id") for e in merged}
            for ref in deduped_refs:
                if ref.get("entry_id") not in ref_ids:
                    merged.append(ref)
        return merged

    def get_leaderboard_entry(self, result_id: str) -> Optional[Dict]:
        """Fetch a single leaderboard entry by result_id."""
        if not result_id:
            return None
        rows = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(rows) if rows else None

    def pin_reference(self, entry_id: str, reference_name: str) -> None:
        """Pin a leaderboard entry as a reference architecture."""
        self.conn.execute(
            """UPDATE leaderboard
               SET is_reference = 1,
                   reference_name = ?,
                   model_source = 'reference'
               WHERE entry_id = ?""",
            (reference_name, entry_id),
        )
        self.notebook._maybe_commit()

    def set_leaderboard_pin(self, entry_id: str, pinned: bool):
        """Pin or unpin a leaderboard entry for dashboard priority."""
        self.notebook._submit_write(
            "UPDATE leaderboard SET is_pinned = ? WHERE entry_id = ?",
            (1 if pinned else 0, entry_id),
        )

    # ── Pre-investigation gate helpers ──────────────────────────────────

    def get_investigation_eligible(
        self,
        max_lr: float,
        min_stability: float,
        min_spectral_norm: float,
        max_spectral_norm: float,
        min_improvement_rate: float,
        ref_lr_ceiling: Optional[float] = None,
    ) -> List[Dict]:
        """Stage A hard reject: return screening candidates that pass all hard filters.

        Joins program_results with leaderboard to return full metric rows for
        candidates eligible for investigation.
        """
        lr_ceiling = ref_lr_ceiling if ref_lr_ceiling is not None else max_lr
        rows = self.conn.execute(
            """SELECT pr.*, l.entry_id, l.tier, l.composite_score,
                      l.screening_loss_ratio, l.screening_novelty,
                      l.pre_inv_score, l.is_reference, l.reference_name
               FROM program_results pr
               JOIN leaderboard l ON l.result_id = pr.result_id
               WHERE l.tier = 'screening'
                 AND COALESCE(l.is_reference, 0) = 0
                 AND pr.stage1_passed = 1
                 AND COALESCE(pr.has_nan_grad, 0) = 0
                 AND COALESCE(pr.has_nan_output, 0) = 0
                 AND COALESCE(pr.has_inf_output, 0) = 0
                 AND COALESCE(pr.has_zero_grad, 0) = 0
                 AND COALESCE(pr.graph_has_gradient_path, 1) = 1
                 AND COALESCE(pr.stability_score, 0) >= ?
                 AND (pr.fp_jacobian_spectral_norm IS NULL
                      OR (pr.fp_jacobian_spectral_norm >= ? AND pr.fp_jacobian_spectral_norm <= ?))
                 AND COALESCE(pr.loss_improvement_rate, 0) >= ?
                 AND COALESCE(pr.loss_ratio, 1.0) < ?
               ORDER BY pr.loss_ratio ASC NULLS LAST""",
            (min_stability, min_spectral_norm, max_spectral_norm,
             min_improvement_rate, lr_ceiling),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def compute_pre_investigation_score(row: Dict, best_ref_lr: Optional[float] = None) -> float:
        """Stage B composite readiness score (0-100 scale).

        Components:
        - Performance (40pts): loss_ratio, discovery_loss_ratio, loss_improvement_rate
        - Stability (20pts): stability_score, spectral_norm (Gaussian around 1.0), grad_norm_std
        - Novelty (20pts): novelty_score * confidence, structural_novelty, behavioral_novelty
        - Fingerprint quality (10pts): fp_intrinsic_dim, fp_isotropy, fp_rank_ratio
        - Efficiency (10pts): throughput_tok_s, peak_memory_mb
        - Reference penalty (-20pts): if loss_ratio > 1.5 * best_reference_lr
        """
        import math
        score = 0.0

        # ── Performance (40 pts) ──
        lr = row.get("loss_ratio")
        if lr is not None and lr > 0:
            # Lower LR is better; LR=0.1 → 40pts, LR=0.8 → ~8pts
            score += max(0, min(40, 40 * (1.0 - float(lr))))

        dlr = row.get("discovery_loss_ratio")
        if dlr is not None and dlr > 0:
            # Bonus: up to 5pts from discovery loss (replaces top of performance)
            score += max(0, min(5, 5 * (1.0 - float(dlr))))

        lir = row.get("loss_improvement_rate")
        if lir is not None and float(lir) > 0:
            # Up to 5pts for improvement rate
            score += min(5, float(lir) * 10)

        # Cap performance at 40
        score = min(40, score)

        # ── Stability (20 pts) ──
        stab = row.get("stability_score")
        if stab is not None:
            score += min(10, float(stab) * 10)

        sn = row.get("fp_jacobian_spectral_norm")
        if sn is not None and float(sn) > 0:
            # Gaussian centered on 1.0: score = 6 * exp(-(log(sn))^2 / 2)
            log_sn = math.log(float(sn))
            score += max(0, min(6, 6 * math.exp(-log_sn * log_sn / 2.0)))

        gns = row.get("grad_norm_std")
        if gns is not None:
            # Lower grad_norm_std is better; up to 4pts
            score += max(0, min(4, 4 * max(0, 1.0 - float(gns))))

        # ── Novelty (20 pts) ──
        ns = row.get("novelty_score")
        nc = row.get("novelty_confidence")
        if ns is not None:
            conf = float(nc) if nc is not None else 0.5
            score += min(10, float(ns) * conf * 10)

        sn_nov = row.get("structural_novelty")
        if sn_nov is not None:
            score += min(5, float(sn_nov) * 5)

        bn = row.get("behavioral_novelty")
        if bn is not None:
            score += min(5, float(bn) * 5)

        # ── Fingerprint quality (10 pts) ──
        fid = row.get("fp_intrinsic_dim")
        if fid is not None and float(fid) > 0:
            # Higher intrinsic dim → better; up to 4pts, cap at dim=20
            score += min(4, float(fid) / 5.0)

        fiso = row.get("fp_isotropy")
        if fiso is not None:
            score += min(3, float(fiso) * 3)

        frr = row.get("fp_rank_ratio")
        if frr is not None:
            score += min(3, float(frr) * 3)

        # ── Efficiency (10 pts) ──
        tp = row.get("throughput_tok_s")
        if tp is not None and float(tp) > 0:
            # Higher throughput → better; up to 5pts, 10k tok/s → 5pts
            score += min(5, float(tp) / 2000.0)

        mem = row.get("peak_memory_mb")
        if mem is not None and float(mem) > 0:
            # Lower memory → better; up to 5pts, 100MB → 5pts, 500MB → 1pt
            score += max(0, min(5, 5 * (1.0 - float(mem) / 600.0)))

        # ── Reference penalty (-20 pts) ──
        if best_ref_lr is not None and lr is not None:
            if float(lr) > 1.5 * float(best_ref_lr):
                score -= 20

        return max(0, min(100, round(score, 2)))

    def get_references(self) -> List[Dict]:
        """Get all pinned reference architectures."""
        rows = self.conn.execute(
            """SELECT l.*, pr.graph_json AS _graph_json,
                      pr.routing_mode AS _routing_mode,
                      pr.graph_fingerprint AS _graph_fingerprint
               FROM leaderboard l
               LEFT JOIN program_results pr ON pr.result_id = l.result_id
               WHERE COALESCE(l.is_reference, 0) = 1
               ORDER BY l.composite_score DESC NULLS LAST, l.reference_name ASC, l.timestamp DESC"""
        ).fetchall()
        refs: List[Dict] = []
        for row in rows:
            entry = dict(row)
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
            entry["architecture_family"] = self.notebook._classify_architecture_family(
                graph_json=entry.pop("_graph_json", None),
                routing_mode=entry.pop("_routing_mode", None),
            )
            entry["screening_novelty"] = self.notebook._reference_novelty_for_display(
                entry.get("screening_novelty")
            )
            refs.append(entry)
        return refs

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

    @staticmethod
    def _classify_architecture_family(
        graph_json: Optional[str],
        routing_mode: Optional[str],
    ) -> str:
        """Map graph structure to a compact architecture family label."""
        if routing_mode:
            return "Routed-MoE"
        if not graph_json:
            return "Unknown"

        try:
            graph = json.loads(graph_json)
            nodes = graph.get("nodes")
            if isinstance(nodes, dict):
                node_iter = [n for n in nodes.values() if isinstance(n, dict)]
            elif isinstance(nodes, list):
                node_iter = [n for n in nodes if isinstance(n, dict)]
            else:
                node_iter = []
            ops = {str(n.get("op_name", "")).strip() for n in node_iter}
        except (json.JSONDecodeError, TypeError, ValueError):
            return "Unknown"

        if not ops:
            return "Unknown"

        attention_ops = {
            "attention", "self_attention", "mha", "multihead_attention", "qkv_attention",
            "softmax_attention", "linear_attention",
        }
        conv_ops = {"conv1d", "conv1d_seq", "depthwise_conv1d", "conv_only"}
        spectral_ops = {"sin", "cos", "fft", "ifft", "fourier_mix", "fourier_mixing", "rfft_seq", "irfft_seq"}
        gating_ops = {"sigmoid", "tanh", "silu", "gelu", "maximum", "minimum", "swiglu", "topk_gate", "moe_topk"}
        mlp_ops = {"linear_proj", "linear_proj_up", "linear_proj_down", "learnable_bias", "swiglu_mlp"}
        ssm_ops = {"state_space", "selective_scan"}
        adaptive_ops = {"mod_topk", "early_exit", "adaptive_recursion", "fixed_point_iter"}

        has_attention = bool(ops & attention_ops)
        has_conv = bool(ops & conv_ops)
        has_spectral = bool(ops & spectral_ops)
        has_gating = bool(ops & gating_ops)
        has_mlp = bool(ops & mlp_ops)
        has_ssm = bool(ops & ssm_ops)
        has_adaptive = bool(ops & adaptive_ops) or routing_mode in ("mod_topk", "early_exit", "adaptive_recursion")

        family = "Hybrid-Mixer"
        if has_ssm:
            family = "Mamba-SSM" if not has_attention else "Hybrid-SSM"
        elif has_attention:
            if has_conv or has_spectral or has_gating:
                family = "Hybrid-Attention"
            else:
                family = "Attention"
        elif has_conv and has_spectral:
            family = "Spectral-Conv"
        elif has_spectral:
            family = "Spectral-Mixer"
        elif has_conv:
            family = "Conv-Mixer"
        elif has_gating and has_mlp:
            family = "Gated-MLP"
        elif has_mlp:
            family = "MLP-Mixer"
        elif has_gating:
            family = "Nonlinear-Mixer"

        # Apply modifiers
        if routing_mode == "moe_topk" or "moe_topk" in ops:
            family = f"MoE-{family}"
        if has_adaptive:
            family = f"Adaptive-{family}"

        return family

    def promote_to_tier(self, entry_id: str, tier: str,
                        **kwargs) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        sets = ["tier = ?"]
        params: List[Any] = [tier]

        # Sanitize all incoming values
        kwargs = self.notebook._sanitize_numeric(kwargs)

        for col in ("investigation_loss_ratio", "investigation_robustness",
                     "investigation_best_training", "investigation_passed",
                     "validation_loss_ratio", "validation_baseline_ratio",
                     "validation_multi_seed_std", "validation_passed",
                     "normalized_baseline_ratio", "param_efficiency",
                     "quant_int8_retention", "quant_quality_per_byte",
                     "robustness_long_ctx_score", "robustness_noise_score",
                     "init_sensitivity_std", "fp_jacobian_spectral_norm",
                     "scaling_param_efficiency", "scaling_flop_efficiency",
                     "scaling_gate_passed", "scaling_best_family",
                     "scaling_d512_param_efficiency", "scaling_confidence",
                     "routing_savings_ratio", "compression_ratio",
                     "activation_sparsity_score", "dead_neuron_ratio",
                     "routing_collapse_score",
                     "wikitext_perplexity", "wikitext_score",
                     "tinystories_perplexity", "tinystories_score",
                     "cross_task_score",
                     "efficiency_wall_score", "max_viable_seq_len",
                     "scaling_regime",
                     "notes"):
            if col in kwargs and kwargs[col] is not None:
                sets.append(f"{col} = ?")
                val = kwargs[col]
                if isinstance(val, bool):
                    val = int(val)
                params.append(val)

        # Recompute composite score
        row = self.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row:
            d = dict(row)
            # Only update with non-None values from kwargs
            d.update({k: v for k, v in kwargs.items() if v is not None})
            # Look up novelty_confidence from linked program_results
            nov_conf = None
            if d.get("result_id"):
                pr = self.conn.execute(
                    "SELECT novelty_confidence FROM program_results WHERE result_id = ?",
                    (d["result_id"],),
                ).fetchone()
                if pr:
                    nov_conf = pr["novelty_confidence"]
            n_routing = self.notebook._count_routing_ops(d["result_id"]) if d.get("result_id") else None
            composite = self.compute_composite_score(
                screening_lr=d.get("screening_loss_ratio"),
                screening_nov=d.get("screening_novelty"),
                inv_lr=d.get("investigation_loss_ratio"),
                inv_robust=d.get("investigation_robustness"),
                val_lr=d.get("validation_loss_ratio"),
                val_baseline=d.get("validation_baseline_ratio"),
                val_std=d.get("validation_multi_seed_std"),
                novelty_confidence=nov_conf,
                scaling_param_efficiency=d.get("scaling_param_efficiency"),
                is_reference=bool(d.get("is_reference")),
                routing_savings=d.get("routing_savings_ratio"),
                compression_ratio=d.get("compression_ratio"),
                n_routing_ops=n_routing,
            )
            sets.append("composite_score = ?")
            params.append(composite)

        sets.append("timestamp = ?")
        params.append(time.time())
        params.append(entry_id)

        self.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
            params,
        )
        try:
            rid_row = self.conn.execute(
                "SELECT result_id FROM leaderboard WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if rid_row and rid_row["result_id"]:
                self._sync_fingerprint_leaderboard(str(rid_row["result_id"]))
        except Exception as e:
            LOGGER.debug("Fingerprint leaderboard sync skipped for entry %s: %s", entry_id, e)
        self.notebook._maybe_commit()


    def upsert_leaderboard(
        self,
        result_id: str,
        model_source: str,
        architecture_desc: str = "",
        tier: str = "screening",
        tags: Optional[str] = None,
        notes: Optional[str] = None,
        is_reference: bool = False,
        reference_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Insert or update a leaderboard entry.

        Accepts all leaderboard columns as keyword arguments.
        Fields are only updated if provided and not None (prevents accidental NULLing).
        """
        # Check if entry exists for this result_id
        existing = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()

        # Combine kwargs with existing data for composite score recomputation
        d = dict(existing) if existing else {}
        # Sanitize all incoming values
        kwargs = self.notebook._sanitize_numeric(kwargs)
        
        d.update({k: v for k, v in kwargs.items() if v is not None})
        if tags is not None: d["tags"] = tags
        if notes is not None: d["notes"] = notes
        d["tier"] = tier
        d["model_source"] = model_source
        if architecture_desc: d["architecture_desc"] = architecture_desc
        d["is_reference"] = int(is_reference)
        if reference_name: d["reference_name"] = reference_name

        # Look up novelty_confidence from linked program_results
        nov_conf = d.get("novelty_confidence")
        if nov_conf is None:
            pr = self.conn.execute(
                "SELECT novelty_confidence FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if pr:
                nov_conf = pr["novelty_confidence"]

        composite = self.compute_composite_score(
            screening_lr=d.get("screening_loss_ratio"),
            screening_nov=d.get("screening_novelty"),
            inv_lr=d.get("investigation_loss_ratio"),
            inv_robust=d.get("investigation_robustness"),
            val_lr=d.get("validation_loss_ratio"),
            val_baseline=d.get("validation_baseline_ratio"),
            val_std=d.get("validation_multi_seed_std"),
            novelty_confidence=nov_conf,
            scaling_param_efficiency=d.get("scaling_param_efficiency"),
            is_reference=bool(is_reference),
            routing_savings=d.get("routing_savings_ratio"),
            compression_ratio=d.get("compression_ratio"),
            discovery_lr=d.get("discovery_loss_ratio"),
            spectral_norm=d.get("fp_jacobian_spectral_norm"),
            robustness_noise=d.get("robustness_noise_score"),
            quant_retention=d.get("quant_int8_retention"),
            long_ctx_score=d.get("robustness_long_ctx_score"),
            init_std=d.get("init_sensitivity_std"),
            loss_improvement_rate=d.get("loss_improvement_rate"),
            quant_quality_per_byte=d.get("quant_quality_per_byte"),
            ncd_score=d.get("ncd_score"),
            n_routing_ops=self.notebook._count_routing_ops(result_id),
            n_sparse_ops=self.notebook._count_sparse_ops(result_id),
            n_moe_ops=self.notebook._count_moe_ops(result_id),
            recursion_savings=d.get("recursion_savings_ratio"),
            depth_savings=d.get("depth_savings_ratio"),
            activation_sparsity=d.get("activation_sparsity_score"),
            max_viable_seq_len=d.get("max_viable_seq_len"),
            long_ctx_scaling=d.get("robustness_long_ctx_scaling_score"),
            long_ctx_passkey=d.get("robustness_long_ctx_passkey_score"),
            long_ctx_multi_hop=d.get("robustness_long_ctx_multi_hop_score"),
            long_ctx_assoc=d.get("robustness_long_ctx_assoc_score"),
            routing_expert_count=d.get("routing_expert_count"),
            routing_confidence_mean=d.get("routing_confidence_mean"),
            routing_drop_rate=d.get("routing_drop_rate"),
        )

        # Compute efficiency_multiple from program_results operational metrics
        eff_mult = kwargs.get("efficiency_multiple")
        if eff_mult is None:
            pr_row = self.conn.execute(
                "SELECT loss_ratio, param_count, flops_forward, "
                "throughput_tok_s, peak_memory_mb, forward_time_ms "
                "FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if pr_row:
                eff_result = self.compute_efficiency_multiple(
                    loss_ratio=pr_row["loss_ratio"],
                    param_count=pr_row["param_count"],
                    flops_forward=pr_row["flops_forward"],
                    throughput_tok_s=pr_row["throughput_tok_s"],
                    peak_memory_mb=pr_row["peak_memory_mb"],
                    forward_time_ms=pr_row["forward_time_ms"],
                )
                if eff_result is not None:
                    eff_mult = eff_result["geomean"]
        if eff_mult is not None:
            kwargs["efficiency_multiple"] = eff_mult

        if existing:
            entry_id = existing["entry_id"]
            sets = ["timestamp = ?", "model_source = ?", "tier = ?", "composite_score = ?", "is_reference = ?"]
            params = [time.time(), model_source, tier, composite, int(is_reference)]
            
            if architecture_desc:
                sets.append("architecture_desc = ?")
                params.append(architecture_desc)
            if tags is not None:
                sets.append("tags = ?")
                params.append(tags)
            if notes is not None:
                sets.append("notes = ?")
                params.append(notes)
            if reference_name is not None:
                sets.append("reference_name = ?")
                params.append(reference_name)

            # Whitelist for other columns from kwargs
            for col in ("screening_loss_ratio", "screening_novelty", "screening_passed",
                         "investigation_loss_ratio", "investigation_robustness",
                         "investigation_best_training", "investigation_passed",
                         "validation_loss_ratio", "validation_baseline_ratio",
                         "validation_multi_seed_std", "validation_passed",
                         "normalized_baseline_ratio", "param_efficiency",
                         "quant_int8_retention", "quant_quality_per_byte",
                         "robustness_long_ctx_score", "robustness_noise_score",
                         "init_sensitivity_std", "fp_jacobian_spectral_norm",
                         "scaling_param_efficiency", "scaling_flop_efficiency",
                         "scaling_gate_passed", "scaling_best_family",
                         "scaling_d512_param_efficiency", "scaling_confidence",
                         "routing_savings_ratio", "compression_ratio",
                         "discovery_loss_ratio", "ncd_score",
                         "robustness_long_ctx_scaling_score",
                         "robustness_long_ctx_assoc_score",
                         "robustness_long_ctx_multi_hop_score",
                         "robustness_long_ctx_passkey_score",
                         "robustness_long_ctx_retrieval_aggregate",
                         "robustness_long_ctx_combined_score",
                         "depth_savings_ratio", "recursion_savings_ratio",
                         "activation_sparsity_score", "routing_expert_count",
                         "routing_confidence_mean", "routing_drop_rate",
                         "efficiency_multiple",
                         "wikitext_perplexity", "wikitext_score",
                         "tinystories_perplexity", "tinystories_score"):
                if col in kwargs and kwargs[col] is not None:
                    sets.append(f"{col} = ?")
                    val = kwargs[col]
                    if isinstance(val, bool): val = int(val)
                    params.append(val)

            params.append(entry_id)
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )
        else:
            entry_id = str(uuid.uuid4())[:12]
            cols = ["entry_id", "result_id", "timestamp", "model_source", "architecture_desc",
                    "tier", "composite_score", "is_reference", "reference_name", "tags", "notes"]
            vals = [entry_id, result_id, time.time(), model_source, architecture_desc,
                    tier, composite, int(is_reference), reference_name, tags, notes]

            for col in ("screening_loss_ratio", "screening_novelty", "screening_passed",
                         "investigation_loss_ratio", "investigation_robustness",
                         "investigation_best_training", "investigation_passed",
                         "validation_loss_ratio", "validation_baseline_ratio",
                         "validation_multi_seed_std", "validation_passed",
                         "normalized_baseline_ratio", "param_efficiency",
                         "quant_int8_retention", "quant_quality_per_byte",
                         "robustness_long_ctx_score", "robustness_noise_score",
                         "init_sensitivity_std", "fp_jacobian_spectral_norm",
                         "scaling_param_efficiency", "scaling_flop_efficiency",
                         "scaling_gate_passed", "scaling_best_family",
                         "scaling_d512_param_efficiency", "scaling_confidence",
                         "routing_savings_ratio", "compression_ratio",
                         "discovery_loss_ratio", "ncd_score",
                         "robustness_long_ctx_scaling_score",
                         "robustness_long_ctx_assoc_score",
                         "robustness_long_ctx_multi_hop_score",
                         "robustness_long_ctx_passkey_score",
                         "robustness_long_ctx_retrieval_aggregate",
                         "robustness_long_ctx_combined_score",
                         "depth_savings_ratio", "recursion_savings_ratio",
                         "activation_sparsity_score", "routing_expert_count",
                         "routing_confidence_mean", "routing_drop_rate",
                         "efficiency_multiple"):
                if col in kwargs and kwargs[col] is not None:
                    cols.append(col)
                    val = kwargs[col]
                    if isinstance(val, bool): val = int(val)
                    vals.append(val)
            
            placeholders = ", ".join(["?"] * len(cols))
            self.conn.execute(
                f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )

        self.notebook._maybe_commit()
        return entry_id

    def get_leaderboard(self, tier: Optional[str] = None,
                        limit: int = 50,
                        sort_by: str = "composite_score",
                        include_family: bool = True,
                        include_references: bool = True) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {"composite_score", "screening_loss_ratio",
                       "investigation_loss_ratio", "validation_loss_ratio",
                       "screening_novelty", "timestamp",
                       "robustness_noise_score", "quant_int8_retention",
                       "robustness_long_ctx_score",
                       "discovery_loss_ratio", "generalization_gap",
                       "efficiency_multiple"}
        if sort_by not in valid_sorts:
            sort_by = "composite_score"

        query = (
            "SELECT l.*, pr.graph_json AS _graph_json, "
            "pr.routing_mode AS _routing_mode, "
            "pr.graph_fingerprint AS _graph_fingerprint, "
            "pr.arch_spec_json AS _arch_spec_json, "
            "pr.param_count AS _param_count, "
            "pr.graph_n_params_estimate AS _graph_n_params_estimate, "
            "pr.novelty_confidence AS _novelty_confidence, "
            "pr.cka_source AS _cka_source, "
            "pr.routing_confidence_mean AS _routing_confidence_mean, "
            "pr.fp_jacobian_spectral_norm AS jacobian_spectral_norm, "
            # Fields for client-side candidateScore computation
            "pr.loss_ratio AS loss_ratio, "
            "pr.discovery_loss AS discovery_loss, "
            "pr.discovery_loss_ratio AS _pr_discovery_loss_ratio, "
            "pr.validation_loss AS validation_loss, "
            "pr.validation_loss_ratio AS _pr_validation_loss_ratio, "
            "pr.generalization_gap AS generalization_gap, "
            "pr.novelty_score AS novelty_score, "
            "pr.final_loss AS final_loss, "
            "pr.throughput_tok_s AS throughput_tok_s, "
            "pr.peak_memory_mb AS peak_memory_mb, "
            "pr.loss_improvement_rate AS loss_improvement_rate, "
            "pr.forward_time_ms AS forward_time_ms, "
            "pr.flops_forward AS flops_forward, "
            "pr.flops_per_param AS flops_per_param, "
            "pr.sparsity_ratio AS sparsity_ratio, "
            "pr.baseline_loss_ratio AS baseline_loss_ratio, "
            "pr.routing_utilization_entropy AS routing_utilization_entropy, "
            "pr.routing_drop_rate AS routing_drop_rate, "
            "pr.routing_confidence_std AS routing_confidence_std, "
            "pr.routing_tokens_total AS routing_tokens_total, "
            "pr.routing_tokens_processed AS routing_tokens_processed, "
            "pr.routing_capacity_overflow_count AS routing_capacity_overflow_count, "
            "pr.depth_savings_ratio AS depth_savings_ratio, "
            "pr.effective_depth_ratio AS effective_depth_ratio, "
            "pr.recursion_savings_ratio AS recursion_savings_ratio, "
            "pr.recursion_depth_ratio AS recursion_depth_ratio, "
            "pr.activation_sparsity_score AS activation_sparsity_score, "
            "pr.routing_expert_count AS routing_expert_count, "
            "pr.routing_confidence_mean AS routing_confidence_mean, "
            "pr.max_viable_seq_len AS max_viable_seq_len, "
            "pr.robustness_long_ctx_scaling_score AS robustness_long_ctx_scaling_score, "
            "pr.robustness_long_ctx_assoc_score AS robustness_long_ctx_assoc_score, "
            "pr.robustness_long_ctx_multi_hop_score AS robustness_long_ctx_multi_hop_score, "
            "pr.robustness_long_ctx_passkey_score AS robustness_long_ctx_passkey_score, "
            "pr.efficiency_multiple AS _pr_efficiency_multiple "
            "FROM leaderboard l "
            "LEFT JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE 1=1"
        )
        params: List[Any] = []
        if tier:
            if include_references:
                query += " AND (l.tier = ? OR COALESCE(l.is_reference, 0) = 1)"
            else:
                query += " AND l.tier = ? AND COALESCE(l.is_reference, 0) = 0"
            params.append(tier)
        elif not include_references:
            query += " AND COALESCE(l.is_reference, 0) = 0"
        oversample = max(limit * 6, 200)
        # Fields sourced from program_results use the SELECT alias directly
        pr_sort_fields = {"discovery_loss_ratio", "generalization_gap"}
        sort_col = sort_by if sort_by in pr_sort_fields else f"l.{sort_by}"
        query += (
            f" ORDER BY COALESCE(l.is_pinned, 0) DESC, "
            f"COALESCE(l.is_reference, 0) DESC, "
            f"{sort_col} DESC NULLS LAST LIMIT ?"
        )
        params.append(oversample)

        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # Prefer leaderboard-curated phase metrics, but backfill from raw
            # program_results when leaderboard fields are absent.
            if d.get("discovery_loss_ratio") is None and d.get("_pr_discovery_loss_ratio") is not None:
                d["discovery_loss_ratio"] = d.get("_pr_discovery_loss_ratio")
            if d.get("validation_loss_ratio") is None and d.get("_pr_validation_loss_ratio") is not None:
                d["validation_loss_ratio"] = d.get("_pr_validation_loss_ratio")
            if include_family:
                d["architecture_family"] = self.notebook._classify_architecture_family(
                    graph_json=d.get("_graph_json"),
                    routing_mode=d.get("_routing_mode"),
                )
            d.pop("_graph_json", None)
            d["routing_mode"] = d.pop("_routing_mode", None)
            d["arch_spec_json"] = d.pop("_arch_spec_json", None)
            d["param_count"] = d.pop("_param_count", None)
            d["graph_n_params_estimate"] = d.pop("_graph_n_params_estimate", None)
            d["novelty_confidence"] = d.pop("_novelty_confidence", None)
            d["cka_source"] = d.pop("_cka_source", None)
            d["routing_confidence_mean"] = d.pop("_routing_confidence_mean", None)
            if d.get("efficiency_multiple") is None and d.get("_pr_efficiency_multiple") is not None:
                d["efficiency_multiple"] = d.get("_pr_efficiency_multiple")
            d.pop("_pr_discovery_loss_ratio", None)
            d.pop("_pr_validation_loss_ratio", None)
            d.pop("_pr_efficiency_multiple", None)
            
            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = json.loads(
                        d["investigation_best_training"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if d.get("is_reference"):
                d["screening_novelty"] = self.notebook._reference_novelty_for_display(
                    d.get("screening_novelty")
                )
                if d.get("novelty_score") is not None:
                    d["novelty_score"] = self.notebook._reference_novelty_for_display(
                        d.get("novelty_score")
                    )
            results.append(d)

        # Separate reference entries so they survive dedup and limit
        references = []
        non_references = []
        for entry in results:
            if include_references and entry.get("is_reference"):
                references.append(entry)
            else:
                non_references.append(entry)

        # Deduplicate references by graph fingerprint first
        seen_ref_fps: Dict[str, int] = {}
        deduped_refs = []
        for entry in references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                if fp in seen_ref_fps:
                    # Keep best reference for this fingerprint
                    existing_idx = seen_ref_fps[fp]
                    if (entry.get("composite_score") or 0) > (deduped_refs[existing_idx].get("composite_score") or 0):
                        deduped_refs[existing_idx] = entry
                    continue
                seen_ref_fps[fp] = len(deduped_refs)
            deduped_refs.append(entry)

        # Deduplicate non-references by graph fingerprint
        seen_fingerprints: Dict[str, int] = {}
        deduped = []
        for entry in non_references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                # If this fingerprint is already in references, skip it in non-references
                if fp in seen_ref_fps:
                    continue
                if fp in seen_fingerprints:
                    # Keep the one with higher composite_score
                    existing_idx = seen_fingerprints[fp]
                    existing_score = deduped[existing_idx].get("composite_score") or 0
                    new_score = entry.get("composite_score") or 0
                    if new_score > existing_score:
                        deduped[existing_idx] = entry
                    continue
                seen_fingerprints[fp] = len(deduped)
            deduped.append(entry)

        # Expose fingerprint as public field, drop internal alias
        for entry in deduped:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
        for entry in deduped_refs:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)

        # Always include reference entries regardless of limit
        merged = deduped[:limit]
        if include_references:
            ref_ids = {e.get("entry_id") for e in merged}
            for ref in deduped_refs:
                if ref.get("entry_id") not in ref_ids:
                    merged.append(ref)
        return merged

    def get_leaderboard_entry(self, result_id: str) -> Optional[Dict]:
        """Fetch a single leaderboard entry by result_id."""
        if not result_id:
            return None
        rows = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(rows) if rows else None

    def pin_reference(self, entry_id: str, reference_name: str) -> None:
        """Pin a leaderboard entry as a reference architecture."""
        self.conn.execute(
            """UPDATE leaderboard
               SET is_reference = 1,
                   reference_name = ?,
                   model_source = 'reference'
               WHERE entry_id = ?""",
            (reference_name, entry_id),
        )
        self.notebook._maybe_commit()

    def set_leaderboard_pin(self, entry_id: str, pinned: bool):
        """Pin or unpin a leaderboard entry for dashboard priority."""
        self.notebook._submit_write(
            "UPDATE leaderboard SET is_pinned = ? WHERE entry_id = ?",
            (1 if pinned else 0, entry_id),
        )

