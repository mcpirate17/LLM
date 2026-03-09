from __future__ import annotations
"""Auto-extracted mixin for LabNotebook."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ._shared import LOGGER, sanitize_for_db


class _LeaderboardMixin:
    """Leaderboard operations for the Lab Notebook."""
    __slots__ = ()

    def _highest_tier(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        tiers = [str(r.get("tier") or "").lower() for r in rows if r.get("tier")]
        if not tiers:
            return None
        return max(tiers, key=lambda t: self._TIER_ORDER.get(t, -1))


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
        kwargs = sanitize_for_db(kwargs)
        
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
            n_routing_ops=self._count_routing_ops(result_id),
            n_sparse_ops=self._count_sparse_ops(result_id),
            n_moe_ops=self._count_moe_ops(result_id),
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

        self._maybe_commit()
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
                d["architecture_family"] = self._classify_architecture_family(
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
                d["screening_novelty"] = self._reference_novelty_for_display(
                    d.get("screening_novelty")
                )
                if d.get("novelty_score") is not None:
                    d["novelty_score"] = self._reference_novelty_for_display(
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


    def set_leaderboard_pin(self, entry_id: str, pinned: bool):
        """Pin or unpin a leaderboard entry for dashboard priority."""
        self._submit_write(
            "UPDATE leaderboard SET is_pinned = ? WHERE entry_id = ?",
            (1 if pinned else 0, entry_id),
        )


    def promote_to_tier(self, entry_id: str, tier: str,
                        **kwargs) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        sets = ["tier = ?"]
        params: List[Any] = [tier]

        # Sanitize all incoming values
        kwargs = sanitize_for_db(kwargs)

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
            n_routing = self._count_routing_ops(d["result_id"]) if d.get("result_id") else None
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
        self._maybe_commit()


    # ── Scaling Summary ──

    def get_scaling_summary(self) -> Dict:
        """Get a summary of scaling gate results for Aria's context.

        Returns aggregate stats on how candidates compare to external
        baselines (GPT-2/Mamba) in parameter efficiency, plus the best
        and worst performers.
        """
        rows = self.conn.execute(
            """SELECT l.entry_id, l.scaling_param_efficiency, l.scaling_flop_efficiency,
                      l.scaling_gate_passed, l.scaling_best_family, l.scaling_confidence,
                      l.screening_loss_ratio, l.screening_novelty, l.composite_score,
                      pr.graph_fingerprint
               FROM leaderboard l
               JOIN program_results pr ON l.result_id = pr.result_id
               WHERE l.scaling_param_efficiency IS NOT NULL
               ORDER BY l.scaling_param_efficiency DESC"""
        ).fetchall()
        if not rows:
            return {
                "n_evaluated": 0,
                "n_gate_passed": 0,
                "message": "No candidates have been evaluated against external scaling laws yet.",
            }

        entries = [dict(r) for r in rows]
        n_passed = sum(1 for e in entries if e.get("scaling_gate_passed"))
        efficiencies = [e["scaling_param_efficiency"] for e in entries]

        return {
            "n_evaluated": len(entries),
            "n_gate_passed": n_passed,
            "target": 3.0,
            "best_param_efficiency": max(efficiencies),
            "worst_param_efficiency": min(efficiencies),
            "mean_param_efficiency": sum(efficiencies) / len(efficiencies),
            "best_entry": {
                "fingerprint": (entries[0].get("graph_fingerprint") or "")[:12],
                "param_efficiency": entries[0]["scaling_param_efficiency"],
                "family": entries[0].get("scaling_best_family", "gpt2"),
                "loss_ratio": entries[0].get("screening_loss_ratio"),
            },
            "worst_entry": {
                "fingerprint": (entries[-1].get("graph_fingerprint") or "")[:12],
                "param_efficiency": entries[-1]["scaling_param_efficiency"],
                "loss_ratio": entries[-1].get("screening_loss_ratio"),
            },
            "entries": [
                {
                    "fingerprint": (e.get("graph_fingerprint") or "")[:12],
                    "param_eff": round(e["scaling_param_efficiency"], 2),
                    "flop_eff": round(e.get("scaling_flop_efficiency") or 0, 2),
                    "gate": bool(e.get("scaling_gate_passed")),
                    "loss_ratio": round(e.get("screening_loss_ratio") or 0, 4),
                }
                for e in entries[:10]
            ],
        }

