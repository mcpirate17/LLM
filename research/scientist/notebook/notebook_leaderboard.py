from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ._shared import LOGGER, sanitize_for_db
from ..leaderboard_scoring import SCORING_VERSION, build_score_kwargs, compute_composite
from ..thresholds import TIER_RANK
from ..trust_policy import is_promotable_entry, sql_trusted_clause


class DuplicateLeaderboardFingerprintError(Exception):
    """Raised when ``upsert_leaderboard`` would create a second leaderboard
    row for a ``graph_fingerprint`` that already has an entry under a
    different ``result_id``.

    Pass ``allow_fingerprint_duplicate=True`` to bypass, or resolve by calling
    ``promote_to_tier(existing_entry_id, ...)`` on the pre-existing entry so
    metrics merge onto one row instead of creating a duplicate.
    """

    def __init__(
        self,
        graph_fingerprint: str,
        existing_entry_id: str,
        existing_result_id: str,
        attempted_result_id: str,
    ) -> None:
        self.graph_fingerprint = graph_fingerprint
        self.existing_entry_id = existing_entry_id
        self.existing_result_id = existing_result_id
        self.attempted_result_id = attempted_result_id
        super().__init__(
            f"fingerprint {graph_fingerprint} is already on the leaderboard at "
            f"entry_id={existing_entry_id} (result_id={existing_result_id}); "
            f"attempted insert for result_id={attempted_result_id}. "
            f"Call promote_to_tier() on the existing entry, or pass "
            f"allow_fingerprint_duplicate=True if the duplicate is intentional."
        )


_LEADERBOARD_MANAGED_COLUMNS = frozenset(
    {
        "entry_id",
        "result_id",
        "timestamp",
        "model_source",
        "architecture_desc",
        "tier",
        "composite_score",
        "is_reference",
        "reference_name",
        "tags",
        "notes",
    }
)


class _LeaderboardMixin:
    """Leaderboard operations for the Lab Notebook."""

    __slots__ = ()

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return num if num == num else None

    def _normalize_benchmark_fields(self, entry: Dict[str, Any]) -> None:
        """Backfill stable benchmark aliases from persisted artifact payloads."""
        raw_payload = entry.pop("_external_benchmarks_json", None)
        payload = None
        if raw_payload and isinstance(raw_payload, str):
            try:
                payload = json.loads(raw_payload)
            except (json.JSONDecodeError, TypeError):
                payload = None

        screening = (
            payload.get("screening_wikitext") if isinstance(payload, dict) else None
        )
        screening_metrics = (
            screening.get("metrics") if isinstance(screening, dict) else {}
        )

        wikitext_ppl = self._coerce_float(
            entry.get("wikitext_ppl")
            or entry.get("wikitext_perplexity")
            or screening_metrics.get("wikitext_perplexity")
        )
        if wikitext_ppl is not None:
            entry["wikitext_ppl"] = wikitext_ppl
            entry.setdefault("peak_ppl", wikitext_ppl)

        improvement_ratio = self._coerce_float(
            entry.get("wikitext_ppl_improvement_ratio")
            or entry.get("wikitext_improvement_ratio")
            or entry.get("wikitext_ppl_improvement")
            or screening_metrics.get("wikitext_ppl_improvement")
        )
        if improvement_ratio is not None:
            entry["improvement_ratio"] = improvement_ratio

        if screening:
            entry.setdefault("screening_wikitext_status", screening.get("status"))
            entry.setdefault(
                "screening_wikitext_metric_version", screening.get("metric_version")
            )
            entry.setdefault("screening_wikitext_variant", screening.get("variant"))
            elapsed_ms = self._coerce_float(screening.get("elapsed_ms"))
            if elapsed_ms is not None:
                entry.setdefault("screening_wikitext_elapsed_ms", elapsed_ms)

        trajectory_payload = (
            payload.get("wikitext_trajectory") if isinstance(payload, dict) else None
        )
        checkpoints = (
            trajectory_payload.get("checkpoints")
            if isinstance(trajectory_payload, dict)
            else None
        )
        if isinstance(checkpoints, dict):
            ordered_steps = []
            for step, values in checkpoints.items():
                try:
                    step_num = int(step)
                except (TypeError, ValueError):
                    continue
                if not isinstance(values, dict):
                    continue
                ppl = self._coerce_float(values.get("ppl"))
                if ppl is None:
                    continue
                ordered_steps.append((step_num, ppl))
            ordered_steps.sort(key=lambda item: item[0])
            if ordered_steps:
                trajectory = [ppl for _, ppl in ordered_steps]
                entry["wikitext_ppl_trajectory"] = trajectory
                entry["peak_ppl"] = min(trajectory)
                entry["eval_budget_steps"] = ordered_steps[-1][0]
                if len(trajectory) >= 2 and trajectory[1] > 0:
                    entry.setdefault("improvement_ratio", trajectory[0] / trajectory[1])

    def _highest_tier(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        tiers = [str(r.get("tier") or "").lower() for r in rows if r.get("tier")]
        if not tiers:
            return None
        return max(tiers, key=lambda t: self._TIER_ORDER.get(t, -1))

    def _leaderboard_update_items(
        self, kwargs: Dict[str, Any]
    ) -> List[tuple[str, Any]]:
        allowed = self._get_leaderboard_columns() - _LEADERBOARD_MANAGED_COLUMNS
        update_items: List[tuple[str, Any]] = []
        for col, val in kwargs.items():
            if col not in allowed or val is None:
                continue
            update_items.append((col, int(val) if isinstance(val, bool) else val))
        return update_items

    @staticmethod
    def _provenance_complete(pr_row: Any) -> bool:
        if not pr_row:
            return False
        raw = (
            pr_row["data_provenance_json"]
            if "data_provenance_json" in pr_row.keys()
            else None
        )
        if not raw or not isinstance(raw, str):
            return False
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
        return bool(payload.get("provenance_complete"))

    def _resolve_allowed_tier(
        self,
        *,
        requested_tier: str,
        existing_tier: str,
        pr_row: Any,
        is_reference: bool,
    ) -> str:
        requested_rank = TIER_RANK.get(requested_tier, 0)
        existing_rank = TIER_RANK.get(existing_tier, 0)
        if requested_rank <= 0 or requested_rank <= existing_rank:
            return requested_tier if requested_rank >= existing_rank else existing_tier
        if is_reference:
            return requested_tier
        trust_entry = dict(pr_row) if pr_row else {}
        if is_promotable_entry(trust_entry) and self._provenance_complete(pr_row):
            return requested_tier
        LOGGER.warning(
            "Blocked promotion above screening for %s: tier=%s trust=%s comparability=%s provenance_complete=%s",
            trust_entry.get("result_id") or "<missing>",
            requested_tier,
            trust_entry.get("trust_label"),
            trust_entry.get("comparability_label"),
            self._provenance_complete(pr_row),
        )
        return existing_tier or "screening"

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
        allow_fingerprint_duplicate: bool = False,
        **kwargs,
    ) -> str:
        """Insert or update a leaderboard entry.

        Accepts all leaderboard columns as keyword arguments.
        Fields are only updated if provided and not None (prevents accidental NULLing).
        """
        self.flush_writes()
        resolved_result_id = result_id
        pr_row = self.conn.execute(
            "SELECT result_id, novelty_confidence, loss_ratio, param_count, flops_forward, "
            "throughput_tok_s, peak_memory_mb, forward_time_ms, graph_json, graph_fingerprint, "
            "result_cohort, trust_label, comparability_label, evaluation_protocol_version, "
            "data_provenance_json "
            "FROM program_results WHERE result_id = ? "
            "OR graph_fingerprint = ? "
            "ORDER BY CASE WHEN result_id = ? THEN 0 ELSE 1 END, timestamp DESC "
            "LIMIT 1",
            (result_id, result_id, result_id),
        ).fetchone()
        if (
            pr_row is None
            and architecture_desc
            and not is_reference
            and str(architecture_desc).strip()
        ):
            # Historical corruption showed callers could reach this method with
            # a bogus result_id but the correct fingerprint in architecture_desc.
            # Rebind to the canonical program row instead of creating an orphan.
            pr_row = self.conn.execute(
                "SELECT result_id, novelty_confidence, loss_ratio, param_count, flops_forward, "
                "throughput_tok_s, peak_memory_mb, forward_time_ms, graph_json, graph_fingerprint, "
                "result_cohort, trust_label, comparability_label, evaluation_protocol_version, "
                "data_provenance_json "
                "FROM program_results WHERE graph_fingerprint = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (str(architecture_desc).strip(),),
            ).fetchone()
        if pr_row:
            resolved_result_id = pr_row["result_id"]
        # Check if entry exists for this result_id
        existing = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (resolved_result_id,),
        ).fetchone()
        if pr_row is None and existing is None and not is_reference:
            LOGGER.error(
                "Blocked orphan leaderboard insert: result_id=%s architecture_desc=%s",
                str(result_id)[:12],
                str(architecture_desc or "")[:40],
            )
            return ""

        # Fingerprint-level dedup gate: if another entry already exists for
        # the same graph_fingerprint but a different result_id, refuse to
        # INSERT a duplicate. References and intentional inserts bypass.
        if (
            not existing
            and not is_reference
            and not allow_fingerprint_duplicate
            and pr_row
            and pr_row["graph_fingerprint"]
        ):
            fp = str(pr_row["graph_fingerprint"]).strip()
            if fp:
                fp_dup = self.conn.execute(
                    "SELECT l.entry_id, l.result_id FROM leaderboard l "
                    "JOIN program_results pr ON l.result_id = pr.result_id "
                    "WHERE pr.graph_fingerprint = ? AND l.result_id != ? "
                    "LIMIT 1",
                    (fp, resolved_result_id),
                ).fetchone()
                if fp_dup is not None:
                    LOGGER.warning(
                        "BLOCKED leaderboard dup insert: fp=%s existing_entry=%s "
                        "(result_id=%s) attempted_result_id=%s",
                        fp,
                        fp_dup["entry_id"],
                        fp_dup["result_id"],
                        resolved_result_id,
                    )
                    raise DuplicateLeaderboardFingerprintError(
                        graph_fingerprint=fp,
                        existing_entry_id=str(fp_dup["entry_id"]),
                        existing_result_id=str(fp_dup["result_id"]),
                        attempted_result_id=str(resolved_result_id),
                    )

        # Combine kwargs with existing data for composite score recomputation
        d = dict(existing) if existing else {}
        # Sanitize all incoming values
        kwargs = sanitize_for_db(kwargs)

        # Merge caller kwargs into d first so derived fields can read them
        for col, val in self._leaderboard_update_items(kwargs):
            d[col] = val
        if tags is not None:
            d["tags"] = tags
        if notes is not None:
            d["notes"] = notes
        if pr_row:
            for key in (
                "result_cohort",
                "trust_label",
                "comparability_label",
                "evaluation_protocol_version",
            ):
                if pr_row[key] is not None and not d.get(key):
                    d[key] = pr_row[key]
                    kwargs.setdefault(key, pr_row[key])
        # Never downgrade tier — only allow promotion or same-tier updates
        existing_tier = str(d.get("tier") or "screening")
        allowed_tier = self._resolve_allowed_tier(
            requested_tier=tier,
            existing_tier=existing_tier,
            pr_row=pr_row,
            is_reference=bool(is_reference),
        )
        if TIER_RANK.get(allowed_tier, 0) >= TIER_RANK.get(existing_tier, 0):
            d["tier"] = allowed_tier
        else:
            import logging as _log

            _log.getLogger(__name__).warning(
                "Blocked tier downgrade for %s: %s -> %s",
                resolved_result_id,
                existing_tier,
                allowed_tier,
            )
            allowed_tier = existing_tier  # preserve existing tier for SQL write below
            d["tier"] = existing_tier
        tier = allowed_tier
        d["model_source"] = model_source
        d["scoring_version"] = SCORING_VERSION
        kwargs.setdefault("scoring_version", SCORING_VERSION)
        if architecture_desc:
            d["architecture_desc"] = architecture_desc
        d["is_reference"] = int(is_reference)
        if reference_name:
            d["reference_name"] = reference_name

        # Auto-derive robustness_grade from investigation_robustness.
        # A: >=2/3, B: 1/3-2/3, C: <1/3, None: untested.
        if not kwargs.get("robustness_grade"):
            inv_rob = d.get("investigation_robustness")
            if inv_rob is not None:
                try:
                    inv_rob_f = float(inv_rob)
                    if inv_rob_f >= 2 / 3:
                        grade = "A"
                    elif inv_rob_f >= 1 / 3:
                        grade = "B"
                    else:
                        grade = "C"
                    d["robustness_grade"] = grade
                    kwargs["robustness_grade"] = grade
                except (TypeError, ValueError):
                    pass

        # Populate per-fingerprint replication aggregates
        _fp = pr_row["graph_fingerprint"] if pr_row else None
        if _fp:
            agg = self.get_fingerprint_aggregates(_fp)
            if agg.get("n_runs", 0) > 0:
                kwargs.setdefault("replication_n", agg["n_runs"])
                kwargs.setdefault("replication_loss_mean", agg["loss_mean"])
                kwargs.setdefault("replication_loss_std", agg["loss_std"])
                kwargs.setdefault(
                    "replication_best_vs_mean_gap", agg["best_vs_mean_gap"]
                )

        update_items = self._leaderboard_update_items(kwargs)

        score_kwargs = build_score_kwargs(
            self.conn,
            self,
            resolved_result_id,
            d,
            bool(is_reference),
        )
        # Pass replication data to scoring
        if _fp:
            agg = kwargs
            score_kwargs["replication_n"] = agg.get("replication_n")
            score_kwargs["replication_loss_mean"] = agg.get("replication_loss_mean")
            score_kwargs["replication_loss_std"] = agg.get("replication_loss_std")
            score_kwargs["replication_best_vs_mean_gap"] = agg.get(
                "replication_best_vs_mean_gap"
            )
        composite = compute_composite(**score_kwargs)

        # Compute efficiency_multiple from program_results operational metrics.
        # MoE models: skip param count penalty (active params < total params).
        from ...synthesis.op_roles import MOE_OPS

        _is_moe = False
        if pr_row and pr_row["graph_json"]:
            try:
                _gj = pr_row["graph_json"]
                if isinstance(_gj, str):
                    _gj = json.loads(_gj)
                _is_moe = any(
                    n.get("op_name") in MOE_OPS
                    for n in (_gj.get("nodes") or {}).values()
                )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass
        eff_mult = kwargs.get("efficiency_multiple")
        if eff_mult is None and pr_row:
            eff_result = self.compute_efficiency_multiple(
                loss_ratio=pr_row["loss_ratio"],
                param_count=pr_row["param_count"],
                flops_forward=pr_row["flops_forward"],
                throughput_tok_s=pr_row["throughput_tok_s"],
                peak_memory_mb=pr_row["peak_memory_mb"],
                forward_time_ms=pr_row["forward_time_ms"],
                is_moe=_is_moe,
            )
            if eff_result is not None:
                eff_mult = eff_result["geomean"]
        if eff_mult is not None:
            kwargs["efficiency_multiple"] = eff_mult

        if existing:
            entry_id = existing["entry_id"]
            sets = [
                "timestamp = ?",
                "model_source = ?",
                "tier = ?",
                "composite_score = ?",
                "is_reference = ?",
            ]
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

            for col, val in update_items:
                sets.append(f"{col} = ?")
                params.append(val)

            params.append(entry_id)
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )
        else:
            entry_id = str(uuid.uuid4())[:12]
            # Denormalize graph_fingerprint for the UNIQUE idx_leaderboard_fp.
            # pr_row carries the fingerprint resolved earlier in this fn.
            fp_for_insert = None
            if pr_row is not None:
                fp_val = pr_row["graph_fingerprint"]
                if fp_val is not None and str(fp_val).strip():
                    fp_for_insert = str(fp_val).strip()
            cols = [
                "entry_id",
                "result_id",
                "timestamp",
                "model_source",
                "architecture_desc",
                "tier",
                "composite_score",
                "is_reference",
                "reference_name",
                "tags",
                "notes",
            ]
            vals = [
                entry_id,
                resolved_result_id,
                time.time(),
                model_source,
                architecture_desc,
                tier,
                composite,
                int(is_reference),
                reference_name,
                tags,
                notes,
            ]

            for col, val in update_items:
                cols.append(col)
                vals.append(val)

            # Populate denormalized graph_fingerprint (idx_leaderboard_fp)
            # only if the column exists and is not already set by kwargs.
            if (
                fp_for_insert is not None
                and "graph_fingerprint" in self._get_leaderboard_columns()
                and "graph_fingerprint" not in cols
            ):
                cols.append("graph_fingerprint")
                vals.append(fp_for_insert)

            placeholders = ", ".join(["?"] * len(cols))
            self.conn.execute(
                f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )

        self._maybe_commit()
        return entry_id

    def get_leaderboard(
        self,
        tier: Optional[str] = None,
        limit: int = 50,
        sort_by: str = "composite_score",
        include_family: bool = True,
        include_references: bool = True,
        trusted_only: bool = False,
        tier_match_mode: str = "reached",
    ) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {
            "composite_score",
            "screening_loss_ratio",
            "investigation_loss_ratio",
            "validation_loss_ratio",
            "screening_novelty",
            "timestamp",
            "robustness_noise_score",
            "quant_int8_retention",
            "robustness_long_ctx_score",
            "discovery_loss_ratio",
            "generalization_gap",
            "efficiency_multiple",
        }
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
            "pr.novelty_valid_for_promotion AS novelty_valid_for_promotion, "
            "pr.novelty_validity_reason AS novelty_validity_reason, "
            "pr.cka_source AS _cka_source, "
            "pr.stage0_passed AS stage0_passed, "
            "pr.stage1_passed AS stage1_passed, "
            "pr.routing_confidence_mean AS _routing_confidence_mean, "
            "pr.fp_jacobian_spectral_norm AS jacobian_spectral_norm, "
            # Program-side fields used by canonical backend score attachment
            "pr.loss_ratio AS loss_ratio, "
            "pr.discovery_loss AS discovery_loss, "
            "pr.discovery_loss_ratio AS _pr_discovery_loss_ratio, "
            "pr.validation_loss AS validation_loss, "
            "pr.validation_loss_ratio AS _pr_validation_loss_ratio, "
            "pr.wikitext_perplexity AS _pr_wikitext_perplexity, "
            "pr.tinystories_perplexity AS _pr_tinystories_perplexity, "
            "pr.hellaswag_acc AS _pr_hellaswag_acc, "
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
            "pr.external_benchmarks_json AS _external_benchmarks_json, "
            "pr.efficiency_multiple AS _pr_efficiency_multiple "
            "FROM leaderboard l "
            "LEFT JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE 1=1"
        )
        params: List[Any] = []
        if trusted_only:
            query += f" AND {sql_trusted_clause(table_alias='l')}"
        if tier:
            normalized_tier = str(tier).strip().lower()
            current_status_clause = {
                "screening": "COALESCE(l.tier, 'screening') = 'screening'",
                "screened_out": "l.tier = 'screened_out'",
                "investigation": "l.tier = 'investigation'",
                "investigation_failed": "l.tier = 'investigation_failed'",
                "investigation_fingerprint_incomplete": (
                    "l.tier = 'investigation_fingerprint_incomplete'"
                ),
                "validation": "l.tier = 'validation' AND COALESCE(l.validation_passed, 0) = 1",
                "validation_pending": "l.tier = 'validation' AND COALESCE(l.validation_passed, 0) = 0",
                "validation_failed": "l.tier = 'validation_failed'",
                "breakthrough": "l.tier = 'breakthrough'",
            }
            reached_stage_clause = {
                "investigation": "l.investigation_passed = 1",
                "validation": "l.validation_passed = 1",
            }
            tier_clause = None
            if tier_match_mode == "current":
                tier_clause = current_status_clause.get(normalized_tier)
            else:
                tier_clause = reached_stage_clause.get(normalized_tier)

            if tier_clause:
                if include_references:
                    query += f" AND ({tier_clause} OR COALESCE(l.is_reference, 0) = 1)"
                else:
                    query += f" AND {tier_clause} AND COALESCE(l.is_reference, 0) = 0"
            elif include_references:
                query += " AND (l.tier = ? OR COALESCE(l.is_reference, 0) = 1)"
                params.append(normalized_tier)
            else:
                query += " AND l.tier = ? AND COALESCE(l.is_reference, 0) = 0"
                params.append(normalized_tier)
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

        try:
            rows = self.conn.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Leaderboard query failed; returning empty results: %s",
                exc,
            )
            return []
        results = []
        for r in rows:
            d = dict(r)
            # Prefer leaderboard-curated phase metrics, but backfill from raw
            # program_results when leaderboard fields are absent.
            if (
                d.get("discovery_loss_ratio") is None
                and d.get("_pr_discovery_loss_ratio") is not None
            ):
                d["discovery_loss_ratio"] = d.get("_pr_discovery_loss_ratio")
            # Only backfill validation metrics for entries actually at
            # validation tier — program_results stores val eval data from
            # training but that doesn't mean the entry was promoted.
            tier = str(d.get("tier") or "").strip().lower()
            if tier in ("validation", "breakthrough"):
                if (
                    d.get("validation_loss_ratio") is None
                    and d.get("_pr_validation_loss_ratio") is not None
                ):
                    d["validation_loss_ratio"] = d.get("_pr_validation_loss_ratio")
            elif (
                str(d.get("result_cohort") or "").strip().lower() == "backfill"
                and d.get("validation_loss_ratio") is None
                and d.get("_pr_validation_loss_ratio") is not None
            ):
                d["validation_loss_ratio"] = d.get("_pr_validation_loss_ratio")
            if (
                d.get("wikitext_perplexity") is None
                and d.get("_pr_wikitext_perplexity") is not None
            ):
                d["wikitext_perplexity"] = d.get("_pr_wikitext_perplexity")
            if (
                d.get("tinystories_perplexity") is None
                and d.get("_pr_tinystories_perplexity") is not None
            ):
                d["tinystories_perplexity"] = d.get("_pr_tinystories_perplexity")
            if (
                d.get("hellaswag_acc") is None
                and d.get("_pr_hellaswag_acc") is not None
            ):
                d["hellaswag_acc"] = d.get("_pr_hellaswag_acc")
            if include_family:
                d["architecture_family"] = self._classify_architecture_family(
                    graph_json=d.get("_graph_json"),
                    routing_mode=d.get("_routing_mode"),
                )
                if d.get("architecture_family") == "Unknown" and d.get("is_reference"):
                    d["architecture_family"] = self._reference_family_fallback(
                        d.get("reference_name")
                    )
            d.pop("_graph_json", None)
            d["routing_mode"] = d.pop("_routing_mode", None)
            d["arch_spec_json"] = d.pop("_arch_spec_json", None)
            d["param_count"] = d.pop("_param_count", None)
            d["graph_n_params_estimate"] = d.pop("_graph_n_params_estimate", None)
            d["novelty_confidence"] = d.pop("_novelty_confidence", None)
            d["cka_source"] = d.pop("_cka_source", None)
            d["routing_confidence_mean"] = d.pop("_routing_confidence_mean", None)
            if (
                d.get("efficiency_multiple") is None
                and d.get("_pr_efficiency_multiple") is not None
            ):
                d["efficiency_multiple"] = d.get("_pr_efficiency_multiple")
            d.pop("_pr_discovery_loss_ratio", None)
            d.pop("_pr_validation_loss_ratio", None)
            d.pop("_pr_wikitext_perplexity", None)
            d.pop("_pr_tinystories_perplexity", None)
            d.pop("_pr_hellaswag_acc", None)
            d.pop("_pr_efficiency_multiple", None)
            self._normalize_benchmark_fields(d)

            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = json.loads(
                        d["investigation_best_training"]
                    )
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
            d["trusted_candidate"] = bool(is_promotable_entry(d))
            results.append(d)

        results = self._attach_canonical_program_scores(results)

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
                    if (entry.get("composite_score") or 0) > (
                        deduped_refs[existing_idx].get("composite_score") or 0
                    ):
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

    def promote_to_tier(self, entry_id: str, tier: str, **kwargs) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        row = self.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return

        from ..leaderboard_scoring import (
            _PR_SELECT_COLS,
            _pr_dict_to_score_kwargs,
            compute_composite,
        )

        pr = None
        if row["result_id"]:
            pr = self.conn.execute(
                f"SELECT {_PR_SELECT_COLS}, data_provenance_json, trust_label, comparability_label "
                "FROM program_results WHERE result_id = ?",
                (row["result_id"],),
            ).fetchone()

        allowed_tier = self._resolve_allowed_tier(
            requested_tier=tier,
            existing_tier=str(row["tier"] or "screening"),
            pr_row=pr,
            is_reference=bool(row["is_reference"]),
        )
        requested_rank = TIER_RANK.get(str(tier or "").lower(), 0)
        allowed_rank = TIER_RANK.get(str(allowed_tier or "").lower(), 0)
        promotion_blocked = requested_rank > allowed_rank
        sets = ["tier = ?"]
        params: List[Any] = [allowed_tier]

        kwargs = sanitize_for_db(kwargs)
        update_items = (
            [] if promotion_blocked else self._leaderboard_update_items(kwargs)
        )

        for col, val in update_items:
            sets.append(f"{col} = ?")
            params.append(val)

        d = dict(row)
        d.update(dict(update_items))
        d["tier"] = allowed_tier

        pr_d: Dict[str, Any] = dict(pr) if pr else {}
        score_kw = _pr_dict_to_score_kwargs(
            pr_d, d, is_reference=bool(d.get("is_reference"))
        )
        composite = compute_composite(**score_kw)
        if isinstance(composite, dict):
            composite = composite["composite_score"]
        sets.append("composite_score = ?")
        params.append(composite)

        # Handle 'notes' explicitly (it's in _LEADERBOARD_MANAGED_COLUMNS
        # so _leaderboard_update_items filters it out, but promote_to_tier
        # should still allow updating it).
        if "notes" in kwargs and kwargs["notes"] is not None:
            sets.append("notes = ?")
            params.append(kwargs["notes"])

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
        except (KeyError, TypeError, ValueError, sqlite3.OperationalError) as e:
            LOGGER.debug(
                "Fingerprint leaderboard sync skipped for entry %s: %s", entry_id, e
            )
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

    def backfill_replication_aggregates(self) -> int:
        """Backfill replication_n and replication_loss_mean on all leaderboard entries.

        Idempotent — safe to call on every startup. Only touches entries where
        the stored replication_n disagrees with the current count from
        program_results (handles new runs arriving since last backfill).

        Returns the number of entries updated.
        """
        rows = self.conn.execute(
            """SELECT l.entry_id, l.replication_n, pr.graph_fingerprint
               FROM leaderboard l
               JOIN program_results pr ON pr.result_id = l.result_id
               WHERE pr.graph_fingerprint IS NOT NULL"""
        ).fetchall()

        # Batch-fetch all fingerprint aggregates in one query
        fps = list({row["graph_fingerprint"] for row in rows})
        agg_map = self.get_fingerprint_aggregates_batch(fps)

        updated = 0
        for row in rows:
            agg = agg_map.get(row["graph_fingerprint"], {})
            n_runs = agg.get("n_runs", 0)
            if n_runs == 0:
                continue
            if row["replication_n"] == n_runs:
                continue
            self.conn.execute(
                """UPDATE leaderboard
                   SET replication_n = ?,
                       replication_loss_mean = ?,
                       replication_loss_std = ?,
                       replication_best_vs_mean_gap = ?
                   WHERE entry_id = ?""",
                (
                    n_runs,
                    agg.get("loss_mean"),
                    agg.get("loss_std"),
                    agg.get("best_vs_mean_gap"),
                    row["entry_id"],
                ),
            )
            updated += 1

        if updated:
            self._maybe_commit()
            LOGGER.info("backfill_replication_aggregates: updated %d entries", updated)
        return updated
