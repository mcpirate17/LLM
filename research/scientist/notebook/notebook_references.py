"""Mixin for LabNotebook — split from notebook_misc."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ..json_utils import fast_loads as _json_loads
from ..leaderboard_scoring import (
    compute_efficiency_multiple as _compute_efficiency_multiple,
    compute_pre_investigation_score as _compute_pre_investigation_score,
)
from .graph_artifacts import resolve_graph_json_value


class _ReferencesMixin:
    """Reference architectures, decisions, novelty calibration."""

    __slots__ = ()

    @staticmethod
    def _decode_reference_json_field(data: Dict[str, Any], key: str) -> None:
        raw = data.get(key)
        if raw:
            try:
                data[key] = _json_loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

    @classmethod
    def _hydrate_decision_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        for key in ("evidence_ids", "alternatives_considered"):
            cls._decode_reference_json_field(data, key)
        if data.get("evidence_pack_json"):
            try:
                data["evidence_pack"] = _json_loads(data["evidence_pack_json"])
            except (json.JSONDecodeError, TypeError):
                data["evidence_pack"] = None
        return data

    @classmethod
    def _hydrate_selection_decision_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        for key in (
            "candidate_pool_summary_json",
            "score_breakdown_json",
            "policy_json",
            "chosen_experiments_json",
            "trigger_json",
        ):
            cls._decode_reference_json_field(data, key)
        return data

    def _graph_structural_counts(
        self, result_id: str, graph_json: Optional[str] = None
    ) -> Dict[str, Optional[int]]:
        """Return routing/sparse/MoE op counts with at most one graph lookup."""
        try:
            raw_graph_json = graph_json
            if raw_graph_json is None:
                row = self.conn.execute(
                    "SELECT graph_json FROM program_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                raw_graph_json = row[0] if row else None
            if not raw_graph_json:
                return {"routing": None, "sparse": None, "moe": None}
            raw_graph_json = resolve_graph_json_value(
                self.conn,
                self.db_path,
                raw_graph_json,
            )
            graph_data = _json_loads(raw_graph_json)
            nodes = graph_data.get("nodes", [])
            if isinstance(nodes, dict):
                node_iter = nodes.values()
            elif isinstance(nodes, list):
                node_iter = nodes
            else:
                node_iter = []

            routing = 0
            sparse = 0
            moe = 0
            for node in node_iter:
                if not isinstance(node, dict):
                    continue
                op_name = node.get("op")
                if not op_name:
                    op_name = node.get("op_name")
                if op_name in self._ROUTING_OPS:
                    routing += 1
                if op_name in self._SPARSE_OPS:
                    sparse += 1
                if op_name in self._MOE_OPS:
                    moe += 1
            return {
                "routing": routing if routing > 0 else None,
                "sparse": sparse if sparse > 0 else None,
                "moe": moe if moe > 0 else None,
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {"routing": None, "sparse": None, "moe": None}

    def compute_efficiency_multiple(
        self,
        loss_ratio: Optional[float] = None,
        param_count: Optional[float] = None,
        flops_forward: Optional[float] = None,
        throughput_tok_s: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
        forward_time_ms: Optional[float] = None,
        is_moe: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Geometric mean of per-dimension ratios vs GPT-2.

        All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
        dimensions to return a result (graceful with missing data).

        For MoE models (is_moe=True), total param count is excluded since
        MoE activates only a fraction of params per token.
        Returns dict with per-dimension ratios and ``geomean``, or None.
        """
        return _compute_efficiency_multiple(
            loss_ratio=loss_ratio,
            param_count=param_count,
            flops_forward=flops_forward,
            throughput_tok_s=throughput_tok_s,
            peak_memory_mb=peak_memory_mb,
            forward_time_ms=forward_time_ms,
            is_moe=is_moe,
        )

    @staticmethod
    def compute_composite_score(**kwargs) -> float:
        """Delegate to v7 scoring (14-component, 565pt max).

        Translates legacy parameter names to v7 equivalents for backward
        compatibility with callers that still use ``wikitext_perplexity``.
        """
        from ..leaderboard_scoring import compute_composite

        # Translate legacy kwargs → current scoring parameter names
        if "wikitext_perplexity" in kwargs and "ppl_screening" not in kwargs:
            kwargs["ppl_screening"] = kwargs.pop("wikitext_perplexity")
        # wikitext_score has no direct scoring equivalent — drop silently
        kwargs.pop("wikitext_score", None)

        result = compute_composite(**kwargs)
        return result if isinstance(result, (int, float)) else float(result)

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

    @staticmethod
    def _reference_family_fallback(reference_name: Optional[str]) -> str:
        """Infer a stable family label for named baselines when graph data is absent."""
        name = str(reference_name or "").strip().lower()
        if not name:
            return "Unknown"
        if "gpt" in name or "transformer" in name:
            return "Attention"
        if "mamba" in name or "ssm" in name:
            return "Mamba-SSM"
        if "rwkv" in name:
            return "Hybrid-Mixer"
        if "retrieval" in name or "rag" in name:
            return "Hybrid-Attention"
        return "Unknown"

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
        self._maybe_commit()

    # ── Pre-investigation gate helpers ──────────────────────────────────

    def get_investigation_eligible(
        self,
        _max_lr: float,
        min_stability: float,
        min_spectral_norm: float,
        max_spectral_norm: float,
        min_improvement_rate: float,
        _ref_lr_ceiling: Optional[float] = None,
        min_composite_score: Optional[float] = None,
    ) -> List[Dict]:
        """Stage A hard reject: return screening candidates that pass health filters.

        Joins program_results with leaderboard to return full metric rows for
        candidates eligible for investigation.  Uses composite_score as the
        primary worthiness gate instead of loss_ratio — a model with excellent
        efficiency/novelty/stability deserves investigation even with moderate loss.
        """
        # Dynamic score floor: the lower of (p25 of non-ref investigation tier)
        # and (p90 of non-ref screening tier).
        #
        # Why both? The p25-of-investigation rule alone drifts upward as a few
        # exceptional candidates promote (they shift the historical baseline),
        # which then locks out the next wave of solid-but-not-stellar templates.
        # Bounding by p90-of-screening keeps the floor anchored to what new
        # candidates can plausibly reach: at least the top 10% of any screening
        # population is always eligible to attempt investigation.
        # Falls back to 75th percentile of screening tier when no
        # non-reference investigation entries exist yet.
        if min_composite_score is None:
            inv_scores = self.conn.execute(
                "SELECT composite_score FROM leaderboard"
                " WHERE tier IN ('investigation', 'validation')"
                " AND COALESCE(is_reference, 0) = 0"
                " AND composite_score IS NOT NULL"
                " ORDER BY composite_score ASC"
            ).fetchall()
            scr_scores = self.conn.execute(
                "SELECT composite_score FROM leaderboard"
                " WHERE tier = 'screening'"
                " AND COALESCE(is_reference, 0) = 0"
                " AND composite_score IS NOT NULL"
                " ORDER BY composite_score ASC"
            ).fetchall()
            inv_floor = inv_scores[len(inv_scores) // 4][0] if inv_scores else None
            screen_p90 = (
                scr_scores[9 * len(scr_scores) // 10][0] if scr_scores else None
            )
            screen_p75 = scr_scores[3 * len(scr_scores) // 4][0] if scr_scores else 0.0
            if inv_floor is not None and screen_p90 is not None:
                min_composite_score = min(inv_floor, screen_p90)
            elif inv_floor is not None:
                min_composite_score = inv_floor
            else:
                min_composite_score = screen_p75
        # v9 Gemini hard gates — see leaderboard_scoring.GEMINI_HARD_GATE_*.
        # Imported lazily to avoid a circular import at module load.
        from ..leaderboard_scoring import (
            GEMINI_HARD_GATE_ERF_DENSITY,
            GEMINI_HARD_GATE_ERF_VARIANCE,
        )

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
                 AND COALESCE(l.composite_score, 0) >= ?
                 -- v9 Gemini gates: NULL-tolerant (rows pre-backfill pass)
                 -- but populated rows must clear the architectural floors.
                 AND (pr.fp_jacobian_erf_density IS NULL
                      OR pr.fp_jacobian_erf_density >= ?)
                 AND (pr.fp_jacobian_erf_variance IS NULL
                      OR pr.fp_jacobian_erf_variance >= ?)
               ORDER BY l.composite_score DESC""",
            (
                min_stability,
                min_spectral_norm,
                max_spectral_norm,
                min_improvement_rate,
                min_composite_score,
                GEMINI_HARD_GATE_ERF_DENSITY,
                GEMINI_HARD_GATE_ERF_VARIANCE,
            ),
        ).fetchall()
        return [dict(r) for r in rows]

    def compute_pre_investigation_score(
        row: Dict, best_ref_lr: Optional[float] = None
    ) -> float:
        """Stage B composite readiness score (0-100 scale).

        Components:
        - Performance (40pts): loss_ratio, discovery_loss_ratio, loss_improvement_rate
        - Stability (20pts): stability_score, spectral_norm (Gaussian around 1.0), grad_norm_std
        - Novelty (20pts): novelty_score * confidence, structural_novelty, behavioral_novelty
        - Fingerprint quality (10pts): fp_intrinsic_dim, fp_isotropy, fp_rank_ratio
        - Efficiency (10pts): throughput_tok_s, peak_memory_mb
        - Reference penalty (-20pts): if loss_ratio > 1.5 * best_reference_lr
        """
        return _compute_pre_investigation_score(row, best_ref_lr=best_ref_lr)

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
            graph_json = entry.pop("_graph_json", None)
            if graph_json:
                graph_json = resolve_graph_json_value(
                    self.conn,
                    self.db_path,
                    graph_json,
                )
            family = self._classify_architecture_family(
                graph_json=graph_json,
                routing_mode=entry.pop("_routing_mode", None),
            )
            if family == "Unknown":
                family = self._reference_family_fallback(entry.get("reference_name"))
            entry["architecture_family"] = family
            entry["screening_novelty"] = self._reference_novelty_for_display(
                entry.get("screening_novelty")
            )
            refs.append(entry)
        return refs

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
            graph = _json_loads(graph_json)
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
            "attention",
            "self_attention",
            "mha",
            "multihead_attention",
            "qkv_attention",
            "softmax_attention",
            "linear_attention",
        }
        conv_ops = {"conv1d", "conv1d_seq", "depthwise_conv1d", "conv_only"}
        spectral_ops = {
            "sin",
            "cos",
            "fft",
            "ifft",
            "fourier_mix",
            "fourier_mixing",
            "rfft_seq",
            "irfft_seq",
        }
        gating_ops = {
            "sigmoid",
            "tanh",
            "silu",
            "gelu",
            "maximum",
            "minimum",
            "swiglu",
            "topk_gate",
            "moe_topk",
        }
        mlp_ops = {
            "linear_proj",
            "linear_proj_up",
            "linear_proj_down",
            "learnable_bias",
            "swiglu_mlp",
        }
        ssm_ops = {"state_space", "selective_scan"}
        adaptive_ops = {
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
            "fixed_point_iter",
        }

        has_attention = bool(ops & attention_ops)
        has_conv = bool(ops & conv_ops)
        has_spectral = bool(ops & spectral_ops)
        has_gating = bool(ops & gating_ops)
        has_mlp = bool(ops & mlp_ops)
        has_ssm = bool(ops & ssm_ops)
        has_adaptive = bool(ops & adaptive_ops) or routing_mode in (
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
        )

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

    def get_unresolved_hypotheses(
        self, campaign_id: Optional[str] = None
    ) -> List[Dict]:
        """Get pending/testing hypotheses."""
        query = "SELECT * FROM hypotheses WHERE status IN ('pending', 'testing')"
        params: List[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        query += " ORDER BY timestamp DESC"
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor]

    # ── Decisions ──

    def record_decision(
        self,
        campaign_id: Optional[str],
        decision_type: str,
        subject: str,
        rationale: str,
        evidence_ids: Optional[List[str]] = None,
        alternatives: Optional[List[Dict]] = None,
        evidence_pack: Optional[Dict] = None,
    ) -> str:
        """Record a go/no-go or other decision. Returns decision_id."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO decisions
            (decision_id, campaign_id, timestamp, decision_type,
             subject, rationale, evidence_ids, alternatives_considered,
             evidence_pack_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                campaign_id,
                now,
                decision_type,
                subject,
                rationale,
                json.dumps(evidence_ids) if evidence_ids else None,
                json.dumps(alternatives) if alternatives else None,
                json.dumps(evidence_pack) if evidence_pack else None,
            ),
        )
        self._maybe_commit()
        return decision_id

    def get_decisions(
        self, campaign_id: Optional[str] = None, decision_type: Optional[str] = None
    ) -> List[Dict]:
        """Get decisions, optionally filtered."""
        query = "SELECT * FROM decisions WHERE 1=1"
        params: List[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        if decision_type:
            query += " AND decision_type = ?"
            params.append(decision_type)
        query += " ORDER BY timestamp DESC"
        cursor = self.conn.execute(query, params)
        return [self._hydrate_decision_row(row) for row in cursor]

    # ── Selection Decisions / Family Bandit Stats ──

    def record_selection_decision(
        self,
        context: str,
        candidate_pool_summary: Dict[str, Any],
        score_breakdown: List[Dict[str, Any]],
        policy: Dict[str, Any],
        reason: str,
        chosen_experiments: List[Dict[str, Any]],
        experiment_id: Optional[str] = None,
        trigger: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record one evidence-based experiment-selection decision."""
        decision_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_decisions
            (decision_id, timestamp, context, experiment_id,
             candidate_pool_summary_json, score_breakdown_json,
             policy_json, reason, chosen_experiments_json, trigger_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                now,
                context,
                experiment_id,
                json.dumps(candidate_pool_summary or {}),
                json.dumps(score_breakdown or []),
                json.dumps(policy or {}),
                reason or "",
                json.dumps(chosen_experiments or []),
                json.dumps(trigger or {}),
            ),
        )
        self._maybe_commit()
        return decision_id

    def get_selection_decisions(
        self,
        context: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return selection decisions newest first."""
        query = "SELECT * FROM selection_decisions WHERE 1=1"
        params: List[Any] = []
        if context:
            query += " AND context = ?"
            params.append(context)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.execute(query, params)
        return [self._hydrate_selection_decision_row(row) for row in cursor]

    def get_selection_family_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return family bandit stats keyed by family name."""
        cursor = self.conn.execute("SELECT * FROM selection_family_stats")
        return {row["family"]: dict(row) for row in cursor}

    def update_selection_family_stats(self, family: str, reward: float) -> None:
        """Update per-family running reward estimate for UCB/uncertainty."""
        family_name = (family or "Unknown").strip() or "Unknown"
        now = time.time()
        self.conn.execute(
            """INSERT INTO selection_family_stats
            (family, n_trials, cumulative_reward, mean_reward, last_reward, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(family) DO UPDATE SET
                n_trials = n_trials + 1,
                cumulative_reward = cumulative_reward + excluded.last_reward,
                mean_reward = (cumulative_reward + excluded.last_reward) * 1.0
                              / (n_trials + 1),
                last_reward = excluded.last_reward,
                last_updated = excluded.last_updated
            """,
            (family_name, float(reward), float(reward), float(reward), now),
        )
        self._maybe_commit()

    # ── Novelty Calibration ──

    def record_novelty_calibration(
        self,
        reference_version: str,
        n_runs: int,
        distribution: Dict[str, Any],
        noise_floor_mean: Optional[float] = None,
        noise_floor_std: Optional[float] = None,
        confidence_low: Optional[float] = None,
        confidence_high: Optional[float] = None,
        cka_source: Optional[str] = None,
        cka_artifact_version: Optional[str] = None,
        probe_protocol_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist novelty baseline calibration stats."""
        calibration_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO novelty_calibration
            (calibration_id, timestamp, reference_version, cka_source,
             cka_artifact_version, probe_protocol_hash, n_runs,
             noise_floor_mean, noise_floor_std, confidence_low, confidence_high,
             distribution_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id,
                now,
                reference_version,
                cka_source,
                cka_artifact_version,
                probe_protocol_hash,
                int(max(1, n_runs)),
                noise_floor_mean,
                noise_floor_std,
                confidence_low,
                confidence_high,
                json.dumps(distribution or {}),
                json.dumps(metadata or {}),
            ),
        )
        self._maybe_commit()
        return calibration_id

    def get_latest_novelty_calibration(
        self,
        reference_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest novelty calibration row, optionally by reference version."""
        query = "SELECT * FROM novelty_calibration WHERE 1=1"
        params: List[Any] = []
        if reference_version:
            query += " AND reference_version = ?"
            params.append(reference_version)
        query += " ORDER BY timestamp DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        out = dict(row)
        for key in ("distribution_json", "metadata_json"):
            raw = out.get(key)
            if raw:
                try:
                    out[key] = _json_loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out
