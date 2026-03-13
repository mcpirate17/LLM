"""Leaderboard CRUD operations — upsert, get, promote, pin, classify."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .json_utils import fast_loads as _json_loads

from .leaderboard_schema import (
    LEADERBOARD_UPSERT_COLUMNS,
    PROMOTE_ALL_COLUMNS,
)
from .leaderboard_scoring import (
    build_score_kwargs,
    compute_composite_score,
    compute_efficiency_multiple,
    reference_novelty_for_display,
)
from .notebook import sanitize_for_db

import logging

LOGGER = logging.getLogger(__name__)


class LeaderboardCRUDMixin:
    """Mixin providing leaderboard CRUD operations."""

    __slots__ = ()

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
        **kwargs: Any,
    ) -> str:
        """Insert or update a leaderboard entry.

        Accepts all leaderboard columns as keyword arguments.
        Fields are only updated if provided and not None.
        """
        existing = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()

        d = dict(existing) if existing else {}
        kwargs = sanitize_for_db(kwargs)

        d.update({k: v for k, v in kwargs.items() if v is not None})
        if tags is not None:
            d["tags"] = tags
        if notes is not None:
            d["notes"] = notes
        d["tier"] = tier
        d["model_source"] = model_source
        if architecture_desc:
            d["architecture_desc"] = architecture_desc
        d["is_reference"] = int(is_reference)
        if reference_name:
            d["reference_name"] = reference_name

        composite = compute_composite_score(
            **build_score_kwargs(self.conn, self.notebook, result_id, d, bool(is_reference)),
        )

        # Compute efficiency_multiple from program_results operational metrics.
        eff_mult = kwargs.get("efficiency_multiple")
        if eff_mult is None:
            pr_row = self.conn.execute(
                "SELECT loss_ratio, param_count, flops_forward, "
                "throughput_tok_s, peak_memory_mb, forward_time_ms "
                "FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if pr_row:
                eff_result = compute_efficiency_multiple(
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
            sets = [
                "timestamp = ?", "model_source = ?", "tier = ?",
                "composite_score = ?", "is_reference = ?",
            ]
            params: List[Any] = [time.time(), model_source, tier, composite, int(is_reference)]

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

            for col in LEADERBOARD_UPSERT_COLUMNS:
                if col in kwargs and kwargs[col] is not None:
                    sets.append(f"{col} = ?")
                    val = kwargs[col]
                    if isinstance(val, bool):
                        val = int(val)
                    params.append(val)

            params.append(entry_id)
            self.conn.execute(
                f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                params,
            )
        else:
            entry_id = str(uuid.uuid4())[:12]
            cols = [
                "entry_id", "result_id", "timestamp", "model_source",
                "architecture_desc", "tier", "composite_score",
                "is_reference", "reference_name", "tags", "notes",
            ]
            vals: List[Any] = [
                entry_id, result_id, time.time(), model_source,
                architecture_desc, tier, composite,
                int(is_reference), reference_name, tags, notes,
            ]

            for col in LEADERBOARD_UPSERT_COLUMNS:
                if col in kwargs and kwargs[col] is not None:
                    cols.append(col)
                    val = kwargs[col]
                    if isinstance(val, bool):
                        val = int(val)
                    vals.append(val)

            placeholders = ", ".join(["?"] * len(cols))
            self.conn.execute(
                f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )

        self.notebook._maybe_commit()
        return entry_id

    def promote_to_tier(self, entry_id: str, tier: str, **kwargs: Any) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        sets = ["tier = ?"]
        params: List[Any] = [tier]

        kwargs = sanitize_for_db(kwargs)

        for col in PROMOTE_ALL_COLUMNS:
            if col in kwargs and kwargs[col] is not None:
                sets.append(f"{col} = ?")
                val = kwargs[col]
                if isinstance(val, bool):
                    val = int(val)
                params.append(val)

        # Recompute composite score.
        row = self.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row:
            d = dict(row)
            d.update({k: v for k, v in kwargs.items() if v is not None})
            composite = (
                compute_composite_score(
                    **build_score_kwargs(
                        self.conn, self.notebook, d["result_id"], d,
                        bool(d.get("is_reference")),
                    )
                )
                if d.get("result_id")
                else 0.0
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
            LOGGER.debug(
                "Fingerprint leaderboard sync skipped for entry %s: %s",
                entry_id, e,
            )
        self.notebook._maybe_commit()

    def get_leaderboard(
        self,
        tier: Optional[str] = None,
        limit: int = 50,
        sort_by: str = "composite_score",
        include_family: bool = True,
        include_references: bool = True,
    ) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {
            "composite_score", "screening_loss_ratio",
            "investigation_loss_ratio", "validation_loss_ratio",
            "screening_novelty", "timestamp",
            "robustness_noise_score", "quant_int8_retention",
            "robustness_long_ctx_score",
            "discovery_loss_ratio", "generalization_gap",
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
            "pr.cka_source AS _cka_source, "
            "pr.routing_confidence_mean AS _routing_confidence_mean, "
            "pr.fp_jacobian_spectral_norm AS jacobian_spectral_norm, "
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
        pr_sort_fields = {"discovery_loss_ratio", "generalization_gap"}
        sort_col = sort_by if sort_by in pr_sort_fields else f"l.{sort_by}"
        query += (
            f" ORDER BY COALESCE(l.is_pinned, 0) DESC, "
            f"COALESCE(l.is_reference, 0) DESC, "
            f"{sort_col} DESC NULLS LAST LIMIT ?"
        )
        params.append(oversample)

        rows = self.conn.execute(query, params).fetchall()
        results = self._process_leaderboard_rows(rows, include_family)

        return self._dedup_and_limit(results, limit, include_references)

    def _process_leaderboard_rows(
        self, rows: list, include_family: bool,
    ) -> List[Dict]:
        """Transform raw SQL rows into cleaned leaderboard dicts."""
        results: List[Dict] = []
        for r in rows:
            d = dict(r)
            if d.get("discovery_loss_ratio") is None and d.get("_pr_discovery_loss_ratio") is not None:
                d["discovery_loss_ratio"] = d["_pr_discovery_loss_ratio"]
            if d.get("validation_loss_ratio") is None and d.get("_pr_validation_loss_ratio") is not None:
                d["validation_loss_ratio"] = d["_pr_validation_loss_ratio"]
            raw_graph_json = d.get("_graph_json")
            if include_family:
                d["architecture_family"] = self._classify_architecture_family(
                    graph_json=raw_graph_json,
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
                d["efficiency_multiple"] = d["_pr_efficiency_multiple"]
            d.pop("_pr_discovery_loss_ratio", None)
            d.pop("_pr_validation_loss_ratio", None)
            d.pop("_pr_efficiency_multiple", None)

            if d.get("investigation_best_training"):
                try:
                    d["investigation_best_training_parsed"] = _json_loads(
                        d["investigation_best_training"]
                    )
                except (json.JSONDecodeError, TypeError):
                    pass
            if d.get("is_reference"):
                d["screening_novelty"] = reference_novelty_for_display(
                    d.get("screening_novelty")
                )
                if d.get("novelty_score") is not None:
                    d["novelty_score"] = reference_novelty_for_display(
                        d.get("novelty_score")
                    )
            # Derive human-readable display_name from graph ops
            if not d.get("is_reference") and not d.get("display_name"):
                d["display_name"] = self._derive_display_name(
                    raw_graph_json, d.get("architecture_family"),
                )
            results.append(d)
        return results

    @staticmethod
    def _dedup_and_limit(
        results: List[Dict], limit: int, include_references: bool,
    ) -> List[Dict]:
        """Dedup by fingerprint and enforce limit, keeping references."""
        references: List[Dict] = []
        non_references: List[Dict] = []
        for entry in results:
            if include_references and entry.get("is_reference"):
                references.append(entry)
            else:
                non_references.append(entry)

        # Dedup references by graph fingerprint.
        seen_ref_fps: Dict[str, int] = {}
        deduped_refs: List[Dict] = []
        for entry in references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                if fp in seen_ref_fps:
                    idx = seen_ref_fps[fp]
                    if (entry.get("composite_score") or 0) > (deduped_refs[idx].get("composite_score") or 0):
                        deduped_refs[idx] = entry
                    continue
                seen_ref_fps[fp] = len(deduped_refs)
            deduped_refs.append(entry)

        # Dedup non-references.
        seen_fingerprints: Dict[str, int] = {}
        deduped: List[Dict] = []
        for entry in non_references:
            fp = entry.get("_graph_fingerprint")
            if fp:
                if fp in seen_ref_fps:
                    continue
                if fp in seen_fingerprints:
                    idx = seen_fingerprints[fp]
                    if (entry.get("composite_score") or 0) > (deduped[idx].get("composite_score") or 0):
                        deduped[idx] = entry
                    continue
                seen_fingerprints[fp] = len(deduped)
            deduped.append(entry)

        for entry in deduped:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)
        for entry in deduped_refs:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)

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
        row = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_investigation_eligible(
        self,
        max_lr: float,
        min_stability: float,
        min_spectral_norm: float,
        max_spectral_norm: float,
        min_improvement_rate: float,
        ref_lr_ceiling: Optional[float] = None,
    ) -> List[Dict]:
        """Return screening candidates that pass all hard filters."""
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
            entry["architecture_family"] = self._classify_architecture_family(
                graph_json=entry.pop("_graph_json", None),
                routing_mode=entry.pop("_routing_mode", None),
            )
            entry["screening_novelty"] = reference_novelty_for_display(
                entry.get("screening_novelty")
            )
            refs.append(entry)
        return refs

    def get_investigated_fingerprints(self) -> set:
        """Return fingerprints that have already been investigated or beyond."""
        fps: set = set()
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM leaderboard l "
            "JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE l.tier IN ('investigation', 'validation', 'breakthrough')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM program_results pr "
            "JOIN experiments e ON e.experiment_id = pr.experiment_id "
            "WHERE e.experiment_type IN ('investigation', 'ablation')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        return fps

    def get_tiers_for_result_ids(self, result_ids: List[str]) -> Dict[str, str]:
        """Return {result_id: tier} for given result IDs."""
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
            "attention", "self_attention", "mha", "multihead_attention",
            "qkv_attention", "softmax_attention", "linear_attention",
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

        if routing_mode == "moe_topk" or "moe_topk" in ops:
            family = f"MoE-{family}"
        if has_adaptive:
            family = f"Adaptive-{family}"

        return family

    @staticmethod
    def _derive_display_name(
        graph_json: Optional[str], architecture_family: Optional[str],
    ) -> Optional[str]:
        """Derive a human-readable name from graph ops.

        Returns a compact name like "Rotor+ReLU Net" or "Clifford+Grade Net".
        Falls back to architecture_family if ops can't be parsed.
        """
        if not graph_json:
            return architecture_family or None

        # Structural/boilerplate ops excluded from naming
        _STRUCTURAL_OPS = frozenset({
            "input", "add", "sub", "mul", "concat", "split2", "split3",
            "identity", "reshape", "transpose", "permute", "flatten",
            "dropout", "layer_norm", "rms_norm", "batch_norm", "group_norm",
            "layernorm", "layernorm_pre", "rmsnorm", "rmsnorm_pre",
            "dynamic_norm", "no_norm",
        })
        # Human-friendly short names for common ops
        _OP_SHORT = {
            "self_attention": "Attn", "attention": "Attn", "mha": "MHA",
            "multihead_attention": "MHA", "qkv_attention": "QKV-Attn",
            "linear_attention": "LinAttn", "softmax_attention": "SoftAttn",
            "clifford_attention": "Clifford", "tropical_attention": "Tropical",
            "ultrametric_attention": "Ultrametric",
            "conv1d": "Conv1D", "conv1d_seq": "Conv1D", "depthwise_conv1d": "DWConv",
            "linear_proj": "Linear", "linear_proj_up": "LinearUp",
            "linear_proj_down": "LinearDown", "grouped_linear": "GrpLinear",
            "bottleneck_proj": "Bottleneck",
            "relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "swiglu": "SwiGLU",
            "sigmoid": "Sigmoid", "tanh": "Tanh", "swiglu_mlp": "SwiGLU-MLP",
            "state_space": "SSM", "selective_scan": "S4",
            "fourier_mix": "Fourier", "fourier_mixing": "Fourier",
            "fft": "FFT", "ifft": "IFFT", "rfft_seq": "RFFT", "irfft_seq": "IRFFT",
            "sin": "Sin", "cos": "Cos",
            "rotor_transform": "Rotor", "geometric_product": "GeomProd",
            "grade_mix": "GradeMix", "padic_expand": "PAdicExp",
            "padic_gate": "PAdicGate",
            "lif_neuron": "LIF", "spike_rate_code": "SpikeRate",
            "tropical_router": "TropRouter", "relu_gate_routing": "ReLUGate",
            "topk_gate": "TopK", "moe_topk": "MoE-TopK", "mod_topk": "ModTopK",
            "cosine_similarity": "CosSim", "tied_proj": "TiedProj",
            "learnable_scale": "Scale", "learnable_bias": "Bias",
            "early_exit": "EarlyExit", "adaptive_recursion": "AdaptRecur",
            "fixed_point_iter": "FixedPt",
            "square": "Sq", "maximum": "Max", "minimum": "Min",
        }

        try:
            graph = _json_loads(graph_json)
            nodes = graph.get("nodes")
            if isinstance(nodes, dict):
                node_iter = [n for n in nodes.values() if isinstance(n, dict)]
            elif isinstance(nodes, list):
                node_iter = [n for n in nodes if isinstance(n, dict)]
            else:
                return architecture_family or None
            ops = [str(n.get("op_name", "")).strip() for n in node_iter]
        except (json.JSONDecodeError, TypeError, ValueError):
            return architecture_family or None

        # Filter to interesting ops, preserving order of first appearance
        seen = set()
        interesting = []
        for op in ops:
            if op in _STRUCTURAL_OPS or op in seen or not op:
                continue
            seen.add(op)
            interesting.append(_OP_SHORT.get(op, op.replace("_", " ").title().replace(" ", "")))

        if not interesting:
            return architecture_family or None

        # Take up to 3 most distinctive ops
        name = "+".join(interesting[:3])
        if len(interesting) > 3:
            name += f"+{len(interesting) - 3}"
        return f"{name} Net"

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

    def set_leaderboard_pin(self, entry_id: str, pinned: bool) -> None:
        """Pin or unpin a leaderboard entry for dashboard priority."""
        self.notebook._submit_write(
            "UPDATE leaderboard SET is_pinned = ? WHERE entry_id = ?",
            (1 if pinned else 0, entry_id),
        )

    def get_scaling_summary(self) -> Dict:
        """Get a summary of scaling gate results for Aria's context."""
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
