"""Support code for motif grammar generation and validation."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional

from .graph import ComputationGraph, ShapeInfo
from .primitives import (
    PRIMITIVE_REGISTRY,
    PrimitiveOp,
    algebraic_types_compatible,
    default_algebraic_type_for_space,
)

logger = logging.getLogger(__name__)

_motif_weight_lock = threading.Lock()
_motif_weight_cache_key: Optional[tuple] = None
_motif_weight_cache_val: Dict[str, float] = {}

OP_TO_TEMPLATE: Dict[str, str] = {
    "div_safe": "safe_division",
    "maximum": "gated_maximum",
    "minimum": "gated_minimum",
    "sub": "residual_difference",
    "split3": "three_way_split",
    "outer_product": "gated_product",
    "geometric_product": "geometric_product_block",
    "tropical_matmul": "tropical_matmul_block",
    "hyp_distance": "hyp_distance_scoring",
    "cumprod_safe": "decay_sequence",
    "lif_neuron": "spiking_moe_block",
    "sparse_threshold": "spiking_moe_block",
    "stdp_attention": "spiking_residual_block",
    "spike_rate_code": "spiking_moe_block",
    "hyp_linear": "hyperbolic_bridge_block",
    "hyp_tangent_nonlinear": "hyperbolic_bridge_block",
    "poincare_add": "poincare_add_bridge",
    "sparse_bottleneck_moe": "n_way_moe_block",
    "conv_only": "conv_residual_block",
    "fixed_point_iter": "iterative_refinement",
    "gated_delta": "recurrent_delta_block",
    "bottleneck_proj": "bottleneck",
    "low_rank_proj": "bottleneck",
    "confidence_token_gate": "cascaded_early_exit",
    "depth_weighted_proj": "recursive_depth_router",
    "depth_token_mask": "depth_token_mask_block",
    "tropical_center": "tropical_center_block",
    "tropical_attention": "tropical_center_block",
    "tropical_add": "tropical_residual",
    "cumsum": "cumulative_sequence",
    "sqrt": "sqrt_gated_ffn",
    "norm_last": "reduce_attend",
    "mean_last": "reduce_attend",
    "max_last": "reduce_attend",
    "sum_last": "reduce_attend",
    "diff_attention": "diff_attention_block",
    "causal_mask": "causal_mix_block",
    "fused_linear_gelu": "fused_gelu_ffn",
    "exp": "exp_gated_residual",
    "integral_kernel": "integral_kernel_block",
    "sliding_window_mask": "windowed_attention",
    "local_window_attn": "local_attention_block",
    "state_space": "state_space_block",
    "rwkv_time_mixing": "rwkv_block",
    "reciprocal": "reciprocal_gated",
    "sign_ste": "sign_ste_gated",
    "log": "log_gated",
    "ultrametric_attention": "ultrametric_attention_block",
    "graph_attention": "graph_attention_block",
    "dual_compression_blend": "signal_routed_compression",
    "signal_conditioned_compression": "signal_routed_compression",
    "score_depth_blend": "mixed_recursion",
    "difficulty_blend_3way": "three_lane_adaptive",
    "relu_gated_moe": "moe",
    "gated_lane_blend": "gated_lane_blend_block",
    "depth_gated_transform": "depth_gated_block",
}

EFFICIENCY_TEMPLATES: FrozenSet[str] = frozenset(
    {
        "sparse_moe_block",
        "routed_bottleneck",
        "token_merge_block",
        "conditional_compute",
        "sparse_ffn",
        "moe",
        "attn_bottleneck_hybrid",
        "attn_sparse_moe",
        "attn_moe_block",
        "latent_attn_sparse_ffn",
        "latent_attn_moe",
        "local_attn_moe",
        "diff_attn_moe",
    }
)

ROUTING_COMPRESSION_MOE_OPS: FrozenSet[str] = frozenset(
    {
        "hybrid_token_gate",
        "sparse_span_builder",
        "hybrid_sparse_router",
        "lane_conditioned_block",
        "default_path",
        "token_entropy",
        "token_class_proj",
        "feature_sparsity",
        "gated_lane_blend",
        "depth_gated_transform",
        "difficulty_blend_3way",
        "score_depth_blend",
        "confidence_token_gate",
        "learned_token_gate",
        "cheap_verify_blend",
        "depth_weighted_proj",
        "depth_token_mask",
        "adjacent_token_merge",
        "relu_gated_moe",
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        "moe_topk",
        "moe_2expert",
        "sparse_bottleneck_moe",
        "tropical_moe",
        "topk_gate",
        "tropical_gate",
        "tropical_router",
        "sparse_threshold",
        "lif_neuron",
        "padic_gate",
        "signal_conditioned_compression",
        "adaptive_rank_gate",
        "dual_compression_blend",
        "latent_attention_compressor",
    }
)

MIN_DIM_OPS: Dict[str, int] = {
    "softmax_attention": 16,
    "linear_attention": 16,
    "graph_attention": 16,
    "multi_head_mix": 4,
    "selective_scan": 8,
    "state_space": 8,
    "rwkv_time_mixing": 8,
    "rwkv_channel": 8,
    "conv1d_seq": 4,
    "moe_topk": 8,
    "moe_2expert": 8,
    "swiglu_mlp": 4,
    "topk_gate": 4,
    "block_sparse_linear": 16,
    "nm_sparse_linear": 8,
    "low_rank_proj": 8,
    "bottleneck_proj": 8,
    "grouped_linear": 8,
    "shared_basis_proj": 8,
    "gated_linear": 8,
    "ternary_projection": 8,
    "linear_proj": 4,
    "linear_proj_down": 4,
    "linear_proj_up": 4,
    "fused_linear_gelu": 4,
    "difficulty_blend_3way": 8,
    "relu_gated_moe": 8,
}

_INDUCTION_STEERING_TARGET = 0.01
_BINDING_STEERING_TARGET = 0.005
_BINDING_COMPOSITE_STEERING_TARGET = 0.006
_AR_STEERING_TARGET = 0.04
_HELLASWAG_STEERING_TARGET = 0.30
_BLIMP_STEERING_TARGET = 0.60
_INDUCTION_V2_STEERING_TARGET = 0.06
_BINDING_V2_STEERING_TARGET = 0.06
_LOSS_STEERING_ANCHOR = 0.65
_SLOT_OUTCOME_MIN_RESCUE_SUPPORT = 3
_SLOT_OUTCOME_SUPPORT_PRIOR = 8.0
_CAPABILITY_SUPPORT_PRIOR = 16.0


def _bounded(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _finite_or_none(value) -> Optional[float]:
    import math as _math

    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if _math.isfinite(out) else None


def _loss_quality_factor(mean_loss, *, k: float = 2.0) -> float:
    import math as _math

    loss = _finite_or_none(mean_loss)
    if loss is None:
        return 1.0
    return _bounded(
        _math.exp(-k * (loss - _LOSS_STEERING_ANCHOR)),
        lo=0.35,
        hi=2.25,
    )


def _s1_quality_factor(s1_rate: float) -> float:
    return _bounded(0.75 + (1.25 * float(s1_rate)), lo=0.75, hi=2.0)


def _support_confidence(n: int, *, prior: float = _CAPABILITY_SUPPORT_PRIOR) -> float:
    n = max(int(n or 0), 0)
    return n / max(n + float(prior), 1.0)


def _support_shrunk_reward(
    reward: float,
    n: int,
    *,
    prior: float = _CAPABILITY_SUPPORT_PRIOR,
) -> float:
    return 1.0 + (
        (_bounded(reward, lo=0.0, hi=4.0) - 1.0) * _support_confidence(n, prior=prior)
    )


def _support_shrunk_multiplier(
    multiplier: float,
    n: int,
    *,
    prior: float = _CAPABILITY_SUPPORT_PRIOR,
) -> float:
    return 1.0 + ((float(multiplier) - 1.0) * _support_confidence(n, prior=prior))


def _capability_reward(
    avg_induction_auc,
    avg_binding_auc,
    avg_binding_composite,
    avg_ar_auc=None,
    avg_hellaswag_acc=None,
    avg_blimp_overall_accuracy=None,
    avg_induction_v2_investigation_auc=None,
    avg_binding_v2_investigation_auc=None,
    math_space_rate=None,
) -> float:
    induction = _bounded(
        (_finite_or_none(avg_induction_auc) or 0.0) / _INDUCTION_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    binding = _bounded(
        (_finite_or_none(avg_binding_auc) or 0.0) / _BINDING_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    binding_composite = _bounded(
        (_finite_or_none(avg_binding_composite) or 0.0)
        / _BINDING_COMPOSITE_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    ar_auc = _bounded(
        (_finite_or_none(avg_ar_auc) or 0.0) / _AR_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    hellaswag = _bounded(
        (_finite_or_none(avg_hellaswag_acc) or 0.0) / _HELLASWAG_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    blimp = _bounded(
        (_finite_or_none(avg_blimp_overall_accuracy) or 0.0) / _BLIMP_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    induction_v2 = _bounded(
        (_finite_or_none(avg_induction_v2_investigation_auc) or 0.0)
        / _INDUCTION_V2_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    binding_v2 = _bounded(
        (_finite_or_none(avg_binding_v2_investigation_auc) or 0.0)
        / _BINDING_V2_STEERING_TARGET,
        lo=0.0,
        hi=2.0,
    )
    math_share = _bounded(_finite_or_none(math_space_rate) or 0.0, lo=0.0, hi=1.0)
    return (
        1.0
        + (0.16 * induction)
        + (0.14 * binding)
        + (0.12 * binding_composite)
        + (0.10 * ar_auc)
        + (0.07 * hellaswag)
        + (0.07 * blimp)
        + (0.17 * induction_v2)
        + (0.17 * binding_v2)
        + (0.10 * math_share)
    )


def _capability_score(
    avg_induction_auc,
    avg_binding_auc,
    avg_binding_composite,
    avg_ar_auc=None,
    avg_hellaswag_acc=None,
    avg_blimp_overall_accuracy=None,
    avg_induction_v2_investigation_auc=None,
    avg_binding_v2_investigation_auc=None,
    math_space_rate=None,
) -> float:
    return (
        _capability_reward(
            avg_induction_auc,
            avg_binding_auc,
            avg_binding_composite,
            avg_ar_auc,
            avg_hellaswag_acc,
            avg_blimp_overall_accuracy,
            avg_induction_v2_investigation_auc,
            avg_binding_v2_investigation_auc,
            math_space_rate,
        )
        - 1.0
    )


def _metric_select_list(columns: set[str], metric_columns: tuple[str, ...]) -> str:
    select_parts = []
    for col_name in metric_columns:
        if col_name in columns:
            select_parts.append(col_name)
        else:
            select_parts.append(f"NULL AS {col_name}")
    return ", ".join(select_parts)


def _slot_outcome_capability(vals: dict) -> float:
    return _capability_score(
        vals.get("mean_induction_auc"),
        vals.get("mean_binding_auc"),
        vals.get("mean_binding_composite"),
        vals.get("mean_ar_auc"),
        vals.get("mean_hellaswag_acc"),
        vals.get("mean_blimp_overall_accuracy"),
        vals.get("mean_induction_v2_investigation_auc"),
        vals.get("mean_binding_v2_investigation_auc"),
        vals.get("math_space_rate"),
    )


def _support_shrunk_slot_outcome_capability(vals: dict) -> Optional[float]:
    try:
        n = int(vals.get("n") or 0)
    except (TypeError, ValueError):
        return None
    if n < _SLOT_OUTCOME_MIN_RESCUE_SUPPORT:
        return None
    capability = _slot_outcome_capability(vals)
    confidence = n / (n + _SLOT_OUTCOME_SUPPORT_PRIOR)
    return capability * confidence


def blend_template_weights_with_db(
    template_weights: Dict[str, float],
    db_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Apply DB empirical template evidence to existing priors.

    Runner-injected template maps may come from older signal paths. Treat those
    as priors and multiply in a bounded DB/default ratio so the refreshed stats
    can steer without allowing either source to dominate outright.
    """
    if not db_weights:
        return dict(template_weights)

    template_items = tuple(
        sorted((str(k), float(v)) for k, v in template_weights.items())
    )
    db_items = tuple(sorted((str(k), float(v)) for k, v in db_weights.items()))
    return dict(_blend_template_weights_cached(template_items, db_items))


@lru_cache(maxsize=128)
def _blend_template_weights_cached(
    template_items: tuple[tuple[str, float], ...],
    db_items: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    from .templates import DEFAULT_TEMPLATE_WEIGHTS

    blended = dict(template_items)
    for tpl_name, db_weight in db_items:
        try:
            db_w = float(db_weight)
        except (TypeError, ValueError):
            continue
        if db_w <= 0.0:
            continue
        default_w = float(DEFAULT_TEMPLATE_WEIGHTS.get(tpl_name, 1.0) or 1.0)
        default_w = max(default_w, 0.01)
        ratio = _bounded(db_w / default_w, lo=0.20, hi=5.0)
        if tpl_name in blended:
            try:
                prior_w = float(blended[tpl_name])
            except (TypeError, ValueError):
                prior_w = default_w
            blended[tpl_name] = prior_w * (ratio**0.5)
        else:
            blended[tpl_name] = db_w
    return tuple(blended.items())


def _load_template_slot_context(conn) -> Dict[str, dict]:
    import json as _json

    columns = {row[1] for row in conn.execute("PRAGMA table_info(slot_stats)")}
    metric_columns = (
        "avg_induction_auc",
        "avg_binding_auc",
        "avg_binding_composite",
        "avg_ar_auc",
        "avg_hellaswag_acc",
        "avg_blimp_overall_accuracy",
        "avg_induction_v2_investigation_auc",
        "avg_binding_v2_investigation_auc",
        "math_space_rate",
    )
    rows = conn.execute(
        f"""SELECT template_name, class_outcomes, wildcard_class_outcomes,
                   {_metric_select_list(columns, metric_columns)}
            FROM slot_stats
            WHERE eval_count >= 3"""
    ).fetchall()
    out: Dict[str, dict] = {}
    for row in rows:
        template_name = str(row[0] or "").strip()
        if not template_name:
            continue
        slot_score = _capability_score(
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
        )
        best_class_score = slot_score
        for raw_payload in (row[1], row[2]):
            if not raw_payload:
                continue
            try:
                payload = _json.loads(raw_payload)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            for vals in payload.values():
                if isinstance(vals, dict):
                    supported_score = _support_shrunk_slot_outcome_capability(vals)
                    if supported_score is not None:
                        best_class_score = max(best_class_score, supported_score)
        ctx = out.setdefault(
            template_name,
            {
                "slot_scores": [],
                "slot_potentials": [],
                "salvage_gaps": [],
            },
        )
        ctx["slot_scores"].append(slot_score)
        ctx["slot_potentials"].append(best_class_score)
        ctx["salvage_gaps"].append(max(0.0, best_class_score - slot_score))
    for template_name, ctx in out.items():
        scores = ctx["slot_scores"]
        potentials = ctx["slot_potentials"]
        gaps = ctx["salvage_gaps"]
        weak_slots = sum(1 for score in scores if score <= 0.15)
        out[template_name] = {
            "mean_slot_score": sum(scores) / len(scores) if scores else 0.0,
            "mean_slot_potential": sum(potentials) / len(potentials)
            if potentials
            else 0.0,
            "max_slot_potential": max(potentials) if potentials else 0.0,
            "mean_salvage_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "weak_slot_fraction": weak_slots / max(len(scores), 1),
        }
    return out


def compute_motif_weights_from_op_weights(
    op_weights: Dict[str, float],
) -> Dict[str, tuple]:
    """Geometric mean of op weights per motif, cached by op_weights content."""
    global _motif_weight_cache_key, _motif_weight_cache_val
    cache_key = tuple(sorted(op_weights.items()))
    with _motif_weight_lock:
        if cache_key == _motif_weight_cache_key:
            return _motif_weight_cache_val

    import math as _math
    from .motifs import ALL_MOTIFS

    result: Dict[str, tuple] = {}
    for motif in ALL_MOTIFS:
        motif_ops = [step.op_name for step in motif.steps]
        factors = [op_weights.get(op, 1.0) for op in motif_ops]
        factor = _math.exp(sum(_math.log(max(f, 0.01)) for f in factors) / len(factors))
        factor = max(0.1, min(8.0, factor))
        result[motif.name] = (factor, motif.lift)

    with _motif_weight_lock:
        _motif_weight_cache_key = cache_key
        _motif_weight_cache_val = result
    return result


def check_graph_space_consistency(graph: ComputationGraph) -> Optional[str]:
    """Validate that a computation graph has no algebraic space conflicts."""
    for _, node in sorted(graph.nodes.items()):
        if node.op_name == "input":
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is None:
            continue
        op_type = op.algebraic_type
        for in_id in node.input_ids:
            in_node = graph.nodes.get(in_id)
            if in_node is None or in_node.op_name == "input":
                continue
            in_op = PRIMITIVE_REGISTRY.get(in_node.op_name)
            if in_op is None:
                continue
            in_type = in_op.algebraic_type
            if not algebraic_types_compatible(in_type, op_type):
                return (
                    f"Space conflict: {in_node.op_name} ({in_type.space}/{in_type.output_guarantee}) -> "
                    f"{node.op_name} ({op_type.space}/{op_type.input_constraint})"
                )
    return None


class DBTemplateWeightCache:
    """TTL-bounded cache for DB template weights. No global mutable state."""

    __slots__ = ("_weights", "_expires", "_ttl")

    def __init__(self, ttl: float = 60.0):
        self._weights: Optional[Dict[str, float]] = None
        self._expires: float = 0.0
        self._ttl = ttl

    def get(
        self, db_path: str = "research/lab_notebook.db"
    ) -> Optional[Dict[str, float]]:
        import sqlite3
        import time as _time

        now = _time.time()
        if self._weights is not None and now < self._expires:
            return self._weights

        try:
            from pathlib import Path

            path = Path(db_path)
            if not path.is_absolute():
                cwd = Path.cwd()
                if (
                    cwd.name == "research"
                    and path.parts
                    and path.parts[0] == "research"
                ):
                    path = cwd.parent / db_path
                else:
                    path = path.resolve()
            if not path.exists():
                return None

            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(template_stats)")
            }
            metric_columns = (
                "avg_induction_auc",
                "avg_binding_auc",
                "avg_binding_composite",
                "avg_ar_auc",
                "avg_hellaswag_acc",
                "avg_blimp_overall_accuracy",
                "avg_induction_v2_investigation_auc",
                "avg_binding_v2_investigation_auc",
                "math_space_rate",
            )
            rows = conn.execute(
                f"""SELECT template_name, eval_count, s1_pass_count, mean_loss,
                           {_metric_select_list(columns, metric_columns)}
                    FROM template_stats WHERE eval_count >= 5"""
            ).fetchall()
            template_slot_context = _load_template_slot_context(conn)
            conn.close()

            if not rows:
                return None

            from .templates import DEFAULT_TEMPLATE_WEIGHTS

            k = 3.0
            db_weights: Dict[str, float] = {}
            for (
                tpl_name,
                eval_count,
                s1_count,
                mean_loss,
                avg_induction_auc,
                avg_binding_auc,
                avg_binding_composite,
                avg_ar_auc,
                avg_hellaswag_acc,
                avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc,
                avg_binding_v2_investigation_auc,
                math_space_rate,
            ) in rows:
                s1_rate = s1_count / max(eval_count, 1)
                static_weight = DEFAULT_TEMPLATE_WEIGHTS.get(tpl_name, 1.0)
                perf_weight = (
                    static_weight
                    * _loss_quality_factor(mean_loss, k=k)
                    * _s1_quality_factor(s1_rate)
                )
                signal_reward = _support_shrunk_reward(
                    _capability_reward(
                        avg_induction_auc,
                        avg_binding_auc,
                        avg_binding_composite,
                        avg_ar_auc,
                        avg_hellaswag_acc,
                        avg_blimp_overall_accuracy,
                        avg_induction_v2_investigation_auc,
                        avg_binding_v2_investigation_auc,
                        math_space_rate,
                    ),
                    eval_count,
                    prior=20.0,
                )
                perf_weight *= signal_reward
                confidence = eval_count / max(eval_count + 12.0, 1.0)
                dynamic_weight = perf_weight
                slot_ctx = template_slot_context.get(str(tpl_name)) or {}
                slot_mean_potential = float(slot_ctx.get("mean_slot_potential") or 0.0)
                slot_max_potential = float(slot_ctx.get("max_slot_potential") or 0.0)
                mean_salvage_gap = float(slot_ctx.get("mean_salvage_gap") or 0.0)
                weak_slot_fraction = float(slot_ctx.get("weak_slot_fraction") or 1.0)
                if slot_ctx:
                    rescue_score = max(slot_mean_potential, 0.75 * slot_max_potential)
                    if rescue_score > (signal_reward - 1.0):
                        rescue_weight = (
                            static_weight
                            * _loss_quality_factor(mean_loss, k=k)
                            * _s1_quality_factor(s1_rate)
                            * (1.0 + rescue_score)
                        )
                        rescue_mix = _bounded(
                            (0.55 * mean_salvage_gap)
                            + (0.20 * (1.0 - weak_slot_fraction)),
                            lo=0.0,
                            hi=0.45,
                        )
                        dynamic_weight = ((1.0 - rescue_mix) * dynamic_weight) + (
                            rescue_mix * rescue_weight
                        )
                db_weights[tpl_name] = ((1.0 - confidence) * static_weight) + (
                    confidence * dynamic_weight
                )

            for tpl_name, w in DEFAULT_TEMPLATE_WEIGHTS.items():
                if tpl_name not in db_weights:
                    db_weights[tpl_name] = w

            eval_counts = {r[0]: r[1] for r in rows}
            if eval_counts:
                sorted_counts = sorted(eval_counts.values())
                median_evals = sorted_counts[len(sorted_counts) // 2]
                if median_evals > 0:
                    for tpl_name in db_weights:
                        n = eval_counts.get(tpl_name, 0)
                        if n < median_evals:
                            curiosity = 1.0 + (1.0 - n / median_evals)
                            db_weights[tpl_name] *= curiosity

            self._weights = db_weights
            self._expires = now + self._ttl
            logger.info(
                "Loaded DB template weights for %d templates (%.0f%% from DB)",
                len(db_weights),
                len(rows) / max(len(db_weights), 1) * 100,
            )
            return db_weights
        except Exception as e:
            logger.debug("Failed to load DB template weights: %s", e)
            return None


class DBOpWeightCache:
    """TTL-bounded cache for DB op weights."""

    __slots__ = ("_weights", "_expires", "_ttl")

    def __init__(self, ttl: float = 60.0):
        self._weights: Optional[Dict[str, float]] = None
        self._expires: float = 0.0
        self._ttl = ttl

    def get(
        self, db_path: str = "research/lab_notebook.db"
    ) -> Optional[Dict[str, float]]:
        import sqlite3
        import time as _time

        now = _time.time()
        if self._weights is not None and now < self._expires:
            return self._weights

        try:
            from pathlib import Path

            path = Path(db_path)
            if not path.is_absolute():
                cwd = Path.cwd()
                if (
                    cwd.name == "research"
                    and path.parts
                    and path.parts[0] == "research"
                ):
                    path = cwd.parent / db_path
                else:
                    path = path.resolve()
            if not path.exists():
                return None

            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(op_stats)")}
            metric_columns = (
                "avg_induction_auc",
                "avg_binding_auc",
                "avg_binding_composite",
                "avg_ar_auc",
                "avg_hellaswag_acc",
                "avg_blimp_overall_accuracy",
                "avg_induction_v2_investigation_auc",
                "avg_binding_v2_investigation_auc",
                "math_space_rate",
            )
            rows = conn.execute(
                f"""SELECT op_name, eval_count, s1_pass_count, mean_loss,
                           {_metric_select_list(columns, metric_columns)}
                    FROM op_stats
                    WHERE eval_count >= 3"""
            ).fetchall()
            conn.close()

            if not rows:
                return None

            weights: Dict[str, float] = {}
            for (
                op_name,
                eval_count,
                s1_count,
                mean_loss,
                avg_induction_auc,
                avg_binding_auc,
                avg_binding_composite,
                avg_ar_auc,
                avg_hellaswag_acc,
                avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc,
                avg_binding_v2_investigation_auc,
                math_space_rate,
            ) in rows:
                s1_rate = s1_count / max(eval_count, 1)
                perf_term = _loss_quality_factor(mean_loss, k=2.0) * _s1_quality_factor(
                    s1_rate
                )
                capability_term = _support_shrunk_reward(
                    _capability_reward(
                        avg_induction_auc,
                        avg_binding_auc,
                        avg_binding_composite,
                        avg_ar_auc,
                        avg_hellaswag_acc,
                        avg_blimp_overall_accuracy,
                        avg_induction_v2_investigation_auc,
                        avg_binding_v2_investigation_auc,
                        math_space_rate,
                    ),
                    eval_count,
                    prior=12.0,
                )
                raw_weight = perf_term * capability_term
                raw_weight = _support_shrunk_multiplier(
                    raw_weight,
                    eval_count,
                    prior=10.0,
                )
                weights[str(op_name)] = round(
                    _bounded(raw_weight, lo=0.25, hi=4.5),
                    4,
                )

            self._weights = weights
            self._expires = now + self._ttl
            if weights:
                logger.info("Loaded DB op weights for %d ops", len(weights))
            return weights
        except Exception as e:
            logger.debug("Failed to load DB op weights: %s", e)
            return None


class SlotAdaptationCache:
    """TTL-bounded cache for slot class adaptations learned from wildcard fills."""

    __slots__ = ("_adaptations", "_expires", "_ttl")

    _MIN_EVALS = 5
    _MAX_EXTRA_CLASSES = 2

    def __init__(self, ttl: float = 120.0):
        self._adaptations: Optional[Dict[str, list]] = None
        self._expires: float = 0.0
        self._ttl = ttl

    def get(self, db_path: str = "research/lab_notebook.db") -> Dict[str, list]:
        import json as _json
        import sqlite3
        import time as _time

        now = _time.time()
        if self._adaptations is not None and now < self._expires:
            return self._adaptations

        adaptations: Dict[str, list] = {}
        try:
            from pathlib import Path

            path = Path(db_path)
            if not path.is_absolute():
                cwd = Path.cwd()
                if (
                    cwd.name == "research"
                    and path.parts
                    and path.parts[0] == "research"
                ):
                    path = cwd.parent / db_path
                else:
                    path = path.resolve()
            if not path.exists():
                return adaptations

            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(slot_stats)")}
            metric_columns = (
                "avg_induction_auc",
                "avg_binding_auc",
                "avg_binding_composite",
                "avg_ar_auc",
                "avg_hellaswag_acc",
                "avg_blimp_overall_accuracy",
                "avg_induction_v2_investigation_auc",
                "avg_binding_v2_investigation_auc",
                "math_space_rate",
            )
            rows = conn.execute(
                f"""SELECT slot_key, slot_classes, s1_pass_count, eval_count,
                           {_metric_select_list(columns, metric_columns)}, wildcard_class_outcomes
                    FROM slot_stats
                    WHERE wildcard_count >= ?""",
                (self._MIN_EVALS,),
            ).fetchall()
            conn.close()

            for (
                slot_key,
                slot_classes_json,
                s1_total,
                eval_total,
                avg_induction_auc,
                avg_binding_auc,
                avg_binding_composite,
                avg_ar_auc,
                avg_hellaswag_acc,
                avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc,
                avg_binding_v2_investigation_auc,
                math_space_rate,
                wc_json,
            ) in rows:
                if not wc_json:
                    continue
                try:
                    prescribed = set(_json.loads(slot_classes_json or "[]"))
                    wc_outcomes = _json.loads(wc_json)
                except (ValueError, TypeError):
                    continue

                baseline_s1_rate = s1_total / max(eval_total, 1)
                baseline_capability = _capability_score(
                    avg_induction_auc,
                    avg_binding_auc,
                    avg_binding_composite,
                    avg_ar_auc,
                    avg_hellaswag_acc,
                    avg_blimp_overall_accuracy,
                    avg_induction_v2_investigation_auc,
                    avg_binding_v2_investigation_auc,
                    math_space_rate,
                )
                extra: list = []
                scored_extra: list[tuple[float, str]] = []
                for cls, vals in wc_outcomes.items():
                    if cls in prescribed:
                        continue
                    n = vals.get("n", 0)
                    s1 = vals.get("s1", 0)
                    if n < self._MIN_EVALS:
                        continue
                    cls_s1_rate = s1 / n
                    cls_capability = _capability_score(
                        vals.get("mean_induction_auc"),
                        vals.get("mean_binding_auc"),
                        vals.get("mean_binding_composite"),
                        vals.get("mean_ar_auc"),
                        vals.get("mean_hellaswag_acc"),
                        vals.get("mean_blimp_overall_accuracy"),
                        vals.get("mean_induction_v2_investigation_auc"),
                        vals.get("mean_binding_v2_investigation_auc"),
                        vals.get("math_space_rate"),
                    )
                    if cls_s1_rate > baseline_s1_rate or cls_capability > (
                        baseline_capability + 0.2
                    ):
                        scored_extra.append(
                            (
                                (cls_s1_rate - baseline_s1_rate)
                                + (0.5 * (cls_capability - baseline_capability)),
                                cls,
                            )
                        )
                for _, cls in sorted(scored_extra, reverse=True):
                    extra.append(cls)
                    if len(extra) >= self._MAX_EXTRA_CLASSES:
                        break

                if extra:
                    adaptations[slot_key] = extra

            self._adaptations = adaptations
            self._expires = now + self._ttl
            if adaptations:
                logger.info(
                    "Loaded slot adaptations: %d slots with expanded classes",
                    len(adaptations),
                )
        except Exception as e:
            logger.debug("Failed to load slot adaptations: %s", e)

        return adaptations


@dataclass(slots=True)
class EfficiencyPrior:
    """Uses historical Pareto frontier data to bias synthesis."""

    op_biases: Dict[str, float]

    def __init__(self, frontier_data: List[Dict]):
        self.op_biases = {}
        for p in frontier_data or []:
            graph_json = p.get("graph_json", "")
            if not graph_json:
                continue
            for motif in [
                "selective_scan",
                "tropical",
                "clifford",
                "low_rank",
                "sparse",
            ]:
                if motif in graph_json:
                    mult = 1.12 if motif == "tropical" else 1.05
                    self.op_biases[motif] = self.op_biases.get(motif, 1.0) * mult

    def get_bias(self, op_name: str) -> float:
        bias = 1.0
        for motif, multiplier in self.op_biases.items():
            if motif in op_name:
                bias *= multiplier
        return min(2.5, bias)


def check_shape_compat(
    op: PrimitiveOp,
    input_shapes: List[ShapeInfo],
    model_dim: int,
    current_space: str = "euclidean",
) -> bool:
    """Quick check if an op is compatible with given input shapes and space."""
    if not algebraic_types_compatible(
        default_algebraic_type_for_space(current_space),
        op.algebraic_type,
    ):
        return False

    if not input_shapes or op.n_inputs != len(input_shapes):
        return False

    s0 = input_shapes[0]
    if op.name == "split2" and (s0.dim % 2 != 0 or s0.dim // 2 < 4):
        return False
    if op.name == "split3" and (s0.dim % 3 != 0 or s0.dim // 3 < 4):
        return False
    if op.shape_rule == "rfft" and not s0.is_standard:
        return False
    if op.shape_rule == "irfft" and not s0.is_freq_domain:
        return False

    if (
        op.name
        in {
            "local_window_attn",
            "sliding_window_mask",
            "token_pool_restore",
            "selective_scan",
            "conv1d_seq",
            "basis_expansion",
            "integral_kernel",
            "fixed_point_iter",
        }
        and not s0.is_standard
    ):
        return False

    min_dim = MIN_DIM_OPS.get(op.name)
    if min_dim and s0.dim < min_dim:
        return False

    if len(input_shapes) == 2:
        s1 = input_shapes[1]
        if op.shape_rule == "binary_broadcast":
            if s0.seq != s1.seq:
                return False
            if s0.dim != s1.dim and s0.dim != 1 and s1.dim != 1:
                return False
        elif op.shape_rule in ("matmul", "concat") and s0.seq != s1.seq:
            return False

    return True
