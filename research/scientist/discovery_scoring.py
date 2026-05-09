from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, Optional

TIER_ORDER = {
    "breakthrough": 4,
    "validation": 3,
    "investigation": 2,
    "screening": 1,
    "screened_out": 0,
}

BONUS_WEIGHTS = {
    "efficiency": 6.0,
    "routing": 7.0,
    "adaptive": 6.0,
    "sparsity": 5.0,
    "learningSpeed": 4.0,
    "externalComparison": 6.0,
    "robustness": 7.0,
    "referenceDelta": 8.0,
    "binding": 10.0,
    "blimp": 5.0,
}

MAX_TOTAL_BONUS = 40.0
PARAM_EXP_RANGE = {"min": 5.0, "max": 10.0}
FLOPS_EXP_RANGE = {"min": 6.0, "max": 13.0}

EXTERNAL_BASELINES = {
    "Attention": {
        "paramEfficiency": 1.0,
        "flopEfficiency": 1.0,
        "throughputRatio": 1.0,
        "learningSpeedRatio": 1.0,
    },
    "Hybrid-Attention": {
        "paramEfficiency": 1.15,
        "flopEfficiency": 1.05,
        "throughputRatio": 0.9,
        "learningSpeedRatio": 1.1,
    },
    "MoE-Attention": {
        "paramEfficiency": 3.5,
        "flopEfficiency": 1.0,
        "throughputRatio": 0.85,
        "learningSpeedRatio": 1.2,
    },
    "Routed-MoE": {
        "paramEfficiency": 3.5,
        "flopEfficiency": 1.0,
        "throughputRatio": 0.85,
        "learningSpeedRatio": 1.2,
    },
    "MoE-Hybrid-Attention": {
        "paramEfficiency": 3.0,
        "flopEfficiency": 0.95,
        "throughputRatio": 0.8,
        "learningSpeedRatio": 1.15,
    },
    "Mamba-SSM": {
        "paramEfficiency": 0.85,
        "flopEfficiency": 1.2,
        "throughputRatio": 4.5,
        "learningSpeedRatio": 0.9,
    },
    "Hybrid-SSM": {
        "paramEfficiency": 1.1,
        "flopEfficiency": 1.15,
        "throughputRatio": 2.5,
        "learningSpeedRatio": 1.1,
    },
    "MoE-Mamba-SSM": {
        "paramEfficiency": 3.0,
        "flopEfficiency": 1.1,
        "throughputRatio": 3.5,
        "learningSpeedRatio": 1.05,
    },
    "Adaptive-Attention": {
        "paramEfficiency": 1.2,
        "flopEfficiency": 1.4,
        "throughputRatio": 1.5,
        "learningSpeedRatio": 1.1,
    },
    "Adaptive-Hybrid-Attention": {
        "paramEfficiency": 1.25,
        "flopEfficiency": 1.35,
        "throughputRatio": 1.4,
        "learningSpeedRatio": 1.1,
    },
    "Adaptive-Mamba-SSM": {
        "paramEfficiency": 0.9,
        "flopEfficiency": 1.5,
        "throughputRatio": 5.0,
        "learningSpeedRatio": 0.95,
    },
    "Adaptive-MLP-Mixer": {
        "paramEfficiency": 1.3,
        "flopEfficiency": 1.1,
        "throughputRatio": 1.1,
        "learningSpeedRatio": 1.05,
    },
    "Conv-Mixer": {
        "paramEfficiency": 0.95,
        "flopEfficiency": 1.1,
        "throughputRatio": 1.2,
        "learningSpeedRatio": 0.95,
    },
    "Spectral-Mixer": {
        "paramEfficiency": 0.9,
        "flopEfficiency": 1.05,
        "throughputRatio": 1.15,
        "learningSpeedRatio": 0.9,
    },
    "Spectral-Conv": {
        "paramEfficiency": 0.92,
        "flopEfficiency": 1.08,
        "throughputRatio": 1.2,
        "learningSpeedRatio": 0.92,
    },
    "Gated-MLP": {
        "paramEfficiency": 0.85,
        "flopEfficiency": 0.95,
        "throughputRatio": 1.3,
        "learningSpeedRatio": 0.85,
    },
    "MLP-Mixer": {
        "paramEfficiency": 0.8,
        "flopEfficiency": 0.9,
        "throughputRatio": 1.4,
        "learningSpeedRatio": 0.8,
    },
    "Nonlinear-Mixer": {
        "paramEfficiency": 0.75,
        "flopEfficiency": 0.85,
        "throughputRatio": 1.5,
        "learningSpeedRatio": 0.75,
    },
    "Hybrid-Mixer": {
        "paramEfficiency": 0.95,
        "flopEfficiency": 1.0,
        "throughputRatio": 1.1,
        "learningSpeedRatio": 0.95,
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_score(score: float) -> int:
    return int(round(max(0.0, score)))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _normalize_loss_ratio(loss_ratio: Any) -> float:
    loss = _to_float(loss_ratio)
    return _clamp01(1.0 - loss) if loss is not None else 0.0


def _normalize_inverse_log10(
    value: Any, min_exp: float, max_exp: float
) -> Optional[float]:
    num = _to_float(value)
    if num is None or num <= 0:
        return None
    exp = math.log10(num)
    return _clamp01(1.0 - (exp - min_exp) / (max_exp - min_exp))


def _pick_first_number(entry: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        num = _to_float(entry.get(key))
        if num is not None:
            return num
    return None


def _average_scores(scores: list[float]) -> Optional[float]:
    if not scores:
        return None
    return sum(scores) / len(scores)


def _parse_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _get_expert_count(entry: Dict[str, Any]) -> Optional[float]:
    direct = _pick_first_number(
        entry, ("routing_expert_count", "expert_count", "n_experts")
    )
    if direct is not None:
        return direct
    parsed = _parse_json_value(entry.get("routing_expert_utilization_json"))
    if isinstance(parsed, list):
        return float(len(parsed))
    if isinstance(parsed, dict):
        return float(len(parsed))
    return None


def _normalize_routing_entropy(
    entropy: Any, n_experts: Optional[float]
) -> Optional[float]:
    ent = _to_float(entropy)
    if ent is None:
        return None
    if n_experts is not None and n_experts > 1:
        max_entropy = math.log2(n_experts)
        if max_entropy > 0:
            return _clamp01(ent / max_entropy)
    return _clamp01(ent)


def _resolve_baseline(family: Any) -> Optional[Dict[str, Any]]:
    fam = str(family or "").strip()
    if not fam or fam == "Unknown":
        return None
    if fam in EXTERNAL_BASELINES:
        return {"key": fam, "baseline": EXTERNAL_BASELINES[fam], "fuzzy": False}

    stripped = fam
    while "-" in stripped:
        stripped = stripped.split("-", 1)[1]
        if stripped in EXTERNAL_BASELINES:
            return {
                "key": stripped,
                "baseline": EXTERNAL_BASELINES[stripped],
                "fuzzy": True,
            }

    best_key = None
    best_len = 0
    for key in EXTERNAL_BASELINES:
        if key in fam and len(key) > best_len:
            best_key = key
            best_len = len(key)
    if best_key is not None:
        return {
            "key": best_key,
            "baseline": EXTERNAL_BASELINES[best_key],
            "fuzzy": True,
        }
    return {
        "key": "Hybrid-Mixer",
        "baseline": EXTERNAL_BASELINES["Hybrid-Mixer"],
        "fuzzy": True,
    }


def _scurve(ratio: float, k: float = 4.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))


def _compute_routing_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    n_experts = _get_expert_count(entry)
    entropy_score = _normalize_routing_entropy(
        entry.get("routing_utilization_entropy"), n_experts
    )
    if entropy_score is not None:
        scores.append(entropy_score)
    drop_rate = _to_float(entry.get("routing_drop_rate"))
    if drop_rate is not None:
        scores.append(_clamp01(1.0 - drop_rate))
    overflow = _to_float(entry.get("routing_capacity_overflow_count"))
    if overflow is not None:
        scores.append(_clamp01(1.0 - min(overflow / 5.0, 1.0)))
    conf_mean = _to_float(entry.get("routing_confidence_mean"))
    if conf_mean is not None:
        scores.append(_clamp01(conf_mean))
    conf_std = _to_float(entry.get("routing_confidence_std"))
    if conf_std is not None:
        scores.append(_clamp01(1.0 - conf_std / 0.3))
    tokens_total = _to_float(entry.get("routing_tokens_total"))
    tokens_processed = _to_float(entry.get("routing_tokens_processed"))
    if tokens_total and tokens_processed:
        scores.append(_clamp01((tokens_processed / tokens_total) / 0.95))
    avg = _average_scores(scores)
    if avg is None:
        return None
    moe_factor = 1.0
    if n_experts is not None and n_experts > 1:
        moe_factor *= min(1.5, 1.0 + math.log2(n_experts) / 6.0)
        if entropy_score is not None and entropy_score > 0.8:
            moe_factor *= 1.2
    if conf_mean is not None and conf_mean > 0.5:
        moe_factor *= 1.0 + 0.3 * (conf_mean - 0.5)
    if drop_rate is not None and drop_rate > 0.3:
        moe_factor *= max(0.5, 1.0 - (drop_rate - 0.3))
    return avg * BONUS_WEIGHTS["routing"] * moe_factor


def _compute_efficiency_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    params = (
        entry.get("param_count")
        if entry.get("param_count") is not None
        else entry.get("graph_n_params_estimate")
    )
    param_score = _normalize_inverse_log10(
        params, PARAM_EXP_RANGE["min"], PARAM_EXP_RANGE["max"]
    )
    if param_score is not None:
        scores.append(param_score)
    flops_score = _normalize_inverse_log10(
        entry.get("flops_forward"), FLOPS_EXP_RANGE["min"], FLOPS_EXP_RANGE["max"]
    )
    if flops_score is not None:
        scores.append(flops_score)
    throughput = _to_float(entry.get("throughput_tok_s"))
    if throughput is not None:
        routing_mode = str(entry.get("routing_mode") or "")
        compute_routing = str(entry.get("compute_routing") or "")
        target = (
            50000.0
            if routing_mode or compute_routing == "depth_token_mask"
            else 25000.0
        )
        scores.append(_clamp01(throughput / target))
    avg = _average_scores(scores)
    return None if avg is None else avg * BONUS_WEIGHTS["efficiency"]


def _compute_adaptive_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    depth_savings = _pick_first_number(
        entry,
        (
            "depth_savings_ratio",
            "adaptive_depth_savings",
            "depth_compute_savings",
            "depth_efficiency_gain",
        ),
    )
    if depth_savings is not None:
        scores.append(_clamp01(depth_savings / 0.5))
    depth_util = _pick_first_number(
        entry, ("effective_depth_ratio", "depth_utilization_ratio", "avg_depth_ratio")
    )
    if depth_util is not None:
        scores.append(_clamp01(1.0 - depth_util))
    recursion_savings = _pick_first_number(
        entry,
        (
            "recursion_savings_ratio",
            "recursion_compute_savings",
            "depth_weighted_proj_savings",
            "recursion_efficiency_gain",
        ),
    )
    if recursion_savings is not None:
        scores.append(_clamp01(recursion_savings / 0.5))
    avg = _average_scores(scores)
    return None if avg is None else avg * BONUS_WEIGHTS["adaptive"]


def _compute_sparsity_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    sparsity_ratio = _to_float(entry.get("sparsity_ratio"))
    if sparsity_ratio is not None:
        scores.append(_clamp01(sparsity_ratio / 0.5))
    params = (
        entry.get("param_count")
        if entry.get("param_count") is not None
        else entry.get("graph_n_params_estimate")
    )
    param_score = _normalize_inverse_log10(params, 4.0, 9.0)
    if param_score is not None:
        scores.append(param_score)
    memory = _to_float(entry.get("peak_memory_mb"))
    if memory is not None:
        scores.append(_clamp01(1.0 - memory / 500.0))
    activation_sparsity = _to_float(entry.get("activation_sparsity_score"))
    if activation_sparsity is not None and activation_sparsity > 0.3:
        scores.append(_clamp01((activation_sparsity - 0.3) / 0.5))
    avg = _average_scores(scores)
    if avg is None:
        return None
    multiplier = 1.3 if sparsity_ratio is not None and sparsity_ratio > 0.5 else 1.0
    return avg * BONUS_WEIGHTS["sparsity"] * multiplier


def _compute_learning_speed_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    lir = _to_float(entry.get("loss_improvement_rate"))
    if lir is not None:
        scores.append(_clamp01(lir))
    throughput = _to_float(entry.get("throughput_tok_s"))
    if throughput is not None:
        scores.append(_clamp01(throughput / 25000.0))
    forward_ms = _to_float(entry.get("forward_time_ms"))
    if forward_ms is not None:
        scores.append(_clamp01(1.0 - forward_ms / 50.0))
    avg = _average_scores(scores)
    return None if avg is None else avg * BONUS_WEIGHTS["learningSpeed"]


def _compute_external_comparison_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scaling_eff = _to_float(entry.get("scaling_param_efficiency"))
    if scaling_eff is not None and scaling_eff < 1.5:
        return 0.0
    resolved = _resolve_baseline(entry.get("architecture_family"))
    if resolved is None:
        return None
    baseline = resolved["baseline"]
    scores: list[float] = []
    loss_ratio = _to_float(entry.get("loss_ratio"))
    params = (
        entry.get("param_count")
        if entry.get("param_count") is not None
        else entry.get("graph_n_params_estimate")
    )
    if loss_ratio is not None and params is not None:
        learning = _clamp01(1.0 - loss_ratio)
        param_norm = _normalize_inverse_log10(
            params, PARAM_EXP_RANGE["min"], PARAM_EXP_RANGE["max"]
        )
        if param_norm is not None:
            scores.append(
                _clamp01(
                    (learning * param_norm) / (0.5 * baseline["paramEfficiency"]) / 1.5
                )
            )
    flops_per_param = _to_float(entry.get("flops_per_param"))
    if flops_per_param is None:
        flops_forward = _to_float(entry.get("flops_forward"))
        params_num = _to_float(params)
        if flops_forward is not None and params_num and params_num > 0:
            flops_per_param = flops_forward / params_num
    if flops_per_param is not None and loss_ratio is not None:
        learning = _clamp01(1.0 - loss_ratio)
        flop_norm = _normalize_inverse_log10(flops_per_param, 0.0, 4.0)
        if flop_norm is not None:
            scores.append(
                _clamp01(
                    (learning * flop_norm) / (0.5 * baseline["flopEfficiency"]) / 1.5
                )
            )
    throughput = _to_float(entry.get("throughput_tok_s"))
    if throughput is not None:
        scores.append(_clamp01(throughput / (25000.0 * baseline["throughputRatio"])))
    lir = _to_float(entry.get("loss_improvement_rate"))
    if lir is not None:
        expected_lir = 0.5 * baseline["learningSpeedRatio"]
        scores.append(_clamp01(lir / (expected_lir * 1.5)))
    avg = _average_scores(scores)
    if avg is None:
        return None
    excellence = 1.3 if avg > 0.75 else 1.1 if avg > 0.5 else 1.0
    return avg * BONUS_WEIGHTS["externalComparison"] * excellence


def _compute_robustness_bonus(entry: Dict[str, Any]) -> Optional[float]:
    scores: list[float] = []
    noise = _to_float(entry.get("robustness_noise_score"))
    if noise is not None:
        scores.append(_clamp01(1.0 - noise))
    long_ctx = _to_float(entry.get("robustness_long_ctx_score"))
    if long_ctx is not None:
        scores.append(_clamp01(long_ctx))
    quant = _to_float(entry.get("quant_int8_retention"))
    if quant is not None:
        q_pct = quant if quant <= 1.0 else quant / 100.0
        scores.append(_clamp01((q_pct - 0.5) / 0.5))
    init_std = _to_float(entry.get("init_sensitivity_std"))
    if init_std is not None:
        scores.append(_clamp01(1.0 - init_std / 0.2))
    spectral = _to_float(entry.get("jacobian_spectral_norm"))
    if spectral is None:
        spectral = _to_float(entry.get("fp_jacobian_spectral_norm"))
    if spectral is not None:
        scores.append(_clamp01(1.0 - spectral / 20.0))
    for key in (
        "robustness_long_ctx_passkey_score",
        "robustness_long_ctx_multi_hop_score",
        "robustness_long_ctx_scaling_score",
        "robustness_long_ctx_assoc_score",
    ):
        val = _to_float(entry.get(key))
        if val is not None:
            scores.append(_clamp01(val))
    avg = _average_scores(scores)
    return None if avg is None else avg * BONUS_WEIGHTS["robustness"]


def _compute_reference_delta_bonus(entry: Dict[str, Any]) -> float:
    baseline_ratio = _to_float(entry.get("validation_baseline_ratio"))
    if baseline_ratio is None:
        baseline_ratio = _to_float(entry.get("baseline_loss_ratio"))
    if baseline_ratio is not None and baseline_ratio < 0.90:
        gain = _clamp01((1.0 - baseline_ratio) / 0.2)
        return gain * BONUS_WEIGHTS["referenceDelta"]
    return 0.0


def _compute_routing_overhead_penalty(entry: Dict[str, Any]) -> float:
    savings = _to_float(entry.get("routing_savings_ratio"))
    if savings is None or savings >= 0.05:
        return 0.0
    effective_lr = _to_float(entry.get("validation_baseline_ratio"))
    if effective_lr is None:
        effective_lr = _to_float(entry.get("validation_loss_ratio"))
    if effective_lr is None:
        effective_lr = _to_float(entry.get("investigation_loss_ratio"))
    if effective_lr is None:
        effective_lr = _to_float(entry.get("screening_loss_ratio"))
    if effective_lr is None or effective_lr <= 0.95:
        return 0.0
    return -3.0 * (1.0 - savings / 0.05)


def _compute_binding_bonus(entry: Dict[str, Any]) -> Optional[float]:
    ar = _to_float(entry.get("ar_legacy_auc"))
    induction = _to_float(entry.get("induction_screening_auc"))
    binding = _to_float(entry.get("binding_screening_auc"))
    if ar is None and induction is None and binding is None:
        return None
    bc = 0.0
    if ar is not None:
        bc += 0.4 * ar
    if induction is not None:
        bc += 0.3 * induction
    if binding is not None:
        bc += 0.3 * binding
    if bc <= 0:
        return 0.0
    return _scurve(bc / 0.15, 6.0) * BONUS_WEIGHTS["binding"]


def _compute_blimp_bonus(entry: Dict[str, Any]) -> float:
    acc = _to_float(entry.get("blimp_overall_accuracy"))
    if acc is None or acc <= 0.50:
        return 0.0
    return _scurve(acc / 0.60, 6.0) * BONUS_WEIGHTS["blimp"]


def compute_bonus_breakdown(entry: Dict[str, Any]) -> Dict[str, float]:
    raw = {
        "efficiencyBonus": _compute_efficiency_bonus(entry) or 0.0,
        "routingBonus": _compute_routing_bonus(entry) or 0.0,
        "adaptiveBonus": _compute_adaptive_bonus(entry) or 0.0,
        "sparsityBonus": _compute_sparsity_bonus(entry) or 0.0,
        "learningSpeedBonus": _compute_learning_speed_bonus(entry) or 0.0,
        "externalComparisonBonus": _compute_external_comparison_bonus(entry) or 0.0,
        "robustnessBonus": _compute_robustness_bonus(entry) or 0.0,
        "referenceDeltaBonus": _compute_reference_delta_bonus(entry),
        "bindingBonus": _compute_binding_bonus(entry) or 0.0,
        "blimpBonus": _compute_blimp_bonus(entry),
        "routingOverheadPenalty": _compute_routing_overhead_penalty(entry),
    }
    total_raw = sum(raw.values())
    if total_raw > MAX_TOTAL_BONUS and total_raw > 0:
        scale = MAX_TOTAL_BONUS / total_raw
        raw = {key: value * scale for key, value in raw.items()}
    return raw


def discovery_score_breakdown(program: Dict[str, Any]) -> Dict[str, Any]:
    loss = _normalize_loss_ratio(program.get("loss_ratio")) * 30.0
    novelty = (_clamp01(_to_float(program.get("novelty_score")) or 0.0)) * 20.0
    baseline_ratio = _to_float(program.get("baseline_loss_ratio"))
    baseline = (
        _clamp01(1.5 - baseline_ratio) * 25.0 if baseline_ratio is not None else 0.0
    )
    identity = 5.0 if program.get("most_similar_to") else 0.0
    params = (
        program.get("param_count")
        if program.get("param_count") is not None
        else program.get("graph_n_params_estimate")
    )
    param_efficiency = _normalize_inverse_log10(params, 4.0, 9.0)
    param_eff = (param_efficiency or 0.0) * 10.0
    learning_speed = (
        _clamp01(_to_float(program.get("loss_improvement_rate")) or 0.0)
    ) * 10.0
    bonus = compute_bonus_breakdown(program)
    total = (
        loss
        + novelty
        + baseline
        + identity
        + param_eff
        + learning_speed
        + sum(bonus.values())
    )

    scaling_eff = _to_float(program.get("scaling_param_efficiency"))
    if scaling_eff is not None and scaling_eff < 1.0:
        total *= max(0.1, scaling_eff)
    scaling_gate = program.get("scaling_gate_passed")
    if scaling_eff is not None and scaling_gate == 0:
        total *= max(0.3, _clamp01(scaling_eff / 3.0))

    return {
        "total": _round_score(total),
        "loss": round(loss / 30.0 * 100.0),
        "novelty": round(novelty / 20.0 * 100.0),
        "baseline": round(baseline / 25.0 * 100.0),
        "id": round(identity / 5.0 * 100.0),
        "paramEfficiency": round(param_eff / 10.0 * 100.0),
        "learningSpeed": round(learning_speed / 10.0 * 100.0),
        **bonus,
    }


def discovery_score(program: Dict[str, Any]) -> int:
    return int(discovery_score_breakdown(program)["total"])


def attach_discovery_score_payload(programs: Iterable[Dict[str, Any]]) -> None:
    """Attach canonical discovery score fields in-place for dashboard/report rows.

    The dashboard previously recomputed a parallel discovery score in JS. This
    helper keeps the backend as the single scoring authority by annotating the
    payload rows directly.
    """
    for program in programs or ():
        if not isinstance(program, dict):
            continue
        breakdown = discovery_score_breakdown(program)
        program["discovery_score"] = int(breakdown.get("total") or 0)
        program["discovery_score_breakdown"] = breakdown
