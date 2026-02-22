from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional


_TARGET_SPECS: List[Dict[str, Any]] = [
    {
        "id": "param_count",
        "label": "Model Size",
        "category": "size",
        "unit": "params",
        "direction": "lte",
        "target": 2_800_000_000,
        "source": "Mamba 2.8B reference scale",
    },
    {
        "id": "total_flops_per_token",
        "label": "FLOPs / Token",
        "category": "flops",
        "unit": "flops",
        "direction": "lte",
        "target": 6_000_000_000,
        "source": "Internal efficiency budget",
    },
    {
        "id": "forward_ms",
        "label": "Forward Latency",
        "category": "speed",
        "unit": "ms",
        "direction": "lte",
        "target": 50.0,
        "source": "Interactive design-loop latency target",
    },
    {
        "id": "stability_score",
        "label": "Stability",
        "category": "quality",
        "unit": "score",
        "direction": "gte",
        "target": 0.95,
        "source": "Sandbox stability gate",
    },
    {
        "id": "efficiency_score",
        "label": "Efficiency",
        "category": "quality",
        "unit": "score",
        "direction": "gte",
        "target": 0.75,
        "source": "Compression+runtime composite target",
    },
    {
        "id": "overall_novelty",
        "label": "Novelty",
        "category": "quality",
        "unit": "score",
        "direction": "gte",
        "target": 0.70,
        "source": "Research novelty objective",
    },
    {
        "id": "mmlu_5shot",
        "label": "MMLU (5-shot)",
        "category": "task",
        "unit": "percent",
        "direction": "gte",
        "target": 66.6,
        "source": "Llama 3 8B base pretrained model card",
    },
    {
        "id": "humaneval_0shot",
        "label": "HumanEval (0-shot)",
        "category": "task",
        "unit": "percent",
        "direction": "gte",
        "target": 62.2,
        "source": "Llama 3 8B instruct model card",
    },
    {
        "id": "gsm8k_8shot_cot",
        "label": "GSM8K (8-shot CoT)",
        "category": "task",
        "unit": "percent",
        "direction": "gte",
        "target": 79.6,
        "source": "Llama 3 8B instruct model card",
    },
    {
        "id": "arc_challenge_25shot",
        "label": "ARC-Challenge (25-shot)",
        "category": "task",
        "unit": "percent",
        "direction": "gte",
        "target": 78.6,
        "source": "Llama 3 8B base pretrained model card",
    },
    {
        "id": "induction_recall_accuracy",
        "label": "Induction Recall Accuracy",
        "category": "recall",
        "unit": "percent",
        "direction": "gte",
        "target": 99.8,
        "source": "Mamba selective induction/copy synthetic evaluations",
    },
    {
        "id": "throughput_vs_transformer",
        "label": "Throughput vs Transformer",
        "category": "speed",
        "unit": "x",
        "direction": "gte",
        "target": 5.0,
        "source": "Mamba efficiency benchmark (inference throughput)",
    },
]


_REFERENCE_SCALING_CURVE: List[Dict[str, float]] = [
    {"params": 130_000_000, "avg_accuracy": 44.7},
    {"params": 370_000_000, "avg_accuracy": 50.0},
    {"params": 790_000_000, "avg_accuracy": 57.1},
    {"params": 1_400_000_000, "avg_accuracy": 59.7},
    {"params": 2_800_000_000, "avg_accuracy": 63.3},
]


_SOURCES: List[Dict[str, str]] = [
    {
        "name": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "url": "https://arxiv.org/abs/2312.00752",
        "notes": "Scaling laws, zero-shot downstream table, throughput/efficiency claims.",
    },
    {
        "name": "Meta Llama 3 model card",
        "url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B",
        "notes": "8B benchmark baselines (MMLU, GSM8K, HumanEval, ARC-Challenge).",
    },
    {
        "name": "Meta Llama 2 model card",
        "url": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
        "notes": "Open baseline context, pretraining scale and model-size reference points.",
    },
    {
        "name": "OpenAI GPT-2 model card",
        "url": "https://huggingface.co/openai-community/gpt2",
        "notes": "124M historical baseline size/context reference.",
    },
]


def benchmark_target_catalog() -> Dict[str, Any]:
    return {
        "version": "benchmark_targets.v1",
        "sources": list(_SOURCES),
        "targets": list(_TARGET_SPECS),
        "scaling_reference": list(_REFERENCE_SCALING_CURVE),
    }


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_observed(metrics: Dict[str, Any], external_observed: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    profiling = metrics.get("profiling") or {}
    sandbox = metrics.get("sandbox") or {}
    compression = metrics.get("compression") or {}
    novelty = metrics.get("novelty") or {}

    observed = {
        "param_count": _to_float(sandbox.get("param_count") or metrics.get("param_count")),
        "total_flops_per_token": _to_float(
            profiling.get("total_flops_per_token") or metrics.get("total_flops_per_token") or metrics.get("flops_per_token")
        ),
        "forward_ms": _to_float(sandbox.get("forward_ms") or metrics.get("forward_ms")),
        "stability_score": _to_float(sandbox.get("stability_score") or metrics.get("stability_score")),
        "efficiency_score": _to_float(compression.get("efficiency_score") or metrics.get("efficiency_score")),
        "overall_novelty": _to_float(novelty.get("overall_novelty") or metrics.get("overall_novelty")),
    }
    if external_observed:
        for k, v in external_observed.items():
            fv = _to_float(v)
            if fv is not None:
                observed[k] = fv
    return observed


def _score_target(observed: Optional[float], target: float, direction: str) -> Dict[str, Any]:
    if observed is None:
        return {
            "status": "not_measured",
            "gap": None,
            "progress": None,
        }

    if direction == "gte":
        gap = observed - target
        on_target = observed >= target
        progress = min(1.0, max(0.0, observed / target)) if target else 1.0
    else:
        gap = target - observed
        on_target = observed <= target
        progress = min(1.0, max(0.0, target / observed)) if observed and observed > 0 else 1.0

    return {
        "status": "on_target" if on_target else "off_target",
        "gap": float(gap),
        "progress": float(progress),
    }


def _project_mamba_avg_accuracy(param_count: Optional[float]) -> Optional[float]:
    if param_count is None or param_count <= 0:
        return None
    points = _REFERENCE_SCALING_CURVE
    if not points:
        return None

    x = math.log10(param_count)
    logs = [math.log10(p["params"]) for p in points]

    if x <= logs[0]:
        return points[0]["avg_accuracy"]
    if x >= logs[-1]:
        x0, y0 = logs[-2], points[-2]["avg_accuracy"]
        x1, y1 = logs[-1], points[-1]["avg_accuracy"]
    else:
        x0 = y0 = x1 = y1 = None
        for i in range(1, len(points)):
            if logs[i] >= x:
                x0, y0 = logs[i - 1], points[i - 1]["avg_accuracy"]
                x1, y1 = logs[i], points[i]["avg_accuracy"]
                break
        if x0 is None:
            return points[-1]["avg_accuracy"]

    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    return y0 + (y1 - y0) * t


def build_benchmark_analysis(metrics: Dict[str, Any], external_observed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed = _extract_observed(metrics, external_observed)
    rows: List[Dict[str, Any]] = []
    on_target = 0
    off_target = 0
    not_measured = 0

    for spec in _TARGET_SPECS:
        observed_value = observed.get(spec["id"])
        result = _score_target(observed_value, float(spec["target"]), spec["direction"])
        status = result["status"]
        if status == "on_target":
            on_target += 1
        elif status == "off_target":
            off_target += 1
        else:
            not_measured += 1
        rows.append({
            **spec,
            "observed": observed_value,
            "status": status,
            "gap": result["gap"],
            "progress": result["progress"],
        })

    measured_total = on_target + off_target
    score = (on_target / measured_total) if measured_total > 0 else 0.0

    current_params = observed.get("param_count")
    projected_avg = _project_mamba_avg_accuracy(current_params)
    ref_top = _REFERENCE_SCALING_CURVE[-1]
    projected_delta_vs_2p8b = None
    if projected_avg is not None:
        projected_delta_vs_2p8b = projected_avg - ref_top["avg_accuracy"]

    return {
        "version": "benchmark_targets.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(_SOURCES),
        "targets": rows,
        "summary": {
            "on_target": on_target,
            "off_target": off_target,
            "not_measured": not_measured,
            "measured": measured_total,
            "score": round(score, 4),
        },
        "scaling_projection": {
            "current_param_count": current_params,
            "reference_curve": list(_REFERENCE_SCALING_CURVE),
            "projected_mamba_avg_accuracy": projected_avg,
            "delta_vs_mamba_2p8b_avg": projected_delta_vs_2p8b,
            "next_param_milestones": [
                790_000_000,
                1_400_000_000,
                2_800_000_000,
                7_000_000_000,
            ],
        },
    }
