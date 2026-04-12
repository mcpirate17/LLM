from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _accumulate_histogram(acc: torch.Tensor | None, hist: torch.Tensor) -> torch.Tensor:
    if acc is None:
        return hist
    if acc.numel() == hist.numel():
        return acc + hist
    if acc.numel() < hist.numel():
        padded = torch.zeros_like(hist)
        padded[: acc.numel()] = acc
        return padded + hist
    padded = torch.zeros_like(acc)
    padded[: hist.numel()] = hist
    return acc + padded


def _init_totals() -> dict[str, Any]:
    return {
        "heatmaps": {},
        "total_savings": 0.0,
        "total_depth_ratio": 0.0,
        "routing_op_count": 0,
        "tokens_total": 0,
        "keep_count": 0,
        "drop_count": 0,
        "default_path_count": 0,
        "routed_token_count": 0,
        "sparse_span_count": 0,
        "sparse_span_width_sum": 0.0,
        "sparse_span_width_count": 0,
        "sparse_span_coverage_tokens": 0,
        "lane_histogram": None,
        "confidence_histogram": None,
        "confidence_sum": 0.0,
        "confidence_sq_sum": 0.0,
        "confidence_count": 0,
        "route_strength_sum": 0.0,
        "route_strength_count": 0,
        "branch_weight_sum": None,
        "branch_weight_count": 0,
        "branch_dominance_sum": 0.0,
        "routed_branch_share_sum": 0.0,
        "medium_branch_share_sum": 0.0,
        "hard_branch_share_sum": 0.0,
        "routing_modes": set(),
        "gate_types": set(),
        "span_types": set(),
        "lane_count_max": 0,
        "trace_payloads": {},
    }


def _merge_module_telemetry(
    totals: dict[str, Any],
    module_name: str,
    rt: dict[str, Any],
) -> None:
    if rt.get("heatmap") is not None:
        totals["heatmaps"][module_name] = rt["heatmap"]
    totals["routing_op_count"] += 1
    totals["tokens_total"] += int(rt.get("tokens_total", 0) or 0)
    totals["keep_count"] += int(rt.get("keep_count", 0) or 0)
    totals["drop_count"] += int(rt.get("drop_count", 0) or 0)
    totals["default_path_count"] += int(rt.get("default_path_count", 0) or 0)
    totals["routed_token_count"] += int(rt.get("routed_token_count", 0) or 0)
    totals["sparse_span_count"] += int(rt.get("sparse_span_count", 0) or 0)
    totals["sparse_span_width_sum"] += float(
        rt.get("sparse_span_width_sum", 0.0) or 0.0
    )
    totals["sparse_span_width_count"] += int(rt.get("sparse_span_width_count", 0) or 0)
    totals["sparse_span_coverage_tokens"] += int(
        rt.get("sparse_span_coverage_tokens", 0) or 0
    )
    totals["confidence_sum"] += float(rt.get("confidence_sum", 0.0) or 0.0)
    totals["confidence_sq_sum"] += float(rt.get("confidence_sq_sum", 0.0) or 0.0)
    totals["confidence_count"] += int(rt.get("confidence_count", 0) or 0)
    totals["route_strength_sum"] += float(rt.get("route_strength_sum", 0.0) or 0.0)
    totals["route_strength_count"] += int(rt.get("route_strength_count", 0) or 0)
    totals["branch_dominance_sum"] += float(rt.get("branch_dominance_sum", 0.0) or 0.0)
    totals["routed_branch_share_sum"] += float(
        rt.get("routed_branch_share_sum", 0.0) or 0.0
    )
    totals["medium_branch_share_sum"] += float(
        rt.get("medium_branch_share_sum", 0.0) or 0.0
    )
    totals["hard_branch_share_sum"] += float(
        rt.get("hard_branch_share_sum", 0.0) or 0.0
    )
    totals["branch_weight_count"] += int(rt.get("branch_weight_count", 0) or 0)
    totals["lane_count_max"] = max(
        totals["lane_count_max"], int(rt.get("lane_count", 0) or 0)
    )
    totals["total_savings"] += float(rt.get("savings_ratio", 0.0) or 0.0)
    totals["total_depth_ratio"] += float(rt.get("depth_ratio", 1.0) or 1.0)

    if rt.get("routing_mode"):
        totals["routing_modes"].add(str(rt["routing_mode"]))
    if rt.get("gate_type"):
        totals["gate_types"].add(str(rt["gate_type"]))
    if rt.get("span_type"):
        totals["span_types"].add(str(rt["span_type"]))
    if rt.get("trace_payload") is not None:
        totals["trace_payloads"][module_name] = rt["trace_payload"]

    if isinstance(rt.get("lane_histogram"), torch.Tensor):
        hist = rt["lane_histogram"].detach().to(torch.float32).cpu()
        totals["lane_histogram"] = _accumulate_histogram(totals["lane_histogram"], hist)
    if isinstance(rt.get("confidence_histogram"), torch.Tensor):
        hist = rt["confidence_histogram"].detach().to(torch.float32).cpu()
        totals["confidence_histogram"] = _accumulate_histogram(
            totals["confidence_histogram"],
            hist,
        )
    if isinstance(rt.get("branch_weight_sum"), torch.Tensor):
        hist = rt["branch_weight_sum"].detach().to(torch.float32).cpu()
        totals["branch_weight_sum"] = _accumulate_histogram(
            totals["branch_weight_sum"], hist
        )


def _add_token_payload(payload: dict[str, Any], totals: dict[str, Any]) -> None:
    tokens_total = totals["tokens_total"]
    if tokens_total <= 0:
        return
    payload["routing_keep_drop_ratio"] = {
        "keep": round(totals["keep_count"] / tokens_total, 4),
        "drop": round(totals["drop_count"] / tokens_total, 4),
    }
    payload["default_path_fraction"] = round(
        totals["default_path_count"] / tokens_total, 4
    )
    payload["routed_compute_fraction"] = round(
        totals["routed_token_count"] / tokens_total, 4
    )
    payload["sparse_span_coverage"] = round(
        totals["sparse_span_coverage_tokens"] / tokens_total,
        4,
    )


def _add_span_payload(payload: dict[str, Any], totals: dict[str, Any]) -> None:
    if totals["sparse_span_width_count"] <= 0:
        return
    payload["sparse_span_count"] = int(totals["sparse_span_count"])
    payload["average_span_width"] = round(
        totals["sparse_span_width_sum"] / totals["sparse_span_width_count"],
        4,
    )


def _add_lane_payload(payload: dict[str, Any], totals: dict[str, Any]) -> None:
    lane_histogram = totals["lane_histogram"]
    if lane_histogram is None:
        return
    lane_probs = lane_histogram / lane_histogram.sum().clamp(min=1.0)
    lane_entropy = float(
        -(lane_probs * torch.log(lane_probs.clamp(min=1e-10))).sum().item()
    )
    payload["lane_utilization_histogram"] = lane_histogram.int().tolist()
    payload["lane_entropy"] = round(lane_entropy, 4)
    payload["lane_utilization"] = payload["lane_utilization_histogram"]
    payload["active_lane_count"] = int((lane_histogram > 0).sum().item())
    payload["dead_lane_count"] = int((lane_histogram == 0).sum().item())


def _add_confidence_payload(payload: dict[str, Any], totals: dict[str, Any]) -> None:
    if totals["confidence_count"] > 0:
        conf_mean = totals["confidence_sum"] / totals["confidence_count"]
        conf_var = max(
            0.0,
            (totals["confidence_sq_sum"] / totals["confidence_count"])
            - (conf_mean * conf_mean),
        )
        payload["route_confidence_mean"] = round(conf_mean, 4)
        payload["route_confidence_std"] = round(conf_var**0.5, 4)
    if totals["confidence_histogram"] is not None:
        payload["confidence_histogram"] = totals["confidence_histogram"].int().tolist()


def _add_branch_payload(payload: dict[str, Any], totals: dict[str, Any]) -> None:
    if totals["route_strength_count"] > 0:
        payload["route_strength_mean"] = round(
            totals["route_strength_sum"] / totals["route_strength_count"],
            4,
        )
    branch_weight_sum = totals["branch_weight_sum"]
    if branch_weight_sum is None or totals["branch_weight_count"] <= 0:
        return
    branch_means = (branch_weight_sum / totals["branch_weight_count"]).tolist()
    payload["branch_weight_mean"] = [round(float(v), 4) for v in branch_means]
    payload["branch_dominance_mean"] = round(
        totals["branch_dominance_sum"] / totals["branch_weight_count"],
        4,
    )
    payload["routed_branch_share"] = round(
        totals["routed_branch_share_sum"] / totals["branch_weight_count"],
        4,
    )
    payload["medium_branch_share"] = round(
        totals["medium_branch_share_sum"] / totals["branch_weight_count"],
        4,
    )
    payload["hard_branch_share"] = round(
        totals["hard_branch_share_sum"] / totals["branch_weight_count"],
        4,
    )


def _finalize_payload(
    totals: dict[str, Any],
    capture_heatmaps: bool,
) -> dict[str, Any] | None:
    routing_op_count = totals["routing_op_count"]
    if routing_op_count <= 0:
        return None
    payload: dict[str, Any] = {
        "routing_savings_ratio": round(totals["total_savings"] / routing_op_count, 4),
        "routing_depth_ratio": round(totals["total_depth_ratio"] / routing_op_count, 4),
    }
    _add_token_payload(payload, totals)
    _add_span_payload(payload, totals)
    _add_lane_payload(payload, totals)
    _add_confidence_payload(payload, totals)
    _add_branch_payload(payload, totals)
    if totals["routing_modes"]:
        payload["routing_modes"] = sorted(totals["routing_modes"])
    if totals["gate_types"]:
        payload["gate_types"] = sorted(totals["gate_types"])
    if totals["span_types"]:
        payload["span_types"] = sorted(totals["span_types"])
    if totals["lane_count_max"] > 0:
        payload["lane_count"] = totals["lane_count_max"]
    if totals["trace_payloads"]:
        payload["routing_traces"] = totals["trace_payloads"]
    if capture_heatmaps and totals["heatmaps"]:
        payload["routing_heatmaps"] = totals["heatmaps"]
    return payload


def collect_routing_telemetry(
    model: nn.Module,
    capture_heatmaps: bool,
) -> dict[str, Any] | None:
    totals = _init_totals()
    for module_name, module in model.named_modules():
        rt = getattr(module, "routing_telemetry", None)
        if rt:
            _merge_module_telemetry(totals, module_name, rt)
    return _finalize_payload(totals, capture_heatmaps)
