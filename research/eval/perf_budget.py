"""
Performance budget gates for experiment-level regression checks.

These checks operate on the aggregated perf report emitted by the runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


DEFAULT_PERF_BUDGETS: Dict[str, Dict[str, float]] = {
    # Budget profile for Stage-1 screening runs.
    "screening_default": {
        "trace_avg_ms.compile": 250.0,
        "trace_avg_ms.forward_pass": 35.0,
        "trace_avg_ms.backward_pass": 55.0,
        "trace_avg_ms.optimizer_step": 20.0,
        "queue_telemetry.scheduling_wait_avg_ms": 40.0,
        "queue_telemetry.submit_wait_avg_ms": 10.0,
        "gpu_starvation.max_stall_ms": 30.0,
        "gpu_starvation.total_stall_ms": 200.0,
    }
}


def _nested_get(payload: Dict[str, Any], dotted_key: str) -> Optional[float]:
    node: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def evaluate_perf_budget_gate(
    perf_report: Optional[Dict[str, Any]],
    budget_profile: str = "screening_default",
    budgets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate experiment perf report against explicit budget thresholds."""
    report = perf_report or {}
    active_budgets = dict(DEFAULT_PERF_BUDGETS.get(budget_profile, {}))
    if budgets:
        active_budgets.update(budgets)

    checks = []
    all_passed = True
    for key, limit in active_budgets.items():
        observed = _nested_get(report, key)
        if observed is None:
            checks.append({
                "metric": key,
                "limit": float(limit),
                "observed": None,
                "passed": False,
                "reason": "missing_metric",
            })
            all_passed = False
            continue
        passed = observed <= float(limit)
        checks.append({
            "metric": key,
            "limit": float(limit),
            "observed": float(observed),
            "passed": passed,
        })
        if not passed:
            all_passed = False

    return {
        "passed": all_passed,
        "profile": budget_profile,
        "checks": checks,
        "n_failed": sum(1 for c in checks if not c["passed"]),
    }
