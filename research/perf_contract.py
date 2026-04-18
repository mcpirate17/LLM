from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from research.eval.perf_budget import evaluate_perf_budget_gate
from research.scientist.json_utils import json_safe
from research.scientist.shared_utils import safe_float

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "research" / "perf_artifacts"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def build_duplicate_work_report(
    *,
    repeated_keys: Optional[Dict[str, int]] = None,
    avoided_keys: Optional[Dict[str, int]] = None,
    wasted_ms: float = 0.0,
    hints: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    repeated = {
        str(key): int(count)
        for key, count in (repeated_keys or {}).items()
        if int(count) > 0
    }
    avoided = {
        str(key): int(count)
        for key, count in (avoided_keys or {}).items()
        if int(count) > 0
    }
    return {
        "detected_count": int(sum(repeated.values())),
        "repeated_keys": repeated,
        "avoided_count": int(sum(avoided.values())),
        "avoided_keys": avoided,
        "estimated_wasted_ms": round(max(0.0, safe_float(wasted_ms)), 4),
        "hints": [str(h) for h in (hints or []) if str(h).strip()],
    }


def build_perf_contract(
    *,
    component: str,
    workload: str,
    metrics: Dict[str, Any],
    identity: Optional[Dict[str, Any]] = None,
    budget_profile: Optional[str] = None,
    budget_verdict: Optional[Dict[str, Any]] = None,
    duplicate_work: Optional[Dict[str, Any]] = None,
    warnings: Optional[Iterable[str]] = None,
    artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    payload_metrics = json_safe(metrics or {})
    contract = {
        "version": 1,
        "generated_at": _utc_now_iso(),
        "component": str(component),
        "workload": str(workload),
        "identity": json_safe(identity or {}),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "pid": os.getpid(),
        },
        "metrics": payload_metrics,
        "budget_profile": budget_profile,
        "budget_verdict": json_safe(budget_verdict)
        if budget_verdict is not None
        else None,
        "duplicate_work": json_safe(duplicate_work or build_duplicate_work_report()),
        "warnings": [str(w) for w in (warnings or []) if str(w).strip()],
        "artifact_path": artifact_path,
    }
    return contract


def build_perf_contract_with_gate(
    *,
    component: str,
    workload: str,
    metrics: Dict[str, Any],
    budget_profile: str,
    identity: Optional[Dict[str, Any]] = None,
    duplicate_work: Optional[Dict[str, Any]] = None,
    warnings: Optional[Iterable[str]] = None,
    gate_payload: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the budget gate and build the perf contract in one step.

    Returns ``(contract, budget_verdict)``. Does not write to disk — callers
    that want to persist call :func:`emit_perf_artifact` with the returned
    contract. Single source of truth for the gate+build pattern; used by the
    research runner, the designer API, and the aria_designer profiler.

    Profiles that gate on flat ``metrics.*`` keys (``designer_interactive``)
    need no ``gate_payload``; the default wraps ``metrics``/``duplicate_work``.
    Profiles that gate on nested report keys (``research_default`` reaches
    ``trace_avg_ms.compile``, ``gpu_starvation.max_stall_ms``, etc.) must pass
    an explicit ``gate_payload``.
    """
    dup = duplicate_work or build_duplicate_work_report()
    payload = gate_payload if gate_payload is not None else {
        "metrics": metrics,
        "duplicate_work": dup,
    }
    verdict = evaluate_perf_budget_gate(payload, budget_profile=budget_profile)
    contract = build_perf_contract(
        component=component,
        workload=workload,
        identity=identity or {},
        metrics=metrics,
        budget_profile=budget_profile,
        budget_verdict=verdict,
        duplicate_work=dup,
        warnings=warnings,
    )
    return contract, verdict


def emit_perf_artifact(
    contract: Dict[str, Any],
    *,
    root: Optional[str] = None,
    slug: Optional[str] = None,
) -> str:
    artifact_root = Path(root) if root else _DEFAULT_ARTIFACT_ROOT
    component = str(contract.get("component") or "unknown")
    workload = str(contract.get("workload") or "run")
    ts = int(time.time() * 1000)
    safe_slug = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in (slug or f"{component}_{workload}_{ts}")
    )
    dated_dir = (
        artifact_root / component / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    dated_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = dated_dir / f"{safe_slug}.json"
    payload = dict(contract)
    payload["artifact_path"] = str(artifact_path)
    with artifact_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")
    return str(artifact_path)


def load_perf_artifact(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def list_recent_perf_artifacts(
    *,
    root: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    artifact_root = Path(root) if root else _DEFAULT_ARTIFACT_ROOT
    if not artifact_root.exists():
        return []
    if component:
        comp_roots = [artifact_root / component]
    else:
        comp_roots = [p for p in artifact_root.iterdir() if p.is_dir()]

    n = max(1, int(limit))
    # Date dirs are YYYY-MM-DD — lexical sort == chronological. Walk newest
    # date dirs across components first, stat only files we might return,
    # and stop once we have enough candidates.
    date_dirs: List[Path] = []
    for comp_root in comp_roots:
        if not comp_root.exists():
            continue
        date_dirs.extend(d for d in comp_root.iterdir() if d.is_dir())
    date_dirs.sort(key=lambda d: d.name, reverse=True)

    candidates: List[Path] = []
    for d in date_dirs:
        files = [f for f in d.iterdir() if f.suffix == ".json"]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(files)
        if len(candidates) >= n:
            break

    results: List[Dict[str, Any]] = []
    for path in candidates[:n]:
        try:
            payload = load_perf_artifact(str(path))
        except Exception:
            continue
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        results.append(
            {
                "artifact_path": str(path),
                "component": payload.get("component"),
                "workload": payload.get("workload"),
                "generated_at": payload.get("generated_at"),
                "identity": payload.get("identity", {}),
                "budget_profile": payload.get("budget_profile"),
                "budget_verdict": payload.get("budget_verdict"),
                "duplicate_work": payload.get("duplicate_work"),
                "total_time_ms": safe_float(metrics.get("total_time_ms"), 0.0),
            }
        )
    return results


def summarize_perf_artifacts(
    artifacts: List[Dict[str, Any]],
    *,
    component: Optional[str] = None,
) -> Dict[str, Any]:
    filtered = [
        item
        for item in artifacts
        if not component or item.get("component") == component
    ]
    failures = 0
    warnings = 0
    total_duplicate = 0
    total_time = 0.0
    total_time_count = 0
    latest = filtered[0] if filtered else None
    for item in filtered:
        verdict = item.get("budget_verdict") or {}
        if verdict.get("passed") is False:
            failures += 1
        checks = verdict.get("checks") or []
        warnings += sum(
            1 for check in checks if check.get("reason") == "missing_metric"
        )
        dup = item.get("duplicate_work") or {}
        total_duplicate += int(dup.get("detected_count", 0) or 0)
        total_time_ms = safe_float(item.get("total_time_ms"), -1.0)
        if total_time_ms >= 0.0:
            total_time += total_time_ms
            total_time_count += 1
    return {
        "component": component,
        "count": len(filtered),
        "latest": latest,
        "failed_budget_runs": failures,
        "missing_metric_warnings": warnings,
        "duplicate_work_events": total_duplicate,
        "avg_total_time_ms": round(total_time / total_time_count, 4)
        if total_time_count
        else None,
    }
