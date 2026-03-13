from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from research.scientist.json_utils import json_safe

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "research" / "perf_artifacts"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    from research.scientist.shared_utils import safe_float
    result = safe_float(value, default)
    return result if result is not None else default



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
        "estimated_wasted_ms": round(max(0.0, _safe_float(wasted_ms)), 4),
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
        "budget_verdict": json_safe(budget_verdict) if budget_verdict is not None else None,
        "duplicate_work": json_safe(duplicate_work or build_duplicate_work_report()),
        "warnings": [str(w) for w in (warnings or []) if str(w).strip()],
        "artifact_path": artifact_path,
    }
    return contract


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
    safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (slug or f"{component}_{workload}_{ts}"))
    dated_dir = artifact_root / component / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = dated_dir / f"{safe_slug}.json"
    payload = dict(contract)
    payload["artifact_path"] = str(artifact_path)
    with artifact_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
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
    search_root = artifact_root / component if component else artifact_root
    if not search_root.exists():
        return []
    items: List[Path] = sorted(search_root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results: List[Dict[str, Any]] = []
    for path in items[: max(1, int(limit))]:
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
                "total_time_ms": _safe_float(metrics.get("total_time_ms"), 0.0),
            }
        )
    return results


def summarize_perf_artifacts(
    artifacts: List[Dict[str, Any]],
    *,
    component: Optional[str] = None,
) -> Dict[str, Any]:
    filtered = [item for item in artifacts if not component or item.get("component") == component]
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
        warnings += sum(1 for check in checks if check.get("reason") == "missing_metric")
        dup = item.get("duplicate_work") or {}
        total_duplicate += int(dup.get("detected_count", 0) or 0)
        total_time_ms = _safe_float(item.get("total_time_ms"), -1.0)
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
        "avg_total_time_ms": round(total_time / total_time_count, 4) if total_time_count else None,
    }
