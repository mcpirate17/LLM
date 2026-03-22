from __future__ import annotations

import argparse
import json

from research.perf_contract import list_recent_perf_artifacts, summarize_perf_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize recent Aria performance artifacts"
    )
    parser.add_argument(
        "--component", choices=["research", "aria_designer"], default=None
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    artifacts = list_recent_perf_artifacts(
        component=args.component, limit=max(1, args.limit)
    )
    payload = {
        "summary": summarize_perf_artifacts(artifacts, component=args.component),
        "artifacts": artifacts,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    summary = payload["summary"]
    print(
        f"component={summary.get('component') or 'all'} "
        f"count={summary.get('count', 0)} "
        f"failed_budget_runs={summary.get('failed_budget_runs', 0)} "
        f"duplicate_work_events={summary.get('duplicate_work_events', 0)} "
        f"avg_total_time_ms={summary.get('avg_total_time_ms')}"
    )
    for item in artifacts:
        verdict = item.get("budget_verdict") or {}
        print(
            f"{item.get('generated_at')} "
            f"{item.get('component')}/{item.get('workload')} "
            f"total_time_ms={item.get('total_time_ms')} "
            f"budget_passed={verdict.get('passed')} "
            f"path={item.get('artifact_path')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
