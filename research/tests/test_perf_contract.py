from __future__ import annotations

from research.perf_contract import (
    build_duplicate_work_report,
    build_perf_contract,
    emit_perf_artifact,
    list_recent_perf_artifacts,
    summarize_perf_artifacts,
)


def test_perf_contract_artifact_roundtrip(tmp_path):
    duplicate_work = build_duplicate_work_report(
        repeated_keys={"graph_compile": 2},
        avoided_keys={"workflow_to_graph": 1},
        wasted_ms=12.5,
        hints=["reuse graph conversion"],
    )
    contract = build_perf_contract(
        component="research",
        workload="experiment_screening",
        identity={"experiment_id": "exp_123"},
        metrics={"total_time_ms": 123.4, "compile_time_ms": 45.6},
        budget_profile="research_default",
        budget_verdict={
            "passed": False,
            "checks": [{"metric": "duplicate_work.detected_count"}],
        },
        duplicate_work=duplicate_work,
    )

    artifact_path = emit_perf_artifact(contract, root=str(tmp_path), slug="exp_123")
    artifacts = list_recent_perf_artifacts(
        root=str(tmp_path), component="research", limit=5
    )
    summary = summarize_perf_artifacts(artifacts, component="research")

    assert artifact_path.endswith("exp_123.json")
    assert artifacts[0]["artifact_path"] == artifact_path
    assert artifacts[0]["duplicate_work"]["detected_count"] == 2
    assert summary["count"] == 1
    assert summary["failed_budget_runs"] == 1
