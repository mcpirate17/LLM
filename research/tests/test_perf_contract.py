from __future__ import annotations

from research.perf_contract import (
    build_duplicate_work_report,
    build_perf_contract,
    build_perf_contract_with_gate,
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


def test_recent_perf_artifacts_skips_non_contract_json(tmp_path):
    contract = build_perf_contract(
        component="research",
        workload="experiment_screening",
        metrics={"total_time_ms": 123.4},
    )
    artifact_path = emit_perf_artifact(contract, root=str(tmp_path), slug="valid")

    junk_dir = tmp_path / "research" / "2099-01-01"
    junk_dir.mkdir(parents=True, exist_ok=True)
    (junk_dir / "profile_dump.json").write_text('{"metrics": null}\n')

    artifacts = list_recent_perf_artifacts(
        root=str(tmp_path), component="research", limit=5
    )

    assert [item["artifact_path"] for item in artifacts] == [artifact_path]


def test_build_perf_contract_with_gate_designer_flat_metrics():
    """designer_interactive gates on flat ``metrics.*`` keys — no gate_payload
    override should be needed."""
    contract, verdict = build_perf_contract_with_gate(
        component="aria_designer",
        workload="workflow_evaluation",
        metrics={
            "total_time_ms": 50.0,
            "compile_time_ms": 10.0,
            "native_coverage": 0.9,
        },
        budget_profile="designer_interactive",
    )
    reasons = [c.get("reason") for c in verdict["checks"]]
    assert "missing_metric" not in reasons
    assert verdict["passed"] is True
    assert contract["component"] == "aria_designer"
    assert contract["budget_profile"] == "designer_interactive"


def test_build_perf_contract_with_gate_research_nested_payload():
    """research_default gates on nested keys (``trace_avg_ms.*``,
    ``gpu_starvation.*``). Passing ``gate_payload`` lets the nested lookups
    resolve even when the contract's own metrics dict is flat."""
    report = {
        "trace_avg_ms": {"compile": 30.0, "forward_pass": 5.0, "backward_pass": 8.0},
        "queue_telemetry": {"scheduling_wait_avg_ms": 2.0},
        "gpu_starvation": {"max_stall_ms": 1.0},
        "duplicate_work": {"detected_count": 0},
    }
    contract, verdict = build_perf_contract_with_gate(
        component="research",
        workload="experiment_screening",
        metrics={"total_time_ms": 100.0, "compile_time_ms": 30.0},
        budget_profile="research_default",
        gate_payload=report,
    )
    reasons = [c.get("reason") for c in verdict["checks"]]
    assert "missing_metric" not in reasons
    assert verdict["passed"] is True
    assert contract["metrics"]["compile_time_ms"] == 30.0
