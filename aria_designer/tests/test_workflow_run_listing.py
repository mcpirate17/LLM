from __future__ import annotations

from pathlib import Path

from aria_designer.api.app import database as db


def test_list_workflow_runs_does_not_call_get_workflow_run(monkeypatch, tmp_path):
    db_path = Path(tmp_path) / "designer_runs.db"
    db.init_db(db_path)

    db.save_workflow_run(
        workflow_id="wf_a",
        run_id="run_a",
        status="ok",
        results={"metric": 1},
        perf={"latency_ms": 3.5},
        stages={"compile": {"status": "done"}},
        error=None,
        semantic_warnings=[{"code": "approximate"}],
        started_at="2026-04-20T12:00:00Z",
        updated_at="2026-04-20T12:00:01Z",
    )
    db.save_workflow_run(
        workflow_id="wf_b",
        run_id="run_b",
        status="failed",
        results={"metric": 2},
        perf=None,
        stages={"sandbox": {"status": "failed"}},
        error={"stage": "sandbox"},
        semantic_warnings=[],
        started_at="2026-04-20T12:00:02Z",
        updated_at="2026-04-20T12:00:03Z",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("list_workflow_runs should hydrate rows directly")

    monkeypatch.setattr(db, "get_workflow_run", _fail)

    runs = db.list_workflow_runs(limit=10)

    assert [run["run_id"] for run in runs] == ["run_b", "run_a"]
    assert runs[0]["stages"]["sandbox"]["status"] == "failed"
    assert runs[1]["perf_contract"]["latency_ms"] == 3.5
