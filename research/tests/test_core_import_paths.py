def test_program_backfill_metrics_imports_root_screening_recompute(monkeypatch):
    from flask import Flask

    from research.scientist import screening_recompute
    from research.scientist.api_routes.programs_routes import program_actions

    calls = []

    def fake_recompute_screening_metrics(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "mode": "full_screening_recompute",
            "updates": {"rapid_screening_passed": True},
            "errors": {},
        }

    class FakeNotebook:
        def get_program_detail(self, result_id):
            return {"result_id": result_id, "graph_json": "{}"}

    monkeypatch.setattr(
        screening_recompute,
        "recompute_screening_metrics",
        fake_recompute_screening_metrics,
    )

    app = Flask(__name__)
    with app.test_request_context(
        "/api/programs/result-1/backfill-metrics",
        method="POST",
        json={"device": "cpu"},
    ):
        response = program_actions._api_program_backfill_metrics(
            "/tmp/lab_notebook.db",
            "result-1",
            nb=FakeNotebook(),
        )

    assert response.status_code == 200
    assert response.get_json()["backfill"]["updates"] == {
        "rapid_screening_passed": True
    }
    assert calls[0]["result_id"] == "result-1"
