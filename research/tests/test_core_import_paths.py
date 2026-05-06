import importlib
from pathlib import Path


def _read(relpath: str) -> str:
    research_root = Path(__file__).resolve().parents[1]
    return (research_root / relpath).read_text()


def test_core_runner_modules_import_without_internal_fallback_binding():
    importlib.import_module("research.scientist.runner.screening_candidate_rank")
    importlib.import_module("research.scientist.runner.execution_screening_pipeline")
    importlib.import_module("research.scientist.api_routes.designer_bp")
    importlib.import_module("research.scientist.api_routes.programs_bp")
    importlib.import_module("research.scientist.api_routes._strategy_preflight")


def test_core_runner_modules_do_not_mask_internal_import_errors():
    screening_rank_src = _read("scientist/runner/screening_candidate_rank.py")
    screening_pipeline_src = _read("scientist/runner/execution_screening_pipeline.py")
    designer_bp_src = _read("scientist/api_routes/designer_bp.py")
    programs_bp_src = _read("scientist/api_routes/programs_bp.py")
    strategy_preflight_src = _read("scientist/api_routes/_strategy_preflight.py")

    assert "except ImportError" not in screening_rank_src
    assert "except ImportError" not in screening_pipeline_src
    assert (
        '("runtime.importer", "aria_designer.runtime.importer")' not in designer_bp_src
    )
    assert "sys.path.insert" not in designer_bp_src
    assert "from api.app import database" not in designer_bp_src
    assert "aria_designer.runtime.importer" not in programs_bp_src
    assert "_graph_to_workflow = None" not in programs_bp_src
    assert "except ImportError as exc" not in strategy_preflight_src


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
