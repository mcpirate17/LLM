import pytest

from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.api


def test_discoveries_endpoint_accepts_fingerprint_for_cross_run_stability(tmp_path):
    db_path = str(tmp_path / "discoveries.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-discovery",
        graph_json="{}",
        stage1_passed=True,
        loss_ratio=0.9,
        novelty_score=0.7,
    )
    nb.flush_writes()
    nb.complete_experiment(exp_id, results={"status": "ok"})
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.9,
        screening_novelty=0.7,
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/discoveries?sort=composite_score&limit=50&view=ranked")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert payload["entries"]
    stability = payload["entries"][0]["cross_run_stability"]
    assert "summary" in stability
    assert "candidates" in stability
    assert "window_size" in stability
