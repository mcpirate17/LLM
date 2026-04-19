import json
import pytest
from research.scientist.api import create_app
from research.scientist.api_routes.deps import get_notebook
from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.e2e


def test_dashboard_pinning_api_and_sorting(tmp_path):
    db_path = str(tmp_path / "pinning_test.db")
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        experiment_type="pinning_test",
        config={"source": "test"},
        hypothesis="test pinning",
        require_preregistration=False,
    )

    # Create two candidates
    res1 = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp1",
        graph_json=json.dumps({"nodes": []}),
        stage1_passed=True,
        loss_ratio=0.5,
        novelty_score=0.1,
    )
    res2 = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp2",
        graph_json=json.dumps({"nodes": []}),
        stage1_passed=True,
        loss_ratio=0.4,  # Better loss than res1
        novelty_score=0.1,
    )

    entry1 = nb.upsert_leaderboard(
        result_id=res1,
        model_source="test",
        architecture_desc="Candidate 1",
        screening_loss_ratio=0.5,
    )
    nb.upsert_leaderboard(
        result_id=res2,
        model_source="test",
        architecture_desc="Candidate 2",
        screening_loss_ratio=0.4,
    )
    nb.flush_writes()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    # Initial check: res2 (0.4) should be above res1 (0.5) because loss is better
    resp = client.get("/api/leaderboard?limit=50&trusted_only=0")
    entries = resp.get_json().get("entries", [])
    assert entries[0]["result_id"] == res2
    assert entries[1]["result_id"] == res1
    assert not entries[0].get("is_pinned")

    # Pin res1 via API
    pin_resp = client.post(
        "/api/leaderboard/pin", json={"entry_id": entry1, "pinned": True}
    )
    assert pin_resp.status_code == 200
    assert pin_resp.get_json()["pinned"] is True

    # Flush the async write so the next read sees it
    shared_nb = get_notebook(db_path, read_only=False)
    shared_nb.flush_writes()
    shared_nb.close()

    # Check sorting: res1 should now be at the top despite worse loss
    resp = client.get("/api/leaderboard?limit=50&trusted_only=0")
    entries = resp.get_json().get("entries", [])
    assert entries[0]["result_id"] == res1
    assert entries[0]["is_pinned"] == 1
    assert entries[1]["result_id"] == res2

    # Unpin res1
    client.post("/api/leaderboard/pin", json={"entry_id": entry1, "pinned": False})
    shared_nb = get_notebook(db_path, read_only=False)
    shared_nb.flush_writes()
    shared_nb.close()

    resp = client.get("/api/leaderboard?limit=50&trusted_only=0")
    entries = resp.get_json().get("entries", [])
    assert entries[0]["result_id"] == res2
    assert entries[0].get("is_pinned") == 0


if __name__ == "__main__":
    pytest.main([__file__])
