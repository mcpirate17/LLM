import json
import pytest
from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook
from research.synthesis.reference_architectures import build_reference

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_full_nas_pipeline_flow(tmp_path):
    """
    Simulate full lifecycle:
    1. Register Reference Arch
    2. Register 2 Synthetic Candidates
    3. Verify Dashboard List
    4. Pin a candidate
    5. Promote a candidate to Investigation
    6. Verify Tier filtering
    """
    db_path = str(tmp_path / "nas_e2e.db")
    nb = LabNotebook(db_path)

    # 1. Start Experiment
    exp_id = nb.start_experiment(
        experiment_type="e2e_test_campaign",
        config={"dim": 64, "n_layers": 2},
        hypothesis="E2E pipeline should work",
    )

    # 2. Register Reference (GPT-2 Small variant)
    ref_graph = build_reference("gpt2", d_model=64)
    ref_res_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="ref_fp",
        graph_json=json.dumps(ref_graph.to_dict()),
        stage1_passed=True,
        loss_ratio=0.45,
        discovery_loss_ratio=0.45,
        validation_loss_ratio=0.46,
        model_source="reference",
    )
    nb.upsert_leaderboard(
        result_id=ref_res_id,
        model_source="reference",
        architecture_desc="GPT-2 Reference",
        screening_loss_ratio=0.45,
        tier="screening",
        is_reference=True,
        reference_name="GPT-2",
    )

    # 3. Register 2 Candidates
    cand1_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="cand1_fp",
        graph_json=json.dumps({"nodes": []}),
        stage1_passed=True,
        loss_ratio=0.40,
        discovery_loss_ratio=0.40,
        validation_loss_ratio=0.41,
    )
    nb.upsert_leaderboard(
        result_id=cand1_id,
        model_source="graph_synthesis",
        architecture_desc="Candidate 1",
        screening_loss_ratio=0.40,
        tier="screening",
    )

    cand2_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="cand2_fp",
        graph_json=json.dumps({"nodes": []}),
        stage1_passed=True,
        loss_ratio=0.38,
        discovery_loss_ratio=0.38,
        validation_loss_ratio=0.39,
    )
    nb.upsert_leaderboard(
        result_id=cand2_id,
        model_source="graph_synthesis",
        architecture_desc="Candidate 2",
        screening_loss_ratio=0.38,
        tier="screening",
    )

    nb.close()

    # --- API Level Checks ---
    app = create_app(notebook_path=db_path)
    client = app.test_client()

    # Verify initial leaderboard (sorted by screening_loss_ratio)
    resp = client.get("/api/leaderboard?limit=50")
    data = resp.get_json()["entries"]
    # Candidate 2 (0.38) should be first
    assert data[0]["result_id"] == cand2_id

    # 4. Pin Candidate 1
    # Find entry_id for cand1
    entry1 = next(e["entry_id"] for e in data if e["result_id"] == cand1_id)
    client.post("/api/leaderboard/pin", json={"entry_id": entry1, "pinned": True})

    # Verify Candidate 1 is now first
    resp = client.get("/api/leaderboard?limit=50")
    data = resp.get_json()["entries"]
    assert data[0]["result_id"] == cand1_id
    assert data[0]["is_pinned"] == 1

    # 5. Promote Candidate 2 to Investigation
    client.post(
        "/api/leaderboard/status",
        json={
            "entry_id": next(e["entry_id"] for e in data if e["result_id"] == cand2_id),
            "tier": "investigation",
            "investigation_loss_ratio": 0.37,
        },
    )

    # 6. Verify Tier filtering
    # Investigation tier should show Candidate 2 AND the Reference (references are always included)
    resp = client.get("/api/leaderboard?tier=investigation")
    data = resp.get_json()["entries"]
    ids = [e["result_id"] for e in data]
    assert cand2_id in ids
    assert ref_res_id in ids
    assert cand1_id not in ids  # Cand 1 is still in screening

    # 7. Check dual metrics in reproducibility manifest
    resp = client.get(f"/api/reproducibility-manifest/{cand1_id}")
    manifest = resp.get_json()
    assert manifest["outcomes"]["discovery_loss_ratio"] == 0.40
    assert manifest["outcomes"]["validation_loss_ratio"] == 0.41


if __name__ == "__main__":
    pytest.main([__file__])
