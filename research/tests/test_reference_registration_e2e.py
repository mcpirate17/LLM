import json

from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook


def test_pinned_reference_visible_in_tier_filtered_endpoints(tmp_path):
    db_path = str(tmp_path / "reference_e2e.db")
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        experiment_type="reference_registration",
        config={"source": "test"},
        hypothesis="register reference",
        require_preregistration=False,
    )

    ref_result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="ref_gpt2_fp",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.42,
        novelty_score=0.01,
        model_source="reference",
    )
    assert ref_result_id

    ref_entry_id = nb.upsert_leaderboard(
        result_id=ref_result_id,
        model_source="reference",
        architecture_desc="GPT-2 Small",
        screening_loss_ratio=0.42,
        screening_novelty=0.01,
        screening_passed=True,
        tier="screening",
        is_reference=True,
        reference_name="GPT-2 Small",
    )
    nb.pin_reference(ref_entry_id, "GPT-2 Small")

    candidate_result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="candidate_fp",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.37,
        novelty_score=0.6,
        model_source="graph_synthesis",
    )
    assert candidate_result_id
    nb.upsert_leaderboard(
        result_id=candidate_result_id,
        model_source="graph_synthesis",
        architecture_desc="Candidate",
        screening_loss_ratio=0.37,
        screening_novelty=0.6,
        screening_passed=True,
        tier="screening",
    )

    refs = nb.get_references()
    assert any(r.get("entry_id") == ref_entry_id for r in refs)
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    leaderboard = client.get("/api/leaderboard?tier=validation&limit=50")
    assert leaderboard.status_code == 200
    lb_entries = leaderboard.get_json().get("entries", [])
    assert any(e.get("entry_id") == ref_entry_id for e in lb_entries)
    assert not any(
        (e.get("result_id") == candidate_result_id and not e.get("is_reference"))
        for e in lb_entries
    )

    discoveries = client.get("/api/discoveries?view=ranked&tier=validation&limit=50")
    assert discoveries.status_code == 200
    disc_entries = discoveries.get_json().get("entries", [])
    assert any(e.get("entry_id") == ref_entry_id for e in disc_entries)
