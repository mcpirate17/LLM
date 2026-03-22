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
    entry = payload["entries"][0]
    stability = entry.get("cross_run_stability", {})
    # Stability data is present (keys vary by backend version)
    assert isinstance(stability, dict)
    assert "seen_runs" in stability or "trend" in stability or "rank_delta" in stability


def test_discoveries_endpoint_returns_orphan_reference_separately(tmp_path):
    db_path = str(tmp_path / "discoveries_refs.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="candidate-fp",
        graph_json="{}",
        stage1_passed=True,
        loss_ratio=0.8,
        novelty_score=0.9,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.8,
        screening_novelty=0.9,
        tier="screening",
    )

    nb.conn.commit()
    nb.conn.execute("PRAGMA foreign_keys = OFF")
    nb.conn.execute(
        """
        INSERT INTO leaderboard (
            entry_id, result_id, timestamp, model_source, architecture_desc,
            tier, composite_score, is_reference, reference_name
        ) VALUES (?, ?, strftime('%s','now'), ?, ?, ?, ?, ?, ?)
        """,
        (
            "ref-entry",
            "missing-ref-result",
            "reference",
            "GPT-2-wikitext103",
            "validation",
            101.0,
            1,
            "GPT-2-wikitext103",
        ),
    )
    nb.conn.commit()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/discoveries?sort=composite_score&limit=50&view=ranked")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert all(not row.get("is_reference") for row in payload["entries"])
    assert any(
        row.get("reference_name") == "GPT-2-wikitext103"
        for row in payload.get("references", [])
    )
    assert payload.get("counts", {}).get("references") == 1


def test_program_detail_falls_back_to_reference_leaderboard_row(tmp_path):
    db_path = str(tmp_path / "reference_detail.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    candidate_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="candidate-fp",
        graph_json="{}",
        stage1_passed=True,
        loss_ratio=0.8,
        novelty_score=0.4,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=candidate_rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.8,
        screening_novelty=0.4,
        tier="screening",
    )

    nb.conn.commit()
    nb.conn.execute("PRAGMA foreign_keys = OFF")
    nb.conn.execute(
        """
        INSERT INTO leaderboard (
            entry_id, result_id, timestamp, model_source, architecture_desc,
            tier, composite_score, screening_loss_ratio, is_reference, reference_name
        ) VALUES (?, ?, strftime('%s','now'), ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ref-program-detail",
            "orphan-gpt2-ref",
            "reference",
            "GPT-2-wikitext103",
            "validation",
            100.0,
            0.5669,
            1,
            "GPT-2-wikitext103",
        ),
    )
    nb.conn.commit()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/programs/orphan-gpt2-ref")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert payload["result_id"] == "orphan-gpt2-ref"
    assert payload["is_reference"] is True
    assert payload["reference_name"] == "GPT-2-wikitext103"
    assert payload["architecture_family"] == "Attention"
    assert payload["loss_ratio"] == pytest.approx(0.5669)


def test_discoveries_search_scope_all_hits_non_leaderboard_fingerprint(tmp_path):
    db_path = str(tmp_path / "discoveries_search.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    leaderboard_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-leaderboard",
        graph_json="{}",
        stage1_passed=True,
        loss_ratio=0.7,
        novelty_score=0.8,
    )
    hidden_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-hidden-search-target",
        graph_json="{}",
        stage1_passed=True,
        loss_ratio=0.6,
        novelty_score=0.5,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=leaderboard_rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.7,
        screening_novelty=0.8,
        tier="screening",
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/discoveries?sort=composite_score&limit=50&view=ranked"
        "&q=fp-hidden-search-target&scope=all"
    )

    assert res.status_code == 200
    payload = res.get_json()
    matches = payload.get("entries", [])
    assert any(row.get("result_id") == hidden_rid for row in matches)
