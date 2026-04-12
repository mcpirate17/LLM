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
        model_source="graph_synthesis",
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
        model_source="graph_synthesis",
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
            tier, composite_score, is_reference, reference_name,
            result_cohort, trust_label, comparability_label,
            evaluation_protocol_version, scoring_version
        ) VALUES (?, ?, strftime('%s','now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            "reference",
            "reference",
            "reference_comparable",
            "reference_v1",
            "v8",
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
        "&q=fp-hidden-search-target&scope=all&trusted_only=0"
    )

    assert res.status_code == 200
    payload = res.get_json()
    matches = payload.get("entries", [])
    assert any(row.get("result_id") == hidden_rid for row in matches)


def test_discoveries_and_leaderboard_default_to_trusted_slice(tmp_path):
    db_path = str(tmp_path / "discoveries_trusted_default.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    trusted_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-trusted",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.5,
        novelty_score=0.8,
    )
    untrusted_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-untrusted",
        graph_json="{}",
        model_source="backfill",
        stage1_passed=True,
        loss_ratio=0.4,
        novelty_score=0.9,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=trusted_rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.5,
        screening_novelty=0.8,
        tier="screening",
    )
    nb.upsert_leaderboard(
        result_id=untrusted_rid,
        model_source="backfill",
        screening_loss_ratio=0.4,
        screening_novelty=0.9,
        tier="screening",
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    lb_default = client.get("/api/leaderboard?limit=50")
    assert lb_default.status_code == 200
    lb_default_payload = lb_default.get_json()
    assert lb_default_payload["trusted_only"] is True
    assert {row.get("result_id") for row in lb_default_payload.get("entries", [])} == {
        trusted_rid
    }

    lb_full = client.get("/api/leaderboard?limit=50&trusted_only=0")
    assert lb_full.status_code == 200
    lb_full_payload = lb_full.get_json()
    assert lb_full_payload["trusted_only"] is False
    assert {row.get("result_id") for row in lb_full_payload.get("entries", [])} == {
        trusted_rid,
        untrusted_rid,
    }

    discoveries_default = client.get("/api/discoveries?limit=50&view=ranked")
    assert discoveries_default.status_code == 200
    discoveries_default_payload = discoveries_default.get_json()
    assert discoveries_default_payload["trusted_only"] is True
    assert {
        row.get("result_id") for row in discoveries_default_payload.get("entries", [])
    } == {trusted_rid}

    discoveries_full = client.get(
        "/api/discoveries?limit=50&view=ranked&trusted_only=0"
    )
    assert discoveries_full.status_code == 200
    discoveries_full_payload = discoveries_full.get_json()
    assert discoveries_full_payload["trusted_only"] is False
    assert {
        row.get("result_id") for row in discoveries_full_payload.get("entries", [])
    } == {trusted_rid, untrusted_rid}


def test_discoveries_search_filters_trusted_slice_before_limit(tmp_path):
    db_path = str(tmp_path / "discoveries_search_trusted_limit.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    trusted_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-search-target-trusted",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.7,
        novelty_score=0.6,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=trusted_rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.7,
        screening_novelty=0.6,
        tier="screening",
    )

    for idx in range(6):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"fp-search-target-untrusted-{idx}",
            graph_json="{}",
            model_source="backfill",
            stage1_passed=True,
            loss_ratio=0.1 + idx * 0.01,
            novelty_score=0.95,
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="backfill",
            screening_loss_ratio=0.1 + idx * 0.01,
            screening_novelty=0.95,
            tier="screening",
        )

    nb.flush_writes()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/discoveries?view=ranked&scope=all&limit=1&q=fp-search-target"
    )

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["trusted_only"] is True
    assert [row.get("result_id") for row in payload.get("entries", [])] == [trusted_rid]


def test_designer_rows_stay_exploratory_even_if_stage1_passed(tmp_path):
    db_path = str(tmp_path / "designer_exploratory.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("designer", {})

    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-designer",
        graph_json="{}",
        model_source="designer_edit",
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.4,
        novelty_score=0.7,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="designer_edit",
        screening_loss_ratio=0.4,
        screening_novelty=0.7,
        tier="screening",
    )
    row = nb.get_program_detail(rid)
    assert row is not None
    assert row["trust_label"] == "exploratory"
    assert row["comparability_label"] == "noncomparable"
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    default_lb = client.get("/api/leaderboard?limit=10")
    assert default_lb.status_code == 200
    assert [
        row.get("result_id") for row in default_lb.get_json().get("entries", [])
    ] == []

    full_lb = client.get("/api/leaderboard?limit=10&trusted_only=0")
    assert full_lb.status_code == 200
    assert [row.get("result_id") for row in full_lb.get_json().get("entries", [])] == [
        rid
    ]
