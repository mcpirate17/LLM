import pytest

from research.scientist.api import create_app
from research.scientist.leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    get_scoring_version,
    prefetch_program_results,
)
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


def test_discoveries_all_graphs_marks_non_leaderboard_failures_as_screened_out(
    tmp_path,
):
    db_path = str(tmp_path / "discoveries_all_graphs.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    failed_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-failed-hidden",
        graph_json="{}",
        bypass_quality_gate=True,
        stage0_passed=True,
        stage1_passed=False,
        hellaswag_acc=0.31,
        trust_label="runtime_observation",
    )
    passed_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-passed-hidden",
        graph_json="{}",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.8,
        trust_label="candidate_screening",
    )
    nb.flush_writes()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/discoveries?view=all_graphs&limit=50&trusted_only=0")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    entries = {entry["result_id"]: entry for entry in payload["entries"]}
    assert entries[failed_rid]["tier"] == "screened_out"
    assert entries[passed_rid]["tier"] == "screening"


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


def test_discoveries_counts_use_current_status_not_stage_pass_history(tmp_path):
    db_path = str(tmp_path / "discoveries_status_counts.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    def make_row(
        fp: str,
        *,
        tier: str,
        investigation_passed=None,
        validation_passed=None,
    ):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json="{}",
            model_source="graph_synthesis",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            tier="screening",
        )
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET tier = ?,
                investigation_passed = COALESCE(?, investigation_passed),
                investigation_loss_ratio = COALESCE(?, investigation_loss_ratio),
                investigation_robustness = COALESCE(?, investigation_robustness),
                validation_passed = COALESCE(?, validation_passed),
                validation_loss_ratio = COALESCE(?, validation_loss_ratio)
            WHERE result_id = ?
            """,
            (
                tier,
                investigation_passed,
                0.2 if investigation_passed is not None else None,
                0.8 if investigation_passed is not None else None,
                validation_passed,
                0.3 if validation_passed is not None else None,
                rid,
            ),
        )

    make_row("fp-screening", tier="screening")
    make_row("fp-screened-out", tier="screened_out")
    make_row("fp-investigation", tier="investigation", investigation_passed=1)
    make_row(
        "fp-validation-pending",
        tier="validation",
        investigation_passed=1,
        validation_passed=0,
    )
    make_row(
        "fp-validation-complete",
        tier="validation",
        investigation_passed=1,
        validation_passed=1,
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/discoveries?sort=composite_score&limit=50&view=ranked")
    assert res.status_code == 200
    payload = res.get_json()
    counts = payload["counts"]

    assert counts["screening"] == 1
    assert counts["screened_out"] == 1
    assert counts["investigation"] == 1
    assert counts["validation_pending"] == 1
    assert counts["validation"] == 1


def test_discoveries_tier_filter_uses_current_status_not_stage_history(tmp_path):
    db_path = str(tmp_path / "discoveries_current_tier_filter.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    def make_row(
        fp: str, *, tier: str, investigation_passed=None, validation_passed=None
    ):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json="{}",
            model_source="graph_synthesis",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            tier="screening",
        )
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET tier = ?,
                investigation_passed = ?,
                validation_passed = ?
            WHERE result_id = ?
            """,
            (tier, investigation_passed, validation_passed, rid),
        )
        return rid

    current_investigation_rid = make_row(
        "fp-current-investigation",
        tier="investigation",
        investigation_passed=0,
    )
    make_row(
        "fp-promoted-validation",
        tier="validation",
        investigation_passed=1,
        validation_passed=1,
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/discoveries?sort=composite_score&limit=50&view=ranked&tier=investigation"
    )

    assert res.status_code == 200
    payload = res.get_json()
    result_ids = {row.get("result_id") for row in payload["entries"]}
    assert current_investigation_rid in result_ids
    assert len(result_ids) == 1


def test_discoveries_validation_filter_excludes_validation_pending_rows(tmp_path):
    db_path = str(tmp_path / "discoveries_validation_filter.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    def make_row(fp: str, *, validation_passed: int):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json="{}",
            model_source="graph_synthesis",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            tier="screening",
        )
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET tier = 'validation',
                investigation_passed = 1,
                validation_passed = ?,
                validation_loss_ratio = 0.3
            WHERE result_id = ?
            """,
            (validation_passed, rid),
        )
        return rid

    validation_pending_rid = make_row("fp-validation-pending-only", validation_passed=0)
    validation_complete_rid = make_row(
        "fp-validation-complete-only", validation_passed=1
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/discoveries?sort=composite_score&limit=50&view=ranked&tier=validation"
    )

    assert res.status_code == 200
    payload = res.get_json()
    result_ids = {row.get("result_id") for row in payload["entries"]}
    assert validation_complete_rid in result_ids
    assert validation_pending_rid not in result_ids


def test_discoveries_expose_capability_quality_separate_from_validation_completion(
    tmp_path,
):
    db_path = str(tmp_path / "discoveries_capability_quality.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    def make_row(fp: str, **kwargs):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json="{}",
            model_source="graph_synthesis",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            tier="screening",
        )
        updates = {
            "tier": "validation",
            "investigation_passed": 1,
            "investigation_loss_ratio": 0.2,
            "investigation_robustness": 0.8,
            "validation_passed": 1,
            "validation_loss_ratio": 0.3,
            **kwargs,
        }
        cols = ", ".join(f"{key} = ?" for key in updates)
        nb.conn.execute(
            f"UPDATE leaderboard SET {cols} WHERE result_id = ?",
            (*updates.values(), rid),
        )
        return rid

    qualified_rid = make_row(
        "fp-qualified",
        validation_baseline_ratio=0.9,
        validation_multi_seed_std=0.05,
        hellaswag_acc=0.45,
        ar_auc=0.12,
        binding_auc=0.12,
        induction_auc=0.12,
    )
    training_only_rid = make_row(
        "fp-training-only",
        validation_baseline_ratio=1.1,
        validation_multi_seed_std=0.05,
        hellaswag_acc=0.22,
        ar_auc=0.01,
        binding_auc=0.02,
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get("/api/discoveries?sort=composite_score&limit=50&view=ranked")
    assert res.status_code == 200
    payload = res.get_json()
    entries = {row["result_id"]: row for row in payload["entries"]}

    assert entries[qualified_rid]["capability_quality"]["status"] == "qualified"
    assert (
        entries[qualified_rid]["capability_quality"]["checks"]["understandingPassed"]
        is True
    )
    assert entries[training_only_rid]["capability_quality"]["status"] == "training_only"
    assert (
        entries[training_only_rid]["capability_quality"]["checks"][
            "understandingPassed"
        ]
        is False
    )


def test_compact_leaderboard_exposes_backend_score_payload(tmp_path):
    db_path = str(tmp_path / "leaderboard_compact_payload.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-compact-payload",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.42,
        novelty_score=0.66,
    )
    nb.flush_writes()
    nb.complete_experiment(exp_id, results={"status": "ok"})
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.42,
        screening_novelty=0.66,
        tier="screening",
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/leaderboard?sort=composite_score&limit=10&compact=1&trusted_only=0"
    )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert payload["entries"]

    entry = payload["entries"][0]
    assert entry["result_id"] == rid
    assert isinstance(entry.get("score_breakdown"), dict)
    assert entry.get("composite_score") is not None
    assert isinstance(entry.get("capability_quality"), dict)
    assert isinstance(entry.get("promotion_evidence"), dict)
    assert "_score" not in entry


def test_backfill_metric_mismatch_warning_is_exposed_in_discoveries_and_compact_leaderboard(
    tmp_path,
):
    db_path = str(tmp_path / "backfill_metric_warning.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("evolution", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-backfill-mismatch",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.04,
        validation_loss_ratio=0.04,
        wikitext_perplexity=812.0,
        hellaswag_acc=0.11,
        result_cohort="backfill",
        trust_label="backfill_observation",
    )
    nb.flush_writes()
    nb.complete_experiment(exp_id, results={"status": "ok"})
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.04,
        screening_novelty=0.33,
        validation_loss_ratio=0.04,
        tier="screening",
        result_cohort="backfill",
        trust_label="backfill_observation",
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    discoveries_res = client.get(
        "/api/discoveries?sort=composite_score&limit=10&view=ranked&trusted_only=0"
    )
    assert discoveries_res.status_code == 200
    discoveries_payload = discoveries_res.get_json()
    assert discoveries_payload is not None
    discoveries_entry = next(
        row for row in discoveries_payload["entries"] if row["result_id"] == rid
    )
    assert discoveries_entry["semantic_warning"]["code"] == "backfill_metric_mismatch"
    assert discoveries_entry["semantic_warning"]["label"] == "Backfill mismatch"
    assert discoveries_entry["semantic_warning_count"] == 1
    assert any(
        "WikiText perplexity" in evidence
        for evidence in discoveries_entry["semantic_warning"]["evidence"]
    )
    assert any(
        "HellaSwag" in evidence
        for evidence in discoveries_entry["semantic_warning"]["evidence"]
    )

    compact_res = client.get(
        "/api/leaderboard?sort=composite_score&limit=10&compact=1&trusted_only=0"
    )
    assert compact_res.status_code == 200
    compact_payload = compact_res.get_json()
    assert compact_payload is not None
    compact_entry = next(
        row for row in compact_payload["entries"] if row["result_id"] == rid
    )
    assert compact_entry["semantic_warning"]["code"] == "backfill_metric_mismatch"
    assert compact_entry["semantic_warning_count"] == 1


def test_report_query_exposes_backend_discovery_score_and_evidence(tmp_path):
    db_path = str(tmp_path / "report_query_backend_scores.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-report-backend-score",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.41,
        novelty_score=0.71,
        baseline_loss_ratio=0.96,
    )
    nb.flush_writes()
    nb.complete_experiment(exp_id, results={"status": "ok"})
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.41,
        screening_novelty=0.71,
        tier="screening",
    )
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/report/query?theme=all&trend=all&limit=10&include_narrative=0&trusted_only=0"
    )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert payload["top_programs"]

    entry = payload["top_programs"][0]
    assert entry["result_id"] == rid
    assert entry.get("discovery_score") is not None
    assert isinstance(entry.get("discovery_score_breakdown"), dict)
    assert isinstance(entry.get("promotion_evidence"), dict)
    assert isinstance(entry.get("decision_gate"), dict)


def test_report_query_handles_duplicate_fingerprint_without_conflicting_insert(
    tmp_path,
):
    db_path = str(tmp_path / "report_query_duplicate_fp.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    canonical_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-report-dup",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.42,
        novelty_score=0.77,
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=canonical_rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.42,
        screening_novelty=0.77,
        tier="screening",
    )
    replay_exp_id = nb.start_experiment("synthesis", {})
    nb.record_program_result(
        experiment_id=replay_exp_id,
        graph_fingerprint="fp-report-dup",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.41,
        novelty_score=0.75,
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
        intentional_rerun_reason="test_fixture_historical_dup",
    )
    nb.flush_writes()
    before_count = nb.conn.execute(
        """
        SELECT COUNT(*)
        FROM leaderboard l
        JOIN program_results pr ON pr.result_id = l.result_id
        WHERE pr.graph_fingerprint = ?
        """,
        ("fp-report-dup",),
    ).fetchone()[0]
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    res = client.get(
        "/api/report/query?theme=all&trend=all&limit=10&include_narrative=0&trusted_only=0"
    )

    assert res.status_code == 200
    payload = res.get_json()
    assert payload is not None
    assert canonical_rid in payload.get("action_eligibility", {})

    nb = LabNotebook(db_path)
    after_count = nb.conn.execute(
        """
        SELECT COUNT(*)
        FROM leaderboard l
        JOIN program_results pr ON pr.result_id = l.result_id
        WHERE pr.graph_fingerprint = ?
        """,
        ("fp-report-dup",),
    ).fetchone()[0]
    nb.close()
    assert before_count == 1
    assert after_count == 1


def test_leaderboard_rescore_api_realigns_payload_with_backend_compute(tmp_path):
    db_path = str(tmp_path / "leaderboard_rescore_api.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-rescore-api",
        graph_json="{}",
        model_source="graph_synthesis",
        stage1_passed=True,
        loss_ratio=0.58,
        novelty_score=0.82,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=rid,
        model_source="graph_synthesis",
        screening_loss_ratio=0.58,
        screening_novelty=0.82,
        tier="screening",
    )

    entry = nb.get_leaderboard_entry(rid)
    assert entry is not None
    pr_cache = prefetch_program_results(nb.conn, [rid])
    expected = float(
        compute_composite(
            **build_score_kwargs_from_prefetch(
                pr_cache[rid],
                dict(entry),
                False,
            )
        )
        or 0.0
    )
    nb.conn.execute(
        """
        UPDATE leaderboard
        SET composite_score = ?, scoring_version = ?
        WHERE result_id = ?
        """,
        (-123.0, "stale-test-version", rid),
    )
    nb.conn.commit()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    rescore = client.post(
        "/api/leaderboard/rescore",
        json={"result_ids": [rid], "only_stale": True},
    )
    assert rescore.status_code == 200
    rescore_payload = rescore.get_json()
    assert rescore_payload["total"] == 1
    assert rescore_payload["changed"] == 1

    leaderboard = client.get(
        "/api/leaderboard?compact=1&trusted_only=0&limit=10&sort=composite_score"
    )
    assert leaderboard.status_code == 200
    payload = leaderboard.get_json()
    rows = {row.get("result_id"): row for row in payload.get("entries", [])}
    assert rid in rows
    assert rows[rid]["composite_score"] == pytest.approx(expected)
    assert isinstance(rows[rid].get("score_breakdown"), dict)


def test_leaderboard_rescore_api_realigns_raw_persisted_rows_with_backend_compute(
    tmp_path,
):
    db_path = str(tmp_path / "leaderboard_rescore_raw_rows.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})

    result_ids = []
    row_specs = [
        ("fp-rescore-screening", 0.58, 0.82, "screening"),
        ("fp-rescore-investigation", 0.47, 0.74, "investigation"),
        ("fp-rescore-validation", 0.39, 0.69, "validation"),
    ]
    for fingerprint, loss_ratio, novelty_score, tier in row_specs:
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fingerprint,
            graph_json="{}",
            model_source="graph_synthesis",
            stage1_passed=True,
            loss_ratio=loss_ratio,
            novelty_score=novelty_score,
        )
        result_ids.append(rid)
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=loss_ratio,
            screening_novelty=novelty_score,
            tier="screening",
        )
        if tier == "investigation":
            nb.conn.execute(
                """
                UPDATE leaderboard
                SET tier = 'investigation',
                    investigation_loss_ratio = ?,
                    investigation_robustness = ?,
                    investigation_passed = 1
                WHERE result_id = ?
                """,
                (loss_ratio * 0.9, 0.81, rid),
            )
        elif tier == "validation":
            nb.conn.execute(
                """
                UPDATE leaderboard
                SET tier = 'validation',
                    investigation_loss_ratio = ?,
                    investigation_robustness = ?,
                    investigation_passed = 1,
                    validation_loss_ratio = ?,
                    validation_baseline_ratio = ?,
                    validation_passed = 1
                WHERE result_id = ?
                """,
                (loss_ratio * 0.92, 0.86, loss_ratio * 0.88, 0.91, rid),
            )

    for index, rid in enumerate(result_ids, start=1):
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET composite_score = ?, scoring_version = ?
            WHERE result_id = ?
            """,
            (-100.0 - index, "stale-test-version", rid),
        )
    nb.conn.commit()
    nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    rescore = client.post(
        "/api/leaderboard/rescore",
        json={"result_ids": result_ids, "only_stale": True},
    )
    assert rescore.status_code == 200
    rescore_payload = rescore.get_json()
    assert rescore_payload["total"] == len(result_ids)
    assert rescore_payload["changed"] == len(result_ids)

    nb = LabNotebook(db_path)
    placeholders = ",".join("?" for _ in result_ids)
    raw_rows = [
        dict(row)
        for row in nb.conn.execute(
            f"""
            SELECT *
            FROM leaderboard
            WHERE result_id IN ({placeholders})
            ORDER BY result_id
            """,
            tuple(result_ids),
        ).fetchall()
    ]
    assert len(raw_rows) == len(result_ids)

    pr_cache = prefetch_program_results(nb.conn, result_ids)
    current_version = get_scoring_version()
    for row in raw_rows:
        rid = str(row["result_id"])
        expected = float(
            compute_composite(
                **build_score_kwargs_from_prefetch(
                    pr_cache[rid],
                    dict(row),
                    bool(row.get("is_reference")),
                )
            )
            or 0.0
        )
        assert float(row["composite_score"]) == pytest.approx(expected)
        assert row.get("scoring_version") == current_version
    nb.close()
