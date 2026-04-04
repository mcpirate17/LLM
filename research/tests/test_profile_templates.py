import json
import sqlite3

from research.tools.profile_templates import build_template_profiles


def _insert_row(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    template_name: str,
    stage0: int,
    stage05: int,
    stage1: int,
    loss_ratio: float | None,
    validation_loss_ratio: float | None,
    error_type: str | None = None,
    slot_usage: list[dict] | None = None,
) -> None:
    graph_json = json.dumps(
        {
            "metadata": {
                "templates_used": [template_name],
                "template_slot_usage": slot_usage or [],
            }
        }
    )
    conn.execute(
        """
        INSERT INTO program_results(
            result_id, timestamp, graph_json,
            stage0_passed, stage05_passed, stage1_passed,
            loss_ratio, validation_loss_ratio, discovery_loss_ratio,
            error_type, stage_at_death, failure_details_json
        ) VALUES (?, 0, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
        """,
        (
            result_id,
            graph_json,
            stage0,
            stage05,
            stage1,
            loss_ratio,
            validation_loss_ratio,
            error_type,
        ),
    )


def test_build_template_profiles_classifies_and_aggregates(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT,
            timestamp REAL,
            graph_json TEXT,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            loss_ratio REAL,
            validation_loss_ratio REAL,
            discovery_loss_ratio REAL,
            error_type TEXT,
            stage_at_death TEXT,
            failure_details_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE template_stats (
            template_name TEXT,
            eval_count INTEGER,
            s0_pass_count INTEGER,
            s1_pass_count INTEGER
        )
        """
    )
    _insert_row(
        conn,
        result_id="q1",
        template_name="attn_spectral_filter",
        stage0=1,
        stage05=1,
        stage1=0,
        loss_ratio=0.9,
        validation_loss_ratio=0.92,
        error_type="insufficient_learning",
    )
    _insert_row(
        conn,
        result_id="q2",
        template_name="attn_spectral_filter",
        stage0=1,
        stage05=1,
        stage1=0,
        loss_ratio=0.88,
        validation_loss_ratio=0.91,
        error_type="failed_convergence",
    )
    _insert_row(
        conn,
        result_id="q3",
        template_name="attn_spectral_filter",
        stage0=1,
        stage05=1,
        stage1=0,
        loss_ratio=0.86,
        validation_loss_ratio=0.89,
        error_type="insufficient_learning",
    )
    _insert_row(
        conn,
        result_id="p1",
        template_name="mamba_reference",
        stage0=1,
        stage05=1,
        stage1=1,
        loss_ratio=0.31,
        validation_loss_ratio=0.29,
        slot_usage=[],
    )
    _insert_row(
        conn,
        result_id="p2",
        template_name="mamba_reference",
        stage0=1,
        stage05=1,
        stage1=1,
        loss_ratio=0.28,
        validation_loss_ratio=0.27,
        slot_usage=[],
    )
    _insert_row(
        conn,
        result_id="p3",
        template_name="mamba_reference",
        stage0=1,
        stage05=1,
        stage1=0,
        loss_ratio=0.4,
        validation_loss_ratio=0.38,
        error_type="failed_convergence",
        slot_usage=[],
    )
    conn.commit()
    conn.close()

    profiles = build_template_profiles(
        db_path,
        templates={"attn_spectral_filter", "mamba_reference", "topk_retrieval"},
    )
    by_name = {item["name"]: item for item in profiles}

    assert by_name["attn_spectral_filter"]["status"] == "quarantine"
    assert (
        by_name["attn_spectral_filter"]["top_failure_reason"] == "insufficient_learning"
    )
    assert by_name["mamba_reference"]["status"] == "promote"
    assert by_name["mamba_reference"]["slot_count"] == 0
    assert by_name["topk_retrieval"]["status"] == "needs_data"
