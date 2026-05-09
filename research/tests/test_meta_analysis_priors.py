from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from research.meta_analysis.priors import (
    apply_meta_analysis_prior_to_grammar,
    build_meta_analysis_prior,
    load_latest_meta_analysis_prior,
    write_meta_analysis_prior,
)
from research.scientist.runner._types import RunConfig


def _insert_op_rows(
    conn: sqlite3.Connection,
    *,
    op_name: str,
    category: str,
    induction: float,
    composite: float,
    stage1: int,
    n: int,
    lambda_affinity: float = 0.0,
) -> None:
    for idx in range(n):
        conn.execute(
            """
            INSERT INTO op_observations (
                result_id, op_name, op_category, induction_screening_auc, composite_score,
                stage1_passed, op_lambda_calculus_affinity,
                op_alternative_math_affinity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{op_name}_{idx}",
                op_name,
                category,
                induction,
                composite,
                stage1,
                lambda_affinity,
                lambda_affinity,
            ),
        )


def test_meta_analysis_prior_builds_artifact_and_applies_to_grammar(tmp_path):
    meta_db = tmp_path / "meta_analysis.db"
    conn = sqlite3.connect(meta_db)
    conn.execute(
        """
        CREATE TABLE op_observations (
            result_id TEXT,
            op_name TEXT,
            op_category TEXT,
            induction_screening_auc REAL,
            composite_score REAL,
            stage1_passed INTEGER,
            op_lambda_calculus_affinity REAL,
            op_alternative_math_affinity REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE template_observations (
            result_id TEXT,
            template_name TEXT,
            induction_screening_auc REAL,
            composite_score REAL,
            stage1_passed INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE op_property_catalog (
            op_name TEXT PRIMARY KEY,
            observed_count INTEGER,
            eval_count INTEGER,
            op_category TEXT,
            op_empirical_probe_needed INTEGER
        )
        """
    )
    _insert_op_rows(
        conn,
        op_name="spectral_filter",
        category="frequency",
        induction=0.18,
        composite=42.0,
        stage1=1,
        n=6,
    )
    _insert_op_rows(
        conn,
        op_name="lambda_map",
        category="functional",
        induction=0.15,
        composite=24.0,
        stage1=1,
        n=6,
        lambda_affinity=0.75,
    )
    _insert_op_rows(
        conn,
        op_name="unstable_math_space",
        category="math_space",
        induction=0.00,
        composite=3.0,
        stage1=0,
        n=6,
    )
    for idx in range(5):
        conn.execute(
            """
            INSERT INTO template_observations (
                result_id, template_name, induction_screening_auc, composite_score, stage1_passed
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (f"tpl_{idx}", "spectral_memory_block", 0.16, 35.0, 1),
        )
    conn.executemany(
        """
        INSERT INTO op_property_catalog (
            op_name, observed_count, eval_count, op_category, op_empirical_probe_needed
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("spectral_filter", 6, 3, "frequency", 1),
            ("unstable_math_space", 6, 0, "math_space", 1),
        ],
    )
    conn.commit()
    conn.close()

    prior = build_meta_analysis_prior(
        meta_db_path=meta_db,
        target="balanced",
        min_support=2,
        probe_queue_limit=4,
        created_at=1_800_000_000.0,
    )

    assert prior["target"] == "balanced"
    assert prior["op_weights"]["spectral_filter"] > 1.0
    assert prior["op_weights"]["rope_rotate"] > 1.0
    assert prior["category_weights"]["frequency"] > 1.0
    assert prior["category_weights"]["math_space"] <= 1.05
    assert prior["template_weights"]["spectral_memory_block"] > 1.0
    assert {row["op_name"] for row in prior["probe_queue"]} == {
        "spectral_filter",
        "unstable_math_space",
    }
    assert "lambda_map" in prior["signals"]["high_lambda_affinity_supported_ops"]

    artifact = write_meta_analysis_prior(prior, output_dir=tmp_path / "priors")
    loaded = load_latest_meta_analysis_prior(artifact.parent, target="balanced")
    assert loaded is not None
    assert loaded["version"] == prior["version"]

    grammar = SimpleNamespace(
        category_weights={"frequency": 2.0, "math_space": 2.0},
        op_weights={"spectral_filter": 2.0},
        template_weights={},
        slot_motif_weight_multipliers={},
        slot_motif_denylist={},
    )
    counts = apply_meta_analysis_prior_to_grammar(grammar, loaded)
    assert counts["op_weights"] == len(prior["op_weights"])
    assert grammar.op_weights["spectral_filter"] > 2.0
    assert grammar.template_weights["spectral_memory_block"] > 1.0


def test_meta_analysis_prior_config_fields_round_trip() -> None:
    config = RunConfig(
        use_meta_analysis_priors=True,
        meta_analysis_prior_target="induction",
        meta_analysis_prior_path="/tmp/meta_priors",
    )

    reconstructed = RunConfig.from_dict(config.to_dict())

    assert reconstructed.use_meta_analysis_priors is True
    assert reconstructed.meta_analysis_prior_target == "induction"
    assert reconstructed.meta_analysis_prior_path == "/tmp/meta_priors"


def test_induction_intermediate_target_ignores_legacy_only_rows(tmp_path) -> None:
    meta_db = tmp_path / "meta_analysis.db"
    conn = sqlite3.connect(meta_db)
    conn.execute(
        """
        CREATE TABLE op_observations (
            result_id TEXT,
            op_name TEXT,
            op_category TEXT,
            induction_screening_auc REAL,
            induction_intermediate_auc REAL,
            composite_score REAL,
            stage1_passed INTEGER,
            op_lambda_calculus_affinity REAL,
            op_alternative_math_affinity REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE template_observations (
            result_id TEXT,
            template_name TEXT,
            induction_screening_auc REAL,
            induction_intermediate_auc REAL,
            composite_score REAL,
            stage1_passed INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE op_property_catalog (
            op_name TEXT PRIMARY KEY,
            observed_count INTEGER,
            eval_count INTEGER,
            op_category TEXT,
            op_empirical_probe_needed INTEGER
        )
        """
    )
    for idx in range(8):
        conn.execute(
            """
            INSERT INTO op_observations (
                result_id, op_name, op_category, induction_screening_auc,
                induction_intermediate_auc, composite_score, stage1_passed,
                op_lambda_calculus_affinity, op_alternative_math_affinity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"legacy_{idx}",
                "legacy_only_winner",
                "math_space",
                0.99,
                None,
                5.0,
                1,
                0.0,
                0.0,
            ),
        )
    for idx in range(4):
        conn.execute(
            """
            INSERT INTO op_observations (
                result_id, op_name, op_category, induction_screening_auc,
                induction_intermediate_auc, composite_score, stage1_passed,
                op_lambda_calculus_affinity, op_alternative_math_affinity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"v2_{idx}", "v2_winner", "frequency", 0.01, 0.42, 5.0, 1, 0.0, 0.0),
        )
    for idx in range(4):
        conn.execute(
            """
            INSERT INTO op_observations (
                result_id, op_name, op_category, induction_screening_auc,
                induction_intermediate_auc, composite_score, stage1_passed,
                op_lambda_calculus_affinity, op_alternative_math_affinity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"v2_low_{idx}", "v2_loser", "reduction", 0.50, 0.02, 5.0, 0, 0.0, 0.0),
        )
    conn.commit()
    conn.close()

    prior = build_meta_analysis_prior(
        meta_db_path=meta_db,
        target="induction_intermediate",
        min_support=2,
        created_at=1_800_000_000.0,
    )

    assert prior["signals"]["target_metric"] == "induction_intermediate_auc"
    assert "legacy_only_winner" not in prior["op_weights"]
    assert prior["op_weights"]["v2_winner"] > 1.0
