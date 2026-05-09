from __future__ import annotations

import json
import sqlite3

from research.meta_analysis import metadata_db
from research.meta_analysis.metadata_db import build_meta_analysis_db
from research.tools.meta_profile_ml_analysis import build_payload as build_ml_payload
from research.tools.externalize_notebook_artifacts import run as externalize_artifacts


def _create_profile_table(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[tuple[str, str], ...],
    primary_key: str,
) -> None:
    defs = ", ".join(f"{name} {sql_type}" for name, sql_type in columns)
    conn.execute(f"CREATE TABLE {table} ({defs}, PRIMARY KEY ({primary_key}))")


def _insert_profile_row(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[tuple[str, str], ...],
    values: dict[str, object],
) -> None:
    names = [name for name, _sql_type in columns]
    conn.execute(
        f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
        tuple(values.get(name) for name in names),
    )


def test_build_meta_analysis_db_materializes_separate_template_slot_tables(
    tmp_path, monkeypatch
):
    source_db = tmp_path / "lab_notebook.db"
    output_db = tmp_path / "meta_analysis.db"
    profiling_db = tmp_path / "component_profiles.db"

    graph = {
        "nodes": {
            "0": {"id": 0, "op_name": "input", "input_ids": []},
            "1": {"id": 1, "op_name": "softmax_attention", "input_ids": [0]},
            "2": {"id": 2, "op_name": "rmsnorm", "input_ids": [1]},
        },
        "metadata": {
            "templates_used": ["typed_slot_memory_block"],
            "template_slot_usage": [
                {
                    "template_name": "typed_slot_memory_block",
                    "slot_index": 2,
                    "slot_key": "typed_slot_memory_block[0].slot2",
                    "slot_key_canonical": "typed_slot_memory_block.slot2",
                    "slot_classes": ["role:global_retrieval", "attention"],
                    "selected_motif": "softmax_attention",
                    "selected_motif_class": "attention",
                    "wildcard": False,
                }
            ],
        },
    }

    src = sqlite3.connect(source_db)
    src.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT NOT NULL,
            graph_fingerprint TEXT,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            loss_ratio REAL,
            tinystories_score REAL,
            induction_intermediate_auc REAL,
            induction_intermediate_status TEXT,
            failure_op TEXT,
            failure_details_json TEXT
        )
        """
    )
    src.execute(
        """
        CREATE TABLE program_graph_features (
            result_id TEXT PRIMARY KEY,
            graph_fingerprint TEXT,
            template_name TEXT,
            templates_json TEXT,
            motifs_json TEXT,
            slot_usage_json TEXT
        )
        """
    )
    src.execute(
        """
        CREATE TABLE program_graph_ops (
            result_id TEXT NOT NULL,
            graph_fingerprint TEXT,
            op_name TEXT NOT NULL,
            PRIMARY KEY (result_id, op_name)
        )
        """
    )
    src.execute(
        """
        INSERT INTO program_results
            (result_id, graph_json, graph_fingerprint, stage0_passed,
             stage05_passed, stage1_passed, loss_ratio,
             tinystories_score, induction_intermediate_auc,
             induction_intermediate_status, failure_op, failure_details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r1",
            json.dumps(graph),
            "fp1",
            1,
            1,
            1,
            0.42,
            0.81,
            0.73,
            "ok",
            "nano_bind",
            '{"reason":"mode_collapse"}',
        ),
    )
    src.execute(
        """
        INSERT INTO program_graph_features (
            result_id, graph_fingerprint, template_name, templates_json,
            motifs_json, slot_usage_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "r1",
            "fp1",
            "typed_slot_memory_block",
            json.dumps(["typed_slot_memory_block"]),
            json.dumps(["norm_layer", "norm_rms", "sparse_semi_structured"]),
            "[]",
        ),
    )
    src.execute(
        """
        INSERT INTO program_graph_ops (result_id, graph_fingerprint, op_name)
        VALUES (?, ?, ?)
        """,
        ("r1", "fp1", "softmax_attention"),
    )
    src.commit()
    src.close()

    prof = sqlite3.connect(profiling_db)
    _create_profile_table(
        prof,
        "op_profiles",
        metadata_db._OP_PROFILE_COLUMNS,
        "op_name, registry",
    )
    _create_profile_table(
        prof,
        "pair_profiles",
        metadata_db._PAIR_PROFILE_COLUMNS,
        "op_a, op_b, composition",
    )
    _create_profile_table(
        prof,
        "triplet_profiles",
        metadata_db._TRIPLET_PROFILE_COLUMNS,
        "op_a, op_b, op_c",
    )
    _insert_profile_row(
        prof,
        "op_profiles",
        metadata_db._OP_PROFILE_COLUMNS,
        {
            "op_name": "input",
            "registry": "primitive",
            "category": "io",
            "forward_time_us": 1.0,
            "backward_time_us": 0.0,
            "peak_memory_bytes": 4,
            "flops_estimate": 0,
            "grad_vanishing": 0,
            "grad_exploding": 0,
            "output_has_nan": 0,
            "lipschitz_estimate": 1.0,
            "jacobian_condition_num": 1.0,
        },
    )
    _insert_profile_row(
        prof,
        "op_profiles",
        metadata_db._OP_PROFILE_COLUMNS,
        {
            "op_name": "softmax_attention",
            "registry": "primitive",
            "category": "mixing",
            "forward_time_us": 12.0,
            "backward_time_us": 20.0,
            "peak_memory_bytes": 128,
            "flops_estimate": 256,
            "grad_vanishing": 0,
            "grad_exploding": 1,
            "output_has_nan": 0,
            "lipschitz_estimate": 3.0,
            "jacobian_condition_num": 10.0,
        },
    )
    _insert_profile_row(
        prof,
        "op_profiles",
        metadata_db._OP_PROFILE_COLUMNS,
        {
            "op_name": "rmsnorm",
            "registry": "primitive",
            "category": "normalization",
            "forward_time_us": 3.0,
            "backward_time_us": 4.0,
            "peak_memory_bytes": 16,
            "flops_estimate": 32,
            "grad_vanishing": 0,
            "grad_exploding": 0,
            "output_has_nan": 0,
            "lipschitz_estimate": 1.2,
            "jacobian_condition_num": 2.0,
        },
    )
    _insert_profile_row(
        prof,
        "pair_profiles",
        metadata_db._PAIR_PROFILE_COLUMNS,
        {
            "op_a": "softmax_attention",
            "op_b": "rmsnorm",
            "composition": "sequential",
            "output_has_nan": 0,
            "grad_has_nan": 0,
            "grad_vanishing": 1,
            "grad_exploding": 0,
            "lipschitz_estimate": 2.0,
        },
    )
    _insert_profile_row(
        prof,
        "triplet_profiles",
        metadata_db._TRIPLET_PROFILE_COLUMNS,
        {
            "op_a": "input",
            "op_b": "softmax_attention",
            "op_c": "rmsnorm",
            "output_has_nan": 0,
            "grad_has_nan": 0,
            "grad_vanishing": 1,
            "grad_exploding": 0,
            "lipschitz_estimate": 2.5,
            "pair_ab_predicted_stable": 1,
            "pair_bc_predicted_stable": 1,
            "triplet_stable": 0,
            "diverges_from_pair_prediction": 1,
        },
    )
    prof.commit()
    prof.close()

    monkeypatch.setattr(
        "research.meta_analysis.metadata_db._infer_active_template_names",
        lambda: {"typed_slot_memory_block"},
    )

    summary = build_meta_analysis_db(
        source_db=source_db,
        output_db=output_db,
        profiling_db=profiling_db,
    )

    assert summary.n_program_rows == 1
    assert summary.n_template_observation_rows == 1
    assert summary.n_slot_observation_rows == 1
    assert summary.n_op_catalog_rows >= 1
    assert summary.n_op_observation_rows == 1
    assert summary.n_alternative_candidate_rows == 4
    assert summary.n_op_profile_rows == 3
    assert summary.n_pair_profile_rows == 1
    assert summary.n_triplet_profile_rows == 1
    assert summary.n_eval_metric_rows >= 10
    assert summary.n_external_component_prior_rows >= 8
    assert summary.n_graph_profile_observation_rows == 1

    out = sqlite3.connect(output_db)
    out.row_factory = sqlite3.Row
    template = out.execute(
        "SELECT * FROM template_property_catalog WHERE template_name = ?",
        ("typed_slot_memory_block",),
    ).fetchone()
    slot = out.execute(
        "SELECT * FROM slot_property_catalog WHERE slot_key = ?",
        ("typed_slot_memory_block.slot2",),
    ).fetchone()
    slot_obs = out.execute(
        "SELECT * FROM slot_observations WHERE result_id = ?",
        ("r1",),
    ).fetchone()
    op = out.execute(
        "SELECT * FROM op_property_catalog WHERE op_name = ?",
        ("softmax_attention",),
    ).fetchone()
    op_obs = out.execute(
        "SELECT * FROM op_observations WHERE result_id = ? AND op_name = ?",
        ("r1", "softmax_attention"),
    ).fetchone()
    candidate = out.execute(
        "SELECT * FROM alternative_math_candidate_catalog WHERE candidate_name = ?",
        ("lambda_calculus",),
    ).fetchone()
    graph_profile = out.execute(
        "SELECT * FROM graph_profile_observations WHERE result_id = ?",
        ("r1",),
    ).fetchone()
    metric = out.execute(
        "SELECT * FROM eval_metric_catalog WHERE metric_name = ?",
        ("language_controluage_nanobind",),
    ).fetchone()
    external_prior = out.execute(
        "SELECT * FROM external_component_prior_catalog WHERE external_family = ?",
        ("attention_gqa_mqa_rope",),
    ).fetchone()
    op_profile = out.execute(
        "SELECT * FROM op_profile_catalog WHERE op_name = ?",
        ("softmax_attention",),
    ).fetchone()
    out.close()

    assert template is not None
    assert template["template_family"] == "memory"
    assert template["template_has_memory"] == 1
    assert template["slot_count"] == 3
    assert template["template_algebraic_linearity_class"] == "bilinear"
    assert template["template_dynamical_memory_length_class"] == "O(L)"
    assert template["template_composition_norm_required"] == "pre"
    assert slot is not None
    assert slot["slot_role"] == "global_retrieval"
    assert slot["slot_role_family"] == "memory_retrieval"
    assert slot["slot_accepts_attention"] == 1
    assert slot["slot_algebraic_linearity_class"] == "bilinear"
    assert slot["slot_spectral_preferred_basis"] == "content"
    assert slot["slot_differentiability_smoothness_class"] == "c_infinity"
    assert slot_obs["selected_motif"] == "softmax_attention"
    assert slot_obs["loss_ratio"] == 0.42
    assert slot_obs["tinystories_score"] == 0.81
    assert slot_obs["failure_op"] == "nano_bind"
    assert slot_obs["motif_count"] == 3
    assert slot_obs["non_norm_motif_count"] == 1
    assert slot_obs["norm_motif_count"] == 2
    assert slot_obs["has_compression_motif"] == 1
    assert slot_obs["has_effective_positional_mixer"] == 0
    assert slot_obs["frequency_collapse_risk"] > 0.8
    assert slot_obs["induction_intermediate_auc"] == 0.73
    assert slot_obs["induction_intermediate_status"] == "ok"
    assert op is not None
    assert op["op_algebraic_linearity_class"] == "bilinear"
    assert op["op_empirical_probe_needed"] == 1
    assert op_obs["motif_count"] == 3
    assert candidate is not None
    assert candidate["candidate_family"] == "symbolic_functional"
    assert candidate["candidate_best_entry_slot"] == "controller_or_program_interpreter"
    assert op_profile is not None
    assert op_profile["forward_time_us"] == 12.0
    assert graph_profile["graph_profile_op_count"] == 3
    assert graph_profile["profile_known_op_count"] == 3
    assert graph_profile["profile_coverage_rate"] == 1.0
    assert graph_profile["profile_slowest_op_name"] == "softmax_attention"
    assert graph_profile["profile_grad_exploding_op_count"] == 1
    assert graph_profile["profile_pair_unstable_count"] == 1
    assert graph_profile["profile_triplet_unstable_count"] == 1
    assert graph_profile["profile_triplet_divergent_count"] == 1
    assert metric is not None
    assert json.loads(metric["source_columns_json"]) == [
        "language_control_s05_sentence_assoc_score",
        "language_control_s05_binding_order_acc",
        "language_control_s05_binding_score",
        "language_control_s10_sentence_assoc_score",
        "language_control_investigation_sentence_assoc_score",
        "failure_op",
        "failure_details_json",
    ]
    assert external_prior is not None
    assert "softmax_attention" in json.loads(external_prior["mapped_ops_json"])

    ml_payload = build_ml_payload(output_db, min_support=1)
    assert ml_payload["summary"]["n_graphs"] == 1
    assert ml_payload["summary"]["n_features"] >= 60
    assert ml_payload["summary"]["target_nano_bind_failure_positives"] == 1
    assert "target_nano_bind_failure" in ml_payload["targets"]

    src_check = sqlite3.connect(source_db)
    source_tables = {
        row[0]
        for row in src_check.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    src_check.close()
    assert source_tables == {
        "program_results",
        "program_graph_features",
        "program_graph_ops",
    }


def test_build_meta_analysis_db_resolves_artifact_backed_graph_json(
    tmp_path, monkeypatch
):
    source_db = tmp_path / "runs.db"
    output_db = tmp_path / "meta_analysis.db"
    graph = {
        "nodes": {
            "0": {"id": 0, "op_name": "input", "input_ids": []},
            "1": {"id": 1, "op_name": "softmax_attention", "input_ids": [0]},
        },
        "metadata": {
            "templates_used": ["typed_slot_memory_block"],
            "template_slot_usage": [
                {
                    "template_name": "typed_slot_memory_block",
                    "slot_index": 0,
                    "slot_key": "typed_slot_memory_block[0].slot0",
                    "slot_classes": ["attention"],
                    "selected_motif": "softmax_attention",
                }
            ],
        },
    }
    src = sqlite3.connect(source_db)
    src.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT NOT NULL,
            graph_fingerprint TEXT,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            loss_ratio REAL
        )
        """
    )
    src.execute(
        """
        INSERT INTO program_results
            (result_id, graph_json, graph_fingerprint, stage0_passed,
             stage05_passed, stage1_passed, loss_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("r-artifact", json.dumps(graph), "fp-artifact", 1, 1, 1, 0.5),
    )
    src.commit()
    src.close()

    externalize_artifacts(
        db_path=source_db,
        min_bytes=16,
        apply=True,
        limit=None,
        vacuum=False,
        include_graph_json=True,
        graph_json_cold_only=False,
    )
    monkeypatch.setattr(
        "research.meta_analysis.metadata_db._infer_active_template_names",
        lambda: {"typed_slot_memory_block"},
    )

    summary = build_meta_analysis_db(source_db=source_db, output_db=output_db)

    assert summary.n_program_rows == 1
    assert summary.n_template_observation_rows == 1
    assert summary.n_slot_observation_rows == 1
    out = sqlite3.connect(output_db)
    assert (
        out.execute(
            "SELECT COUNT(*) FROM graph_profile_observations WHERE result_id = ?",
            ("r-artifact",),
        ).fetchone()[0]
        == 1
    )
    out.close()
