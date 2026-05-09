from __future__ import annotations

import sqlite3

from research.tools.meta_experiment_queue import build_payload


def test_meta_experiment_queue_builds_profile_and_compression_actions(tmp_path):
    meta_db = tmp_path / "meta.db"
    conn = sqlite3.connect(meta_db)
    conn.executescript(
        """
        CREATE TABLE template_observations (
            result_id TEXT,
            template_name TEXT,
            slot_count INTEGER,
            failure_op TEXT,
            routing_fast_lane_ppl_improvement REAL,
            wikitext_perplexity REAL,
            language_control_s05_sentence_assoc_score REAL
        );
        CREATE TABLE graph_profile_observations (
            result_id TEXT,
            profile_missing_op_count INTEGER,
            profile_coverage_rate REAL
        );
        CREATE TABLE op_observations (
            result_id TEXT,
            op_name TEXT
        );
        CREATE TABLE op_profile_catalog (
            op_name TEXT PRIMARY KEY
        );
        CREATE TABLE slot_observations (
            result_id TEXT,
            template_name TEXT,
            selected_motif TEXT,
            selected_motif_class TEXT,
            has_compression_motif INTEGER,
            failure_op TEXT,
            wikitext_perplexity REAL,
            language_control_s05_sentence_assoc_score REAL,
            frequency_collapse_risk REAL,
            has_effective_positional_mixer INTEGER
        );
        """
    )
    for idx in range(12):
        result_id = f"r{idx}"
        conn.execute(
            "INSERT INTO template_observations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result_id,
                "token_merge_block",
                3,
                "nano_bind" if idx < 4 else "",
                0.1 if idx < 9 else 0.0,
                120.0,
                0.8,
            ),
        )
        conn.execute(
            "INSERT INTO graph_profile_observations VALUES (?, ?, ?)",
            (result_id, 8, 0.5),
        )
        conn.execute(
            "INSERT INTO op_observations VALUES (?, ?)",
            (result_id, "adjacent_token_merge"),
        )
        conn.execute(
            "INSERT INTO slot_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result_id,
                "token_merge_block",
                "adjacent_token_merge",
                "compression",
                1,
                "nano_bind" if idx < 4 else "",
                120.0,
                0.8,
                0.75,
                0,
            ),
        )
    conn.commit()
    conn.close()

    payload = build_payload(
        meta_db, min_support=3, profile_limit=5, compression_limit=5
    )

    assert payload["summary"]["profile_queue_count"] == 1
    assert payload["summary"]["compression_queue_count"] == 1
    profile = payload["profile_refresh_queue"][0]
    assert profile["op_name"] == "adjacent_token_merge"
    assert profile["recommended_scaffold_family"] == "gpt2_replace"
    assert "profile_component_scaffolds" in profile["scaffold_command"]
    compression = payload["compression_safety_queue"][0]
    assert compression["selected_motif"] == "adjacent_token_merge"
    assert compression["recommended_variant"] == (
        "add_or_preserve_positional_or_content_mixer_after_compression"
    )
