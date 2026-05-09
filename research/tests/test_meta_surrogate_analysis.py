from __future__ import annotations

import sqlite3

from research.tools.meta_surrogate_analysis import (
    analyze_templates,
    load_graph_rows,
)


def test_surrogate_uses_template_observations_for_template_risk(tmp_path):
    db = tmp_path / "meta.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE template_observations (
            result_id TEXT,
            template_name TEXT,
            slot_count INTEGER,
            failure_op TEXT,
            wikitext_perplexity REAL,
            tinystories_score REAL,
            language_control_s05_sentence_assoc_score REAL,
            motif_count INTEGER,
            non_norm_motif_count INTEGER,
            norm_motif_count INTEGER,
            norm_dominance REAL,
            has_attention_motif INTEGER,
            has_ssm_motif INTEGER,
            has_conv_motif INTEGER,
            has_recurrent_motif INTEGER,
            has_routing_motif INTEGER,
            has_compression_motif INTEGER,
            has_effective_positional_mixer INTEGER,
            mixer_after_compression INTEGER,
            motif_thinness_score REAL,
            frequency_collapse_risk REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_observations (
            template_name TEXT,
            slot_index INTEGER,
            selected_motif TEXT,
            selected_motif_class TEXT,
            failure_op TEXT,
            language_control_s05_sentence_assoc_score REAL,
            frequency_collapse_risk REAL,
            has_effective_positional_mixer INTEGER
        )
        """
    )
    rows = []
    for idx in range(10):
        rows.append(
            (
                f"r{idx}",
                "primary_template",
                3,
                None,
                100.0,
                0.5,
                0.9,
                3,
                1,
                2,
                0.6667,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0.75,
                0.8,
            )
        )
        rows.append(
            (
                f"r{idx}",
                "secondary_token_merge",
                1,
                "nano_bind" if idx < 7 else None,
                50.0,
                0.4,
                0.1,
                3,
                1,
                2,
                0.6667,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0.75,
                0.8,
            )
        )
    conn.executemany(
        """
        INSERT INTO template_observations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    conn.commit()
    conn.close()

    graph_rows = load_graph_rows(db)
    assert len(graph_rows) == 10
    assert all(row.template_name == "primary_template" for row in graph_rows)

    template_rows = analyze_templates(db)
    by_template = {row["template_name"]: row for row in template_rows}
    assert by_template["secondary_token_merge"]["n"] == 10
    assert by_template["secondary_token_merge"]["nano_bind_rate"] == 0.7
