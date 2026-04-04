"""Tests for metric-aware template backfill counting."""

from __future__ import annotations

import sqlite3

from research.tools.backfill_templates import get_template_counts, get_template_stats


def test_get_template_stats_and_counts_from_template_stats(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE template_stats (
               template_name TEXT,
               eval_count INTEGER,
               s0_pass_count INTEGER,
               s1_pass_count INTEGER
           )"""
    )
    conn.executemany(
        "INSERT INTO template_stats(template_name, eval_count, s0_pass_count, s1_pass_count) "
        "VALUES (?, ?, ?, ?)",
        [
            ("latent_attn_ffn_block", 20, 8, 3),
            ("local_attn_ffn_block", 11, 7, 1),
        ],
    )
    conn.commit()
    conn.close()

    stats = get_template_stats(db_path)
    assert stats["latent_attn_ffn_block"] == {"eval": 20, "s0": 8, "s1": 3}
    assert stats["local_attn_ffn_block"] == {"eval": 11, "s0": 7, "s1": 1}

    assert get_template_counts(db_path, metric="eval")["latent_attn_ffn_block"] == 20
    assert get_template_counts(db_path, metric="s0")["latent_attn_ffn_block"] == 8
    assert get_template_counts(db_path, metric="s1")["latent_attn_ffn_block"] == 3


def test_get_template_stats_prefers_live_program_results(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE template_stats (
               template_name TEXT,
               eval_count INTEGER,
               s0_pass_count INTEGER,
               s1_pass_count INTEGER
           )"""
    )
    conn.execute(
        """CREATE TABLE program_results (
               graph_json TEXT,
               stage0_passed INTEGER,
               stage1_passed INTEGER
           )"""
    )
    conn.execute(
        "INSERT INTO template_stats(template_name, eval_count, s0_pass_count, s1_pass_count) "
        "VALUES ('latent_attn_ffn_block', 0, 0, 0)"
    )
    graph_json = '{"metadata":{"templates_used":["latent_attn_ffn_block","latent_attn_ffn_block"]}}'
    conn.executemany(
        "INSERT INTO program_results(graph_json, stage0_passed, stage1_passed) "
        "VALUES (?, ?, ?)",
        [
            (graph_json, 1, 1),
            (graph_json, 1, 0),
        ],
    )
    conn.commit()
    conn.close()

    stats = get_template_stats(db_path)
    assert stats["latent_attn_ffn_block"] == {"eval": 2, "s0": 2, "s1": 1}
