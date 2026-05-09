from __future__ import annotations

import sqlite3

from research.tools import real_lm_quickcheck_audit as audit


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE leaderboard (
            result_id TEXT PRIMARY KEY,
            entry_id TEXT,
            tier TEXT,
            composite_score REAL,
            induction_screening_auc REAL,
            binding_screening_composite REAL
        );
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            wikitext_perplexity REAL,
            wikitext_score REAL,
            hellaswag_acc REAL,
            blimp_overall_accuracy REAL,
            tinystories_score REAL,
            language_control_s05_sentence_assoc_score REAL,
            language_control_investigation_sentence_assoc_score REAL
        );
        CREATE TABLE program_graph_features (
            result_id TEXT PRIMARY KEY,
            template_name TEXT
        );
        """
    )
    rows = [
        (
            "attn_good",
            "attention",
            100.0,
            50.0,
            0.9,
            0.30,
            0.60,
            0.8,
        ),
        (
            "merge_good",
            "token_merge_block",
            90.0,
            40.0,
            0.8,
            0.25,
            0.55,
            0.7,
        ),
        (
            "ssm_mid",
            "latent_attn_ssm_hybrid",
            50.0,
            100.0,
            0.5,
            0.20,
            0.53,
            0.4,
        ),
        (
            "other_bad",
            "sparse_ffn",
            10.0,
            500.0,
            0.1,
            0.10,
            0.50,
            0.2,
        ),
    ]
    for rid, template, composite, ppl, wt, hs, blimp, tiny in rows:
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?)",
            (rid, rid, "screening", composite, 0.0, 0.0),
        )
        conn.execute(
            "INSERT INTO program_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, ppl, wt, hs, blimp, tiny, 0.0, 0.0),
        )
        conn.execute(
            "INSERT INTO program_graph_features VALUES (?, ?)",
            (rid, template),
        )
    conn.commit()
    conn.close()


def test_build_report_scores_existing_real_text_metrics(tmp_path):
    db = tmp_path / "lab.db"
    _make_db(db)

    report = audit.build_report(db)

    assert report["coverage"]["rows"] == 4
    assert report["coverage"]["families"]["attention"] == 1
    assert report["coverage"]["families"]["token_merge"] == 1
    assert report["top_core_rows"][0]["result_id"] == "attn_good"
    assert report["top_core_rows"][1]["result_id"] == "merge_good"
    assert report["top_core_rows"][0]["real_lm_core_score"] == 1.0
    assert report["top_core_rows"][-1]["real_lm_core_score"] == 0.0

    corr = {item["metric"]: item for item in report["correlations"]}
    assert corr["real_lm_core_score"]["spearman_vs_wikitext_score"] > 0.999


def test_markdown_report_names_core_score(tmp_path):
    db = tmp_path / "lab.db"
    _make_db(db)
    report = audit.build_report(db)

    text = audit._markdown(report)

    assert "Primary score: `real_lm_core_score`" in text
    assert "## Top Core Rows" in text
    assert "`merge_good`" in text
