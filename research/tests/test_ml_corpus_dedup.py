from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from research.tests._ml_corpus_test_support import create_test_db, graph_json


def test_graph_training_corpus_dedupes_metadata_only_reruns(tmp_path: Path) -> None:
    from research.scientist.intelligence.ml_corpus import (
        load_deduped_graph_training_rows,
    )

    db_path = tmp_path / "ml_corpus.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    graph_a = graph_json('{"templates_used":["a"]}')
    graph_b = graph_json('{"templates_used":["b"],"lineage":{"parent":"x"}}')
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r1",
            graph_a,
            "stale_fp_1",
            '{"isotropy": 0.1}',
            0.2,
            0.1,
            1.2,
            12.0,
            1,
            0,
            0,
            1.0,
            "candidate_screening",
            "screening_only",
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r2",
            graph_b,
            "stale_fp_2",
            '{"isotropy": 0.9}',
            0.8,
            0.7,
            0.4,
            8.0,
            1,
            1,
            1,
            2.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    conn.execute(
        "INSERT INTO leaderboard VALUES (?, ?, ?)",
        ("r2", 0.25, "validation"),
    )
    conn.commit()
    conn.close()

    rows = load_deduped_graph_training_rows(db_path)
    assert len(rows) == 1

    row = rows[0]
    assert row["n_rows"] == 2
    assert row["stage1_any_passed"] is True
    assert np.isclose(row["stage1_pass_rate"], 0.5)
    assert row["loss_ratio_best"] == 0.4
    assert row["wikitext_perplexity_best"] == 8.0
    assert row["stage05_any_passed"] is True


def test_graph_training_corpus_excludes_untrusted_rows_when_labels_exist(
    tmp_path: Path,
) -> None:
    from research.scientist.intelligence.ml_corpus import (
        load_deduped_graph_training_rows,
    )

    db_path = tmp_path / "ml_corpus_trust.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    shared_graph = graph_json('{"templates_used":["trust_test"]}')
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "trusted_row",
            shared_graph,
            "trusted_fp",
            '{"isotropy": 0.2}',
            0.4,
            0.3,
            0.7,
            9.0,
            1,
            1,
            1,
            1.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "backfill_row",
            shared_graph,
            "backfill_fp",
            '{"isotropy": 0.8}',
            0.9,
            0.8,
            0.2,
            4.0,
            1,
            1,
            1,
            2.0,
            "backfill_observation",
            "reconstructed_init_variant",
        ),
    )
    conn.commit()
    conn.close()

    rows = load_deduped_graph_training_rows(db_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["n_rows"] == 1
    assert row["loss_ratio_best"] == 0.7
    assert row["wikitext_perplexity_best"] == 9.0


def test_graph_training_corpus_cache_hits_when_db_unchanged(monkeypatch) -> None:
    from research.scientist.intelligence import ml_corpus

    ml_corpus._clear_corpus_cache()
    call_count = {"n": 0}

    def _fake_builder(_db_path: str):
        call_count["n"] += 1
        return [
            {
                "canonical_fingerprint": "fp_a",
                "graph_json": graph_json('{"templates_used":["a"]}'),
                "stage1_any_passed": True,
                "stage1_pass_rate": 1.0,
                "stage0_any_passed": True,
                "stage05_any_passed": True,
                "wikitext_perplexity_best": 7.0,
                "loss_ratio_best": 0.4,
                "n_rows": 1,
                "latest_timestamp": 1.0,
            }
        ]

    monkeypatch.setattr(
        ml_corpus,
        "_build_graph_training_rows",
        _fake_builder,
    )
    monkeypatch.setattr(
        ml_corpus,
        "_db_cache_signature",
        lambda _db_path: ((1, 100, 1), (0, 0, 0)),
    )

    rows_a = ml_corpus.load_deduped_graph_training_rows("/tmp/cache_test.sqlite3")
    rows_b = ml_corpus.load_deduped_graph_training_rows("/tmp/cache_test.sqlite3")

    assert call_count["n"] == 1
    assert rows_a is rows_b


def test_graph_training_corpus_cache_invalidates_on_db_signature_change(
    monkeypatch,
) -> None:
    from research.scientist.intelligence import ml_corpus

    ml_corpus._clear_corpus_cache()
    call_count = {"n": 0}

    def _fake_builder(_db_path: str):
        call_count["n"] += 1
        return [
            {
                "canonical_fingerprint": f"fp_{call_count['n']}",
                "graph_json": graph_json('{"templates_used":["a"]}'),
                "stage1_any_passed": True,
                "stage1_pass_rate": 1.0,
                "stage0_any_passed": True,
                "stage05_any_passed": True,
                "wikitext_perplexity_best": 7.0,
                "loss_ratio_best": 0.4,
                "n_rows": 1,
                "latest_timestamp": 1.0,
            }
        ]

    signatures = iter(
        [
            ((1, 100, 1), (0, 0, 0)),
            ((2, 100, 1), (0, 0, 0)),
        ]
    )
    monkeypatch.setattr(ml_corpus, "_build_graph_training_rows", _fake_builder)
    monkeypatch.setattr(
        ml_corpus, "_db_cache_signature", lambda _db_path: next(signatures)
    )

    rows_a = ml_corpus.load_deduped_graph_training_rows("/tmp/cache_test.sqlite3")
    rows_b = ml_corpus.load_deduped_graph_training_rows("/tmp/cache_test.sqlite3")

    assert call_count["n"] == 2
    assert rows_a[0]["canonical_fingerprint"] == "fp_1"
    assert rows_b[0]["canonical_fingerprint"] == "fp_2"


def test_graph_training_corpus_cache_stats_track_hit_miss_and_validation(
    monkeypatch,
) -> None:
    from research.scientist.intelligence import ml_corpus

    ml_corpus._clear_corpus_cache()

    monkeypatch.setattr(
        ml_corpus,
        "_build_graph_training_rows",
        lambda _db_path: [
            {
                "canonical_fingerprint": ml_corpus._graph_fingerprint(
                    graph_json('{"templates_used":["a"]}')
                ),
                "graph_json": graph_json('{"templates_used":["a"]}'),
                "stage1_any_passed": True,
                "stage1_pass_rate": 1.0,
                "stage0_any_passed": True,
                "stage05_any_passed": True,
                "wikitext_perplexity_best": 7.0,
                "loss_ratio_best": 0.4,
                "n_rows": 1,
                "latest_timestamp": 1.0,
            }
        ],
    )
    monkeypatch.setattr(
        ml_corpus,
        "_db_cache_signature",
        lambda _db_path: ((1, 100, 1), (0, 0, 0)),
    )

    ml_corpus.load_deduped_graph_training_rows(
        "/tmp/cache_stats.sqlite3", validate=True
    )
    ml_corpus.load_deduped_graph_training_rows(
        "/tmp/cache_stats.sqlite3", validate=True
    )

    stats = ml_corpus.get_corpus_cache_stats()
    assert stats["misses"] >= 1
    assert stats["hits"] >= 1
    assert stats["validations"] >= 1


def test_graph_training_corpus_validation_rejects_duplicate_canonical_rows() -> None:
    from research.scientist.intelligence.ml_corpus import (
        CorpusIntegrityError,
        _graph_fingerprint,
        validate_graph_training_rows,
    )

    graph = graph_json('{"templates_used":["a"]}')
    canonical = _graph_fingerprint(graph)
    rows = [
        {
            "canonical_fingerprint": canonical,
            "graph_json": graph,
            "stage1_any_passed": True,
            "stage1_pass_rate": 1.0,
            "n_rows": 1,
        },
        {
            "canonical_fingerprint": canonical,
            "graph_json": graph,
            "stage1_any_passed": False,
            "stage1_pass_rate": 0.0,
            "n_rows": 1,
        },
    ]

    with pytest.raises(CorpusIntegrityError, match="duplicate canonical_fingerprint"):
        validate_graph_training_rows(rows)


def test_predictor_query_reraises_corpus_integrity_error(monkeypatch) -> None:
    from research.scientist.intelligence.ml_corpus import CorpusIntegrityError
    from research.scientist.intelligence.predictor import _query_training_data

    def _broken_loader(*_args, **_kwargs):
        raise CorpusIntegrityError("predictor corpus broken")

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor_ridge.load_deduped_predictor_training_rows",
        _broken_loader,
    )

    class _NotebookStub:
        db_path = Path("/tmp/unused.sqlite3")

    with pytest.raises(CorpusIntegrityError, match="predictor corpus broken"):
        _query_training_data(_NotebookStub())


def test_gbm_query_reraises_corpus_integrity_error(monkeypatch) -> None:
    from research.scientist.intelligence.ml_corpus import CorpusIntegrityError
    from research.scientist.intelligence.predictor import _query_graph_training_data

    def _broken_loader(*_args, **_kwargs):
        raise CorpusIntegrityError("graph corpus broken")

    monkeypatch.setattr(
        "research.scientist.intelligence.predictor_gbm.load_screening_predictor_corpus_rows",
        _broken_loader,
    )

    with pytest.raises(CorpusIntegrityError, match="graph corpus broken"):
        _query_graph_training_data("/tmp/unused.sqlite3")


def test_predictor_query_uses_deduped_corpus(tmp_path: Path) -> None:
    from research.scientist.intelligence.predictor import _query_training_data

    db_path = tmp_path / "predictor_corpus.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    graph = graph_json('{"templates_used":["a"]}')
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "screening_row",
            graph,
            "old_fp_a",
            '{"isotropy": 0.1}',
            0.2,
            0.3,
            0.8,
            11.0,
            1,
            0,
            1,
            1.0,
            "candidate_screening",
            "screening_only",
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "validation_row",
            graph,
            "old_fp_b",
            '{"isotropy": 0.9}',
            0.7,
            0.8,
            0.6,
            9.0,
            1,
            1,
            1,
            2.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    conn.execute(
        "INSERT INTO leaderboard VALUES (?, ?, ?)",
        ("screening_row", None, "screening"),
    )
    conn.execute(
        "INSERT INTO leaderboard VALUES (?, ?, ?)",
        ("validation_row", 0.25, "validation"),
    )
    conn.commit()
    conn.close()

    class _NotebookStub:
        def __init__(self, path: Path):
            self.db_path = path

    X, y, w = _query_training_data(_NotebookStub(db_path))

    assert X.shape == (1, 18)
    assert y.shape == (1,)
    assert w.shape == (1,)
    assert np.isclose(y[0], 0.25)
    assert w[0] >= 6.0


def test_predictor_query_excludes_non_promotable_rows(tmp_path: Path) -> None:
    from research.scientist.intelligence.predictor import _query_training_data

    db_path = tmp_path / "predictor_corpus_untrusted.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "backfill_only",
            graph_json('{"templates_used":["backfill_only"]}'),
            "backfill_fp",
            '{"isotropy": 0.5}',
            0.9,
            0.7,
            0.3,
            6.0,
            1,
            1,
            1,
            1.0,
            "backfill_observation",
            "reconstructed_init_variant",
        ),
    )
    conn.execute(
        "INSERT INTO leaderboard VALUES (?, ?, ?)",
        ("backfill_only", 0.2, "validation"),
    )
    conn.commit()
    conn.close()

    class _NotebookStub:
        def __init__(self, path: Path):
            self.db_path = path

    X, y, w = _query_training_data(_NotebookStub(db_path))

    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert w.shape[0] == 0


def test_gbm_query_uses_deduped_graph_rows(tmp_path: Path) -> None:
    from research.scientist.intelligence.predictor import _query_graph_training_data

    db_path = tmp_path / "gbm_corpus.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "g1",
            graph_json('{"templates_used":["a"]}'),
            "stale_1",
            '{"isotropy": 0.1}',
            0.1,
            0.2,
            0.9,
            10.0,
            1,
            0,
            0,
            1.0,
            "candidate_screening",
            "screening_only",
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "g2",
            graph_json('{"templates_used":["b"],"lineage":{"parent":"x"}}'),
            "stale_2",
            '{"isotropy": 0.9}',
            0.8,
            0.9,
            0.4,
            7.0,
            1,
            1,
            1,
            2.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    conn.commit()
    conn.close()

    (
        feat_dicts,
        y_gate,
        y_rank_ppl,
        y_rank_composite,
        sample_weights,
        latest_timestamps,
        graph_signatures,
    ) = _query_graph_training_data(str(db_path))
    assert len(feat_dicts) == 1
    assert y_gate.tolist() == [1]
    assert y_rank_ppl.tolist() == [7.0]
    assert np.isnan(y_rank_composite[0])
    assert sample_weights.shape == (1,)
    assert latest_timestamps.shape == (1,)
    assert len(graph_signatures) == 1


def test_op_embeddings_cooccurrence_pairs_dedupe_reruns(tmp_path: Path) -> None:
    from research.scientist.intelligence.op_embeddings import (
        _extract_cooccurrence_pairs,
    )

    db_path = tmp_path / "op_embeddings.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "p1",
            graph_json('{"templates_used":["a"]}'),
            "old_a",
            '{"isotropy": 0.1}',
            0.1,
            0.2,
            1.0,
            9.0,
            1,
            0,
            0,
            1.0,
            "candidate_screening",
            "screening_only",
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "p2",
            graph_json('{"templates_used":["b"],"lineage":{"parent":"x"}}'),
            "old_b",
            '{"isotropy": 0.2}',
            0.2,
            0.3,
            0.5,
            8.0,
            1,
            1,
            1,
            2.0,
            "candidate_grade",
            "candidate_comparable",
        ),
    )
    conn.commit()
    conn.close()

    op_to_idx = {"add": 0, "layernorm": 1}
    positive, negative = _extract_cooccurrence_pairs(db_path, op_to_idx)
    assert positive == [(0, 1)]
    assert negative == []


def test_graph_fingerprint_fallback_uses_shared_graph_parser(monkeypatch) -> None:
    from research.scientist.intelligence import ml_corpus
    import research.synthesis.serializer as serializer_mod

    calls = {"n": 0}
    original_graph_from_json = serializer_mod.graph_from_json

    def _counted_graph_from_json(payload: str):
        calls["n"] += 1
        return original_graph_from_json(payload)

    monkeypatch.setattr(ml_corpus, "_try_import_rust_scheduler", lambda: None)
    monkeypatch.setattr(serializer_mod, "graph_from_json", _counted_graph_from_json)

    payload = graph_json('{"templates_used":["fp"]}')
    fingerprint = ml_corpus._graph_fingerprint(payload)

    assert fingerprint
    assert calls["n"] == 1
