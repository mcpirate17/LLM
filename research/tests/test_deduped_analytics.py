from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research.scientist.analytics._exp_comparisons import _ComparisonsMixin
from research.scientist.analytics.analytics_grammar import _GrammarMixin
from research.scientist.analytics._exp_weights import _WeightsMixin
from research.scientist.intelligence.analyzer import cluster_architecture_families
from research.scientist.intelligence.analyzer import analyze_efficiency_profiles
from research.scientist.intelligence.digest import ArchitectureFamily
from research.tests._ml_corpus_test_support import create_test_db, graph_json


class _NotebookStub:
    def __init__(self, path: Path):
        self.db_path = path


class _AnalyticsStub(_WeightsMixin):
    __slots__ = ("nb",)

    def __init__(self, path: Path):
        self.nb = _NotebookStub(path)


class _ComparisonsStub(_ComparisonsMixin):
    __slots__ = ("nb",)

    def __init__(self, path: Path):
        self.nb = _NotebookStub(path)

    @staticmethod
    def _extract_ops_fast(graph_json_str: str):
        return _extract_ops(graph_json_str)

    @staticmethod
    def _extract_ops_fallback(graph_json_str: str):
        return _extract_ops(graph_json_str)


class _GrammarStub(_GrammarMixin):
    __slots__ = ("nb",)

    def __init__(self, path: Path):
        self.nb = _NotebookStub(path)

    @staticmethod
    def _extract_ops_fast(graph_json_str: str):
        return _extract_ops(graph_json_str)

    @staticmethod
    def _extract_ops_fallback(graph_json_str: str):
        return _extract_ops(graph_json_str)

    @staticmethod
    def _depth_bucket(depth):
        depth_value = float(depth or 0.0)
        if depth_value < 3:
            return "shallow"
        if depth_value < 6:
            return "medium"
        return "deep"


def _create_full_analysis_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE program_results (
            result_id TEXT,
            experiment_id TEXT,
            graph_json TEXT,
            graph_fingerprint TEXT,
            fingerprint_json TEXT,
            novelty_score REAL,
            structural_novelty REAL,
            loss_ratio REAL,
            wikitext_perplexity REAL,
            param_count REAL,
            graph_n_params_estimate REAL,
            graph_depth REAL,
            graph_uses_math_spaces INTEGER,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            timestamp REAL
        );
        """
    )
    conn.close()


def _extract_ops(graph_json_str: str):
    try:
        graph = json.loads(graph_json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    nodes = graph.get("nodes", {})
    ops = []
    if isinstance(nodes, dict):
        values = nodes.values()
    elif isinstance(nodes, list):
        values = nodes
    else:
        return None
    for node in values:
        if not isinstance(node, dict):
            continue
        op = node.get("op_name") or node.get("op_type") or node.get("op")
        if op and op not in {"input", "output"}:
            ops.append(op)
    return sorted(set(ops))


def test_template_weights_ignore_metadata_only_reruns(tmp_path: Path) -> None:
    db_path = tmp_path / "weights.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "a_fail",
                graph_json('{"templates_used":["template_a"]}'),
                "stale_a1",
                None,
                None,
                None,
                1.1,
                11.0,
                1,
                0,
                0,
                1.0,
                "candidate_screening",
                "screening_only",
            ),
            (
                "a_pass",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "stale_a2",
                None,
                None,
                None,
                0.6,
                8.0,
                1,
                1,
                1,
                2.0,
                "candidate_grade",
                "candidate_comparable",
            ),
            (
                "b_pass",
                graph_json('{"templates_used":["template_b"]}', middle_op="relu"),
                "stale_b1",
                None,
                None,
                None,
                0.7,
                9.0,
                1,
                1,
                1,
                3.0,
                "candidate_grade",
                "candidate_comparable",
            ),
        ],
    )
    conn.commit()
    conn.close()

    analytics = _AnalyticsStub(db_path)
    weights = analytics.compute_template_weights(min_used=1)

    assert weights == {"template_a": 1.0, "template_b": 1.0}


def test_architecture_family_clustering_requires_unique_s1_graphs(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "families.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "dup_1",
                graph_json('{"templates_used":["template_a"]}'),
                "old_dup_1",
                None,
                None,
                None,
                0.5,
                8.0,
                1,
                1,
                1,
                1.0,
                "candidate_grade",
                "candidate_comparable",
            ),
            (
                "dup_2",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "old_dup_2",
                None,
                None,
                None,
                0.4,
                7.5,
                1,
                1,
                1,
                2.0,
                "candidate_grade",
                "candidate_comparable",
            ),
            (
                "unique_pass",
                graph_json('{"templates_used":["template_b"]}', middle_op="relu"),
                "old_unique",
                None,
                None,
                None,
                0.6,
                8.5,
                1,
                1,
                1,
                3.0,
                "candidate_grade",
                "candidate_comparable",
            ),
        ],
    )
    conn.commit()
    conn.close()

    families = cluster_architecture_families(_NotebookStub(db_path), min_cluster_size=3)
    assert families == []


def test_top_op_combinations_dedupe_stage1_reruns(tmp_path: Path) -> None:
    db_path = tmp_path / "op_combos.sqlite3"
    _create_full_analysis_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results (
            result_id, experiment_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            param_count, graph_n_params_estimate, graph_depth, graph_uses_math_spaces,
            stage0_passed, stage05_passed, stage1_passed, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "dup_1",
                "exp_a",
                graph_json('{"templates_used":["template_a"]}'),
                "old_dup_1",
                None,
                0.2,
                None,
                0.6,
                8.0,
                128.0,
                128.0,
                3.0,
                0,
                1,
                1,
                1,
                1.0,
            ),
            (
                "dup_2",
                "exp_b",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "old_dup_2",
                None,
                0.9,
                None,
                0.5,
                7.5,
                128.0,
                128.0,
                3.0,
                0,
                1,
                1,
                1,
                2.0,
            ),
            (
                "other",
                "exp_c",
                graph_json('{"templates_used":["template_b"]}', middle_op="relu"),
                "old_other",
                None,
                0.4,
                None,
                0.7,
                8.5,
                256.0,
                256.0,
                4.0,
                1,
                1,
                1,
                1,
                3.0,
            ),
        ],
    )
    conn.commit()
    conn.close()

    combos = _ComparisonsStub(db_path).top_op_combinations(5)
    assert combos[0]["ops"] == ["add", "layernorm"]
    assert combos[0]["count"] == 1


def test_program_factor_rows_dedupe_metadata_only_reruns(tmp_path: Path) -> None:
    db_path = tmp_path / "grammar.sqlite3"
    _create_full_analysis_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results (
            result_id, experiment_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            param_count, graph_n_params_estimate, graph_depth, graph_uses_math_spaces,
            stage0_passed, stage05_passed, stage1_passed, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "g1",
                "exp_1",
                graph_json('{"templates_used":["template_a"]}'),
                "old_1",
                None,
                0.1,
                None,
                0.9,
                10.0,
                64.0,
                64.0,
                2.0,
                0,
                1,
                0,
                0,
                1.0,
            ),
            (
                "g2",
                "exp_2",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "old_2",
                None,
                0.6,
                None,
                0.4,
                7.0,
                64.0,
                64.0,
                2.0,
                1,
                1,
                1,
                1,
                2.0,
            ),
            (
                "g3",
                "exp_3",
                graph_json('{"templates_used":["template_b"]}', middle_op="relu"),
                "old_3",
                None,
                0.3,
                None,
                0.8,
                9.0,
                96.0,
                96.0,
                4.0,
                0,
                1,
                0,
                0,
                3.0,
            ),
        ],
    )
    conn.commit()
    conn.close()

    rows = _GrammarStub(db_path)._load_program_factor_rows()
    assert len(rows) == 2
    layernorm_row = next(row for row in rows if "layernorm" in row["ops"])
    assert layernorm_row["stage1_passed"] == 1
    assert layernorm_row["math_space"] is True


def test_efficiency_profiles_dedupe_stage1_reruns(tmp_path: Path) -> None:
    db_path = tmp_path / "efficiency.sqlite3"
    _create_full_analysis_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results (
            result_id, experiment_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            param_count, graph_n_params_estimate, graph_depth, graph_uses_math_spaces,
            stage0_passed, stage05_passed, stage1_passed, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "e1",
                "exp_1",
                graph_json('{"templates_used":["template_a"]}'),
                "old_1",
                None,
                0.1,
                None,
                0.7,
                9.0,
                100.0,
                100.0,
                3.0,
                0,
                1,
                1,
                1,
                1.0,
            ),
            (
                "e2",
                "exp_2",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "old_2",
                None,
                0.2,
                None,
                0.5,
                7.0,
                100.0,
                100.0,
                3.0,
                0,
                1,
                1,
                1,
                2.0,
            ),
        ],
    )
    conn.commit()
    conn.close()

    families = [
        ArchitectureFamily(
            family_id=1, representative_ops=["add", "layernorm"], n_members=1
        )
    ]
    profiles = analyze_efficiency_profiles(_NotebookStub(db_path), families)
    assert len(profiles) == 1
    assert profiles[0].avg_params == 100.0
