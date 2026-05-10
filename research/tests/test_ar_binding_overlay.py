"""Tests for advisory AR/binding overlay joins."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research.meta_analysis.ar_binding_overlay import (
    empty_overlay,
    overlay_for_chain,
    overlay_for_graph,
    overlay_for_pair,
    overlay_for_routing_decision,
)
from research.scientist.intelligence.ar_binding_reranker import (
    rerank_graphs_by_ar_binding,
    score_ar_binding_overlay,
)
from research.scientist.runner._types import RunConfig


def _create_meta_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE op_observations (
            result_id TEXT NOT NULL,
            op_name TEXT NOT NULL,
            ar_curriculum_auc_pair_final REAL,
            ar_validation_rank_score REAL,
            ar_intermediate_auc REAL,
            ar_gate_score REAL,
            ar_legacy_auc REAL,
            binding_multislot_auc REAL,
            binding_intermediate_auc REAL,
            binding_curriculum_auc REAL,
            binding_screening_auc REAL,
            ar_curriculum_s0_retention REAL,
            frequency_collapse_risk REAL
        )
        """
    )
    rows = [
        ("r1", "a", 0.80, None, None, None, None, 0.60, None, None, None, 0.90, 0.20),
        ("r1", "b", 0.80, None, None, None, None, 0.60, None, None, None, 0.90, 0.20),
        ("r2", "a", 0.20, None, None, None, None, 0.20, None, None, None, 0.40, 0.70),
        ("r2", "c", 0.20, None, None, None, None, 0.20, None, None, None, 0.40, 0.70),
    ]
    conn.executemany(
        """
        INSERT INTO op_observations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _create_runs_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE program_results (
            graph_json TEXT NOT NULL,
            ar_curriculum_auc_pair_final REAL,
            ar_validation_rank_score REAL,
            ar_intermediate_auc REAL,
            ar_gate_score REAL,
            ar_legacy_auc REAL,
            binding_multislot_auc REAL,
            binding_intermediate_auc REAL,
            binding_curriculum_auc REAL,
            binding_screening_auc REAL,
            ar_curriculum_s0_retention REAL,
            routing_collapse_score REAL
        )
        """
    )
    graph = {
        "metadata": {
            "routing_decisions": [
                {
                    "template_name": "tpl_multiscale",
                    "decision_key": "hard_num_experts",
                    "value": 4,
                    "source": "policy",
                }
            ]
        }
    }
    other = {
        "metadata": {
            "routing_decisions": [
                {
                    "template_name": "tpl_multiscale",
                    "decision_key": "hard_num_experts",
                    "value": 8,
                    "source": "policy",
                }
            ]
        }
    }
    conn.executemany(
        "INSERT INTO program_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                json.dumps(graph),
                0.70,
                None,
                None,
                None,
                None,
                0.40,
                None,
                None,
                None,
                0.85,
                0.10,
            ),
            (
                json.dumps(other),
                0.30,
                None,
                None,
                None,
                None,
                0.20,
                None,
                None,
                None,
                0.50,
                0.50,
            ),
        ],
    )
    conn.commit()
    conn.close()


def test_overlay_for_pair_keeps_gain_and_risk_separate(tmp_path: Path):
    meta_db = tmp_path / "meta.db"
    _create_meta_db(meta_db)

    overlay = overlay_for_pair("a", "b", meta_db_path=meta_db)

    assert overlay["expected_ar_gain"] == 0.3
    assert overlay["ar_gain_n"] == 1
    assert overlay["expected_binding_gain"] == 0.2
    assert overlay["binding_gain_n"] == 1
    assert overlay["retention_risk"] == 0.1
    assert overlay["collapse_risk"] == 0.2
    assert overlay["holdout_required"] is True


def test_overlay_for_chain_and_graph_use_same_shape(tmp_path: Path):
    meta_db = tmp_path / "meta.db"
    _create_meta_db(meta_db)

    chain_overlay = overlay_for_chain(["a", "b"], meta_db_path=meta_db)
    graph_overlay = overlay_for_graph(
        {"nodes": [{"op_name": "a"}, {"component_type": "b"}]},
        meta_db_path=meta_db,
    )

    assert graph_overlay == chain_overlay
    assert set(graph_overlay) == set(empty_overlay())


def test_overlay_for_missing_database_returns_empty(tmp_path: Path):
    assert (
        overlay_for_pair("a", "b", meta_db_path=tmp_path / "missing.db")
        == empty_overlay()
    )


def test_overlay_for_routing_decision_matches_audit_payload(tmp_path: Path):
    runs_db = tmp_path / "runs.db"
    _create_runs_db(runs_db)

    overlay = overlay_for_routing_decision(
        "tpl_multiscale",
        "hard_num_experts",
        4,
        runs_db_path=runs_db,
    )

    assert overlay["expected_ar_gain"] == 0.2
    assert overlay["expected_binding_gain"] == 0.1
    assert overlay["ar_gain_n"] == 1
    assert overlay["binding_gain_n"] == 1
    assert overlay["retention_risk"] == 0.15
    assert overlay["collapse_risk"] == 0.1
    assert overlay["holdout_required"] is True


def test_ar_binding_reranker_scores_only_holdout_cleared_overlays():
    assert (
        round(
            score_ar_binding_overlay(
                {
                    "expected_ar_gain": 0.4,
                    "ar_gain_n": 30,
                    "expected_binding_gain": 0.2,
                    "binding_gain_n": 30,
                    "retention_risk": 0.1,
                    "collapse_risk": 0.05,
                    "holdout_required": False,
                }
            ),
            6,
        )
        == 0.45
    )
    assert score_ar_binding_overlay(empty_overlay()) is None


def test_rerank_graphs_by_ar_binding_tags_graph_metadata():
    class Graph:
        def __init__(self, graph_id: str):
            self.graph_id = graph_id
            self.metadata = {}

        def to_dict(self):
            return {"graph_id": self.graph_id, "nodes": []}

    def overlay_for(graph_dict):
        ar_gain = {"a": 0.1, "b": 0.6, "c": -0.1}[graph_dict["graph_id"]]
        return {
            "expected_ar_gain": ar_gain,
            "ar_gain_n": 30,
            "expected_binding_gain": 0.0,
            "binding_gain_n": 30,
            "retention_risk": 0.0,
            "collapse_risk": 0.0,
            "holdout_required": False,
        }

    graphs, stats = rerank_graphs_by_ar_binding(
        [Graph("a"), Graph("b"), Graph("c")],
        overlay_fn=overlay_for,
    )

    assert [g.graph_id for g in graphs] == ["b", "a", "c"]
    assert stats["used"] is True
    assert graphs[0].metadata["ar_binding_rerank_score"] == 0.6
    assert graphs[0].metadata["ar_binding_overlay"]["expected_ar_gain"] == 0.6


def test_ar_binding_overlay_config_round_trips() -> None:
    config = RunConfig(ar_binding_overlay_enabled=True)
    reconstructed = RunConfig.from_dict(config.to_dict())
    assert reconstructed.ar_binding_overlay_enabled is True
