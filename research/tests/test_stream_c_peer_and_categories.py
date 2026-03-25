"""Tests for Stream C: Refinement & Selection Intelligence.

C2: Nearest-peer retrieval via Jaccard similarity + peer_comparison scorer
C3: Higher-resolution family features (category distribution, routing ops, template)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers — in-memory notebook with program_results + leaderboard
# ---------------------------------------------------------------------------


def _make_graph_json(ops: List[str], template: str = "") -> str:
    """Build a minimal graph_json with the given op names."""
    nodes: Dict[str, Any] = {"input_0": {"op_name": "input", "input_ids": []}}
    prev_id = "input_0"
    for i, op in enumerate(ops):
        nid = f"n{i}"
        nodes[nid] = {"op_name": op, "input_ids": [prev_id]}
        prev_id = nid
    data: Dict[str, Any] = {"nodes": nodes}
    if template:
        data["metadata"] = {"template": template}
    return json.dumps(data)


def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with program_results and leaderboard tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            graph_fingerprint TEXT,
            graph_json TEXT,
            stage0_passed INTEGER DEFAULT 1,
            stage05_passed INTEGER DEFAULT 1,
            stage1_passed INTEGER DEFAULT 0,
            loss_ratio REAL,
            novelty_score REAL,
            novelty_confidence REAL,
            graph_n_ops INTEGER,
            graph_category_histogram TEXT,
            fp_interaction_sparsity REAL,
            fp_cka_vs_transformer REAL,
            fp_cka_vs_ssm REAL,
            fp_cka_vs_conv REAL,
            fingerprint_json TEXT,
            error_type TEXT,
            timestamp REAL DEFAULT 0
        )"""
    )
    conn.execute(
        """CREATE TABLE leaderboard (
            entry_id TEXT PRIMARY KEY,
            result_id TEXT,
            tier TEXT,
            composite_score REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE failure_signatures (
            signature TEXT PRIMARY KEY,
            n_failures INTEGER DEFAULT 0,
            n_successes INTEGER DEFAULT 0,
            error_types TEXT,
            last_updated REAL DEFAULT 0
        )"""
    )
    return conn


def _insert_result(
    conn: sqlite3.Connection,
    result_id: str,
    fingerprint: str,
    ops: List[str],
    stage1: bool = False,
    loss_ratio: float = 0.8,
    novelty: float = 0.5,
    tier: str = "",
    composite: float = 0.0,
    template: str = "",
):
    graph_json = _make_graph_json(ops, template=template)
    conn.execute(
        """INSERT INTO program_results
           (result_id, experiment_id, graph_fingerprint, graph_json,
            stage1_passed, loss_ratio, novelty_score, graph_n_ops, timestamp)
           VALUES (?, 'exp1', ?, ?, ?, ?, ?, ?, 1.0)""",
        (
            result_id,
            fingerprint,
            graph_json,
            int(stage1),
            loss_ratio,
            novelty,
            len(ops),
        ),
    )
    if tier:
        conn.execute(
            """INSERT INTO leaderboard (entry_id, result_id, tier, composite_score)
               VALUES (?, ?, ?, ?)""",
            (f"e_{result_id}", result_id, tier, composite),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# C2: Nearest-peer retrieval
# ---------------------------------------------------------------------------


class TestGetNearestPeers:
    """get_nearest_peers() returns structurally similar fingerprints by Jaccard."""

    @staticmethod
    def _make_notebook(conn):
        """Build a lightweight object with _AnalyticsMixin + _MiscMixin methods and a conn."""
        from research.scientist.notebook.notebook_analytics import _AnalyticsMixin
        from research.scientist.notebook.notebook_misc import _MiscMixin

        class _TestNotebook(_AnalyticsMixin, _MiscMixin):
            pass  # no __slots__ so conn can be set as instance attr

        nb = _TestNotebook()
        nb.conn = conn
        nb._maybe_commit = lambda: None
        return nb

    def test_returns_peers_sorted_by_jaccard(self):
        conn = _create_test_db()
        # Target: ops = {linear_proj, rmsnorm, softmax_attention}
        _insert_result(
            conn,
            "target",
            "fp_target",
            ["linear_proj", "rmsnorm", "softmax_attention"],
            stage1=True,
            loss_ratio=0.5,
        )
        # Peer A: 2/3 overlap → Jaccard = 2/4 = 0.5
        _insert_result(
            conn,
            "peer_a",
            "fp_a",
            ["linear_proj", "rmsnorm", "swiglu_mlp"],
            stage1=True,
            loss_ratio=0.6,
        )
        # Peer B: 1/3 overlap → Jaccard = 1/5 = 0.2
        _insert_result(
            conn,
            "peer_b",
            "fp_b",
            ["linear_proj", "conv1d_seq", "selective_scan"],
            stage1=False,
            loss_ratio=0.9,
        )
        # Peer C: 0/3 overlap → Jaccard = 0, should not appear
        _insert_result(
            conn,
            "peer_c",
            "fp_c",
            ["conv1d_seq", "selective_scan", "lif_neuron"],
            stage1=False,
            loss_ratio=1.0,
        )

        nb = self._make_notebook(conn)
        peers = nb.get_nearest_peers("fp_target", n=5)

        assert len(peers) >= 2
        # Peer A should be most similar
        assert peers[0]["fingerprint"] == "fp_a"
        assert peers[0]["jaccard_similarity"] > peers[1]["jaccard_similarity"]
        # Peer C should not appear (0 overlap)
        fps = [p["fingerprint"] for p in peers]
        assert "fp_c" not in fps

    def test_returns_empty_for_unknown_fingerprint(self):
        conn = _create_test_db()
        nb = self._make_notebook(conn)
        assert nb.get_nearest_peers("nonexistent", n=5) == []

    def test_includes_tier_and_scores(self):
        conn = _create_test_db()
        _insert_result(
            conn,
            "target",
            "fp_target",
            ["linear_proj", "rmsnorm"],
            stage1=True,
            loss_ratio=0.5,
        )
        _insert_result(
            conn,
            "peer",
            "fp_peer",
            ["linear_proj", "rmsnorm", "swiglu_mlp"],
            stage1=True,
            loss_ratio=0.4,
            tier="investigation",
            composite=150.0,
        )

        nb = self._make_notebook(conn)
        peers = nb.get_nearest_peers("fp_target", n=5)

        assert len(peers) == 1
        p = peers[0]
        assert p["tier"] == "investigation"
        assert p["composite_score"] == 150.0
        assert p["loss_ratio"] == 0.4
        assert p["stage1_passed"] is True


# ---------------------------------------------------------------------------
# C2: Peer comparison scorer in judgment
# ---------------------------------------------------------------------------


class TestPeerComparisonScorer:
    """_score_peer_comparison rewards similarity to successful peers,
    penalizes similarity to failed ones."""

    def test_rewards_successful_peers(self):
        from research.scientist.judgment import (
            _score_peer_comparison,
            JudgmentContext,
        )

        signals = {
            "nearest_peers": [
                {
                    "fingerprint": "fp_good",
                    "jaccard_similarity": 0.8,
                    "stage1_passed": True,
                    "loss_ratio": 0.3,
                    "novelty_score": 0.7,
                    "tier": "investigation",
                    "composite_score": 200.0,
                },
            ],
        }
        ctx = JudgmentContext()
        delta, confidence, evidence = _score_peer_comparison({}, ctx, signals)
        assert delta > 0, f"Expected positive delta for successful peer, got {delta}"

    def test_penalizes_failed_peers(self):
        from research.scientist.judgment import (
            _score_peer_comparison,
            JudgmentContext,
        )

        signals = {
            "nearest_peers": [
                {
                    "fingerprint": "fp_bad",
                    "jaccard_similarity": 0.9,
                    "stage1_passed": False,
                    "loss_ratio": None,
                    "novelty_score": None,
                    "tier": "",
                    "composite_score": None,
                },
            ],
        }
        ctx = JudgmentContext()
        delta, confidence, evidence = _score_peer_comparison({}, ctx, signals)
        assert delta < 0, f"Expected negative delta for failed peer, got {delta}"

    def test_empty_peers_returns_zero(self):
        from research.scientist.judgment import (
            _score_peer_comparison,
            JudgmentContext,
        )

        ctx = JudgmentContext()
        delta, confidence, evidence = _score_peer_comparison({}, ctx, {})
        assert delta == 0.0
        assert confidence == 0.0
        assert evidence == []


# ---------------------------------------------------------------------------
# C3: Higher-resolution bucket features
# ---------------------------------------------------------------------------


class TestBucketCategoryDistribution:
    """get_fingerprint_buckets() returns op_category_distribution, top_routing_ops,
    and template_signature alongside existing fields."""

    @staticmethod
    def _make_notebook(conn):
        from research.scientist.notebook.notebook_analytics import _AnalyticsMixin
        from research.scientist.notebook.notebook_misc import _MiscMixin

        class _TestNotebook(_AnalyticsMixin, _MiscMixin):
            pass

        nb = _TestNotebook()
        nb.conn = conn
        nb._maybe_commit = lambda: None
        return nb

    def test_bucket_has_new_fields(self):
        conn = _create_test_db()
        _insert_result(
            conn,
            "r1",
            "fp1",
            ["linear_proj", "rmsnorm", "softmax_attention", "moe_topk"],
            stage1=True,
            loss_ratio=0.5,
            template="moe_router",
        )
        _insert_result(
            conn,
            "r2",
            "fp2",
            ["linear_proj", "rmsnorm", "moe_2expert"],
            stage1=False,
            loss_ratio=0.9,
            template="moe_router",
        )

        nb = self._make_notebook(conn)
        buckets = nb.get_fingerprint_buckets(limit=10)

        assert len(buckets) > 0
        b = buckets[0]
        # Existing fields still present
        assert "bucket" in b
        assert "s1_rate" in b
        assert "n_graphs" in b
        assert "top_ops" in b
        # New fields
        assert "op_category_distribution" in b
        assert isinstance(b["op_category_distribution"], dict)
        assert len(b["op_category_distribution"]) == 11  # all OpCategory values
        # Values sum to ~1.0
        total = sum(b["op_category_distribution"].values())
        assert 0.99 <= total <= 1.01, f"Category distribution sums to {total}"

        assert "top_routing_ops" in b
        assert isinstance(b["top_routing_ops"], list)

        assert "template_signature" in b

    def test_routing_ops_extracted(self):
        conn = _create_test_db()
        # Insert graphs with routing ops
        _insert_result(
            conn,
            "r1",
            "fp1",
            ["moe_topk", "moe_topk", "linear_proj", "moe_2expert"],
            stage1=True,
            loss_ratio=0.5,
        )

        nb = self._make_notebook(conn)
        buckets = nb.get_fingerprint_buckets(limit=10)
        assert len(buckets) > 0
        routing = buckets[0]["top_routing_ops"]
        # moe_topk and moe_2expert should be detected
        assert "moe_topk" in routing or "moe_2expert" in routing

    def test_template_signature_from_metadata(self):
        conn = _create_test_db()
        _insert_result(conn, "r1", "fp1", ["linear_proj"], template="my_template")
        _insert_result(conn, "r2", "fp2", ["linear_proj"], template="my_template")
        _insert_result(conn, "r3", "fp3", ["linear_proj"], template="other")

        nb = self._make_notebook(conn)
        buckets = nb.get_fingerprint_buckets(limit=10)
        # At least one bucket should have "my_template" as signature (most common)
        templates = [b["template_signature"] for b in buckets]
        assert any(t == "my_template" for t in templates)


# ---------------------------------------------------------------------------
# C3: Category distribution used in judgment scoring
# ---------------------------------------------------------------------------


class TestCategoryDistributionScoring:
    """_score_fingerprint_bucket uses op_category_distribution for scoring."""

    def test_diverse_categories_get_bonus(self):
        from research.scientist.judgment import (
            _score_fingerprint_bucket,
            JudgmentContext,
        )

        # Bucket with diverse categories: 5 active categories
        signals = {
            "fingerprint_buckets": [
                {
                    "bucket": "hybrid",
                    "s1_rate": 0.5,  # baseline delta = (0.5-0.3)*1.5 = 0.3
                    "n_graphs": 50,
                    "top_routing_ops": ["moe_topk"],
                    "op_category_distribution": {
                        "parameterized": 0.25,
                        "mixing": 0.20,
                        "sequence": 0.20,
                        "functional": 0.15,
                        "reduction": 0.10,
                        "elementwise_unary": 0.04,
                        "elementwise_binary": 0.03,
                        "linear_algebra": 0.02,
                        "structural": 0.01,
                        "frequency": 0.0,
                        "math_space": 0.0,
                    },
                    "template_signature": "test",
                },
            ],
        }
        ctx = JudgmentContext(fingerprint_bucket="hybrid")
        delta_diverse, _, _ = _score_fingerprint_bucket({}, ctx, signals)

        # Same bucket but dominated by one category
        signals_dominated = {
            "fingerprint_buckets": [
                {
                    "bucket": "hybrid",
                    "s1_rate": 0.5,
                    "n_graphs": 50,
                    "top_routing_ops": ["moe_topk"],
                    "op_category_distribution": {
                        "elementwise_unary": 0.75,
                        "parameterized": 0.10,
                        "mixing": 0.05,
                        "sequence": 0.04,
                        "functional": 0.03,
                        "reduction": 0.02,
                        "elementwise_binary": 0.01,
                        "linear_algebra": 0.0,
                        "structural": 0.0,
                        "frequency": 0.0,
                        "math_space": 0.0,
                    },
                    "template_signature": "test",
                },
            ],
        }
        delta_dominated, _, _ = _score_fingerprint_bucket({}, ctx, signals_dominated)

        assert delta_diverse > delta_dominated, (
            f"Diverse categories ({delta_diverse}) should score higher "
            f"than dominated ({delta_dominated})"
        )
