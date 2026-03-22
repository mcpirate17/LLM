"""Tests for the under-observed component exploration script."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from research.tools.explore_under_observed import (
    OpCoverage,
    ExplorationResult,
    _build_summary,
    _find_motifs_containing_op,
    _make_config_for_op,
    _ops_in_graph,
    discover_targets,
    evaluate_graph,
    generate_forced_graph,
    generate_weighted_batch,
    run_exploration,
    update_coverage,
    write_reports,
)
from research.synthesis.grammar import GrammarConfig, generate_layer_graph


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DB with op_success_rates table."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE op_success_rates (
            op_name TEXT PRIMARY KEY,
            n_used INTEGER DEFAULT 0,
            n_stage0_passed INTEGER DEFAULT 0,
            n_stage05_passed INTEGER DEFAULT 0,
            n_stage1_passed INTEGER DEFAULT 0,
            avg_loss_ratio REAL,
            avg_novelty REAL,
            avg_novelty_confidence REAL,
            last_updated REAL
        )
    """)
    # Insert some ops with varying observation counts
    ops = [
        ("linear_proj", 500),
        ("rmsnorm", 400),
        ("gelu", 350),
        ("softmax_attention", 200),
        ("layernorm", 15),  # under threshold
        ("silu", 8),  # under threshold
    ]
    for name, n in ops:
        conn.execute(
            "INSERT INTO op_success_rates (op_name, n_used) VALUES (?, ?)",
            (name, n),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_db(tmp_path):
    """DB with the table but no rows."""
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE op_success_rates (
            op_name TEXT PRIMARY KEY,
            n_used INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    return db_path


# ── Target discovery ─────────────────────────────────────────────────


class TestDiscoverTargets:
    def test_finds_under_observed(self, tmp_db):
        targets = discover_targets(tmp_db, threshold=20)
        assert "layernorm" in targets
        assert "silu" in targets
        assert targets["layernorm"] == 15
        assert targets["silu"] == 8

    def test_excludes_well_observed(self, tmp_db):
        targets = discover_targets(tmp_db, threshold=20)
        assert "linear_proj" not in targets
        assert "rmsnorm" not in targets

    def test_includes_zero_observation_ops(self, empty_db):
        """Ops in PRIMITIVE_REGISTRY with no DB rows should appear as targets."""
        targets = discover_targets(empty_db, threshold=20)
        # Should include many ops from PRIMITIVE_REGISTRY
        assert len(targets) > 10
        # All should have n_used=0
        assert all(n == 0 for n in targets.values())

    def test_nonexistent_db(self, tmp_path):
        """Non-existent DB should treat all ops as zero-observation."""
        targets = discover_targets(str(tmp_path / "missing.db"), threshold=20)
        assert len(targets) > 10

    def test_threshold_zero(self, tmp_db):
        """Threshold=0 should return no targets (0 < 0 is False)."""
        targets = discover_targets(tmp_db, threshold=0)
        assert len(targets) == 0

    def test_skips_pseudo_ops(self, empty_db):
        """Pseudo-ops like 'input', 'add' should not appear."""
        targets = discover_targets(empty_db, threshold=20)
        for skip in ("input", "output", "add", "concat", "split"):
            assert skip not in targets


# ── Graph ops extraction ─────────────────────────────────────────────


class TestOpsInGraph:
    def test_extracts_ops(self):
        g = generate_layer_graph(GrammarConfig(composition_depth=1), seed=42)
        ops = _ops_in_graph(g)
        assert len(ops) > 0
        assert "input" not in ops


# ── Config generation ────────────────────────────────────────────────


class TestMakeConfigForOp:
    def test_sets_exploration_targets(self):
        cfg = _make_config_for_op("linear_proj")
        assert "linear_proj" in cfg.exploration_targets

    def test_uses_exploration_boost(self):
        cfg = _make_config_for_op("linear_proj")
        assert cfg.exploration_boost_factor >= 4.0

    def test_unknown_op_returns_default(self):
        cfg = _make_config_for_op("nonexistent_op_xyz")
        assert isinstance(cfg, GrammarConfig)


# ── Motif lookup ─────────────────────────────────────────────────────


class TestFindMotifsContainingOp:
    def test_finds_motifs_for_common_op(self):
        motifs = _find_motifs_containing_op("linear_proj")
        assert len(motifs) > 0

    def test_no_motifs_for_unknown_op(self):
        motifs = _find_motifs_containing_op("nonexistent_op_xyz")
        assert motifs == []


# ── Forced graph generation ──────────────────────────────────────────


class TestGenerateForcedGraph:
    def test_generates_graph_with_common_op(self):
        graph, retries = generate_forced_graph("linear_proj", seed=42, max_retries=10)
        assert graph is not None
        assert "linear_proj" in _ops_in_graph(graph)

    def test_generates_graph_with_rmsnorm(self):
        graph, retries = generate_forced_graph("rmsnorm", seed=42, max_retries=10)
        assert graph is not None
        assert "rmsnorm" in _ops_in_graph(graph)

    def test_returns_none_for_impossible_op(self):
        graph, retries = generate_forced_graph(
            "nonexistent_op_xyz", seed=42, max_retries=3
        )
        assert graph is None
        assert retries == 3


# ── Weighted batch generation ────────────────────────────────────────


class TestGenerateWeightedBatch:
    def test_generates_requested_count(self):
        targets = {"linear_proj": 5, "rmsnorm": 3}
        graphs = generate_weighted_batch(targets, n_graphs=5, base_seed=42)
        assert len(graphs) <= 5
        assert len(graphs) > 0

    def test_boosted_ops_appear_more(self):
        targets = {"linear_proj": 0, "rmsnorm": 0, "gelu": 0}
        graphs = generate_weighted_batch(targets, n_graphs=10, base_seed=42)
        hit_count = sum(1 for g in graphs if set(targets.keys()) & _ops_in_graph(g))
        # At least some graphs should contain target ops
        assert hit_count > 0


# ── Coverage tracking ────────────────────────────────────────────────


class TestUpdateCoverage:
    def test_updates_on_compile_pass(self):
        coverage = {"op_a": OpCoverage("op_a", 5)}
        graph = generate_layer_graph(seed=42)
        present = _ops_in_graph(graph)
        # Pick a real op from the graph
        op_name = next(iter(present))
        coverage = {op_name: OpCoverage(op_name, 5)}
        result = ExplorationResult(
            graph_fingerprint="abc",
            target_ops=[op_name],
            ops_present=list(present),
            compile_ok=True,
            forward_ok=True,
            rapid_ok=True,
            s1_ok=False,
        )
        update_coverage(coverage, graph, result, {op_name})
        assert coverage[op_name].inserted == 1
        assert coverage[op_name].compile_pass == 1
        assert coverage[op_name].forward_pass == 1
        assert coverage[op_name].rapid_pass == 1
        assert coverage[op_name].s1_fail == 1

    def test_skips_ops_not_in_graph(self):
        coverage = {"nonexistent_xyz": OpCoverage("nonexistent_xyz", 0)}
        graph = generate_layer_graph(seed=42)
        result = ExplorationResult(
            graph_fingerprint="abc",
            target_ops=["nonexistent_xyz"],
            ops_present=[],
            compile_ok=True,
        )
        update_coverage(coverage, graph, result, {"nonexistent_xyz"})
        assert coverage["nonexistent_xyz"].inserted == 0


# ── Summary builder ──────────────────────────────────────────────────


class TestBuildSummary:
    def test_summary_counts(self):
        cov = {
            "a": OpCoverage(
                "a",
                5,
                inserted=1,
                compile_pass=1,
                forward_pass=1,
                rapid_pass=1,
                s1_pass=1,
            ),
            "b": OpCoverage("b", 3, inserted=1, compile_pass=1, forward_pass=0),
            "c": OpCoverage("c", 0, inserted=0),
        }
        s = _build_summary(cov)
        assert s["n_targets"] == 3
        assert s["n_covered"] == 2
        assert s["n_compile_pass"] == 2
        assert s["n_forward_pass"] == 1
        assert s["n_s1_pass"] == 1
        assert s["coverage_rate"] == pytest.approx(2 / 3)


# ── Report writing ───────────────────────────────────────────────────


class TestWriteReports:
    def test_writes_both_formats(self, tmp_path):
        coverage = {
            "op_a": OpCoverage("op_a", 5, inserted=1, compile_pass=1),
        }
        results = [
            ExplorationResult(
                graph_fingerprint="fp1",
                target_ops=["op_a"],
                ops_present=["op_a", "linear_proj"],
                compile_ok=True,
            ),
        ]
        md, js = write_reports(
            coverage,
            results,
            str(tmp_path),
            "forced",
            20,
            10.0,
        )
        assert os.path.exists(md)
        assert os.path.exists(js)

        with open(js) as f:
            data = json.load(f)
        assert data["mode"] == "forced"
        assert data["n_targets"] == 1

        with open(md) as f:
            text = f.read()
        assert "op_a" in text
        assert "forced" in text


# ── End-to-end dry run ───────────────────────────────────────────────


class TestEndToEndDryRun:
    def test_forced_dry_run(self, tmp_db, tmp_path):
        coverage, results = run_exploration(
            db_path=tmp_db,
            mode="forced",
            threshold=20,
            device="cpu",
            max_retries_forced=5,
            run_s1=False,
            output_dir=str(tmp_path / "reports"),
            base_seed=42,
            dry_run=True,
        )
        # Should have found layernorm (15) and silu (8) as under-observed
        # Plus any ops in PRIMITIVE_REGISTRY with 0 observations in this DB
        assert len(coverage) > 0
        # Check at least some were covered
        covered = sum(1 for c in coverage.values() if c.inserted > 0)
        assert covered > 0

    def test_weighted_dry_run(self, tmp_db, tmp_path):
        coverage, results = run_exploration(
            db_path=tmp_db,
            mode="weighted",
            threshold=20,
            device="cpu",
            n_graphs_weighted=5,
            max_retries_forced=3,
            run_s1=False,
            output_dir=str(tmp_path / "reports"),
            base_seed=42,
            dry_run=True,
        )
        assert len(coverage) > 0
        assert len(results) > 0

    def test_reports_generated(self, tmp_db, tmp_path):
        out = str(tmp_path / "reports")
        run_exploration(
            db_path=tmp_db,
            mode="forced",
            threshold=20,
            device="cpu",
            max_retries_forced=3,
            run_s1=False,
            output_dir=out,
            base_seed=42,
            dry_run=True,
        )
        # Check reports exist
        files = os.listdir(out)
        md_files = [f for f in files if f.endswith(".md")]
        json_files = [f for f in files if f.endswith(".json")]
        assert len(md_files) == 1
        assert len(json_files) == 1


# ── Pipeline evaluation (CPU, no S1) ────────────────────────────────


class TestEvaluateGraph:
    def test_compile_and_forward_cpu(self):
        """Verify a simple graph compiles and passes forward on CPU."""
        graph = generate_layer_graph(
            GrammarConfig(composition_depth=1, max_ops=8),
            seed=42,
        )
        result = evaluate_graph(graph, device="cpu", run_s1=False)
        # At minimum it should compile
        assert result.compile_ok, f"compile failed: {result.compile_error}"
        assert result.param_count > 0
        assert len(result.ops_present) > 0

    def test_forward_pass_basic(self):
        """Multiple seeds to verify forward pass works."""
        passed = 0
        seed = 0
        attempts = 0
        while attempts < 5 and seed < 20:
            try:
                graph = generate_layer_graph(
                    GrammarConfig(composition_depth=1, max_ops=12),
                    seed=seed,
                )
                result = evaluate_graph(graph, device="cpu", run_s1=False)
                attempts += 1
                if result.forward_ok:
                    passed += 1
            except ValueError:
                pass  # Grammar validation rejection — try next seed
            seed += 1
        # At least some should pass forward
        assert passed > 0, "No graphs passed forward on CPU"
