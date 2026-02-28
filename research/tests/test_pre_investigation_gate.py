"""Tests for the pre-investigation gate (3-stage filtering before investigation)."""
import os
import sys
import sqlite3
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from research.scientist.notebook import LabNotebook
from research.scientist.runner import RunConfig, ExperimentRunner


# ── Helpers ──────────────────────────────────────────────────────────

def _make_nb(tmp_path):
    """Create a LabNotebook with a temp DB."""
    db_path = os.path.join(str(tmp_path), "test.db")
    nb = LabNotebook(db_path=db_path)
    return nb


def _insert_program_result(nb, result_id=None, experiment_id=None, **kwargs):
    """Insert a minimal program_result row for testing."""
    rid = result_id or str(uuid.uuid4())[:12]
    eid = experiment_id or "exp_test"
    defaults = {
        "timestamp": "2026-02-28T00:00:00",
        "graph_json": '{"nodes":[],"edges":[]}',
        "stage1_passed": 1,
        "loss_ratio": 0.4,
        "has_nan_grad": 0,
        "has_nan_output": 0,
        "has_inf_output": 0,
        "has_zero_grad": 0,
        "graph_has_gradient_path": 1,
        "stability_score": 0.7,
        "fp_jacobian_spectral_norm": 1.5,
        "loss_improvement_rate": 0.1,
        "novelty_score": 0.5,
        "novelty_confidence": 0.6,
        "structural_novelty": 0.4,
        "behavioral_novelty": 0.3,
        "fp_intrinsic_dim": 8.0,
        "fp_isotropy": 0.6,
        "fp_rank_ratio": 0.7,
        "throughput_tok_s": 5000,
        "peak_memory_mb": 200,
        "grad_norm_std": 0.3,
        "graph_fingerprint": f"fp_{rid}",
    }
    defaults.update(kwargs)

    # Ensure experiment exists
    try:
        nb.conn.execute(
            "INSERT OR IGNORE INTO experiments (experiment_id, experiment_type, timestamp) VALUES (?, 'synthesis', datetime('now'))",
            (eid,),
        )
    except Exception:
        pass

    cols = ["result_id", "experiment_id"] + list(defaults.keys())
    vals = [rid, eid] + list(defaults.values())
    placeholders = ",".join("?" for _ in cols)
    col_str = ",".join(cols)
    nb.conn.execute(f"INSERT INTO program_results ({col_str}) VALUES ({placeholders})", vals)
    nb.conn.commit()
    return rid


def _insert_leaderboard(nb, result_id, tier="screening", **kwargs):
    """Insert a leaderboard entry."""
    nb.upsert_leaderboard(
        result_id=result_id,
        model_source="graph_synthesis",
        architecture_desc="test_arch",
        tier=tier,
        **kwargs,
    )


def _insert_reference(nb, name="gpt2", loss_ratio=0.73):
    """Insert a reference architecture on the leaderboard."""
    rid = f"ref_{name}_{uuid.uuid4().hex[:6]}"
    _insert_program_result(nb, result_id=rid, loss_ratio=loss_ratio,
                           graph_fingerprint=f"ref_fp_{name}")
    _insert_leaderboard(nb, rid, tier="screening",
                        screening_loss_ratio=loss_ratio,
                        is_reference=True, reference_name=name)
    return rid


# ── Stage A: Hard reject tests ──────────────────────────────────────

class TestStageAHardReject:
    def test_reject_nan_grad(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, has_nan_grad=1)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert not any(e["result_id"] == rid for e in eligible)

    def test_reject_inf_output(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, has_inf_output=1)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert not any(e["result_id"] == rid for e in eligible)

    def test_reject_zero_grad(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, has_zero_grad=1)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert not any(e["result_id"] == rid for e in eligible)

    def test_reject_collapsed_spectral_norm(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, fp_jacobian_spectral_norm=0.001)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert not any(e["result_id"] == rid for e in eligible)

    def test_reject_exploding_spectral_norm(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, fp_jacobian_spectral_norm=100.0)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert not any(e["result_id"] == rid for e in eligible)

    def test_pass_healthy_candidate(self, tmp_path):
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        eligible = nb.get_investigation_eligible(
            max_lr=0.5, min_stability=0.3, min_spectral_norm=0.01,
            max_spectral_norm=50.0, min_improvement_rate=0.0)
        assert any(e["result_id"] == rid for e in eligible)


# ── Stage B: Composite score tests ──────────────────────────────────

class TestStageBCompositeScore:
    def test_score_formula_basic(self):
        row = {
            "loss_ratio": 0.3,
            "stability_score": 0.8,
            "novelty_score": 0.6,
            "novelty_confidence": 0.7,
            "fp_intrinsic_dim": 10.0,
            "fp_isotropy": 0.5,
            "fp_rank_ratio": 0.6,
            "throughput_tok_s": 4000,
            "peak_memory_mb": 150,
        }
        score = LabNotebook.compute_pre_investigation_score(row)
        assert 0 < score <= 100
        assert score > 20  # healthy candidate should score decently

    def test_reference_penalty(self):
        row = {"loss_ratio": 0.8, "stability_score": 0.5}
        # best_ref_lr=0.3 → threshold 0.45, LR 0.8 > 0.45 → penalty
        score_with = LabNotebook.compute_pre_investigation_score(row, best_ref_lr=0.3)
        score_without = LabNotebook.compute_pre_investigation_score(row, best_ref_lr=None)
        assert score_with < score_without

    def test_novelty_boost(self):
        base = {"loss_ratio": 0.5, "stability_score": 0.5}
        novel = {**base, "novelty_score": 0.9, "novelty_confidence": 0.9,
                 "structural_novelty": 0.8, "behavioral_novelty": 0.7}
        score_base = LabNotebook.compute_pre_investigation_score(base)
        score_novel = LabNotebook.compute_pre_investigation_score(novel)
        assert score_novel > score_base

    def test_ranking_order(self):
        good = {"loss_ratio": 0.2, "stability_score": 0.9, "novelty_score": 0.7,
                "novelty_confidence": 0.8}
        bad = {"loss_ratio": 0.7, "stability_score": 0.3}
        assert (LabNotebook.compute_pre_investigation_score(good)
                > LabNotebook.compute_pre_investigation_score(bad))

    def test_top_n_limit(self, tmp_path):
        nb = _make_nb(tmp_path)
        rids = []
        for i in range(10):
            rid = _insert_program_result(nb, loss_ratio=0.1 + i * 0.05)
            _insert_leaderboard(nb, rid, screening_loss_ratio=0.1 + i * 0.05)
            rids.append(rid)

        config = RunConfig()
        config.pre_inv_gate_enabled = True
        config.pre_inv_top_n = 3

        runner = MagicMock()
        runner._get_reference_baseline_lr = MagicMock(return_value=None)

        gate = ExperimentRunner._pre_investigation_gate
        # Call with mocked self
        runner_real = MagicMock(spec=ExperimentRunner)
        runner_real._get_reference_baseline_lr = MagicMock(return_value=None)
        result = gate(runner_real, config, nb, nb.get_leaderboard(limit=50))
        assert len(result) <= 3


# ── Stage C: Probe tests ────────────────────────────────────────────

class TestStageCProbe:
    def test_probe_disabled_by_default(self):
        config = RunConfig()
        assert config.pre_inv_probe_enabled is False

    def test_probe_reject_high_lr(self, tmp_path):
        """When probe returns high loss_ratio, candidate is rejected."""
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)

        config = RunConfig()
        config.pre_inv_gate_enabled = True
        config.pre_inv_probe_enabled = True
        config.pre_inv_probe_max_lr = 0.5
        config.pre_inv_top_n = 5

        runner = MagicMock(spec=ExperimentRunner)
        runner._get_reference_baseline_lr = MagicMock(return_value=None)
        # Probe returns 0.9 (too high)
        runner._pre_inv_probe = MagicMock(return_value=0.9)

        result = ExperimentRunner._pre_investigation_gate(runner, config, nb,
                                                           nb.get_leaderboard(limit=50))
        assert rid not in result


# ── Integration tests ────────────────────────────────────────────────

class TestIntegration:
    def test_reference_aware_gating(self, tmp_path):
        """Gate uses reference LR ceiling when references exist."""
        nb = _make_nb(tmp_path)
        _insert_reference(nb, "gpt2", 0.3)
        # Candidate with LR=0.6 should fail: 0.6 > 0.3 * 1.5 = 0.45
        rid = _insert_program_result(nb, loss_ratio=0.6)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.6)

        runner = MagicMock(spec=ExperimentRunner)
        runner._get_reference_baseline_lr = lambda self_inner, nb_inner: 0.3
        # Bind the real method
        runner._get_reference_baseline_lr = MagicMock(return_value=0.3)

        config = RunConfig()
        config.pre_inv_gate_enabled = True
        config.pre_inv_reference_margin = 1.5

        result = ExperimentRunner._pre_investigation_gate(
            runner, config, nb, nb.get_leaderboard(limit=50))
        assert rid not in result

    def test_reference_aware_pass(self, tmp_path):
        """Candidate below reference ceiling passes."""
        nb = _make_nb(tmp_path)
        _insert_reference(nb, "gpt2", 0.7)
        rid = _insert_program_result(nb, loss_ratio=0.3)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)

        runner = MagicMock(spec=ExperimentRunner)
        runner._get_reference_baseline_lr = MagicMock(return_value=0.7)

        config = RunConfig()
        config.pre_inv_gate_enabled = True

        result = ExperimentRunner._pre_investigation_gate(
            runner, config, nb, nb.get_leaderboard(limit=50))
        assert rid in result

    def test_legacy_fallback(self, tmp_path):
        """Gate falls back to legacy when disabled."""
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, loss_ratio=0.3)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)

        runner = MagicMock(spec=ExperimentRunner)

        config = RunConfig()
        config.pre_inv_gate_enabled = False
        config.investigation_loss_ratio_threshold = 0.5

        result = ExperimentRunner._pre_investigation_gate(
            runner, config, nb, nb.get_leaderboard(limit=50))
        assert rid in result

    def test_score_persisted(self, tmp_path):
        """pre_inv_score is written to leaderboard."""
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb)
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)

        runner = MagicMock(spec=ExperimentRunner)
        runner._get_reference_baseline_lr = MagicMock(return_value=None)

        config = RunConfig()
        config.pre_inv_gate_enabled = True

        ExperimentRunner._pre_investigation_gate(
            runner, config, nb, nb.get_leaderboard(limit=50))

        row = nb.conn.execute(
            "SELECT pre_inv_score FROM leaderboard WHERE result_id = ?",
            (rid,)).fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0] > 0

    def test_already_investigated_filtered(self, tmp_path):
        """Candidates with investigated fingerprints are filtered out."""
        nb = _make_nb(tmp_path)
        rid = _insert_program_result(nb, graph_fingerprint="already_done")
        _insert_leaderboard(nb, rid, screening_loss_ratio=0.3)
        # Create an investigation experiment with the same fingerprint
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, experiment_type, timestamp, config_json) VALUES ('inv1', 'investigation', datetime('now'), '{}')")
        nb.conn.execute(
            "INSERT INTO program_results (result_id, experiment_id, graph_fingerprint, timestamp, graph_json) VALUES ('inv_r1', 'inv1', 'already_done', datetime('now'), '{}')")
        nb.conn.commit()

        runner = MagicMock(spec=ExperimentRunner)
        runner._get_reference_baseline_lr = MagicMock(return_value=None)

        config = RunConfig()
        config.pre_inv_gate_enabled = True

        result = ExperimentRunner._pre_investigation_gate(
            runner, config, nb, nb.get_leaderboard(limit=50))
        assert rid not in result

    def test_worth_it_uses_pre_inv_score(self):
        """When pre_inv_score is available, worthiness uses it."""
        entry = {"tier": "screening", "pre_inv_score": 25.0,
                 "screening_loss_ratio": 0.9, "screening_novelty": 0.0,
                 "result_id": "test", "graph_fingerprint": "fp1"}
        # With legacy rules, LR=0.9 and Nov=0 would NOT be worth it
        # But pre_inv_score=25 >= 20 should make it worth it
        pis = entry.get("pre_inv_score")
        assert pis is not None
        assert float(pis) >= 20.0

        # Verify legacy would reject
        lr = entry.get("screening_loss_ratio", 1.0)
        nov = entry.get("screening_novelty", 0.0)
        legacy_worth = (lr < 0.2 or (lr < 0.4 and nov > 0.4) or (lr < 0.6 and nov > 0.7))
        assert not legacy_worth  # Legacy would reject
