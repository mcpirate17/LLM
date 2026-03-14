"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


try:
    from research.scientist.notebook import LabNotebook, ExperimentEntry
    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    from research.scientist.persona import Aria
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    import research.scientist.llm.prompts as _prompts_mod  # noqa: F401
    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    import research.scientist.llm.context as _context_mod  # noqa: F401
    HAS_CONTEXT = True
except Exception as e:
    HAS_CONTEXT = False
    print(f"Context import failed: {e}")


@unittest.skipUnless(HAS_TORCH, "requires torch for graph/metrics modules")
class TestNoveltyScoring(unittest.TestCase):
    """Test that novelty scoring no longer always returns 1.0."""

    def _make_graph(self, n_ops=5, op_names=None):
        """Create a simple computation graph for testing."""
        from research.synthesis.graph import ComputationGraph
        graph = ComputationGraph(model_dim=256)

        # Add input node
        input_id = graph.add_input()

        if op_names is None:
            op_names = ["relu", "gelu", "tanh", "sigmoid", "silu"]

        prev_id = input_id
        for op_name in op_names[:n_ops]:
            prev_id = graph.add_op(op_name, [prev_id])

        graph.set_output(prev_id)
        return graph

    def test_structural_novelty_not_always_one(self):
        """Structural novelty should NOT be ~1.0 for every graph."""
        from research.eval.metrics import novelty_score

        # Simple graph with few unique ops (all the same)
        simple_graph = self._make_graph(
            op_names=["relu", "relu", "relu", "relu", "relu"])

        # Diverse graph with many unique ops
        diverse_graph = self._make_graph(
            op_names=["relu", "gelu", "tanh", "sigmoid", "silu"])

        simple_nov = novelty_score(simple_graph)
        diverse_nov = novelty_score(diverse_graph)

        # Neither should be 1.0 (the old bug)
        self.assertLess(simple_nov.structural_novelty, 0.95,
                        "Simple graph should NOT have max novelty")

        # Diverse should be higher than simple
        self.assertGreater(diverse_nov.structural_novelty,
                           simple_nov.structural_novelty,
                           "Diverse graph should have higher novelty than simple")

    def test_no_fingerprint_discount(self):
        """Without behavioral fingerprint, overall_novelty should be discounted."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        nov = novelty_score(graph, fingerprint=None)

        # Should be discounted (0.6x structural)
        expected_max = nov.structural_novelty * 0.6 + 0.01  # small tolerance
        self.assertLessEqual(nov.overall_novelty, expected_max,
                             "No-fingerprint novelty should be discounted")

    def test_with_fingerprint_uses_behavioral(self):
        """With fingerprint, overall should use 70% behavioral weight."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint

        graph = self._make_graph()
        fp = BehavioralFingerprint(
            novelty_score=0.9,
            cka_vs_transformer=0.3,
            cka_vs_ssm=0.2,
            cka_vs_conv=0.1,
        )

        nov = novelty_score(graph, fingerprint=fp)

        # raw_novelty should be weighted: 0.3 * structural + 0.7 * behavioral
        # overall_novelty = raw_novelty * CKA reference penalty
        expected_raw = 0.3 * nov.structural_novelty + 0.7 * 0.9
        self.assertAlmostEqual(nov.raw_novelty, expected_raw, places=2)
        # overall_novelty should be <= raw_novelty (penalty is <= 1.0)
        self.assertLessEqual(nov.overall_novelty, expected_raw + 0.01)

    def test_behavior_signature_contributes_to_fingerprint_novelty(self):
        """Fingerprint novelty is not just 1 - max(CKA)."""
        from research.eval.fingerprint import BehavioralFingerprint, _blend_behavioral_novelty

        fp = BehavioralFingerprint(
            cka_vs_transformer=0.3,
            cka_vs_ssm=0.2,
            cka_vs_conv=0.1,
            interaction_locality=1.0,
            interaction_sparsity=0.0,
            interaction_symmetry=1.0,
            interaction_hierarchy=0.0,
            isotropy=1.0,
            rank_ratio=0.0,
            sensitivity_uniformity=1.0,
            hierarchy_fitness=1.0,
        )
        expected_cka_only = 0.7
        blended = _blend_behavioral_novelty(fp)
        self.assertGreater(blended, expected_cka_only)
        self.assertLessEqual(blended, 1.0)

    def test_duplicate_penalization(self):
        """Exact duplicate fingerprints should be penalized."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        fp_str = graph.fingerprint()

        nov_fresh = novelty_score(graph, known_fingerprints=[])
        nov_dup = novelty_score(graph, known_fingerprints=[fp_str])

        self.assertGreater(nov_fresh.overall_novelty,
                           nov_dup.overall_novelty,
                           "Duplicate should be penalized")


class TestNoveltyCalibration(unittest.TestCase):
    """Regression tests for novelty confidence/quality tracking (#4, #10)."""

    def test_fingerprint_quality_defaults(self):
        """BehavioralFingerprint defaults to quality='none', analyses_succeeded=0."""
        from research.eval.fingerprint import BehavioralFingerprint

        fp = BehavioralFingerprint()
        self.assertEqual(fp.quality, "none")
        self.assertEqual(fp.analyses_succeeded, 0)

    def test_novelty_confidence_defaults(self):
        """NoveltyMetrics defaults to novelty_confidence=0.0."""
        from research.eval.metrics import NoveltyMetrics

        nm = NoveltyMetrics()
        self.assertEqual(nm.novelty_confidence, 0.0)

    def test_confidence_no_fingerprint(self):
        """Without fingerprint, novelty_confidence should be 0.2."""
        from research.eval.metrics import novelty_score
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        nov = novelty_score(graph, fingerprint=None)
        self.assertAlmostEqual(nov.novelty_confidence, 0.2)

    def test_confidence_full_quality_fingerprint(self):
        """Full-quality fingerprint gives confidence=0.9."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(
            novelty_score=0.7, quality="full", analyses_succeeded=4,
        )
        nov = novelty_score(graph, fingerprint=fp)
        self.assertAlmostEqual(nov.novelty_confidence, 0.9)

    def test_confidence_partial_quality_fingerprint(self):
        """Partial-quality fingerprint gives confidence=0.4 + n*0.1."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        for n in (1, 2, 3):
            fp = BehavioralFingerprint(
                novelty_score=0.5, quality="partial", analyses_succeeded=n,
            )
            nov = novelty_score(graph, fingerprint=fp)
            expected = 0.4 + n * 0.1
            self.assertAlmostEqual(nov.novelty_confidence, expected,
                                   msg=f"analyses_succeeded={n}")

    def test_confidence_none_quality_with_fingerprint(self):
        """quality='none' but fingerprint provided gives confidence=0.3."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(novelty_score=0.5, quality="none",
                                   analyses_succeeded=0)
        nov = novelty_score(graph, fingerprint=fp)
        self.assertAlmostEqual(nov.novelty_confidence, 0.3)

    def test_heuristic_fingerprint_not_valid_for_promotion_by_default(self):
        """Heuristic novelty should require override policy downstream."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(
            novelty_score=0.6,
            cka_source="heuristic_fallback",
            novelty_valid_for_promotion=False,
            novelty_validity_reason="heuristic_fallback_reference",
        )
        nov = novelty_score(graph, fingerprint=fp)
        self.assertFalse(nov.novelty_valid_for_promotion)
        self.assertEqual(nov.novelty_validity_reason, "heuristic_fallback_reference")

    def test_novelty_confidence_persisted_in_db(self):
        """novelty_confidence column exists and round-trips through DB."""
        from research.scientist.notebook import LabNotebook
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            rid = nb.record_program_result(
                exp_id, "fp123", "{}",
                novelty_score=0.7, novelty_confidence=0.85,
            )
            nb.flush_writes()
            detail = nb.get_program_detail(rid)
            self.assertIsNotNone(detail, "get_program_detail returned None — async write may not have flushed")
            self.assertAlmostEqual(detail["novelty_confidence"], 0.85)
            nb.close()

    def test_op_success_rates_tracks_novelty_confidence(self):
        """update_op_success_rates persists avg_novelty_confidence."""
        from research.scientist.notebook import LabNotebook
        import tempfile, os, json

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            graph = {"nodes": {"n1": {"op_name": "relu", "inputs": ["input"]}}}
            nb.record_program_result(
                exp_id, "fp1", json.dumps(graph),
                novelty_score=0.6, novelty_confidence=0.9,
                stage0_passed=True, stage1_passed=True,
            )
            nb.record_program_result(
                exp_id, "fp2", json.dumps(graph),
                novelty_score=0.4, novelty_confidence=0.3,
                stage0_passed=True, stage1_passed=False,
            )
            nb.flush_writes()
            nb.update_op_success_rates(exp_id)
            rates = nb.get_op_success_rates()
            relu_rate = [r for r in rates if r["op_name"] == "relu"][0]
            self.assertIsNotNone(relu_rate["avg_novelty_confidence"])
            self.assertAlmostEqual(relu_rate["avg_novelty_confidence"], 0.6)
            nb.close()

    def test_breakthrough_requires_novelty_confidence(self):
        """Runner breakthrough gate requires novelty_confidence >= 0.5."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(
            config.auto_validate_min_novelty_confidence, 0.5,
            "Breakthrough gate must require novelty_confidence >= 0.5",
        )

    def test_breakthrough_requires_5_seeds(self):
        """Runner breakthrough gate requires >= 5 seeds passed."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(
            config.validation_n_seeds, 5,
            "Breakthrough gate must require >= 5 seeds",
        )

    def test_validation_n_seeds_default_is_5(self):
        """RunConfig.validation_n_seeds default must be >= 5."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(config.validation_n_seeds, 5,
                                "validation_n_seeds must default to >= 5 for breakthrough eligibility")

    def test_grammar_weights_discount_low_confidence_novelty(self):
        """Grammar weight novelty factor should be scaled by confidence."""
        from research.scientist.analytics import ExperimentAnalytics
        from unittest.mock import MagicMock

        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)
        analytics.nb = MagicMock()

        # High confidence novelty vs low confidence novelty
        stats_high_conf = {
            "total": 100, "s1_total": 20, "novelty_sum": 50.0, "count": 100,
            "conf_sum": 90.0, "conf_count": 100,  # avg conf = 0.9
        }
        stats_low_conf = {
            "total": 100, "s1_total": 20, "novelty_sum": 50.0, "count": 100,
            "conf_sum": 20.0, "conf_count": 100,  # avg conf = 0.2
        }

        weights_high = analytics._compute_weights_from_stats(
            {"activation": stats_high_conf})
        weights_low = analytics._compute_weights_from_stats(
            {"activation": stats_low_conf})

        # Both should produce weights, but high-conf should weight novelty more
        self.assertIsNotNone(weights_high)
        self.assertIsNotNone(weights_low)
        # With same s1_rate (only one category), both hit statistical guard
        # and return default. Use two categories to get past the guard.
        stats_good = {
            "total": 100, "s1_total": 30, "novelty_sum": 80.0, "count": 100,
            "conf_sum": 90.0, "conf_count": 100,
        }
        stats_bad = {
            "total": 100, "s1_total": 5, "novelty_sum": 10.0, "count": 100,
            "conf_sum": 20.0, "conf_count": 100,
        }
        w_high = analytics._compute_weights_from_stats({
            "activation": stats_good, "linear": stats_bad,
        })
        # Replace good stats with low confidence
        stats_good_lowconf = dict(stats_good)
        stats_good_lowconf["conf_sum"] = 10.0  # avg conf = 0.1
        w_low = analytics._compute_weights_from_stats({
            "activation": stats_good_lowconf, "linear": stats_bad,
        })
        self.assertIsNotNone(w_high)
        self.assertIsNotNone(w_low)
        # High-confidence novelty should give a higher weight
        self.assertGreater(w_high["activation"], w_low["activation"],
                           "High-confidence novelty should produce higher grammar weight")

    def test_grammar_weights_cap_repeated_fingerprint_influence(self):
        """Fingerprint-capped weighting should reduce repeated architecture dominance."""
        from research.scientist.analytics import ExperimentAnalytics
        from unittest.mock import patch as _patch

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fingerprint_cap.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("synthesis", {}, "fingerprint-cap")

            dominant_graph = {
                "nodes": {
                    "n1": {"op_name": "relu"},
                    "n2": {"op_name": "gelu"},
                    "n3": {"op_name": "tanh"},
                    "n4": {"op_name": "matmul"},
                    "n5": {"op_name": "layernorm"},
                }
            }
            contrast_graph = {
                "nodes": {
                    "n1": {"op_name": "sin"},
                    "n2": {"op_name": "cos"},
                    "n3": {"op_name": "exp"},
                    "n4": {"op_name": "sum_reduce"},
                    "n5": {"op_name": "mean_reduce"},
                }
            }

            for i in range(24):
                nb.record_program_result(
                    exp_id,
                    "fp_repeat_dominant",
                    json.dumps(dominant_graph),
                    stage0_passed=True,
                    stage1_passed=True,
                    novelty_score=0.8,
                    novelty_confidence=0.9,
                    loss_ratio=0.4,
                    timestamp=time.time() + i,
                )
            for i in range(10):
                nb.record_program_result(
                    exp_id,
                    f"fp_unique_{i}",
                    json.dumps(contrast_graph),
                    stage0_passed=True,
                    stage1_passed=False,
                    novelty_score=0.2,
                    novelty_confidence=0.6,
                    loss_ratio=0.9,
                    timestamp=time.time() + 100 + i,
                )

            nb.flush_writes()

            analytics_capped = ExperimentAnalytics(nb)
            analytics_uncapped = ExperimentAnalytics(nb)

            capped_rates, capped_diag = analytics_capped._collect_fingerprint_capped_op_rates(3.0)
            uncapped_rates, _ = analytics_uncapped._collect_fingerprint_capped_op_rates(1_000_000.0)

            self.assertIn("relu", capped_rates)
            self.assertIn("relu", uncapped_rates)
            self.assertLess(capped_rates["relu"]["n_used"], uncapped_rates["relu"]["n_used"])
            self.assertGreater(capped_diag["rerun_ratio"], 0.5)
            self.assertGreater(capped_diag["top_fingerprint_concentration"], 0.5)

            with _patch.object(ExperimentAnalytics, 'FINGERPRINT_WEIGHT_CAP', 3.0):
                capped_weights = analytics_capped.compute_grammar_weights()
                diag = analytics_capped.grammar_weight_learning_diagnostics()
            with _patch.object(ExperimentAnalytics, 'FINGERPRINT_WEIGHT_CAP', 1_000_000.0):
                uncapped_weights = analytics_uncapped.compute_grammar_weights()
            self.assertIsNotNone(capped_weights)
            self.assertIsNotNone(uncapped_weights)

            self.assertEqual(diag.get("mode"), "fingerprint_capped")
            self.assertTrue(diag.get("used_fingerprint_capping"))
            self.assertEqual(diag.get("fingerprint_cap"), 3.0)

            nb.close()

    def test_composite_score_discounts_low_confidence_novelty(self):
        """Composite score should weight novelty contribution by confidence."""
        from research.scientist.notebook import LabNotebook

        # Full confidence: novelty fully counted
        score_full = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.9)
        # Low confidence: novelty discounted
        score_low = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.2)
        # No confidence param: defaults to 1.0 (backward compat)
        score_none = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8)

        self.assertGreater(score_full, score_low,
                           "High confidence should yield higher composite score")
        self.assertEqual(score_none, LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=1.0),
            "None confidence should behave like 1.0")
        # Zero confidence should eliminate novelty contribution entirely
        score_zero = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.0)
        score_no_nov = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.0)
        self.assertAlmostEqual(score_zero, score_no_nov, places=6,
                               msg="Zero confidence should be equivalent to zero novelty")

    def test_upsert_leaderboard_passes_novelty_confidence(self):
        """upsert_leaderboard should use novelty_confidence in composite score."""
        import tempfile, os
        from research.scientist.notebook import LabNotebook

        with tempfile.TemporaryDirectory() as d:
            nb = LabNotebook(os.path.join(d, "test.db"))
            exp_id = nb.start_experiment("test", {}, "test")
            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="fp1",
                graph_json="{}",
                loss_ratio=0.5,
                novelty_score=0.8,
                novelty_confidence=0.9,
            )
            # High confidence
            eid_high = nb.upsert_leaderboard(
                result_id=rid, model_source="test",
                screening_loss_ratio=0.5, screening_novelty=0.8,
                novelty_confidence=0.9,
            )
            # Low confidence
            rid2 = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="fp2",
                graph_json="{}",
                loss_ratio=0.5,
                novelty_score=0.8,
                novelty_confidence=0.2,
            )
            eid_low = nb.upsert_leaderboard(
                result_id=rid2, model_source="test",
                screening_loss_ratio=0.5, screening_novelty=0.8,
                novelty_confidence=0.2,
            )
            lb = nb.get_leaderboard(limit=10)
            scores = {e["entry_id"]: e["composite_score"] for e in lb}
            self.assertGreater(scores[eid_high], scores[eid_low])

    def test_composite_score_rewards_real_token_quality(self):
        """WikiText quality should materially improve composite score."""
        from research.scientist.notebook import LabNotebook

        weak = LabNotebook.compute_composite_score(
            screening_lr=0.15,
            screening_nov=0.8,
            wikitext_score=0.45,
            wikitext_perplexity=180.0,
        )
        strong = LabNotebook.compute_composite_score(
            screening_lr=0.15,
            screening_nov=0.8,
            wikitext_score=0.68,
            wikitext_perplexity=30.0,
        )
        self.assertGreater(strong, weak)

    def test_failed_investigation_only_penalized_with_negative_evidence(self):
        """Failed investigation should hurt only when robustness/token quality are weak."""
        from research.scientist.notebook import LabNotebook

        weak = LabNotebook.compute_composite_score(
            screening_lr=0.02,
            screening_nov=0.8,
            inv_lr=0.2,
            inv_robust=0.3333,
            investigation_passed=False,
            validation_passed=False,
            wikitext_score=0.45,
            wikitext_perplexity=180.0,
        )
        strong = LabNotebook.compute_composite_score(
            screening_lr=0.02,
            screening_nov=0.8,
            inv_lr=0.2,
            inv_robust=0.3333,
            investigation_passed=False,
            validation_passed=False,
            wikitext_score=0.67,
            wikitext_perplexity=35.0,
        )
        self.assertGreater(strong, weak)


class TestBaselineDataFn(unittest.TestCase):
    """Tests for baseline training with custom data functions."""

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_with_data_fn(self):
        """Baseline trains using provided data_fn instead of random tokens."""
        from research.eval.baseline import TransformerBaseline
        import torch

        call_count = [0]
        def fake_data(batch_size, seq_len, dev):
            call_count[0] += 1
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            loss = bl.get_baseline_loss(
                d_model=64, seq_len=32, n_steps=5, vocab_size=1024,
                batch_size=2, device="cpu", data_fn=fake_data, data_tag="test",
            )
            self.assertTrue(0 < loss < 20)
            self.assertGreater(call_count[0], 0, "data_fn should have been called")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_compare_with_data_fn(self):
        """compare() passes data_fn through to training."""
        from research.eval.baseline import TransformerBaseline
        import torch

        def fake_data(batch_size, seq_len, dev):
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            ratio = bl.compare(
                program_loss=5.0,
                d_model=64, seq_len=32, n_steps=5, vocab_size=1024,
                batch_size=2, device="cpu", data_fn=fake_data, data_tag="test",
            )
            self.assertIsInstance(ratio, float)
            self.assertGreater(ratio, 0)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_data_tag_separates_cache(self):
        """Different data_tags produce separate cache entries."""
        from research.eval.baseline import TransformerBaseline

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            key_random = bl._config_key(64, 32, 5, 1024, data_tag="random")
            key_hydra = bl._config_key(64, 32, 5, 1024, data_tag="hydra")
            self.assertNotEqual(key_random, key_hydra)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_hydra_batch_fallback_to_random(self):
        """_get_hydra_batch returns None when HYDRA loader fails to init."""
        from research.scientist.runner import ExperimentRunner, RunConfig
        import torch

        config = RunConfig(data_mode="hydra", hydra_project_root="/nonexistent")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._hydra_loader = None
        runner._hydra_iter = None
        runner._hydra_signature = ""
        # Mock the import to raise immediately rather than risk hanging
        with patch.dict("sys.modules", {"hydra.data": None}):
            result = runner._get_hydra_batch(config, 2, 32, torch.device("cpu"))
        self.assertIsNone(result, "Should return None when HYDRA unavailable")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_random_mode(self):
        """_make_baseline_data_fn returns (None, 'random', False) for random mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="random")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag, cache_data_fn = runner._make_baseline_data_fn(config)
        self.assertIsNone(data_fn)
        self.assertEqual(data_tag, "random")
        self.assertFalse(cache_data_fn)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_hydra_mode(self):
        """_make_baseline_data_fn returns a callable for hydra mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="hydra")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag, cache_data_fn = runner._make_baseline_data_fn(config)
        self.assertIsNotNone(data_fn)
        self.assertEqual(data_tag, "hydra")
        self.assertFalse(cache_data_fn)


class TestCkaReferenceArtifacts(unittest.TestCase):
    """Tests for CKA reference artifact loader/validator/cache (#28/#43 Phase A)."""

    def _make_artifact_dir(self, tmpdir, manifest_override=None, families=None):
        """Helper: create a valid artifact directory with manifest and .pt files."""
        import json, torch
        art_dir = os.path.join(tmpdir, "cka_references", "v1")
        os.makedirs(art_dir, exist_ok=True)

        manifest = {
            "artifact_version": "v1",
            "schema_version": "1",
            "created_at": "2026-01-01T00:00:00Z",
            "code_version": "test",
            "reference_families": ["transformer", "ssm", "conv"],
            "probe_protocol_hash": "abc123",
            "activation_shape": [16, 32],
            "quality_flags": {"overall": "good"},
        }
        if manifest_override:
            manifest.update(manifest_override)

        with open(os.path.join(art_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        # Create .pt files
        shape = manifest["activation_shape"]
        for family in (families or ["transformer", "ssm", "conv"]):
            data = {
                "activations": torch.randn(shape[0], shape[1]),
                "config": {"family": family},
                "training_info": {},
            }
            torch.save(data, os.path.join(art_dir, f"{family}.pt"))

        return art_dir

    def test_load_manifest_valid(self):
        """Valid manifest loads without error."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            self.assertEqual(m.artifact_version, "v1")
            self.assertEqual(m.schema_version, "1")
            self.assertEqual(set(m.reference_families), {"transformer", "ssm", "conv"})
            self.assertEqual(m.activation_shape, [16, 32])

    def test_load_manifest_missing_file(self):
        """Missing manifest.json raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError, msg="No manifest.json"):
                load_manifest(Path(d))

    def test_load_manifest_malformed_json(self):
        """Malformed JSON raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                f.write("{bad json")
            with self.assertRaises(ValueError):
                load_manifest(Path(d))

    def test_load_manifest_missing_fields(self):
        """Manifest missing required fields raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                json.dump({"artifact_version": "v1"}, f)
            with self.assertRaises(ValueError, msg="missing required fields"):
                load_manifest(Path(d))

    def test_load_manifest_unsupported_schema(self):
        """Unsupported schema version raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"schema_version": "99"})
            with self.assertRaises(ValueError, msg="Unsupported schema"):
                load_manifest(Path(art_dir))

    def test_load_manifest_missing_family(self):
        """Manifest with incomplete families raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(
                d, {"reference_families": ["transformer"]}
            )
            with self.assertRaises(ValueError, msg="missing reference families"):
                load_manifest(Path(art_dir))

    def test_load_manifest_bad_activation_shape(self):
        """Invalid activation_shape raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"activation_shape": [0, 32]})
            with self.assertRaises(ValueError):
                load_manifest(Path(art_dir))

    def test_load_reference_activations_valid(self):
        """Valid .pt files load as tensors with correct shape."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            refs = load_reference_activations(Path(art_dir), m)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            for t in refs.values():
                self.assertEqual(tuple(t.shape[-2:]), (16, 32))

    def test_load_reference_activations_missing_file(self):
        """Missing .pt file raises ValueError."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            os.remove(os.path.join(art_dir, "ssm.pt"))
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="Missing artifact file"):
                load_reference_activations(Path(art_dir), m)

    def test_load_reference_activations_shape_mismatch(self):
        """Tensor with wrong shape raises ValueError."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile, torch
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            # Overwrite one file with wrong shape
            torch.save(
                {"activations": torch.randn(8, 32)},
                os.path.join(art_dir, "conv.pt"),
            )
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="shape mismatch"):
                load_reference_activations(Path(art_dir), m)

    def test_store_no_artifacts_returns_none(self):
        """ReferenceCkaStore with no artifacts returns None references."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = ReferenceCkaStore(artifact_dir=os.path.join(d, "nonexistent"))
            self.assertIsNone(store.get_references())
            self.assertFalse(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "heuristic_fallback")

    def test_store_with_valid_artifacts(self):
        """ReferenceCkaStore loads valid artifacts successfully."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_store_reset_clears_cache(self):
        """reset() clears loaded state so next access reloads."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            self.assertTrue(store.is_artifact_backed)
            store.reset()
            # Point to nonexistent dir after reset
            store._artifact_dir = os.path.join(d, "gone")
            self.assertFalse(store.is_artifact_backed)

    def test_store_metadata_provenance_fields(self):
        """Metadata includes all expected provenance fields."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            meta = store.get_metadata()
            self.assertIn("cka_source", meta)
            self.assertIn("cka_artifact_version", meta)
            self.assertIn("cka_probe_protocol_hash", meta)
            self.assertIn("cka_reference_quality", meta)
            self.assertEqual(meta["cka_reference_quality"], "good")

    # ── Phase C: Runtime CKA switchover tests ──

    def test_compute_reference_cka_with_artifacts(self):
        """_compute_reference_cka uses artifact activations when provided."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        # Create fake candidate reps and reference activations
        S, D = 16, 32
        reps = torch.randn(1, S, D)
        ref_activations = {
            "transformer": torch.randn(S, D),
            "ssm": torch.randn(S, D),
            "conv": torch.randn(S, D),
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])
        for family in ("transformer", "ssm", "conv"):
            self.assertGreaterEqual(result[family], 0.0)
            self.assertLessEqual(result[family], 1.0)

    def test_compute_reference_cka_heuristic_fallback(self):
        """_compute_reference_cka falls back to heuristic when no artifacts."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        reps = torch.randn(1, 16, 32)
        result = _compute_reference_cka(reps, ref_activations=None)
        self.assertTrue(result["_succeeded"])
        # Heuristic should still produce valid CKA values
        for family in ("transformer", "ssm", "conv"):
            self.assertGreaterEqual(result[family], 0.0)
            self.assertLessEqual(result[family], 1.0)

    def test_compute_reference_cka_seq_len_mismatch(self):
        """Artifact CKA handles different seq lengths between candidate and reference."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        reps = torch.randn(1, 16, 32)  # seq_len=16
        ref_activations = {
            "transformer": torch.randn(24, 32),  # seq_len=24 (longer)
            "ssm": torch.randn(8, 32),           # seq_len=8 (shorter)
            "conv": torch.randn(16, 32),          # seq_len=16 (same)
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])

    def test_fingerprint_records_cka_source(self):
        """Fingerprint records cka_source provenance."""
        from research.eval.cka_references import reset_default_store
        reset_default_store()  # ensure clean state

        fp = self._make_fingerprint()
        # Should be one of the valid sources
        self.assertIn(fp.cka_source, ("artifact", "heuristic_fallback", "none"))

    def test_fingerprint_heuristic_fallback_when_no_artifacts(self):
        """Fingerprint falls back to heuristic when artifact dir is missing."""
        from unittest.mock import patch
        from research.eval import cka_references
        from research.eval.cka_references import ReferenceCkaStore, reset_default_store

        reset_default_store()
        # Force a store pointing to nonexistent dir
        fake_store = ReferenceCkaStore(artifact_dir="/nonexistent/path")
        with patch.object(cka_references, '_default_store', fake_store):
            with patch.object(cka_references, '_default_lock', cka_references.threading.Lock()):
                # Override get_default_store to return our fake store
                with patch('research.eval.cka_references.get_default_store', return_value=fake_store):
                    fp = self._make_fingerprint()
        self.assertIn(fp.cka_source, ("heuristic_fallback",))
        reset_default_store()

    def _make_fingerprint(self):
        """Helper: compute fingerprint on a tiny model."""
        import torch.nn as nn
        from research.eval.fingerprint import compute_fingerprint

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 100)
            def forward(self, x):
                return self.linear(self.embed(x))

        model = TinyModel()
        return compute_fingerprint(model, seq_len=8, model_dim=32,
                                   vocab_size=100, device="cpu", n_probes=4)

    def test_fingerprint_cka_provenance_fields_exist(self):
        """BehavioralFingerprint has cka_source and cka_artifact_version fields."""
        from research.eval.fingerprint import BehavioralFingerprint

        fp = BehavioralFingerprint()
        self.assertEqual(fp.cka_source, "none")
        self.assertIsNone(fp.cka_artifact_version)
        d = fp.to_dict()
        self.assertIn("cka_source", d)
        self.assertIn("cka_artifact_version", d)

    def test_export_produces_loadable_artifacts(self):
        """Export tool produces artifacts that ReferenceCkaStore can load."""
        from research.tools.export_cka_references import export_artifacts
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = str(Path(d) / "refs" / "v1")
            export_artifacts(
                output_dir=art_dir, seed=123, n_steps=10, device="cpu",
            )
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_export_deterministic(self):
        """Same seed produces same probe_protocol_hash."""
        from research.tools.export_cka_references import export_artifacts

        with tempfile.TemporaryDirectory() as d:
            d1 = str(Path(d) / "run1")
            d2 = str(Path(d) / "run2")
            export_artifacts(output_dir=d1, seed=99, n_steps=5, device="cpu")
            export_artifacts(output_dir=d2, seed=99, n_steps=5, device="cpu")

            with open(Path(d1) / "manifest.json") as f:
                m1 = json.load(f)
            with open(Path(d2) / "manifest.json") as f:
                m2 = json.load(f)
            self.assertEqual(
                m1["probe_protocol_hash"], m2["probe_protocol_hash"]
            )
            self.assertEqual(m1["activation_shape"], m2["activation_shape"])




if __name__ == '__main__':
    unittest.main()
