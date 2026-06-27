"""Novelty confidence/quality tracking regression tests.

Split from the test_novelty.py omnibus on 2026-06-13."""

import pytest
import importlib
import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch  # noqa: F401

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
    from research.scientist.notebook import LabNotebook

    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
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
            novelty_score=0.7,
            quality="full",
            analyses_succeeded=4,
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
                novelty_score=0.5,
                quality="partial",
                analyses_succeeded=n,
            )
            nov = novelty_score(graph, fingerprint=fp)
            expected = 0.4 + n * 0.1
            self.assertAlmostEqual(
                nov.novelty_confidence, expected, msg=f"analyses_succeeded={n}"
            )

    def test_confidence_none_quality_with_fingerprint(self):
        """quality='none' but fingerprint provided gives confidence=0.3."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(
            novelty_score=0.5, quality="none", analyses_succeeded=0
        )
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
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            rid = nb.record_program_result(
                exp_id,
                "fp123",
                "{}",
                novelty_score=0.7,
                novelty_confidence=0.85,
            )
            nb.flush_writes()
            detail = nb.get_program_detail(rid)
            self.assertIsNotNone(
                detail,
                "get_program_detail returned None — async write may not have flushed",
            )
            self.assertAlmostEqual(detail["novelty_confidence"], 0.85)
            nb.close()

    def test_op_success_rates_tracks_novelty_confidence(self):
        """update_op_success_rates persists avg_novelty_confidence."""
        from research.scientist.notebook import LabNotebook
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            graph = {"nodes": {"n1": {"op_name": "relu", "inputs": ["input"]}}}
            nb.record_program_result(
                exp_id,
                "fp1",
                json.dumps(graph),
                novelty_score=0.6,
                novelty_confidence=0.9,
                stage0_passed=True,
                stage1_passed=True,
                trust_label="test_fixture",
            )
            nb.record_program_result(
                exp_id,
                "fp2",
                json.dumps(graph),
                novelty_score=0.4,
                novelty_confidence=0.3,
                stage0_passed=True,
                stage1_passed=False,
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
            config.auto_validate_min_novelty_confidence,
            0.5,
            "Breakthrough gate must require novelty_confidence >= 0.5",
        )

    def test_breakthrough_requires_5_seeds(self):
        """Runner breakthrough gate requires >= 5 seeds passed."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(
            config.validation_n_seeds,
            5,
            "Breakthrough gate must require >= 5 seeds",
        )

    def test_validation_n_seeds_default_is_5(self):
        """RunConfig.validation_n_seeds default must be >= 5."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(
            config.validation_n_seeds,
            5,
            "validation_n_seeds must default to >= 5 for breakthrough eligibility",
        )

    def test_grammar_weights_discount_low_confidence_novelty(self):
        """Grammar weight novelty factor should be scaled by confidence."""
        from research.scientist.analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)
        analytics.nb = MagicMock()

        # High confidence novelty vs low confidence novelty
        stats_high_conf = {
            "total": 100,
            "s1_total": 20,
            "novelty_sum": 50.0,
            "count": 100,
            "conf_sum": 90.0,
            "conf_count": 100,  # avg conf = 0.9
        }
        stats_low_conf = {
            "total": 100,
            "s1_total": 20,
            "novelty_sum": 50.0,
            "count": 100,
            "conf_sum": 20.0,
            "conf_count": 100,  # avg conf = 0.2
        }

        weights_high = analytics._compute_weights_from_stats(
            {"activation": stats_high_conf}
        )
        weights_low = analytics._compute_weights_from_stats(
            {"activation": stats_low_conf}
        )

        # Both should produce weights, but high-conf should weight novelty more
        self.assertIsNotNone(weights_high)
        self.assertIsNotNone(weights_low)
        # With same s1_rate (only one category), both hit statistical guard
        # and return default. Use two categories to get past the guard.
        stats_good = {
            "total": 100,
            "s1_total": 30,
            "novelty_sum": 80.0,
            "count": 100,
            "conf_sum": 90.0,
            "conf_count": 100,
        }
        stats_bad = {
            "total": 100,
            "s1_total": 5,
            "novelty_sum": 10.0,
            "count": 100,
            "conf_sum": 20.0,
            "conf_count": 100,
        }
        w_high = analytics._compute_weights_from_stats(
            {
                "activation": stats_good,
                "linear": stats_bad,
            }
        )
        # Replace good stats with low confidence
        stats_good_lowconf = dict(stats_good)
        stats_good_lowconf["conf_sum"] = 10.0  # avg conf = 0.1
        w_low = analytics._compute_weights_from_stats(
            {
                "activation": stats_good_lowconf,
                "linear": stats_bad,
            }
        )
        self.assertIsNotNone(w_high)
        self.assertIsNotNone(w_low)
        # High-confidence novelty should give a higher weight
        self.assertGreater(
            w_high["activation"],
            w_low["activation"],
            "High-confidence novelty should produce higher grammar weight",
        )

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
                loop_exp_id = nb.start_experiment(
                    "synthesis", {}, f"fingerprint-cap-dominant-{i}"
                )
                nb.record_program_result(
                    loop_exp_id,
                    "fp_repeat_dominant",
                    json.dumps(dominant_graph),
                    intentional_rerun_reason="replication_measurement",
                    stage0_passed=True,
                    stage1_passed=True,
                    novelty_score=0.8,
                    novelty_confidence=0.9,
                    loss_ratio=0.4,
                    timestamp=time.time() + i,
                    trust_label="test_fixture",
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

            capped_rates, capped_diag = (
                analytics_capped._collect_fingerprint_capped_op_rates(3.0)
            )
            uncapped_rates, _ = analytics_uncapped._collect_fingerprint_capped_op_rates(
                1_000_000.0
            )

            self.assertIn("relu", capped_rates)
            self.assertIn("relu", uncapped_rates)
            self.assertLess(
                capped_rates["relu"]["n_used"], uncapped_rates["relu"]["n_used"]
            )
            self.assertGreater(capped_diag["rerun_ratio"], 0.5)
            self.assertGreater(capped_diag["top_fingerprint_concentration"], 0.5)

            with _patch.object(ExperimentAnalytics, "FINGERPRINT_WEIGHT_CAP", 3.0):
                capped_weights = analytics_capped.compute_grammar_weights()
                diag = analytics_capped.grammar_weight_learning_diagnostics()
            with _patch.object(
                ExperimentAnalytics, "FINGERPRINT_WEIGHT_CAP", 1_000_000.0
            ):
                uncapped_weights = analytics_uncapped.compute_grammar_weights()
            self.assertIsNotNone(capped_weights)
            self.assertIsNotNone(uncapped_weights)

            self.assertEqual(diag.get("mode"), "fingerprint_capped")
            self.assertTrue(diag.get("used_fingerprint_capping"))
            self.assertEqual(diag.get("fingerprint_cap"), 3.0)

            nb.close()

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


if __name__ == "__main__":
    unittest.main()
