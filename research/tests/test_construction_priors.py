"""Tests for multi-metric construction priors built from ablation evidence."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.scientist.notebook import LabNotebook
from research.scientist.construction_priors import (
    USE_THRESHOLD,
    AVOID_THRESHOLD,
    _classify,
    _composite_score,
    compute_construction_prior,
    construction_prior_as_grammar_adjustments,
    get_active_construction_prior,
    list_construction_prior_snapshots,
    record_construction_prior_snapshot,
)


class TestCompositeScore(unittest.TestCase):
    def test_score_zero_when_all_metrics_missing(self):
        score, weight = _composite_score({})
        self.assertEqual(score, 0.0)
        self.assertEqual(weight, 0.0)

    def test_score_positive_when_metrics_indicate_useful(self):
        score, weight = _composite_score({
            "induction": 0.10, "binding": 0.08, "ar": 0.05,
            "blimp": 0.02, "hellaswag": 0.02, "ppl_pct": 0.20, "loss": 0.10,
        })
        self.assertGreater(score, 0.3)
        self.assertGreater(weight, 0.9)

    def test_score_negative_when_metrics_indicate_baggage(self):
        score, _ = _composite_score({
            "induction": -0.10, "binding": -0.08,
            "ar": -0.05, "blimp": -0.02, "hellaswag": -0.02,
            "ppl_pct": -0.20, "loss": -0.10,
        })
        self.assertLess(score, -0.3)

    def test_score_partial_metric_coverage(self):
        score, weight = _composite_score({"induction": 0.20, "binding": 0.15})
        self.assertGreater(score, 0.5)  # both saturate positive
        self.assertAlmostEqual(weight, 0.40, places=2)

    def test_classify_thresholds(self):
        self.assertEqual(_classify(USE_THRESHOLD + 0.01), "use")
        self.assertEqual(_classify(AVOID_THRESHOLD - 0.01), "avoid")
        self.assertEqual(_classify(0.0), "mixed")


class TestConstructionPriorEndToEnd(unittest.TestCase):
    """Build evidence in a real notebook, compute prior, snapshot, read back."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "lab_notebook.db"
        self.nb = LabNotebook(str(self.db_path))

    def tearDown(self) -> None:
        self.nb.close()
        self._tmp.cleanup()

    def _write_parent(self, *, rid: str, fp: str, **metrics) -> None:
        exp_id = self.nb.start_experiment("synthesis", {}, "parent")
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            result_id=rid,
            stage1_passed=True,
            loss_ratio=metrics.get("loss_ratio", 0.4),
            wikitext_perplexity=metrics.get("wikitext_perplexity", 200.0),
            hellaswag_acc=metrics.get("hellaswag_acc", 0.30),
            blimp_overall_accuracy=metrics.get("blimp_overall_accuracy", 0.55),
            induction_auc=metrics.get("induction_auc", 0.30),
            binding_auc=metrics.get("binding_auc", 0.20),
            binding_composite=metrics.get("binding_composite", 0.20),
            ar_auc=metrics.get("ar_auc", 0.10),
            model_source="graph_synthesis",
        )
        self.nb.flush_writes()

    def _write_child(self, *, parent_rid: str, parent_fp: str,
                     rid: str, fp: str, rule_type: str, rule_key: str,
                     **metrics) -> None:
        # Use intentional_rerun_reason to avoid duplicate-fingerprint blocks for
        # children that happen to share fingerprints across rules in tests.
        exp_id = self.nb.start_experiment(
            "ablation",
            {},
            f"ablation child for {rule_key}",
        )
        from research.scientist.runner._helpers import program_result_kwargs_from_s1
        s1 = {
            "passed": True, "loss_ratio": metrics.get("loss_ratio", 0.5),
            "final_loss": 5.0,
            "wikitext_perplexity": metrics.get("wikitext_perplexity", 250.0),
            "wikitext_score": 0.4,
            "hellaswag_acc": metrics.get("hellaswag_acc", 0.27),
            "blimp_overall_accuracy": metrics.get("blimp_overall_accuracy", 0.52),
            "induction_auc": metrics.get("induction_auc", 0.20),
            "binding_auc": metrics.get("binding_auc", 0.10),
            "binding_composite": metrics.get("binding_composite", 0.10),
            "ar_auc": metrics.get("ar_auc", 0.05),
            "fp_jacobian_erf_density": 0.5, "fp_icld_delta_loss": -0.3,
            "fp_logit_margin_delta": 0.2,
        }
        kwargs = program_result_kwargs_from_s1(s1, model_source="ablation")
        rid_returned = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            result_id=rid,
            stage0_passed=True, stage05_passed=True, stage1_passed=True,
            intentional_rerun_reason="ablation_counterfactual",
            **kwargs,
        )
        self.nb.flush_writes()
        # Record the linkage in causal_ablation_child_observations
        evidence_id = self.nb.record_causal_rule_evidence({
            "parent_experiment_id": "exp_parent",
            "parent_result_id": parent_rid,
            "parent_fingerprint": parent_fp,
            "ablation_experiment_id": exp_id,
            "rule_type": rule_type, "rule_key": rule_key,
            "rule_context": "{}",
            "original_loss_ratio": 0.4,
            "ablation_best_loss_ratio": metrics.get("loss_ratio", 0.5),
            "effect_size": metrics.get("loss_ratio", 0.5) - 0.4,
            "original_stage1_passed": 1,
            "ablation_stage1_pass_count": 1,
            "ablation_total": 1,
            "outcome": "supported",
            "confidence": 0.5,
            "evidence_json": "{}",
        })
        self.nb.record_causal_ablation_child_observations(evidence_id, [{
            "parent_result_id": parent_rid,
            "parent_experiment_id": "exp_parent",
            "parent_fingerprint": parent_fp,
            "child_result_id": rid_returned,
            "child_experiment_id": exp_id,
            "child_fingerprint": fp,
            "ablation_experiment_id": exp_id,
            "source": "executed",
            "rule_type": rule_type,
            "rule_key": rule_key,
            "stage1_passed": 1,
            "loss_ratio": metrics.get("loss_ratio", 0.5),
            "model_source": "ablation",
            "trust_label": "executed",
            "comparability_label": "ablation_counterfactual",
            "provenance": {},
        }])
        self.nb.flush_writes()

    def test_compute_use_verdict_for_helpful_op(self) -> None:
        # Parent has strong probes; children all show probes drop sharply when
        # this op is removed → verdict 'use'.
        self._write_parent(
            rid="parent_useful", fp="fp_parent_useful",
            induction_auc=0.50, binding_composite=0.40, ar_auc=0.20,
            hellaswag_acc=0.32, blimp_overall_accuracy=0.60,
            wikitext_perplexity=150.0,
        )
        for i in range(5):
            self._write_child(
                parent_rid="parent_useful", parent_fp="fp_parent_useful",
                rid=f"child_useful_{i}", fp=f"fp_child_useful_{i}",
                rule_type="op", rule_key="proj_shared_basis",
                induction_auc=0.30,           # Δ +0.20 helpful
                binding_composite=0.25,        # Δ +0.15 helpful
                ar_auc=0.10,                   # Δ +0.10 helpful
                hellaswag_acc=0.27,            # Δ +0.05 helpful
                blimp_overall_accuracy=0.55,   # Δ +0.05 helpful
                wikitext_perplexity=250.0,     # Δ +66% helpful
            )
        prior = compute_construction_prior(self.nb, min_n=3, min_metric_complete=3)
        rules = prior["payload"]["rules"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["rule_key"], "proj_shared_basis")
        self.assertEqual(rules[0]["verdict"], "use")
        self.assertGreater(rules[0]["score"], USE_THRESHOLD)
        self.assertGreater(rules[0]["multiplier"], 1.0)
        self.assertIn("proj_shared_basis", prior["payload"]["op_weights"])

    def test_compute_avoid_verdict_for_harmful_op(self) -> None:
        # Children improve across every metric when this op is removed → 'avoid'.
        self._write_parent(
            rid="parent_bag", fp="fp_parent_bag",
            induction_auc=0.10, binding_composite=0.05, ar_auc=0.03,
            hellaswag_acc=0.27, blimp_overall_accuracy=0.50,
            wikitext_perplexity=600.0,
        )
        for i in range(5):
            self._write_child(
                parent_rid="parent_bag", parent_fp="fp_parent_bag",
                rid=f"child_bag_{i}", fp=f"fp_child_bag_{i}",
                rule_type="op", rule_key="bad_op",
                induction_auc=0.40,           # Δ -0.30 (children better)
                binding_composite=0.25,
                ar_auc=0.18,
                hellaswag_acc=0.32,
                blimp_overall_accuracy=0.58,
                wikitext_perplexity=150.0,    # children PPL much lower
            )
        prior = compute_construction_prior(self.nb, min_n=3, min_metric_complete=3)
        rules = prior["payload"]["rules"]
        self.assertEqual(rules[0]["verdict"], "avoid")
        self.assertLess(rules[0]["score"], AVOID_THRESHOLD)
        self.assertLess(rules[0]["multiplier"], 1.0)

    def test_snapshot_round_trip_and_grammar_adjustments(self) -> None:
        self._write_parent(
            rid="p1", fp="fp_p1",
            induction_auc=0.5, binding_composite=0.4, ar_auc=0.2,
            hellaswag_acc=0.32, blimp_overall_accuracy=0.6,
            wikitext_perplexity=150.0,
        )
        for i in range(4):
            self._write_child(
                parent_rid="p1", parent_fp="fp_p1",
                rid=f"c_{i}", fp=f"fp_c_{i}",
                rule_type="op", rule_key="useful_op",
                induction_auc=0.3, binding_composite=0.25,
                ar_auc=0.10, hellaswag_acc=0.27,
                blimp_overall_accuracy=0.55, wikitext_perplexity=250.0,
            )
        prior = compute_construction_prior(self.nb, min_n=3, min_metric_complete=3)
        version = record_construction_prior_snapshot(
            self.nb, prior, activate=True, notes="test"
        )
        self.assertTrue(version)
        active = get_active_construction_prior(self.nb)
        self.assertIsNotNone(active)
        self.assertEqual(active["version"], version)
        snaps = list_construction_prior_snapshots(self.nb)
        self.assertEqual(len(snaps), 1)
        self.assertTrue(snaps[0]["is_active"])
        adj = construction_prior_as_grammar_adjustments(active)
        self.assertEqual(adj["version"], version)
        self.assertIn("useful_op", adj["op_weights"])

    def test_activating_new_snapshot_demotes_old(self) -> None:
        self._write_parent(
            rid="p2", fp="fp_p2",
            induction_auc=0.5, binding_composite=0.4, ar_auc=0.2,
            hellaswag_acc=0.32, blimp_overall_accuracy=0.6,
            wikitext_perplexity=150.0,
        )
        for i in range(3):
            self._write_child(
                parent_rid="p2", parent_fp="fp_p2",
                rid=f"c2_{i}", fp=f"fp_c2_{i}",
                rule_type="op", rule_key="useful_op",
                induction_auc=0.3, binding_composite=0.25, ar_auc=0.10,
                hellaswag_acc=0.27, blimp_overall_accuracy=0.55,
                wikitext_perplexity=250.0,
            )
        prior_a = compute_construction_prior(self.nb, min_n=3, min_metric_complete=3)
        v_a = record_construction_prior_snapshot(self.nb, prior_a, activate=True)
        # Force a different version string by pre-populating
        prior_a["payload"]["version"] = "v_test_b"
        v_b = record_construction_prior_snapshot(self.nb, prior_a, activate=True)
        self.assertNotEqual(v_a, v_b)
        snaps = list_construction_prior_snapshots(self.nb)
        active = [s for s in snaps if s["is_active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["version"], v_b)


if __name__ == "__main__":
    unittest.main()
