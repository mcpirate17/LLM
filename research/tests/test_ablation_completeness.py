"""Tests that ablation S1 rows persist the same post-S1 metric coverage as
normal screening S1 rows. This is the regression that would have caught the
2026-04-29 incident where the ablation runner shipped a 1500-row dataset
with only loss persisted.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.scientist.notebook import LabNotebook
from research.scientist.runner._helpers import (
    S1_REQUIRED_POST_METRIC_COLUMNS,
    program_result_kwargs_from_s1,
)


def _full_s1() -> dict:
    return {
        "passed": True,
        "final_loss": 5.0,
        "loss_ratio": 0.42,
        "param_count": 2_000_000,  # above the nano floor (a valid S1-pass write)
        "wikitext_perplexity": 200.0,
        "wikitext_score": 0.5,
        "hellaswag_acc": 0.31,
        "blimp_overall_accuracy": 0.55,
        "induction_screening_auc": 0.21,
        "binding_screening_auc": 0.18,
        "binding_screening_composite": 0.12,
        "ar_legacy_auc": 0.06,
        "fp_jacobian_erf_density": 0.55,
        "fp_icld_delta_loss": -0.4,
        "fp_logit_margin_delta": 0.3,
    }


def _bare_s1() -> dict:
    """Loss-only s1 — exactly what the original buggy ablation runner produced."""
    return {
        "passed": True,
        "final_loss": 5.0,
        "loss_ratio": 0.42,
        "param_count": 1234,
    }


class TestAblationCompleteness(unittest.TestCase):
    """Ablation S1 writes must carry the full post-S1 metric set or be rejected."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "lab_notebook.db"
        self.nb = LabNotebook(str(self.db_path))

    def tearDown(self) -> None:
        self.nb.close()
        self._tmp.cleanup()

    def _record(
        self,
        *,
        fp: str,
        kwargs: dict,
        experiment_type: str = "ablation",
    ) -> str:
        exp_id = self.nb.start_experiment(experiment_type, {}, "test")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            intentional_rerun_reason="ablation_counterfactual",
            **kwargs,
        )
        self.nb.flush_writes()
        return rid

    def test_canonical_kwargs_satisfies_guardrail(self) -> None:
        """A row built via program_result_kwargs_from_s1 from a complete s1
        must persist successfully and have every required column populated."""
        kwargs = program_result_kwargs_from_s1(_full_s1(), model_source="ablation")
        rid = self._record(fp="fp_full_ablation", kwargs=kwargs)
        self.assertTrue(rid)
        row = self.nb.conn.execute(
            "SELECT "
            + ",".join(S1_REQUIRED_POST_METRIC_COLUMNS)
            + " FROM program_results_compat WHERE result_id = ?",
            (rid,),
        ).fetchone()
        for col in S1_REQUIRED_POST_METRIC_COLUMNS:
            self.assertIsNotNone(
                row[col], f"required post-S1 column {col} is NULL after ablation write"
            )

    def test_loss_only_ablation_write_is_blocked(self) -> None:
        """The original bug: ablation passed stage1 with only loss populated.
        The guardrail must refuse it loudly."""
        with self.assertRaises(ValueError) as ctx:
            self._record(
                fp="fp_loss_only_ablation",
                kwargs={
                    "model_source": "ablation",
                    "loss_ratio": 0.5,
                    "final_loss": 5.0,
                },
            )
        self.assertIn("missing post-S1 metrics", str(ctx.exception))

    def test_partial_metrics_still_blocked(self) -> None:
        """A row missing even one required metric must be rejected."""
        kwargs = program_result_kwargs_from_s1(_full_s1(), model_source="ablation")
        kwargs.pop(
            "induction_screening_auc", None
        )  # ablation runner used to swallow this
        with self.assertRaises(ValueError) as ctx:
            self._record(fp="fp_partial_ablation", kwargs=kwargs)
        self.assertIn("induction_screening_auc", str(ctx.exception))

    def test_backfill_replay_provenance_bypasses_guardrail(self) -> None:
        """The backfill tool sets a trust_label and is the *fix* path; it must
        still be allowed through even if a metric is genuinely unavailable
        post-replay (e.g. probe fails on certain ops)."""
        kwargs = program_result_kwargs_from_s1(
            _bare_s1(),
            model_source="ablation",
            extra={
                "trust_label": "ablation_metric_backfill_replay",
                "comparability_label": "reconstructed_init_variant",
                "evaluation_protocol_version": "ablation_metric_backfill_v1",
            },
        )
        rid = self._record(fp="fp_replay_ablation", kwargs=kwargs)
        self.assertTrue(rid)

    def test_universal_guardrail_blocks_synthesis_loss_only_s1(self) -> None:
        """User rule: 'we never enter missing data for any experiments'.
        The guardrail now applies to every model_source, not just ablation.
        A graph_synthesis row claiming stage1_passed=True with only loss
        populated must be rejected the same way an ablation row would be.
        """
        with self.assertRaises(ValueError) as ctx:
            self._record(
                fp="fp_screening_loss_only",
                kwargs={
                    "model_source": "graph_synthesis",
                    "loss_ratio": 0.5,
                    "final_loss": 5.0,
                },
                experiment_type="synthesis",
            )
        self.assertIn("missing post-S1 metrics", str(ctx.exception))

    def test_universal_guardrail_blocks_fingerprint_refine_partial(self) -> None:
        """fingerprint_refine candidate_grade rows that only ran wikitext+hellaswag
        but skipped induction/binding/ar/blimp must also be rejected."""
        with self.assertRaises(ValueError) as ctx:
            self._record(
                fp="fp_refine_partial",
                kwargs={
                    "model_source": "fingerprint_refine",
                    "trust_label": "candidate_grade",
                    "loss_ratio": 0.5,
                    "wikitext_perplexity": 200.0,
                    "hellaswag_acc": 0.25,
                },
                experiment_type="synthesis",
            )
        self.assertIn("missing post-S1 metrics", str(ctx.exception))

    def test_loss_only_s1_blocked_for_any_model_source(self) -> None:
        """Spot-check every common model_source: stage1_passed=True with no
        post-S1 probe metrics must always raise."""
        for source in (
            "graph_synthesis",
            "fingerprint_refine",
            "novelty",
            "forced_exploration",
            "evolution",
            "grammar",
            "mixed",
            "exact_graph_replay",
        ):
            with self.subTest(model_source=source):
                with self.assertRaises(ValueError):
                    self._record(
                        fp=f"fp_loss_only_{source}",
                        kwargs={
                            "model_source": source,
                            "loss_ratio": 0.5,
                            "final_loss": 5.0,
                        },
                        experiment_type="synthesis",
                    )

    def test_explicit_backfill_observation_bypass(self) -> None:
        """Reconstruction/init-variant backfills carry trust_label='backfill_observation'
        and are exempted via the bypass-prefix whitelist."""
        rid = self._record(
            fp="fp_reconstruct",
            kwargs={
                "model_source": "graph_synthesis",
                "trust_label": "backfill_observation",
                "comparability_label": "reconstructed_init_variant",
                "loss_ratio": 0.5,
                "final_loss": 5.0,
            },
            experiment_type="synthesis",
        )
        self.assertTrue(rid)

    def test_debug_quality_gate_bypass_does_not_bypass_metric_completeness(
        self,
    ) -> None:
        exp_id = self.nb.start_experiment("synthesis", {}, "debug bypass test")
        with self.assertRaises(ValueError) as ctx:
            self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="fp_debug_bypass_loss_only",
                graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
                bypass_quality_gate=True,
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                model_source="graph_synthesis",
                loss_ratio=0.5,
                final_loss=5.0,
            )
        self.assertIn("missing post-S1 metrics", str(ctx.exception))

    def test_failed_ablation_write_not_blocked(self) -> None:
        """An ablation child that failed S1 (stage1_passed=False) is allowed
        through — the guardrail only applies to claimed-passing rows."""
        exp_id = self.nb.start_experiment("ablation", {}, "failed test")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_failed_ablation",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            error_type="OOM",
            error_message="cuda OOM",
            model_source="ablation",
            intentional_rerun_reason="ablation_counterfactual",
        )
        self.nb.flush_writes()
        self.assertTrue(rid)


if __name__ == "__main__":
    unittest.main()
