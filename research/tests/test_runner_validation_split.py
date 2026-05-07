"""Smoke tests for the execution_validation.py split.

Verifies that extracted methods exist on _ExecutionValidationMixin and
that the mixin can be composed into a minimal stub class.
"""

import inspect
import os
import sys
import unittest
from unittest.mock import patch

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from research.scientist.runner.execution_validation import (
    _ExecutionValidationMixin,
)
from research.scientist.runner._types import RunConfig

pytestmark = pytest.mark.unit


class TestExtractedMethodsExist(unittest.TestCase):
    """All extracted helpers must be present on the mixin class."""

    EXPECTED_METHODS = [
        # shared error handling
        "_handle_thread_error",
        "_handle_thread_fatal",
        # validation thread extractions
        "_run_single_validation_candidate",
        "_validation_cka_check",
        "_record_validation_candidate",
        "_validation_promote_and_record",
        "_validation_baseline_comparisons",
        "_validation_record_and_checkpoint",
        # scale-up thread extractions
        "_scale_up_candidate",
        "_scale_up_fetch_and_compile",
        "_scale_up_train",
        "_scale_up_collect_training_metrics",
        "_scale_up_baseline_comparison",
        "_scale_up_evals",
        "_scale_up_novelty",
        "_scale_up_record_result",
    ]

    def test_all_extracted_methods_exist(self):
        for name in self.EXPECTED_METHODS:
            with self.subTest(method=name):
                attr = getattr(_ExecutionValidationMixin, name, None)
                self.assertIsNotNone(
                    attr, f"{name} missing from _ExecutionValidationMixin"
                )
                self.assertTrue(callable(attr), f"{name} is not callable")

    def test_original_thread_methods_still_exist(self):
        for name in ("_run_validation_thread", "_run_scale_up_thread"):
            attr = getattr(_ExecutionValidationMixin, name, None)
            self.assertIsNotNone(attr, f"{name} missing after refactor")

    def test_no_method_exceeds_150_lines(self):
        """All methods on the mixin must be <= 150 lines."""
        for name, method in inspect.getmembers(
            _ExecutionValidationMixin, predicate=inspect.isfunction
        ):
            if name.startswith("_"):
                src = inspect.getsource(method)
                n_lines = len(src.splitlines())
                self.assertLessEqual(
                    n_lines,
                    150,
                    f"{name} is {n_lines} lines (limit 150)",
                )


class TestMixinComposition(unittest.TestCase):
    """The mixin must compose into a concrete class without errors."""

    def test_mixin_has_slots(self):
        self.assertEqual(_ExecutionValidationMixin.__slots__, ())

    def test_mixin_composes_into_stub(self):
        class _Stub(_ExecutionValidationMixin):
            pass

        obj = _Stub()
        self.assertIsInstance(obj, _ExecutionValidationMixin)
        for name in TestExtractedMethodsExist.EXPECTED_METHODS:
            self.assertTrue(hasattr(obj, name))


class TestChampionConfirmationPolicy(unittest.TestCase):
    def test_confirmation_survivor_is_not_novelty_gated(self):
        class _Stub(_ExecutionValidationMixin):
            def __init__(self):
                self.events = []

            def _resolve_novelty_promotion_validity(self, *_args):
                return False, "duplicate_champion", False

            def _emit_event(self, event_type, payload):
                self.events.append((event_type, payload))

        class _Graph:
            def fingerprint(self):
                return "fp_parent"

        class _Novelty:
            novelty_valid_for_promotion = False
            novelty_validity_reason = "duplicate_champion"
            structural_novelty = 0.1
            behavioral_novelty = 0.2
            novelty_confidence = 0.3
            most_similar_to = "parent"

        class _Notebook:
            conn = None

            def get_program_detail(self, result_id):
                return {"graph_fingerprint": "fp_parent"}

            def record_program_result(self, **kwargs):
                self.recorded = kwargs
                return "child-confirm"

            def store_training_curve(self, *_args):
                raise AssertionError("no curve should be stored in this test")

        config = RunConfig(mode="confirmation")
        results = {
            "novel_count": 0,
            "confirmed_count": 0,
            "survivors": [],
            "best_loss_ratio": None,
            "best_novelty_score": None,
        }
        nb = _Notebook()

        with patch(
            "research.scientist.runner.execution_validation_scale.graph_to_json",
            return_value="{}",
        ):
            _Stub()._scale_up_record_result(
                exp_id="exp-confirm",
                source_result_id="parent-rid",
                prog_idx=0,
                total=1,
                config=config,
                nb=nb,
                results=results,
                graph=_Graph(),
                model=None,
                s1_passed=True,
                loss_ratio=0.53,
                final_loss=6.3,
                throughput=None,
                training_curve=None,
                n_score=0.1,
                nov=_Novelty(),
                program_metrics={},
            )

        self.assertEqual(results["confirmed_count"], 1)
        self.assertEqual(results["novel_count"], 1)
        self.assertTrue(results["survivors"][0]["confirmation"])
        self.assertEqual(
            nb.recorded["intentional_rerun_reason"], "champion_confirmation"
        )
        self.assertEqual(nb.recorded["graph_fingerprint"], "fp_parent")


if __name__ == "__main__":
    unittest.main()
