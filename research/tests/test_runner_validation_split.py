"""Smoke tests for the execution_validation.py split.

Verifies that extracted methods exist on _ExecutionValidationMixin and
that the mixin can be composed into a minimal stub class.
"""

import inspect
import os
import sys
import unittest

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from research.scientist.runner.execution_validation import (
    _ExecutionValidationMixin,
)

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


if __name__ == "__main__":
    unittest.main()
