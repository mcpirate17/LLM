"""Smoke tests for the _run_inline_investigation split into sub-methods.

Verifies that the extracted methods exist, are callable, and compose
correctly on _ContinuousInvestigationMixin.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from research.scientist.runner.continuous_investigation import (
    _ContinuousInvestigationMixin,
)

pytestmark = pytest.mark.unit


class TestContinuousInvestigationSplitStructure(unittest.TestCase):
    """Verify the split produced the expected methods with correct signatures."""

    EXPECTED_METHODS = [
        "_inline_investigate_candidate_training",
        "_inline_investigate_fingerprint_completion",
        "_record_inline_investigation_candidate",
        "_inline_investigate_one_candidate",
        "_inline_investigation_loop",
        "_run_inline_investigation",
    ]

    def test_all_extracted_methods_exist_on_mixin(self):
        for name in self.EXPECTED_METHODS:
            self.assertTrue(
                hasattr(_ContinuousInvestigationMixin, name),
                f"Missing method: {name}",
            )
            self.assertTrue(
                callable(getattr(_ContinuousInvestigationMixin, name)),
                f"Not callable: {name}",
            )

    def test_no_method_exceeds_150_lines(self):
        import ast

        src_path = os.path.join(
            _RESEARCH_ROOT,
            "scientist",
            "runner",
            "continuous_investigation.py",
        )
        with open(src_path) as f:
            tree = ast.parse(f.read())

        # Only check the methods we split — pre-existing methods are out of scope
        split_methods = set(self.EXPECTED_METHODS)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "_ContinuousInvestigationMixin"
            ):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in split_methods:
                        end = item.end_lineno or item.lineno
                        lines = end - item.lineno + 1
                        self.assertLessEqual(
                            lines,
                            150,
                            f"{item.name} is {lines} lines (max 150)",
                        )


class TestContinuousInvestigationFingerprintCompletion(unittest.TestCase):
    """Smoke-test _inline_investigate_fingerprint_completion in isolation."""

    def _make_mixin_instance(self):
        """Create a bare mixin instance (no __init__ needed due to __slots__=())."""
        return _ContinuousInvestigationMixin()

    def test_fingerprint_skipped_when_model_is_none(self):
        mixin = self._make_mixin_instance()
        config = MagicMock()
        config.max_seq_len = 128
        config.model_dim = 64
        config.vocab_size = 1000
        nb = MagicMock()
        source = {"_behavioral_fingerprint": {"some": "data"}}

        attempted, completed, downgrade = (
            mixin._inline_investigate_fingerprint_completion(
                config=config,
                nb=nb,
                source=source,
                source_result_id="abc12345",
                best_inv_model=None,
                dev="cpu",
            )
        )
        self.assertFalse(attempted)
        self.assertFalse(completed)
        self.assertFalse(downgrade)

    def test_fingerprint_skipped_when_no_fp_dict(self):
        mixin = self._make_mixin_instance()
        config = MagicMock()
        nb = MagicMock()
        source = {}  # no _behavioral_fingerprint key
        fake_model = MagicMock()

        attempted, completed, downgrade = (
            mixin._inline_investigate_fingerprint_completion(
                config=config,
                nb=nb,
                source=source,
                source_result_id="abc12345",
                best_inv_model=fake_model,
                dev="cpu",
            )
        )
        self.assertFalse(attempted)
        self.assertFalse(completed)
        self.assertFalse(downgrade)


class _TestableInvestigationMixin(_ContinuousInvestigationMixin):
    """Concrete subclass that allows setting attributes (mixin uses __slots__=())."""

    pass


class TestInlineInvestigationLoop(unittest.TestCase):
    """Smoke-test _inline_investigation_loop delegates to _inline_investigate_one_candidate."""

    def test_loop_skips_candidates_before_resume_point(self):
        """If checkpoint says resume from candidate 2, candidates 0 and 1 are skipped."""
        mixin = _TestableInvestigationMixin()

        # Mock out all the attributes the loop touches
        mixin._stop_event = MagicMock()
        mixin._stop_event.is_set.return_value = False
        mixin.aria = MagicMock()
        mixin.aria.total_cost = 0
        mixin._update_progress = MagicMock()
        mixin._emit_event = MagicMock()
        mixin._inline_investigate_one_candidate = MagicMock()

        config = MagicMock()
        config.device = "cpu"
        config.investigation_steps = 100
        config.investigation_batch_size = 4
        config.stage1_steps = 50
        config.early_stop_patience = 10
        config.early_stop_min_steps = 5
        config.max_cost_dollars = 0
        config.n_training_programs = 1

        nb = MagicMock()

        result_ids = ["aaa", "bbb", "ccc"]
        inv_map = {
            "aaa": {"graph_json": "{}", "loss_ratio": 0.3},
            "bbb": {"graph_json": "{}", "loss_ratio": 0.25},
            "ccc": {"graph_json": "{}", "loss_ratio": 0.2},
        }

        # Checkpoint says resume from candidate 2 (skip 0 and 1)
        ckpt = MagicMock()
        ckpt.load_phase.return_value = {"candidate_idx": 2}

        with patch(
            "research.scientist.runner.continuous_investigation.CheckpointManager"
        ) as MockCM:
            MockCM.phase_resume_candidate_idx.return_value = 2

            with patch(
                "research.scientist.runner.continuous_investigation._build_source_map",
                return_value=inv_map,
            ):
                with patch(
                    "research.scientist.runner.continuous_investigation.resolve_device",
                    return_value="cpu",
                ):
                    mixin._inline_investigation_loop(
                        config,
                        nb,
                        result_ids,
                        inv_map,
                        "exp-1",
                        ckpt,
                    )

        # Only candidate "ccc" (index 2) should have been investigated
        self.assertEqual(mixin._inline_investigate_one_candidate.call_count, 1)
        call_kwargs = mixin._inline_investigate_one_candidate.call_args[1]
        self.assertEqual(call_kwargs["source_result_id"], "ccc")
        self.assertEqual(call_kwargs["prog_idx"], 2)


if __name__ == "__main__":
    unittest.main()
