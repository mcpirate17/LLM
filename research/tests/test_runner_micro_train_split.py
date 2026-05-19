"""Behavior tests for selected _micro_train helper methods."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch


class TestMicroTrainContext(unittest.TestCase):
    """Verify _MicroTrainContext dataclass works correctly."""

    def test_context_creation(self):
        from research.scientist.runner.execution_training import _MicroTrainContext

        ctx = _MicroTrainContext(
            model=MagicMock(),
            config=MagicMock(),
            dev=torch.device("cpu"),
            seed=42,
            graph_json="",
            graph_data=None,
            result={"passed": False},
            progress=MagicMock(),
            optimizer=MagicMock(),
            model_params=(),
            routing_modules=[],
            early_exit_modules=[],
            lm_head=None,
            norm=None,
            tracer=None,
            trace_totals_ms={"forward_pass": 0.0},
            starvation_detector=MagicMock(),
            op_profiler=MagicMock(),
            run_profiler=MagicMock(),
            use_synthesized_training=False,
            collect_curve=False,
            grad_clip_norm=1.0,
            total_steps=100,
            seq_len=128,
            random_mode=True,
            seed_int=42,
            t_start=0.0,
        )
        self.assertEqual(ctx.total_steps, 100)
        self.assertEqual(ctx.seed, 42)

    def test_trace_ctx_returns_nullcontext_when_no_tracer(self):
        from contextlib import nullcontext

        from research.scientist.runner.execution_training import _MicroTrainContext

        ctx = _MicroTrainContext(
            model=MagicMock(),
            config=MagicMock(),
            dev=torch.device("cpu"),
            seed=42,
            graph_json="",
            graph_data=None,
            result={"passed": False},
            progress=MagicMock(),
            optimizer=MagicMock(),
            model_params=(),
            routing_modules=[],
            early_exit_modules=[],
            lm_head=None,
            norm=None,
            tracer=None,  # No tracer
            trace_totals_ms={},
            starvation_detector=MagicMock(),
            op_profiler=MagicMock(),
            run_profiler=MagicMock(),
            use_synthesized_training=False,
            collect_curve=False,
            grad_clip_norm=1.0,
            total_steps=100,
            seq_len=128,
            random_mode=True,
            seed_int=42,
            t_start=0.0,
        )
        result = ctx.trace_ctx("test")
        self.assertIsInstance(result, nullcontext)


class TestAdaptiveBudget(unittest.TestCase):
    """Test _micro_train_adaptive_budget."""

    def test_no_graph_data_returns_base_steps(self):
        from research.scientist.runner.execution_training import (
            _ExecutionTrainingMixin,
        )

        mixin = _ExecutionTrainingMixin()
        config = MagicMock()
        config.stage1_steps = 500
        result = {}
        steps = mixin._micro_train_adaptive_budget(config, None, result)
        self.assertEqual(steps, 500)
        self.assertNotIn("adaptive_budget_novelty_bonus", result)


class TestPruningEval(unittest.TestCase):
    """Test _micro_train_pruning_eval skips when not configured."""

    def test_skips_when_no_final_loss(self):
        from research.scientist.runner.execution_training import (
            _ExecutionTrainingMixin,
        )

        mixin = _ExecutionTrainingMixin()
        result = {"passed": True}  # no final_loss
        config = MagicMock()
        config.one_shot_pruning_baseline = True
        mixin._micro_train_pruning_eval(
            MagicMock(), config, torch.device("cpu"), 42, result
        )
        self.assertNotIn("pruning_method", result)

    def test_skips_when_pruning_disabled(self):
        from research.scientist.runner.execution_training import (
            _ExecutionTrainingMixin,
        )

        mixin = _ExecutionTrainingMixin()
        result = {"passed": True, "final_loss": 1.5}
        config = MagicMock()
        config.one_shot_pruning_baseline = False
        mixin._micro_train_pruning_eval(
            MagicMock(), config, torch.device("cpu"), 42, result
        )
        self.assertNotIn("pruning_method", result)


if __name__ == "__main__":
    unittest.main()
