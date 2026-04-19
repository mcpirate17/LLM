"""Smoke tests for the _micro_train split into phase methods."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch


class TestMicroTrainMethodsExist(unittest.TestCase):
    """Verify all extracted methods exist on the mixin."""

    def test_all_extracted_methods_exist(self):
        from research.scientist.runner.execution_training import (
            _ExecutionTrainingMixin,
        )

        expected = [
            "_micro_train",
            "_micro_train_build_context",
            "_micro_train_record_optimizer_info",
            "_micro_train_should_use_cuda_graph",
            "_micro_train_setup_optimizer",
            "_micro_train_adaptive_budget",
            "_micro_train_cuda_graph_capture",
            "_micro_train_cuda_graph_loop",
            "_micro_train_sample_data",
            "_micro_train_execute_step",
            "_micro_train_standard_loop",
            "_micro_train_post_step",
            "_micro_train_pruning_eval",
            "_micro_train_finalize_perf",
        ]
        for name in expected:
            with self.subTest(method=name):
                self.assertTrue(
                    hasattr(_ExecutionTrainingMixin, name),
                    f"{name} not found on _ExecutionTrainingMixin",
                )
                self.assertTrue(
                    callable(getattr(_ExecutionTrainingMixin, name)),
                    f"{name} is not callable",
                )


class TestMicroTrainMethodSizes(unittest.TestCase):
    """No newly-split method over 150 lines."""

    # Pre-existing methods that were already over 150 lines before this split.
    _PREEXISTING_LARGE = frozenset(
        {
            "_collect_post_training_metrics",
            "_run_post_s1_screening_probes",
            "_train_with_program",
        }
    )

    def test_no_split_method_exceeds_150_lines(self):
        src = (
            Path(__file__).parent.parent
            / "scientist"
            / "runner"
            / "execution_training.py"
        ).read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "_ExecutionTrainingMixin"
            ):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in self._PREEXISTING_LARGE:
                            continue
                        n_lines = item.end_lineno - item.lineno + 1
                        with self.subTest(method=item.name):
                            self.assertLessEqual(
                                n_lines,
                                150,
                                f"{item.name} is {n_lines} lines (max 150)",
                            )


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
