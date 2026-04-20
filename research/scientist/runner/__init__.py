"""
Experiment Runner package.

Keep package import light so dashboard and API processes do not pull in the
full runner implementation, torch, or training stack unless a real run starts.
"""

from __future__ import annotations

import importlib
import threading as threading  # noqa: F401 - re-exported for test patching

from ._types import LiveProgress as LiveProgress, RunConfig as RunConfig

_EXPERIMENT_RUNNER_CLASS = None

__all__ = [
    "ExperimentRunner",
    "LiveProgress",
    "RunConfig",
    "_native_proactive_gating",
    "_native_runner_progress_report",
    "_rebuild_graph_with_overrides",
    "clear_gpu_memory",
    "propose_ablation_suite",
]


def __getattr__(name: str):
    if name == "ExperimentRunner":
        global _EXPERIMENT_RUNNER_CLASS
        if _EXPERIMENT_RUNNER_CLASS is not None:
            return _EXPERIMENT_RUNNER_CLASS

        from .core import _CoreMixin
        from ._lifecycle import _LifecycleMixin
        from .control_start import _ControlStartMixin
        from .control_actions import _ControlActionsMixin
        from .control_cycle import _ControlCycleMixin
        from .screening import _ScreeningMixin
        from .cycle import _CycleMixin
        from .continuous_loop import _ContinuousLoopMixin
        from .continuous_modes import _ContinuousModesMixin
        from .continuous_investigation import _ContinuousInvestigationMixin
        from .continuous_validation import _ContinuousValidationMixin
        from .continuous_inline_validation_phase7 import (
            _ContinuousInlineValidationPhase7Mixin,
        )
        from .execution_screening import _ExecutionScreeningMixin
        from .execution_screening_pipeline import _ExecutionScreeningPipelineMixin
        from .execution_investigation import _ExecutionInvestigationMixin
        from .execution_validation import _ExecutionValidationMixin
        from .execution_search import _ExecutionSearchMixin
        from .execution_training import _ExecutionTrainingMixin
        from .execution_candidates import _ExecutionCandidatesMixin
        from .execution_experiment_phase3 import _ExecutionExperimentPhase3Mixin
        from .execution_validation_phase3 import _ExecutionValidationPhase3Mixin
        from .execution_micro_train_phase3 import _ExecutionMicroTrainPhase3Mixin
        from .synthesis import _SynthesisMixin
        from .results_analysis import _ResultsAnalysisMixin
        from .results_automation import _ResultsAutomationMixin
        from .results_knowledge import _ResultsKnowledgeMixin
        from .results_auto_escalate_phase7 import _ResultsAutoEscalatePhase7Mixin
        from .selection import _SelectionMixin
        from .dashboard_panel import _DashboardPanelMixin
        from .dashboard_perf import _DashboardPerfMixin
        from .dashboard_hypothesis import _DashboardHypothesisMixin
        from .dashboard_orchestrator import _DashboardOrchestratorMixin

        class ExperimentRunner(
            _CoreMixin,
            _LifecycleMixin,
            _ControlStartMixin,
            _ControlActionsMixin,
            _ControlCycleMixin,
            _ScreeningMixin,
            _CycleMixin,
            _ContinuousLoopMixin,
            _ContinuousModesMixin,
            _ContinuousInvestigationMixin,
            _ContinuousValidationMixin,
            _ContinuousInlineValidationPhase7Mixin,
            _ExecutionScreeningMixin,
            _ExecutionScreeningPipelineMixin,
            _ExecutionInvestigationMixin,
            _ExecutionValidationMixin,
            _ExecutionSearchMixin,
            _ExecutionTrainingMixin,
            _ExecutionCandidatesMixin,
            _ExecutionExperimentPhase3Mixin,
            _ExecutionValidationPhase3Mixin,
            _ExecutionMicroTrainPhase3Mixin,
            _SynthesisMixin,
            _ResultsAnalysisMixin,
            _ResultsAutomationMixin,
            _ResultsKnowledgeMixin,
            _ResultsAutoEscalatePhase7Mixin,
            _SelectionMixin,
            _DashboardPanelMixin,
            _DashboardPerfMixin,
            _DashboardHypothesisMixin,
            _DashboardOrchestratorMixin,
        ):
            """Autonomous experiment execution engine with background support."""

        _EXPERIMENT_RUNNER_CLASS = ExperimentRunner
        return _EXPERIMENT_RUNNER_CLASS

    if name == "results":
        return importlib.import_module(".results_automation", __name__)

    if name in {
        "_native_proactive_gating",
        "_native_runner_progress_report",
        "_rebuild_graph_with_overrides",
        "clear_gpu_memory",
        "propose_ablation_suite",
    }:
        module = importlib.import_module("._helpers", __name__)
        return getattr(module, name)

    if name == "torch":
        import torch

        return torch

    allowed = {
        "control",
        "control_start",
        "control_cycle",
        "control_actions",
        "core",
        "screening",
        "cycle",
        "continuous",
        "execution",
        "synthesis",
        "selection",
        "dashboard",
        "results_auto_escalate_phase7",
    }
    if name in allowed:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
