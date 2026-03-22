"""
Experiment Runner

The autonomous experiment execution engine. Aria uses this to:
1. Generate batches of synthesized programs
2. Evaluate them through the funnel
3. Record results in the lab notebook
4. Analyze patterns and formulate new hypotheses
5. Adjust strategy based on outcomes

Supports background execution controlled from the dashboard.
"""

from __future__ import annotations

import importlib
import threading as threading  # noqa: F401 — re-exported for test patching
import torch as torch  # noqa: F401 — re-exported for test patching

from ._helpers import (
    _native_proactive_gating as _native_proactive_gating,
    _native_runner_progress_report as _native_runner_progress_report,
    _rebuild_graph_with_overrides as _rebuild_graph_with_overrides,
    clear_gpu_memory as clear_gpu_memory,
    propose_ablation_suite as propose_ablation_suite,
)
from ._types import RunConfig as RunConfig, LiveProgress as LiveProgress


# Core infrastructure
from .core import _CoreMixin

# Control — split mixins replace monolith control.py
from .control_start import _ControlStartMixin
from .control_actions import _ControlActionsMixin
from .control_cycle import _ControlCycleMixin

# Screening & cycle
from .screening import _ScreeningMixin
from .cycle import _CycleMixin

# Continuous — split mixins replace monolith continuous.py
from .continuous_loop import _ContinuousLoopMixin
from .continuous_modes import _ContinuousModesMixin
from .continuous_investigation import _ContinuousInvestigationMixin
from .continuous_validation import _ContinuousValidationMixin
from .continuous_inline_validation_phase7 import _ContinuousInlineValidationPhase7Mixin

# Execution — split mixins replace monolith execution.py
from .execution_screening import _ExecutionScreeningMixin
from .execution_investigation import _ExecutionInvestigationMixin
from .execution_validation import _ExecutionValidationMixin
from .execution_search import _ExecutionSearchMixin
from .execution_training import _ExecutionTrainingMixin
from .execution_candidates import _ExecutionCandidatesMixin
from .execution_experiment_phase3 import _ExecutionExperimentPhase3Mixin
from .execution_validation_phase3 import _ExecutionValidationPhase3Mixin
from .execution_micro_train_phase3 import _ExecutionMicroTrainPhase3Mixin

# Synthesis
from .synthesis import _SynthesisMixin

# Results — split mixins replace monolith results.py
from .results_analysis import _ResultsAnalysisMixin
from .results_automation import _ResultsAutomationMixin
from .results_knowledge import _ResultsKnowledgeMixin
from .results_auto_escalate_phase7 import _ResultsAutoEscalatePhase7Mixin

# Selection & Dashboard
from .selection import _SelectionMixin
from .dashboard import _DashboardMixin


class ExperimentRunner(
    _CoreMixin,
    # Control splits
    _ControlStartMixin,
    _ControlActionsMixin,
    _ControlCycleMixin,
    # Screening & cycle
    _ScreeningMixin,
    _CycleMixin,
    # Continuous splits
    _ContinuousLoopMixin,
    _ContinuousModesMixin,
    _ContinuousInvestigationMixin,
    _ContinuousValidationMixin,
    _ContinuousInlineValidationPhase7Mixin,
    # Execution splits
    _ExecutionScreeningMixin,
    _ExecutionInvestigationMixin,
    _ExecutionValidationMixin,
    _ExecutionSearchMixin,
    _ExecutionTrainingMixin,
    _ExecutionCandidatesMixin,
    _ExecutionExperimentPhase3Mixin,
    _ExecutionValidationPhase3Mixin,
    _ExecutionMicroTrainPhase3Mixin,
    # Synthesis
    _SynthesisMixin,
    # Results splits
    _ResultsAnalysisMixin,
    _ResultsAutomationMixin,
    _ResultsKnowledgeMixin,
    _ResultsAutoEscalatePhase7Mixin,
    # Selection & Dashboard
    _SelectionMixin,
    _DashboardMixin,
):
    """Autonomous experiment execution engine with background support.

    Composed from split mixins in runner/ submodules.
    Monolith files (execution.py, continuous.py, control.py, results.py)
    are retained for reference but no longer in the MRO.
    """


def __getattr__(name: str):
    if name == "results":
        return importlib.import_module(".results_automation", __name__)
    # Allow accessing submodules by name for patching in tests.
    _allowed = {
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
    if name in _allowed:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
