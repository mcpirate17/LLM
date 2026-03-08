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

import threading
import torch

from ._helpers import (
    _native_proactive_gating,
    _native_runner_progress_report,
    _rebuild_graph_with_overrides,
    propose_ablation_suite,
)
from ._types import RunConfig, LiveProgress

from .core import _CoreMixin
from .control import _ControlMixin
from .screening import _ScreeningMixin
from .cycle import _CycleMixin
from .continuous import _ContinuousMixin
from .execution import _ExecutionMixin
from .synthesis import _SynthesisMixin
from .results import _ResultsMixin
from .selection import _SelectionMixin
from .dashboard import _DashboardMixin


class ExperimentRunner(_CoreMixin, _ControlMixin, _ScreeningMixin, _CycleMixin, _ContinuousMixin, _ExecutionMixin, _SynthesisMixin, _ResultsMixin, _SelectionMixin, _DashboardMixin):
    """Autonomous experiment execution engine with background support.

    Composed from mixins in runner/ submodules.
    """
