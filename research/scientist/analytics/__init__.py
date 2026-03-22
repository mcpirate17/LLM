"""Experiment Analytics — Learning Feedback Engine.

Analyzes experiment history to learn which operations, structures, and
combinations correlate with success. Feeds back into grammar weights
to improve synthesis over time.

Split into domain mixins under analytics/ directory.
"""

from __future__ import annotations

from typing import Optional, Dict

from .analytics_ops import _OpsMixin
from .analytics_grammar import _GrammarMixin
from .analytics_routing import _RoutingMixin
from .analytics_experiments import _ExperimentsMixin
from .analytics_campaigns import _CampaignsMixin
from .analytics_refinement import RefinementAnalyzer


class ExperimentAnalytics(
    _OpsMixin,
    _GrammarMixin,
    _RoutingMixin,
    _ExperimentsMixin,
    _CampaignsMixin,
):
    """Data-driven analytics over experiment history.

    Composed from targeted mixins under analytics/ directory.
    """

    __slots__ = ("nb", "_last_grammar_weight_diagnostics")

    LEARNING_TRAJECTORY_MIN_EXPERIMENTS = 5
    FINGERPRINT_WEIGHT_CAP = 3.0

    def __init__(self, notebook):
        self.nb = notebook
        self._last_grammar_weight_diagnostics: Optional[Dict] = None


__all__ = ["ExperimentAnalytics", "RefinementAnalyzer"]
