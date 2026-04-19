"""Electronic Lab Notebook public API."""

from __future__ import annotations

from ._shared import (
    ExperimentEntry as ExperimentEntry,
    NOTEBOOK_SCHEMA as NOTEBOOK_SCHEMA,
    _PROGRAM_RESULTS_NEW_COLUMNS as _PROGRAM_RESULTS_NEW_COLUMNS,
    infer_insight_identity as infer_insight_identity,
    sanitize_for_db as sanitize_for_db,
)

__all__ = [
    "ExperimentEntry",
    "NOTEBOOK_SCHEMA",
    "_PROGRAM_RESULTS_NEW_COLUMNS",
    "infer_insight_identity",
    "sanitize_for_db",
    "LabNotebook",
]


_LAB_NOTEBOOK_CLASS = None


def __getattr__(name: str):
    global _LAB_NOTEBOOK_CLASS
    if name == "LabNotebook":
        if _LAB_NOTEBOOK_CLASS is not None:
            return _LAB_NOTEBOOK_CLASS
        from .notebook_core import _NotebookCore
        from .notebook_experiments import _ExperimentsMixin
        from .notebook_programs import _ProgramsMixin
        from .notebook_leaderboard import _LeaderboardMixin
        from .notebook_campaigns import _CampaignsMixin
        from .notebook_knowledge import _KnowledgeMixin
        from .notebook_healer import _HealerMixin
        from .notebook_chat import _ChatMixin
        from .notebook_analytics import _AnalyticsMixin
        from .notebook_advanced_analytics import _AdvancedAnalyticsMixin
        from .notebook_misc import _MiscMixin

        class LabNotebook(
            _NotebookCore,
            _ExperimentsMixin,
            _ProgramsMixin,
            _LeaderboardMixin,
            _CampaignsMixin,
            _KnowledgeMixin,
            _HealerMixin,
            _ChatMixin,
            _AnalyticsMixin,
            _AdvancedAnalyticsMixin,
            _MiscMixin,
        ):
            """Electronic lab notebook for the AI scientist."""

        _LAB_NOTEBOOK_CLASS = LabNotebook
        return _LAB_NOTEBOOK_CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
