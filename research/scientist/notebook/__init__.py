"""Electronic Lab Notebook — composed from domain mixin modules.

Persistent, structured record of all experiments, hypotheses,
observations, and conclusions. Stored as SQLite for queryability
and served to the React dashboard via API.
"""
from __future__ import annotations

# Re-export public API from _shared
from ._shared import (
    ExperimentEntry,
    NOTEBOOK_SCHEMA,
    _PROGRAM_RESULTS_NEW_COLUMNS,
    infer_insight_identity,
    sanitize_for_db,
)

# Mixin imports
from .notebook_core import _NotebookCore
from .notebook_experiments import _ExperimentsMixin
from .notebook_programs import _ProgramsMixin
from .notebook_leaderboard import _LeaderboardMixin
from .notebook_campaigns import _CampaignsMixin
from .notebook_knowledge import _KnowledgeMixin
from .notebook_healer import _HealerMixin
from .notebook_chat import _ChatMixin
from .notebook_analytics import _AnalyticsMixin
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
    _MiscMixin,
):
    """Electronic lab notebook for the AI scientist.

    Composed from targeted mixins under notebook/ directory.
    """
    pass
