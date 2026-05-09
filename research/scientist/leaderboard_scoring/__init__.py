"""Leaderboard scoring — pure-arithmetic composite scoring formula.

Public API surface (preserved verbatim from the pre-2026-05 single-file
module): consumers may continue importing directly from this package as
``from research.scientist.leaderboard_scoring import compute_composite``.

Layout:
  _utils, _config        — leaves: utilities and loaded config dicts
  kwargs, pre_investigation, components, ar_validation, penalties
                         — independent scoring inputs
  champion_tiny          — 50pt champion protocol + v12 hard-failure gate
  generic                — _compute_composite_generic shared by all versions
  v10, v11, v12, v14     — version-specific composite functions
  compute                — public dispatcher
"""

from __future__ import annotations

from ._config import (
    ACTIVE_SCORING_VERSION,
    GEMINI_HARD_GATE_ERF_DENSITY,
    GEMINI_HARD_GATE_ERF_VARIANCE,
    _V10_CONFIG,
    _V11_CONFIG,
    _V14_CONFIG,
)
from ._utils import compute_efficiency_multiple
from .champion_tiny import (
    CHAMPION_INDUCTION_V3_PROTOCOLS,
    CHAMPION_TINY_MODEL_SCORE_V1,
    compute_champion_tiny_model_score_v1,
)
from .components import _score_understanding_v8
from .compute import (
    composite_score_ceiling,
    compute_composite,
    get_scoring_version,
)
from .kwargs import (
    _PR_SELECT_COLS,
    _pr_dict_to_score_kwargs,
    build_score_kwargs,
    build_score_kwargs_from_prefetch,
    prefetch_program_results,
)
from .pre_investigation import compute_pre_investigation_score
from .v10 import compute_composite_v10
from .v11 import compute_composite_v11
from .v12 import compute_composite_v12
from .v14 import compute_composite_v14


__all__ = [
    # Public dispatcher
    "compute_composite",
    "composite_score_ceiling",
    "get_scoring_version",
    "ACTIVE_SCORING_VERSION",
    # Versioned composites (used by tests + leaderboard rescore tooling)
    "compute_composite_v10",
    "compute_composite_v11",
    "compute_composite_v12",
    "compute_composite_v14",
    # Loaded weight configs (re-exported for tooling that introspects them)
    "_V10_CONFIG",
    "_V11_CONFIG",
    "_V14_CONFIG",
    # Score-kwargs builders
    "build_score_kwargs",
    "build_score_kwargs_from_prefetch",
    "prefetch_program_results",
    # Public utilities
    "compute_efficiency_multiple",
    "compute_pre_investigation_score",
    # Champion tiny-model 50pt protocol
    "compute_champion_tiny_model_score_v1",
    "CHAMPION_TINY_MODEL_SCORE_V1",
    "CHAMPION_INDUCTION_V3_PROTOCOLS",
    # Pre-investigation hard gates (read by notebook_references)
    "GEMINI_HARD_GATE_ERF_DENSITY",
    "GEMINI_HARD_GATE_ERF_VARIANCE",
    # Test/internal-imported privates re-exported defensively to preserve
    # the existing import surface (notebook_leaderboard, test_*).
    "_PR_SELECT_COLS",
    "_pr_dict_to_score_kwargs",
    "_score_understanding_v8",
]
