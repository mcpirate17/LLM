"""programs_bp split into focused submodules.

Re-exports every route handler so programs_bp.py can register them. Splitting
the original 1884-line god file is what unblocks the guardrail audit.
"""

from .program_detail import (
    _api_program_detail,
    _api_program_explanation,
    _api_program_lineage,
    _api_program_refine_analysis,
    _api_programs,
    _api_training_curve,
)
from .program_actions import (
    _api_program_morph,
    _api_program_external_benchmarks,
    _api_program_backfill_metrics,
    _api_program_backfill_loss,
    _api_program_rescreen,
    _api_program_promote_screening,
    _api_purge_junk_programs,
)
from .validation_rerun import (
    _api_program_queue_validation_rerun,
    _api_program_pending_reruns,
    _api_drain_pending_validation_rerun,
    _api_program_cancel_rerun,
)
from .causal_ablation import (
    _api_program_causal_evidence,
    _api_program_causal_ablation,
    _api_bulk_causal_ablation_start,
    _api_causal_ablation_summary,
    _api_causal_ablation_champions,
    _api_causal_ablation_components,
    _api_causal_ablation_recommendations,
    _api_causal_ablation_children_for_rule,
    _api_construction_prior_active,
    _api_construction_prior_refresh,
)

__all__ = [
    "_api_program_detail",
    "_api_program_explanation",
    "_api_program_lineage",
    "_api_program_refine_analysis",
    "_api_programs",
    "_api_training_curve",
    "_api_program_morph",
    "_api_program_external_benchmarks",
    "_api_program_backfill_metrics",
    "_api_program_backfill_loss",
    "_api_program_rescreen",
    "_api_program_promote_screening",
    "_api_purge_junk_programs",
    "_api_program_queue_validation_rerun",
    "_api_program_pending_reruns",
    "_api_drain_pending_validation_rerun",
    "_api_program_cancel_rerun",
    "_api_program_causal_evidence",
    "_api_program_causal_ablation",
    "_api_bulk_causal_ablation_start",
    "_api_causal_ablation_summary",
    "_api_causal_ablation_champions",
    "_api_causal_ablation_components",
    "_api_causal_ablation_recommendations",
    "_api_causal_ablation_children_for_rule",
    "_api_construction_prior_active",
    "_api_construction_prior_refresh",
]
