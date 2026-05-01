"""programs API route registration.

Thin orchestrator. The route handlers live in `programs_routes/` (split out
from the original 1884-line god file on 2026-04-30 to satisfy the no-god-files
rule in CLAUDE.md and the guardrail-audit pre-commit hook). This file only:
  1. imports the route handlers from the package
  2. registers them with the Flask app via the existing `register_notebook_routes` helper

If you're adding a new route, drop the handler into the topical submodule
(programs_routes/{program_detail,program_actions,validation_rerun,causal_ablation}.py)
and re-export it from `programs_routes/__init__.py`. Then add a tuple here.
"""

from __future__ import annotations

import logging

from .deps import ApiRouteContext
from ._utils import register_notebook_routes, with_notebook_context
from .programs_routes import (
    _api_bulk_causal_ablation_start,
    _api_causal_ablation_champions,
    _api_causal_ablation_children_for_rule,
    _api_causal_ablation_components,
    _api_causal_ablation_recommendations,
    _api_causal_ablation_summary,
    _api_construction_prior_active,
    _api_construction_prior_refresh,
    _api_drain_pending_validation_rerun,
    _api_program_backfill_loss,
    _api_program_backfill_metrics,
    _api_program_cancel_rerun,
    _api_program_causal_ablation,
    _api_program_causal_evidence,
    _api_program_detail,
    _api_program_explanation,
    _api_program_external_benchmarks,
    _api_program_lineage,
    _api_program_morph,
    _api_program_pending_reruns,
    _api_program_promote_screening,
    _api_program_queue_validation_rerun,
    _api_program_refine_analysis,
    _api_program_rescreen,
    _api_programs,
    _api_purge_junk_programs,
    _api_training_curve,
)

logger = logging.getLogger(__name__)


def _program_detail_routes(notebook_path: str):
    return (
        ("/api/programs/<result_id>", "api_program_detail", _api_program_detail),
        (
            "/api/programs/<result_id>/explanation",
            "api_program_explanation",
            _api_program_explanation,
            ("POST",),
        ),
        (
            "/api/programs/<result_id>/lineage",
            "api_program_lineage",
            _api_program_lineage,
        ),
        (
            "/api/programs/<result_id>/refine-analysis",
            "api_program_refine_analysis",
            _api_program_refine_analysis,
        ),
        ("/api/programs", "api_programs", _api_programs),
        (
            "/api/programs/<result_id>/training-curve",
            "api_training_curve",
            _api_training_curve,
        ),
    )


def _program_action_routes(notebook_path: str):
    return (
        (
            "/api/programs/<result_id>/morph",
            "api_program_morph",
            _api_program_morph,
            ("POST",),
        ),
        (
            "/api/programs/<result_id>/external-benchmarks",
            "api_program_external_benchmarks",
            _api_program_external_benchmarks,
            ("POST",),
        ),
        (
            "/api/programs/<result_id>/backfill-metrics",
            "api_program_backfill_metrics",
            _api_program_backfill_metrics,
            ("POST",),
            (notebook_path,),
        ),
        (
            "/api/programs/<result_id>/backfill-loss",
            "api_program_backfill_loss",
            _api_program_backfill_loss,
            ("POST",),
            (notebook_path,),
        ),
        (
            "/api/programs/<result_id>/rescreen",
            "api_program_rescreen",
            _api_program_rescreen,
            ("POST",),
            (notebook_path,),
        ),
        (
            "/api/programs/<result_id>/promote-screening",
            "api_program_promote_screening",
            _api_program_promote_screening,
            ("POST",),
        ),
        (
            "/api/programs/purge-junk",
            "api_purge_junk_programs",
            _api_purge_junk_programs,
            ("POST",),
        ),
    )


def _validation_rerun_routes(notebook_path: str):
    return (
        (
            "/api/programs/<result_id>/queue-validation-rerun",
            "api_program_queue_validation_rerun",
            _api_program_queue_validation_rerun,
            ("POST",),
        ),
        (
            "/api/programs/<result_id>/pending-reruns",
            "api_program_pending_reruns",
            _api_program_pending_reruns,
        ),
        (
            "/api/programs/<result_id>/pending-reruns/<task_id>/cancel",
            "api_program_cancel_rerun",
            _api_program_cancel_rerun,
            ("POST",),
        ),
        (
            "/api/runner/drain-pending-validation-rerun",
            "api_drain_pending_validation_rerun",
            _api_drain_pending_validation_rerun,
            ("POST",),
            (notebook_path,),
        ),
    )


def _causal_ablation_routes(notebook_path: str):
    return (
        (
            "/api/programs/<result_id>/causal-evidence",
            "api_program_causal_evidence",
            _api_program_causal_evidence,
        ),
        (
            "/api/programs/<result_id>/causal-ablation",
            "api_program_causal_ablation",
            _api_program_causal_ablation,
            ("POST",),
            (notebook_path,),
        ),
        (
            "/api/ablations/bulk/start",
            "api_bulk_causal_ablation_start",
            _api_bulk_causal_ablation_start,
            ("POST",),
            (notebook_path,),
        ),
        (
            "/api/ablations/causal-summary",
            "api_causal_ablation_summary",
            _api_causal_ablation_summary,
        ),
        (
            "/api/ablations/champions",
            "api_causal_ablation_champions",
            _api_causal_ablation_champions,
        ),
        (
            "/api/ablations/components",
            "api_causal_ablation_components",
            _api_causal_ablation_components,
        ),
        (
            "/api/ablations/recommendations",
            "api_causal_ablation_recommendations",
            _api_causal_ablation_recommendations,
        ),
        (
            "/api/ablations/children-for-rule",
            "api_causal_ablation_children_for_rule",
            _api_causal_ablation_children_for_rule,
        ),
        (
            "/api/ablations/construction-prior",
            "api_construction_prior_active",
            _api_construction_prior_active,
        ),
        (
            "/api/ablations/construction-prior/refresh",
            "api_construction_prior_refresh",
            _api_construction_prior_refresh,
            ("POST",),
        ),
    )


def register_programs_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    routes = (
        *_program_detail_routes(notebook_path),
        *_program_action_routes(notebook_path),
        *_validation_rerun_routes(notebook_path),
        *_causal_ablation_routes(notebook_path),
    )
    register_notebook_routes(app, wnb, routes)
