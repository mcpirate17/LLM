#!/usr/bin/env python
"""Backfill under-sampled templates with targeted experiments.

Usage:
    python -m research.tools.backfill_templates [--target 50] [--device cuda]
    python -m research.tools.backfill_templates --dry-run
    python -m research.tools.backfill_templates --templates arch_router_block compute_budget_block

Uses the full screening pipeline (fingerprint, novelty, wikitext, leaderboard)
but skips LLM hypothesis/summary calls.  Results appear on the dashboard and
leaderboard like any other experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools._db_maintenance import connect_readonly

DB_PATH = Path(__file__).resolve().parents[1] / "runs.db"
_VALID_TARGET_METRICS = ("eval", "s0", "s1")
_VALID_WEIGHT_MODES = ("uniform", "random", "default", "scaffold_guided")
_VALID_PHASES = ("isolation", "stack")
_VALID_POLICY_MODES = ("off", "auto")
_VALID_TEMPLATE_MODES = ("rehab", "coverage", "harvest", "frozen")

_NON_ROUTING_TEMPLATES = {
    "gpt2_reference",
    "mamba_reference",
    "residual_block",
    "sequential",
    "transformer_block",
    "spiking_stdp_block",
    "spiking_residual_block",
    "rwkv_block",
    "rwkv_double_norm",
    "rwkv_sparse_chain",
    "token_merge_block",
    "token_merge_conv",
    "sparse_ffn",
    "fused_gelu_ffn",
    "bottleneck",
    "normalized_matmul",
    "gated_product",
    "gated_residual",
    "dense_cascade",
    # Attention templates without routing ops
    "attn_residual_block",
    "attn_gated_residual",
    "attn_cross_dim",
    "attn_multi_head_mix",
    "latent_attn_ffn_block",
    "local_attn_ffn_block",
    "diff_attn_ffn_block",
    "linear_attn_ffn_block",
    "latent_attn_sparse_ffn",
    "local_attn_swiglu",
    "diff_attn_gated_ffn",
    "graph_attn_ffn_block",
    "attn_ssm_hybrid",
    "attn_conv_hybrid",
    "attn_rwkv_hybrid",
    "attn_bottleneck_hybrid",
    "dual_attn_block",
    "attn_state_space_hybrid",
    "cascaded_attn_ffn",
    "attn_exp_gated",
    "attn_reciprocal_gated",
    "attn_decay_sequence",
    "attn_gated_product",
    "attn_chebyshev_hybrid",
    "attn_kronecker_hybrid",
    "attn_log_gated",
    "attn_gated_maximum",
    "attn_hyperbolic",
    "attn_spectral_filter",
    "attn_normalized_matmul",
    "attn_softmax_normalized_matmul",
    "attn_softmax_normalized_matmul_v2",
    "attn_softmax_normalized_matmul_compact_ffn",
    "attn_softmax_normalized_matmul_fixed_tail_norm",
    "attn_linear_no_matmul_ffn",
    "attn_linear_no_matmul_ffn_dense_tail",
    "attn_linear_no_matmul_ffn_direct_recovery",
    "latent_attn_conv_hybrid",
    "diff_attn_conv_hybrid",
    "attn_safe_division",
    "latent_attn_ssm_hybrid",
    "local_attn_ssm_hybrid",
    "attn_spiking_hybrid",
    "linear_attn_sparse_ffn",
    "graph_attn_sparse_ffn",
    # New attention-style blocks use mixing ops directly but no routing ops.
    "difficulty_routed_attention_block",
    "strided_attention_block",
    "gated_progressive_attention_block",
    "gated_linear_attention_block",
    "long_conv_hyena_block",
    "associative_memory_block",
    "mixture_of_recursions_block",
    "codex_ssm_retention_block",
    "codex_ssm_delta_memory_block",
    "codex_ssm_mla_gated_block",
    "codex_ssm_local_recall_block",
}

_STACK_PHASE_OVERRIDES: dict[str, dict[str, Any]] = {
    # This template already consumes nearly the full screening depth budget
    # on its own; composing two copies in stack mode produces validator
    # failures ("Too deep: 19/20 > 18") rather than useful backfill data.
    "hybrid_sparse_triplet_router": {
        "composition_depth": 1,
    },
    # This multiscale router has three medium branches plus a hard expert path.
    # Stacking two copies in stack mode creates coupled routing graphs that
    # fail screening even though a single instance validates cleanly.
    "multiscale_difficulty_router": {
        "composition_depth": 1,
    },
    # Even one attn_safe_division block is already fairly dense. Stacking two
    # copies in stack mode breaches the screening op budget instead of yielding
    # useful evidence about the division scaffold itself.
    "attn_safe_division": {
        "composition_depth": 1,
    },
    "multiscale_rich_lane_router": {
        "composition_depth": 1,
    },
    "intelligent_multilane_router": {
        "composition_depth": 1,
    },
}


@dataclass(frozen=True)
class TemplateBackfillPolicy:
    mode: str = "coverage"
    min_batch: int = 15
    max_batch: int | None = None
    max_retries: int = 5
    preferred_weight_mode: str | None = None
    structural_stop_error: str | None = None
    zero_s1_stop_after_s0: int | None = None
    notes: str = ""


_DEFAULT_TEMPLATE_POLICY = TemplateBackfillPolicy()

_TEMPLATE_BACKFILL_POLICIES: dict[str, TemplateBackfillPolicy] = {
    "intelligent_multilane_router": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=10,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        structural_stop_error="causality_violation",
        zero_s1_stop_after_s0=8,
        notes="Structural causality fix landed; continue coverage with neutral sampling and stop quickly if conversion regresses.",
    ),
    "hybrid_sparse_triplet_router": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=12,
        max_batch=24,
        max_retries=4,
        notes="Search-space widening landed; gather coverage with moderate batches.",
    ),
    "recursive_depth_router": TemplateBackfillPolicy(
        mode="harvest",
        min_batch=12,
        max_batch=20,
        max_retries=3,
        preferred_weight_mode="scaffold_guided",
        zero_s1_stop_after_s0=10,
        notes="Stable production winner; bias toward high-yield harvest runs.",
    ),
    "multiscale_rich_lane_router": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=8,
        max_batch=12,
        max_retries=2,
        notes="Secondary family: allow low-priority coverage backfill, but do not prioritize harvest budget here.",
    ),
    "multiscale_difficulty_router": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=8,
        max_batch=12,
        max_retries=2,
        notes="Secondary family: allow low-priority coverage backfill, but keep batches capped.",
    ),
    "depth_gated_block_matmul_norm": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=8,
        max_batch=16,
        max_retries=3,
        preferred_weight_mode="scaffold_guided",
        notes="Promoted experimental family: gather coverage carefully after the matmul+rmsnorm variant showed repeated low-loss survivors.",
    ),
    "attn_safe_division": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=10,
        max_batch=20,
        max_retries=3,
        preferred_weight_mode="scaffold_guided",
        notes="Sparse but clean early evidence; gather more coverage before changing weights.",
    ),
    "latent_attn_ssm_hybrid": TemplateBackfillPolicy(
        mode="harvest",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="High-yield frontier family; use neutral sampling for trusted low-loss harvest.",
    ),
    "local_attn_ssm_hybrid": TemplateBackfillPolicy(
        mode="harvest",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="High-yield frontier family; use neutral sampling for trusted low-loss harvest.",
    ),
    "attn_routing_block": TemplateBackfillPolicy(
        mode="harvest",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="Promoted routing winner; preserve neutral sampling while chasing frontier loss.",
    ),
    "linear_attn_ffn_block": TemplateBackfillPolicy(
        mode="harvest",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="Low-loss linear attention family; preserve neutral sampling while chasing frontier loss.",
    ),
    "diff_attn_conv_hybrid": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="Frontier-adjacent hybrid family; keep neutral sampling until stronger conversion is established.",
    ),
    "attn_softmax_normalized_matmul_compact_ffn": TemplateBackfillPolicy(
        mode="coverage",
        min_batch=16,
        max_batch=24,
        max_retries=3,
        preferred_weight_mode="uniform",
        notes="Seed-sensitive attention-tail family; keep neutral sampling while validating frontier behavior.",
    ),
}


def get_template_stats(db_path: Path) -> dict[str, dict[str, int]]:
    """Return per-template eval/S0/S1 counts from live program_results rows."""
    db = connect_readonly(db_path)
    stats: dict[str, dict[str, int]] = {}

    try:
        rows = db.execute(
            "SELECT graph_json, stage0_passed, stage1_passed "
            "FROM program_results_compat "
            "WHERE graph_json IS NOT NULL"
        ).fetchall()
        for gj, s0, s1 in rows:
            try:
                gj = resolve_graph_json_value(db, db_path, gj)
                g = json.loads(gj)
                for t in set(g.get("metadata", {}).get("templates_used", [])):
                    bucket = stats.setdefault(t, {"eval": 0, "s0": 0, "s1": 0})
                    bucket["eval"] += 1
                    bucket["s0"] += 1 if s0 else 0
                    bucket["s1"] += 1 if s1 else 0
            except (json.JSONDecodeError, TypeError):
                logger.debug("Skipping invalid graph_json while loading template stats")
    except Exception as exc:
        logger.warning("program_results scan failed: %s", exc)
        try:
            rows = db.execute(
                "SELECT template_name, eval_count, s0_pass_count, s1_pass_count "
                "FROM template_stats"
            ).fetchall()
            for name, ev, s0, s1 in rows:
                stats[name] = {
                    "eval": int(ev or 0),
                    "s0": int(s0 or 0),
                    "s1": int(s1 or 0),
                }
        except Exception:
            logger.debug("template_stats fallback table unavailable")
    finally:
        db.close()

    return stats


def get_template_counts(db_path: Path, metric: str = "eval") -> Counter:
    """Count template observations using the requested metric."""
    if metric not in _VALID_TARGET_METRICS:
        raise ValueError(f"Unsupported target metric: {metric}")

    counts: Counter = Counter()
    for name, tpl_stats in get_template_stats(db_path).items():
        counts[name] = int(tpl_stats.get(metric, 0))
    return counts


def _fmt_stats(stats: dict[str, int] | None) -> str:
    stats = stats or {}
    return (
        f"eval={int(stats.get('eval', 0)):3d} "
        f"s0={int(stats.get('s0', 0)):3d} "
        f"s1={int(stats.get('s1', 0)):3d}"
    )


def _make_category_weights(mode: str) -> dict[str, float] | None:
    """Build category weights for backfill grammar config."""
    if mode == "default":
        return None  # use GrammarConfig defaults
    from research.synthesis.grammar import GrammarConfig

    cats = list(GrammarConfig().category_weights)
    if mode == "uniform":
        return {k: 1.0 for k in cats}
    if mode == "random":
        return {k: round(random.uniform(0.3, 3.0), 2) for k in cats}
    return None


def _scaffold_guided_priors(
    db_path: str,
    *,
    min_support: int = 5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build op/category priors from scaffold profiling evidence."""
    from research.scientist.notebook import LabNotebook
    from research.synthesis.primitives import get_primitive

    nb = LabNotebook(db_path)
    try:
        stats = nb.get_scaffold_component_stats(min_support=min_support)
    finally:
        nb.close()

    op_weights: dict[str, float] = {}
    category_buckets: dict[str, list[float]] = {}
    for op_name, stat in stats.items():
        support = int(stat.get("support") or 0)
        prior_rate = float(stat.get("prior_rate") or 0.5)
        confidence = min(1.0, support / 12.0)
        weight = 1.0 + ((prior_rate - 0.5) * 2.4 * confidence)
        clamped = round(max(0.35, min(2.5, weight)), 3)
        op_weights[op_name] = clamped
        try:
            category = get_primitive(op_name).category.value
        except (KeyError, ValueError):
            continue
        category_buckets.setdefault(category, []).append(clamped)

    category_weights = {
        category: round(sum(weights) / len(weights), 3)
        for category, weights in category_buckets.items()
        if weights
    }
    return op_weights, category_weights


def _phase_settings(phase: str, template_name: str | None = None) -> dict[str, Any]:
    """Backfill settings for isolation vs stack validation."""
    if phase == "stack":
        settings = {
            "composition_depth": 2,
            "n_layers": 3,
            "stage1_steps": 750,
        }
        if template_name and template_name in _STACK_PHASE_OVERRIDES:
            settings.update(_STACK_PHASE_OVERRIDES[template_name])
        return settings
    return {
        "composition_depth": 1,
        "n_layers": 2,
        "stage1_steps": 500,
    }


def get_template_backfill_policy(template_name: str) -> TemplateBackfillPolicy:
    """Return the resolved per-template backfill policy."""
    policy = _TEMPLATE_BACKFILL_POLICIES.get(template_name, _DEFAULT_TEMPLATE_POLICY)
    if policy.mode not in _VALID_TEMPLATE_MODES:
        raise ValueError(f"Unsupported template backfill mode: {policy.mode}")
    return policy


def resolve_weight_mode(
    template_name: str,
    requested_weight_mode: str,
    policy_mode: str = "auto",
) -> str:
    """Resolve the effective weight mode after policy overrides."""
    if policy_mode not in _VALID_POLICY_MODES:
        raise ValueError(f"Unsupported backfill policy mode: {policy_mode}")
    if requested_weight_mode not in _VALID_WEIGHT_MODES:
        raise ValueError(f"Unsupported weight mode: {requested_weight_mode}")
    if policy_mode == "off":
        return requested_weight_mode
    preferred = get_template_backfill_policy(template_name).preferred_weight_mode
    return preferred or requested_weight_mode


def plan_batch_size(
    *,
    template_name: str,
    requested_batch_size: int,
    metric_deficit: int,
    s1_deficit: int,
    policy_mode: str = "auto",
) -> int:
    """Compute the next batch size under the template policy."""
    if policy_mode not in _VALID_POLICY_MODES:
        raise ValueError(f"Unsupported backfill policy mode: {policy_mode}")
    if policy_mode == "off":
        return max(requested_batch_size, metric_deficit, s1_deficit)
    policy = get_template_backfill_policy(template_name)
    if policy.mode == "frozen":
        return 0
    batch = max(requested_batch_size, metric_deficit, s1_deficit, policy.min_batch)
    if policy.max_batch is not None:
        batch = min(batch, policy.max_batch)
    return batch


def summarize_experiment_batch(db_path: str, experiment_id: str) -> dict[str, Any]:
    """Return compact persisted-row outcomes for a completed or partial batch."""
    db = connect_readonly(Path(db_path))
    try:
        row = db.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN stage0_passed THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN stage1_passed THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN rapid_screening_passed THEN 1 ELSE 0 END), 0)
            FROM program_results_compat
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
        rows, s0, s1, rapid = (int(v or 0) for v in (row or (0, 0, 0, 0)))
        error_counts = {
            str(name): int(count)
            for name, count in db.execute(
                """
                SELECT COALESCE(error_type, 'none'), COUNT(*)
                FROM program_results_compat
                WHERE experiment_id = ?
                GROUP BY COALESCE(error_type, 'none')
                """,
                (experiment_id,),
            ).fetchall()
        }
        return {
            "rows": rows,
            "s0": s0,
            "s1": s1,
            "rapid": rapid,
            "error_counts": error_counts,
        }
    finally:
        db.close()


def should_stop_backfill_attempt(
    *,
    template_name: str,
    batch_summary: dict[str, Any],
    policy_mode: str = "auto",
) -> tuple[bool, str | None]:
    """Return whether the template should stop further backfill attempts."""
    if policy_mode == "off":
        return False, None
    policy = get_template_backfill_policy(template_name)
    if policy.mode == "frozen":
        return True, "template_frozen"
    rows = int(batch_summary.get("rows", 0) or 0)
    s0 = int(batch_summary.get("s0", 0) or 0)
    s1 = int(batch_summary.get("s1", 0) or 0)
    error_counts = batch_summary.get("error_counts") or {}
    if (
        policy.structural_stop_error
        and rows > 0
        and int(error_counts.get(policy.structural_stop_error, 0) or 0) * 2 >= rows
    ):
        return True, f"structural_stop:{policy.structural_stop_error}"
    if (
        policy.zero_s1_stop_after_s0 is not None
        and s0 >= policy.zero_s1_stop_after_s0
        and s1 == 0
    ):
        return True, "zero_s1_conversion"
    return False, None


def _start_backfill_experiment_with_retry(
    runner: Any,
    *,
    experiment_type: str,
    config: dict[str, Any],
    hypothesis: str,
    hypothesis_metadata: dict[str, Any],
    created_by: str,
    max_attempts: int = 6,
) -> tuple[str, Any]:
    """Start a preregistered experiment, retrying transient SQLite lock contention."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        nb = runner._make_notebook()
        try:
            runner._populate_refuted_cache(nb)
            exp_id = runner._start_preregistered_experiment(
                nb=nb,
                experiment_type=experiment_type,
                config=config,
                hypothesis=hypothesis,
                hypothesis_metadata=hypothesis_metadata,
                created_by=created_by,
            )
            return exp_id, nb
        except Exception as exc:
            nb.close()
            if "locked" not in str(exc).lower() or attempt >= max_attempts:
                raise
            last_error = exc
            sleep_s = min(0.5 * (2 ** (attempt - 1)), 8.0)
            logger.warning(
                "Backfill experiment start hit db lock on attempt %d/%d; retrying in %.1fs",
                attempt,
                max_attempts,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise last_error or RuntimeError("database is locked")


def run_template_batch_detailed(
    template_name: str,
    n_programs: int,
    device: str,
    db_path: str,
    weight_mode: str = "uniform",
    phase: str = "isolation",
    *,
    model_dim_override: int | None = None,
    n_layers_override: int | None = None,
    stage1_steps_override: int | None = None,
    composition_depth_override: int | None = None,
) -> dict[str, Any]:
    """Run the full screening pipeline biased toward a single template."""
    from research.scientist.runner import ExperimentRunner, RunConfig
    from research.synthesis.templates import TEMPLATES

    # Fully specify every registered template so pick_template cannot fall back
    # to default priors for omitted names.
    tpl_weights = {t: 0.0 for t in TEMPLATES}
    tpl_weights[template_name] = 1.0

    cat_weights = _make_category_weights(weight_mode)
    op_weights = None
    if weight_mode == "scaffold_guided":
        op_weights, scaffold_cat_weights = _scaffold_guided_priors(db_path)
        cat_weights = scaffold_cat_weights or None
    phase_cfg = _phase_settings(phase, template_name)
    if n_layers_override is not None:
        phase_cfg["n_layers"] = int(n_layers_override)
    if stage1_steps_override is not None:
        phase_cfg["stage1_steps"] = int(stage1_steps_override)
    if composition_depth_override is not None:
        phase_cfg["composition_depth"] = int(composition_depth_override)

    config = RunConfig(
        n_programs=n_programs,
        device=device,
        mode="single",
        model_dim=(
            int(model_dim_override)
            if model_dim_override is not None
            else 256  # RunConfig default; can't use RunConfig.model_dim with slots=True
        ),
        composition_depth=int(phase_cfg["composition_depth"]),
        n_layers=int(phase_cfg["n_layers"]),
        stage1_steps=int(phase_cfg["stage1_steps"]),
        template_weights=tpl_weights,
        category_weights=cat_weights,
        op_weights=op_weights,
        use_learned_candidate_weights=False,
        use_screening_signal_weights=False,
        routing_mandatory=template_name not in _NON_ROUTING_TEMPLATES,
        persist_screening_failures=True,
        disable_runtime_dedup=True,
        enable_stage09_cheap_train_gate=False,
        gbm_prescreener_enabled=False,  # backfill needs ALL graphs for data collection
    )

    runner = ExperimentRunner(db_path)
    # Reset category weight overrides so backfill gets unbiased op distribution.
    # Without this, DB-persisted chat/Aria overrides (e.g. mixing: 0.3) skew
    # which ops appear inside template slots across ALL backfill runs.
    runner._grammar_weight_overrides = {}
    runner._op_weights_overrides = {}
    if cat_weights:
        logger.info(f"Category weights ({weight_mode}): {cat_weights}")
    if op_weights:
        logger.info(
            "Scaffold-guided op weights loaded: %d ops",
            len(op_weights),
        )
    logger.info("Cleared DB grammar/op weight overrides for backfill")
    logger.info(
        "Backfill forcing neutral weight mode: learned candidate weights off, screening signal weights off"
    )
    runner._ensure_math_spaces()

    hypothesis = f"Backfill ({phase}): gather data on template '{template_name}'"
    config_payload = config.to_dict()
    config_payload.update(
        {
            "backfill_template": template_name,
            "backfill_phase": phase,
            "backfill_weight_mode": weight_mode,
            "backfill_n_programs": int(n_programs),
        }
    )
    exp_id, nb = _start_backfill_experiment_with_retry(
        runner,
        experiment_type="backfill",
        config=config_payload,
        hypothesis=hypothesis,
        hypothesis_metadata={"source": "backfill_tool", "phase": phase},
        created_by="backfill_templates",
    )
    nb.close()

    logger.info(f"Started experiment {exp_id}")

    # Run the full screening pipeline in the current thread (blocking)
    nb = runner._make_notebook()
    try:
        results = runner._execute_experiment(
            exp_id, config, nb, use_learned_grammar=False
        )

        # Complete experiment without LLM summary/analysis
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=f"Backfill {phase} run for {template_name}: "
            f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1",
        )

        # Update op stats
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        else:
            nb.update_op_success_rates(exp_id)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)

        total = results.get("total", 0)
        persisted_rows = int(
            ((results.get("funnel_counts") or {}).get("persisted_rows", 0) or 0)
        )
        s1 = results.get("stage1_passed", 0)
        batch_summary = summarize_experiment_batch(db_path, exp_id)
        logger.info(f"Experiment {exp_id} done: {s1}/{total} S1 passed")
        if persisted_rows <= 0 and total > 0:
            logger.info(
                "Experiment %s produced %d candidates but 0 persisted rows "
                "(runtime-dedup/early screening consumed the batch)",
                exp_id,
                total,
            )
        return {
            "experiment_id": exp_id,
            "persisted_rows": persisted_rows,
            "results": results,
            "batch_summary": batch_summary,
            "status": "completed",
        }

    except KeyboardInterrupt:
        logger.info(f"Experiment {exp_id} interrupted — saving partial results")
        nb.fail_experiment(exp_id, error="KeyboardInterrupt")
        raise
    except Exception as e:
        logger.error(f"Experiment {exp_id} failed: {e}")
        nb.fail_experiment(exp_id, error=str(e))
        return {
            "experiment_id": exp_id,
            "persisted_rows": 0,
            "results": {},
            "batch_summary": summarize_experiment_batch(db_path, exp_id),
            "status": "failed",
            "error": str(e),
        }
    finally:
        nb.close()


def run_template_batch(
    template_name: str,
    n_programs: int,
    device: str,
    db_path: str,
    weight_mode: str = "uniform",
    phase: str = "isolation",
    *,
    model_dim_override: int | None = None,
    n_layers_override: int | None = None,
    stage1_steps_override: int | None = None,
    composition_depth_override: int | None = None,
) -> int:
    """Compatibility wrapper returning only persisted rows."""
    result = run_template_batch_detailed(
        template_name,
        n_programs,
        device,
        db_path,
        weight_mode,
        phase,
        model_dim_override=model_dim_override,
        n_layers_override=n_layers_override,
        stage1_steps_override=stage1_steps_override,
        composition_depth_override=composition_depth_override,
    )
    return int(result.get("persisted_rows", 0) or 0)


def main():
    parser = argparse.ArgumentParser(description="Backfill under-sampled templates")
    parser.add_argument(
        "--target", type=int, default=50, help="Minimum samples per template"
    )
    parser.add_argument(
        "--target-metric",
        default="eval",
        choices=list(_VALID_TARGET_METRICS),
        help="Which count must reach --target: eval, s0, or s1",
    )
    parser.add_argument(
        "--min-s1",
        type=int,
        default=0,
        help="Optional minimum S1 survivors per template in addition to --target",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Min programs per template",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without running"
    )
    parser.add_argument(
        "--templates",
        nargs="*",
        default=None,
        help="Only backfill these specific templates",
    )
    parser.add_argument(
        "--weights",
        default="scaffold_guided",
        choices=list(_VALID_WEIGHT_MODES),
        help="Weight mode: uniform, random, default, or scaffold_guided",
    )
    parser.add_argument(
        "--phase",
        default="isolation",
        choices=list(_VALID_PHASES),
        help="Backfill phase: isolation = single-block evidence, stack = survivability under deeper composition",
    )
    parser.add_argument(
        "--model-dim-override",
        type=int,
        default=None,
        help="Override default model_dim for the backfill run.",
    )
    parser.add_argument(
        "--n-layers-override",
        type=int,
        default=None,
        help="Override phase default for n_layers.",
    )
    parser.add_argument(
        "--stage1-steps-override",
        type=int,
        default=None,
        help="Override phase default for stage1_steps.",
    )
    parser.add_argument(
        "--composition-depth-override",
        type=int,
        default=None,
        help="Override phase default for composition_depth.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip ML model refresh after backfill (stats + Bayesian + graph predictor)",
    )
    parser.add_argument(
        "--policy",
        default="auto",
        choices=list(_VALID_POLICY_MODES),
        help="Template policy layer: auto applies rehab/coverage/harvest/frozen rules; off preserves legacy behavior.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_all",
        help="Show sample counts for all templates and exit",
    )
    args = parser.parse_args()

    from research.synthesis.templates import TEMPLATES

    stats = get_template_stats(Path(args.db))
    counts = Counter(
        {
            name: int(tpl_stats.get(args.target_metric, 0))
            for name, tpl_stats in stats.items()
        }
    )

    if args.list_all:
        print(
            f"{'Template':<35s} {'Eval':>5s} {'S0':>5s} {'S1':>5s} {'Target':>7s} {'Mode':>10s}"
        )
        print("-" * 78)
        for name in sorted(TEMPLATES.keys(), key=lambda n: counts.get(n, 0)):
            tpl_stats = stats.get(name, {})
            current = counts.get(name, 0)
            s1 = int(tpl_stats.get("s1", 0))
            missing_target = current < args.target
            missing_s1 = args.min_s1 > 0 and s1 < args.min_s1
            policy = get_template_backfill_policy(name)
            flag = " <--" if (missing_target or missing_s1) else ""
            print(
                f"  {name:<35s} {int(tpl_stats.get('eval', 0)):5d} "
                f"{int(tpl_stats.get('s0', 0)):5d} {s1:5d} "
                f"{current:7d} {policy.mode:>10s}{flag}"
            )
        total = sum(counts.get(n, 0) for n in TEMPLATES)
        print(
            f"\n{len(TEMPLATES)} templates, {total} total {args.target_metric} samples, "
            f"{sum(1 for n in TEMPLATES if counts.get(n, 0) < args.target or (args.min_s1 > 0 and int(stats.get(n, {}).get('s1', 0)) < args.min_s1))} "
            f"below target"
        )
        return

    # Find under-sampled templates
    candidates = args.templates if args.templates else list(TEMPLATES.keys())
    needs_data = {}
    for name in candidates:
        if name not in TEMPLATES:
            print(f"WARNING: '{name}' is not a registered template, skipping")
            continue
        tpl_stats = stats.get(name, {})
        current = counts.get(name, 0)
        s1 = int(tpl_stats.get("s1", 0))
        metric_deficit = max(args.target - current, 0)
        s1_deficit = max(args.min_s1 - s1, 0)
        if metric_deficit > 0 or s1_deficit > 0:
            policy = get_template_backfill_policy(name)
            needs_data[name] = {
                "metric_deficit": metric_deficit,
                "s1_deficit": s1_deficit,
                "current_metric": current,
                "current_stats": {
                    "eval": int(tpl_stats.get("eval", 0)),
                    "s0": int(tpl_stats.get("s0", 0)),
                    "s1": s1,
                },
                "policy": policy,
            }

    if not needs_data:
        print(
            f"All templates have >= {args.target} {args.target_metric} samples"
            + (f" and >= {args.min_s1} S1 survivors." if args.min_s1 > 0 else ".")
        )
        return

    total_programs = sum(
        plan_batch_size(
            template_name=name,
            requested_batch_size=args.batch_size,
            metric_deficit=data["metric_deficit"],
            s1_deficit=data["s1_deficit"],
            policy_mode=args.policy,
        )
        for name, data in needs_data.items()
    )
    print(
        f"Templates below target ({args.target_metric}>={args.target}"
        f"{', s1>=' + str(args.min_s1) if args.min_s1 > 0 else ''}) "
        f"({len(needs_data)} templates, ~{total_programs} programs):\n"
    )
    for name, data in sorted(
        needs_data.items(),
        key=lambda x: (x[1]["metric_deficit"], x[1]["s1_deficit"]),
        reverse=True,
    ):
        batch = plan_batch_size(
            template_name=name,
            requested_batch_size=args.batch_size,
            metric_deficit=data["metric_deficit"],
            s1_deficit=data["s1_deficit"],
            policy_mode=args.policy,
        )
        print(
            f"  {name:<35s}  {_fmt_stats(data['current_stats'])}  "
            f"need_{args.target_metric}={data['metric_deficit']:3d}  "
            f"need_s1={data['s1_deficit']:3d}  batch={batch:3d}  mode={data['policy'].mode}"
        )
    print()

    if args.dry_run:
        return

    completed = 0
    max_retries_per_template = 5
    try:
        for name, data in sorted(
            needs_data.items(),
            key=lambda x: (x[1]["metric_deficit"], x[1]["s1_deficit"]),
            reverse=True,
        ):
            policy = data["policy"]
            if args.policy == "auto" and policy.mode == "frozen":
                print(
                    f"\n=== {name} skipped (mode=frozen) "
                    f"stats: {_fmt_stats(data['current_stats'])} ==="
                )
                if policy.notes:
                    print(f"  policy note: {policy.notes}")
                continue

            current = data["current_metric"]
            current_s1 = int(data["current_stats"].get("s1", 0))

            print(
                f"\n=== {name} ({args.target_metric} {current} → {args.target}, "
                f"phase={args.phase}, policy={policy.mode}, stats: {_fmt_stats(data['current_stats'])}) ==="
            )
            t0 = time.time()
            recorded = 0
            updated_stats = data["current_stats"]
            new_count = current
            attempts = 0
            new_s1 = current_s1
            max_retries = (
                policy.max_retries
                if args.policy == "auto"
                else max_retries_per_template
            )
            while attempts < max_retries and (
                new_count < args.target or new_s1 < args.min_s1
            ):
                attempts += 1
                n_programs = plan_batch_size(
                    template_name=name,
                    requested_batch_size=args.batch_size,
                    metric_deficit=max(args.target - new_count, 0),
                    s1_deficit=max(args.min_s1 - new_s1, 0),
                    policy_mode=args.policy,
                )
                if n_programs <= 0:
                    print("  Policy skipped batch generation for this template.")
                    break
                effective_weight_mode = resolve_weight_mode(
                    name, args.weights, args.policy
                )
                batch_result = run_template_batch_detailed(
                    name,
                    n_programs,
                    args.device,
                    args.db,
                    effective_weight_mode,
                    args.phase,
                    model_dim_override=args.model_dim_override,
                    n_layers_override=args.n_layers_override,
                    stage1_steps_override=args.stage1_steps_override,
                    composition_depth_override=args.composition_depth_override,
                )
                recorded = int(batch_result.get("persisted_rows", 0) or 0)
                updated_stats = get_template_stats(Path(args.db)).get(name, {})
                new_count = int(updated_stats.get(args.target_metric, 0))
                new_s1 = int(updated_stats.get("s1", 0))
                batch_summary = batch_result.get("batch_summary") or {}
                stop_now, stop_reason = should_stop_backfill_attempt(
                    template_name=name,
                    batch_summary=batch_summary,
                    policy_mode=args.policy,
                )
                if stop_now:
                    print(
                        f"  Stopping early after attempt {attempts}: {stop_reason} "
                        f"(rows={batch_summary.get('rows', 0)} s0={batch_summary.get('s0', 0)} s1={batch_summary.get('s1', 0)})"
                    )
                    break
                if new_count > current:
                    current = new_count
                    current_s1 = new_s1
                if recorded == 0:
                    print(
                        f"  Attempt {attempts}/{max_retries}: "
                        f"0 persisted rows after runtime dedup. Retrying..."
                    )
            elapsed = time.time() - t0
            print(
                f"  Recorded {recorded} persisted programs in {elapsed:.0f}s, "
                f"{name} now has {new_count} {args.target_metric} samples "
                f"({_fmt_stats(updated_stats)})"
            )
            completed += 1
    except KeyboardInterrupt:
        print(
            f"\n\nInterrupted after {completed}/{len(needs_data)} templates. Partial results saved."
        )
        return

    # ── Refresh ML models after backfill ──
    if not args.no_refresh:
        print("\nRefreshing analytics stats + ML models...")
        try:
            from research.tools.backfill_stats import backfill

            backfill(args.db)
            print("  Stats tables rebuilt (op_stats, template_stats, motif_stats)")
        except Exception as e:
            print(f"  Stats backfill failed: {e}")

        try:
            from research.tools.train_predictors import (
                train_bayesian,
                train_ensemble_full,
                train_graph_predictor,
            )

            train_bayesian(save=True)
            print("  Bayesian tracker refreshed")
            train_graph_predictor(save=True)
            print("  Graph predictor refreshed")
            train_ensemble_full(save=True)
            print("  Ensemble predictor refreshed")
        except Exception as e:
            print(f"  ML model refresh failed: {e}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
