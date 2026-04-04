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
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "lab_notebook.db"
_VALID_TARGET_METRICS = ("eval", "s0", "s1")
_VALID_WEIGHT_MODES = ("uniform", "random", "default", "scaffold_guided")
_VALID_PHASES = ("isolation", "stack")


def get_template_stats(db_path: Path) -> dict[str, dict[str, int]]:
    """Return per-template eval/S0/S1 counts from live program_results rows."""
    db = sqlite3.connect(str(db_path))
    stats: dict[str, dict[str, int]] = {}

    try:
        rows = db.execute(
            "SELECT graph_json, stage0_passed, stage1_passed "
            "FROM program_results "
            "WHERE graph_json IS NOT NULL"
        ).fetchall()
        for gj, s0, s1 in rows:
            try:
                g = json.loads(gj)
                for t in set(g.get("metadata", {}).get("templates_used", [])):
                    bucket = stats.setdefault(t, {"eval": 0, "s0": 0, "s1": 0})
                    bucket["eval"] += 1
                    bucket["s0"] += 1 if s0 else 0
                    bucket["s1"] += 1 if s1 else 0
            except Exception:
                pass
    except sqlite3.OperationalError as exc:
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
        except sqlite3.OperationalError:
            pass
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
    min_support: int = 2,
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


def _phase_settings(phase: str) -> dict[str, Any]:
    """Backfill settings for isolation vs stack validation."""
    if phase == "stack":
        return {
            "composition_depth": 2,
            "n_layers": 3,
            "stage1_steps": 750,
        }
    return {
        "composition_depth": 1,
        "n_layers": 2,
        "stage1_steps": 500,
    }


def run_template_batch(
    template_name: str,
    n_programs: int,
    device: str,
    db_path: str,
    weight_mode: str = "uniform",
    phase: str = "isolation",
) -> int:
    """Run the full screening pipeline biased toward a single template."""
    from research.scientist.runner import ExperimentRunner, RunConfig
    from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS

    # Fully specify template weights so pick_template does not fall back to
    # default priors for every other template.
    tpl_weights = {t: 0.0 for t in DEFAULT_TEMPLATE_WEIGHTS}
    tpl_weights[template_name] = 1.0

    cat_weights = _make_category_weights(weight_mode)
    op_weights = None
    if weight_mode == "scaffold_guided":
        op_weights, scaffold_cat_weights = _scaffold_guided_priors(db_path)
        cat_weights = scaffold_cat_weights or None
    phase_cfg = _phase_settings(phase)

    # Reference/non-routing templates need routing_mandatory=False
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
        "latent_attn_conv_hybrid",
        "diff_attn_conv_hybrid",
        "attn_safe_division",
        "latent_attn_ssm_hybrid",
        "local_attn_ssm_hybrid",
        "attn_spiking_hybrid",
        "linear_attn_sparse_ffn",
        "graph_attn_sparse_ffn",
    }
    config = RunConfig(
        n_programs=n_programs,
        device=device,
        mode="single",
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

    nb = runner._make_notebook()
    runner._populate_refuted_cache(nb)

    hypothesis = f"Backfill ({phase}): gather data on template '{template_name}'"
    exp_id = runner._start_preregistered_experiment(
        nb=nb,
        experiment_type="backfill",
        config=config.to_dict(),
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
        s1 = results.get("stage1_passed", 0)
        logger.info(f"Experiment {exp_id} done: {s1}/{total} S1 passed")
        return total

    except KeyboardInterrupt:
        logger.info(f"Experiment {exp_id} interrupted — saving partial results")
        nb.fail_experiment(exp_id, error="KeyboardInterrupt")
        raise
    except Exception as e:
        logger.error(f"Experiment {exp_id} failed: {e}")
        nb.fail_experiment(exp_id, error=str(e))
        return 0
    finally:
        nb.close()


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
        "--no-refresh",
        action="store_true",
        help="Skip ML model refresh after backfill (stats + Bayesian + graph predictor)",
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
        print(f"{'Template':<35s} {'Eval':>5s} {'S0':>5s} {'S1':>5s} {'Target':>7s}")
        print("-" * 66)
        for name in sorted(TEMPLATES.keys(), key=lambda n: counts.get(n, 0)):
            tpl_stats = stats.get(name, {})
            current = counts.get(name, 0)
            s1 = int(tpl_stats.get("s1", 0))
            missing_target = current < args.target
            missing_s1 = args.min_s1 > 0 and s1 < args.min_s1
            flag = " <--" if (missing_target or missing_s1) else ""
            print(
                f"  {name:<35s} {int(tpl_stats.get('eval', 0)):5d} "
                f"{int(tpl_stats.get('s0', 0)):5d} {s1:5d} "
                f"{current:7d}{flag}"
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
            needs_data[name] = {
                "metric_deficit": metric_deficit,
                "s1_deficit": s1_deficit,
                "current_metric": current,
                "current_stats": {
                    "eval": int(tpl_stats.get("eval", 0)),
                    "s0": int(tpl_stats.get("s0", 0)),
                    "s1": s1,
                },
            }

    if not needs_data:
        print(
            f"All templates have >= {args.target} {args.target_metric} samples"
            + (f" and >= {args.min_s1} S1 survivors." if args.min_s1 > 0 else ".")
        )
        return

    total_programs = sum(
        max(args.batch_size, data["metric_deficit"], data["s1_deficit"])
        for data in needs_data.values()
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
        batch = max(args.batch_size, data["metric_deficit"], data["s1_deficit"])
        print(
            f"  {name:<35s}  {_fmt_stats(data['current_stats'])}  "
            f"need_{args.target_metric}={data['metric_deficit']:3d}  "
            f"need_s1={data['s1_deficit']:3d}  batch={batch}"
        )
    print()

    if args.dry_run:
        return

    completed = 0
    try:
        for name, data in sorted(
            needs_data.items(),
            key=lambda x: (x[1]["metric_deficit"], x[1]["s1_deficit"]),
            reverse=True,
        ):
            n_programs = max(
                args.batch_size, data["metric_deficit"], data["s1_deficit"]
            )
            current = data["current_metric"]

            print(
                f"\n=== {name} ({args.target_metric} {current} → {args.target}, "
                f"phase={args.phase}, stats: {_fmt_stats(data['current_stats'])}) ==="
            )
            t0 = time.time()
            recorded = run_template_batch(
                name, n_programs, args.device, args.db, args.weights, args.phase
            )
            elapsed = time.time() - t0

            updated_stats = get_template_stats(Path(args.db)).get(name, {})
            new_count = int(updated_stats.get(args.target_metric, 0))
            print(
                f"  Recorded {recorded} programs in {elapsed:.0f}s, "
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
                train_graph_predictor,
            )

            train_bayesian(save=True)
            print("  Bayesian tracker refreshed")
            train_graph_predictor()
            print("  Graph predictor refreshed")
        except Exception as e:
            print(f"  ML model refresh failed: {e}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
