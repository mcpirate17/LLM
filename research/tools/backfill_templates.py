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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "lab_notebook.db"


def get_template_counts(db_path: Path) -> Counter:
    """Count how many program_results reference each template.

    Uses SQL LIKE to avoid fetching/parsing full graph_json blobs.
    Templates are stored in metadata.templates_used inside graph_json,
    so we extract via json_extract where available, falling back to LIKE.
    """
    db = sqlite3.connect(str(db_path))
    counts: Counter = Counter()

    # Try json_extract first (SQLite 3.38+)
    try:
        rows = db.execute(
            "SELECT json_extract(graph_json, '$.metadata.templates_used') "
            "FROM program_results WHERE graph_json IS NOT NULL"
        ).fetchall()
        for (tpl_json,) in rows:
            if not tpl_json:
                continue
            try:
                for t in set(json.loads(tpl_json)):
                    counts[t] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        # Fallback: fetch and parse (older SQLite)
        rows = db.execute(
            "SELECT graph_json FROM program_results WHERE graph_json IS NOT NULL"
        ).fetchall()
        for (gj,) in rows:
            try:
                g = json.loads(gj)
                for t in set(g.get("metadata", {}).get("templates_used", [])):
                    counts[t] += 1
            except Exception:
                pass

    db.close()
    return counts


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


def run_template_batch(
    template_name: str,
    n_programs: int,
    device: str,
    db_path: str,
    weight_mode: str = "uniform",
) -> int:
    """Run the full screening pipeline biased toward a single template."""
    from research.scientist.runner import ExperimentRunner, RunConfig
    from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS

    # Heavily bias toward target template
    tpl_weights = {t: 0.01 for t in DEFAULT_TEMPLATE_WEIGHTS}
    tpl_weights[template_name] = 100.0

    cat_weights = _make_category_weights(weight_mode)

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
    }
    config = RunConfig(
        n_programs=n_programs,
        device=device,
        mode="single",
        template_weights=tpl_weights,
        category_weights=cat_weights,
        routing_mandatory=template_name not in _NON_ROUTING_TEMPLATES,
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
    logger.info("Cleared DB grammar/op weight overrides for backfill")
    runner._ensure_math_spaces()

    nb = runner._make_notebook()
    runner._populate_refuted_cache(nb)

    hypothesis = f"Backfill: gather data on template '{template_name}'"
    exp_id = runner._start_preregistered_experiment(
        nb=nb,
        experiment_type="backfill",
        config=config.to_dict(),
        hypothesis=hypothesis,
        hypothesis_metadata={"source": "backfill_tool"},
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
            aria_summary=f"Backfill run for {template_name}: "
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
        default="uniform",
        choices=["uniform", "random", "default"],
        help="Category weight mode: uniform (all 1.0), random, or default (GrammarConfig defaults)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_all",
        help="Show sample counts for all templates and exit",
    )
    args = parser.parse_args()

    from research.synthesis.templates import TEMPLATES

    counts = get_template_counts(Path(args.db))

    if args.list_all:
        print(f"{'Template':<35s} {'Samples':>7s}  {'S1%':>5s}")
        print("-" * 50)
        for name in sorted(TEMPLATES.keys(), key=lambda n: counts.get(n, 0)):
            c = counts.get(name, 0)
            flag = " <--" if c < args.target else ""
            print(f"  {name:<35s} {c:5d}{flag}")
        total = sum(counts.get(n, 0) for n in TEMPLATES)
        print(
            f"\n{len(TEMPLATES)} templates, {total} total samples, "
            f"{sum(1 for n in TEMPLATES if counts.get(n, 0) < args.target)} below {args.target}"
        )
        return

    # Find under-sampled templates
    candidates = args.templates if args.templates else list(TEMPLATES.keys())
    needs_data = {}
    for name in candidates:
        if name not in TEMPLATES:
            print(f"WARNING: '{name}' is not a registered template, skipping")
            continue
        current = counts.get(name, 0)
        if current < args.target:
            needs_data[name] = args.target - current

    if not needs_data:
        print(f"All templates have >= {args.target} samples.")
        return

    total_programs = sum(max(args.batch_size, d) for d in needs_data.values())
    print(
        f"Templates below {args.target} samples "
        f"({len(needs_data)} templates, ~{total_programs} programs):\n"
    )
    for name, deficit in sorted(needs_data.items(), key=lambda x: x[1], reverse=True):
        current = counts.get(name, 0)
        batch = max(args.batch_size, deficit)
        print(f"  {name:<35s}  have={current:3d}  need={deficit:3d}  batch={batch}")
    print()

    if args.dry_run:
        return

    completed = 0
    try:
        for name, deficit in sorted(needs_data.items(), key=lambda x: -x[1]):
            n_programs = max(args.batch_size, deficit)
            current = counts.get(name, 0)

            print(f"\n=== {name} ({current} → {args.target}) ===")
            t0 = time.time()
            recorded = run_template_batch(
                name, n_programs, args.device, args.db, args.weights
            )
            elapsed = time.time() - t0

            new_count = get_template_counts(Path(args.db)).get(name, 0)
            print(
                f"  Recorded {recorded} programs in {elapsed:.0f}s, "
                f"{name} now has {new_count} samples"
            )
            completed += 1
    except KeyboardInterrupt:
        print(
            f"\n\nInterrupted after {completed}/{len(needs_data)} templates. Partial results saved."
        )
        return

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
