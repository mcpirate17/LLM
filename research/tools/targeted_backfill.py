"""Targeted backfill: run specific template combinations that historically produce low loss.

Unlike backfill_templates.py which biases toward a single template, this script
forces SPECIFIC template combinations via composition_depth and heavy co-weighting.
Designed for gathering high-quality data on known winning patterns.

Usage:
    python -m research.tools.targeted_backfill --combo sparse_ffn+token_merge_block --n 80
    python -m research.tools.targeted_backfill --combo token_merge_block+token_merge_block --n 50
    python -m research.tools.targeted_backfill --all --n 50
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "lab_notebook.db"

# Known winning template combos (from data analysis, sorted by best loss)
WINNING_COMBOS = {
    "sparse_ffn+token_merge_block": {
        "templates": ["sparse_ffn", "token_merge_block"],
        "best_loss": 0.006,
        "historical_s1_rate": 0.75,
        "notes": "3x S1 passes at loss 0.006-0.043 — best token_merge results",
    },
    "token_merge_block": {
        "templates": ["token_merge_block"],
        "best_loss": 0.064,
        "historical_s1_rate": 0.50,
        "notes": "Standalone token_merge with conv1d+swiglu+sparse",
    },
    "token_merge_block+token_merge_block": {
        "templates": ["token_merge_block", "token_merge_block"],
        "best_loss": None,
        "historical_s1_rate": None,
        "notes": "Double merge — untested, could discover deeper merge patterns",
    },
    "conditional_compute+token_merge_block": {
        "templates": ["conditional_compute", "token_merge_block"],
        "best_loss": 0.104,
        "historical_s1_rate": 0.33,
        "notes": "Routing + merge combo",
    },
    "three_lane_adaptive+token_merge_block": {
        "templates": ["three_lane_adaptive", "token_merge_block"],
        "best_loss": None,
        "historical_s1_rate": None,
        "notes": "3-lane routing + merge — untested but both templates are strong",
    },
    "fused_gelu_ffn+token_merge_block": {
        "templates": ["fused_gelu_ffn", "token_merge_block"],
        "best_loss": None,
        "historical_s1_rate": None,
        "notes": "Fused FFN + merge",
    },
}


def run_combo(
    combo_name: str,
    templates: list[str],
    n_programs: int,
    device: str,
    db_path: str,
) -> int:
    """Run screening pipeline biased toward a specific template combination."""
    from research.scientist.runner import ExperimentRunner, RunConfig
    from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS

    # Suppress all templates except the target combo.
    # Also boost templates with efficiency ops so the screening pipeline's
    # Gate 5 (requires routing/sparse/MoE op) doesn't drop all candidates.
    # The screening pipeline enforces this gate independently of RunConfig.
    tpl_weights = {t: 0.01 for t in DEFAULT_TEMPLATE_WEIGHTS}
    for t in templates:
        tpl_weights[t] = 100.0
    # Ensure at least some companion templates with efficiency ops
    # get generated in dedup rounds (these score on the 135pt efficiency budget)
    _EFFICIENCY_COMPANIONS = [
        "sparse_ffn",
        "routed_bottleneck",
        "sparse_moe_block",
        "feature_sparse_block",
        "three_lane_adaptive",
    ]
    for t in _EFFICIENCY_COMPANIONS:
        if t not in templates:
            tpl_weights.setdefault(t, max(tpl_weights.get(t, 0), 5.0))

    config = RunConfig(
        n_programs=n_programs,
        device=device,
        mode="single",
        template_weights=tpl_weights,
        category_weights={
            k: 1.0
            for k in [
                "elementwise_unary",
                "elementwise_binary",
                "reduction",
                "linear_algebra",
                "structural",
                "parameterized",
                "mixing",
                "sequence",
                "frequency",
                "math_space",
                "functional",
            ]
        },
        use_learned_candidate_weights=False,
        use_screening_signal_weights=False,
        routing_mandatory=False,
        gbm_prescreener_enabled=False,
    )

    runner = ExperimentRunner(db_path)
    runner._grammar_weight_overrides = {}
    runner._op_weights_overrides = {}
    logger.info(
        "Targeted backfill forcing neutral weight mode: learned candidate weights off, screening signal weights off"
    )
    runner._ensure_math_spaces()

    nb = runner._make_notebook()
    runner._populate_refuted_cache(nb)

    hypothesis = f"Targeted backfill: {combo_name} ({'+'.join(templates)})"
    exp_id = runner._start_preregistered_experiment(
        nb=nb,
        experiment_type="backfill",
        config=config.to_dict(),
        hypothesis=hypothesis,
        hypothesis_metadata={"source": "targeted_backfill", "combo": combo_name},
        created_by="targeted_backfill",
    )
    nb.close()

    logger.info(f"Started experiment {exp_id} for {combo_name}")

    nb = runner._make_notebook()
    try:
        results = runner._execute_experiment(
            exp_id, config, nb, use_learned_grammar=False
        )
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=f"Targeted backfill {combo_name}: "
            f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1",
        )
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        else:
            nb.update_op_success_rates(exp_id)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)

        total = results.get("total", 0)
        s1 = results.get("stage1_passed", 0)
        best = results.get("best_loss_ratio")
        best_str = f", best_loss={best:.4f}" if best else ""
        logger.info(f"{combo_name}: {s1}/{total} S1 passed{best_str}")
        return total
    except KeyboardInterrupt:
        nb.fail_experiment(exp_id, error="KeyboardInterrupt")
        raise
    except Exception as e:
        logger.error(f"{combo_name} failed: {e}")
        nb.fail_experiment(exp_id, error=str(e))
        return 0
    finally:
        nb.close()


def main():
    parser = argparse.ArgumentParser(
        description="Targeted backfill for winning template combos"
    )
    parser.add_argument(
        "--combo",
        type=str,
        default=None,
        help="Combo name (e.g. sparse_ffn+token_merge_block)",
    )
    parser.add_argument("--all", action="store_true", help="Run all winning combos")
    parser.add_argument("--n", type=int, default=50, help="Programs per combo")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--list", action="store_true", help="List available combos")
    args = parser.parse_args()

    if args.list:
        print(f"{'Combo':<45} {'Best Loss':>10} {'S1%':>6}  Notes")
        print("-" * 90)
        for name, info in WINNING_COMBOS.items():
            bl = f"{info['best_loss']:.4f}" if info["best_loss"] else "untested"
            sr = (
                f"{info['historical_s1_rate']:.0%}"
                if info["historical_s1_rate"]
                else "?"
            )
            print(f"  {name:<45} {bl:>10} {sr:>6}  {info['notes']}")
        return

    if args.all:
        combos = list(WINNING_COMBOS.keys())
    elif args.combo:
        if args.combo not in WINNING_COMBOS:
            # Allow ad-hoc combos: "tpl1+tpl2"
            templates = [t.strip() for t in args.combo.split("+")]
            WINNING_COMBOS[args.combo] = {"templates": templates}
        combos = [args.combo]
    else:
        parser.print_help()
        return

    t0 = time.time()
    total_programs = 0
    for combo_name in combos:
        info = WINNING_COMBOS[combo_name]
        templates = info["templates"]
        print(f"\n=== {combo_name} ===")
        recorded = run_combo(combo_name, templates, args.n, args.device, args.db)
        total_programs += recorded

    elapsed = time.time() - t0
    print(
        f"\nDone: {len(combos)} combos, {total_programs} total programs, {elapsed:.0f}s"
    )

    # Refresh stats
    try:
        from research.tools.backfill_stats import backfill

        backfill(args.db)
        print("Stats refreshed")
    except Exception as e:
        print(f"Stats refresh failed: {e}")


if __name__ == "__main__":
    main()
