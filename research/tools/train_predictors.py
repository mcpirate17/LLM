"""Train and evaluate all ML predictors from historical data.

Backfill script for initializing or refreshing the intelligence layer:
- Temporal Bayesian tracker (op/template/motif posteriors)
- Op embeddings (16-dim contrastive)
- Interaction model (factored bilinear pair prediction)
- Graph topology predictor (structural features)
- Full ensemble (combines all above + GBM)

Usage:
    python -m research.tools.train_predictors                    # train all
    python -m research.tools.train_predictors --component bayesian
    python -m research.tools.train_predictors --evaluate          # train + eval
    python -m research.tools.train_predictors --save-state        # persist to disk
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_NOTEBOOK_DB = Path("research/lab_notebook.db")
_PROFILING_DB = Path("research/profiling/component_profiles.db")
_STATE_DIR = Path("research/runtime/learning")


def train_bayesian(save: bool = False) -> dict:
    """Train temporal Bayesian tracker from DB."""
    from research.scientist.intelligence.temporal_bayesian import (
        TemporalBayesianTracker,
    )

    t0 = time.time()
    tracker = TemporalBayesianTracker.from_db(_NOTEBOOK_DB)
    elapsed = time.time() - t0

    diag = tracker.diagnostics()
    logger.info("Bayesian tracker: %s (%.1fs)", diag, elapsed)

    if save:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tracker.save_state(_STATE_DIR / "bayesian_state.json")

    return {"component": "bayesian", "elapsed_s": elapsed, **diag}


def train_embeddings(save: bool = False) -> dict:
    """Train op embeddings from profiling + experiment data."""
    from research.scientist.intelligence.op_embeddings import OpEmbeddings

    t0 = time.time()
    emb = OpEmbeddings.train(_NOTEBOOK_DB, _PROFILING_DB, n_epochs=50)
    elapsed = time.time() - t0

    logger.info(
        "Op embeddings: %d ops, trained=%s (%.1fs)", emb.n_ops, emb._trained, elapsed
    )

    if save:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        emb.save(_STATE_DIR / "op_embeddings.npz")

    return {
        "component": "embeddings",
        "n_ops": emb.n_ops,
        "trained": emb._trained,
        "elapsed_s": elapsed,
    }


def train_interaction(save: bool = False) -> dict:
    """Train factored interaction model."""
    from research.scientist.intelligence.interaction_model import InteractionModel

    t0 = time.time()
    model = InteractionModel.train(_NOTEBOOK_DB, _PROFILING_DB, n_epochs=80)
    elapsed = time.time() - t0

    logger.info(
        "Interaction model: %d ops, metrics=%s (%.1fs)",
        model.n_ops,
        model._train_metrics,
        elapsed,
    )

    if save:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(_STATE_DIR / "interaction_model.npz")

    return {
        "component": "interaction",
        "n_ops": model.n_ops,
        "elapsed_s": elapsed,
        **model._train_metrics,
    }


def train_graph_predictor(save: bool = False) -> dict:
    """Train topology-aware graph predictor."""
    from research.scientist.intelligence.gnn_predictor import GraphPredictor

    t0 = time.time()
    model = GraphPredictor.train(_NOTEBOOK_DB, _PROFILING_DB)
    elapsed = time.time() - t0

    logger.info("Graph predictor: metrics=%s (%.1fs)", model._train_metrics, elapsed)

    if save and model.is_fitted():
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(_STATE_DIR / "graph_predictor.npz")

    return {
        "component": "graph_predictor",
        "elapsed_s": elapsed,
        **model._train_metrics,
    }


def train_ensemble_full(save: bool = False) -> dict:
    """Train full ensemble (all components)."""
    from research.scientist.intelligence.predictor import train_ensemble

    t0 = time.time()
    ensemble = train_ensemble(
        db_path=str(_NOTEBOOK_DB),
        profiling_db=str(_PROFILING_DB),
    )
    elapsed = time.time() - t0

    diag = ensemble.diagnostics()
    logger.info("Ensemble: %s (%.1fs)", diag, elapsed)

    if save and ensemble.is_fitted():
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        ensemble.save(_STATE_DIR)

    return {"component": "ensemble", "elapsed_s": elapsed, **diag}


def evaluate_all() -> dict:
    """Evaluate all predictors and report metrics."""
    from research.scientist.intelligence.predictor import evaluate_gbm

    results = {}

    # GBM evaluation
    try:
        gbm_eval = evaluate_gbm(str(_NOTEBOOK_DB))
        results["gbm"] = gbm_eval
        logger.info(
            "GBM eval: AUC=%.3f, skip_rate=%.2f",
            gbm_eval.get("gate_auc", 0),
            gbm_eval.get("skip_rate", 0),
        )
    except Exception as e:
        results["gbm"] = {"error": str(e)}

    # Graph predictor evaluation
    try:
        from research.scientist.intelligence.gnn_predictor import GraphPredictor

        gp = GraphPredictor.train(_NOTEBOOK_DB, _PROFILING_DB)
        results["graph_predictor"] = gp._train_metrics
        logger.info("GraphPredictor eval: %s", gp._train_metrics)
    except Exception as e:
        results["graph_predictor"] = {"error": str(e)}

    # Interaction analysis summary
    try:
        from research.scientist.intelligence.interaction_analysis import (
            build_interaction_matrix,
        )

        matrix = build_interaction_matrix(_NOTEBOOK_DB, _PROFILING_DB)
        results["interaction_analysis"] = matrix.summary()
        logger.info("Interaction analysis: %s", matrix.summary())
    except Exception as e:
        results["interaction_analysis"] = {"error": str(e)}

    return results


def generate_heatmaps() -> None:
    """Generate all interaction heatmaps to artifacts/."""
    from research.scientist.intelligence.interaction_analysis import (
        build_interaction_matrix,
        render_heatmap,
        export_json,
    )

    output_dir = Path("research/artifacts")
    matrix = build_interaction_matrix(_NOTEBOOK_DB, _PROFILING_DB, min_observations=3)

    for metric in ("s1_rate", "loss", "obs"):
        render_heatmap(
            matrix,
            metric=metric,
            output_path=output_dir / f"interaction_{metric}.png",
            title=f"Op Interaction: {metric}",
        )

    # Category rollup
    rollup = matrix.category_rollup()
    render_heatmap(
        rollup,
        metric="s1_rate",
        output_path=output_dir / "interaction_s1_rate_category.png",
        title="Category×Category S1 Rate",
    )

    # Math-space slice
    math_matrix = matrix.filter_category("math_space")
    if math_matrix.ops:
        render_heatmap(
            math_matrix,
            metric="s1_rate",
            output_path=output_dir / "interaction_s1_rate_math_space.png",
            title="Math-Space Op Interaction: S1 Rate",
        )

    export_json(matrix, output_dir / "interaction_data.json")
    logger.info("Heatmaps generated in %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ML predictors")
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=[
            "all",
            "bayesian",
            "embeddings",
            "interaction",
            "graph",
            "ensemble",
            "heatmaps",
        ],
        help="Which component to train",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation after training"
    )
    parser.add_argument(
        "--save-state", action="store_true", help="Persist trained models to disk"
    )
    parser.add_argument(
        "--heatmaps", action="store_true", help="Generate interaction heatmaps"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = {}
    t0 = time.time()

    if args.component in ("all", "bayesian"):
        results["bayesian"] = train_bayesian(save=args.save_state)
    if args.component in ("all", "embeddings"):
        results["embeddings"] = train_embeddings(save=args.save_state)
    if args.component in ("all", "interaction"):
        results["interaction"] = train_interaction(save=args.save_state)
    if args.component in ("all", "graph"):
        results["graph"] = train_graph_predictor(save=args.save_state)
    if args.component in ("all", "ensemble"):
        results["ensemble"] = train_ensemble_full(save=args.save_state)
    if args.component == "heatmaps" or args.heatmaps:
        generate_heatmaps()

    if args.evaluate:
        results["evaluation"] = evaluate_all()

    total = time.time() - t0
    logger.info("Total training time: %.1fs", total)
    for name, res in results.items():
        logger.info("  %s: %s", name, res)


if __name__ == "__main__":
    main()
