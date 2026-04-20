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
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.scientist.intelligence.predictor_artifacts import (
    BAYESIAN_STATE_PATH,
    ENSEMBLE_META_PATH,
    GRAPH_PREDICTOR_PATH,
    INTERACTION_MODEL_PATH,
    METRICS_REPORT_PATH,
    OP_EMBEDDINGS_PATH,
    STATE_DIR,
    ensure_state_dir,
)
from research.tools._script_audit import (
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)

logger = logging.getLogger(__name__)

_NOTEBOOK_DB = Path("research/lab_notebook.db")
_PROFILING_DB = Path("research/profiling/component_profiles.db")


def _safe_report_section(name: str, fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc), "section": name}


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _derive_confusion_from_summary(
    *,
    n: int,
    accuracy: float,
    precision: float,
    recall: float,
) -> dict[str, float]:
    from research.scientist.intelligence.metrics_utils import (
        binary_classification_metrics,
    )

    for positives in range(n + 1):
        tp_raw = recall * positives
        tp = int(round(tp_raw))
        if not np.isclose(tp_raw, tp):
            continue
        if precision <= 0.0:
            pred_pos = tp
        else:
            pred_pos_raw = tp / precision
            pred_pos = int(round(pred_pos_raw))
            if not np.isclose(pred_pos_raw, pred_pos):
                continue
        fp = pred_pos - tp
        fn = positives - tp
        tn = n - positives - fp
        if min(tp, fp, tn, fn) < 0:
            continue
        if np.isclose((tp + tn) / max(n, 1), accuracy):
            return binary_classification_metrics(
                np.array([1] * positives + [0] * (n - positives), dtype=np.int32),
                np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn, dtype=np.float64),
                0.5,
            )
    raise ValueError("unable to derive confusion matrix from summary metrics")


def _graph_predictor_metrics_report(state_dir: Path) -> dict[str, Any]:
    graph_meta_path = state_dir / "graph_predictor.json"
    if not graph_meta_path.exists():
        return {"error": "graph_predictor_not_fitted"}
    with open(graph_meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    train_metrics = dict(meta.get("train_metrics", {}))
    derived = dict(train_metrics.get("val_gate_metrics", {}))
    if not derived:
        derived = _derive_confusion_from_summary(
            n=int(train_metrics.get("n_val", 0)),
            accuracy=float(train_metrics.get("val_accuracy", 0.0)),
            precision=float(train_metrics.get("val_precision", 0.0)),
            recall=float(train_metrics.get("val_recall", 0.0)),
        )
        derived["roc_auc"] = float(train_metrics.get("val_auc", 0.0))
        derived["threshold"] = float(train_metrics.get("gate_threshold", 0.5))
    return {
        "persisted_train_metrics": train_metrics,
        "derived_val_classification_metrics": derived,
        "note": (
            "Uses persisted holdout metrics from graph_predictor.json. "
            "This is the most faithful report for the saved GraphPredictor artifact."
        ),
    }


def _gbm_metrics_report(db_path: str, state_dir: Path) -> dict[str, Any]:
    from research.scientist.intelligence.metrics_utils import (
        binary_classification_metrics,
    )
    from research.scientist.intelligence.ml_corpus import (
        build_dense_feature_matrix,
        grouped_stratified_split,
    )
    from research.scientist.intelligence.predictor import (
        GBMPredictor,
        _query_graph_training_data,
    )

    feat_dicts, y_gate, y_rank, _sample_weights, graph_signatures = (
        _query_graph_training_data(db_path)
    )
    X, feature_names = build_dense_feature_matrix(feat_dicts)
    _train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y_gate, seed=42
    )
    gbm = GBMPredictor.load(state_dir)
    if not gbm.is_fitted():
        return {"error": "gbm_not_fitted"}

    gate_names = gbm.gate_feature_names or gbm.feature_names
    gate_col_idx = [
        feature_names.index(name) for name in gate_names if name in feature_names
    ]
    X_gate = X[:, gate_col_idx] if gate_col_idx else X

    if gbm.train_metrics:
        operating_points = dict(gbm.train_metrics.get("operating_points", {}))
        selected = dict(gbm.train_metrics.get("gate_metrics", {}))
        skip_metrics = binary_classification_metrics(
            y_gate[val_idx], gbm.gate_model.predict(X_gate[val_idx]), threshold=0.1
        )
        skipped = skip_metrics["tn"] + skip_metrics["fn"]
        false_skip_rate = float(skip_metrics["fn"] / skipped) if skipped else 0.0
        return {
            "split_stats": split_stats,
            "train_metrics": gbm.train_metrics,
            "val_metrics_selected_threshold": selected,
            "val_metrics_threshold_0_5": binary_classification_metrics(
                y_gate[val_idx], gbm.gate_model.predict(X_gate[val_idx]), threshold=0.5
            ),
            "val_metrics_threshold_0_1_skip_rule": skip_metrics,
            "operating_points": operating_points,
            "skip_rate": float(skipped / max(len(val_idx), 1)),
            "false_skip_rate": false_skip_rate,
            "rank_spearman_val": gbm.train_metrics.get("rank_spearman"),
        }

    gate_scores = gbm.gate_model.predict(X_gate[val_idx])
    rank_spearman = None
    if gbm.rank_model is not None:
        rank_mask = np.isfinite(y_rank[val_idx])
        if rank_mask.sum() >= 2:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(
                y_rank[val_idx][rank_mask],
                gbm.rank_model.predict(X[val_idx][rank_mask]),
            )
            rank_spearman = float(rho) if np.isfinite(rho) else 0.0

    skip_metrics = binary_classification_metrics(
        y_gate[val_idx], gate_scores, threshold=0.1
    )
    skipped = skip_metrics["tn"] + skip_metrics["fn"]
    false_skip_rate = float(skip_metrics["fn"] / skipped) if skipped else 0.0
    return {
        "split_stats": split_stats,
        "val_metrics_threshold_0_5": binary_classification_metrics(
            y_gate[val_idx], gate_scores, threshold=0.5
        ),
        "val_metrics_threshold_0_1_skip_rule": skip_metrics,
        "skip_rate": float(skipped / max(len(val_idx), 1)),
        "false_skip_rate": false_skip_rate,
        "rank_spearman_val": rank_spearman,
    }


def _component_scores(
    ensemble,
    db_path: str,
    sample_limit: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    from research.scientist.intelligence.ml_corpus import (
        grouped_stratified_split,
        load_screening_predictor_corpus_rows,
    )
    from research.synthesis.graph_features import (
        enrich_with_op_stats,
        extract_graph_features_bundle,
        load_op_stats,
    )

    rows = [
        row
        for row in load_screening_predictor_corpus_rows(db_path)
        if bool(row.get("stage0_any_passed"))
    ]
    if len(rows) > sample_limit:
        rows = random.Random(0).sample(rows, sample_limit)

    op_stats_cache = load_op_stats(db_path)
    score_rows: list[list[float]] = []
    labels: list[int] = []
    graph_signatures: list[str] = []

    for row in rows:
        graph_json = row["graph_json"]
        label = int(bool(row["stage1_any_passed"]))
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature:
            continue
        try:
            graph = (
                json.loads(graph_json) if isinstance(graph_json, str) else graph_json
            )
        except (json.JSONDecodeError, TypeError):
            continue

        component_scores: list[float] = []
        if ensemble.gbm is not None and ensemble.gbm.is_fitted():
            feats, ops = extract_graph_features_bundle(graph)
            if feats:
                for op in ops:
                    if op:
                        feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
                enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
                component_scores.append(ensemble.gbm.predict_gate(feats))
            else:
                component_scores.append(0.5)
        else:
            component_scores.append(0.5)

        if ensemble.graph_pred is not None and ensemble.graph_pred.is_fitted():
            component_scores.append(ensemble.graph_pred.predict_gate(graph))
        else:
            component_scores.append(0.5)

        score_rows.append(component_scores)
        labels.append(label)
        graph_signatures.append(signature)

    X = np.array(score_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    _train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y, seed=42
    )
    scores_norm = (X - ensemble._score_mean) / ensemble._score_std
    val_scores = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                scores_norm[val_idx] @ ensemble.w_ensemble + ensemble.b_ensemble,
                -15,
                15,
            )
        )
    )
    return y[val_idx], val_scores, split_stats


def _ensemble_metrics_report(
    db_path: str,
    profiling_db: str,
    state_dir: Path,
    fresh_ensemble: bool,
) -> dict[str, Any]:
    from research.scientist.intelligence.metrics_utils import (
        binary_classification_metrics,
    )
    from research.scientist.intelligence.predictor import (
        EnsemblePredictor,
        train_ensemble,
    )

    if fresh_ensemble:
        ensemble = train_ensemble(db_path=db_path, profiling_db=profiling_db)
        source = "fresh_train_ensemble"
    else:
        ensemble = EnsemblePredictor.load(
            state_dir=state_dir, profiling_db=profiling_db
        )
        source = "saved_runtime_artifacts"

    if (
        ensemble.w_ensemble.size == 0
        or ensemble._score_mean.size == 0
        or ensemble._score_std.size == 0
    ):
        return {"error": "ensemble_not_calibrated", "source": source}

    y_true, y_score, split_stats = _component_scores(ensemble, db_path)
    return {
        "source": source,
        "persisted_meta": _load_json_if_exists(state_dir / ENSEMBLE_META_PATH.name),
        "split_stats": split_stats,
        "val_metrics_selected_threshold": binary_classification_metrics(
            y_true, y_score, threshold=ensemble.gate_threshold
        ),
        "val_metrics_threshold_0_5": binary_classification_metrics(
            y_true, y_score, threshold=0.5
        ),
        "weights": [float(x) for x in ensemble.w_ensemble.tolist()],
        "bias": float(ensemble.b_ensemble),
    }


def _interaction_metrics_report(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "interaction_model.json"
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    return {
        "persisted_train_metrics": dict(meta.get("train_metrics", {})),
        "note": (
            "Current pipeline does not persist holdout ROC/PPV/NPV metrics for "
            "InteractionModel; only optimization loss and sample counts are available."
        ),
    }


def _investigation_predictor_metrics_report(db_path: str) -> dict[str, Any]:
    from research.scientist.intelligence.predictor import (
        evaluate as evaluate_investigation_predictor,
    )
    from research.scientist.notebook import LabNotebook

    nb = LabNotebook(db_path)
    try:
        return evaluate_investigation_predictor(nb)
    finally:
        nb.close()


def _build_predictor_metrics_report(
    *,
    db_path: str,
    profiling_db: str,
    state_dir: Path,
    fresh_ensemble: bool,
) -> dict[str, Any]:
    return {
        "paths": {
            "db_path": db_path,
            "profiling_db": profiling_db,
            "state_dir": str(state_dir),
        },
        "graph_predictor": _safe_report_section(
            "graph_predictor",
            _graph_predictor_metrics_report,
            state_dir,
        ),
        "gbm_gate": _safe_report_section(
            "gbm_gate",
            _gbm_metrics_report,
            db_path,
            state_dir,
        ),
        "ensemble_calibrated": _safe_report_section(
            "ensemble_calibrated",
            _ensemble_metrics_report,
            db_path,
            profiling_db,
            state_dir,
            fresh_ensemble,
        ),
        "interaction_model": _safe_report_section(
            "interaction_model",
            _interaction_metrics_report,
            state_dir,
        ),
        "bayesian_tracker": {
            "note": (
                "Temporal Bayesian tracker is used as a posterior/ranking component. "
                "The current saved runtime state does not emit ROC/PPV/NPV metrics."
            )
        },
        "investigation_predictor": _safe_report_section(
            "investigation_predictor",
            _investigation_predictor_metrics_report,
            db_path,
        ),
    }


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
        state_dir = ensure_state_dir(STATE_DIR)
        tracker.save_state(state_dir / BAYESIAN_STATE_PATH.name)

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
        state_dir = ensure_state_dir(STATE_DIR)
        emb.save(state_dir / OP_EMBEDDINGS_PATH.name)

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
        state_dir = ensure_state_dir(STATE_DIR)
        model.save(state_dir / INTERACTION_MODEL_PATH.name)

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
        state_dir = ensure_state_dir(STATE_DIR)
        model.save(state_dir / GRAPH_PREDICTOR_PATH.name)

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
        ensemble.save(ensure_state_dir(STATE_DIR))

    return {"component": "ensemble", "elapsed_s": elapsed, **diag}


def write_metrics_report(fresh_ensemble: bool = False) -> dict:
    """Write predictor metrics report to the runtime state directory."""
    report = _build_predictor_metrics_report(
        db_path=str(_NOTEBOOK_DB),
        profiling_db=str(_PROFILING_DB),
        state_dir=STATE_DIR,
        fresh_ensemble=fresh_ensemble,
    )
    state_dir = ensure_state_dir(STATE_DIR)
    report_path = state_dir / METRICS_REPORT_PATH.name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("Predictor metrics report written to %s", report_path)
    return {"report_path": str(report_path)}


def evaluate_all() -> dict:
    """Evaluate all predictors and report metrics."""
    from research.scientist.intelligence.predictor import (
        analyze_graph_label_quality,
        evaluate_gbm,
        evaluate_gbm_induction,
    )

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

    # Induction-specific GBM evaluation
    try:
        gbm_induction_eval = evaluate_gbm_induction(str(_NOTEBOOK_DB))
        results["gbm_induction"] = gbm_induction_eval
        logger.info(
            "GBM induction eval: learner_auc=%.3f, induction_spearman=%.3f",
            gbm_induction_eval.get("learner_auc", 0),
            gbm_induction_eval.get("induction_spearman", 0),
        )
    except Exception as e:
        results["gbm_induction"] = {"error": str(e)}

    # Label-quality summary
    try:
        label_quality = analyze_graph_label_quality(str(_NOTEBOOK_DB))
        results["label_quality"] = label_quality
        logger.info("Label quality: %s", label_quality)
    except Exception as e:
        results["label_quality"] = {"error": str(e)}

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
    nb, exp_id = start_script_experiment(
        db_path=_NOTEBOOK_DB,
        experiment_type="predictor_training",
        config={
            "component": args.component,
            "evaluate": bool(args.evaluate),
            "save_state": bool(args.save_state),
            "heatmaps": bool(args.heatmaps),
        },
        source_script="train_predictors",
        hypothesis=f"Train predictor components ({args.component})",
    )

    try:
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
            results["heatmaps"] = {"generated": True}

        if args.evaluate:
            results["evaluation"] = evaluate_all()
        if args.save_state:
            try:
                results["metrics_report"] = write_metrics_report(
                    fresh_ensemble=args.component in ("all", "ensemble")
                )
            except Exception as e:
                logger.warning("Failed to write predictor metrics report: %s", e)
                results["metrics_report"] = {"error": str(e)}

        total = time.time() - t0
        logger.info("Total training time: %.1fs", total)
        for name, res in results.items():
            logger.info("  %s: %s", name, res)

        complete_script_experiment(
            nb,
            exp_id,
            results={
                "components_trained": len(results),
                "component_names": sorted(results.keys()),
                "elapsed_s": round(total, 3),
                "save_state": bool(args.save_state),
                "evaluate": bool(args.evaluate),
            },
            summary=(
                f"Predictor training complete: components={len(results)} "
                f"component={args.component}"
            ),
        )
    except KeyboardInterrupt:
        fail_script_experiment(
            nb,
            exp_id,
            error="KeyboardInterrupt",
            results={"components_trained": len(results)},
        )
        nb.close()
        raise
    except Exception as exc:
        fail_script_experiment(
            nb,
            exp_id,
            error=str(exc),
            results={"components_trained": len(results)},
        )
        nb.close()
        raise
    nb.close()


if __name__ == "__main__":
    main()
