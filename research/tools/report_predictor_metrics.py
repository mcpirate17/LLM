#!/usr/bin/env python3
"""Report held-out metrics for the research ML predictors.

Usage:
    python -m research.tools.report_predictor_metrics
    python -m research.tools.report_predictor_metrics --json
    python -m research.tools.report_predictor_metrics --fresh-ensemble
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

from research.scientist.intelligence.ml_corpus import (
    load_deduped_screening_predictor_rows,
)
from research.scientist.intelligence.metrics_utils import binary_classification_metrics
from research.scientist.intelligence.predictor import (
    GBMPredictor,
    EnsemblePredictor,
    _dicts_to_matrix,
    _grouped_stratified_split,
    _query_graph_training_data,
    evaluate as evaluate_investigation_predictor,
    train_ensemble,
)
from research.synthesis.graph_features import (
    enrich_with_op_stats,
    extract_graph_features,
    load_op_stats,
)

DEFAULT_DB = "research/lab_notebook.db"
DEFAULT_PROFILING_DB = "research/profiling/component_profiles.db"
DEFAULT_STATE_DIR = Path("research/runtime/learning")


def _safe_report_section(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc), "section": name}


def _derive_confusion_from_summary(
    *,
    n: int,
    accuracy: float,
    precision: float,
    recall: float,
) -> Dict[str, float]:
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


def _graph_predictor_report(state_dir: Path) -> Dict[str, Any]:
    graph_meta_path = state_dir / "graph_predictor.json"
    if not graph_meta_path.exists():
        return {"error": "graph_predictor_not_fitted"}
    with open(graph_meta_path, encoding="utf-8") as f:
        meta = json.load(f)
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


def _gbm_report(db_path: str, state_dir: Path) -> Dict[str, Any]:
    feat_dicts, y_gate, y_rank, _sample_weights, graph_signatures = (
        _query_graph_training_data(db_path)
    )
    X, feature_names = _dicts_to_matrix(feat_dicts)
    _train_idx, val_idx, split_stats = _grouped_stratified_split(
        graph_signatures, y_gate, seed=42
    )
    gbm = GBMPredictor.load(state_dir)
    if not gbm.is_fitted():
        return {"error": "gbm_not_fitted"}
    # Build gate-only feature matrix (gate model has fewer features than rank)
    gate_names = gbm.gate_feature_names or gbm.feature_names
    gate_col_idx = [feature_names.index(fn) for fn in gate_names if fn in feature_names]
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
    ensemble: EnsemblePredictor,
    db_path: str,
    sample_limit: int = 2000,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rows = [
        row
        for row in load_deduped_screening_predictor_rows(db_path)
        if bool(row.get("stage0_any_passed"))
    ]
    if len(rows) > sample_limit:
        rows = random.Random(0).sample(rows, sample_limit)

    op_stats_cache = load_op_stats(db_path)
    score_rows = []
    labels = []
    graph_signatures = []

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

        component_scores = []
        if ensemble.gbm is not None and ensemble.gbm.is_fitted():
            feats = extract_graph_features(graph)
            if feats:
                nodes = graph.get("nodes") or {}
                ops = [
                    node.get("op_name", "")
                    for node in nodes.values()
                    if node.get("op_name", "") != "input"
                ]
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

        # Bayesian worst-op excluded from scoring — weight was -0.006 (noise).
        # InteractionModel excluded — calibration weight was ≈ -0.06.

        score_rows.append(component_scores)
        labels.append(label)
        graph_signatures.append(signature)

    X = np.array(score_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    _train_idx, val_idx, split_stats = _grouped_stratified_split(
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


def _ensemble_report(
    db_path: str,
    profiling_db: str,
    state_dir: Path,
    fresh_ensemble: bool,
) -> Dict[str, Any]:
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
        "persisted_meta": _load_json_if_exists(state_dir / "ensemble_state.json"),
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


def _interaction_report(state_dir: Path) -> Dict[str, Any]:
    path = state_dir / "interaction_model.json"
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "persisted_train_metrics": dict(meta.get("train_metrics", {})),
        "note": (
            "Current pipeline does not persist holdout ROC/PPV/NPV metrics for "
            "InteractionModel; only optimization loss and sample counts are available."
        ),
    }


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def build_report(
    *,
    db_path: str,
    profiling_db: str,
    state_dir: Path,
    fresh_ensemble: bool,
) -> Dict[str, Any]:
    report = {
        "paths": {
            "db_path": db_path,
            "profiling_db": profiling_db,
            "state_dir": str(state_dir),
        },
        "graph_predictor": _safe_report_section(
            "graph_predictor",
            _graph_predictor_report,
            state_dir,
        ),
        "gbm_gate": _safe_report_section(
            "gbm_gate",
            _gbm_report,
            db_path,
            state_dir,
        ),
        "ensemble_calibrated": _safe_report_section(
            "ensemble_calibrated",
            _ensemble_report,
            db_path,
            profiling_db,
            state_dir,
            fresh_ensemble,
        ),
        "interaction_model": _safe_report_section(
            "interaction_model",
            _interaction_report,
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
            lambda: _investigation_predictor_report(db_path),
        ),
    }
    return report


def _investigation_predictor_report(db_path: str) -> Dict[str, Any]:
    from research.scientist.notebook import LabNotebook

    nb = LabNotebook(db_path)
    try:
        return evaluate_investigation_predictor(nb)
    finally:
        nb.close()


def _print_summary(report: Dict[str, Any]) -> None:
    gp = report["graph_predictor"]["derived_val_classification_metrics"]
    gbm = (
        report["gbm_gate"].get("val_metrics_selected_threshold")
        or report["gbm_gate"]["val_metrics_threshold_0_5"]
    )
    ens = report["ensemble_calibrated"].get("val_metrics_selected_threshold") or report[
        "ensemble_calibrated"
    ].get("val_metrics_threshold_0_5", {})
    print("Predictor held-out metrics")
    print(
        "GraphPredictor  "
        f"auc={gp['roc_auc']:.4f} acc={gp['accuracy']:.4f} "
        f"ppv={gp['precision_ppv']:.4f} tpr={gp['recall_tpr_sensitivity']:.4f} "
        f"npv={gp['npv']:.4f}"
    )
    print(
        "GBM gate        "
        f"auc={gbm['roc_auc']:.4f} acc={gbm['accuracy']:.4f} "
        f"ppv={gbm['precision_ppv']:.4f} tpr={gbm['recall_tpr_sensitivity']:.4f} "
        f"npv={gbm['npv']:.4f}"
    )
    if ens:
        print(
            "Ensemble        "
            f"auc={ens['roc_auc']:.4f} acc={ens['accuracy']:.4f} "
            f"ppv={ens['precision_ppv']:.4f} tpr={ens['recall_tpr_sensitivity']:.4f} "
            f"npv={ens['npv']:.4f}"
        )
    skip_rate = report["gbm_gate"]["skip_rate"]
    false_skip_rate = report["gbm_gate"]["false_skip_rate"]
    print(
        "GBM skip rule    "
        f"skip_rate={skip_rate:.4f} false_skip_rate={false_skip_rate:.4f}"
    )
    print(f"Interaction      {report['interaction_model']['persisted_train_metrics']}")
    print(report["graph_predictor"]["note"])
    if report["ensemble_calibrated"].get("source") == "saved_runtime_artifacts":
        print(
            "Ensemble note    loading saved artifacts is fastest; "
            "use --fresh-ensemble for exact current-codepath metrics."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report held-out metrics for research ML predictors."
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--profiling-db", default=DEFAULT_PROFILING_DB)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--fresh-ensemble", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = build_report(
        db_path=args.db,
        profiling_db=args.profiling_db,
        state_dir=Path(args.state_dir),
        fresh_ensemble=args.fresh_ensemble,
    )
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
