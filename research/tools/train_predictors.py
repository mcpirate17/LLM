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
from datetime import datetime, timezone
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
_MODEL_REGISTRY_PATH = STATE_DIR / "model_registry.json"


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


def _selected_metrics_from_section(
    section: dict[str, Any] | None,
    key: str,
) -> dict[str, Any]:
    payload = section or {}
    selected = (payload.get(key) or {}).get("selected_metrics") or {}
    return dict(selected) if isinstance(selected, dict) else {}


def _build_model_registry(report: dict[str, Any]) -> dict[str, Any]:
    from research.scientist.ml_influence_policy import build_ml_influence_policy
    from research.scientist.runner import RunConfig

    policy = build_ml_influence_policy(config=RunConfig(), report=report)
    components = dict(policy.get("components") or {})

    def _component_status(name: str, default_status: str) -> str:
        comp = components.get(name) or {}
        if comp.get("proven") is True:
            return default_status
        quality_tier = str(comp.get("quality_tier") or "")
        if quality_tier == "usable":
            return "restricted"
        if quality_tier == "weak":
            return "advisory_only"
        if quality_tier == "monitor_only":
            return "monitor_only"
        return "insufficient_evidence"

    ensemble = report.get("ensemble_calibrated") or {}
    gbm = report.get("gbm_gate") or {}
    graph = report.get("graph_predictor") or {}
    investigation = report.get("investigation_predictor") or {}

    models: dict[str, Any] = {
        "ensemble_calibrated": {
            "status": _component_status("screening_ensemble", "production"),
            "quality_tier": (components.get("screening_ensemble") or {}).get(
                "quality_tier"
            ),
            "metric_source": (components.get("screening_ensemble") or {}).get(
                "metric_source"
            ),
            "runtime_active": bool(
                (components.get("screening_ensemble") or {}).get("allowed")
            ),
            "saved_runtime_artifact_metrics": _selected_metrics_from_section(
                ensemble, "saved_runtime_artifact_evaluation"
            ),
            "temporal_holdout_metrics": _selected_metrics_from_section(
                ensemble, "temporal_holdout_evaluation"
            ),
        },
        "gbm_gate": {
            "status": _component_status("gbm_gate", "production"),
            "quality_tier": (components.get("gbm_gate") or {}).get("quality_tier"),
            "metric_source": (components.get("gbm_gate") or {}).get("metric_source"),
            "saved_runtime_artifact_metrics": _selected_metrics_from_section(
                gbm, "saved_runtime_artifact_evaluation"
            ),
            "temporal_holdout_metrics": _selected_metrics_from_section(
                gbm, "temporal_holdout_evaluation"
            ),
        },
        "gbm_rank_ppl": {
            "status": (
                "production"
                if ((gbm.get("persisted_train_metrics") or {}).get("rank_heads") or {})
                .get("ppl", {})
                .get("spearman", 0.0)
                >= 0.5
                else "needs_review"
            ),
            "saved_runtime_artifact_metrics": dict(
                (gbm.get("rank_heads") or {}).get("ppl") or {}
            ),
            "temporal_holdout_metrics": dict(
                (gbm.get("temporal_rank_heads") or {}).get("ppl") or {}
            ),
        },
        "gbm_rank_composite": {
            "status": (
                "production"
                if ((gbm.get("persisted_train_metrics") or {}).get("rank_heads") or {})
                .get("composite", {})
                .get("spearman", 0.0)
                >= 0.5
                else "needs_review"
            ),
            "saved_runtime_artifact_metrics": dict(
                (gbm.get("rank_heads") or {}).get("composite") or {}
            ),
            "temporal_holdout_metrics": dict(
                (gbm.get("temporal_rank_heads") or {}).get("composite") or {}
            ),
        },
        "graph_predictor": {
            "status": _component_status("graph_predictor", "advisory_only"),
            "quality_tier": (components.get("graph_predictor") or {}).get(
                "quality_tier"
            ),
            "metric_source": (components.get("graph_predictor") or {}).get(
                "metric_source"
            ),
            "saved_runtime_artifact_metrics": _selected_metrics_from_section(
                graph, "saved_runtime_artifact_evaluation"
            ),
            "temporal_holdout_metrics": _selected_metrics_from_section(
                graph, "temporal_holdout_evaluation"
            ),
        },
        "investigation_predictor": {
            "status": _component_status("investigation_predictor", "production"),
            "quality_tier": (components.get("investigation_predictor") or {}).get(
                "quality_tier"
            ),
            "metric_source": "saved_runtime_artifact_metrics",
            "runtime_active": bool(
                (components.get("investigation_predictor") or {}).get("allowed")
            ),
            "saved_runtime_artifact_metrics": {
                key: value
                for key, value in dict(investigation).items()
                if isinstance(value, (int, float, str, bool))
            },
        },
        "interaction_model": {
            "status": "insufficient_evidence",
            "saved_runtime_artifact_metrics": dict(
                ((report.get("interaction_model") or {}).get("persisted_train_metrics"))
                or {}
            ),
        },
        "bayesian_tracker": {
            "status": "insufficient_evidence",
            "saved_runtime_artifact_metrics": dict(
                report.get("bayesian_tracker") or {}
            ),
        },
    }

    return {
        "registry_version": 2,
        "updated_at": _iso_utc_now(),
        "source_report_schema_version": report.get("report_schema_version"),
        "policy_active_generation_influencers": list(
            policy.get("active_generation_influencers") or []
        ),
        "models": models,
    }


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _regression_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = np.asarray(y_true[mask], dtype=np.float64)
    y_pred_f = np.asarray(y_pred[mask], dtype=np.float64)
    if y_true_f.size < 2:
        return {"n": int(y_true_f.size), "error": "insufficient_data"}

    metrics: dict[str, Any] = {
        "n": int(y_true_f.size),
        "mae": float(np.mean(np.abs(y_true_f - y_pred_f))),
        "rmse": float(np.sqrt(np.mean((y_true_f - y_pred_f) ** 2))),
    }
    try:
        from scipy.stats import kendalltau, pearsonr, spearmanr

        rho, _ = spearmanr(y_true_f, y_pred_f)
        pearson = pearsonr(y_true_f, y_pred_f).statistic
        kendall = kendalltau(y_true_f, y_pred_f).statistic
        metrics.update(
            {
                "spearman": float(rho) if np.isfinite(rho) else 0.0,
                "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
                "kendall": float(kendall) if np.isfinite(kendall) else 0.0,
            }
        )
    except Exception:
        metrics.update({"spearman": 0.0, "pearson": 0.0, "kendall": 0.0})
    return metrics


def _binary_evaluation_bundle(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    selected_threshold: float,
    source: str,
    split_stats: dict[str, Any] | None = None,
    extra_thresholds: tuple[float, ...] = (),
    include_predictions: bool = True,
) -> dict[str, Any]:
    from research.scientist.intelligence.metrics_utils import (
        binary_classification_metrics,
        operating_point_profiles,
        reliability_curve,
    )

    y_true_arr = np.asarray(y_true).astype(np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    threshold_metrics = {
        str(float(threshold)): binary_classification_metrics(
            y_true_arr, y_score_arr, threshold=float(threshold)
        )
        for threshold in (selected_threshold, 0.5, *extra_thresholds)
    }
    bundle: dict[str, Any] = {
        "evaluated_at": _iso_utc_now(),
        "source": source,
        "split_stats": dict(split_stats or {}),
        "selected_threshold": float(selected_threshold),
        "selected_metrics": threshold_metrics[str(float(selected_threshold))],
        "threshold_0_5_metrics": threshold_metrics[str(0.5)],
        "operating_points": operating_point_profiles(y_true_arr, y_score_arr),
        "reliability_curve": reliability_curve(y_true_arr, y_score_arr),
    }
    if extra_thresholds:
        bundle["extra_threshold_metrics"] = {
            str(float(threshold)): threshold_metrics[str(float(threshold))]
            for threshold in extra_thresholds
        }
    if include_predictions:
        bundle["holdout_predictions"] = {
            "n": int(y_true_arr.size),
            "y_true": y_true_arr.astype(int).tolist(),
            "y_score": y_score_arr.astype(float).tolist(),
        }
    return bundle


def _attach_primary_eval_aliases(
    section: dict[str, Any],
    *,
    primary_key: str,
) -> dict[str, Any]:
    primary = dict(section.get(primary_key) or {})
    if not primary:
        return section
    section["primary_source"] = str(primary.get("source") or primary_key)
    section["source"] = str(primary.get("source") or primary_key)
    if "split_stats" in primary:
        section["split_stats"] = dict(primary.get("split_stats") or {})
    if "selected_metrics" in primary:
        section["val_metrics_selected_threshold"] = dict(
            primary.get("selected_metrics") or {}
        )
    if "threshold_0_5_metrics" in primary:
        section["val_metrics_threshold_0_5"] = dict(
            primary.get("threshold_0_5_metrics") or {}
        )
    if "operating_points" in primary:
        section["operating_points"] = dict(primary.get("operating_points") or {})
    if "reliability_curve" in primary:
        section["reliability_curve"] = list(primary.get("reliability_curve") or [])
    return section


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


def _evaluate_graph_predictor_temporal(
    *,
    db_path: str,
    state_dir: Path,
    selected_threshold: float,
) -> dict[str, Any]:
    from research.scientist.intelligence.gnn_predictor import GraphPredictor
    from research.scientist.intelligence.ml_corpus import (
        grouped_temporal_split,
        load_screening_predictor_corpus_rows,
    )

    graph_path = state_dir / GRAPH_PREDICTOR_PATH.name
    if not graph_path.exists():
        return {"error": "graph_predictor_not_fitted"}
    model = GraphPredictor.load(graph_path)
    if not model.is_fitted():
        return {"error": "graph_predictor_not_fitted"}

    labels: list[int] = []
    scores: list[float] = []
    signatures: list[str] = []
    timestamps: list[float] = []
    for row in load_screening_predictor_corpus_rows(db_path):
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature:
            continue
        graph_json = row.get("graph_json")
        try:
            graph = (
                json.loads(graph_json) if isinstance(graph_json, str) else graph_json
            )
        except (json.JSONDecodeError, TypeError):
            continue
        labels.append(int(bool(row.get("stage1_any_passed"))))
        scores.append(float(model.predict_gate(graph)))
        signatures.append(signature)
        timestamps.append(float(row.get("latest_timestamp") or 0.0))

    if len(labels) < 10:
        return {"error": "insufficient_data"}
    y = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    _train_idx, val_idx, split_stats = grouped_temporal_split(
        signatures,
        y,
        np.asarray(timestamps, dtype=np.float64),
    )
    if len(val_idx) == 0:
        return {"error": "temporal_split_failed", "split_stats": split_stats}
    return _binary_evaluation_bundle(
        y_true=y[val_idx],
        y_score=y_score[val_idx],
        selected_threshold=selected_threshold,
        source="saved_runtime_artifact_temporal",
        split_stats=split_stats,
    )


def _graph_predictor_metrics_report(
    state_dir: Path, db_path: str | None = None
) -> dict[str, Any]:
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
    section: dict[str, Any] = {
        "persisted_train_metrics": train_metrics,
        "saved_runtime_artifact_evaluation": {
            "evaluated_at": _iso_utc_now(),
            "source": "saved_runtime_artifact",
            "split_stats": {
                "n_val": int(train_metrics.get("n_val", 0)),
                "n_positive": int(train_metrics.get("n_positive", 0)),
            },
            "selected_threshold": float(derived.get("threshold", 0.5)),
            "selected_metrics": derived,
            "threshold_0_5_metrics": (
                dict(derived)
                if np.isclose(float(derived.get("threshold", 0.5)), 0.5)
                else {}
            ),
            "operating_points": dict(train_metrics.get("operating_points", {})),
            "reliability_curve": [],
        },
        "derived_val_classification_metrics": derived,
        "note": (
            "Uses persisted holdout metrics from graph_predictor.json. "
            "This is the most faithful report for the saved GraphPredictor artifact."
        ),
    }
    if db_path:
        section["temporal_holdout_evaluation"] = _evaluate_graph_predictor_temporal(
            db_path=db_path,
            state_dir=state_dir,
            selected_threshold=float(derived.get("threshold", 0.5)),
        )
    return _attach_primary_eval_aliases(
        section,
        primary_key="saved_runtime_artifact_evaluation",
    )


def _gbm_metrics_report(db_path: str, state_dir: Path) -> dict[str, Any]:
    from research.scientist.intelligence.ml_corpus import (
        build_dense_feature_matrix,
        grouped_stratified_split,
        grouped_temporal_split,
    )
    from research.scientist.intelligence.predictor_gbm import _ranking_diagnostics
    from research.scientist.intelligence.predictor import (
        GBMPredictor,
        _query_graph_training_data,
    )

    (
        feat_dicts,
        y_gate,
        y_rank_ppl,
        y_rank_composite,
        _sample_weights,
        latest_timestamps,
        graph_signatures,
    ) = _query_graph_training_data(db_path)
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

    gate_scores = gbm.gate_model.predict(X_gate[val_idx])
    saved_eval = _binary_evaluation_bundle(
        y_true=y_gate[val_idx],
        y_score=gate_scores,
        selected_threshold=float(gbm.gate_threshold),
        source="saved_runtime_artifact",
        split_stats=split_stats,
        extra_thresholds=(0.1,),
    )
    skip_metrics = dict(
        (saved_eval.get("extra_threshold_metrics") or {}).get(str(0.1), {})
    )
    skipped = int(skip_metrics.get("tn", 0) + skip_metrics.get("fn", 0))
    false_skip_rate = float(skip_metrics.get("fn", 0) / skipped) if skipped else 0.0

    rank_heads: dict[str, Any] = {}
    if gbm.rank_model_ppl is not None:
        rank_mask = np.isfinite(y_rank_ppl[val_idx])
        if rank_mask.sum() >= 2:
            rank_heads["ppl"] = _ranking_diagnostics(
                y_rank_ppl[val_idx][rank_mask],
                gbm.rank_model_ppl.predict(X[val_idx][rank_mask]),
                target_kind="ppl",
            )
        else:
            rank_heads["ppl"] = {
                "n": int(rank_mask.sum()),
                "error": "insufficient_data",
            }
    if gbm.rank_model_composite is not None:
        rank_mask = np.isfinite(y_rank_composite[val_idx])
        if rank_mask.sum() >= 2:
            rank_heads["composite"] = _ranking_diagnostics(
                y_rank_composite[val_idx][rank_mask],
                gbm.rank_model_composite.predict(X[val_idx][rank_mask]),
                target_kind="composite",
            )
        else:
            rank_heads["composite"] = {
                "n": int(rank_mask.sum()),
                "error": "insufficient_data",
            }
    if not rank_heads and gbm.legacy_mixed_rank_model_loaded:
        rank_heads["legacy_mixed"] = {
            "error": "legacy_mixed_rank_model_ignored_until_retrained"
        }

    _train_idx_tm, val_idx_tm, temporal_split_stats = grouped_temporal_split(
        graph_signatures,
        y_gate,
        latest_timestamps,
    )
    temporal_eval = {
        "error": "temporal_split_failed",
        "split_stats": temporal_split_stats,
    }
    temporal_rank_heads: dict[str, Any] = {}
    if len(val_idx_tm) > 0:
        gate_scores_tm = gbm.gate_model.predict(X_gate[val_idx_tm])
        temporal_eval = _binary_evaluation_bundle(
            y_true=y_gate[val_idx_tm],
            y_score=gate_scores_tm,
            selected_threshold=float(gbm.gate_threshold),
            source="saved_runtime_artifact_temporal",
            split_stats=temporal_split_stats,
            extra_thresholds=(0.1,),
        )
        if gbm.rank_model_ppl is not None:
            rank_mask = np.isfinite(y_rank_ppl[val_idx_tm])
            if rank_mask.sum() >= 2:
                raw_preds = gbm.rank_model_ppl.predict(X[val_idx_tm][rank_mask])
                ppl_preds = np.exp(
                    np.clip(raw_preds, gbm.rank_ppl_log_min, gbm.rank_ppl_log_max)
                )
                temporal_rank_heads["ppl"] = _ranking_diagnostics(
                    y_rank_ppl[val_idx_tm][rank_mask],
                    ppl_preds,
                    target_kind="ppl",
                )
            else:
                temporal_rank_heads["ppl"] = {
                    "n": int(rank_mask.sum()),
                    "error": "insufficient_data",
                }
        if gbm.rank_model_composite is not None:
            rank_mask = np.isfinite(y_rank_composite[val_idx_tm])
            if rank_mask.sum() >= 2:
                temporal_rank_heads["composite"] = _ranking_diagnostics(
                    y_rank_composite[val_idx_tm][rank_mask],
                    gbm.rank_model_composite.predict(X[val_idx_tm][rank_mask]),
                    target_kind="composite",
                )
            else:
                temporal_rank_heads["composite"] = {
                    "n": int(rank_mask.sum()),
                    "error": "insufficient_data",
                }

    section: dict[str, Any] = {
        "split_stats": split_stats,
        "persisted_train_metrics": dict(gbm.train_metrics or {}),
        "saved_runtime_artifact_evaluation": saved_eval,
        "temporal_holdout_evaluation": temporal_eval,
        "val_metrics_threshold_0_1_skip_rule": skip_metrics,
        "skip_rate": float(skipped / max(len(val_idx), 1)),
        "false_skip_rate": false_skip_rate,
        "rank_heads": rank_heads,
        "temporal_rank_heads": temporal_rank_heads,
        "rank_spearman_val": (rank_heads.get("ppl") or {}).get("spearman"),
    }
    return _attach_primary_eval_aliases(
        section,
        primary_key="saved_runtime_artifact_evaluation",
    )


def _component_score_frame(
    ensemble,
    db_path: str,
    sample_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    from research.scientist.intelligence.ml_corpus import (
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
    if sample_limit is not None and len(rows) > sample_limit:
        rows = random.Random(0).sample(rows, sample_limit)

    op_stats_cache = load_op_stats(db_path)
    score_rows: list[list[float]] = []
    labels: list[int] = []
    graph_signatures: list[str] = []
    latest_timestamps: list[float] = []

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
        latest_timestamps.append(float(row.get("latest_timestamp") or 0.0))

    X = np.array(score_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return y, X, graph_signatures, np.asarray(latest_timestamps, dtype=np.float64)


def _ensemble_scores_from_components(
    ensemble,
    X: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    scores_norm = (X - ensemble._score_mean) / ensemble._score_std
    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                scores_norm[idx] @ ensemble.w_ensemble + ensemble.b_ensemble,
                -15,
                15,
            )
        )
    )


def _ensemble_metrics_report(
    db_path: str,
    profiling_db: str,
    state_dir: Path,
    fresh_ensemble: bool,
) -> dict[str, Any]:
    from research.scientist.intelligence.predictor import (
        EnsemblePredictor,
        train_ensemble,
    )
    from research.scientist.intelligence.ml_corpus import (
        grouped_stratified_split,
        grouped_temporal_split,
    )

    ensemble = EnsemblePredictor.load(state_dir=state_dir, profiling_db=profiling_db)
    if (
        ensemble.w_ensemble.size == 0
        or ensemble._score_mean.size == 0
        or ensemble._score_std.size == 0
    ):
        return {"error": "ensemble_not_calibrated", "source": "saved_runtime_artifact"}

    y_all, X_all, graph_signatures, latest_timestamps = _component_score_frame(
        ensemble,
        db_path,
    )
    _train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures,
        y_all,
        seed=42,
    )
    y_true = y_all[val_idx]
    y_score = _ensemble_scores_from_components(ensemble, X_all, val_idx)
    section: dict[str, Any] = {
        "persisted_meta": _load_json_if_exists(state_dir / ENSEMBLE_META_PATH.name),
        "saved_runtime_artifact_evaluation": _binary_evaluation_bundle(
            y_true=y_true,
            y_score=y_score,
            selected_threshold=float(ensemble.gate_threshold),
            source="saved_runtime_artifact",
            split_stats=split_stats,
        ),
        "weights": [float(x) for x in ensemble.w_ensemble.tolist()],
        "bias": float(ensemble.b_ensemble),
    }
    _train_idx_tm, val_idx_tm, temporal_split_stats = grouped_temporal_split(
        graph_signatures,
        y_all,
        latest_timestamps,
    )
    if len(val_idx_tm) > 0:
        section["temporal_holdout_evaluation"] = _binary_evaluation_bundle(
            y_true=y_all[val_idx_tm],
            y_score=_ensemble_scores_from_components(ensemble, X_all, val_idx_tm),
            selected_threshold=float(ensemble.gate_threshold),
            source="saved_runtime_artifact_temporal",
            split_stats=temporal_split_stats,
        )
    else:
        section["temporal_holdout_evaluation"] = {
            "error": "temporal_split_failed",
            "split_stats": temporal_split_stats,
        }
    if fresh_ensemble:
        fresh = train_ensemble(db_path=db_path, profiling_db=profiling_db)
        if (
            fresh.w_ensemble.size == 0
            or fresh._score_mean.size == 0
            or fresh._score_std.size == 0
        ):
            section["fresh_train_comparison"] = {
                "error": "ensemble_not_calibrated",
                "source": "fresh_train_comparison",
            }
        else:
            (
                fresh_y_all,
                fresh_X_all,
                fresh_graph_signatures,
                _fresh_latest_timestamps,
            ) = _component_score_frame(
                fresh,
                db_path,
            )
            _fresh_train_idx, fresh_val_idx, fresh_split_stats = (
                grouped_stratified_split(
                    fresh_graph_signatures,
                    fresh_y_all,
                    seed=42,
                )
            )
            section["fresh_train_comparison"] = _binary_evaluation_bundle(
                y_true=fresh_y_all[fresh_val_idx],
                y_score=_ensemble_scores_from_components(
                    fresh, fresh_X_all, fresh_val_idx
                ),
                selected_threshold=float(fresh.gate_threshold),
                source="fresh_train_comparison",
                split_stats=fresh_split_stats,
            )
    return _attach_primary_eval_aliases(
        section,
        primary_key="saved_runtime_artifact_evaluation",
    )


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
        "report_schema_version": 2,
        "report_generated_at": _iso_utc_now(),
        "paths": {
            "db_path": db_path,
            "profiling_db": profiling_db,
            "state_dir": str(state_dir),
        },
        "graph_predictor": _safe_report_section(
            "graph_predictor",
            _graph_predictor_metrics_report,
            state_dir,
            db_path,
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
    registry_path = state_dir / _MODEL_REGISTRY_PATH.name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    registry = _build_model_registry(report)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    logger.info("Predictor metrics report written to %s", report_path)
    logger.info("Model registry written to %s", registry_path)
    return {
        "report_path": str(report_path),
        "model_registry_path": str(registry_path),
    }


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
