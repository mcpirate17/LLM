#!/usr/bin/env python
"""Audit individual cheap-probe predictor heads for NAS policy decisions.

This intentionally sits beside the blended runtime GBM.  It trains/report heads
one target at a time so NAS gates can say which probe/head accepted, ranked, or
rejected a graph.

Usage:
    python -m research.tools.audit_cheap_probe_predictors
    python -m research.tools.audit_cheap_probe_predictors --json-out research/reports/cheap_probe_heads.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.defaults import RUNS_DB
from research.scientist.intelligence.metrics_utils import binary_classification_metrics
from research.scientist.intelligence.ml_corpus import _graph_fingerprint
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.notebook.native_conn import NativeConnectionWrapper
from research.synthesis.graph_features import extract_graph_features_bundle
from research.tools.annotate_literature_attribution import classify_graph_family


CHEAP_FEATURE_COLUMNS: tuple[str, ...] = (
    "ar_gate_score",
    "nano_induction_nearest_max_accuracy",
    "language_control_s05_sentence_assoc_score",
    "language_control_s05_binding_score",
    "language_control_s10_sentence_assoc_score",
    "language_control_s10_binding_score",
)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    feature_mode: str
    target_columns: tuple[str, ...]
    threshold: float
    positive_when: str = "high"
    target_definition: str = ""
    target_mode: str = "max"
    model_kind: str = "regressor"


HEAD_SPECS: tuple[HeadSpec, ...] = (
    HeadSpec(
        name="predict_ar_gate_from_graph",
        feature_mode="graph",
        target_columns=("ar_gate_score",),
        threshold=0.95,
        target_definition="static graph -> real ar_gate_score; >=0.95 is a strong pass axis",
    ),
    HeadSpec(
        name="predict_nano_induction_nearest_from_graph",
        feature_mode="graph",
        target_columns=("nano_induction_nearest_max_accuracy",),
        threshold=0.50,
        target_definition="static graph -> nano_induction_nearest_max_accuracy; >=0.50 is strong positive evidence",
    ),
    HeadSpec(
        name="predict_nb05_binding_from_graph",
        feature_mode="graph",
        target_columns=("language_control_s05_binding_score",),
        threshold=0.95,
        target_definition="static graph -> nb0.5 binding score; split from sentence-assoc so binding failures are visible",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_nb05_sentence_assoc_from_graph",
        feature_mode="graph",
        target_columns=("language_control_s05_sentence_assoc_score",),
        threshold=0.95,
        target_definition="static graph -> nb0.5 sentence-assoc score; positive/rescue evidence, not a binding substitute",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_nb05_joint_from_graph",
        feature_mode="graph",
        target_columns=(
            "language_control_s05_binding_score",
            "language_control_s05_sentence_assoc_score",
        ),
        threshold=0.95,
        target_definition="static graph -> joint nb0.5 pass; both binding and sentence-assoc must clear threshold",
        target_mode="joint_min",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_nb10_binding_from_graph",
        feature_mode="graph",
        target_columns=("language_control_s10_binding_score",),
        threshold=0.95,
        target_definition="static graph -> nb1.0 binding score; avoids sentence-assoc saturation poisoning NPV",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_nb10_sentence_assoc_from_graph",
        feature_mode="graph",
        target_columns=("language_control_s10_sentence_assoc_score",),
        threshold=0.95,
        target_definition="static graph -> nb1.0 sentence-assoc score; positive/rescue evidence, not a binding substitute",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_nb10_joint_from_graph",
        feature_mode="graph",
        target_columns=(
            "language_control_s10_binding_score",
            "language_control_s10_sentence_assoc_score",
        ),
        threshold=0.95,
        target_definition="static graph -> joint nb1.0 pass; both binding and sentence-assoc must clear threshold",
        target_mode="joint_min",
        model_kind="classifier",
    ),
    HeadSpec(
        name="predict_large_induction_from_cheap",
        feature_mode="cheap",
        target_columns=("large_induction_intermediate_auc",),
        threshold=0.50,
        target_definition="cheap probes -> large-run induction_intermediate_auc",
    ),
    HeadSpec(
        name="predict_large_binding_from_cheap",
        feature_mode="cheap",
        target_columns=(
            "large_binding_screening_composite",
            "large_binding_screening_auc",
        ),
        threshold=0.50,
        target_definition="cheap probes -> max(large-run binding composite, binding auc)",
    ),
    HeadSpec(
        name="predict_large_blimp_from_cheap",
        feature_mode="cheap",
        target_columns=("large_blimp_overall_accuracy",),
        threshold=0.55,
        target_definition="cheap probes -> large-run BLiMP overall accuracy",
    ),
    HeadSpec(
        name="predict_ar_curriculum_from_cheap",
        feature_mode="cheap",
        target_columns=("large_ar_curriculum_auc_pair_final",),
        threshold=0.50,
        target_definition="cheap probes -> large-run ar_curriculum_auc_pair_final",
    ),
    HeadSpec(
        name="predict_failure_from_cheap",
        feature_mode="cheap",
        target_columns=("failure_target",),
        threshold=0.50,
        positive_when="high",
        target_definition="cheap probes -> no-go/failure class; 1 means no successful S1 observed",
    ),
)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _max_opt(left: Any, right: Any) -> float | None:
    lval = _finite_float(left)
    rval = _finite_float(right)
    if lval is None:
        return rval
    if rval is None:
        return lval
    return max(lval, rval)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}  # nosec B608
    except sqlite3.OperationalError:
        return set()


def _select_col(available: set[str], name: str) -> str:
    return f"pr.{name}" if name in available else f"NULL AS {name}"


def _audit_select_columns(available: set[str]) -> list[str]:
    wanted = [
        "result_id",
        "graph_json",
        "timestamp",
        "stage1_passed",
        "stage0_passed",
        "stage05_passed",
        "param_count",
        "graph_n_params_estimate",
        "train_budget_steps",
        "n_train_steps",
        "failure_op",
        "error_type",
        "ar_gate_score",
        "nano_induction_nearest_max_accuracy",
        "language_control_s05_sentence_assoc_score",
        "language_control_s05_binding_score",
        "language_control_s10_sentence_assoc_score",
        "language_control_s10_binding_score",
        "induction_intermediate_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "blimp_overall_accuracy",
        "ar_curriculum_auc_pair_final",
    ]
    return [_select_col(available, name) for name in wanted]


def _op_names(graph: Mapping[str, Any]) -> set[str]:
    nodes = graph.get("nodes") or {}
    if isinstance(nodes, Mapping):
        values = nodes.values()
    elif isinstance(nodes, list):
        values = nodes
    else:
        values = []
    return {
        str(node.get("op_name") or "")
        for node in values
        if isinstance(node, Mapping) and node.get("op_name")
    }


def _parse_graph_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _new_group(
    canonical: str, graph_json: str, graph: Mapping[str, Any]
) -> dict[str, Any]:
    ops = _op_names(graph)
    return {
        "canonical_fingerprint": canonical,
        "graph_json": graph_json,
        "family": classify_graph_family(ops),
        "latest_timestamp": 0.0,
        "n_rows": 0,
        "stage1_any_passed": False,
        "failure_target": None,
        **{col: None for col in CHEAP_FEATURE_COLUMNS},
        "large_induction_intermediate_auc": None,
        "large_binding_screening_auc": None,
        "large_binding_screening_composite": None,
        "large_blimp_overall_accuracy": None,
        "large_ar_curriculum_auc_pair_final": None,
    }


def _row_is_large_run(
    row: sqlite3.Row,
    *,
    large_min_params: int,
    large_min_steps: int,
) -> bool:
    params = _finite_float(row["param_count"])
    if params is None:
        params = _finite_float(row["graph_n_params_estimate"])
    steps = _finite_float(row["train_budget_steps"])
    if steps is None:
        steps = _finite_float(row["n_train_steps"])
    return (
        params is not None
        and params >= float(large_min_params)
        and steps is not None
        and steps >= float(large_min_steps)
    )


def load_audit_rows(
    db_path: str | Path = RUNS_DB,
    *,
    large_min_params: int = 100_000_000,
    large_min_steps: int = 20_000,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Load one row per graph fingerprint with cheap and large-run targets.

    Cheap probe targets are max-aggregated across trusted observations for the
    fingerprint. Large-run targets are max-aggregated only from rows meeting the
    supplied parameter and step floors.
    """

    conn = NativeConnectionWrapper(str(db_path), read_only=True)
    table = "program_results_compat"
    available = _table_columns(conn, table)
    if not available:
        table = "program_results"
        available = _table_columns(conn, table)
    if not available:
        return []

    cols = ",\n               ".join(_audit_select_columns(available))
    limit_sql = f" LIMIT {int(limit)}" if limit and limit > 0 else ""
    rows = conn.execute(
        f"""
        SELECT {cols}
        FROM {table} pr
        WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
          AND pr.graph_json <> '{{}}'
        ORDER BY COALESCE(pr.timestamp, 0.0) ASC{limit_sql}
        """  # nosec B608 - table/columns are internal and validated from PRAGMA
    ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            graph_json = resolve_graph_json_value(conn, str(db_path), row["graph_json"])
        except Exception:
            graph_json = row["graph_json"]
        graph = _parse_graph_json(graph_json)
        if graph is None:
            continue
        canonical = _graph_fingerprint(json.dumps(graph, sort_keys=True))
        group = grouped.get(canonical)
        if group is None:
            graph_text = json.dumps(graph, sort_keys=True, separators=(",", ":"))
            group = _new_group(canonical, graph_text, graph)
            grouped[canonical] = group

        group["n_rows"] += 1
        group["latest_timestamp"] = max(
            float(group["latest_timestamp"]), float(row["timestamp"] or 0.0)
        )
        stage1_passed = bool(row["stage1_passed"])
        group["stage1_any_passed"] = bool(group["stage1_any_passed"] or stage1_passed)
        if not stage1_passed:
            group["failure_target"] = 1.0
        elif group["failure_target"] is None:
            group["failure_target"] = 0.0

        for col in CHEAP_FEATURE_COLUMNS:
            group[col] = _max_opt(group.get(col), row[col])

        if _row_is_large_run(
            row,
            large_min_params=large_min_params,
            large_min_steps=large_min_steps,
        ):
            group["large_induction_intermediate_auc"] = _max_opt(
                group["large_induction_intermediate_auc"],
                row["induction_intermediate_auc"],
            )
            group["large_binding_screening_auc"] = _max_opt(
                group["large_binding_screening_auc"], row["binding_screening_auc"]
            )
            group["large_binding_screening_composite"] = _max_opt(
                group["large_binding_screening_composite"],
                row["binding_screening_composite"],
            )
            group["large_blimp_overall_accuracy"] = _max_opt(
                group["large_blimp_overall_accuracy"], row["blimp_overall_accuracy"]
            )
            group["large_ar_curriculum_auc_pair_final"] = _max_opt(
                group["large_ar_curriculum_auc_pair_final"],
                row["ar_curriculum_auc_pair_final"],
            )

    return sorted(grouped.values(), key=lambda item: str(item["canonical_fingerprint"]))


def _target_value(row: Mapping[str, Any], spec: HeadSpec) -> float | None:
    values = [_finite_float(row.get(col)) for col in spec.target_columns]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    if spec.target_mode == "joint_min":
        return min(finite) if len(finite) == len(spec.target_columns) else None
    if spec.positive_when == "low":
        return min(finite)
    return max(finite)


def _graph_features(row: Mapping[str, Any]) -> dict[str, float]:
    graph = _parse_graph_json(row.get("graph_json"))
    if graph is None:
        return {}
    feats, ops = extract_graph_features_bundle(graph)
    out = {str(key): float(value) for key, value in feats.items()}
    for op in ops:
        if op:
            key = f"op_{op}"
            out[key] = out.get(key, 0.0) + 1.0
    return out


def _cheap_features(row: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in CHEAP_FEATURE_COLUMNS:
        value = _finite_float(row.get(col))
        present = 1.0 if value is not None else 0.0
        out[col] = value if value is not None else float("nan")
        out[f"{col}__present"] = present
    return out


def _features_for(row: Mapping[str, Any], spec: HeadSpec) -> dict[str, float]:
    if spec.feature_mode == "graph":
        return _graph_features(row)
    if spec.feature_mode == "cheap":
        return _cheap_features(row)
    raise ValueError(f"unknown feature mode: {spec.feature_mode}")


def build_head_dataset(
    rows: Sequence[Mapping[str, Any]],
    spec: HeadSpec,
) -> tuple[list[dict[str, float]], np.ndarray, list[Mapping[str, Any]]]:
    feat_rows: list[dict[str, float]] = []
    targets: list[float] = []
    kept_rows: list[Mapping[str, Any]] = []
    for row in rows:
        target = _target_value(row, spec)
        if target is None:
            continue
        feats = _features_for(row, spec)
        if not feats:
            continue
        feat_rows.append(feats)
        targets.append(float(target))
        kept_rows.append(row)
    return feat_rows, np.asarray(targets, dtype=np.float64), kept_rows


def _materialize_matrix(
    feat_rows: Sequence[Mapping[str, float]],
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if not feat_rows:
        return np.zeros((0, 0), dtype=np.float64), []
    names = list(feature_names or sorted({key for feats in feat_rows for key in feats}))
    X = np.full((len(feat_rows), len(names)), np.nan, dtype=np.float64)
    col_idx = {name: idx for idx, name in enumerate(names)}
    for row_idx, feats in enumerate(feat_rows):
        for key, value in feats.items():
            idx = col_idx.get(key)
            if idx is not None:
                X[row_idx, idx] = float(value)
    return X, names


def _impute_train_val(
    X_train: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if X_train.size == 0:
        return X_train, X_val
    med = np.zeros(X_train.shape[1], dtype=np.float64)
    finite = np.isfinite(X_train)
    has_value = np.any(finite, axis=0)
    if np.any(has_value):
        med[has_value] = np.nanmedian(X_train[:, has_value], axis=0)
    train = np.where(np.isfinite(X_train), X_train, med)
    val = np.where(np.isfinite(X_val), X_val, med)
    return train, val


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr

        value = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        rx = np.argsort(np.argsort(y_true))
        ry = np.argsort(np.argsort(y_pred))
        value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else 0.0


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    target_threshold: float,
    model_kind: str,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    """Fit one small head and return validation predictions + importances."""

    X_train, X_val = _impute_train_val(X_train, X_val)
    if y_train.size < 4 or np.unique(y_train).size < 2:
        return (
            np.full(X_val.shape[0], float(np.mean(y_train)) if y_train.size else 0.0),
            np.zeros(X_train.shape[1]),
            "constant_mean",
            [],
        )

    if model_kind == "classifier":
        y_bin = (
            np.asarray(y_train, dtype=np.float64) >= float(target_threshold)
        ).astype(np.int32)
        if np.unique(y_bin).size < 2:
            rate = float(np.mean(y_bin)) if y_bin.size else 0.0
            return (
                np.full(X_val.shape[0], rate),
                np.zeros(X_train.shape[1]),
                "constant_rate",
                [],
            )
        return _fit_classifier_candidates(
            X_train,
            y_bin,
            X_val,
            (np.asarray(y_val, dtype=np.float64) >= float(target_threshold)).astype(
                np.int32
            ),
            seed=seed,
            n_estimators=n_estimators,
        )

    return _fit_regressor_candidates(
        X_train,
        y_train,
        X_val,
        y_val,
        target_threshold=target_threshold,
        seed=seed,
        n_estimators=n_estimators,
    )


def _fit_classifier_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    candidates: list[tuple[str, Any]] = []
    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        leaf = max(1, min(20, y_train.size // 100))
        candidates.extend(
            [
                (
                    "hist_gradient_boosting_classifier",
                    HistGradientBoostingClassifier(
                        max_iter=max(32, int(n_estimators)),
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        class_weight="balanced",
                        random_state=int(seed),
                    ),
                ),
                (
                    "extra_trees_classifier",
                    ExtraTreesClassifier(
                        n_estimators=max(32, int(n_estimators)),
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        random_state=int(seed),
                        n_jobs=1,
                    ),
                ),
                (
                    "random_forest_classifier",
                    RandomForestClassifier(
                        n_estimators=max(32, int(n_estimators)),
                        max_depth=10,
                        min_samples_leaf=leaf,
                        class_weight="balanced_subsample",
                        random_state=int(seed),
                        n_jobs=1,
                    ),
                ),
                (
                    "logistic_regression_balanced",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ),
            ]
        )
    except Exception:
        preds, imp = _fit_linear_fallback(X_train, y_train.astype(np.float64), X_val)
        return preds, imp, "linear_fallback", []

    return _select_candidate_model(
        candidates,
        X_train,
        y_train,
        X_val,
        y_val,
        decision_threshold=0.5,
        target_threshold=0.5,
    )


def _fit_regressor_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    target_threshold: float,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    candidates: list[tuple[str, Any]] = []
    try:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )

        leaf = max(1, min(20, y_train.size // 100))
        candidates.extend(
            [
                (
                    "random_forest_regressor",
                    RandomForestRegressor(
                        n_estimators=max(32, int(n_estimators)),
                        max_depth=6,
                        min_samples_leaf=leaf,
                        random_state=int(seed),
                        n_jobs=1,
                    ),
                ),
                (
                    "extra_trees_regressor",
                    ExtraTreesRegressor(
                        n_estimators=max(32, int(n_estimators)),
                        min_samples_leaf=leaf,
                        random_state=int(seed),
                        n_jobs=1,
                    ),
                ),
                (
                    "hist_gradient_boosting_regressor",
                    HistGradientBoostingRegressor(
                        max_iter=max(32, int(n_estimators)),
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    except Exception:
        preds, imp = _fit_linear_fallback(X_train, y_train, X_val)
        return preds, imp, "linear_fallback", []

    return _select_candidate_model(
        candidates,
        X_train,
        y_train,
        X_val,
        y_val,
        decision_threshold=target_threshold,
        target_threshold=target_threshold,
    )


def _select_candidate_model(
    candidates: Sequence[tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    decision_threshold: float,
    target_threshold: float,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    best_name = "linear_fallback"
    best_preds, best_importance = _fit_linear_fallback(X_train, y_train, X_val)
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    comparison: list[dict[str, Any]] = []
    for name, model in candidates:
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                preds = np.asarray(model.predict_proba(X_val)[:, 1], dtype=np.float64)
            else:
                preds = np.asarray(model.predict(X_val), dtype=np.float64)
            preds = np.clip(preds, 0.0, 1.0)
            y_val_bin = (
                np.asarray(y_val, dtype=np.float64) >= float(target_threshold)
            ).astype(np.int32)
            metrics = binary_classification_metrics(
                y_val_bin, preds, threshold=float(decision_threshold)
            )
            score = float(metrics["roc_auc"])
            tie_break = float(metrics["balanced_accuracy"])
            comparison.append(
                {
                    "model": name,
                    "selection_score": float(score),
                    "balanced_accuracy": tie_break,
                    "ppv": float(metrics["precision_ppv"]),
                    "npv": float(metrics["npv"]),
                    "accuracy": float(metrics["accuracy"]),
                    "roc_auc": float(metrics["roc_auc"]),
                    "prediction_mean": float(np.mean(preds)) if preds.size else 0.0,
                    "prediction_std": float(np.std(preds)) if preds.size else 0.0,
                }
            )
            combined_score = (score, tie_break, float(np.std(preds)))
            if combined_score > best_score:
                best_score = combined_score
                best_name = name
                best_preds = preds
                if hasattr(model, "feature_importances_"):
                    best_importance = np.asarray(
                        model.feature_importances_, dtype=np.float64
                    )
                else:
                    best_importance = _permutation_importance(model, X_val, preds)
        except Exception:
            comparison.append({"model": name, "error": "fit_failed"})
    comparison.sort(
        key=lambda item: (
            float(item.get("selection_score", -1.0)),
            float(item.get("balanced_accuracy", 0.0)),
        ),
        reverse=True,
    )
    return best_preds, best_importance, best_name, comparison


def _fit_linear_fallback(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    Xn = (X_train - mean) / std
    Xv = (X_val - mean) / std
    reg = 1e-3
    xtx = Xn.T @ Xn + reg * np.eye(Xn.shape[1])
    coef = np.linalg.pinv(xtx) @ Xn.T @ y_train
    bias = float(np.mean(y_train))
    preds = Xv @ coef + bias
    return np.clip(preds, 0.0, 1.0), np.abs(coef)


def _permutation_importance(
    model: Any,
    X_val: np.ndarray,
    baseline_preds: np.ndarray,
    *,
    max_features: int = 256,
) -> np.ndarray:
    """Cheap deterministic proxy importance for sklearn models without gain."""

    if X_val.size == 0:
        return np.zeros(0, dtype=np.float64)
    n_features = X_val.shape[1]
    out = np.zeros(n_features, dtype=np.float64)
    # Bound report cost for very wide op histograms.
    variances = np.var(X_val, axis=0)
    candidates = np.argsort(-variances)[: min(max_features, n_features)]
    rng = np.random.RandomState(17)
    baseline = np.asarray(baseline_preds, dtype=np.float64)
    for idx in candidates:
        Xp = X_val.copy()
        Xp[:, idx] = Xp[rng.permutation(Xp.shape[0]), idx]
        if hasattr(model, "predict_proba"):
            pred = np.asarray(model.predict_proba(Xp)[:, 1], dtype=np.float64)
        else:
            pred = np.asarray(model.predict(Xp), dtype=np.float64)
        out[idx] = float(np.mean(np.abs(pred - baseline)))
    return out


def _temporal_split(
    rows: Sequence[Mapping[str, Any]], train_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(len(rows)),
        key=lambda idx: (
            float(rows[idx].get("latest_timestamp") or 0.0),
            str(rows[idx].get("canonical_fingerprint") or idx),
        ),
    )
    if len(order) < 2:
        return np.asarray(order, dtype=np.int32), np.zeros(0, dtype=np.int32)
    split_at = int(math.floor(len(order) * float(train_fraction)))
    split_at = max(1, min(len(order) - 1, split_at))
    return (
        np.asarray(order[:split_at], dtype=np.int32),
        np.asarray(order[split_at:], dtype=np.int32),
    )


def _metric_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_bin = (y_true >= float(threshold)).astype(np.int32)
    decision = float(threshold if decision_threshold is None else decision_threshold)
    return {
        "n": int(y_true.size),
        "target_mean": float(np.mean(y_true)) if y_true.size else 0.0,
        "target_positive_count": int(np.sum(y_bin)),
        "target_prevalence": float(np.mean(y_bin)) if y_bin.size else 0.0,
        "spearman": _spearman(y_true, y_pred),
        "binary_at_threshold": binary_classification_metrics(
            y_bin, y_pred, threshold=decision
        ),
    }


def _calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    if y_true.size == 0:
        return []
    y_bin = (np.asarray(y_true, dtype=np.float64) >= float(threshold)).astype(
        np.float64
    )
    score = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (
            (score >= lo) & (score <= hi) if lo == 0.0 else (score > lo) & (score <= hi)
        )
        if not np.any(mask):
            continue
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(np.sum(mask)),
                "mean_pred": float(np.mean(score[mask])),
                "observed_positive_rate": float(np.mean(y_bin[mask])),
            }
        )
    return out


def _top_importances(
    names: Sequence[str], importances: np.ndarray, *, top_n: int
) -> list[dict[str, Any]]:
    if importances.size == 0:
        return []
    pairs = [
        (str(name), float(value))
        for name, value in zip(names, importances)
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": value} for name, value in pairs[:top_n]]


def _stability_flags(binary_metrics: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    positives = int(binary_metrics.get("positives") or 0)
    negatives = int(binary_metrics.get("negatives") or 0)
    predicted_positive = int(binary_metrics.get("tp") or 0) + int(
        binary_metrics.get("fp") or 0
    )
    predicted_negative = int(binary_metrics.get("tn") or 0) + int(
        binary_metrics.get("fn") or 0
    )
    if positives < 50:
        flags.append("unstable_ppv_few_actual_positives")
    if negatives < 50:
        flags.append("unstable_npv_few_actual_negatives")
    if predicted_positive < 50:
        flags.append("unstable_ppv_few_predicted_positives")
    if predicted_negative < 50:
        flags.append("unstable_npv_few_predicted_negatives")
    return flags


def _band_key(row: Mapping[str, Any]) -> str:
    def val(name: str) -> float:
        value = _finite_float(row.get(name))
        return value if value is not None else -1.0

    weak: list[str] = []
    if val("ar_gate_score") < 0.95:
        weak.append("ar")
    if val("nano_induction_nearest_max_accuracy") < 0.20:
        weak.append("nano")
    if val("language_control_s05_binding_score") < 0.80:
        weak.append("s05_binding")
    if val("language_control_s05_sentence_assoc_score") < 0.30:
        weak.append("s05_sentence")
    return "+".join(weak) if weak else "no_weak_cheap_band"


def _stratified_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    target_threshold: float,
    decision_threshold: float,
    min_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    y_bin = (np.asarray(y_true, dtype=np.float64) >= float(target_threshold)).astype(
        np.int32
    )
    score = np.asarray(y_pred, dtype=np.float64)

    def collect(key_fn) -> list[dict[str, Any]]:
        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            buckets[str(key_fn(row))].append(idx)
        out: list[dict[str, Any]] = []
        for key, idxs in buckets.items():
            if len(idxs) < min_n:
                continue
            idx = np.asarray(idxs, dtype=np.int32)
            metrics = binary_classification_metrics(
                y_bin[idx], score[idx], threshold=float(decision_threshold)
            )
            out.append(
                {
                    "stratum": key,
                    "n": int(idx.size),
                    "positives": int(metrics["positives"]),
                    "negatives": int(metrics["negatives"]),
                    "ppv": float(metrics["precision_ppv"]),
                    "npv": float(metrics["npv"]),
                    "accuracy": float(metrics["accuracy"]),
                    "recall": float(metrics["recall_tpr_sensitivity"]),
                    "predicted_negative": int(metrics["tn"] + metrics["fn"]),
                    "stability_flags": _stability_flags(metrics),
                }
            )
        out.sort(key=lambda item: (-int(item["n"]), str(item["stratum"])))
        return out

    return {
        "family": collect(lambda row: row.get("family") or "unknown"),
        "cheap_signal_band": collect(_band_key),
    }


def _operating_point_profiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_threshold: float,
    min_support: int = 50,
) -> dict[str, dict[str, Any]]:
    y_bin = (np.asarray(y_true, dtype=np.float64) >= float(target_threshold)).astype(
        np.int32
    )
    score = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    if score.size == 0:
        base = binary_classification_metrics(y_bin, score, threshold=0.5)
        return {name: dict(base) for name in ("balanced", "f1", "high_ppv", "high_npv")}
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray(
                    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
                ),
                np.quantile(score, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    profiles: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(
            y_bin, score, threshold=float(threshold)
        )
        metrics["predicted_positive"] = int(metrics["tp"] + metrics["fp"])
        metrics["predicted_negative"] = int(metrics["tn"] + metrics["fn"])
        metrics["stability_flags"] = _stability_flags(metrics)
        profiles.append(metrics)

    def eligible_ppv(item: Mapping[str, Any]) -> bool:
        return int(item.get("predicted_positive") or 0) >= min_support

    def eligible_npv(item: Mapping[str, Any]) -> bool:
        return int(item.get("predicted_negative") or 0) >= min_support

    def best(key_fn, eligible_fn=lambda _item: True) -> dict[str, Any]:
        pool = [item for item in profiles if eligible_fn(item)] or profiles
        return dict(max(pool, key=key_fn))

    return {
        "balanced": best(
            lambda item: (
                float(item["balanced_accuracy"]),
                float(item["roc_auc"]),
                float(item["accuracy"]),
            )
        ),
        "f1": best(
            lambda item: (
                float(item["f1"]),
                float(item["balanced_accuracy"]),
                float(item["accuracy"]),
            )
        ),
        "high_ppv": best(
            lambda item: (
                float(item["precision_ppv"]),
                float(item["recall_tpr_sensitivity"]),
                float(item["accuracy"]),
            ),
            eligible_ppv,
        ),
        "high_npv": best(
            lambda item: (
                float(item["npv"]),
                float(item["specificity_tnr"]),
                float(item["accuracy"]),
            ),
            eligible_npv,
        ),
    }


def _evaluate_split(
    feat_rows: Sequence[Mapping[str, float]],
    y: np.ndarray,
    source_rows: Sequence[Mapping[str, Any]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    spec: HeadSpec,
    *,
    seed: int,
    n_estimators: int,
    top_features: int,
) -> dict[str, Any]:
    if train_idx.size == 0 or val_idx.size == 0:
        return {
            "error": "empty_split",
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        }
    X_all, names = _materialize_matrix(feat_rows)
    preds, importances, selected_model, model_comparison = _fit_predict(
        X_all[train_idx],
        y[train_idx],
        X_all[val_idx],
        y[val_idx],
        target_threshold=spec.threshold,
        model_kind=spec.model_kind,
        seed=seed,
        n_estimators=n_estimators,
    )
    decision_threshold = 0.5 if spec.model_kind == "classifier" else spec.threshold
    metrics = _metric_payload(
        y[val_idx],
        preds,
        threshold=spec.threshold,
        decision_threshold=decision_threshold,
    )
    stability = _stability_flags(metrics["binary_at_threshold"])
    operating_points = _operating_point_profiles(
        y[val_idx],
        preds,
        target_threshold=spec.threshold,
    )
    metrics.update(
        {
            "model_kind": spec.model_kind,
            "selected_model": selected_model,
            "model_comparison": model_comparison,
            "decision_threshold": float(decision_threshold),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "feature_count": int(len(names)),
            "stability_flags": stability,
            "operating_points": operating_points,
            "top_feature_importances": _top_importances(
                names, importances, top_n=top_features
            ),
            "calibration_bins": _calibration_bins(
                y[val_idx], preds, threshold=spec.threshold
            ),
            "stratified_diagnostics": _stratified_diagnostics(
                y[val_idx],
                preds,
                [source_rows[idx] for idx in val_idx],
                target_threshold=spec.threshold,
                decision_threshold=decision_threshold,
            ),
            "validation_fingerprints": [
                str(source_rows[idx].get("canonical_fingerprint") or "")
                for idx in val_idx[:20]
            ],
        }
    )
    return metrics


def _leave_family_out(
    feat_rows: Sequence[Mapping[str, float]],
    y: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    spec: HeadSpec,
    *,
    min_family_holdout: int,
    seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    families = sorted({str(row.get("family") or "unknown") for row in rows})
    payloads: list[dict[str, Any]] = []
    X_all, _names = _materialize_matrix(feat_rows)
    for family in families:
        val_idx = np.asarray(
            [
                idx
                for idx, row in enumerate(rows)
                if str(row.get("family") or "unknown") == family
            ],
            dtype=np.int32,
        )
        if val_idx.size < min_family_holdout:
            continue
        train_idx = np.asarray(
            [
                idx
                for idx, row in enumerate(rows)
                if str(row.get("family") or "unknown") != family
            ],
            dtype=np.int32,
        )
        if train_idx.size < max(4, min_family_holdout):
            continue
        preds, _imp, _model_name, _comparison = _fit_predict(
            X_all[train_idx],
            y[train_idx],
            X_all[val_idx],
            y[val_idx],
            target_threshold=spec.threshold,
            model_kind=spec.model_kind,
            seed=seed,
            n_estimators=n_estimators,
        )
        decision_threshold = 0.5 if spec.model_kind == "classifier" else spec.threshold
        metrics = _metric_payload(
            y[val_idx],
            preds,
            threshold=spec.threshold,
            decision_threshold=decision_threshold,
        )
        payloads.append(
            {
                "family": family,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "spearman": float(metrics["spearman"]),
                "roc_auc": float(metrics["binary_at_threshold"]["roc_auc"]),
                "precision": float(metrics["binary_at_threshold"]["precision_ppv"]),
                "recall": float(
                    metrics["binary_at_threshold"]["recall_tpr_sensitivity"]
                ),
            }
        )
    if not payloads:
        return {"families_evaluated": 0, "error": "insufficient_family_holdouts"}
    return {
        "families_evaluated": len(payloads),
        "mean_spearman": float(np.mean([p["spearman"] for p in payloads])),
        "mean_roc_auc": float(np.mean([p["roc_auc"] for p in payloads])),
        "worst_family_by_spearman": min(payloads, key=lambda p: p["spearman"]),
        "families": payloads,
    }


def audit_heads(
    rows: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[HeadSpec] = HEAD_SPECS,
    min_samples: int = 30,
    min_eval: int = 8,
    train_fraction: float = 0.8,
    min_family_holdout: int = 8,
    seed: int = 42,
    n_estimators: int = 96,
    top_features: int = 12,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        feat_rows, y, kept_rows = build_head_dataset(rows, spec)
        y_bin = (
            (y >= spec.threshold).astype(np.int32)
            if y.size
            else np.zeros(0, dtype=np.int32)
        )
        head_result: dict[str, Any] = {
            "head": spec.name,
            "feature_mode": spec.feature_mode,
            "model_kind": spec.model_kind,
            "target_mode": spec.target_mode,
            "target_columns": list(spec.target_columns),
            "target_definition": spec.target_definition,
            "threshold": float(spec.threshold),
            "sample_count": int(y.size),
            "positive_count": int(np.sum(y_bin)),
            "prevalence": float(np.mean(y_bin)) if y_bin.size else 0.0,
        }
        if y.size < min_samples or np.unique(y).size < 2:
            head_result["error"] = "insufficient_samples_or_target_variance"
            results.append(head_result)
            continue
        train_idx, val_idx = _temporal_split(kept_rows, train_fraction)
        if val_idx.size < min_eval:
            head_result["error"] = "insufficient_temporal_eval_rows"
            head_result["n_temporal_val"] = int(val_idx.size)
            results.append(head_result)
            continue
        head_result["temporal_holdout"] = _evaluate_split(
            feat_rows,
            y,
            kept_rows,
            train_idx,
            val_idx,
            spec,
            seed=seed,
            n_estimators=n_estimators,
            top_features=top_features,
        )
        head_result["leave_family_out"] = _leave_family_out(
            feat_rows,
            y,
            kept_rows,
            spec,
            min_family_holdout=min_family_holdout,
            seed=seed,
            n_estimators=n_estimators,
        )
        results.append(head_result)
    return {
        "audit_version": "cheap_probe_heads_v3_model_threshold_search",
        "row_count": int(len(rows)),
        "heads": results,
    }


def _fmt_float(value: Any, digits: int = 3) -> str:
    val = _finite_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def format_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cheap-Probe Predictor Head Audit",
        "",
        f"Rows: {int(report.get('row_count') or 0)}",
        f"Audit version: `{report.get('audit_version')}`",
        "",
        "| head | model | n | pos | threshold | PPV | NPV | accuracy | ROC AUC | flags | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for head in report.get("heads", []):
        temporal = head.get("temporal_holdout") or {}
        binary = temporal.get("binary_at_threshold") or {}
        status = head.get("error") or "ok"
        flags = ",".join(temporal.get("stability_flags") or [])
        lines.append(
            "| {head} | {model} | {n} | {pos} | {thr} | {ppv} | {npv} | {acc} | {auc} | {flags} | {status} |".format(
                head=head.get("head"),
                model=(temporal.get("selected_model") or head.get("model_kind")),
                n=head.get("sample_count"),
                pos=head.get("positive_count"),
                thr=_fmt_float(head.get("threshold"), 2),
                ppv=_fmt_float(binary.get("precision_ppv")),
                npv=_fmt_float(binary.get("npv")),
                acc=_fmt_float(binary.get("accuracy")),
                auc=_fmt_float(binary.get("roc_auc")),
                flags=flags or "",
                status=status,
            )
        )
    for head in report.get("heads", []):
        lines.extend(
            ["", f"## {head.get('head')}", "", str(head.get("target_definition") or "")]
        )
        temporal = head.get("temporal_holdout") or {}
        binary = temporal.get("binary_at_threshold") or {}
        if head.get("error"):
            lines.append(f"- status: `{head['error']}`")
            continue
        lines.extend(
            [
                f"- samples: {head.get('sample_count')} total, {head.get('positive_count')} positive at threshold {_fmt_float(head.get('threshold'), 2)}",
                f"- temporal holdout: model={temporal.get('selected_model') or temporal.get('model_kind')} decision_threshold={_fmt_float(temporal.get('decision_threshold'), 2)} n_train={temporal.get('n_train')} n_val={temporal.get('n_val')} spearman={_fmt_float(temporal.get('spearman'))} roc_auc={_fmt_float(binary.get('roc_auc'))}",
                f"- PPV/NPV/accuracy: ppv={_fmt_float(binary.get('precision_ppv'))} npv={_fmt_float(binary.get('npv'))} accuracy={_fmt_float(binary.get('accuracy'))} recall={_fmt_float(binary.get('recall_tpr_sensitivity'))} tp={binary.get('tp')} fp={binary.get('fp')} tn={binary.get('tn')} fn={binary.get('fn')}",
            ]
        )
        comparisons = temporal.get("model_comparison") or []
        if comparisons:
            lines.append(
                "- model candidates: "
                + "; ".join(
                    f"{item.get('model')} auc={_fmt_float(item.get('roc_auc'))} bal={_fmt_float(item.get('balanced_accuracy'))}"
                    for item in comparisons[:4]
                    if not item.get("error")
                )
            )
        operating = temporal.get("operating_points") or {}
        if operating:
            parts = []
            for name in ("balanced", "f1", "high_ppv", "high_npv"):
                item = operating.get(name) or {}
                parts.append(
                    f"{name}@{_fmt_float(item.get('threshold'), 2)} ppv={_fmt_float(item.get('precision_ppv'))} npv={_fmt_float(item.get('npv'))} acc={_fmt_float(item.get('accuracy'))}"
                )
            lines.append("- operating points: " + "; ".join(parts))
        flags = temporal.get("stability_flags") or []
        if flags:
            lines.append("- stability flags: " + ", ".join(flags))
        top = temporal.get("top_feature_importances") or []
        if top:
            lines.append(
                "- top features: "
                + ", ".join(
                    f"{item['feature']}={_fmt_float(item['importance'])}"
                    for item in top[:8]
                )
            )
        strata = temporal.get("stratified_diagnostics") or {}
        for key in ("family", "cheap_signal_band"):
            items = strata.get(key) or []
            if items:
                preview = "; ".join(
                    f"{item['stratum']} n={item['n']} ppv={_fmt_float(item['ppv'])} npv={_fmt_float(item['npv'])}"
                    for item in items[:5]
                )
                lines.append(f"- top {key} strata: {preview}")
        bins = temporal.get("calibration_bins") or []
        if bins:
            lines.append(
                "- calibration bins: "
                + "; ".join(
                    f"{_fmt_float(b['lo'], 1)}-{_fmt_float(b['hi'], 1)} n={b['n']} pred={_fmt_float(b['mean_pred'])} obs={_fmt_float(b['observed_positive_rate'])}"
                    for b in bins[:10]
                )
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--large-min-params", type=int, default=100_000_000)
    parser.add_argument("--large-min-steps", type=int, default=20_000)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-eval", type=int, default=8)
    parser.add_argument("--min-family-holdout", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=96)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    rows = load_audit_rows(
        args.db,
        large_min_params=args.large_min_params,
        large_min_steps=args.large_min_steps,
        limit=args.limit,
    )
    report = audit_heads(
        rows,
        min_samples=args.min_samples,
        min_eval=args.min_eval,
        min_family_holdout=args.min_family_holdout,
        n_estimators=args.n_estimators,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    markdown = format_markdown(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
