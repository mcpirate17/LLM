from __future__ import annotations

from typing import Any, Dict

import numpy as np


def safe_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    if np.unique(y_true).size < 2:
        return 0.0
    try:
        from sklearn.metrics import roc_auc_score

        score = float(roc_auc_score(y_true, y_score))
        return score if np.isfinite(score) else 0.0
    except Exception:
        return 0.0


def safe_binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision score (area under precision-recall curve)."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score

        score = float(average_precision_score(y_true, y_score))
        return score if np.isfinite(score) else 0.0
    except Exception:
        return 0.0


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score (mean squared error of probability estimates). Lower is better."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return 1.0
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — weighted average gap between predicted and observed rates."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return 1.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (
            (y_score > lo) & (y_score <= hi)
            if lo > 0
            else (y_score >= lo) & (y_score <= hi)
        )
        if mask.sum() == 0:
            continue
        bin_frac = mask.sum() / len(y_true)
        bin_acc = y_true[mask].mean()
        bin_conf = y_score[mask].mean()
        ece += bin_frac * abs(bin_acc - bin_conf)
    return float(ece)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score > threshold).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    accuracy = float((tp + tn) / max(len(y_true), 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    roc_auc = safe_binary_roc_auc(y_true, y_score)
    pr_auc = safe_binary_pr_auc(y_true, y_score)
    brier = brier_score(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score)

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "negatives": int(len(y_true) - np.sum(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_ppv": precision,
        "recall_tpr_sensitivity": recall,
        "specificity_tnr": specificity,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "prevalence": float(np.mean(y_true)) if len(y_true) else 0.0,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
    }


def operating_point_profiles(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    high_precision_floor: float = 0.7,
    high_recall_floor: float = 0.9,
    max_thresholds: int = 512,
) -> Dict[str, Dict[str, Any]]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) == 0:
        base = binary_classification_metrics(y_true, y_score, threshold=0.5)
        return {"f1": base, "youden": base, "high_precision": base, "high_recall": base}

    thresholds = np.unique(np.clip(y_score, 0.0, 1.0))
    if len(thresholds) > max_thresholds:
        quantiles = np.linspace(0.0, 1.0, max_thresholds)
        thresholds = np.unique(np.quantile(thresholds, quantiles))
    thresholds = np.unique(
        np.concatenate(([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0], thresholds))
    )

    metrics = [
        binary_classification_metrics(y_true, y_score, threshold=float(th))
        for th in thresholds
    ]

    def _best_f1() -> Dict[str, Any]:
        return max(
            metrics,
            key=lambda m: (
                m["f1"],
                m["balanced_accuracy"],
                m["precision_ppv"],
                -m["threshold"],
            ),
        )

    def _best_youden() -> Dict[str, Any]:
        return max(
            metrics,
            key=lambda m: (
                m["recall_tpr_sensitivity"] - m["fpr"],
                m["balanced_accuracy"],
                m["f1"],
                -m["threshold"],
            ),
        )

    def _best_high_precision() -> Dict[str, Any]:
        eligible = [m for m in metrics if m["precision_ppv"] >= high_precision_floor]
        if eligible:
            return max(
                eligible,
                key=lambda m: (
                    m["recall_tpr_sensitivity"],
                    m["precision_ppv"],
                    m["f1"],
                    -m["threshold"],
                ),
            )
        return max(
            metrics,
            key=lambda m: (
                m["precision_ppv"],
                m["recall_tpr_sensitivity"],
                m["f1"],
                -m["threshold"],
            ),
        )

    def _best_high_recall() -> Dict[str, Any]:
        eligible = [
            m for m in metrics if m["recall_tpr_sensitivity"] >= high_recall_floor
        ]
        if eligible:
            return max(
                eligible,
                key=lambda m: (
                    m["precision_ppv"],
                    m["recall_tpr_sensitivity"],
                    m["f1"],
                    -m["threshold"],
                ),
            )
        return max(
            metrics,
            key=lambda m: (
                m["recall_tpr_sensitivity"],
                m["precision_ppv"],
                m["f1"],
                -m["threshold"],
            ),
        )

    return {
        "f1": _best_f1(),
        "youden": _best_youden(),
        "high_precision": _best_high_precision(),
        "high_recall": _best_high_recall(),
    }
