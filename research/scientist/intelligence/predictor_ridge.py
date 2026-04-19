"""Performance predictor split module. Re-exported via predictor."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics_utils import (
    binary_classification_metrics,
    operating_point_profiles,
    safe_binary_roc_auc,
)
from .ml_corpus import (
    CorpusIntegrityError,
    build_dense_feature_matrix,
    grouped_stratified_split,
    load_deduped_screening_predictor_rows,
    load_deduped_graph_training_rows,
    load_deduped_predictor_training_rows,
    rerun_confidence_weight,
)

logger = logging.getLogger(__name__)

_TIER_WEIGHT = {
    "screening": 1.0,
    "screened_out": 0.5,
    "investigation": 4.0,
    "investigation_failed": 2.0,
    "investigation_fingerprint_incomplete": 2.0,
    "validation": 6.0,
    "breakthrough": 6.0,
}

_FINGERPRINT_KEYS = [
    "interaction_locality",
    "interaction_sparsity",
    "interaction_symmetry",
    "interaction_hierarchy",
    "intrinsic_dim",
    "isotropy",
    "rank_ratio",
    "jacobian_spectral_norm",
    "jacobian_effective_rank",
    "sensitivity_uniformity",
    "cka_vs_transformer",
    "cka_vs_ssm",
    "cka_vs_conv",
    "hierarchy_fitness",
    "routing_selectivity",
    "routing_compute_ratio",
]

_STATE_DIR = Path("research/runtime/learning")
_GBM_GATE_MODEL_PATH = _STATE_DIR / "gbm_gate_model.txt"
_GBM_RANK_MODEL_PATH = _STATE_DIR / "gbm_rank_model.txt"
_GBM_META_PATH = _STATE_DIR / "gbm_predictor.json"
_GRAPH_PREDICTOR_PATH = _STATE_DIR / "graph_predictor.npz"
_INTERACTION_MODEL_PATH = _STATE_DIR / "interaction_model.npz"
_BAYESIAN_STATE_PATH = _STATE_DIR / "bayesian_state.json"
_ENSEMBLE_STATE_PATH = _STATE_DIR / "ensemble_state.npz"
_ENSEMBLE_META_PATH = _STATE_DIR / "ensemble_state.json"


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _extract_features(
    fingerprint_json, novelty_score: float, structural_novelty: float
) -> Optional[np.ndarray]:
    """Extract 18D feature vector from fingerprint JSON + novelty scores.

    Returns None if fingerprint_json is unparseable.
    """
    try:
        fp = (
            json.loads(fingerprint_json)
            if isinstance(fingerprint_json, str)
            else fingerprint_json
        )
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(fp, dict):
        return None

    features = []
    for key in _FINGERPRINT_KEYS:
        val = fp.get(key)
        features.append(float(val) if val is not None else 0.0)

    features.append(float(novelty_score) if novelty_score is not None else 0.0)
    features.append(
        float(structural_novelty) if structural_novelty is not None else 0.0
    )

    return np.array(features, dtype=np.float64)


@dataclass
class PerformancePredictor:
    """Weighted Ridge regression model for predicting loss_ratio."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_clip_lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_clip_hi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_train: int = 0
    n_investigation: int = 0  # how many investigation+ samples contributed

    def is_fitted(self) -> bool:
        return self.n_train > 0 and self.weights.size > 0


def _query_training_data(nb) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query historical data across all tiers with per-tier sample weights.

    Returns (X, y, w) where X is (n, 18) features, y is (n,) loss_ratios,
    and w is (n,) sample weights reflecting tier reliability.

    Investigation/validation entries use investigation_loss_ratio when available,
    falling back to loss_ratio. Screening entries use loss_ratio (short training).
    """
    try:
        rows = load_deduped_predictor_training_rows(nb.db_path, validate=True)
    except CorpusIntegrityError:
        raise
    except Exception as e:
        logger.warning("Predictor training data query failed: %s", e)
        return np.zeros((0, 18)), np.zeros(0), np.zeros(0)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    w_list: List[float] = []
    n_inv = 0

    for row in rows:
        fp_json = row["fingerprint_json"]
        novelty = row["novelty_score"]
        struct_nov = row["structural_novelty"]
        target = row["target_loss_ratio"]
        tier = row["tier"]
        rerun_weight = rerun_confidence_weight(int(row.get("n_rows", 1)))

        feats = _extract_features(fp_json, novelty, struct_nov)
        if feats is None:
            continue

        lr = float(target)
        if not np.isfinite(lr):
            continue
        # target_loss_ratio should be in [0, ~1.5] — values above 2.0
        # indicate corrupted rows (e.g. raw perplexity leaking into the
        # loss_ratio column via backfill scripts).  Filter them out so
        # they don't dominate the Ridge fit.
        if lr < 0.0 or lr > 2.0:
            continue

        tier_str = str(tier) if tier else "screening"
        weight = _TIER_WEIGHT.get(tier_str, 1.0) * rerun_weight

        X_list.append(feats)
        y_list.append(lr)
        w_list.append(weight)

        if tier_str in (
            "investigation",
            "investigation_failed",
            "investigation_fingerprint_incomplete",
            "validation",
            "breakthrough",
        ):
            n_inv += 1

    if not X_list:
        return np.zeros((0, 18)), np.zeros(0), np.zeros(0)

    return np.array(X_list), np.array(y_list), np.array(w_list)


def train(nb, alpha: float = 1.0) -> PerformancePredictor:
    """Train a weighted Ridge regression predictor from notebook history.

    Uses weighted normal equations: w = (X^T W X + alpha*I)^{-1} X^T W y
    where W is a diagonal matrix of per-sample weights (tier-based).

    Returns a fitted PerformancePredictor (in-memory, no pickle).
    """
    X, y, sample_weights = _query_training_data(nb)

    if len(X) < 10:
        logger.info(
            "Predictor: insufficient data (%d samples), skipping training", len(X)
        )
        return PerformancePredictor()

    # Count investigation-tier contributions
    n_inv = int(np.sum(sample_weights > 1.5))

    # Winsorise features at the 1st/99th percentile to prevent extreme
    # outliers (e.g. jacobian_spectral_norm ~55k) from dominating the
    # Ridge fit and causing wild extrapolation on unseen data.
    p01 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)
    X = np.clip(X, p01, p99)

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    X_norm = (X - mean) / std

    # Weighted Ridge regression via normal equations:
    # w = (X^T W X + alpha*I)^{-1} X^T W y
    n_features = X_norm.shape[1]
    W = sample_weights  # (n,) vector — applied via broadcasting
    XtWX = (X_norm * W[:, None]).T @ X_norm + alpha * np.eye(n_features)
    XtWy = (X_norm * W[:, None]).T @ y
    try:
        weights = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        logger.warning("Predictor: singular matrix in normal equations")
        return PerformancePredictor()

    # Bias: weighted mean of residuals
    predictions = X_norm @ weights
    weighted_residual = np.sum(W * (y - predictions)) / np.sum(W)
    bias = float(weighted_residual)

    model = PerformancePredictor(
        weights=weights,
        bias=bias,
        feature_mean=mean,
        feature_std=std,
        feature_clip_lo=p01,
        feature_clip_hi=p99,
        n_train=len(X),
        n_investigation=n_inv,
    )

    logger.info(
        "Predictor trained: %d samples (%d investigation+), %d features, bias=%.4f",
        len(X),
        n_inv,
        n_features,
        bias,
    )
    return model


def predict(
    model: PerformancePredictor,
    fingerprint_dict: dict,
    novelty_score: float = 0.0,
    structural_novelty: float = 0.0,
) -> float:
    """Predict loss_ratio for a candidate.

    Returns predicted loss_ratio (lower = better). Returns 1.0 if model
    is not fitted or features can't be extracted.
    """
    if not model.is_fitted():
        return 1.0

    feats = _extract_features(fingerprint_dict, novelty_score, structural_novelty)
    if feats is None:
        return 1.0

    if model.feature_clip_lo.size == feats.size:
        feats = np.clip(feats, model.feature_clip_lo, model.feature_clip_hi)
    x_norm = (feats - model.feature_mean) / model.feature_std
    return float(x_norm @ model.weights + model.bias)


def evaluate(nb, alpha: float = 1.0) -> Dict:
    """Hold-out evaluation: train on first 80%, test on last 20%.

    Returns dict with spearman_rho, n_train, n_test, mean_error.
    """
    X, y, sample_weights = _query_training_data(nb)

    if len(X) < 15:
        return {"error": "insufficient_data", "n_total": len(X)}

    # Winsorise before split (same as train()) to prevent extreme
    # outliers from dominating the Ridge fit.
    p01 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)
    X = np.clip(X, p01, p99)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = sample_weights[:split]

    # Standardize using train stats
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Weighted fit
    n_features = X_train_norm.shape[1]
    W = w_train
    XtWX = (X_train_norm * W[:, None]).T @ X_train_norm + alpha * np.eye(n_features)
    XtWy = (X_train_norm * W[:, None]).T @ y_train
    try:
        weights = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return {"error": "singular_matrix", "n_total": len(X)}

    preds_train = X_train_norm @ weights
    bias = float(np.sum(W * (y_train - preds_train)) / np.sum(W))

    # Predict on test
    preds_test = X_test_norm @ weights + bias

    # Spearman rank correlation
    from scipy.stats import spearmanr

    rho, p_value = spearmanr(y_test, preds_test)

    mean_error = float(np.mean(np.abs(y_test - preds_test)))
    n_inv = int(np.sum(sample_weights > 1.5))

    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "spearman_p": float(p_value) if np.isfinite(p_value) else 1.0,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_investigation": n_inv,
        "mean_absolute_error": mean_error,
    }


