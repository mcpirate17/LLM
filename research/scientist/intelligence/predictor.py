"""Performance predictors: Ridge (fingerprint) + LightGBM (graph-structure).

Ridge regression on 18D fingerprint feature vector — trained in-memory from
historical notebook data. Used for investigation-stage ranking.

GBMPredictor (LightGBM) operates on graph-structure features available BEFORE
eval, enabling cheap rejection of hopeless graphs at screening time:
  - gate_model: P(pass_s1 | graph_features) — skip if P < 0.1
  - rank_model: predicted wikitext_perplexity — prioritize promising graphs
"""

from __future__ import annotations

import functools
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STATE_DIR = Path("research/runtime/learning")
_GBM_GATE_MODEL_PATH = _STATE_DIR / "gbm_gate_model.txt"
_GBM_RANK_MODEL_PATH = _STATE_DIR / "gbm_rank_model.txt"
_GBM_META_PATH = _STATE_DIR / "gbm_predictor.json"
_GRAPH_PREDICTOR_PATH = _STATE_DIR / "graph_predictor.npz"
_INTERACTION_MODEL_PATH = _STATE_DIR / "interaction_model.npz"
_BAYESIAN_STATE_PATH = _STATE_DIR / "bayesian_state.json"
_ENSEMBLE_STATE_PATH = _STATE_DIR / "ensemble_state.npz"
_ENSEMBLE_META_PATH = _STATE_DIR / "ensemble_state.json"

# 16 fingerprint features + 2 novelty features = 18D
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

# Sample weights by tier: investigation/validation ran more training steps,
# so their loss_ratio is a more reliable target signal.
_TIER_WEIGHT = {
    "screening": 1.0,
    "screened_out": 0.5,
    "investigation": 4.0,
    "investigation_failed": 2.0,
    "investigation_fingerprint_incomplete": 2.0,
    "validation": 6.0,
    "breakthrough": 6.0,
}


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
        rows = nb.conn.execute(
            """
            SELECT pr.fingerprint_json, pr.novelty_score,
                   pr.structural_novelty, pr.loss_ratio,
                   l.investigation_loss_ratio, l.tier
            FROM program_results pr
            JOIN leaderboard l ON l.result_id = pr.result_id
            WHERE pr.fingerprint_json IS NOT NULL
              AND pr.loss_ratio IS NOT NULL
            ORDER BY pr.timestamp ASC
            """
        ).fetchall()
    except Exception as e:
        logger.warning("Predictor training data query failed: %s", e)
        return np.zeros((0, 18)), np.zeros(0), np.zeros(0)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    w_list: List[float] = []
    n_inv = 0

    for row in rows:
        fp_json = row[0] if not isinstance(row, dict) else row["fingerprint_json"]
        novelty = row[1] if not isinstance(row, dict) else row["novelty_score"]
        struct_nov = row[2] if not isinstance(row, dict) else row["structural_novelty"]
        loss_ratio = row[3] if not isinstance(row, dict) else row["loss_ratio"]
        inv_lr = (
            row[4] if not isinstance(row, dict) else row["investigation_loss_ratio"]
        )
        tier = row[5] if not isinstance(row, dict) else row["tier"]

        feats = _extract_features(fp_json, novelty, struct_nov)
        if feats is None:
            continue

        # Use investigation_loss_ratio when available (more training steps),
        # fall back to screening loss_ratio
        target = inv_lr if inv_lr is not None else loss_ratio
        lr = float(target)
        if not np.isfinite(lr):
            continue

        tier_str = str(tier) if tier else "screening"
        weight = _TIER_WEIGHT.get(tier_str, 1.0)

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

    x_norm = (feats - model.feature_mean) / model.feature_std
    return float(x_norm @ model.weights + model.bias)


def evaluate(nb, alpha: float = 1.0) -> Dict:
    """Hold-out evaluation: train on first 80%, test on last 20%.

    Returns dict with spearman_rho, n_train, n_test, mean_error.
    """
    X, y, sample_weights = _query_training_data(nb)

    if len(X) < 15:
        return {"error": "insufficient_data", "n_total": len(X)}

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


# ─────────────────────────────────────────────────────────────────────────────
# GBMPredictor: LightGBM graph-structure pre-screener
# ─────────────────────────────────────────────────────────────────────────────

_MIN_GBM_SAMPLES = 50  # minimum rows to train
_RETRAIN_INTERVAL = 50  # retrain every N new experiments


def _query_graph_training_data(
    db_path: str,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Query graph_json + labels from ALL program_results for GBM training.

    Uses every graph — including failures without final_loss (they're known
    negatives). This gives the gate model a much clearer picture of what
    "hopeless" looks like vs only training on graphs that survived long enough
    to produce a loss.

    Returns (feature_dicts, y_gate, y_rank) where:
      - feature_dicts: list of dicts from extract_graph_features (with full op histogram)
      - y_gate: binary array (1 = passed S1)
      - y_rank: float array of wikitext_perplexity (NaN where unavailable)
    """
    from ...synthesis.graph_features import (
        extract_graph_features,
        enrich_with_op_stats,
        load_op_stats,
    )

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        rows = conn.execute(
            """SELECT graph_json, stage1_passed, wikitext_perplexity
               FROM program_results
               WHERE graph_json IS NOT NULL
               ORDER BY timestamp ASC"""
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning("GBM training data query failed: %s", e)
        return [], np.zeros(0), np.zeros(0)

    # Load op_stats ONCE for all rows (avoids N+1 DB queries)
    op_stats_cache = load_op_stats(db_path)

    feat_dicts: List[Dict[str, float]] = []
    gate_labels: List[int] = []
    rank_labels: List[float] = []

    for row in rows:
        gj = row["graph_json"]
        if not gj:
            continue
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        feats = extract_graph_features(gj_dict)
        if not feats:
            continue
        # Add per-op counts for ALL ops in this graph (full op histogram)
        nodes = gj_dict.get("nodes") or {}
        ops = [
            n.get("op_name", "")
            for n in nodes.values()
            if n.get("op_name", "") != "input"
        ]
        for op in ops:
            if op:
                feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
        enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
        feat_dicts.append(feats)
        gate_labels.append(1 if row["stage1_passed"] else 0)
        ppl = row["wikitext_perplexity"]
        rank_labels.append(float(ppl) if ppl is not None else float("nan"))

    return (
        feat_dicts,
        np.array(gate_labels, dtype=np.int32),
        np.array(rank_labels, dtype=np.float64),
    )


def _dicts_to_matrix(
    feat_dicts: List[Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    """Convert list of feature dicts to (n, d) matrix + ordered column names."""
    if not feat_dicts:
        return np.zeros((0, 0)), []
    all_keys = sorted(feat_dicts[0].keys())
    # Union of all keys across dicts (handles missing keys)
    key_set = set()
    for d in feat_dicts:
        key_set.update(d.keys())
    all_keys = sorted(key_set)

    X = np.zeros((len(feat_dicts), len(all_keys)), dtype=np.float32)
    for i, d in enumerate(feat_dicts):
        for j, k in enumerate(all_keys):
            X[i, j] = d.get(k, 0.0)
    return X, all_keys


@dataclass(slots=True)
class GBMPredictor:
    """LightGBM-based graph pre-screener.

    Two models:
      gate_model: classifier — P(pass_s1 | graph_features)
      rank_model: regressor — predicted wikitext_perplexity

    Both operate on graph-structure features only (no forward pass needed).
    """

    gate_model: Any = None  # lgb.Booster
    rank_model: Any = None  # lgb.Booster
    feature_names: List[str] = field(default_factory=list)
    n_train: int = 0
    gate_importance: Optional[Dict[str, float]] = None
    rank_importance: Optional[Dict[str, float]] = None

    def is_fitted(self) -> bool:
        return self.gate_model is not None and self.n_train > 0

    def predict_gate(self, features: Dict[str, float]) -> float:
        """Predict P(pass_s1) for a single graph. Returns 0.5 if not fitted."""
        if not self.is_fitted():
            return 0.5
        x = np.array(
            [[features.get(k, 0.0) for k in self.feature_names]], dtype=np.float32
        )
        try:
            return float(self.gate_model.predict(x)[0])
        except (TypeError, ValueError):
            return 0.5

    def predict_rank(self, features: Dict[str, float]) -> float:
        """Predict wikitext_perplexity for a single graph. Returns 1e6 if not fitted."""
        if self.rank_model is None:
            return 1e6
        x = np.array(
            [[features.get(k, 0.0) for k in self.feature_names]], dtype=np.float32
        )
        try:
            return float(self.rank_model.predict(x)[0])
        except (TypeError, ValueError):
            return 1e6

    def save(self, state_dir: Path) -> None:
        """Persist LightGBM models plus feature metadata."""
        if not self.is_fitted():
            return
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.gate_model.save_model(str(state_dir / _GBM_GATE_MODEL_PATH.name))
        if self.rank_model is not None:
            self.rank_model.save_model(str(state_dir / _GBM_RANK_MODEL_PATH.name))
        with open(state_dir / _GBM_META_PATH.name, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "n_train": self.n_train,
                    "gate_importance": self.gate_importance,
                    "rank_importance": self.rank_importance,
                    "has_rank_model": self.rank_model is not None,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, state_dir: Path) -> "GBMPredictor":
        """Load persisted LightGBM models and metadata from disk."""
        state_dir = Path(state_dir)
        meta_path = state_dir / _GBM_META_PATH.name
        gate_model_path = state_dir / _GBM_GATE_MODEL_PATH.name
        if not meta_path.exists() or not gate_model_path.exists():
            return cls()
        try:
            import lightgbm as lgb
        except ImportError:
            logger.info("lightgbm not installed, persisted GBM predictor unavailable")
            return cls()
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        gate_model = lgb.Booster(model_file=str(gate_model_path))
        rank_model = None
        if meta.get("has_rank_model"):
            rank_model_path = state_dir / _GBM_RANK_MODEL_PATH.name
            if rank_model_path.exists():
                rank_model = lgb.Booster(model_file=str(rank_model_path))
        return cls(
            gate_model=gate_model,
            rank_model=rank_model,
            feature_names=list(meta.get("feature_names", [])),
            n_train=int(meta.get("n_train", 0)),
            gate_importance=meta.get("gate_importance"),
            rank_importance=meta.get("rank_importance"),
        )


def train_gbm(
    db_path: str = "research/lab_notebook.db",
) -> GBMPredictor:
    """Train GBM gate + rank models from notebook history.

    Returns a fitted GBMPredictor. Falls back gracefully if lightgbm
    is unavailable or insufficient data.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.info("lightgbm not installed, GBM predictor unavailable")
        return GBMPredictor()

    feat_dicts, y_gate, y_rank = _query_graph_training_data(db_path)

    if len(feat_dicts) < _MIN_GBM_SAMPLES:
        logger.info(
            "GBM predictor: insufficient data (%d < %d), skipping",
            len(feat_dicts),
            _MIN_GBM_SAMPLES,
        )
        return GBMPredictor()

    X, feature_names = _dicts_to_matrix(feat_dicts)
    n_total = len(X)

    # Random stratified 80/20 split (NOT temporal — distribution shifts over time
    # as grammar rules evolve, causing massive train/test divergence with temporal splits)
    n_pos = int(y_gate.sum())
    n_neg = n_total - n_pos
    if n_pos < 5 or n_neg < 5:
        logger.info(
            "GBM predictor: insufficient class balance (pos=%d, neg=%d)", n_pos, n_neg
        )
        return GBMPredictor()

    rng = np.random.RandomState(42)
    pos_idx = np.where(y_gate == 1)[0]
    neg_idx = np.where(y_gate == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_split = int(len(pos_idx) * 0.8)
    neg_split = int(len(neg_idx) * 0.8)
    train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
    val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train, X_val = X[train_idx], X[val_idx]
    y_gate_train, y_gate_val = y_gate[train_idx], y_gate[val_idx]
    y_rank_train_full, y_rank_val_full = y_rank[train_idx], y_rank[val_idx]

    # Class imbalance handling
    pos_count = int(y_gate_train.sum())
    neg_count = len(y_gate_train) - pos_count
    spw = neg_count / max(pos_count, 1)

    # ── Gate model (binary classifier) ──
    # Strong regularization to prevent overfitting: feature/bagging subsampling,
    # high min_data_in_leaf, lambda_l1/l2, and generous early stopping patience.
    gate_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "scale_pos_weight": spw,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": 1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X_train, label=y_gate_train, feature_name=feature_names)
    val_set = lgb.Dataset(
        X_val, label=y_gate_val, feature_name=feature_names, reference=train_set
    )

    gate_model = lgb.train(
        gate_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    gate_importance = dict(
        zip(feature_names, gate_model.feature_importance("gain").tolist())
    )

    # ── Rank model (regression on wikitext_perplexity) ──
    rank_model = None
    rank_importance = None
    rank_mask_tr = np.isfinite(y_rank_train_full)
    rank_mask_va = np.isfinite(y_rank_val_full)
    if rank_mask_tr.sum() >= 30 and rank_mask_va.sum() >= 5:
        X_rank_train = X_train[rank_mask_tr]
        y_rank_train = y_rank_train_full[rank_mask_tr]
        X_rank_val = X_val[rank_mask_va]
        y_rank_val = y_rank_val_full[rank_mask_va]

        rank_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_jobs": 1,
            "seed": 42,
        }
        r_train = lgb.Dataset(
            X_rank_train, label=y_rank_train, feature_name=feature_names
        )
        r_val = lgb.Dataset(
            X_rank_val, label=y_rank_val, feature_name=feature_names, reference=r_train
        )
        rank_model = lgb.train(
            rank_params,
            r_train,
            num_boost_round=500,
            valid_sets=[r_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        rank_importance = dict(
            zip(feature_names, rank_model.feature_importance("gain").tolist())
        )

    n_trained = len(X_train)
    predictor = GBMPredictor(
        gate_model=gate_model,
        rank_model=rank_model,
        feature_names=feature_names,
        n_train=n_trained,
        gate_importance=gate_importance,
        rank_importance=rank_importance,
    )

    logger.info(
        "GBM predictor trained: %d samples, %d features, gate_spw=%.1f, rank=%s",
        n_trained,
        len(feature_names),
        spw,
        "yes" if rank_model else "no",
    )
    return predictor


def evaluate_gbm(
    db_path: str = "research/lab_notebook.db",
) -> Dict[str, Any]:
    """Train + evaluate GBM predictor with hold-out metrics.

    Returns dict with gate_auc, rank_spearman, skip_rate, n_train, n_test.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm_not_installed"}

    feat_dicts, y_gate, y_rank = _query_graph_training_data(db_path)
    n_total = len(feat_dicts)
    if n_total < _MIN_GBM_SAMPLES:
        return {"error": "insufficient_data", "n_total": n_total}

    X, feature_names = _dicts_to_matrix(feat_dicts)

    # Random stratified split (matches train_gbm)
    n_pos = int(y_gate.sum())
    n_neg = n_total - n_pos
    if n_pos < 5 or n_neg < 5:
        return {"error": "insufficient_balance", "n_pos": n_pos, "n_neg": n_neg}

    rng = np.random.RandomState(42)
    pos_idx = np.where(y_gate == 1)[0]
    neg_idx = np.where(y_gate == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_split = int(len(pos_idx) * 0.8)
    neg_split = int(len(neg_idx) * 0.8)
    train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
    val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])

    X_train, X_test = X[train_idx], X[val_idx]
    y_gate_train, y_gate_test = y_gate[train_idx], y_gate[val_idx]
    y_rank_train_full, y_rank_test_full = y_rank[train_idx], y_rank[val_idx]
    n_train = len(X_train)
    n_test = len(X_test)

    pos_count = int(y_gate_train.sum())
    neg_count = n_train - pos_count
    spw = neg_count / max(pos_count, 1)

    gate_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "scale_pos_weight": spw,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": 1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X_train, label=y_gate_train, feature_name=feature_names)
    val_set = lgb.Dataset(
        X_test, label=y_gate_test, feature_name=feature_names, reference=train_set
    )

    gate_model = lgb.train(
        gate_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Gate AUC
    from sklearn.metrics import roc_auc_score

    gate_preds = gate_model.predict(X_test)
    try:
        gate_auc = float(roc_auc_score(y_gate_test, gate_preds))
    except ValueError:
        gate_auc = 0.0

    # Skip rate at P < 0.1
    skip_mask = gate_preds < 0.1
    skip_rate = float(skip_mask.mean())
    if skip_mask.sum() > 0:
        false_skip_rate = float(y_gate_test[skip_mask].mean())
    else:
        false_skip_rate = 0.0

    # Rank Spearman on wikitext_perplexity subset
    rank_spearman = 0.0
    rank_mask_tr = np.isfinite(y_rank_train_full)
    rank_mask_te = np.isfinite(y_rank_test_full)
    if rank_mask_tr.sum() >= 30 and rank_mask_te.sum() >= 10:
        rank_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_jobs": 1,
            "seed": 42,
        }
        r_train = lgb.Dataset(
            X_train[rank_mask_tr],
            label=y_rank_train_full[rank_mask_tr],
            feature_name=feature_names,
        )
        r_val = lgb.Dataset(
            X_test[rank_mask_te],
            label=y_rank_test_full[rank_mask_te],
            feature_name=feature_names,
            reference=r_train,
        )
        rank_model = lgb.train(
            rank_params,
            r_train,
            num_boost_round=500,
            valid_sets=[r_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        rank_preds = rank_model.predict(X_test[rank_mask_te])
        from scipy.stats import spearmanr

        rho, _ = spearmanr(y_rank_test_full[rank_mask_te], rank_preds)
        rank_spearman = float(rho) if np.isfinite(rho) else 0.0

    # Top feature importances
    importance = dict(
        zip(feature_names, gate_model.feature_importance("gain").tolist())
    )
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    return {
        "gate_auc": gate_auc,
        "rank_spearman": rank_spearman,
        "skip_rate": skip_rate,
        "false_skip_rate": false_skip_rate,
        "n_train": n_train,
        "n_test": n_test,
        "n_positive": int(y_gate.sum()),
        "n_total": n_total,
        "top_features": top_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EnsemblePredictor: combines GBM + GraphPredictor + Bayesian + InteractionModel
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EnsemblePredictor:
    """Meta-learner combining all prediction components.

    Combines predictions from:
    1. GBMPredictor (existing LightGBM gate/rank)
    2. GraphPredictor (topology-aware features)
    3. TemporalBayesianTracker (op-level Bayesian posteriors)
    4. InteractionModel (pairwise stability/loss)

    Uses logistic regression on component outputs + uncertainty-driven
    exploration bonus. Gracefully degrades when components are unavailable.
    """

    gbm: Optional[GBMPredictor] = None
    graph_pred: Optional[Any] = None  # GraphPredictor
    bayesian: Optional[Any] = None  # TemporalBayesianTracker
    interaction: Optional[Any] = None  # InteractionModel
    # Learned meta-learner weights (calibrated from held-out data)
    w_ensemble: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_ensemble: float = 0.0
    _score_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _score_std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _n_components: int = 0
    _n_score_dims: int = 0  # number of score dimensions the weights were trained on

    def is_fitted(self) -> bool:
        # Fitted if at least one component is available
        return any(
            [
                self.gbm is not None and self.gbm.is_fitted(),
                self.graph_pred is not None and self.graph_pred.is_fitted(),
            ]
        )

    def predict_gate(
        self,
        graph_json: Any = None,
        graph_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Predict P(pass_s1) by combining all available components.

        Args:
            graph_json: Raw graph JSON (for GraphPredictor + InteractionModel).
            graph_features: Pre-extracted flat features (for GBM).

        Returns P(pass_s1) in [0, 1].
        """
        scores = []
        uncertainties = []

        # GBM prediction
        if self.gbm is not None and self.gbm.is_fitted() and graph_features is not None:
            p = self.gbm.predict_gate(graph_features)
            scores.append(p)
            uncertainties.append(0.1)  # GBM has fixed uncertainty estimate

        # Topology predictor
        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
        ):
            p = self.graph_pred.predict_gate(graph_json)
            scores.append(p)
            uncertainties.append(0.15)

        # Bayesian worst-op score
        if self.bayesian is not None and graph_json is not None:
            ops = self._extract_ops(graph_json)
            if ops:
                op_weights = self.bayesian.op_weights(mode="mean")
                worst = min(op_weights.get(op, 0.5) for op in ops)
                # Map weight [0.1, 8.0] → probability [0, 1]
                bayes_p = np.clip(worst / 3.0, 0.0, 1.0)
                scores.append(float(bayes_p))
                # Uncertainty from posterior variance
                variances = []
                for op in ops:
                    if op in self.bayesian.op_posteriors:
                        variances.append(self.bayesian.op_posteriors[op].std)
                uncertainties.append(float(np.mean(variances)) if variances else 0.2)

        # Interaction model: mean pair stability
        if (
            self.interaction is not None
            and self.interaction._trained
            and graph_json is not None
        ):
            ops = self._extract_ops(graph_json)
            if len(ops) >= 2:
                stabilities = []
                for a in ops:
                    for b in ops:
                        if a != b:
                            stabilities.append(self.interaction.predict_stability(a, b))
                if stabilities:
                    scores.append(float(np.mean(stabilities)))
                    uncertainties.append(0.15)

        if not scores:
            return 0.5

        # Learned blend weights (calibrated logistic regression on component scores)
        if self.w_ensemble.size > 0 and self._n_score_dims > 0:
            x = np.zeros(self._n_score_dims, dtype=np.float64)
            for i, s in enumerate(scores[: self._n_score_dims]):
                x[i] = s
            # Standardize using training stats
            if self._score_mean.size == self._n_score_dims:
                x = (x - self._score_mean) / np.maximum(self._score_std, 1e-8)
            logit = float(x @ self.w_ensemble + self.b_ensemble)
            final = float(1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10))))
        else:
            # Fallback: simple average (no exploration bonus — exploration
            # happens at grammar level via Thompson sampling, not screening gate)
            final = float(np.mean(scores))

        return float(np.clip(final, 0.01, 0.99))

    def predict_rank(
        self,
        graph_json: Any = None,
        graph_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Predict ranking score (lower = better). Combines available components."""
        ranks = []

        if self.gbm is not None and self.gbm.is_fitted() and graph_features is not None:
            r = self.gbm.predict_rank(graph_features)
            if r < 1e5:
                ranks.append(r)

        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
        ):
            r = self.graph_pred.predict_rank(graph_json)
            if r < 1e5:
                ranks.append(r)

        if not ranks:
            return 1e6

        return float(np.mean(ranks))

    def _extract_ops(self, graph_json: Any) -> List[str]:
        """Extract op names from graph JSON."""
        if isinstance(graph_json, str):
            try:
                graph_json = json.loads(graph_json)
            except (json.JSONDecodeError, TypeError):
                return []
        if not isinstance(graph_json, dict):
            return []
        nodes = graph_json.get("nodes") or {}
        return [
            n.get("op_name", "")
            for n in nodes.values()
            if n.get("op_name", "") and n.get("op_name") != "input"
        ]

    def diagnostics(self) -> Dict:
        """Return diagnostic info about ensemble state."""
        return {
            "gbm_fitted": self.gbm is not None and self.gbm.is_fitted(),
            "graph_pred_fitted": self.graph_pred is not None
            and self.graph_pred.is_fitted(),
            "bayesian_loaded": self.bayesian is not None,
            "interaction_fitted": self.interaction is not None
            and self.interaction._trained,
            "n_components": sum(
                [
                    self.gbm is not None and self.gbm.is_fitted(),
                    self.graph_pred is not None and self.graph_pred.is_fitted(),
                    self.bayesian is not None,
                    self.interaction is not None and self.interaction._trained,
                ]
            ),
            "calibrated": self.w_ensemble.size > 0,
            "n_score_dims": self._n_score_dims,
        }

    def save(self, state_dir: Path) -> None:
        """Persist ensemble calibration plus any fitted submodels."""
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if self.gbm is not None and self.gbm.is_fitted():
            self.gbm.save(state_dir)
        if self.graph_pred is not None and self.graph_pred.is_fitted():
            self.graph_pred.save(state_dir / _GRAPH_PREDICTOR_PATH.name)
        if self.bayesian is not None:
            self.bayesian.save_state(state_dir / _BAYESIAN_STATE_PATH.name)
        if self.interaction is not None and self.interaction._trained:
            self.interaction.save(state_dir / _INTERACTION_MODEL_PATH.name)
        np.savez_compressed(
            str(state_dir / _ENSEMBLE_STATE_PATH.name),
            w_ensemble=self.w_ensemble,
            score_mean=self._score_mean,
            score_std=self._score_std,
        )
        with open(state_dir / _ENSEMBLE_META_PATH.name, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "b_ensemble": self.b_ensemble,
                    "n_score_dims": self._n_score_dims,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        state_dir: Path = _STATE_DIR,
        profiling_db: str = "research/profiling/component_profiles.db",
    ) -> "EnsemblePredictor":
        """Load persisted ensemble state without retraining."""
        state_dir = Path(state_dir)
        ensemble = cls()

        gbm = GBMPredictor.load(state_dir)
        if gbm.is_fitted():
            ensemble.gbm = gbm

        try:
            from .gnn_predictor import GraphPredictor

            graph_path = state_dir / _GRAPH_PREDICTOR_PATH.name
            if graph_path.exists():
                ensemble.graph_pred = GraphPredictor.load(
                    graph_path, profiling_db=Path(profiling_db)
                )
        except Exception as exc:
            logger.debug("GraphPredictor load skipped: %s", exc)

        try:
            from .temporal_bayesian import TemporalBayesianTracker

            bayes_path = state_dir / _BAYESIAN_STATE_PATH.name
            if bayes_path.exists():
                ensemble.bayesian = TemporalBayesianTracker.load_state(bayes_path)
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Bayesian state load skipped: %s", exc)

        try:
            from .interaction_model import InteractionModel

            interaction_path = state_dir / _INTERACTION_MODEL_PATH.name
            if interaction_path.exists():
                ensemble.interaction = InteractionModel.load(interaction_path)
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Interaction model load skipped: %s", exc)

        ensemble_state_path = state_dir / _ENSEMBLE_STATE_PATH.name
        ensemble_meta_path = state_dir / _ENSEMBLE_META_PATH.name
        if ensemble_state_path.exists() and ensemble_meta_path.exists():
            try:
                data = np.load(str(ensemble_state_path))
                with open(ensemble_meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                ensemble.w_ensemble = data["w_ensemble"]
                ensemble._score_mean = data["score_mean"]
                ensemble._score_std = data["score_std"]
                ensemble.b_ensemble = float(meta.get("b_ensemble", 0.0))
                ensemble._n_score_dims = int(meta.get("n_score_dims", 0))
            except Exception as exc:
                logger.debug("Ensemble calibration load skipped: %s", exc)

        return ensemble


@functools.lru_cache(maxsize=4)
def load_runtime_ensemble(
    state_dir: str = str(_STATE_DIR),
    profiling_db: str = "research/profiling/component_profiles.db",
) -> EnsemblePredictor:
    """Load persisted ensemble state for runtime use.

    This path is intentionally load-only. Training belongs in offline tooling.
    """
    return EnsemblePredictor.load(
        state_dir=Path(state_dir),
        profiling_db=profiling_db,
    )


def train_ensemble(
    db_path: str = "research/lab_notebook.db",
    profiling_db: str = "research/profiling/component_profiles.db",
) -> EnsemblePredictor:
    """Train the full ensemble predictor from notebook + profiling data.

    Trains each component independently, then combines them. Gracefully
    degrades if any component fails to train.
    """
    from pathlib import Path

    notebook_path = Path(db_path)
    profiling_path = Path(profiling_db)

    ensemble = EnsemblePredictor()

    # 1. GBM (existing)
    try:
        ensemble.gbm = train_gbm(db_path)
        if ensemble.gbm.is_fitted():
            logger.info("Ensemble: GBM trained (%d samples)", ensemble.gbm.n_train)
    except Exception as e:
        logger.warning("Ensemble: GBM training failed: %s", e)

    # 2. Topology predictor
    try:
        from .gnn_predictor import GraphPredictor

        ensemble.graph_pred = GraphPredictor.train(notebook_path, profiling_path)
        if ensemble.graph_pred.is_fitted():
            logger.info(
                "Ensemble: GraphPredictor trained (acc=%.3f)",
                ensemble.graph_pred._train_metrics.get("val_accuracy", 0),
            )
    except Exception as e:
        logger.warning("Ensemble: GraphPredictor training failed: %s", e)

    # 3. Bayesian tracker
    try:
        from .temporal_bayesian import TemporalBayesianTracker

        ensemble.bayesian = TemporalBayesianTracker.from_db(notebook_path)
        n_ops = len(ensemble.bayesian.op_posteriors)
        logger.info("Ensemble: Bayesian tracker loaded (%d ops)", n_ops)
    except Exception as e:
        logger.warning("Ensemble: Bayesian tracker failed: %s", e)

    # 4. Interaction model
    try:
        from .interaction_model import InteractionModel

        ensemble.interaction = InteractionModel.train(
            notebook_path, profiling_path, n_epochs=50
        )
        if ensemble.interaction._trained:
            logger.info(
                "Ensemble: InteractionModel trained (%d ops)",
                ensemble.interaction.n_ops,
            )
    except Exception as e:
        logger.warning("Ensemble: InteractionModel training failed: %s", e)

    # 5. Calibrate blend weights from held-out data
    try:
        _calibrate_ensemble(ensemble, db_path)
    except Exception as e:
        logger.warning("Ensemble calibration failed (using fallback averaging): %s", e)

    logger.info("Ensemble ready: %s", ensemble.diagnostics())
    return ensemble


def _calibrate_ensemble(
    ensemble: EnsemblePredictor,
    db_path: str,
    n_samples: int = 2000,
    n_epochs: int = 60,
    lr: float = 0.01,
) -> None:
    """Calibrate ensemble blend weights from held-out program_results.

    Fits a logistic regression on component scores → actual S1 labels.
    This learns which components to trust more and the optimal threshold.
    """
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    rows = conn.execute(
        """SELECT graph_json, stage1_passed
           FROM program_results
           WHERE graph_json IS NOT NULL AND stage0_passed = 1
           ORDER BY RANDOM() LIMIT ?""",
        (n_samples,),
    ).fetchall()
    conn.close()

    if len(rows) < 100:
        return

    # Collect component scores for each graph
    from ...synthesis.graph_features import (
        extract_graph_features,
        enrich_with_op_stats,
        load_op_stats,
    )

    op_stats_cache = load_op_stats(db_path)
    score_rows: list = []
    labels: list = []

    for gj, s1 in rows:
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue

        scores = []

        # GBM score
        if ensemble.gbm is not None and ensemble.gbm.is_fitted():
            feats = extract_graph_features(gj_dict)
            if feats:
                nodes = gj_dict.get("nodes") or {}
                ops = [
                    n.get("op_name", "")
                    for n in nodes.values()
                    if n.get("op_name", "") != "input"
                ]
                enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
                scores.append(ensemble.gbm.predict_gate(feats))
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # GraphPredictor score
        if ensemble.graph_pred is not None and ensemble.graph_pred.is_fitted():
            scores.append(ensemble.graph_pred.predict_gate(gj_dict))
        else:
            scores.append(0.5)

        # Bayesian worst-op score
        if ensemble.bayesian is not None:
            nodes = gj_dict.get("nodes") or {}
            ops = [
                n.get("op_name", "")
                for n in nodes.values()
                if n.get("op_name", "") and n.get("op_name") != "input"
            ]
            if ops:
                op_weights = ensemble.bayesian.op_weights(mode="mean")
                worst = min(op_weights.get(op, 0.5) for op in ops)
                scores.append(float(np.clip(worst / 3.0, 0.0, 1.0)))
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # Interaction model score
        if ensemble.interaction is not None and ensemble.interaction._trained:
            nodes = gj_dict.get("nodes") or {}
            ops = [
                n.get("op_name", "")
                for n in nodes.values()
                if n.get("op_name", "") and n.get("op_name") != "input"
            ]
            if len(ops) >= 2:
                stabs = [
                    ensemble.interaction.predict_stability(a, b)
                    for a in ops
                    for b in ops
                    if a != b
                ]
                scores.append(float(np.mean(stabs)) if stabs else 0.5)
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        score_rows.append(scores)
        labels.append(int(s1 or 0))

    if len(score_rows) < 100:
        return

    X = np.array(score_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    n_dims = X.shape[1]

    # Standardize scores (components have different scales/ranges)
    score_mean = X.mean(axis=0)
    score_std = X.std(axis=0)
    score_std[score_std < 1e-8] = 1.0
    X_norm = (X - score_mean) / score_std

    # Stratified split
    rng = np.random.RandomState(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    split_p = int(len(pos_idx) * 0.8)
    split_n = int(len(neg_idx) * 0.8)
    train_idx = np.concatenate([pos_idx[:split_p], neg_idx[:split_n]])
    val_idx = np.concatenate([pos_idx[split_p:], neg_idx[split_n:]])

    X_tr, X_va = X_norm[train_idx], X_norm[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    # Logistic regression via SGD
    w = rng.randn(n_dims).astype(np.float64) * 0.01
    b = 0.0

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    for _ in range(n_epochs):
        perm = rng.permutation(len(X_tr))
        for start in range(0, len(X_tr), 128):
            idx = perm[start : start + 128]
            x_b = X_tr[idx]
            y_b = y_tr[idx]
            preds = _sig(x_b @ w + b)
            grad = (preds - y_b)[:, None] * x_b
            w -= lr * grad.mean(axis=0) + lr * 0.01 * w
            b -= lr * float((preds - y_b).mean())

    # Validate
    val_preds = _sig(X_va @ w + b)
    val_correct = int(np.sum((val_preds > 0.5) == y_va))
    val_acc = val_correct / max(len(X_va), 1)

    ensemble.w_ensemble = w.astype(np.float32)
    ensemble.b_ensemble = float(b)
    ensemble._score_mean = score_mean.astype(np.float32)
    ensemble._score_std = score_std.astype(np.float32)
    ensemble._n_score_dims = n_dims

    logger.info(
        "Ensemble calibrated: %d-dim logistic regression, val_acc=%.3f, "
        "weights=[%s], bias=%.3f (%d train, %d val)",
        n_dims,
        val_acc,
        ", ".join(f"{wi:.3f}" for wi in w),
        b,
        len(X_tr),
        len(X_va),
    )
