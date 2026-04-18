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
        return


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


# ─────────────────────────────────────────────────────────────────────────────
# GBMPredictor: LightGBM graph-structure pre-screener
# ─────────────────────────────────────────────────────────────────────────────

_MIN_GBM_SAMPLES = 50  # minimum rows to train


def _graph_signature(graph_json: Any) -> Optional[str]:
    """Return a stable hash for exact-graph grouping."""
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(graph_json, dict):
        return None
    try:
        canonical = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _load_screening_predictor_corpus_rows(
    db_path: str,
    *,
    validate: bool,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper that preserves local monkeypatch points."""
    db_file = Path(db_path)
    if not db_file.exists():
        return load_deduped_graph_training_rows(db_path, validate=validate)
    try:
        from ..notebook.shared_conn import get_notebook_conn
        conn = get_notebook_conn(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(program_results)")}
    except sqlite3.Error:
        return load_deduped_graph_training_rows(db_path, validate=validate)
    required = {
        "result_cohort",
        "data_provenance_json",
        "trust_label",
        "comparability_label",
    }
    if not required.issubset(cols):
        return load_deduped_graph_training_rows(db_path, validate=validate)
    return load_deduped_screening_predictor_rows(db_path, validate=validate)


def analyze_graph_label_quality(db_path: str) -> Dict[str, Any]:
    """Summarize duplicate-graph ambiguity and hard-negative composition."""
    try:
        from ..notebook.shared_conn import get_notebook_conn
        conn = get_notebook_conn(db_path)
        rows = conn.execute(
            """SELECT graph_json, stage1_passed, stage0_passed, stage05_passed
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        ).fetchall()
    except Exception as e:
        return {"error": f"label_quality_query_failed: {e}"}

    group_labels: Dict[str, List[int]] = defaultdict(list)
    n_rows = 0
    n_pos = 0
    n_fail_s05 = 0
    n_fail_pre_s0 = 0
    for row in rows:
        gj = row["graph_json"]
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        signature = _graph_signature(gj_dict)
        if signature is None:
            continue
        s1 = int(row["stage1_passed"] or 0)
        s0 = bool(row["stage0_passed"])
        s05 = bool(row["stage05_passed"])
        n_rows += 1
        n_pos += s1
        if not s1:
            if s05:
                n_fail_s05 += 1
            elif not s0:
                n_fail_pre_s0 += 1
        group_labels[signature].append(s1)

    duplicate_groups = [vals for vals in group_labels.values() if len(vals) > 1]
    ambiguous_groups = [
        vals for vals in duplicate_groups if 0.0 < float(np.mean(vals)) < 1.0
    ]
    ambiguous_rows = int(sum(len(vals) for vals in ambiguous_groups))
    ambiguity_rates = [float(np.mean(vals)) for vals in ambiguous_groups]

    return {
        "n_rows": n_rows,
        "n_unique_graphs": len(group_labels),
        "n_duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": int(sum(len(vals) for vals in duplicate_groups)),
        "n_ambiguous_duplicate_groups": len(ambiguous_groups),
        "rows_in_ambiguous_duplicate_groups": ambiguous_rows,
        "ambiguous_row_fraction": (float(ambiguous_rows / n_rows) if n_rows else 0.0),
        "ambiguous_group_mean_s1_rate": (
            float(np.mean(ambiguity_rates)) if ambiguity_rates else 0.0
        ),
        "n_positive": n_pos,
        "n_fail_s05": n_fail_s05,
        "n_fail_pre_s0": n_fail_pre_s0,
        "fail_s05_fraction_of_negatives": (float(n_fail_s05 / max(n_rows - n_pos, 1))),
    }


def _query_graph_training_data(
    db_path: str,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Query graph_json + labels from ALL program_results for GBM training.

    Uses every graph — including failures without final_loss (they're known
    negatives). This gives the gate model a much clearer picture of what
    "hopeless" looks like vs only training on graphs that survived long enough
    to produce a loss.

    Returns (feature_dicts, y_gate, y_rank) where:
      - feature_dicts: list of dicts from extract_graph_features (with full op histogram
        + probe features: hellaswag_acc, induction_auc, ar_auc, blimp, binding_composite)
      - y_gate: binary array (1 = passed S1)
      - y_rank: float array of composite_score (NaN where unavailable)
    """
    from ...synthesis.graph_features import (
        extract_graph_features_bundle,
        enrich_with_op_stats,
        load_op_stats,
    )

    try:
        rows = _load_screening_predictor_corpus_rows(db_path, validate=True)
    except CorpusIntegrityError:
        raise
    except Exception as e:
        logger.warning("GBM training data query failed: %s", e)
        return [], np.zeros(0), np.zeros(0), np.zeros(0), []

    # Load op_stats ONCE for all rows (avoids N+1 DB queries)
    op_stats_cache = load_op_stats(db_path)

    feat_dicts: List[Dict[str, float]] = []
    gate_labels: List[int] = []
    rank_labels: List[float] = []
    sample_weights: List[float] = []
    graph_signatures: List[str] = []

    for row in rows:
        gj = row["graph_json"]
        if not gj:
            continue
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature:
            signature = _graph_signature(gj_dict) or ""
        if not signature:
            continue
        feats, ops = extract_graph_features_bundle(gj_dict)
        if not feats:
            continue
        for op in ops:
            if op:
                feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
        enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
        # Post-eval features (NaN where unavailable — LightGBM handles
        # missing natively).  Excluded from gate model to prevent leakage;
        # used by rank model only.
        for post_key in (
            "hellaswag_acc_best",
            "induction_auc_best",
            "ar_auc_best",
            "blimp_overall_accuracy_best",
            "binding_composite_best",
            "initial_loss_best",
            "mean_grad_norm_best",
            "max_grad_norm_best",
            "grad_norm_std_best",
        ):
            v = row.get(post_key)
            feats[post_key] = float(v) if v is not None else float("nan")
        feat_dicts.append(feats)
        graph_signatures.append(signature)
        gate_labels.append(1 if row["stage1_any_passed"] else 0)
        sample_weights.append(rerun_confidence_weight(int(row.get("n_rows", 1))))
        # Rank target: prefer composite_score when available, otherwise fall back
        # to best observed wikitext perplexity for corpora without leaderboard data.
        comp = row.get("composite_score_best")
        if comp is not None:
            rank_labels.append(float(comp))
        else:
            ppl = row.get("wikitext_perplexity_best")
            rank_labels.append(float(ppl) if ppl is not None else float("nan"))

    return (
        feat_dicts,
        np.array(gate_labels, dtype=np.int32),
        np.array(rank_labels, dtype=np.float64),
        np.array(sample_weights, dtype=np.float64),
        graph_signatures,
    )


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
    feature_names: List[str] = field(default_factory=list)  # all features (rank model)
    gate_feature_names: List[str] = field(default_factory=list)  # structure only (gate)
    n_train: int = 0
    gate_threshold: float = 0.5
    gate_importance: Optional[Dict[str, float]] = None
    rank_importance: Optional[Dict[str, float]] = None
    train_metrics: Dict[str, Any] = field(default_factory=dict)

    def is_fitted(self) -> bool:
        return self.gate_model is not None and self.n_train > 0

    def predict_gate(self, features: Dict[str, float]) -> float:
        """Predict P(pass_s1) for a single graph. Returns 0.5 if not fitted."""
        if not self.is_fitted():
            return 0.5
        names = self.gate_feature_names or self.feature_names
        x = np.array([[features.get(k, 0.0) for k in names]], dtype=np.float32)
        try:
            return float(self.gate_model.predict(x)[0])
        except (TypeError, ValueError):
            return 0.5

    def predict_rank(self, features: Dict[str, float]) -> float:
        """Predict composite_score for a single graph. Returns 1e6 if not fitted."""
        if self.rank_model is None:
            return 1e6
        x = np.array(
            [[features.get(k, float("nan")) for k in self.feature_names]],
            dtype=np.float32,
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
                    "gate_feature_names": self.gate_feature_names,
                    "n_train": self.n_train,
                    "gate_threshold": self.gate_threshold,
                    "gate_importance": self.gate_importance,
                    "rank_importance": self.rank_importance,
                    "has_rank_model": self.rank_model is not None,
                    "train_metrics": self.train_metrics,
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
            gate_feature_names=list(
                meta.get("gate_feature_names", meta.get("feature_names", []))
            ),
            n_train=int(meta.get("n_train", 0)),
            gate_threshold=float(meta.get("gate_threshold", 0.5)),
            gate_importance=meta.get("gate_importance"),
            rank_importance=meta.get("rank_importance"),
            train_metrics=dict(meta.get("train_metrics", {})),
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

    feat_dicts, y_gate, y_rank, sample_weights, graph_signatures = (
        _query_graph_training_data(db_path)
    )

    if len(feat_dicts) < _MIN_GBM_SAMPLES:
        logger.info(
            "GBM predictor: insufficient data (%d < %d), skipping",
            len(feat_dicts),
            _MIN_GBM_SAMPLES,
        )
        return GBMPredictor()

    X, feature_names = build_dense_feature_matrix(feat_dicts)
    n_total = len(X)

    # Post-eval features leak the gate label (only non-NaN for entries
    # that completed training).  Excluded from gate; kept for rank model.
    _PROBE_FEATURE_NAMES = {
        "hellaswag_acc_best",
        "induction_auc_best",
        "ar_auc_best",
        "blimp_overall_accuracy_best",
        "binding_composite_best",
        "initial_loss_best",
        "mean_grad_norm_best",
        "max_grad_norm_best",
        "grad_norm_std_best",
    }
    gate_col_mask = np.array(
        [fn not in _PROBE_FEATURE_NAMES for fn in feature_names], dtype=bool
    )
    gate_feature_names = [fn for fn in feature_names if fn not in _PROBE_FEATURE_NAMES]
    X_gate = X[:, gate_col_mask]

    n_pos = int(y_gate.sum())
    n_neg = n_total - n_pos
    if n_pos < 5 or n_neg < 5:
        logger.info(
            "GBM predictor: insufficient class balance (pos=%d, neg=%d)", n_pos, n_neg
        )
        return GBMPredictor()

    train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y_gate, seed=42
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        logger.info("GBM predictor: grouped split failed, skipping")
        return GBMPredictor()

    # Full feature matrices (with probes) for rank model
    X_train, X_val = X[train_idx], X[val_idx]
    # Gate feature matrices (no probes — probes leak the S1 label)
    X_gate_train, X_gate_val = X_gate[train_idx], X_gate[val_idx]
    y_gate_train, y_gate_val = y_gate[train_idx], y_gate[val_idx]
    y_rank_train_full, y_rank_val_full = y_rank[train_idx], y_rank[val_idx]
    w_train, w_val = sample_weights[train_idx], sample_weights[val_idx]

    # Class imbalance handling
    pos_count = int(y_gate_train.sum())
    neg_count = len(y_gate_train) - pos_count
    spw = neg_count / max(pos_count, 1)

    # ── Gate model (binary classifier) ──
    # Uses structure features only (no probes) to avoid label leakage.
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
    train_set = lgb.Dataset(
        X_gate_train,
        label=y_gate_train,
        weight=w_train,
        feature_name=gate_feature_names,
    )
    val_set = lgb.Dataset(
        X_gate_val,
        label=y_gate_val,
        weight=w_val,
        feature_name=gate_feature_names,
        reference=train_set,
    )

    gate_model = lgb.train(
        gate_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    gate_importance = dict(
        zip(gate_feature_names, gate_model.feature_importance("gain").tolist())
    )

    # ── Rank model (regression on composite_score, with probe features) ──
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
            X_rank_train,
            label=y_rank_train,
            weight=w_train[rank_mask_tr],
            feature_name=feature_names,
        )
        r_val = lgb.Dataset(
            X_rank_val,
            label=y_rank_val,
            weight=w_val[rank_mask_va],
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
        rank_importance = dict(
            zip(feature_names, rank_model.feature_importance("gain").tolist())
        )

    gate_scores = gate_model.predict(X_gate_val)
    operating_points = operating_point_profiles(y_gate_val, gate_scores)
    gate_threshold = float(operating_points["f1"]["threshold"])
    gate_metrics = binary_classification_metrics(
        y_gate_val, gate_scores, gate_threshold
    )

    rank_spearman = 0.0
    if rank_model is not None:
        rank_mask_va = np.isfinite(y_rank_val_full)
        if rank_mask_va.sum() >= 10:
            rank_preds = rank_model.predict(X_val[rank_mask_va])
            from scipy.stats import spearmanr

            rho, _ = spearmanr(y_rank_val_full[rank_mask_va], rank_preds)
            rank_spearman = float(rho) if np.isfinite(rho) else 0.0

    n_trained = len(X_train)
    predictor = GBMPredictor(
        gate_model=gate_model,
        rank_model=rank_model,
        feature_names=feature_names,
        gate_feature_names=gate_feature_names,
        n_train=n_trained,
        gate_threshold=gate_threshold,
        gate_importance=gate_importance,
        rank_importance=rank_importance,
        train_metrics={
            "gate_threshold": gate_threshold,
            "gate_metrics": gate_metrics,
            "operating_points": operating_points,
            "rank_spearman": rank_spearman,
            "n_train": n_trained,
            "n_val": len(X_val),
            "n_positive": int(y_gate.sum()),
            **split_stats,
        },
    )

    logger.info(
        "GBM predictor trained: %d samples, %d features, gate_spw=%.1f, rank=%s, unique_graphs=%d, dup_groups=%d, ambiguous_dup_groups=%d",
        n_trained,
        len(feature_names),
        spw,
        "yes" if rank_model else "no",
        split_stats["n_unique_graphs"],
        split_stats["n_duplicate_groups"],
        split_stats["n_ambiguous_duplicate_groups"],
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

    feat_dicts, y_gate, y_rank, sample_weights, graph_signatures = (
        _query_graph_training_data(db_path)
    )
    n_total = len(feat_dicts)
    if n_total < _MIN_GBM_SAMPLES:
        return {"error": "insufficient_data", "n_total": n_total}

    X, feature_names = build_dense_feature_matrix(feat_dicts)

    # Strip post-eval probe features from gate evaluation — same as train_gbm().
    # These features leak the gate label (only non-NaN for entries that completed
    # training). Kept for rank model; excluded from gate.
    _PROBE_FEATURE_NAMES = {
        "hellaswag_acc_best",
        "induction_auc_best",
        "ar_auc_best",
        "blimp_overall_accuracy_best",
        "binding_composite_best",
        "initial_loss_best",
        "mean_grad_norm_best",
        "max_grad_norm_best",
        "grad_norm_std_best",
    }
    gate_col_mask = np.array(
        [fn not in _PROBE_FEATURE_NAMES for fn in feature_names], dtype=bool
    )
    gate_feature_names = [fn for fn in feature_names if fn not in _PROBE_FEATURE_NAMES]
    X_gate = X[:, gate_col_mask]

    n_pos = int(y_gate.sum())
    n_neg = n_total - n_pos
    if n_pos < 5 or n_neg < 5:
        return {"error": "insufficient_balance", "n_pos": n_pos, "n_neg": n_neg}

    train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y_gate, seed=42
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        return {"error": "grouped_split_failed", "n_total": n_total}

    X_gate_train, X_gate_test = X_gate[train_idx], X_gate[val_idx]
    X_train, X_test = X[train_idx], X[val_idx]
    y_gate_train, y_gate_test = y_gate[train_idx], y_gate[val_idx]
    y_rank_train_full, y_rank_test_full = y_rank[train_idx], y_rank[val_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[val_idx]
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
    train_set = lgb.Dataset(
        X_gate_train,
        label=y_gate_train,
        weight=w_train,
        feature_name=gate_feature_names,
    )
    val_set = lgb.Dataset(
        X_gate_test,
        label=y_gate_test,
        weight=w_test,
        feature_name=gate_feature_names,
        reference=train_set,
    )

    gate_model = lgb.train(
        gate_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Gate AUC (use probe-stripped features, matching train_gbm)
    gate_preds = gate_model.predict(X_gate_test)
    operating_points = operating_point_profiles(y_gate_test, gate_preds)
    gate_threshold = float(operating_points["f1"]["threshold"])
    gate_metrics = binary_classification_metrics(
        y_gate_test, gate_preds, gate_threshold
    )
    gate_auc = safe_binary_roc_auc(y_gate_test, gate_preds)

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
            weight=w_train[rank_mask_tr],
            feature_name=feature_names,
        )
        r_val = lgb.Dataset(
            X_test[rank_mask_te],
            label=y_rank_test_full[rank_mask_te],
            weight=w_test[rank_mask_te],
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
        zip(gate_feature_names, gate_model.feature_importance("gain").tolist())
    )
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    return {
        "gate_auc": gate_auc,
        "gate_threshold": gate_threshold,
        "gate_metrics": gate_metrics,
        "operating_points": operating_points,
        "rank_spearman": rank_spearman,
        "skip_rate": skip_rate,
        "false_skip_rate": false_skip_rate,
        "n_train": n_train,
        "n_test": n_test,
        "n_positive": int(y_gate.sum()),
        "n_total": n_total,
        **split_stats,
        "top_features": top_features,
    }


def evaluate_gbm_induction(
    db_path: str = "research/lab_notebook.db",
) -> Dict[str, Any]:
    """Train + evaluate GBM models for canonical induction labels."""
    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm_not_installed"}

    from ...synthesis.graph_features import (
        extract_graph_features_bundle,
        enrich_with_op_stats,
        load_op_stats,
    )

    try:
        rows = _load_screening_predictor_corpus_rows(db_path, validate=True)
    except CorpusIntegrityError:
        raise
    except Exception as e:
        return {"error": f"induction_query_failed: {e}"}

    op_stats_cache = load_op_stats(db_path)
    feat_dicts: List[Dict[str, float]] = []
    sample_weights_list: List[float] = []
    graph_signatures: List[str] = []
    y_auc_list: List[float] = []

    for row in rows:
        induction_auc = row.get("induction_auc_500")
        if induction_auc is None or not np.isfinite(float(induction_auc)):
            continue
        gj = row.get("graph_json")
        if not gj:
            continue
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature:
            signature = _graph_signature(gj_dict) or ""
        if not signature:
            continue
        feats, ops = extract_graph_features_bundle(gj_dict)
        if not feats:
            continue
        for op in ops:
            if op:
                feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
        enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
        feat_dicts.append(feats)
        sample_weights_list.append(rerun_confidence_weight(int(row.get("n_rows", 1))))
        graph_signatures.append(signature)
        y_auc_list.append(float(induction_auc))

    if len(feat_dicts) < _MIN_GBM_SAMPLES:
        return {
            "error": "insufficient_induction_data",
            "n_total": len(feat_dicts),
        }

    sample_weights = np.array(sample_weights_list, dtype=np.float64)
    y_auc = np.array(y_auc_list, dtype=np.float64)
    y_learner = (y_auc >= 0.02).astype(np.int32)

    n_pos = int(y_learner.sum())
    n_neg = len(y_learner) - n_pos
    if n_pos < 5 or n_neg < 5:
        return {
            "error": "insufficient_induction_balance",
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    X, feature_names = build_dense_feature_matrix(feat_dicts)
    train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y_learner, seed=42
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        return {"error": "grouped_split_failed", "n_total": len(X)}

    X_train, X_test = X[train_idx], X[val_idx]
    y_auc_train, y_auc_test = y_auc[train_idx], y_auc[val_idx]
    y_cls_train, y_cls_test = y_learner[train_idx], y_learner[val_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[val_idx]

    cls_spw = (len(y_cls_train) - int(y_cls_train.sum())) / max(
        int(y_cls_train.sum()), 1
    )
    cls_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "scale_pos_weight": cls_spw,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": 1,
        "seed": 42,
    }
    cls_train = lgb.Dataset(
        X_train, label=y_cls_train, weight=w_train, feature_name=feature_names
    )
    cls_val = lgb.Dataset(
        X_test,
        label=y_cls_test,
        weight=w_test,
        feature_name=feature_names,
        reference=cls_train,
    )
    cls_model = lgb.train(
        cls_params,
        cls_train,
        num_boost_round=500,
        valid_sets=[cls_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    reg_params = {
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
    reg_train = lgb.Dataset(
        X_train, label=y_auc_train, weight=w_train, feature_name=feature_names
    )
    reg_val = lgb.Dataset(
        X_test,
        label=y_auc_test,
        weight=w_test,
        feature_name=feature_names,
        reference=reg_train,
    )
    reg_model = lgb.train(
        reg_params,
        reg_train,
        num_boost_round=500,
        valid_sets=[reg_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    cls_preds = cls_model.predict(X_test)
    reg_preds = np.clip(reg_model.predict(X_test), 0.0, 1.0)
    learner_auc = safe_binary_roc_auc(y_cls_test, cls_preds)
    learner_acc = float(np.mean((reg_preds >= 0.02).astype(np.int32) == y_cls_test))
    induction_mae = float(np.mean(np.abs(y_auc_test - reg_preds)))
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(y_auc_test, reg_preds)
        induction_spearman = float(rho) if np.isfinite(rho) else 0.0
    except Exception:
        induction_spearman = 0.0

    importance = dict(zip(feature_names, reg_model.feature_importance("gain").tolist()))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    return {
        "learner_auc": learner_auc,
        "learner_acc_from_regression": learner_acc,
        "induction_mae": induction_mae,
        "induction_spearman": induction_spearman,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(X),
        "n_learners": int(y_learner.sum()),
        **split_stats,
        "top_features": top_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EnsemblePredictor: combines GBM + GraphPredictor + Bayesian + InteractionModel
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EnsemblePredictor:
    """Meta-learner combining prediction components.

    Scoring components (3D logistic regression):
    1. GBMPredictor (LightGBM gate/rank on graph-structure features)
    2. GraphPredictor (topology-aware features)
    3. TemporalBayesianTracker (op-level Bayesian posteriors)

    InteractionModel is still loaded for research (heatmaps, pair analysis)
    but excluded from the scoring blend — calibration weight was ≈ -0.06.

    Gate threshold uses the high_precision operating point to prioritise
    PPV (≥0.70) over recall, satisfying the screening trust policy.
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
    gate_threshold: float = 0.5
    _calibration_metrics: Dict[str, Any] = field(default_factory=dict)

    def is_fitted(self) -> bool:
        # Runtime-usable if at least one component is available.
        return any(
            [
                self.gbm is not None and self.gbm.is_fitted(),
                self.graph_pred is not None and self.graph_pred.is_fitted(),
                self.bayesian is not None,
                self.interaction is not None and self.interaction._trained,
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

        # GBM prediction
        if self.gbm is not None and self.gbm.is_fitted() and graph_features is not None:
            scores.append(self.gbm.predict_gate(graph_features))

        # Topology predictor
        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
        ):
            scores.append(self.graph_pred.predict_gate(graph_json))

        # Bayesian worst-op excluded from scoring — calibration weight was
        # -0.006 (noise).  Bayesian tracker remains active for grammar
        # advisory weights via op_weights()/template_weights().

        # InteractionModel excluded — calibration weight was ≈ -0.06.

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

    def predict_induction_auc(
        self,
        graph_json: Any = None,
        graph_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Predict canonical induction AUC from available components."""
        preds: List[float] = []

        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
            and hasattr(self.graph_pred, "predict_induction_auc")
        ):
            preds.append(float(self.graph_pred.predict_induction_auc(graph_json)))

        if self.gbm is not None and self.gbm.is_fitted() and graph_features is not None:
            logger.debug(
                "GBM induction prediction skipped: no persisted induction head is available"
            )

        if not preds:
            return 0.0
        return float(np.clip(np.mean(preds), 0.0, 1.0))

    def predict_induction_learner_prob(
        self,
        graph_json: Any = None,
        graph_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Predict probability that a graph will be an induction learner."""
        auc = self.predict_induction_auc(
            graph_json=graph_json, graph_features=graph_features
        )
        # 0.02 is the current learner threshold; scale around that boundary.
        return float(np.clip(auc / 0.02, 0.0, 1.0))

    def predict_planning_score(
        self,
        graph_json: Any = None,
        graph_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Return a planning score that prioritizes likely survivors and likely winners."""
        p_pass = self.predict_gate(graph_json=graph_json, graph_features=graph_features)
        induction_auc = self.predict_induction_auc(
            graph_json=graph_json, graph_features=graph_features
        )
        p_induction = self.predict_induction_learner_prob(
            graph_json=graph_json, graph_features=graph_features
        )
        quality_terms: List[float] = []

        if self.gbm is not None and self.gbm.is_fitted() and graph_features is not None:
            gbm_rank = float(self.gbm.predict_rank(graph_features))
            if np.isfinite(gbm_rank) and gbm_rank < 1e5:
                quality_terms.append(float(np.exp(-max(gbm_rank, 0.0) / 25.0)))

        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
        ):
            graph_rank = float(self.graph_pred.predict_rank(graph_json))
            if np.isfinite(graph_rank) and graph_rank < 1e5:
                quality_terms.append(float(np.exp(-max(graph_rank, 0.0) / 25.0)))
            if hasattr(self.graph_pred, "predict_loss"):
                predicted_loss = float(self.graph_pred.predict_loss(graph_json))
                if np.isfinite(predicted_loss):
                    quality_terms.append(
                        float(np.clip(1.0 - (max(predicted_loss, 0.0) / 0.7), 0.0, 1.0))
                    )

        quality_score = (
            float(np.clip(np.mean(quality_terms), 0.0, 1.0)) if quality_terms else 0.0
        )
        # Induction head has Spearman 0.21 — not useful for ranking.
        # Removed from blend until induction predictor achieves Spearman >= 0.5.
        # Previous: 0.5*p_pass + 0.25*quality + 0.25*induction (25% noise).
        if quality_terms:
            blended = float(
                np.clip(
                    0.65 * p_pass + 0.35 * quality_score, 0.0, 1.0
                )
            )
        else:
            blended = float(np.clip(p_pass, 0.0, 1.0))
        return {
            "p_pass": float(p_pass),
            "predicted_induction_auc": float(induction_auc),
            "p_induction_learner": float(p_induction),
            "predicted_quality_score": float(quality_score),
            "planning_score": blended,
        }

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
            "graph_pred_has_induction": self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and hasattr(self.graph_pred, "predict_induction_auc"),
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
        else:
            _unlink_if_exists(state_dir / _GBM_GATE_MODEL_PATH.name)
            _unlink_if_exists(state_dir / _GBM_RANK_MODEL_PATH.name)
            _unlink_if_exists(state_dir / _GBM_META_PATH.name)
        if self.graph_pred is not None and self.graph_pred.is_fitted():
            self.graph_pred.save(state_dir / _GRAPH_PREDICTOR_PATH.name)
        else:
            _unlink_if_exists(state_dir / _GRAPH_PREDICTOR_PATH.name)
            _unlink_if_exists(
                (state_dir / _GRAPH_PREDICTOR_PATH.name).with_suffix(".json")
            )
        if self.bayesian is not None:
            self.bayesian.save_state(state_dir / _BAYESIAN_STATE_PATH.name)
        else:
            _unlink_if_exists(state_dir / _BAYESIAN_STATE_PATH.name)
        if self.interaction is not None and self.interaction._trained:
            self.interaction.save(state_dir / _INTERACTION_MODEL_PATH.name)
        else:
            _unlink_if_exists(state_dir / _INTERACTION_MODEL_PATH.name)
            _unlink_if_exists(
                (state_dir / _INTERACTION_MODEL_PATH.name).with_suffix(".json")
            )
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
                    "gate_threshold": self.gate_threshold,
                    "calibration_metrics": self._calibration_metrics,
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
                ensemble.gate_threshold = float(meta.get("gate_threshold", 0.5))
                ensemble._calibration_metrics = dict(
                    meta.get("calibration_metrics", {})
                )
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
    n_epochs: int = 60,
    lr: float = 0.01,
) -> None:
    """Calibrate ensemble blend weights from held-out program_results.

    Fits a logistic regression on component scores → actual S1 labels.
    This learns which components to trust more and the optimal threshold.
    Uses the high_precision operating point for the gate threshold to
    prioritise PPV (fewer false positives) over recall.
    """

    rows = [
        row
        for row in _load_screening_predictor_corpus_rows(db_path, validate=False)
        if bool(row.get("stage0_any_passed"))
    ]

    if len(rows) < 100:
        return

    # Collect component scores for each graph
    from ...synthesis.graph_features import (
        extract_graph_features_bundle,
        enrich_with_op_stats,
        load_op_stats,
    )

    op_stats_cache = load_op_stats(db_path)
    score_rows: list = []
    labels: list = []
    sample_weights: list = []
    graph_signatures: List[str] = []

    for row in rows:
        gj = row["graph_json"]
        s1 = int(bool(row["stage1_any_passed"]))
        rerun_weight = rerun_confidence_weight(int(row.get("n_rows", 1)))
        try:
            gj_dict = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        signature = str(row.get("canonical_fingerprint") or "")
        if not signature:
            signature = _graph_signature(gj_dict) or ""
        if not signature:
            continue

        scores = []

        # GBM score
        if ensemble.gbm is not None and ensemble.gbm.is_fitted():
            feats, ops = extract_graph_features_bundle(gj_dict)
            if feats:
                for op in ops:
                    if op:
                        feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
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

        # Bayesian worst-op excluded from calibration — weight was -0.006
        # (noise).  Bayesian tracker stays active for grammar advisory.

        # InteractionModel excluded from calibration — learned weight was
        # ≈ -0.06 (effectively zero/harmful).  Kept for research use but
        # not fed to the meta-learner.

        score_rows.append(scores)
        labels.append(int(s1 or 0))
        sample_weights.append(rerun_weight)
        graph_signatures.append(signature)

    if len(score_rows) < 100:
        return

    X = np.array(score_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    sample_w = np.array(sample_weights, dtype=np.float64)
    n_dims = X.shape[1]
    if np.unique(y).size < 2:
        logger.info(
            "Ensemble calibration skipped: insufficient class balance (classes=%d)",
            np.unique(y).size,
        )
        ensemble.w_ensemble = np.zeros(0, dtype=np.float32)
        ensemble._score_mean = np.zeros(0, dtype=np.float32)
        ensemble._score_std = np.zeros(0, dtype=np.float32)
        ensemble._n_score_dims = 0
        ensemble.gate_threshold = 0.5
        ensemble._calibration_metrics = {
            "error": "insufficient_class_balance",
            "n_samples": len(y),
            "n_classes": int(np.unique(y).size),
        }
        return

    train_idx, val_idx, split_stats = grouped_stratified_split(
        graph_signatures, y.astype(np.int32), seed=42
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        return

    # Standardize scores using train statistics only.
    score_mean = X[train_idx].mean(axis=0)
    score_std = X[train_idx].std(axis=0)
    score_std[score_std < 1e-8] = 1.0
    X_norm = (X - score_mean) / score_std

    rng = np.random.RandomState(42)
    X_tr, X_va = X_norm[train_idx], X_norm[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    w_tr = sample_w[train_idx]
    if np.unique(y_tr).size < 2 or np.unique(y_va).size < 2:
        logger.info(
            "Ensemble calibration skipped: split lost class diversity (train=%d classes, val=%d classes)",
            np.unique(y_tr).size,
            np.unique(y_va).size,
        )
        ensemble.w_ensemble = np.zeros(0, dtype=np.float32)
        ensemble._score_mean = np.zeros(0, dtype=np.float32)
        ensemble._score_std = np.zeros(0, dtype=np.float32)
        ensemble._n_score_dims = 0
        ensemble.gate_threshold = 0.5
        ensemble._calibration_metrics = {
            "error": "split_lost_class_diversity",
            "n_train": len(y_tr),
            "n_val": len(y_va),
            **split_stats,
        }
        return

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
            w_b = w_tr[idx]
            preds = _sig(x_b @ w + b)
            err = (preds - y_b) * w_b
            grad = err[:, None] * x_b
            denom = max(float(np.sum(w_b)), 1e-8)
            w -= lr * (grad.sum(axis=0) / denom) + lr * 0.01 * w
            b -= lr * float(err.sum() / denom)

    # Validate
    val_preds = _sig(X_va @ w + b)
    operating_points = operating_point_profiles(y_va, val_preds)
    # Use the F1-optimal operating point — balances precision and recall.
    # high_precision (PPV≥0.70) was discarding ~50% of true positives,
    # starving downstream models of labeled data.  F1 recovers ~30% more
    # recall at an acceptable PPV trade-off (~0.50) and feeds the data
    # flywheel: more evals → more labels → better models.
    gate_threshold = float(operating_points["f1"]["threshold"])
    selected_metrics = operating_points["f1"]
    val_acc = float(selected_metrics["accuracy"])

    ensemble.w_ensemble = w.astype(np.float32)
    ensemble.b_ensemble = float(b)
    ensemble._score_mean = score_mean.astype(np.float32)
    ensemble._score_std = score_std.astype(np.float32)
    ensemble._n_score_dims = n_dims
    ensemble.gate_threshold = gate_threshold
    ensemble._calibration_metrics = {
        "gate_threshold": gate_threshold,
        "val_metrics": binary_classification_metrics(y_va, val_preds, gate_threshold),
        "operating_points": operating_points,
        "n_train": len(X_tr),
        "n_val": len(X_va),
        **split_stats,
    }

    logger.info(
        "Ensemble calibrated: %d-dim logistic regression, val_acc=%.3f, "
        "weights=[%s], bias=%.3f (%d train, %d val, unique_graphs=%d, dup_groups=%d, ambiguous_dup_groups=%d)",
        n_dims,
        val_acc,
        ", ".join(f"{wi:.3f}" for wi in w),
        b,
        len(X_tr),
        len(X_va),
        split_stats["n_unique_graphs"],
        split_stats["n_duplicate_groups"],
        split_stats["n_ambiguous_duplicate_groups"],
    )
