"""Performance predictor split module. Re-exported via predictor."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

from .metrics_utils import (
    binary_classification_metrics,
    operating_point_profiles,
    safe_binary_roc_auc,
)
from .ml_corpus import (
    CorpusIntegrityError,
    build_dense_feature_matrix,
    grouped_stratified_split,
    load_screening_predictor_corpus_rows,
    rerun_confidence_weight,
)
from .predictor_artifacts import (
    GBM_GATE_MODEL_PATH as _GBM_GATE_MODEL_PATH,
    GBM_META_PATH as _GBM_META_PATH,
    GBM_RANK_COMPOSITE_MODEL_PATH as _GBM_RANK_COMPOSITE_MODEL_PATH,
    GBM_RANK_MODEL_PATH as _GBM_RANK_MODEL_PATH,
    GBM_RANK_PPL_MODEL_PATH as _GBM_RANK_PPL_MODEL_PATH,
    ensure_state_dir,
    read_json,
    unlink_paths,
    write_json,
)

logger = logging.getLogger(__name__)

from .predictor_ridge import _extract_features  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# GBMPredictor: LightGBM graph-structure pre-screener
# ─────────────────────────────────────────────────────────────────────────────

_MIN_GBM_SAMPLES = 50  # minimum rows to train
_MIN_PPL_RANK_SPEARMAN = 0.30
_MIN_PPL_RANK_NDCG = 0.50
_MIN_COMPOSITE_RANK_SPEARMAN = 0.50
_MIN_COMPOSITE_RANK_NDCG = 0.60


_POST_EVAL_FEATURE_NAMES = (
    "hellaswag_acc_best",
    "induction_screening_auc_best",
    "ar_legacy_auc_best",
    "blimp_overall_accuracy_best",
    "binding_screening_composite_best",
    "induction_intermediate_auc_best",
    "binding_intermediate_auc_best",
    "validation_loss_ratio_best",
    "rapid_screening_passed_best",
    "initial_loss_best",
    "mean_grad_norm_best",
    "max_grad_norm_best",
    "grad_norm_std_best",
    # v9 Gemini trajectory metrics — added 2026-04-25. These should
    # noticeably move the predictor's ROC AUC if the smoke-data signal
    # holds at scale (top-ind-1 had 400× erf_variance vs GPT-2).
    "fp_jacobian_erf_density_best",
    "fp_jacobian_erf_variance_best",
    "fp_icld_velocity_best",
    "fp_logit_margin_velocity_best",
    "fp_id_collapse_rate_best",
    "fp_jacobian_spectral_norm_best",
    # Understanding-tier eval features (added 2026-04-26 after the
    # diagnostic + cross_task backfill closed the population gap).
    "diagnostic_score_best",
    "cross_task_score_best",
)
_PROBE_FEATURE_NAMES = set(_POST_EVAL_FEATURE_NAMES)


def _quality_from_ppl(value: float) -> float:
    return float(np.exp(-max(float(value), 0.0) / 25.0))


def _quality_from_composite(value: float) -> float:
    return float(np.clip(float(value) / 100.0, 0.0, 1.0))


def _rank_target_quality(
    values: np.ndarray,
    *,
    target_kind: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if target_kind == "ppl":
        return np.array([_quality_from_ppl(v) for v in arr], dtype=np.float64)
    if target_kind == "composite":
        return np.array([_quality_from_composite(v) for v in arr], dtype=np.float64)
    raise ValueError(f"unknown rank target kind: {target_kind}")


def _ranking_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_kind: str,
) -> Dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = np.asarray(y_true[mask], dtype=np.float64)
    y_pred_f = np.asarray(y_pred[mask], dtype=np.float64)
    n = int(y_true_f.size)
    if n < 2:
        return {
            "target_kind": target_kind,
            "n": n,
            "error": "insufficient_data",
        }

    from scipy.stats import kendalltau, pearsonr, spearmanr

    rho, _ = spearmanr(y_true_f, y_pred_f)
    pearson = pearsonr(y_true_f, y_pred_f).statistic
    kendall = kendalltau(y_true_f, y_pred_f).statistic

    true_quality = _rank_target_quality(y_true_f, target_kind=target_kind)
    pred_quality = _rank_target_quality(y_pred_f, target_kind=target_kind)
    k = min(max(int(np.ceil(n * 0.1)), 5), n)

    def _dcg(order: np.ndarray) -> float:
        gains = true_quality[order]
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        return float(np.sum(gains * discounts))

    pred_order = np.argsort(-pred_quality)
    ideal_order = np.argsort(-true_quality)
    ndcg_at_k = 0.0
    ideal_dcg = _dcg(ideal_order[:k])
    if ideal_dcg > 0.0:
        ndcg_at_k = float(_dcg(pred_order[:k]) / ideal_dcg)

    pred_top = set(pred_order[:k].tolist())
    true_top = set(ideal_order[:k].tolist())
    overlap = len(pred_top & true_top)

    return {
        "target_kind": target_kind,
        "n": n,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
        "kendall": float(kendall) if np.isfinite(kendall) else 0.0,
        "mae": float(np.mean(np.abs(y_true_f - y_pred_f))),
        "rmse": float(np.sqrt(np.mean((y_true_f - y_pred_f) ** 2))),
        "ndcg_at_top_decile": ndcg_at_k,
        "top_decile_n": int(k),
        "top_decile_overlap": int(overlap),
        "top_decile_hit_rate": float(overlap / max(k, 1)),
        "top_decile_mean_true_quality": float(np.mean(true_quality[pred_order[:k]])),
    }


def _ppl_log_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return float(np.log(1.0)), float(np.log(1e6))
    y_log = np.log(np.maximum(finite, 1.0))
    log_min = float(np.percentile(y_log, 0.5))
    log_max = float(np.percentile(y_log, 99.5))
    if log_max < log_min:
        log_max = log_min
    return log_min, log_max


def _fit_rank_head(
    lgb: Any,
    *,
    X_train: np.ndarray,
    y_train_full: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val_full: np.ndarray,
    w_val: np.ndarray,
    feature_names: List[str],
    target_kind: str,
) -> tuple[Any, Optional[Dict[str, float]], Dict[str, Any], Dict[str, Any]]:
    rank_mask_tr = np.isfinite(y_train_full)
    rank_mask_va = np.isfinite(y_val_full)
    n_train = int(rank_mask_tr.sum())
    n_val = int(rank_mask_va.sum())
    if n_train < 30 or n_val < 10:
        return (
            None,
            None,
            {
                "target_kind": target_kind,
                "n_train": n_train,
                "n_val": n_val,
                "error": "insufficient_data",
            },
            {},
        )

    y_train_target = np.asarray(y_train_full[rank_mask_tr], dtype=np.float64)
    y_val_target = np.asarray(y_val_full[rank_mask_va], dtype=np.float64)
    target_meta: Dict[str, Any] = {}
    if target_kind == "ppl":
        log_min, log_max = _ppl_log_bounds(y_train_target)
        y_train_target = np.clip(
            np.log(np.maximum(y_train_target, 1.0)), log_min, log_max
        )
        y_val_target = np.clip(np.log(np.maximum(y_val_target, 1.0)), log_min, log_max)
        target_meta = {
            "log_min": float(log_min),
            "log_max": float(log_max),
            "training_target_space": "clipped_log_ppl",
        }

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
        label=y_train_target,
        weight=w_train[rank_mask_tr],
        feature_name=feature_names,
    )
    r_val = lgb.Dataset(
        X_val[rank_mask_va],
        label=y_val_target,
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
    importance = dict(
        zip(feature_names, rank_model.feature_importance("gain").tolist())
    )
    rank_preds = rank_model.predict(X_val[rank_mask_va])
    if target_kind == "ppl":
        log_min = float(target_meta["log_min"])
        log_max = float(target_meta["log_max"])
        rank_preds = np.exp(np.clip(rank_preds, log_min, log_max))
    diagnostics = _ranking_diagnostics(
        y_val_full[rank_mask_va],
        rank_preds,
        target_kind=target_kind,
    )
    diagnostics.update({"n_train": n_train, "n_val": n_val, **target_meta})
    return rank_model, importance, diagnostics, target_meta


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
    """Backward-compatible monkeypatch point over the canonical corpus loader."""
    return load_screening_predictor_corpus_rows(db_path, validate=validate)


def analyze_graph_label_quality(db_path: str) -> Dict[str, Any]:
    """Summarize duplicate-graph ambiguity and hard-negative composition."""
    try:
        from ..notebook.shared_conn import get_notebook_conn

        conn = get_notebook_conn(db_path)
        rows = conn.execute(
            """SELECT graph_json, stage1_passed, stage0_passed, stage05_passed
               FROM program_results_compat
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
        try:
            gj = resolve_graph_json_value(conn, db_path, row["graph_json"])
        except Exception:
            continue
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


def _load_gbm_corpus_rows(db_path: str) -> List[Dict[str, Any]]:
    try:
        return _load_screening_predictor_corpus_rows(db_path, validate=True)
    except CorpusIntegrityError:
        raise
    except Exception as e:
        logger.warning("GBM training data query failed: %s", e)
        return []


def _row_graph_and_signature(row: Dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    gj = row["graph_json"]
    if not gj:
        return None
    try:
        gj_dict = json.loads(gj) if isinstance(gj, str) else gj
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(gj_dict, dict):
        return None
    signature = str(row.get("canonical_fingerprint") or "")
    if not signature:
        signature = _graph_signature(gj_dict) or ""
    if not signature:
        return None
    return gj_dict, signature


def _add_op_histogram(feats: Dict[str, float], ops: list[str]) -> None:
    for op in ops:
        if op:
            feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0


def _add_post_eval_features(feats: Dict[str, float], row: Dict[str, Any]) -> None:
    for post_key in _POST_EVAL_FEATURE_NAMES:
        value = row.get(post_key)
        feats[post_key] = float(value) if value is not None else float("nan")


def _training_feature_row(row: Dict[str, Any], op_stats_cache):
    from ...synthesis.graph_features import (
        enrich_with_op_stats,
        extract_graph_features_bundle,
    )

    parsed = _row_graph_and_signature(row)
    if parsed is None:
        return None
    graph, signature = parsed
    feats, ops = extract_graph_features_bundle(graph)
    if not feats:
        return None
    _add_op_histogram(feats, ops)
    enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
    _add_post_eval_features(feats, row)
    return feats, signature


def _append_training_labels(
    row: Dict[str, Any],
    *,
    gate_labels: List[int],
    rank_labels_ppl: List[float],
    rank_labels_composite: List[float],
    sample_weights: List[float],
    latest_timestamps: List[float],
) -> None:
    gate_labels.append(1 if row["stage1_any_passed"] else 0)
    sample_weights.append(rerun_confidence_weight(int(row.get("n_rows", 1))))
    latest_timestamps.append(float(row.get("latest_timestamp") or 0.0))
    comp = row.get("composite_score_best")
    ppl = row.get("wikitext_perplexity_best")
    rank_labels_composite.append(float(comp) if comp is not None else float("nan"))
    rank_labels_ppl.append(float(ppl) if ppl is not None else float("nan"))


def _query_graph_training_data(
    db_path: str,
) -> Tuple[
    List[Dict[str, float]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """Query graph_json + labels from ALL program_results for GBM training.

    Uses every graph — including failures without final_loss (they're known
    negatives). This gives the gate model a much clearer picture of what
    "hopeless" looks like vs only training on graphs that survived long enough
    to produce a loss.

    Returns (feature_dicts, y_gate, y_rank_ppl, y_rank_composite, sample_weights,
    latest_timestamps, graph_signatures) where:
      - feature_dicts: list of dicts from extract_graph_features (with full op histogram
        + probe features: hellaswag_acc, induction_screening_auc, ar_legacy_auc, blimp, binding_screening_composite)
      - y_gate: binary array (1 = passed S1)
      - y_rank_ppl: float array of best wikitext perplexity (NaN where unavailable)
      - y_rank_composite: float array of best composite score (NaN where unavailable)
    """
    from ...synthesis.graph_features import load_op_stats

    rows = _load_gbm_corpus_rows(db_path)
    if not rows:
        return [], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), []

    op_stats_cache = load_op_stats(db_path)
    feat_dicts: List[Dict[str, float]] = []
    gate_labels: List[int] = []
    rank_labels_ppl: List[float] = []
    rank_labels_composite: List[float] = []
    sample_weights: List[float] = []
    latest_timestamps: List[float] = []
    graph_signatures: List[str] = []

    for row in rows:
        training_row = _training_feature_row(row, op_stats_cache)
        if training_row is None:
            continue
        feats, signature = training_row
        feat_dicts.append(feats)
        graph_signatures.append(signature)
        _append_training_labels(
            row,
            gate_labels=gate_labels,
            rank_labels_ppl=rank_labels_ppl,
            rank_labels_composite=rank_labels_composite,
            sample_weights=sample_weights,
            latest_timestamps=latest_timestamps,
        )

    return (
        feat_dicts,
        np.array(gate_labels, dtype=np.int32),
        np.array(rank_labels_ppl, dtype=np.float64),
        np.array(rank_labels_composite, dtype=np.float64),
        np.array(sample_weights, dtype=np.float64),
        np.array(latest_timestamps, dtype=np.float64),
        graph_signatures,
    )


@dataclass(slots=True)
class GBMPredictor:
    """LightGBM-based graph pre-screener.

    Three models:
      gate_model: classifier — P(pass_s1 | graph_features)
      rank_model_ppl: regressor — predicted wikitext_perplexity (lower is better)
      rank_model_composite: regressor — predicted composite_score (higher is better)

    Both operate on graph-structure features only (no forward pass needed).
    """

    gate_model: Any = None  # lgb.Booster
    rank_model_ppl: Any = None  # lgb.Booster
    rank_model_composite: Any = None  # lgb.Booster
    feature_names: List[str] = field(default_factory=list)  # all features (rank model)
    gate_feature_names: List[str] = field(default_factory=list)  # structure only (gate)
    n_train: int = 0
    gate_threshold: float = 0.5
    gate_importance: Optional[Dict[str, float]] = None
    rank_importance_ppl: Optional[Dict[str, float]] = None
    rank_importance_composite: Optional[Dict[str, float]] = None
    rank_ppl_log_min: float = float(np.log(1.0))
    rank_ppl_log_max: float = float(np.log(1e6))
    legacy_mixed_rank_model_loaded: bool = False
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

    def _predict_rank_head(self, model: Any, features: Dict[str, float]) -> float:
        if model is None:
            return 1e6
        x = np.array(
            [[features.get(k, float("nan")) for k in self.feature_names]],
            dtype=np.float32,
        )
        try:
            return float(model.predict(x)[0])
        except (TypeError, ValueError):
            return 1e6

    def predict_rank(self, features: Dict[str, float]) -> float:
        """Predict perplexity-style rank (lower is better). Returns 1e6 if unavailable."""
        return self.predict_rank_ppl(features)

    def _rank_head_metrics(self, target_kind: str) -> Dict[str, Any]:
        return dict((self.train_metrics.get("rank_heads") or {}).get(target_kind) or {})

    def rank_head_is_usable(self, target_kind: str) -> bool:
        metrics = self._rank_head_metrics(target_kind)
        spearman = float(metrics.get("spearman", 0.0) or 0.0)
        ndcg = float(metrics.get("ndcg_at_top_decile", 0.0) or 0.0)
        if target_kind == "ppl":
            return (
                self.rank_model_ppl is not None
                and spearman >= _MIN_PPL_RANK_SPEARMAN
                and ndcg >= _MIN_PPL_RANK_NDCG
            )
        if target_kind == "composite":
            return (
                self.rank_model_composite is not None
                and spearman >= _MIN_COMPOSITE_RANK_SPEARMAN
                and ndcg >= _MIN_COMPOSITE_RANK_NDCG
            )
        return False

    def predict_rank_ppl(self, features: Dict[str, float]) -> float:
        if not self.rank_head_is_usable("ppl"):
            return 1e6
        log_ppl = self._predict_rank_head(self.rank_model_ppl, features)
        if not np.isfinite(log_ppl) or log_ppl >= 1e5:
            return 1e6
        log_ppl = float(np.clip(log_ppl, self.rank_ppl_log_min, self.rank_ppl_log_max))
        return float(np.exp(log_ppl))

    def predict_rank_composite(self, features: Dict[str, float]) -> float:
        if not self.rank_head_is_usable("composite"):
            return 1e6
        return self._predict_rank_head(self.rank_model_composite, features)

    def predict_quality_score(self, features: Dict[str, float]) -> float:
        quality_terms: List[float] = []
        ppl = self.predict_rank_ppl(features)
        if np.isfinite(ppl) and ppl < 1e5:
            quality_terms.append(_quality_from_ppl(ppl))
        composite = self.predict_rank_composite(features)
        if np.isfinite(composite) and composite > -1e5:
            quality_terms.append(_quality_from_composite(composite))
        if not quality_terms:
            return 0.0
        return float(np.clip(np.mean(quality_terms), 0.0, 1.0))

    def save(self, state_dir: Path) -> None:
        """Persist LightGBM models plus feature metadata."""
        if not self.is_fitted():
            return
        state_dir = ensure_state_dir(state_dir)
        self.gate_model.save_model(str(state_dir / _GBM_GATE_MODEL_PATH.name))
        if self.rank_model_ppl is not None:
            self.rank_model_ppl.save_model(
                str(state_dir / _GBM_RANK_PPL_MODEL_PATH.name)
            )
        if self.rank_model_composite is not None:
            self.rank_model_composite.save_model(
                str(state_dir / _GBM_RANK_COMPOSITE_MODEL_PATH.name)
            )
        unlink_paths(state_dir / _GBM_RANK_MODEL_PATH.name)
        write_json(
            state_dir / _GBM_META_PATH.name,
            {
                "feature_names": self.feature_names,
                "gate_feature_names": self.gate_feature_names,
                "n_train": self.n_train,
                "gate_threshold": self.gate_threshold,
                "gate_importance": self.gate_importance,
                "rank_importance_ppl": self.rank_importance_ppl,
                "rank_importance_composite": self.rank_importance_composite,
                "rank_ppl_log_min": self.rank_ppl_log_min,
                "rank_ppl_log_max": self.rank_ppl_log_max,
                "has_rank_model_ppl": self.rank_model_ppl is not None,
                "has_rank_model_composite": self.rank_model_composite is not None,
                "rank_semantics_version": 2,
                "train_metrics": self.train_metrics,
            },
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
        meta = read_json(meta_path)
        gate_model = lgb.Booster(model_file=str(gate_model_path))
        rank_model_ppl = None
        if meta.get("has_rank_model_ppl"):
            rank_model_path = state_dir / _GBM_RANK_PPL_MODEL_PATH.name
            if rank_model_path.exists():
                rank_model_ppl = lgb.Booster(model_file=str(rank_model_path))
        rank_model_composite = None
        if meta.get("has_rank_model_composite"):
            rank_model_path = state_dir / _GBM_RANK_COMPOSITE_MODEL_PATH.name
            if rank_model_path.exists():
                rank_model_composite = lgb.Booster(model_file=str(rank_model_path))
        legacy_mixed_rank_model_loaded = False
        if (
            rank_model_ppl is None
            and rank_model_composite is None
            and meta.get("has_rank_model")
            and (state_dir / _GBM_RANK_MODEL_PATH.name).exists()
        ):
            legacy_mixed_rank_model_loaded = True
        return cls(
            gate_model=gate_model,
            rank_model_ppl=rank_model_ppl,
            rank_model_composite=rank_model_composite,
            feature_names=list(meta.get("feature_names", [])),
            gate_feature_names=list(
                meta.get("gate_feature_names", meta.get("feature_names", []))
            ),
            n_train=int(meta.get("n_train", 0)),
            gate_threshold=float(meta.get("gate_threshold", 0.5)),
            gate_importance=meta.get("gate_importance"),
            rank_importance_ppl=meta.get("rank_importance_ppl"),
            rank_importance_composite=meta.get("rank_importance_composite"),
            rank_ppl_log_min=float(meta.get("rank_ppl_log_min", np.log(1.0))),
            rank_ppl_log_max=float(meta.get("rank_ppl_log_max", np.log(1e6))),
            legacy_mixed_rank_model_loaded=legacy_mixed_rank_model_loaded,
            train_metrics=dict(meta.get("train_metrics", {})),
        )


def train_gbm(
    db_path: str = RUNS_DB,
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

    (
        feat_dicts,
        y_gate,
        y_rank_ppl,
        y_rank_composite,
        sample_weights,
        _latest_timestamps,
        graph_signatures,
    ) = _query_graph_training_data(db_path)

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
    # that completed training).  Excluded from gate; kept for rank heads.
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
    y_rank_ppl_train, y_rank_ppl_val = y_rank_ppl[train_idx], y_rank_ppl[val_idx]
    y_rank_composite_train = y_rank_composite[train_idx]
    y_rank_composite_val = y_rank_composite[val_idx]
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

    # ── Rank heads (semantically separate targets) ──
    rank_model_ppl, rank_importance_ppl, rank_diag_ppl, rank_ppl_meta = _fit_rank_head(
        lgb,
        X_train=X_train,
        y_train_full=y_rank_ppl_train,
        w_train=w_train,
        X_val=X_val,
        y_val_full=y_rank_ppl_val,
        w_val=w_val,
        feature_names=feature_names,
        target_kind="ppl",
    )
    (
        rank_model_composite,
        rank_importance_composite,
        rank_diag_composite,
        _rank_composite_meta,
    ) = _fit_rank_head(
        lgb,
        X_train=X_train,
        y_train_full=y_rank_composite_train,
        w_train=w_train,
        X_val=X_val,
        y_val_full=y_rank_composite_val,
        w_val=w_val,
        feature_names=feature_names,
        target_kind="composite",
    )

    gate_scores = gate_model.predict(X_gate_val)
    operating_points = operating_point_profiles(y_gate_val, gate_scores)
    gate_threshold = float(operating_points["f1"]["threshold"])
    gate_metrics = binary_classification_metrics(
        y_gate_val, gate_scores, gate_threshold
    )

    n_trained = len(X_train)
    predictor = GBMPredictor(
        gate_model=gate_model,
        rank_model_ppl=rank_model_ppl,
        rank_model_composite=rank_model_composite,
        feature_names=feature_names,
        gate_feature_names=gate_feature_names,
        n_train=n_trained,
        gate_threshold=gate_threshold,
        gate_importance=gate_importance,
        rank_importance_ppl=rank_importance_ppl,
        rank_importance_composite=rank_importance_composite,
        rank_ppl_log_min=float(rank_ppl_meta.get("log_min", np.log(1.0))),
        rank_ppl_log_max=float(rank_ppl_meta.get("log_max", np.log(1e6))),
        train_metrics={
            "gate_threshold": gate_threshold,
            "gate_metrics": gate_metrics,
            "operating_points": operating_points,
            "rank_heads": {
                "ppl": rank_diag_ppl,
                "composite": rank_diag_composite,
            },
            "rank_spearman_ppl": rank_diag_ppl.get("spearman", 0.0),
            "rank_spearman_composite": rank_diag_composite.get("spearman", 0.0),
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
        "yes"
        if (rank_model_ppl is not None or rank_model_composite is not None)
        else "no",
        split_stats["n_unique_graphs"],
        split_stats["n_duplicate_groups"],
        split_stats["n_ambiguous_duplicate_groups"],
    )
    return predictor


def evaluate_gbm(
    db_path: str = RUNS_DB,
    *,
    excluded_feature_names: Optional[set] = None,
) -> Dict[str, Any]:
    """Train + evaluate GBM predictor with hold-out metrics.

    Returns dict with gate_auc, rank_spearman, skip_rate, n_train, n_test.

    ``excluded_feature_names`` drops the listed feature columns from BOTH
    the gate matrix and the rank-head matrix before training. This is
    how the A/B harness compares "with vs without Gemini features" —
    pass ``{"fp_jacobian_erf_density_best", ...}`` to disable them.
    The drop happens after dense-matrix construction, so the corpus
    query is unchanged.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm_not_installed"}

    (
        feat_dicts,
        y_gate,
        y_rank_ppl,
        y_rank_composite,
        sample_weights,
        _latest_timestamps,
        graph_signatures,
    ) = _query_graph_training_data(db_path)
    n_total = len(feat_dicts)
    if n_total < _MIN_GBM_SAMPLES:
        return {"error": "insufficient_data", "n_total": n_total}

    X, feature_names = build_dense_feature_matrix(feat_dicts)

    # Drop A/B-excluded columns from BOTH gate and rank matrices.
    if excluded_feature_names:
        keep_mask = np.array(
            [fn not in excluded_feature_names for fn in feature_names], dtype=bool
        )
        X = X[:, keep_mask]
        feature_names = [fn for fn in feature_names if fn not in excluded_feature_names]

    # Strip post-eval probe features from gate evaluation — same as train_gbm().
    # These features leak the gate label (only non-NaN for entries that completed
    # training). Kept for rank heads; excluded from gate.
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
    y_rank_ppl_train, y_rank_ppl_test = y_rank_ppl[train_idx], y_rank_ppl[val_idx]
    y_rank_composite_train = y_rank_composite[train_idx]
    y_rank_composite_test = y_rank_composite[val_idx]
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

    rank_model_ppl, _rank_importance_ppl, rank_diag_ppl, _rank_ppl_meta = (
        _fit_rank_head(
            lgb,
            X_train=X_train,
            y_train_full=y_rank_ppl_train,
            w_train=w_train,
            X_val=X_test,
            y_val_full=y_rank_ppl_test,
            w_val=w_test,
            feature_names=feature_names,
            target_kind="ppl",
        )
    )
    (
        _rank_model_composite,
        _rank_importance_composite,
        rank_diag_composite,
        _rank_composite_meta,
    ) = _fit_rank_head(
        lgb,
        X_train=X_train,
        y_train_full=y_rank_composite_train,
        w_train=w_train,
        X_val=X_test,
        y_val_full=y_rank_composite_test,
        w_val=w_test,
        feature_names=feature_names,
        target_kind="composite",
    )

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
        "rank_heads": {
            "ppl": rank_diag_ppl,
            "composite": rank_diag_composite,
        },
        "rank_spearman": rank_diag_ppl.get("spearman", 0.0),
        "rank_spearman_ppl": rank_diag_ppl.get("spearman", 0.0),
        "rank_spearman_composite": rank_diag_composite.get("spearman", 0.0),
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
    db_path: str = RUNS_DB,
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
        induction_screening_auc = row.get("induction_screening_auc_500")
        if induction_screening_auc is None or not np.isfinite(
            float(induction_screening_auc)
        ):
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
        y_auc_list.append(float(induction_screening_auc))

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
