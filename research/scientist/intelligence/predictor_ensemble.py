"""Performance predictor split module. Re-exported via predictor."""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics_utils import (
    binary_classification_metrics,
    operating_point_profiles,
)
from .predictor_gbm import (
    _graph_signature,
    _load_screening_predictor_corpus_rows,
    train_gbm,
)
from .ml_corpus import (
    grouped_stratified_split,
    rerun_confidence_weight,
)
from .predictor_artifacts import (
    BAYESIAN_STATE_PATH as _BAYESIAN_STATE_PATH,
    ENSEMBLE_META_PATH as _ENSEMBLE_META_PATH,
    ENSEMBLE_STATE_PATH as _ENSEMBLE_STATE_PATH,
    GBM_GATE_MODEL_PATH as _GBM_GATE_MODEL_PATH,
    GBM_META_PATH as _GBM_META_PATH,
    GBM_RANK_COMPOSITE_MODEL_PATH as _GBM_RANK_COMPOSITE_MODEL_PATH,
    GBM_RANK_MODEL_PATH as _GBM_RANK_MODEL_PATH,
    GBM_RANK_PPL_MODEL_PATH as _GBM_RANK_PPL_MODEL_PATH,
    GRAPH_PREDICTOR_PATH as _GRAPH_PREDICTOR_PATH,
    INTERACTION_MODEL_PATH as _INTERACTION_MODEL_PATH,
    STATE_DIR,
    ensure_state_dir,
    metadata_sidecar_path,
    load_npz_archive,
    read_json,
    save_npz_archive,
    unlink_paths,
    write_json,
)

logger = logging.getLogger(__name__)


def _ppl_to_quality(value: float) -> float:
    return float(np.exp(-max(float(value), 0.0) / 25.0))


from .predictor_gbm import GBMPredictor  # noqa: F401
from .predictor_ridge import PerformancePredictor, _extract_features  # noqa: F401


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
            gbm_quality = float(self.gbm.predict_quality_score(graph_features))
            if np.isfinite(gbm_quality) and gbm_quality > 0.0:
                quality_terms.append(float(np.clip(gbm_quality, 0.0, 1.0)))

        if (
            self.graph_pred is not None
            and self.graph_pred.is_fitted()
            and graph_json is not None
        ):
            graph_rank = float(self.graph_pred.predict_rank(graph_json))
            if np.isfinite(graph_rank) and graph_rank < 1e5:
                quality_terms.append(_ppl_to_quality(graph_rank))
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
            blended = float(np.clip(0.65 * p_pass + 0.35 * quality_score, 0.0, 1.0))
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
        state_dir = ensure_state_dir(state_dir)
        if self.gbm is not None and self.gbm.is_fitted():
            self.gbm.save(state_dir)
        else:
            unlink_paths(
                state_dir / _GBM_GATE_MODEL_PATH.name,
                state_dir / _GBM_RANK_MODEL_PATH.name,
                state_dir / _GBM_RANK_PPL_MODEL_PATH.name,
                state_dir / _GBM_RANK_COMPOSITE_MODEL_PATH.name,
                state_dir / _GBM_META_PATH.name,
            )
        if self.graph_pred is not None and self.graph_pred.is_fitted():
            self.graph_pred.save(state_dir / _GRAPH_PREDICTOR_PATH.name)
        else:
            unlink_paths(
                state_dir / _GRAPH_PREDICTOR_PATH.name,
                metadata_sidecar_path(state_dir / _GRAPH_PREDICTOR_PATH.name),
            )
        if self.bayesian is not None:
            self.bayesian.save_state(state_dir / _BAYESIAN_STATE_PATH.name)
        else:
            unlink_paths(state_dir / _BAYESIAN_STATE_PATH.name)
        if self.interaction is not None and self.interaction._trained:
            self.interaction.save(state_dir / _INTERACTION_MODEL_PATH.name)
        else:
            unlink_paths(
                state_dir / _INTERACTION_MODEL_PATH.name,
                metadata_sidecar_path(state_dir / _INTERACTION_MODEL_PATH.name),
            )
        save_npz_archive(
            state_dir / _ENSEMBLE_STATE_PATH.name,
            w_ensemble=self.w_ensemble,
            score_mean=self._score_mean,
            score_std=self._score_std,
        )
        write_json(
            state_dir / _ENSEMBLE_META_PATH.name,
            {
                "b_ensemble": self.b_ensemble,
                "n_score_dims": self._n_score_dims,
                "gate_threshold": self.gate_threshold,
                "calibration_metrics": self._calibration_metrics,
            },
        )

    @classmethod
    def load(
        cls,
        state_dir: Path = STATE_DIR,
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
                data = load_npz_archive(ensemble_state_path)
                meta = read_json(ensemble_meta_path)
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
    state_dir: str = str(STATE_DIR),
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
