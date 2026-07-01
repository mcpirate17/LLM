"""Learned surrogate over the fab ledger (WS-3).

Replaces ``property_miner.predicted_lift`` — the geometric mean of per-axis
marginal pass rates, which *assumes axis independence* (weakness #2) — with a
gradient-boosted model that captures axis interactions natively (trees split on
combinations, so a pair like ``tropical + has_state`` is learnable where the
marginal model multiplies two independent rates).

Read-only over ``catalog/ledger.jsonl``. Trains two targets:
  - ``composite_score`` (regression) — the acquisition signal; a median GBM plus
    an upper-quantile GBM give the (mean, optimistic-bound) pair UCB needs.
  - ``promoted`` (classification) — reported as AUC; sparse (≈10/576), advisory.

The K-fold report (``catalog/surrogate_report.json``) carries the WS-3 acceptance
check: out-of-fold top-K recall of eventual promotions, surrogate vs the marginal
baseline (the same independence model ``predicted_lift`` encodes). If the surrogate
does not beat the baseline, ``acceptance_passed`` is False and selection stays legacy.

Features per ledger entry: one-hot ``category`` + ``synthesis_kind`` + a curated set
of ``math_axes`` (value-as-category, missing → its own level) + multi-hot
``math_knobs``. Tree models supply interactions, so no explicit pairwise blow-up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._stats import spearman
from .ledger import DEFAULT_LEDGER_PATH, write_json_report
from .ledger import read_last_grades_and_statuses as _read_ledger
from .math_sweep_features import math_sweep_surrogate_features

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = _REPO / "component_fab" / "catalog" / "surrogate_report.json"

# math_axes keys with enough ledger coverage to be worth encoding (the long tail
# of rare keys adds noise + sparsity for no signal). Mirrors the high-coverage
# axes observed in the ledger.
FEATURE_AXES: tuple[str, ...] = (
    "op_algebraic_space",
    "op_geometric_receptive_field",
    "op_dynamical_has_state",
    "op_dynamical_memory_length_class",
    "op_activation_sparsity_pattern",
    "op_spectral_preferred_basis",
    "op_routing_kind",
    "op_block_template",
    "op_math_family",
    "op_sparse_matrix_pattern",
    "op_calculus_operator",
    "op_linear_algebra_structure",
)
UPPER_QUANTILE = 0.9
DEFAULT_RECALL_KS: tuple[int, ...] = (25, 50, 100)


@dataclass(slots=True)
class TrainingRow:
    proposal_id: str
    features: dict[str, float]
    composite: float
    promoted: int


@dataclass(slots=True)
class SurrogateReport:
    n_rows: int
    n_promoted: int
    n_features: int
    composite_spearman_oof: float
    composite_r2_oof: float
    promoted_auc_surrogate: float | None
    promoted_auc_marginal: float | None
    recall_at_k: dict[int, dict[str, float]]  # k -> {"surrogate":, "marginal":}
    acceptance_passed: bool
    findings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Ledger ingest + feature encoding
# --------------------------------------------------------------------------- #
def features_from_metadata(
    *,
    category: str,
    synthesis_kind: str,
    math_axes: dict[str, Any],
    math_knobs: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build the one-hot/multi-hot feature dict for a single candidate.

    Shared by training (ledger rows) and inference (live ProposalSpecs) so the
    encoding never drifts between them.
    """
    feat: dict[str, float] = {
        f"category={category}": 1.0,
        f"synthesis_kind={synthesis_kind}": 1.0,
    }
    for axis in FEATURE_AXES:
        feat[f"{axis}={math_axes.get(axis)}"] = 1.0
    for knob in math_knobs:
        if knob:
            feat[f"knob={knob}"] = 1.0
    feat.update(math_sweep_surrogate_features(metadata or math_axes))
    return feat


def features_for_spec(spec: Any) -> dict[str, float]:
    """Encode a live ProposalSpec identically to a ledger training row."""
    axes = dict(spec.math_axes or {})
    knobs = [p for p in str(axes.get("op_math_knobs") or "").split("+") if p]
    return features_from_metadata(
        category=str(spec.category or ""),
        synthesis_kind=str(spec.synthesis_kind or ""),
        math_axes=axes,
        math_knobs=knobs,
    )


def _training_rows(
    last_grade: dict[str, dict[str, Any]], last_status: dict[str, str]
) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for pid, grade in last_grade.items():
        meta = grade.get("metadata") or {}
        axes = meta.get("math_axes") or {}
        knobs = meta.get("math_knobs") or []

        # WS-2 Relabeling: prefer the honest frontier-relative delta from the
        # deep probe if it exists. Otherwise fallback to the nano composite.
        # We also want to penalize candidates that fail mechanistic probes.
        composite = float(grade.get("composite_score") or 0.0)

        # If the candidate failed mechanistic probes (e.g. routing collapse),
        # we demote its score.
        # Note: these fields are added by our new validator/mechanism.py
        if meta.get("routing_entropy_mean", 1.0) < 0.05:
            composite *= 0.5
        if meta.get("state_degeneracy", 0.0) > 0.9:
            composite *= 0.5
        # A candidate whose forward measures as a convex token-averager has
        # reconverged on the softmax basin — the mission's pathology to steer
        # away from. Demote so search prefers genuinely novel geometry.
        if meta.get("softmax_twin_score", 0.0) > 0.85:
            composite *= 0.5

        frontier_delta = meta.get("deep_probe_mean_delta")
        slope = meta.get("slope")
        if frontier_delta is not None:
            # Shift the delta into a similar range as the 0-1 composite
            # (frontier deltas are typically small, e.g. -0.1 to +0.1).
            # We want 'beats frontier' (delta > 0) to be a strong signal.
            target = 0.6 + float(frontier_delta) * 2.0

            # Incorporate slope: a positive slope is a strong indicator of
            # scaling potential. (typical slopes are small, e.g. 0.05).
            if slope is not None:
                target += float(slope) * 2.0
        else:
            target = composite

        rows.append(
            TrainingRow(
                proposal_id=pid,
                features=features_from_metadata(
                    category=str(grade.get("category") or ""),
                    synthesis_kind=str(grade.get("synthesis_kind") or ""),
                    math_axes=axes,
                    math_knobs=knobs,
                    metadata=meta,
                ),
                composite=max(0.0, min(1.0, target)),
                promoted=1 if last_status.get(pid) == "promoted" else 0,
            )
        )
    return rows


def _vectorize(
    rows: Sequence[TrainingRow],
) -> tuple[np.ndarray, list[str]]:
    """Dict features -> dense one-hot matrix via DictVectorizer."""
    from sklearn.feature_extraction import DictVectorizer

    dv = DictVectorizer(sparse=False)
    X = np.asarray(dv.fit_transform([r.features for r in rows]), dtype=float)
    return X, list(dv.get_feature_names_out())


# --------------------------------------------------------------------------- #
# Marginal baseline (the predicted_lift / axis-independence model)
# --------------------------------------------------------------------------- #
def _marginal_scores(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    """Additive per-feature mean-composite model — the independence assumption.

    Each active feature contributes the train-set mean composite of rows that
    had it; a candidate's score is the mean over its active features. This is the
    same information the surrogate sees, minus interactions — so the head-to-head
    isolates the value of modeling interactions.
    """
    global_mean = float(y_train.mean()) if len(y_train) else 0.0
    n_features = X_train.shape[1]
    feat_mean = np.full(n_features, global_mean, dtype=float)
    for j in range(n_features):
        mask = X_train[:, j] > 0
        if mask.any():
            feat_mean[j] = float(y_train[mask].mean())
    scores = np.empty(X_eval.shape[0], dtype=float)
    for i in range(X_eval.shape[0]):
        active = X_eval[i] > 0
        scores[i] = float(feat_mean[active].mean()) if active.any() else global_mean
    return scores


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _recall_at_k(scores: np.ndarray, promoted: np.ndarray, k: int) -> float:
    total = int(promoted.sum())
    if total == 0:
        return 0.0
    topk = np.argsort(scores)[::-1][:k]
    return float(promoted[topk].sum()) / total


# --------------------------------------------------------------------------- #
# K-fold report
# --------------------------------------------------------------------------- #
def _fit_composite_models(X: np.ndarray, y: np.ndarray):
    """Return (median_model, upper_quantile_model) fit on (X, y)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    median = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=8,
        random_state=0,
    ).fit(X, y)
    upper = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=UPPER_QUANTILE,
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=8,
        random_state=0,
    ).fit(X, y)
    return median, upper


def compute_surrogate_report(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    *,
    n_folds: int = 5,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> SurrogateReport:
    """K-fold out-of-fold evaluation + the WS-3 acceptance comparison."""

    last_grade, last_status = _read_ledger(Path(ledger_path))
    rows = _training_rows(last_grade, last_status)
    if len(rows) < n_folds * 2:
        return SurrogateReport(
            n_rows=len(rows),
            n_promoted=sum(r.promoted for r in rows),
            n_features=0,
            composite_spearman_oof=0.0,
            composite_r2_oof=0.0,
            promoted_auc_surrogate=None,
            promoted_auc_marginal=None,
            recall_at_k={},
            acceptance_passed=False,
            findings=[f"insufficient ledger rows ({len(rows)}) for {n_folds}-fold CV"],
        )
    X, feat_names = _vectorize(rows)
    y = np.array([r.composite for r in rows], dtype=float)
    promoted = np.array([r.promoted for r in rows], dtype=int)

    oof_surrogate, oof_marginal = _composite_oof(X, y, n_folds)
    spearman_oof = spearman(oof_surrogate, y)
    r2 = _r2(y, oof_surrogate)

    # Acceptance ranks by the PROMOTION target (the sparse end-goal), not the
    # composite proxy: surrogate = GBM promotion probability, marginal = additive
    # per-feature promotion rate (the predicted_lift / independence analogue),
    # both out-of-fold. Fall back to composite ranking only when positives are
    # too few to train a promotion classifier.
    surr_p, marg_p = _promotion_oof(X, promoted, n_folds)
    if surr_p is not None and marg_p is not None:
        rank_surrogate, rank_marginal, rank_target = surr_p, marg_p, "promotion"
        auc_s, auc_m = _safe_auc(promoted, surr_p), _safe_auc(promoted, marg_p)
    else:
        rank_surrogate, rank_marginal, rank_target = (
            oof_surrogate,
            oof_marginal,
            "composite",
        )
        auc_s = auc_m = None
    recall = {
        int(k): {
            "surrogate": round(_recall_at_k(rank_surrogate, promoted, k), 4),
            "marginal": round(_recall_at_k(rank_marginal, promoted, k), 4),
        }
        for k in recall_ks
    }

    # Acceptance: surrogate promotion AUC strictly beats the marginal/independence
    # AUC (a whole-ranking metric, stable where recall@K is noisy at ~10 positives).
    # When AUC is unavailable (too few positives), fall back to top-K recall.
    primary_k = sorted(recall_ks)[len(recall_ks) // 2]
    if auc_s is not None and auc_m is not None:
        passed = auc_s > auc_m
    else:
        passed = recall[primary_k]["surrogate"] > recall[primary_k]["marginal"]
    findings = _surrogate_findings(
        spearman_oof, r2, recall, auc_s, auc_m, passed, primary_k, rank_target
    )
    return SurrogateReport(
        n_rows=len(rows),
        n_promoted=int(promoted.sum()),
        n_features=len(feat_names),
        composite_spearman_oof=round(spearman_oof, 4),
        composite_r2_oof=round(r2, 4),
        promoted_auc_surrogate=round(auc_s, 4) if auc_s is not None else None,
        promoted_auc_marginal=round(auc_m, 4) if auc_m is not None else None,
        recall_at_k=recall,
        acceptance_passed=passed,
        findings=findings,
    )


def _composite_oof(
    X: np.ndarray, y: np.ndarray, n_folds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold composite predictions: (GBM, marginal/independence)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    oof_surrogate = np.zeros(len(y), dtype=float)
    oof_marginal = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(X):
        reg = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=8,
            random_state=0,
        ).fit(X[train_idx], y[train_idx])
        oof_surrogate[test_idx] = reg.predict(X[test_idx])
        oof_marginal[test_idx] = _marginal_scores(
            X[train_idx], y[train_idx], X[test_idx]
        )
    return oof_surrogate, oof_marginal


def _promotion_oof(
    X: np.ndarray, promoted: np.ndarray, n_folds: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Out-of-fold promotion scores: (GBM probability, marginal rate).

    Returns ``(None, None)`` when positives are too few to stratify — the caller
    then falls back to composite ranking. The marginal arm is the additive
    per-feature promotion rate (axis-independence), the direct predicted_lift
    analogue for the promotion target.
    """
    if int(promoted.sum()) < n_folds:
        return None, None
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    surrogate = np.zeros(len(promoted), dtype=float)
    marginal = np.zeros(len(promoted), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(X, promoted):
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=8,
            random_state=0,
        ).fit(X[train_idx], promoted[train_idx])
        surrogate[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        marginal[test_idx] = _marginal_scores(
            X[train_idx], promoted[train_idx].astype(float), X[test_idx]
        )
    return surrogate, marginal


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _surrogate_findings(
    spearman: float,
    r2: float,
    recall: dict[int, dict[str, float]],
    auc_s: float | None,
    auc_m: float | None,
    passed: bool,
    primary_k: int,
    rank_target: str,
) -> list[str]:
    findings: list[str] = []
    s = recall[primary_k]["surrogate"]
    m = recall[primary_k]["marginal"]
    verdict = "PASS" if passed else "FAIL"
    if auc_s is not None and auc_m is not None:
        basis = (
            f"promotion AUC surrogate={auc_s:.3f} vs marginal/independence "
            f"={auc_m:.3f} (Δ{auc_s - auc_m:+.3f})"
        )
    else:
        basis = f"top-K recall at K={primary_k}: surrogate {s:.0%} vs marginal {m:.0%}"
    findings.append(
        f"ACCEPTANCE {verdict} (ranked by {rank_target}): {basis}. "
        + (
            "Surrogate beats the axis-independence baseline → safe to flip "
            "selection to surrogate."
            if passed
            else "Surrogate does not beat the marginal baseline → keep selection=legacy."
        )
    )
    if auc_s is not None and auc_m is not None and 0.0 < (auc_s - auc_m) < 0.02:
        findings.append(
            "CAVEAT: the AUC margin is thin and may sit within CV noise at this few "
            "positives — interactions add only modest promotion-ranking value here. "
            "The surrogate's firmer advantage is the per-candidate (mean, upper-quantile) "
            "it gives UCB acquisition; the marginal independence model has no uncertainty."
        )
    findings.append(
        "recall of eventual promotions by K: "
        + ", ".join(
            f"@{k}: surr {v['surrogate']:.0%}/marg {v['marginal']:.0%}"
            for k, v in sorted(recall.items())
        )
    )
    findings.append(
        f"composite OOF Spearman={spearman:.3f}, R2={r2:.3f} "
        f"({'usable' if spearman >= 0.3 else 'weak'} acquisition ranking signal)."
    )
    return findings


def write_surrogate_report(
    report: SurrogateReport, output_path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    payload = asdict(report)
    payload["recall_at_k"] = {str(k): v for k, v in report.recall_at_k.items()}
    return write_json_report(payload, output_path)


# --------------------------------------------------------------------------- #
# Fitted approximant for acquisition (proposer/acquisition.py)
# --------------------------------------------------------------------------- #
class MeanFieldApproximant:
    """Fitted composite predictor exposing (median, upper_quantile) per candidate."""

    def __init__(self, feature_names: list[str], median_model: Any, upper_model: Any):
        self._feature_names = feature_names
        self._index = {name: i for i, name in enumerate(feature_names)}
        self._median = median_model
        self._upper = upper_model

    @classmethod
    def fit(
        cls, ledger_path: Path | str = DEFAULT_LEDGER_PATH
    ) -> "MeanFieldApproximant | None":
        last_grade, last_status = _read_ledger(Path(ledger_path))
        rows = _training_rows(last_grade, last_status)
        if len(rows) < 10:
            return None
        X, names = _vectorize(rows)
        y = np.array([r.composite for r in rows], dtype=float)
        median, upper = _fit_composite_models(X, y)
        return cls(names, median, upper)

    def _encode(self, features: dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self._feature_names), dtype=float)
        for name, val in features.items():
            j = self._index.get(name)
            if j is not None:
                vec[j] = val
        return vec.reshape(1, -1)

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        """Return (median, upper_quantile) predicted composite for one candidate."""
        x = self._encode(features)
        return float(self._median.predict(x)[0]), float(self._upper.predict(x)[0])
