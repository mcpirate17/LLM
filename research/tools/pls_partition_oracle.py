#!/usr/bin/env python
"""PLS -> recursive-partition per-axis capability oracle + distance-novelty routing.

The corrected modeling stack (supersedes the flat-GBM screener for capability
prediction). For each capability axis independently:

  1. PCA  — diagnostic only: intrinsic dimensionality of the property space.
  2. PLS  — latent projection of the (unbounded) math+structure property space onto
            the axis label; uses ALL features, Q^2 plateaus instead of degrading as
            properties grow (the GBM "more features hurt" finding was a wrong-model
            artifact).
  3. DecisionTree on the PLS latent scores — interpretable nonlinear partition of the
            projected space into capability rules.

One model PER axis because induction (retrieval) and ar_curriculum (reasoning) load on
near-orthogonal property directions. A kNN distance-novelty scorer routes genuinely
novel designs to the probe instead of trusting a confident-wrong prediction.

Additive: persists to research/runtime/pls_partition_oracle/, never touches the
deployed GBM screener (research/runtime/capability_screener/).

Usage::
    python -m research.tools.pls_partition_oracle backtest   # A/B vs GBM, OOF + temporal
    python -m research.tools.pls_partition_oracle train      # fit + persist both axes
    python -m research.tools.pls_partition_oracle score      # decisions on probed winners
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _capability_map,
    _shrink,
    _template_map,
)
from research.tools.induction_predictor_foundation import (
    _fingerprint_timestamps,
    _fit_gbm,
    _ranking_metrics,
)
from research.tools.novelty_scorer import NoveltyScorer

logger = logging.getLogger(__name__)

# axis name -> (graph_runs column, capable threshold). Each axis is a near-orthogonal
# capability dimension (pairwise label spearman < 0.15), so each gets its own model.
#   induction      — in-context retrieval (rare-positive, ~3%).
#   ar_curriculum  — deep multi-stage AR/reasoning (rare-positive ~7%, data-starved ~1.3k).
#   ar_gate        — cheap AR go/no-go (well-populated ~4.5k, saturated; thr at the strong tail,
#                    not 0.5, since 73% clear 0.5). Weakly correlated w/ curriculum (rho 0.13) and
#                    induction (rho 0.05) — a distinct dimension, NOT a curriculum substitute.
#   nano_induction_nearest — brand-new nearest-token induction probe, populated for all s1
#                    (~8.5k, 99.6% feature-backed, rare-positive ~3%). Rank-distinct from the
#                    induction axis (label rho 0.14, < 0.15) — 50 nano-capable rows are
#                    induction-negative — so it earns its own model, not an induction substitute.
AXES: Dict[str, Tuple[str, float]] = {
    "induction": ("induction_screening_auc", 0.35),
    "ar_curriculum": ("ar_curriculum_auc_pair_final", 0.5),
    "ar_gate": ("ar_gate_score", 0.9),
    "nano_induction_nearest": ("nano_induction_nearest_max_accuracy", 0.5),
}

_STATE_DIR = Path("research/runtime/pls_partition_oracle")
_MODEL_PATH = _STATE_DIR / "oracle.joblib"
_META_PATH = _STATE_DIR / "oracle_meta.json"


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
@dataclass
class AxisCorpus:
    X: np.ndarray  # [n, n_features] math+structure properties
    names: List[str]
    fps: List[str]  # temporal order (oldest first)
    y: np.ndarray  # raw mean label
    clusters: List[str]  # template family per row


def _axis_corpus(db_path: str, axis_col: str, ts_map: Dict[str, float]) -> AxisCorpus:
    """Semantic features + raw label for one axis, sorted oldest-first (temporal split)."""
    cap = _capability_map(db_path, axis_col)
    tmpl = _template_map(db_path)
    fps = [fp for fp in cap if fp in ts_map]
    fps.sort(key=lambda fp: ts_map[fp])
    X, names, present = load_semantic_features(fps)  # preserves input order
    y = np.array([cap[fp] for fp in present], dtype=np.float64)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]
    return AxisCorpus(X, names, present, y, clusters)


# --------------------------------------------------------------------------- #
# the PLS -> partition cascade (one axis)
# --------------------------------------------------------------------------- #
@dataclass
class PLSPartition:
    scaler: StandardScaler
    pls: PLSRegression
    head: Any  # recursive partitioner on PLS latent scores (tree or gbm)
    feature_names: List[str]
    n_components: int
    head_kind: str

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        names: List[str],
        n_components: int,
        tree_depth: int,
        min_leaf: int,
        head_kind: str = "tree",
    ) -> "PLSPartition":
        scaler = StandardScaler().fit(X)
        Xz = np.asarray(scaler.transform(X))
        n_comp = max(1, min(n_components, Xz.shape[1], Xz.shape[0] - 1))
        pls = PLSRegression(n_components=n_comp).fit(Xz, y)
        scores = np.asarray(pls.transform(Xz))
        if head_kind == "gbm":
            head: Any = _fit_gbm(scores, y)
        elif head_kind == "bayes":
            # Bayesian linear head on the PLS latent scores — calibrated, low-variance
            # regularization that does not overfit the rare-positive tail.
            head = BayesianRidge().fit(scores, y)
        else:
            head = DecisionTreeRegressor(
                max_depth=tree_depth, min_samples_leaf=min_leaf, random_state=42
            ).fit(scores, y)
        return cls(scaler, pls, head, list(names), n_comp, head_kind)

    def _scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.pls.transform(self.scaler.transform(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Cascade prediction: scale -> PLS project -> partition head."""
        return np.asarray(self.head.predict(self._scores(X)), dtype=np.float64)

    def pls_predict(self, X: np.ndarray) -> np.ndarray:
        """PLS-only prediction (no tree) — the A/B ablation."""
        z = np.asarray(self.scaler.transform(X))
        return np.asarray(self.pls.predict(z), dtype=np.float64).ravel()


@dataclass
class RawGBM:
    """A raw-feature GBM with the same predict contract as PLSPartition (no PLS)."""

    model: Any
    feature_names: List[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            return np.asarray(self.model.predict(X), dtype=np.float64)


_PLS_HEAD_KIND = {"pls_gbm": "gbm", "pls_bayes": "bayes", "pls_tree": "tree"}


def _raw_estimator(kind: str, tree_depth: int, min_leaf: int) -> Any:
    """Unfit raw-feature estimator (no PLS projection) for a recursive-partition kind."""
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1,
        )
    if kind == "tree_raw":
        return DecisionTreeRegressor(
            max_depth=tree_depth, min_samples_leaf=min_leaf, random_state=42
        )
    raise ValueError(f"not a raw-estimator kind: {kind}")


def _fit_axis_model(
    kind: str, X: np.ndarray, y: np.ndarray, names: List[str], params: Dict[str, int]
) -> Any:
    """Fit one deployable axis model by kind (uniform .predict)."""
    if kind == "gbm":
        return RawGBM(_fit_gbm(X, y), list(names))
    if kind in ("extra_trees", "tree_raw"):
        est = _raw_estimator(kind, params["tree_depth"], params["min_leaf"]).fit(X, y)
        return RawGBM(est, list(names))
    return PLSPartition.fit(X, y, names, head_kind=_PLS_HEAD_KIND[kind], **params)


# --------------------------------------------------------------------------- #
# the deployed oracle: per-axis cascade + novelty routing
# --------------------------------------------------------------------------- #
@dataclass
class AxisOracle:
    models: Dict[str, Any]  # axis -> RawGBM | PLSPartition (uniform .predict)
    thresholds: Dict[str, float]
    scorer: NoveltyScorer
    feature_names: List[str]
    novelty_pctile_thr: float

    def evaluate_features(self, feats: Dict[str, float]) -> Dict[str, Any]:
        x = np.array(
            [[feats.get(n, 0.0) for n in self.feature_names]], dtype=np.float64
        )
        preds = {ax: round(float(m.predict(x)[0]), 4) for ax, m in self.models.items()}
        pctile = self.scorer.percentile(float(self.scorer.novelty(x)[0]))
        good = any(preds[ax] >= self.thresholds[ax] for ax in preds)
        if pctile >= self.novelty_pctile_thr:
            rec = "EXPLORE_PROBE"
        elif good:
            rec = "PREDICT_GOOD"
        else:
            rec = "PREDICT_BAD"
        return {
            "predicted": preds,
            "novelty_pctile": round(pctile, 3),
            "recommendation": rec,
        }

    def save(self, meta: Dict[str, Any]) -> None:
        import joblib

        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, _MODEL_PATH)
        _META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))

    @staticmethod
    def load() -> "AxisOracle":
        import joblib
        import sys

        # Older oracle artifacts were saved from ``python -m`` execution, so
        # joblib may resolve these classes against the caller's ``__main__``.
        # Register aliases before loading so runtime tools can consume the
        # artifact without retraining it.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            for cls in (AxisOracle, PLSPartition, RawGBM):
                setattr(main_mod, cls.__name__, cls)

        return joblib.load(_MODEL_PATH)


# --------------------------------------------------------------------------- #
# A/B model zoo (same train/test contract for every candidate)
# --------------------------------------------------------------------------- #
def _fit_predict(
    kind: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    names: List[str],
    n_components: int,
    tree_depth: int,
    min_leaf: int,
) -> np.ndarray:
    if kind == "gbm":
        return np.asarray(_fit_gbm(Xtr, ytr).predict(Xte), dtype=np.float64)
    if kind in ("tree_raw", "extra_trees"):
        est = _raw_estimator(kind, tree_depth, min_leaf).fit(Xtr, ytr)
        return np.asarray(est.predict(Xte), dtype=np.float64)
    head_kind = _PLS_HEAD_KIND[kind] if kind != "pls_only" else "tree"
    model = PLSPartition.fit(
        Xtr, ytr, names, n_components, tree_depth, min_leaf, head_kind=head_kind
    )
    if kind == "pls_only":
        return model.pls_predict(Xte)
    if kind in ("pls_tree", "pls_gbm", "pls_bayes"):
        return model.predict(Xte)
    raise ValueError(f"unknown model kind: {kind}")


# Full A/B zoo: raw GBM, raw recursive-partition (single tree + ExtraTrees ensemble),
# PLS-only linear, and PLS-projected heads (tree / GBM / Bayesian). One contract per kind.
_KINDS = (
    "gbm",
    "tree_raw",
    "extra_trees",
    "pls_only",
    "pls_tree",
    "pls_gbm",
    "pls_bayes",
)


def _metrics(pred: np.ndarray, y_raw: np.ndarray, thr: float) -> Dict[str, Any]:
    m = _ranking_metrics(pred, y_raw, thr)
    return {
        "roc": round(m["roc_auc_gt_thr"], 4) if m["roc_auc_gt_thr"] else None,
        "spearman": round(m["spearman_rho"], 4),
    }


def _oof_ab(
    cp: AxisCorpus, thr: float, shrink_f: float, k: int, params: Dict[str, int]
) -> Dict[str, Any]:
    """Honest out-of-fold A/B: each row predicted by a model that never saw it."""
    n = len(cp.fps)
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(n), k)
    oof = {kind: np.zeros(n, dtype=np.float64) for kind in _KINDS}
    for f in range(k):
        held = folds[f]
        rest = np.concatenate([folds[j] for j in range(k) if j != f])
        mask = np.zeros(n, dtype=bool)
        mask[rest] = True
        ys = _shrink(cp.y, cp.clusters, mask, shrink_f)  # train-only stats
        for kind in _KINDS:
            oof[kind][held] = _fit_predict(
                kind, cp.X[rest], ys[rest], cp.X[held], cp.names, **params
            )
    return {kind: _metrics(oof[kind], cp.y, thr) for kind in _KINDS}


def _temporal_ab(
    cp: AxisCorpus, thr: float, shrink_f: float, params: Dict[str, int]
) -> Dict[str, Any]:
    """Forward-in-time A/B: train on oldest 80%, score newest 20% vs raw label."""
    n = len(cp.fps)
    cut = int(n * 0.8)
    mask = np.zeros(n, dtype=bool)
    mask[:cut] = True
    ys = _shrink(cp.y, cp.clusters, mask, shrink_f)
    out: Dict[str, Any] = {}
    for kind in _KINDS:
        pred = _fit_predict(kind, cp.X[:cut], ys[:cut], cp.X[cut:], cp.names, **params)
        out[kind] = _metrics(pred, cp.y[cut:], thr)
    return out


def _leave_family_out_ab(
    cp: AxisCorpus, thr: float, shrink_f: float, params: Dict[str, int], k: int = 5
) -> Dict[str, Any]:
    """OOD-honest A/B: hold out WHOLE template families (GroupKFold by cluster).

    Each row is predicted exactly once, by a model that never saw any graph from its
    template family — the closest in-corpus proxy for scoring a novel, off-manifold
    design. This (not random/in-dist holdout) is the go/no-go metric for the screener.
    """
    n = len(cp.fps)
    groups = np.asarray(cp.clusters)
    n_groups = int(len({*cp.clusters}))
    splits = min(k, n_groups)
    if splits < 2:
        return {kind: {"roc": None, "spearman": 0.0} for kind in _KINDS}
    gkf = GroupKFold(n_splits=splits)
    oof = {kind: np.zeros(n, dtype=np.float64) for kind in _KINDS}
    for rest, held in gkf.split(cp.X, cp.y, groups):
        mask = np.zeros(n, dtype=bool)
        mask[rest] = True
        ys = _shrink(cp.y, cp.clusters, mask, shrink_f)  # train-family stats only
        for kind in _KINDS:
            oof[kind][held] = _fit_predict(
                kind, cp.X[rest], ys[rest], cp.X[held], cp.names, **params
            )
    return {kind: _metrics(oof[kind], cp.y, thr) for kind in _KINDS}


def _pca_dim(X: np.ndarray) -> Dict[str, int]:
    pca = PCA().fit(np.asarray(StandardScaler().fit_transform(X)))
    cum = np.cumsum(pca.explained_variance_ratio_)
    return {
        "n_features": int(X.shape[1]),
        "dim_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
        "dim_95pct_var": int(np.searchsorted(cum, 0.95) + 1),
    }


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def backtest(
    db_path: str, shrink_f: float, k: int, params: Dict[str, int]
) -> Dict[str, Any]:
    ts_map = _fingerprint_timestamps(db_path)
    report: Dict[str, Any] = {
        "shrink_f": shrink_f,
        "oof_folds": k,
        "pls_params": params,
        "axes": {},
    }
    for axis, (col, thr) in AXES.items():
        cp = _axis_corpus(db_path, col, ts_map)
        n = len(cp.fps)
        n_cap = int((cp.y > thr).sum())
        if n < 100 or not (0 < n_cap < n):
            report["axes"][axis] = {"n_graphs": n, "n_capable": n_cap, "skipped": True}
            continue
        report["axes"][axis] = {
            "n_graphs": n,
            "n_capable": n_cap,
            "threshold": thr,
            "pca": _pca_dim(cp.X),
            "out_of_fold": _oof_ab(cp, thr, shrink_f, k, params),
            "leave_family_out": _leave_family_out_ab(cp, thr, shrink_f, params),
            "temporal_80_20": _temporal_ab(cp, thr, shrink_f, params),
        }
    report["note"] = (
        "pls_tree is the deployed cascade; gbm is the baseline screener model. "
        "ROC scored vs RAW held-out label; target shrunk train-only."
    )
    return report


_CANDIDATES = ("gbm", "extra_trees", "pls_gbm", "pls_bayes")


def train(
    db_path: str,
    shrink_f: float,
    params: Dict[str, int],
    novelty_k: int,
    novelty_pctile_thr: float,
) -> Dict[str, Any]:
    """Per axis: leave-family-out-select the best regressor among _CANDIDATES, fit, persist.

    Self-tuning by design — as properties grow the PLS-family may overtake the raw
    GBM on the induction axis too; re-running train re-selects. Novelty routing is
    the axis-independent piece that handles the genuinely-unknown.
    """
    ts_map = _fingerprint_timestamps(db_path)
    models: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}
    selected: Dict[str, Dict[str, Any]] = {}
    trained: Dict[str, int] = {}
    novelty_corpus: Optional[AxisCorpus] = None
    for axis, (col, thr) in AXES.items():
        cp = _axis_corpus(db_path, col, ts_map)
        if len(cp.fps) < 100 or not (0 < (cp.y > thr).sum() < len(cp.fps)):
            continue
        # OOD-honest self-selection: pick the kind that ranks held-out template
        # families best (not random/in-dist OOF — that rewards regress-to-familiar).
        lfo = _leave_family_out_ab(cp, thr, shrink_f, params)
        best = max(_CANDIDATES, key=lambda k: lfo[k]["roc"] or 0.0)
        ys = _shrink(cp.y, cp.clusters, np.ones(len(cp.fps), dtype=bool), shrink_f)
        models[axis] = _fit_axis_model(best, cp.X, ys, cp.names, params)
        thresholds[axis] = thr
        trained[axis] = len(cp.fps)
        selected[axis] = {
            "kind": best,
            "leave_family_out_roc": {k: lfo[k]["roc"] for k in _CANDIDATES},
        }
        if novelty_corpus is None or len(cp.fps) > len(novelty_corpus.fps):
            novelty_corpus = cp
    if not models or novelty_corpus is None:
        raise SystemExit("no axis had enough labeled graphs to train")
    names = novelty_corpus.names
    scorer = NoveltyScorer(novelty_corpus.X, names, k=novelty_k)
    oracle = AxisOracle(models, thresholds, scorer, names, novelty_pctile_thr)
    meta = {
        "axes_trained": trained,
        "selected_per_axis": selected,
        "thresholds": thresholds,
        "n_features": len(names),
        "pls_params": params,
        "novelty_k": novelty_k,
        "novelty_pctile_thr": novelty_pctile_thr,
        "novelty_corpus_axis": max(trained, key=lambda a: trained[a]),
    }
    oracle.save(meta)
    logger.info("saved oracle -> %s", _MODEL_PATH)
    return {"persisted": str(_MODEL_PATH), **meta}


def score(db_path: str, probes_file: str) -> Dict[str, Any]:
    from research.tools.graph_semantic_features import GraphSemanticExtractor
    from research.tools.probe_novel_candidates import _collect_pool

    oracle = AxisOracle.load()
    pf = Path(probes_file)
    if not pf.exists():
        return {"error": f"{probes_file} missing", "axes": list(oracle.models)}
    probes = [
        r
        for r in json.loads(pf.read_text())["results"]
        if r.get("actual_induction_auc") is not None
    ]
    cand = {c["fingerprint"]: c for c in _collect_pool(db_path, 600, 12000, 3_000_000)}
    ext = GraphSemanticExtractor(db_path)
    rows: List[Dict[str, Any]] = []
    for r in probes:
        c = cand.get(r["fingerprint"])
        if c is None:
            continue
        decision = oracle.evaluate_features(ext.features(c["graph"].to_dict()["nodes"]))
        rows.append(
            {
                "fingerprint": r["fingerprint"],
                "actual_induction": round(float(r["actual_induction_auc"]), 4),
                **decision,
                "novel_mixers": r.get("novel_mixers", []),
            }
        )
    rows.sort(key=lambda x: -x["actual_induction"])
    capable = [x for x in rows if x["actual_induction"] > 0.35]
    return {
        "n_probed": len(rows),
        "summary": {
            "n_capable": len(capable),
            "capable_routed_to_probe": sum(
                1 for x in capable if x["recommendation"] == "EXPLORE_PROBE"
            ),
            "capable_not_predicted_bad": sum(
                1 for x in capable if x["recommendation"] != "PREDICT_BAD"
            ),
        },
        "decisions": rows,
    }


def _params(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "n_components": args.pls_components,
        "tree_depth": args.tree_depth,
        "min_leaf": args.min_leaf,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["backtest", "train", "score"])
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--oof-folds", type=int, default=5)
    p.add_argument("--pls-components", type=int, default=20)
    p.add_argument("--tree-depth", type=int, default=4)
    p.add_argument("--min-leaf", type=int, default=20)
    p.add_argument("--novelty-k", type=int, default=10)
    p.add_argument("--novelty-pctile", type=float, default=0.9)
    p.add_argument("--probes", default="research/reports/novel_candidate_probes.json")
    p.add_argument("--out", default="research/reports/pls_partition_backtest.json")
    args = p.parse_args()

    if args.mode == "backtest":
        report = backtest(args.db, args.shrink, args.oof_folds, _params(args))
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        logger.info("wrote %s", out_path)
    elif args.mode == "train":
        report = train(
            args.db,
            args.shrink,
            _params(args),
            args.novelty_k,
            args.novelty_pctile,
        )
    else:
        report = score(args.db, args.probes)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
