#!/usr/bin/env python
"""Train and evaluate a lightweight predictor for ar_curriculum_auc_pair_final.

Standalone — does NOT touch the project's main predictor stack
(predictor_gbm.py, predictor_ensemble.py). This script exists to answer the
question:

  "Given the cheap upstream features we already collect on every candidate,
   how well can we predict the ar_curriculum AUC without running the probe?"

If Spearman ρ on hold-out ≥ 0.7, the predictor can triage which candidates
need the actual probe vs which we can predict. If < 0.7, more backfill is
needed before the predictor is reliable.

Output:
  research/runtime/ar_curriculum_experiment/predictor_<run_id>.json
  research/runtime/ar_curriculum_experiment/predictor_<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from research.scientist.notebook import LabNotebook
from research.tools.ar_curriculum_trends import UPSTREAM_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "research/runtime/ar_curriculum_experiment"
DEFAULT_DB = REPO_ROOT / "research/runs.db"
TARGETS = ("ar_curriculum_auc_pair_final", "ar_curriculum_s0_retention")


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    rx = _ranks(xs.tolist())
    ry = _ranks(ys.tolist())
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(xs: list[float]) -> list[float]:
    paired = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg
        i = j + 1
    return ranks


def load_rows(db: Path) -> list[dict[str, Any]]:
    nb = LabNotebook(str(db), read_only=True)
    feature_cols = ", ".join(f"pr.{f}" for f in UPSTREAM_FEATURES)
    target_cols = ", ".join(f"pr.{t}" for t in TARGETS)
    sql = f"""
        SELECT pr.graph_fingerprint, l.tier, l.composite_score,
               {feature_cols}, {target_cols}
        FROM program_results pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.ar_curriculum_auc_pair_final IS NOT NULL
    """
    rows = [
        dict(r) if hasattr(r, "keys") else {k: r[i] for i, k in enumerate(r.keys())}
        for r in nb.conn.execute(sql).fetchall()
    ]
    nb.close()
    return rows


def to_matrix(
    rows: list[dict[str, Any]], target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y) with median imputation. Drop all-null feature columns."""
    candidate_features: list[str] = list(UPSTREAM_FEATURES)
    populated = {
        f: sum(1 for r in rows if r.get(f) is not None) for f in candidate_features
    }
    feature_names = [f for f in candidate_features if populated[f] > 0]
    medians: dict[str, float] = {}
    for f in feature_names:
        vals: list[float] = []
        for r in rows:
            v = r.get(f)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        medians[f] = st.median(vals) if vals else 0.0
    X: list[list[float]] = []
    y: list[float] = []
    for r in rows:
        if r.get(target) is None:
            continue
        feats = []
        for f in feature_names:
            v = r.get(f)
            if v is None:
                feats.append(medians[f])
                continue
            try:
                feats.append(float(v))
            except (TypeError, ValueError):
                feats.append(medians[f])
        X.append(feats)
        y.append(float(r[target]))
    return np.array(X), np.array(y), feature_names


def _make_model(kind: str, seed: int):
    if kind == "gbm":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.85,
            random_state=seed,
        )
    if kind == "gbm_deep":
        return GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            min_samples_leaf=3,
            subsample=0.75,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "tree":
        return DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=10,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def _eval_one_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    kind: str,
    seed: int,
    n_folds: int,
) -> dict[str, Any]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_spearmans: list[float] = []
    test_spearmans: list[float] = []
    test_maes: list[float] = []
    importances_acc = np.zeros(len(feature_names))
    for fold_seed, (tr_idx, te_idx) in enumerate(kf.split(X)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        model = _make_model(kind, seed + fold_seed)
        model.fit(Xtr, ytr)
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
        train_spearmans.append(_spearman(np.array(pred_tr), ytr))
        test_spearmans.append(_spearman(np.array(pred_te), yte))
        test_maes.append(float(mean_absolute_error(yte, pred_te)))
        if hasattr(model, "feature_importances_"):
            importances_acc += np.asarray(model.feature_importances_)
    importances = sorted(
        zip(feature_names, (importances_acc / max(n_folds, 1)).tolist()),
        key=lambda p: p[1],
        reverse=True,
    )
    return {
        "model": kind,
        "spearman_train_mean": round(float(np.mean(train_spearmans)), 3),
        "spearman_train_std": round(float(np.std(train_spearmans)), 3),
        "spearman_test_mean": round(float(np.mean(test_spearmans)), 3),
        "spearman_test_std": round(float(np.std(test_spearmans)), 3),
        "mae_test_mean": round(float(np.mean(test_maes)), 4),
        "fold_test_spearmans": [round(float(s), 3) for s in test_spearmans],
        "feature_importances": [
            {"feature": f, "importance": round(float(imp), 4)} for f, imp in importances
        ],
    }


def evaluate_target(
    rows: list[dict[str, Any]],
    target: str,
    *,
    seed: int,
    test_frac: float,
    n_folds: int = 5,
    model_kinds: tuple[str, ...] = ("gbm", "gbm_deep", "rf", "tree"),
) -> dict[str, Any]:
    X, y, feature_names = to_matrix(rows, target)
    if len(X) < 30:
        return {"target": target, "status": "too_few_samples", "n": len(X)}
    per_model = [
        _eval_one_model(X, y, feature_names, kind=k, seed=seed, n_folds=n_folds)
        for k in model_kinds
    ]
    best = max(per_model, key=lambda d: d["spearman_test_mean"])
    return {
        "target": target,
        "n_total": len(X),
        "n_folds": n_folds,
        "y_mean": round(float(np.mean(y)), 3),
        "y_std": round(float(np.std(y)), 3),
        "models": per_model,
        "best_model": best["model"],
        "best_spearman_test_mean": best["spearman_test_mean"],
        "best_spearman_test_std": best["spearman_test_std"],
        "status": "ok",
    }


def write_report(
    results: dict[str, Any], out_dir: Path, run_id: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"predictor_{run_id}.json"
    md_path = out_dir / f"predictor_{run_id}.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines: list[str] = [
        f"# AR curriculum predictor — {run_id}",
        "",
        f"n_rows_with_curriculum = {results['n_rows']}",
        "",
        "Multi-model 5-fold CV. Models: gbm (default GBM), gbm_deep (GBM with deeper trees), "
        "rf (random forest), tree (single decision tree depth=5).",
        "Decision threshold: best_model Spearman ρ ≥ 0.7 → predictor reliable enough to triage; < 0.7 → run more backfill.",
        "",
        "## Per-target best model summary",
        "",
        "| target | n | best model | best Spearman test | y mean ± std | verdict |",
        "|---|---:|---|---|---|---|",
    ]
    for r in results.get("targets", []):
        if r.get("status") != "ok":
            lines.append(
                f"| {r['target']} | — | — | — | — | {r.get('status', 'fail')} |"
            )
            continue
        verdict = (
            "RELIABLE" if r["best_spearman_test_mean"] >= 0.7 else "MORE DATA NEEDED"
        )
        lines.append(
            f"| {r['target']} | {r['n_total']} | {r['best_model']} | "
            f"{r['best_spearman_test_mean']:+.3f} ± {r['best_spearman_test_std']:.3f} | "
            f"{r['y_mean']:.3f} ± {r['y_std']:.3f} | {verdict} |"
        )

    for r in results.get("targets", []):
        if r.get("status") != "ok":
            continue
        lines += [
            "",
            f"## {r['target']} — model comparison (5-fold CV)",
            "",
            "| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |",
            "|---|---|---|---|---:|",
        ]
        for m in r["models"]:
            per_fold = ", ".join(f"{s:+.2f}" for s in m["fold_test_spearmans"])
            lines.append(
                f"| {m['model']} | "
                f"{m['spearman_train_mean']:+.3f} ± {m['spearman_train_std']:.3f} | "
                f"{m['spearman_test_mean']:+.3f} ± {m['spearman_test_std']:.3f} | "
                f"{per_fold} | {m['mae_test_mean']:.3f} |"
            )
        # Use best model's feature importance
        best = max(r["models"], key=lambda d: d["spearman_test_mean"])
        lines += [
            "",
            f"### Feature importances ({best['model']})",
            "",
            "| feature | importance |",
            "|---|---:|",
        ]
        for fi in best["feature_importances"][:10]:
            lines.append(f"| {fi['feature']} | {fi['importance']:.3f} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    rows = load_rows(args.db)
    logger.info("Loaded %d rows with ar_curriculum data", len(rows))
    if not rows:
        return 0

    target_results = []
    for target in TARGETS:
        logger.info("Evaluating predictor for %s", target)
        target_results.append(
            evaluate_target(
                rows, target, seed=int(args.seed), test_frac=float(args.test_frac)
            )
        )

    payload = {
        "run_id": run_id,
        "n_rows": len(rows),
        "targets": target_results,
    }
    json_path, md_path = write_report(payload, RUNTIME_ROOT, run_id)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
