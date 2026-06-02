#!/usr/bin/env python
"""Numerical-estimation companion to ``audit_cheap_probe_predictors``.

The audit head trains *classifiers* for nb05/nb10 (pass/fail at a threshold) and
a *regressor* for ar_gate. This script answers a different question for the same
targets: instead of "will it pass?", "what continuous score does the static
graph predict, and how accurate is that estimate?".

For each target it reuses the audit's fingerprint-deduped row loader, graph
features, and temporal split, fits regressor candidates, and reports regression
accuracy (R^2, MAE, RMSE) plus rank accuracy (Spearman) on a forward-in-time
holdout. Reuses ``audit_cheap_probe_predictors`` machinery; no duplication.

Usage:
    python -m research.tools.probe_numeric_estimation_eval \
        --json-out research/reports/probe_numeric_estimation.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.defaults import RUNS_DB
from research.tools.audit_cheap_probe_predictors import (
    HeadSpec,
    _impute_train_val,
    _materialize_matrix,
    _spearman,
    _temporal_split,
    build_head_dataset,
    load_audit_rows,
)

# Continuous targets the user asked about: AR gate, nb0.5, nb1.0. Sentence-assoc
# and nano_induction included for context. All predicted from the static graph.
NUMERIC_SPECS: tuple[HeadSpec, ...] = (
    HeadSpec(
        name="estimate_ar_gate_score",
        feature_mode="graph",
        target_columns=("ar_gate_score",),
        threshold=0.95,
        target_definition="static graph -> continuous ar_gate_score",
        model_kind="regressor",
    ),
    HeadSpec(
        name="estimate_nb05_binding_score",
        feature_mode="graph",
        target_columns=("language_control_s05_binding_score",),
        threshold=0.95,
        target_definition="static graph -> continuous nb0.5 binding score",
        model_kind="regressor",
    ),
    HeadSpec(
        name="estimate_nb05_sentence_assoc_score",
        feature_mode="graph",
        target_columns=("language_control_s05_sentence_assoc_score",),
        threshold=0.95,
        target_definition="static graph -> continuous nb0.5 sentence-assoc score",
        model_kind="regressor",
    ),
    HeadSpec(
        name="estimate_nb10_binding_score",
        feature_mode="graph",
        target_columns=("language_control_s10_binding_score",),
        threshold=0.95,
        target_definition="static graph -> continuous nb1.0 binding score",
        model_kind="regressor",
    ),
    HeadSpec(
        name="estimate_nb10_sentence_assoc_score",
        feature_mode="graph",
        target_columns=("language_control_s10_sentence_assoc_score",),
        threshold=0.95,
        target_definition="static graph -> continuous nb1.0 sentence-assoc score",
        model_kind="regressor",
    ),
    HeadSpec(
        name="estimate_nano_induction_nearest",
        feature_mode="graph",
        target_columns=("nano_induction_nearest_max_accuracy",),
        threshold=0.50,
        target_definition="static graph -> continuous nano_induction_nearest_max_accuracy",
        model_kind="regressor",
    ),
)


def _regressor_candidates(seed: int, n_estimators: int) -> list[tuple[str, Any]]:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )

    return [
        (
            "random_forest_regressor",
            RandomForestRegressor(
                n_estimators=max(64, int(n_estimators)),
                max_depth=10,
                min_samples_leaf=3,
                random_state=int(seed),
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees_regressor",
            ExtraTreesRegressor(
                n_estimators=max(64, int(n_estimators)),
                min_samples_leaf=3,
                random_state=int(seed),
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting_regressor",
            HistGradientBoostingRegressor(
                max_iter=max(64, int(n_estimators)),
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=int(seed),
            ),
        ),
    ]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")  # constant target — R^2 undefined
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _fit_select_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Fit each candidate regressor, return (best_by_r2, comparison)."""
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    comparison: list[dict[str, Any]] = []
    for name, model in _regressor_candidates(seed, n_estimators):
        try:
            model.fit(X_train, y_train)
            preds = np.clip(
                np.asarray(model.predict(X_val), dtype=np.float64), 0.0, 1.0
            )
            r2 = _r2(y_val, preds)
            entry = {
                "model": name,
                "r2": r2,
                "mae": float(np.mean(np.abs(y_val - preds))),
                "rmse": float(math.sqrt(np.mean((y_val - preds) ** 2))),
                "spearman": _spearman(y_val, preds),
            }
            comparison.append(entry)
            # Select on R^2 (nan treated as -inf), tie-break on spearman.
            score = (-1e18 if (r2 != r2) else r2, entry["spearman"])
            if best is None or score > best[0]:
                best = (score, entry)
        except Exception as exc:  # noqa: BLE001
            comparison.append({"model": name, "error": str(exc)[:120]})
    comparison.sort(
        key=lambda item: (
            float(item["r2"])
            if isinstance(item.get("r2"), float) and math.isfinite(item["r2"])
            else -1e18
        ),
        reverse=True,
    )
    return (best[1] if best else None), comparison


def _evaluate_numeric_head(
    rows: Sequence[Mapping[str, Any]],
    spec: HeadSpec,
    *,
    train_fraction: float,
    min_samples: int,
    min_eval: int,
    seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    feat_rows, y, kept_rows = build_head_dataset(rows, spec)
    out: dict[str, Any] = {
        "head": spec.name,
        "target_columns": list(spec.target_columns),
        "target_definition": spec.target_definition,
        "sample_count": int(y.size),
    }
    if y.size < min_samples or np.unique(y).size < 2:
        out["error"] = "insufficient_samples_or_target_variance"
        return out
    out["target_mean"] = float(np.mean(y))
    out["target_std"] = float(np.std(y))

    train_idx, val_idx = _temporal_split(kept_rows, train_fraction)
    if val_idx.size < min_eval:
        out["error"] = "insufficient_temporal_eval_rows"
        out["n_temporal_val"] = int(val_idx.size)
        return out

    X_all, _ = _materialize_matrix(feat_rows)
    X_train, X_val = _impute_train_val(X_all[train_idx], X_all[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    baseline_mae = float(np.mean(np.abs(y_val - float(np.mean(y_train)))))

    best, comparison = _fit_select_regressor(
        X_train, y_train, X_val, y_val, seed=seed, n_estimators=n_estimators
    )
    out.update(
        {
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "feature_count": int(X_all.shape[1]),
            "baseline_mae_predict_mean": baseline_mae,
            "best_model": best["model"] if best else None,
            "best_r2": best["r2"] if best else float("nan"),
            "best_mae": best["mae"] if best else float("nan"),
            "best_rmse": best["rmse"] if best else float("nan"),
            "best_spearman": best["spearman"] if best else 0.0,
            "mae_improvement_vs_mean": (
                baseline_mae - best["mae"] if best else float("nan")
            ),
            "model_comparison": comparison,
        }
    )
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(v) else f"{v:.{digits}f}"


def format_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Numerical-Estimation Head Audit (regression)",
        "",
        f"Rows (fingerprint-deduped): {int(report.get('row_count') or 0)}",
        "",
        "| target | n | model | R^2 | MAE | RMSE | Spearman | MAE vs mean-baseline |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for head in report.get("heads", []):
        if head.get("error"):
            lines.append(
                f"| {head['head']} | {head.get('sample_count')} | — | — | — | — | — | {head['error']} |"
            )
            continue
        lines.append(
            "| {h} | {n} | {m} | {r2} | {mae} | {rmse} | {rho} | {imp} |".format(
                h=head["head"],
                n=head.get("n_val"),
                m=head.get("best_model"),
                r2=_fmt(head.get("best_r2")),
                mae=_fmt(head.get("best_mae")),
                rmse=_fmt(head.get("best_rmse")),
                rho=_fmt(head.get("best_spearman")),
                imp=_fmt(head.get("mae_improvement_vs_mean")),
            )
        )
    lines.append("")
    lines.append(
        "R^2>0 means the graph-feature regressor beats predicting the dataset mean; "
        "Spearman is rank accuracy (robust to calibration). 'MAE vs mean-baseline' "
        "is how many absolute-error points the model saves over the naive mean."
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-eval", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    rows = load_audit_rows(args.db, limit=args.limit)
    heads = [
        _evaluate_numeric_head(
            rows,
            spec,
            train_fraction=args.train_fraction,
            min_samples=args.min_samples,
            min_eval=args.min_eval,
            seed=args.seed,
            n_estimators=args.n_estimators,
        )
        for spec in NUMERIC_SPECS
    ]
    report = {
        "audit_version": "probe_numeric_estimation_v1",
        "row_count": int(len(rows)),
        "heads": heads,
    }
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
