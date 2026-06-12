#!/usr/bin/env python
"""Before/after eval: cascade-stacked + calibrated + cost-thresholded probe heads.

Answers "can we lift PPV/NPV/TPR/TNR over the graph-only audit heads?" by holding
the model family fixed (HistGradientBoosting, the audit's selected model) and
varying only three levers, cumulatively:

  cond 1  graph_only            graph features, uncalibrated, decision @ target thr  (== audit "before")
  cond 2  + cheap               graph + STRICTLY-CHEAPER probe features (cascade-safe)
  cond 3  + calibration         isotonic calibration fit on a held calibration fold
  cond 4  + cost_threshold      decision threshold minimizing expected cost (c_fp:c_fn)

Cascade safety: a head predicting an s10 score may only consume ar_gate / nano /
s05 features — never any s10 column (that is leakage). Enforced per-head via
``cheap_inputs``. Reuses ``audit_cheap_probe_predictors`` primitives; no dup.

3-way temporal split (train / calibration / test) keeps calibration honest:
isotonic is fit on calib, every metric is read on the forward-in-time test fold.

Usage:
    python -m research.tools.stacked_calibrated_probe_eval \
        --json-out research/reports/stacked_calibrated_probe.json \
        --markdown-out research/reports/stacked_calibrated_probe.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.defaults import RUNS_DB
from research.scientist.intelligence.metrics_utils import binary_classification_metrics
from research.tools.audit_cheap_probe_predictors import (
    _finite_float,
    _graph_features,
    _impute_train_val,
    _materialize_matrix,
    load_audit_rows,
)


@dataclass(frozen=True)
class CascadeHead:
    name: str
    target_columns: tuple[str, ...]
    threshold: float
    # Strictly-cheaper probe columns this head may consume (cascade order:
    # ar_gate -> nano -> nb0.5 -> nb1.0). NEVER include the head's own tier.
    cheap_inputs: tuple[str, ...] = field(default_factory=tuple)
    target_mode: str = "max"  # "max" single col, "joint_min" = min across cols


_AR = "ar_gate_score"
_NANO = "nano_induction_nearest_max_accuracy"
_S05_BIND = "language_control_s05_binding_score"
_S05_SENT = "language_control_s05_sentence_assoc_score"

CASCADE_HEADS: tuple[CascadeHead, ...] = (
    CascadeHead("ar_gate", ("ar_gate_score",), 0.95, cheap_inputs=()),
    CascadeHead("nb05_binding", (_S05_BIND,), 0.95, cheap_inputs=(_AR, _NANO)),
    CascadeHead(
        "nb10_binding",
        ("language_control_s10_binding_score",),
        0.95,
        cheap_inputs=(_AR, _NANO, _S05_BIND, _S05_SENT),
    ),
    CascadeHead(
        "nb10_joint",
        (
            "language_control_s10_binding_score",
            "language_control_s10_sentence_assoc_score",
        ),
        0.95,
        cheap_inputs=(_AR, _NANO, _S05_BIND, _S05_SENT),
        target_mode="joint_min",
    ),
)


def _target(row: Mapping[str, Any], head: CascadeHead) -> float | None:
    vals = [_finite_float(row.get(c)) for c in head.target_columns]
    finite = [v for v in vals if v is not None]
    if not finite:
        return None
    if head.target_mode == "joint_min":
        return min(finite) if len(finite) == len(head.target_columns) else None
    return max(finite)


def _cheap_feats(row: Mapping[str, Any], cols: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        value = _finite_float(row.get(col))
        out[f"cheap_{col}"] = value if value is not None else float("nan")
        out[f"cheap_{col}__present"] = 1.0 if value is not None else 0.0
    return out


def build_dataset(
    rows: Sequence[Mapping[str, Any]], head: CascadeHead, *, use_cheap: bool
) -> tuple[list[dict[str, float]], np.ndarray, list[Mapping[str, Any]]]:
    feats: list[dict[str, float]] = []
    targets: list[float] = []
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        tgt = _target(row, head)
        if tgt is None:
            continue
        f = _graph_features(row)
        if not f:
            continue
        if use_cheap and head.cheap_inputs:
            f = {**f, **_cheap_feats(row, head.cheap_inputs)}
        feats.append(f)
        targets.append(float(tgt))
        kept.append(row)
    return feats, np.asarray(targets, dtype=np.float64), kept


def _three_way_temporal(
    rows: Sequence[Mapping[str, Any]], train_frac: float, calib_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            float(rows[i].get("latest_timestamp") or 0.0),
            str(rows[i].get("canonical_fingerprint") or i),
        ),
    )
    n = len(order)
    a = max(1, int(math.floor(n * train_frac)))
    b = max(a + 1, int(math.floor(n * (train_frac + calib_frac))))
    b = min(b, n - 1)
    return (
        np.asarray(order[:a], dtype=np.int32),
        np.asarray(order[a:b], dtype=np.int32),
        np.asarray(order[b:], dtype=np.int32),
    )


def _model(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=int(seed),
    )


def _isotonic(p_calib: np.ndarray, y_calib: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    if np.unique(y_calib).size < 2 or p_calib.size < 10:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, y_calib.astype(np.float64))
    return iso


def _cost_threshold(
    y_true: np.ndarray, score: np.ndarray, *, c_fp: float, c_fn: float
) -> float:
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.98, 49), score]))
    best_thr, best_cost = 0.5, float("inf")
    for thr in grid:
        pred = score >= thr
        fp = float(np.sum(pred & (y_true == 0)))
        fn = float(np.sum(~pred & (y_true == 1)))
        cost = c_fp * fp + c_fn * fn
        if cost < best_cost or (cost == best_cost and thr < best_thr):
            best_cost, best_thr = cost, float(thr)
    return best_thr


def _row_metrics(
    y_true: np.ndarray, score: np.ndarray, thr: float, label: str
) -> dict[str, Any]:
    m = binary_classification_metrics(y_true.astype(np.int32), score, threshold=thr)
    return {
        "condition": label,
        "threshold": round(float(thr), 3),
        "roc_auc": round(float(m["roc_auc"]), 3),
        "ppv": round(float(m["precision_ppv"]), 3),
        "npv": round(float(m["npv"]), 3),
        "tpr": round(float(m["recall_tpr_sensitivity"]), 3),
        "tnr": round(float(m["specificity_tnr"]), 3),
        "accuracy": round(float(m["accuracy"]), 3),
        "f1": round(float(m["f1"]), 3),
        "tp": int(m["tp"]),
        "fp": int(m["fp"]),
        "tn": int(m["tn"]),
        "fn": int(m["fn"]),
    }


def evaluate_head(
    rows: Sequence[Mapping[str, Any]],
    head: CascadeHead,
    *,
    train_frac: float,
    calib_frac: float,
    seed: int,
    c_fp: float,
    c_fn: float,
    min_eval: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"head": head.name, "threshold": head.threshold}
    conditions: list[dict[str, Any]] = []

    for use_cheap in (False, True):
        if use_cheap and not head.cheap_inputs:
            continue  # no cheaper probe exists (e.g. ar_gate)
        feats, y, kept = build_dataset(rows, head, use_cheap=use_cheap)
        if y.size < 40 or np.unique((y >= head.threshold)).size < 2:
            continue
        tr, ca, test_idx = _three_way_temporal(kept, train_frac, calib_frac)
        if test_idx.size < min_eval or tr.size < 20:
            continue
        X_all, _ = _materialize_matrix(feats)
        Xtr, Xca = _impute_train_val(X_all[tr], X_all[ca])
        _, Xtest = _impute_train_val(X_all[tr], X_all[test_idx])
        ybin = (y >= head.threshold).astype(np.int32)
        ytr, yca, ytest = ybin[tr], ybin[ca], ybin[test_idx]
        if np.unique(ytr).size < 2:
            continue

        model = _model(seed)
        model.fit(Xtr, ytr)
        p_test = np.clip(model.predict_proba(Xtest)[:, 1], 0.0, 1.0)
        tag = "graph+cheap" if use_cheap else "graph_only"

        # cond: uncalibrated @ 0.5
        conditions.append(_row_metrics(ytest, p_test, 0.5, f"{tag} | uncal @0.50"))

        if use_cheap or not head.cheap_inputs:
            # calibration + cost only profiled on the richest feature set
            p_ca = np.clip(model.predict_proba(Xca)[:, 1], 0.0, 1.0)
            iso = _isotonic(p_ca, yca)
            if iso is not None:
                p_cal = np.clip(iso.predict(p_test), 0.0, 1.0)
                conditions.append(
                    _row_metrics(ytest, p_cal, 0.5, f"{tag} | calibrated @0.50")
                )
                # cost-optimal threshold chosen on calib, applied to test
                p_ca_cal = np.clip(iso.predict(p_ca), 0.0, 1.0)
                thr = _cost_threshold(yca, p_ca_cal, c_fp=c_fp, c_fn=c_fn)
                conditions.append(
                    _row_metrics(
                        ytest,
                        p_cal,
                        thr,
                        f"{tag} | calibrated @cost({c_fp:g}:{c_fn:g})",
                    )
                )
        out["n_test"] = int(test_idx.size)
        out["test_prevalence"] = round(float(np.mean(ytest)), 3)

    out["conditions"] = conditions
    return out


def _fit_eval_family(
    feats: Sequence[Mapping[str, float]],
    ybin: np.ndarray,
    fams: np.ndarray,
    held: str,
    seed: int,
    min_family: int,
) -> dict[str, Any] | None:
    """Train on all families != held, test on held. Uncalibrated @0.5."""
    test_idx = np.flatnonzero(fams == held)
    tr = np.flatnonzero(fams != held)
    if test_idx.size < min_family or tr.size < 40:
        return None
    if np.unique(ybin[tr]).size < 2 or np.unique(ybin[test_idx]).size < 2:
        return None
    X_all, _ = _materialize_matrix(feats)
    Xtr, Xtest = _impute_train_val(X_all[tr], X_all[test_idx])
    model = _model(seed)
    model.fit(Xtr, ybin[tr])
    p = np.clip(model.predict_proba(Xtest)[:, 1], 0.0, 1.0)
    m = binary_classification_metrics(ybin[test_idx], p, threshold=0.5)
    return {
        "family": held,
        "n": int(test_idx.size),
        "roc_auc": float(m["roc_auc"]),
        "ppv": float(m["precision_ppv"]),
        "npv": float(m["npv"]),
        "tpr": float(m["recall_tpr_sensitivity"]),
        "tnr": float(m["specificity_tnr"]),
    }


def evaluate_head_lfo(
    rows: Sequence[Mapping[str, Any]], head: CascadeHead, *, seed: int, min_family: int
) -> dict[str, Any]:
    """Leave-one-family-out OOD: the deployment-relevant generalization test."""
    out: dict[str, Any] = {"head": head.name, "conditions": []}
    for use_cheap in (False, True):
        if use_cheap and not head.cheap_inputs:
            continue
        feats, y, kept = build_dataset(rows, head, use_cheap=use_cheap)
        if y.size < 60:
            continue
        ybin = (y >= head.threshold).astype(np.int32)
        fams = np.asarray([str(r.get("family") or "unknown") for r in kept])
        per_family = [
            r
            for fam in sorted(set(fams))
            if (r := _fit_eval_family(feats, ybin, fams, fam, seed, min_family))
        ]
        if not per_family:
            continue
        tag = "graph+cheap" if use_cheap else "graph_only"
        out["conditions"].append(
            {
                "condition": f"{tag} | LFO-OOD @0.50",
                "families_evaluated": len(per_family),
                "mean_roc_auc": round(
                    float(np.mean([p["roc_auc"] for p in per_family])), 3
                ),
                "mean_ppv": round(float(np.mean([p["ppv"] for p in per_family])), 3),
                "mean_npv": round(float(np.mean([p["npv"] for p in per_family])), 3),
                "mean_tpr": round(float(np.mean([p["tpr"] for p in per_family])), 3),
                "mean_tnr": round(float(np.mean([p["tnr"] for p in per_family])), 3),
                "worst_family_roc": round(
                    float(min(p["roc_auc"] for p in per_family)), 3
                ),
            }
        )
    return out


def format_markdown_lfo(report: Mapping[str, Any]) -> str:
    lines = [
        "# Leave-Family-Out (OOD) — stacking gain on UNSEEN architecture families",
        "",
        "| head | condition | families | mean ROC | mean PPV | mean NPV | mean TPR | mean TNR | worst-family ROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for head in report.get("lfo_heads", []):
        for c in head.get("conditions", []):
            lines.append(
                "| {h} | {cond} | {fam} | {roc} | {ppv} | {npv} | {tpr} | {tnr} | {worst} |".format(
                    h=head["head"],
                    cond=c["condition"],
                    fam=c["families_evaluated"],
                    roc=c["mean_roc_auc"],
                    ppv=c["mean_ppv"],
                    npv=c["mean_npv"],
                    tpr=c["mean_tpr"],
                    tnr=c["mean_tnr"],
                    worst=c["worst_family_roc"],
                )
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def format_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stacked + Calibrated + Cost-Thresholded Probe Heads",
        "",
        f"Rows (fingerprint-deduped): {report.get('row_count')}  | "
        f"cost ratio c_fp:c_fn = {report.get('c_fp')}:{report.get('c_fn')}",
        "",
    ]
    for head in report.get("heads", []):
        lines += [
            f"## {head['head']}  (target>={head['threshold']}, "
            f"n_test={head.get('n_test')}, test_prevalence={head.get('test_prevalence')})",
            "",
            "| condition | thr | ROC | PPV | NPV | TPR | TNR | acc | F1 | tp/fp/tn/fn |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in head.get("conditions", []):
            lines.append(
                "| {cond} | {thr} | {roc} | {ppv} | {npv} | {tpr} | {tnr} | {acc} | {f1} | {cm} |".format(
                    cond=c["condition"],
                    thr=c["threshold"],
                    roc=c["roc_auc"],
                    ppv=c["ppv"],
                    npv=c["npv"],
                    tpr=c["tpr"],
                    tnr=c["tnr"],
                    acc=c["accuracy"],
                    f1=c["f1"],
                    cm=f"{c['tp']}/{c['fp']}/{c['tn']}/{c['fn']}",
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(RUNS_DB))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.64)
    ap.add_argument("--calib-frac", type=float, default=0.16)
    ap.add_argument("--c-fp", type=float, default=3.0, help="cost of a false accept")
    ap.add_argument("--c-fn", type=float, default=1.0, help="cost of a false reject")
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--min-family", type=int, default=15)
    ap.add_argument("--mode", choices=("temporal", "lfo", "both"), default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--markdown-out", type=Path)
    args = ap.parse_args(argv)

    rows = load_audit_rows(args.db, limit=args.limit)
    report: dict[str, Any] = {
        "audit_version": "stacked_calibrated_probe_v1",
        "row_count": int(len(rows)),
        "c_fp": args.c_fp,
        "c_fn": args.c_fn,
    }
    md_parts: list[str] = []
    if args.mode in ("temporal", "both"):
        report["heads"] = [
            evaluate_head(
                rows,
                head,
                train_frac=args.train_frac,
                calib_frac=args.calib_frac,
                seed=args.seed,
                c_fp=args.c_fp,
                c_fn=args.c_fn,
                min_eval=args.min_eval,
            )
            for head in CASCADE_HEADS
        ]
        md_parts.append(format_markdown(report))
    if args.mode in ("lfo", "both"):
        report["lfo_heads"] = [
            evaluate_head_lfo(rows, head, seed=args.seed, min_family=args.min_family)
            for head in CASCADE_HEADS
        ]
        md_parts.append(format_markdown_lfo(report))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    md = "\n".join(md_parts)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
