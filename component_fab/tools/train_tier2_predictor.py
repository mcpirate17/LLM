"""Train + leave-architecture-out validate a Tier-2 value predictor.

Target = Tier-2 ``mean_delta`` (how much a candidate beats the baseline on net).
Features = measured position-Jacobian descriptors + op-multiset size + key axis
indicators, computed at train time from each row's ``math_axes``.

DEPLOY GATE: this tool REFUSES to save a model unless it generalizes out-of-
distribution — enough labels, enough distinct architectures, and a positive
leave-architecture-out Spearman + R². As of 2026-06-03 the data (n=34, 8 archs)
fails this gate by a wide margin (leave-arch-out R² < 0); the table must grow
(run more diverse cohorts) before a model is worth trusting. This is deliberate:
a predictor that doesn't beat predict-the-mean OOD is worse than no predictor.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from component_fab.proposer.capability_screen import fab_op_multiset
from component_fab.proposer.measured_screen import measured_screen_for_spec
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    category_from_axes,
    synthesis_kind_for_axes,
)
from component_fab.state.tier2_training import load_tier2_labels

_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "research" / "runtime" / "tier2_value_predictor"

# Deploy gate — a model is saved only if it clears ALL of these OOD bars.
MIN_LABELS = 60
MIN_ARCHS = 12
MIN_LEAVE_ARCH_OUT_RHO = 0.35

_DESCRIPTORS = (
    "long_range_reach",
    "content_dependence",
    "content_match_gating",
    "causality_violation",
    "measured_lipschitz",
    "effective_rank",
    "nonlinearity",
    "self_dominance",
)
_FEATURE_NAMES = (
    *_DESCRIPTORS,
    "op_count",
    "n_distinct_ops",
    "has_state",
    "memory_o_l",
    "global_receptive",
)


def _spec_from_axes(pid: str, name: str, axes: dict[str, Any]) -> ProposalSpec:
    return ProposalSpec(
        proposal_id=pid,
        name=name or pid,
        category=category_from_axes(axes),
        synthesis_kind=str(
            axes.get("synthesis_kind") or synthesis_kind_for_axes(axes, axes)
        ),
        math_axes=dict(axes),
        anchor_witness_op="",
        anchor_witnesses_all=(),
        declared_property_row=dict(axes),
        predicted_lift=0.5,
        rationale="",
    )


def _features_for_row(row: dict[str, Any], extractor: Any) -> list[float] | None:
    axes = dict(row.get("math_axes") or {})
    spec = _spec_from_axes(row.get("proposal_id", ""), row.get("name", ""), axes)
    ms = measured_screen_for_spec(spec, extractor=extractor)
    if not ms.available or ms.descriptors is None:
        return None
    d = ms.descriptors
    ops = fab_op_multiset(spec)
    feat = [float(d.get(k, 0.0)) for k in _DESCRIPTORS]
    feat += [float(len(ops)), float(len(set(ops)))]
    feat += [
        float(axes.get("op_dynamical_has_state") or 0),
        1.0 if axes.get("op_dynamical_memory_length_class") == "O(L)" else 0.0,
        1.0
        if axes.get("op_geometric_receptive_field") in ("global", "hybrid_local_global")
        else 0.0,
    ]
    return feat if all(np.isfinite(feat)) else None


def _build_dataset(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from research.tools.measured_descriptors import MeasuredDescriptorExtractor

    extractor = MeasuredDescriptorExtractor(n_seeds=2)
    X: list[list[float]] = []
    y: list[float] = []
    groups: list[str] = []
    for row in rows:
        feat = _features_for_row(row, extractor)
        if feat is None:
            continue
        X.append(feat)
        y.append(float(row.get("mean_delta") or 0.0))
        groups.append(str(row.get("arch_group") or "unknown"))
    return np.asarray(X), np.asarray(y), groups


def _leave_arch_out_eval(
    X: np.ndarray, y: np.ndarray, groups: list[str]
) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    gids = np.array([sorted(set(groups)).index(g) for g in groups])
    pred = np.zeros_like(y)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, gids):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=2)
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    rho = spearmanr(y, pred).correlation
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "spearman": float(rho) if rho == rho else 0.0,  # nan-guard
        "n_archs": len(set(groups)),
        "n_labels": len(y),
    }


def _save_model(X: np.ndarray, y: np.ndarray, metrics: dict[str, float]) -> Path:
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(n_estimators=100, max_depth=2).fit(X, y)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _OUT_DIR / "model.joblib")
    (_OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "target": "tier2_mean_delta",
                "feature_names": list(_FEATURE_NAMES),
                "leave_arch_out": metrics,
                "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return _OUT_DIR / "model.joblib"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="train Tier-2 value predictor")
    parser.add_argument("--min-labels", type=int, default=MIN_LABELS)
    parser.add_argument("--min-archs", type=int, default=MIN_ARCHS)
    parser.add_argument("--min-rho", type=float, default=MIN_LEAVE_ARCH_OUT_RHO)
    parser.add_argument(
        "--force",
        action="store_true",
        help="save the model even if it fails the OOD deploy gate (debug only)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rows = load_tier2_labels()
    print(f"loaded {len(rows)} Tier-2 training labels")
    if len(rows) < 2:
        print("not enough labels to train anything yet — run cohorts to accumulate.")
        return 0
    X, y, groups = _build_dataset(rows)
    n_archs = len(set(groups))
    print(f"usable rows: {len(y)}  distinct architectures: {n_archs}")
    if n_archs < 2:
        print("need >=2 distinct architectures for leave-arch-out CV.")
        return 0

    metrics = _leave_arch_out_eval(X, y, groups)
    print(
        f"leave-architecture-out: R2={metrics['r2']:+.3f} "
        f"Spearman={metrics['spearman']:+.3f} "
        f"(n_labels={metrics['n_labels']}, n_archs={metrics['n_archs']})"
    )
    passes = (
        len(y) >= args.min_labels
        and n_archs >= args.min_archs
        and metrics["r2"] > 0.0
        and metrics["spearman"] >= args.min_rho
    )
    if not passes and not args.force:
        print(
            "DEPLOY GATE FAILED — not saving. Need "
            f">={args.min_labels} labels, >={args.min_archs} archs, R2>0, "
            f"Spearman>={args.min_rho}. The blocker is DATA: run more diverse "
            "Tier-2 cohorts to grow research/data/tier2_predictor/labels.jsonl."
        )
        return 0
    path = _save_model(X, y, metrics)
    print(f"{'(forced) ' if not passes else ''}saved predictor → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
