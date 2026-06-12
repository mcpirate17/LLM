#!/usr/bin/env python
"""Learned BLiMP surrogate from cheap screening metrics (runs.db).

Fills the gap the existing predictor ecosystem leaves: the oracle / pls_partition /
capability_screener all predict ``induction_screening_auc`` from graph-semantic
features, and ``audit_cheap_probe_predictors`` has a BLiMP head but only for the
``large_blimp`` 100M+ subset from 6 probes. Nothing trains on the FULL
``runs.db.program_results.blimp_overall_accuracy`` distribution using the broad
cheap battery (induction/binding screening AUC + the fp_* jacobian/ICLD/logit-margin
fingerprints + diagnostic_score + loss-trajectory).

The 2026-06-10 correlation study (see research/notes/aria_history_and_blimp_correlation)
showed: no single cheap metric predicts BLiMP (best partial |rho|~0.2), loss-based
metrics are ANTI-predictive, but a GBM *combination* roughly doubles the signal
(OOF Spearman ~0.43, top-30% screen recall ~0.52 vs 0.30 chance). This trains that
combination as a reusable screening proxy.

OOD-honest by default: out-of-fold predictions are LEAVE-ONE-GRAPH-FAMILY-OUT
(``classify_graph_family``), because runs.db has near-duplicate architectures that
random K-fold would leak across folds — inflating the score. A random-fold number
is reported alongside so the leakage gap is visible.

Usage:
    python -m research.tools.blimp_cheap_surrogate            # train + eval + save
    python -m research.tools.blimp_cheap_surrogate --no-size  # drop param/steps features
    python -m research.tools.blimp_cheap_surrogate --target large_blimp_overall_accuracy
    python -m research.tools.blimp_cheap_surrogate --score    # prioritize rows missing BLiMP
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.defaults import RUNS_DB
from research.tools.annotate_literature_attribution import classify_graph_family

ARTIFACT_DIR = Path("research/runtime/blimp_cheap_surrogate")
UPPER_QUANTILE = 0.9

# Broad cheap battery — computed during screening, BEFORE a full BLiMP eval.
# Excludes the expensive eval outcomes (hellaswag/wikitext) and the known-broken
# >70%-zero fingerprints (fp_intrinsic_dim/isotropy/rank_ratio).
FEATURE_COLUMNS: tuple[str, ...] = (
    "induction_screening_auc",
    "binding_screening_composite",
    "binding_screening_auc",
    "fp_jacobian_erf_density",
    "fp_jacobian_erf_decay_slope",
    "fp_jacobian_spectral_norm",
    "fp_jacobian_effective_rank",
    "fp_icld_velocity",
    "fp_icld_delta_loss",
    "fp_id_collapse_rate",
    "fp_logit_margin_velocity",
    "fp_logit_margin_delta",
    "diagnostic_score",
    "ncd_score",
    "stability_score",
    "mean_grad_norm",
    "loss_improvement_rate",
    "validation_loss_ratio",
    "loss_ratio",
    "final_loss",
    "min_loss",
    "screening_loss_50",
    "induction_intermediate_auc",
    "binding_intermediate_auc",
)
# Known at screen time, predictive but a confound for same-budget ranking — opt-out.
SIZE_COLUMNS: tuple[str, ...] = ("param_count", "n_train_steps", "flops_per_token")


@dataclass(slots=True)
class FeatureCoverage:
    feature: str
    pct_present: float
    importance: float | None = None


@dataclass(slots=True)
class SurrogateReport:
    target: str
    n_rows: int
    n_families: int
    feature_names: list[str]
    family_out_spearman: float
    family_out_r2: float
    random_fold_spearman: float
    screen_recall_at_30pct: float
    chance_recall: float
    coverage: list[FeatureCoverage] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _ops_from_graph(graph_json: Any) -> set[str]:
    if isinstance(graph_json, str) and graph_json.strip():
        try:
            graph = json.loads(graph_json)
        except (json.JSONDecodeError, ValueError):
            return set()
    elif isinstance(graph_json, Mapping):
        graph = dict(graph_json)
    else:
        return set()
    nodes = graph.get("nodes") or {}
    values = nodes.values() if isinstance(nodes, Mapping) else nodes
    if not isinstance(values, (list, tuple)) and not hasattr(values, "__iter__"):
        return set()
    return {
        str(n.get("op_name") or "")
        for n in values
        if isinstance(n, Mapping) and n.get("op_name")
    }


def _available_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(program_results)")}


def load_data(
    db_path: Path | str,
    *,
    target: str,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (X, y, families, feature_names), deduped to one row per graph.

    Dedup keeps the MAX target per graph fingerprint (best observed capability),
    matching the existing audit harness — so a single architecture appearing in
    many runs cannot dominate or leak across folds.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        avail = _available_columns(conn)
        feats = [f for f in features if f in avail]
        if target not in avail:
            raise ValueError(f"target {target!r} not in program_results")
        cols = ["graph_fingerprint", "graph_json", target, *feats]
        sel = ", ".join(cols)
        rows = conn.execute(
            f"SELECT {sel} FROM program_results "  # noqa: S608 — cols are validated
            f"WHERE {target} IS NOT NULL AND {target} > 0"
        ).fetchall()
    finally:
        conn.close()

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        fp = row[0] or row[1]
        if fp is None:
            continue
        y = _finite(row[2])
        if math.isnan(y):
            continue
        prev = best.get(str(fp))
        if prev is None or y > prev["y"]:
            best[str(fp)] = {"y": y, "graph_json": row[1], "feat": row[3:]}

    X = np.array([[_finite(v) for v in r["feat"]] for r in best.values()], dtype=float)
    yv = np.array([r["y"] for r in best.values()], dtype=float)
    families = [
        classify_graph_family(_ops_from_graph(r["graph_json"])) or "unknown"
        for r in best.values()
    ]
    return X, yv, families, feats


# --------------------------------------------------------------------------- #
# Metrics (stdlib/numpy)
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return 0.0
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _recall_at_frac(
    scores: np.ndarray, y: np.ndarray, keep: float
) -> tuple[float, float]:
    top = y >= np.quantile(y, 0.90)
    if top.sum() == 0:
        return 0.0, keep
    thr = np.quantile(scores, 1 - keep)
    kept = scores >= thr
    return float((top & kept).sum()) / float(top.sum()), keep


# --------------------------------------------------------------------------- #
# Training / OOF
# --------------------------------------------------------------------------- #
def _new_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=0,
    )


def _leave_family_out_oof(
    X: np.ndarray, y: np.ndarray, families: Sequence[str]
) -> np.ndarray:
    """OOD-honest OOF: predict each family from a model trained on all others."""
    fam_idx: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(families):
        fam_idx[f].append(i)
    oof = np.full(len(y), np.nan)
    for idx in fam_idx.values():
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        if mask.sum() < 50:
            continue
        model = _new_regressor().fit(X[mask], y[mask])
        oof[idx] = model.predict(X[idx])
    return oof


def _random_fold_oof(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
    from sklearn.model_selection import KFold

    oof = np.zeros(len(y))
    for train_idx, test_idx in KFold(n_folds, shuffle=True, random_state=0).split(X):
        oof[test_idx] = (
            _new_regressor().fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        )
    return oof


def _coverage(X: np.ndarray, names: Sequence[str]) -> list[FeatureCoverage]:
    return [
        FeatureCoverage(feature=n, pct_present=float((~np.isnan(X[:, j])).mean()))
        for j, n in enumerate(names)
    ]


def evaluate(
    X: np.ndarray, y: np.ndarray, families: Sequence[str], names: Sequence[str]
) -> SurrogateReport:
    fam_oof = _leave_family_out_oof(X, y, families)
    valid = ~np.isnan(fam_oof)
    fam_sp = _spearman(fam_oof[valid], y[valid])
    fam_r2 = _r2(y[valid], fam_oof[valid])
    rand_oof = _random_fold_oof(X, y)
    rand_sp = _spearman(rand_oof, y)
    recall, chance = _recall_at_frac(fam_oof[valid], y[valid], 0.30)
    cov = _coverage(X, names)
    # permutation importance on a full-data fit (directional, for feature triage)
    from sklearn.inspection import permutation_importance

    full = _new_regressor().fit(X, y)
    pi = permutation_importance(full, X, y, n_repeats=5, random_state=0, scoring="r2")
    for c, imp in zip(cov, pi.importances_mean):
        c.importance = float(imp)
    report = SurrogateReport(
        target="",
        n_rows=len(y),
        n_families=len(set(families)),
        feature_names=list(names),
        family_out_spearman=round(fam_sp, 4),
        family_out_r2=round(fam_r2, 4),
        random_fold_spearman=round(rand_sp, 4),
        screen_recall_at_30pct=round(recall, 4),
        chance_recall=round(chance, 4),
        coverage=sorted(cov, key=lambda c: -(c.importance or 0.0)),
    )
    report.findings = _findings(report)
    return report


def _findings(r: SurrogateReport) -> list[str]:
    out = [
        f"Leave-family-out OOF Spearman={r.family_out_spearman:.3f} (the honest number); "
        f"random-fold={r.random_fold_spearman:.3f} — the gap is architecture-family leakage.",
        f"Screening: keep top-30% by surrogate -> retain {r.screen_recall_at_30pct:.0%} of "
        f"the true top-10%-BLiMP (vs {r.chance_recall:.0%} chance, "
        f"{r.screen_recall_at_30pct / max(r.chance_recall, 1e-9):.1f}x).",
    ]
    broken = [c.feature for c in r.coverage if c.pct_present < 0.30]
    if broken:
        out.append(
            f"LOW-COVERAGE features (<30% present, weak/unreliable): {', '.join(broken)}."
        )
    top = [c for c in r.coverage if (c.importance or 0) > 0][:6]
    if top:
        out.append(
            "Top features by permutation importance: "
            + ", ".join(f"{c.feature}({c.importance:+.3f})" for c in top)
        )
    return out


# --------------------------------------------------------------------------- #
# Persist / score
# --------------------------------------------------------------------------- #
def train_and_save(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], target: str, out_dir: Path
) -> Path:
    import joblib

    median = _new_regressor().fit(X, y)
    upper = _new_regressor()
    upper.set_params(loss="quantile", quantile=UPPER_QUANTILE)
    upper.fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.joblib"
    joblib.dump(
        {
            "median": median,
            "upper": upper,
            "features": list(names),
            "target": target,
            # Sorted in-sample predictions → at score time a new prediction's
            # percentile vs the training population is a searchsorted lookup, so
            # priority output can say "this is in the top-X% of what we've trained on".
            "train_pred_sorted": np.sort(median.predict(X)),
        },
        path,
    )
    return path


def load_model(out_dir: Path) -> dict[str, Any]:
    import joblib

    path = out_dir / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained surrogate at {path}; run without --score first to train it"
        )
    return joblib.load(path)


def _load_score_rows(
    db_path: Path | str,
    *,
    target: str,
    features: Sequence[str],
    only_missing: bool,
    require_stage1: bool,
    require_screened: bool,
    limit: int,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix + result_ids for rows to prioritize (default: missing BLiMP).

    ``require_screened`` keeps only rows whose primary cheap screen
    (``induction_screening_auc``) was actually computed — gating a row with no
    screening signal is low-information (the model would mostly return its prior).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        avail = _available_columns(conn)
        feats = [f for f in features if f in avail]
        where = ["graph_json IS NOT NULL"]
        if only_missing and target in avail:
            where.append(f"({target} IS NULL OR {target} <= 0)")
        if require_stage1 and "stage1_passed" in avail:
            where.append("stage1_passed = 1")
        if require_screened and "induction_screening_auc" in avail:
            where.append("induction_screening_auc IS NOT NULL")
        sel = ", ".join(["result_id", *feats])
        sql = (
            f"SELECT {sel} FROM program_results "  # noqa: S608 — validated columns
            f"WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT {int(limit)}"
        )
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    ids = [str(r[0]) for r in rows]
    X = np.array([[_finite(v) for v in r[1:]] for r in rows], dtype=float)
    if X.size == 0:
        X = np.empty((0, len(feats)), dtype=float)
    return X, ids


def _percentile(value: float, sorted_pop: np.ndarray | None) -> float:
    if sorted_pop is None or len(sorted_pop) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_pop, value) / len(sorted_pop))


def score_and_gate(
    db_path: Path | str,
    model: dict[str, Any],
    *,
    keep_frac: float,
    keep_threshold: float | None,
    only_missing: bool,
    require_stage1: bool,
    require_screened: bool,
    limit: int,
    budget: int = 0,
) -> list[dict[str, Any]]:
    """Predict BLiMP for un-evaluated rows → an advisory BLiMP eval priority list.

    Records are sorted best-predicted-first (so a consumer evaluates the most
    promising candidates first and can stop at a budget). ``recommended_for_eval``
    is true when, in priority order:
      - ``budget`` > 0: this row is in the top ``budget`` by predicted BLiMP, OR
      - predicted percentile (vs the training population) >= 1 - ``keep_frac``, OR
      - predicted BLiMP >= ``keep_threshold`` (if given).

    This is NOT a hard gate. Until prospective validation shows >=95% frontier
    recall, low-priority rows must remain eligible for random/novelty/exploration
    lanes and downstream consumers must not discard them from ``priority=low``.
    """
    feats = model["features"]
    sorted_pop = model.get("train_pred_sorted")
    X, ids = _load_score_rows(
        db_path,
        target=model["target"],
        features=feats,
        only_missing=only_missing,
        require_stage1=require_stage1,
        require_screened=require_screened,
        limit=limit,
    )
    if not ids:
        return []
    med = model["median"].predict(X)
    upp = model["upper"].predict(X)
    order = sorted(range(len(ids)), key=lambda i: float(med[i]), reverse=True)
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(order):
        m, u = float(med[i]), float(upp[i])
        pct = _percentile(m, sorted_pop)
        keep = (budget > 0 and rank < budget) or pct >= (1.0 - keep_frac)
        if keep_threshold is not None:
            keep = keep or m >= keep_threshold
        if keep:
            priority = "high"
        elif pct >= 0.50 or (keep_threshold is not None and u >= keep_threshold):
            priority = "medium"
        else:
            priority = "low"
        out.append(
            {
                "rank": rank,
                "result_id": ids[i],
                "predicted_blimp": round(m, 4),
                "predicted_blimp_upper": round(u, 4),
                "train_percentile": round(pct, 4),
                "priority": priority,
                "recommended_for_eval": bool(keep),
                "hard_gate": False,
                "advisory_only": True,
            }
        )
    return out


def _run_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    model = load_model(out_dir)
    results = score_and_gate(
        args.db,
        model,
        keep_frac=args.keep_frac,
        keep_threshold=args.keep_threshold,
        only_missing=not args.score_all,
        require_stage1=args.require_stage1,
        require_screened=not args.no_require_screened,
        limit=args.limit,
        budget=args.budget,
    )
    score_out = Path(args.score_out)
    score_out.parent.mkdir(parents=True, exist_ok=True)
    with score_out.open("w", encoding="utf-8") as handle:
        for rec in results:  # already ranked best-predicted-first
            handle.write(json.dumps(rec) + "\n")
    n = len(results)
    kept = sum(1 for r in results if r["recommended_for_eval"])
    if not args.quiet:
        budget_note = f" budget={args.budget}" if args.budget else ""
        print(
            f"scored {n} candidate rows (ranked best-first{budget_note}); "
            f"priority=high for {kept} ({kept / max(n, 1):.0%}). "
            "No rows are marked skip; this is advisory until frontier-recall validation passes."
        )
        for rec in results[:10]:
            print(
                f"  #{rec['rank']:<4d} {rec['priority']:6s} pred={rec['predicted_blimp']:.3f} "
                f"(top {1 - rec['train_percentile']:.0%}) {rec['result_id']}"
            )
        print(f"wrote ranked worklist: {score_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BLiMP cheap-metric surrogate")
    parser.add_argument("--db", default=str(RUNS_DB), type=str)
    parser.add_argument("--target", default="blimp_overall_accuracy", type=str)
    parser.add_argument("--no-size", action="store_true", help="drop param/steps/flops")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR), type=str)
    parser.add_argument("--quiet", action="store_true")
    # --score priority mode: load the trained model and prioritize un-evaluated
    # rows. Low-priority rows are advisory, not discardable.
    parser.add_argument(
        "--score",
        action="store_true",
        help="score un-evaluated rows with the saved model instead of training",
    )
    parser.add_argument(
        "--keep-frac",
        default=0.30,
        type=float,
        help="priority=high for the top this-fraction by predicted BLiMP",
    )
    parser.add_argument(
        "--budget",
        default=0,
        type=int,
        help="GPU budget hint: priority=high for the top-N by predicted BLiMP (0 = use keep-frac)",
    )
    parser.add_argument(
        "--keep-threshold",
        default=None,
        type=float,
        help="also mark priority=high for any row with predicted BLiMP >= this absolute value",
    )
    parser.add_argument(
        "--score-all",
        action="store_true",
        help="score ALL rows, not only those missing a BLiMP eval",
    )
    parser.add_argument("--require-stage1", action="store_true")
    parser.add_argument(
        "--no-require-screened",
        action="store_true",
        help="score even rows where the primary cheap screen was never computed",
    )
    parser.add_argument("--limit", default=5000, type=int)
    parser.add_argument(
        "--score-out",
        default="research/reports/blimp_surrogate_scores.jsonl",
        type=str,
    )
    args = parser.parse_args(argv)

    features = list(FEATURE_COLUMNS) + ([] if args.no_size else list(SIZE_COLUMNS))
    if args.score:
        return _run_score(args)
    X, y, families, names = load_data(args.db, target=args.target, features=features)
    report = evaluate(X, y, families, names)
    report.target = args.target
    out_dir = Path(args.out_dir)
    model_path = train_and_save(X, y, names, args.target, out_dir)
    (out_dir / "report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.quiet:
        print(
            f"target={report.target}  rows={report.n_rows}  families={report.n_families}  "
            f"features={len(names)}"
        )
        for f in report.findings:
            print(f"  - {f}")
        print(f"saved model -> {model_path}")
        print(f"saved report -> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
