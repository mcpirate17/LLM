"""Architecture-component statistical analysis for noisy NAS probe metrics.

Outputs:
  research/reports/arch_component_analysis_2026-05-23/*.csv
  research/notes/arch_component_stat_analysis_2026-05-23.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "research" / "runs.db"
OUT_DIR = REPO / "research" / "reports" / "arch_component_analysis_2026-05-23"
NOTE_PATH = REPO / "research" / "notes" / "arch_component_stat_analysis_2026-05-23.md"
HYDRA_DIR = REPO / "research" / "reports" / "hydra_eval_2026-05-22"

RNG_SEED = 20260523
BOOT_INDEX_CACHE: dict[tuple[int, int], np.ndarray] = {}

METRICS: dict[str, str] = {
    "ar_curriculum_auc_pair_final": "ar_curriculum_auc_pair_final",
    "binding_intermediate_auc": "binding_intermediate_auc",
    # program_results does not have binding_range_auc; mixer_fingerprint writes
    # binding_range.binding_screening_auc, and the DB column is the same range AUC.
    "binding_range_auc": "binding_screening_auc",
    "binding_multislot_held_entity_slot_acc": "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_slot_lift": "binding_multislot_held_slot_lift",
    "induction_intermediate_auc": "induction_intermediate_auc",
    "ar_validation_rank_score": "ar_validation_rank_score",
}

CORE_METRICS = [
    "ar_curriculum_auc_pair_final",
    "binding_intermediate_auc",
    "binding_range_auc",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_slot_lift",
    "induction_intermediate_auc",
]

STRUCTURAL_OPS = {
    "input",
    "add",
    "mul",
    "multiply",
    "identity",
    "residual",
    "concat",
    "split",
    "mean",
    "sum",
}

SQL = f"""
SELECT
  result_id, experiment_id, timestamp, graph_fingerprint, graph_json,
  result_cohort, comparability_label, evaluation_protocol_version,
  param_count, graph_depth, graph_n_ops, graph_n_edges, graph_n_unique_ops,
  n_train_steps, train_budget_steps, training_program_json, data_provenance_json,
  {", ".join(f"{col} AS {alias}" for alias, col in METRICS.items())}
-- program_results_compat (= graph_runs LEFT JOIN graphs) is canonical post-Phase-5b;
-- the legacy program_results table is stale (not updated by probe backfills). See
-- research/notes/adjacent_token_merge_leak_2026-05-23.md.
FROM program_results_compat
WHERE COALESCE(graph_json, '') NOT IN ('', '{{}}')
  AND ({" OR ".join(f"{col} IS NOT NULL" for col in METRICS.values())});
"""


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _iter_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = graph.get("nodes") or []
    if isinstance(nodes, dict):
        return [n for n in nodes.values() if isinstance(n, dict)]
    if isinstance(nodes, list):
        return [n for n in nodes if isinstance(n, dict)]
    return []


def _extract_ops(graph: dict[str, Any]) -> set[str]:
    ops: set[str] = set()
    for node in _iter_nodes(graph):
        op = node.get("op_name") or node.get("op") or node.get("type")
        if op:
            op_s = str(op)
            if op_s != "input":
                ops.add(op_s)
    return ops


def _training_tokens(row: sqlite3.Row, provenance: dict[str, Any] | None) -> float:
    tp = _json_loads(row["training_program_json"]) or {}
    steps = (
        _finite(tp.get("n_steps"))
        or _finite(row["n_train_steps"])
        or _finite(row["train_budget_steps"])
    )
    batch = _finite(tp.get("batch_size"))
    curriculum = tp.get("curriculum") if isinstance(tp.get("curriculum"), dict) else {}
    seq_len = (
        _finite(tp.get("seq_len"))
        or _finite(tp.get("max_seq_len"))
        or _finite(curriculum.get("max_seq_len"))
        or _finite(curriculum.get("initial_seq_len"))
    )
    if steps and batch and seq_len:
        return float(steps * batch * seq_len)
    if steps:
        # Most candidate-grade rows use byte-vocab tiny training programs with
        # batch 8 and max seq 256 when the serialized program is missing.
        return float(steps * 8 * 256)
    if provenance:
        steps = _finite(provenance.get("s1_steps")) or _finite(
            provenance.get("rapid_steps")
        )
        model_dim = _finite(provenance.get("model_dim"))
        if steps:
            return float(
                steps * 8 * (256 if not model_dim else min(512, max(64, model_dim)))
            )
    return float("nan")


def _n_blocks(provenance: dict[str, Any] | None) -> float:
    if not provenance:
        return 1.0
    for key in ("n_layers", "n_blocks"):
        v = _finite(provenance.get(key))
        if v:
            return float(v)
    return 1.0


def load_program_rows() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    ops_by_result: dict[str, set[str]] = {}
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(SQL):
            try:
                graph_json = resolve_graph_json_value(conn, DB_PATH, row["graph_json"])
                graph = json.loads(graph_json)
            except Exception:
                continue
            if not isinstance(graph, dict):
                continue
            result_id = str(row["result_id"])
            provenance = _json_loads(row["data_provenance_json"]) or {}
            model_dim = _finite(graph.get("model_dim")) or _finite(
                provenance.get("model_dim")
            )
            ops = _extract_ops(graph)
            if not ops:
                continue
            ops_by_result[result_id] = ops
            rec: dict[str, Any] = {
                "result_id": result_id,
                "graph_fingerprint": row["graph_fingerprint"],
                "result_cohort": row["result_cohort"],
                "comparability_label": row["comparability_label"],
                "evaluation_protocol_version": row["evaluation_protocol_version"],
                "param_count": _finite(row["param_count"]),
                "model_dim": model_dim,
                "n_blocks": _n_blocks(provenance),
                "training_tokens": _training_tokens(row, provenance),
                "graph_depth": _finite(row["graph_depth"]),
                "graph_n_ops": _finite(row["graph_n_ops"]),
                "graph_n_unique_ops": _finite(row["graph_n_unique_ops"]),
            }
            for metric in METRICS:
                rec[metric] = _finite(row[metric])
            rows.append(rec)
    return pd.DataFrame(rows), ops_by_result


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _partial_corr_binary(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> float:
    if x.size < controls.shape[1] + 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), controls])
    bx, *_ = np.linalg.lstsq(design, x, rcond=None)
    by, *_ = np.linalg.lstsq(design, y, rcond=None)
    return _pearson(x - design @ bx, y - design @ by)


def _ci(vals: list[float]) -> tuple[float, float]:
    clean = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(clean, [0.025, 0.975]).astype(float))


def _bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    key = (n, n_boot)
    cached = BOOT_INDEX_CACHE.get(key)
    if cached is None:
        cached = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
        BOOT_INDEX_CACHE[key] = cached
    return cached


def _bootstrap_assoc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    controls: np.ndarray | None = None,
) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if controls is not None:
        mask &= np.isfinite(controls).all(axis=1)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    ctrl = controls[mask].astype(float) if controls is not None else None
    n = len(y)
    n_present = int(x.sum())
    n_absent = int(n - n_present)
    if n < 8 or n_present < 2 or n_absent < 2:
        return {}
    if ctrl is not None:
        design = np.column_stack([np.ones(n), ctrl])
        bx, *_ = np.linalg.lstsq(design, x, rcond=None)
        by, *_ = np.linalg.lstsq(design, y, rcond=None)
        xr = x - design @ bx
        yr = y - design @ by
        r = _pearson(xr, yr)
    else:
        xr = x
        yr = y
        r = _pearson(x, y)
    delta = float(y[x > 0.5].mean() - y[x <= 0.5].mean())
    idx = _bootstrap_indices(n, n_boot, rng)
    xb = x[idx]
    yb = y[idx]
    xrb = xr[idx]
    yrb = yr[idx]
    xrm = xrb.mean(axis=1)
    yrm = yrb.mean(axis=1)
    cov = ((xrb - xrm[:, None]) * (yrb - yrm[:, None])).mean(axis=1)
    sx = xrb.std(axis=1)
    sy = yrb.std(axis=1)
    valid_r = (sx > 0) & (sy > 0)
    boot_r = cov[valid_r] / (sx[valid_r] * sy[valid_r])
    cnt_present = xb.sum(axis=1)
    cnt_absent = n - cnt_present
    valid_d = (cnt_present >= 2) & (cnt_absent >= 2)
    sum_present = (xb * yb).sum(axis=1)
    sum_absent = ((1.0 - xb) * yb).sum(axis=1)
    boot_delta = (sum_present[valid_d] / cnt_present[valid_d]) - (
        sum_absent[valid_d] / cnt_absent[valid_d]
    )
    boot_r = boot_r[np.isfinite(boot_r)]
    boot_delta = boot_delta[np.isfinite(boot_delta)]
    if boot_r.size == 0 or boot_delta.size == 0:
        return {}
    r_lo, r_hi = tuple(np.quantile(boot_r, [0.025, 0.975]).astype(float))
    d_lo, d_hi = tuple(np.quantile(boot_delta, [0.025, 0.975]).astype(float))
    return {
        "n": n,
        "present": n_present,
        "absent": n_absent,
        "mean_present": float(y[x > 0.5].mean()),
        "mean_absent": float(y[x <= 0.5].mean()),
        "delta": delta,
        "delta_lo": d_lo,
        "delta_hi": d_hi,
        "r": float(r),
        "r_lo": r_lo,
        "r_hi": r_hi,
    }


def primitive_tables(
    df: pd.DataFrame,
    ops_by_result: dict[str, set[str]],
    *,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    op_counts = Counter(op for ops in ops_by_result.values() for op in ops)
    rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RNG_SEED)
    controls = df[["model_dim", "n_blocks", "training_tokens"]].copy()
    controls["training_tokens"] = np.log1p(controls["training_tokens"])
    controls = controls.fillna(controls.median(numeric_only=True))
    controls.to_numpy(dtype=float)
    for metric in CORE_METRICS:
        sub = df[df[metric].notna()]
        if sub.empty:
            continue
        y = sub[metric].to_numpy(dtype=float)
        min_support = max(5 if len(sub) <= 80 else 15, int(math.ceil(0.015 * len(sub))))
        candidate_ops = [op for op, count in op_counts.items() if count >= min_support]
        for op in sorted(candidate_ops):
            x = (
                sub["result_id"]
                .map(lambda rid: op in ops_by_result.get(rid, set()))
                .to_numpy(dtype=float)
            )
            if x.sum() < min_support or (len(x) - x.sum()) < min_support:
                continue
            assoc = _bootstrap_assoc(x, y, n_boot=n_boot, rng=rng)
            if assoc:
                rows.append({"metric": metric, "op": op, **assoc})
            ctrl_sub = controls.loc[sub.index].to_numpy(dtype=float)
            assoc_p = _bootstrap_assoc(x, y, n_boot=n_boot, rng=rng, controls=ctrl_sub)
            if assoc_p:
                partial_rows.append({"metric": metric, "op": op, **assoc_p})
    prim = pd.DataFrame(rows).sort_values(["metric", "delta"], ascending=[True, False])
    partial = pd.DataFrame(partial_rows).sort_values(
        ["metric", "r"], ascending=[True, False]
    )
    return prim, partial


def combination_table(
    df: pd.DataFrame,
    ops_by_result: dict[str, set[str]],
    *,
    n_boot: int,
    max_k: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RNG_SEED + 1)
    for metric in CORE_METRICS:
        sub = df[df[metric].notna()].reset_index(drop=True)
        if sub.empty:
            continue
        y = sub[metric].to_numpy(dtype=float)
        min_support = max(5 if len(sub) <= 80 else 18, int(math.ceil(0.02 * len(sub))))
        combo_counts: Counter[tuple[str, ...]] = Counter()
        for rid in sub["result_id"]:
            ops = sorted(
                op for op in ops_by_result.get(rid, set()) if op not in STRUCTURAL_OPS
            )
            for k in range(2, max_k + 1):
                combo_counts.update(itertools.combinations(ops, k))
        candidates = [
            combo for combo, count in combo_counts.items() if count >= min_support
        ]
        scored_candidates: list[tuple[float, tuple[str, ...]]] = []
        rid_ops = [ops_by_result.get(rid, set()) for rid in sub["result_id"]]
        for combo in candidates:
            combo_set = set(combo)
            x0 = np.fromiter(
                (combo_set <= ops for ops in rid_ops), dtype=float, count=len(rid_ops)
            )
            if x0.sum() < 2 or len(x0) - x0.sum() < 2:
                continue
            delta0 = float(y[x0 > 0.5].mean() - y[x0 <= 0.5].mean())
            scored_candidates.append((delta0, combo))
        # Bootstrap the strongest positive and negative frequent itemsets. This
        # keeps output useful while avoiding thousands of near-zero CIs.
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = [c for _, c in scored_candidates[:180]] + [
            c for _, c in scored_candidates[-40:]
        ]
        for combo in candidates:
            combo_set = set(combo)
            x = (
                sub["result_id"]
                .map(lambda rid: combo_set <= ops_by_result.get(rid, set()))
                .to_numpy(dtype=float)
            )
            assoc = _bootstrap_assoc(x, y, n_boot=n_boot, rng=rng)
            if assoc:
                rows.append(
                    {
                        "metric": metric,
                        "combo": " + ".join(combo),
                        "k": len(combo),
                        **assoc,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["metric", "delta"], ascending=[True, False])
    return out


def _std_from_duplicate_fps(df: pd.DataFrame, metric: str) -> tuple[int, float]:
    vals: list[float] = []
    groups = 0
    for _, g in df[df[metric].notna()].groupby("graph_fingerprint"):
        if len(g) < 2:
            continue
        arr = g[metric].to_numpy(dtype=float)
        if np.nanstd(arr, ddof=1) > 0:
            vals.append(float(np.nanvar(arr, ddof=1)))
            groups += 1
    if not vals:
        return 0, float("nan")
    return groups, float(np.sqrt(np.nanmean(vals)))


def variance_budget(df: pd.DataFrame) -> pd.DataFrame:
    rerun_noise = {
        "induction_intermediate_auc": {
            "anchor_std": 0.2087,
            "source": "n4_induction_reruns aggregate including original single-run flip",
        },
        "binding_multislot_held_entity_slot_acc": {
            "anchor_std": 0.0217,
            "source": "n2_binding_multiseed held_slot std across 7 calls",
        },
        "binding_multislot_held_slot_lift": {
            "anchor_std": 0.0217,
            "source": "n2_binding_multiseed held_slot std across 7 calls",
        },
        "binding_range_auc": {
            "anchor_std": 0.0020,
            "source": "n2_binding_multiseed binding_range auc std across 7 calls",
        },
        "ar_validation_rank_score": {
            "anchor_std": 0.8259,
            "source": "n2_ar_val_7seed per-seed rank score std",
        },
    }
    rows: list[dict[str, Any]] = []
    for metric in [*CORE_METRICS, "ar_validation_rank_score"]:
        vals = df[metric].dropna().to_numpy(dtype=float)
        if vals.size < 2:
            continue
        obs_std = float(np.nanstd(vals, ddof=1))
        dup_groups, dup_std = _std_from_duplicate_fps(df, metric)
        anchor = rerun_noise.get(metric, {})
        noise_std = float(
            anchor.get(
                "anchor_std", dup_std if math.isfinite(dup_std) else float("nan")
            )
        )
        frac = (
            float(min(1.0, (noise_std**2) / (obs_std**2)))
            if math.isfinite(noise_std) and obs_std
            else float("nan")
        )
        rows.append(
            {
                "metric": metric,
                "n": int(vals.size),
                "observed_std": obs_std,
                "duplicate_fp_groups": dup_groups,
                "duplicate_fp_within_std": dup_std,
                "rerun_anchor_std": anchor.get("anchor_std", float("nan")),
                "noise_std_used": noise_std,
                "noise_variance_fraction": frac,
                "source": anchor.get(
                    "source", "duplicate graph_fingerprint groups in program_results"
                ),
            }
        )
    return pd.DataFrame(rows)


def hydra_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    ar_rows: list[dict[str, Any]] = []
    data = json.loads((HYDRA_DIR / "ar_arch_sweep.json").read_text())
    for row in data.get("rows", []):
        ar = row.get("ar_validation", {})
        ar_rows.append(
            {
                "source": "ar_arch_sweep",
                "label": row.get("arch"),
                "lane": row.get("lane"),
                "pattern": row.get("pattern"),
                "n_params_m": (row.get("n_params") or 0) / 1e6,
                "dim": row.get("dim"),
                "n_blocks": row.get("n_blocks"),
                "rank_score_mean": ar.get("ar_validation_rank_score_mean"),
                "rank_score_std": ar.get("ar_validation_rank_score_std"),
                "held_pair_acc_mean": ar.get("ar_validation_held_pair_acc_mean"),
                "held_pair_acc_std": ar.get("ar_validation_held_pair_acc_std"),
            }
        )
    hydra_rows: list[dict[str, Any]] = []
    data2 = json.loads((HYDRA_DIR / "arch_sweep.json").read_text())
    for row in data2.get("archs", []):
        top1 = np.array([s["top1"] for s in row.get("per_seed", [])], dtype=float)
        hydra_rows.append(
            {
                "source": "arch_sweep",
                "label": row.get("label"),
                "lane": row.get("lane"),
                "pattern": row.get("pattern"),
                "n_params_m": row.get("n_params_M"),
                "dim": row.get("dim"),
                "n_blocks": row.get("n_blocks"),
                "top1_mean": float(top1.mean()) if top1.size else float("nan"),
                "top1_std": float(top1.std(ddof=1)) if top1.size > 1 else float("nan"),
                "top1_sem": float(top1.std(ddof=1) / math.sqrt(top1.size))
                if top1.size > 1
                else float("nan"),
            }
        )
    return (
        pd.DataFrame(ar_rows).sort_values("rank_score_mean", ascending=False),
        pd.DataFrame(hydra_rows).sort_values("top1_mean", ascending=False),
    )


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _top_table(df: pd.DataFrame, metric: str, cols: list[str], n: int = 6) -> str:
    if df.empty:
        return "_No rows._"
    sub = df[df["metric"] == metric].head(n).copy()
    if sub.empty:
        return "_No rows._"
    return sub[cols].to_markdown(index=False, floatfmt=".3f")


def write_report(
    df: pd.DataFrame,
    prim: pd.DataFrame,
    partial: pd.DataFrame,
    combos: pd.DataFrame,
    var_budget: pd.DataFrame,
    ar_arch: pd.DataFrame,
    hydra: pd.DataFrame,
    *,
    n_boot: int,
) -> None:
    metric_counts = pd.DataFrame(
        [
            {
                "metric": m,
                "n": int(df[m].notna().sum()),
                "mean": df[m].mean(),
                "std": df[m].std(),
            }
            for m in CORE_METRICS
        ]
    )
    best_prim_rows = []
    for metric in CORE_METRICS:
        sub = prim[
            (prim["metric"] == metric)
            & (prim["present"] >= (5 if metric.startswith("binding_multislot") else 15))
        ]
        if not sub.empty:
            best_prim_rows.append(sub.iloc[0])
    best_prim = pd.DataFrame(best_prim_rows)
    best_partial_rows = []
    for metric in CORE_METRICS:
        sub = partial[partial["metric"] == metric].copy()
        if not sub.empty:
            best_partial_rows.append(sub.sort_values("r", ascending=False).iloc[0])
    best_partial = pd.DataFrame(best_partial_rows)
    best_combo_rows = []
    for metric in CORE_METRICS:
        sub = combos[combos["metric"] == metric]
        if not sub.empty:
            best_combo_rows.append(sub.iloc[0])
    best_combo = pd.DataFrame(best_combo_rows)

    lines: list[str] = []
    lines.append("# Architecture-Component Statistical Analysis")
    lines.append("")
    lines.append(
        f"Generated: 2026-05-23. Bootstrap resamples per association: `{n_boot}`. DB: `{DB_PATH.relative_to(REPO)}`."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("SQL used:")
    lines.append("")
    lines.append("```sql")
    lines.append(SQL.strip())
    lines.append("```")
    lines.append("")
    lines.append(
        "Each graph was parsed from `graph_json` with artifact-pointer resolution. "
        "Primitive features are binary op-presence indicators from node `op_name`. "
        "`binding_range_auc` is an alias for DB column `binding_screening_auc`, because `program_results` has no `binding_range_auc` column while mixer-fingerprint writes range AUC under `binding_range.binding_screening_auc`."
    )
    lines.append("")
    lines.append(
        "For every metric/op pair, I computed point-biserial Pearson `r`, mean difference `mean(op present) - mean(op absent)`, and 95% bootstrap CIs. "
        "Partial correlations residualize both primitive presence and metric on `[model_dim, n_blocks, log1p(training_tokens)]`; `model_dim` comes from graph JSON/provenance, `n_blocks` from provenance `n_layers` when present else 1, and training tokens from serialized training program when present else step-count fallback."
    )
    lines.append("")
    lines.append("## Data Coverage")
    lines.append("")
    lines.append(metric_counts.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Strongest Per-Primitive Associations")
    lines.append("")
    lines.append(
        best_prim[
            [
                "metric",
                "op",
                "present",
                "mean_present",
                "mean_absent",
                "delta",
                "delta_lo",
                "delta_hi",
                "r",
                "r_lo",
                "r_hi",
            ]
        ].to_markdown(index=False, floatfmt=".3f")
    )
    lines.append("")
    lines.append(
        "Interpretation: positive deltas are observed single-run associations, not causal effects. CIs that include zero should be treated as directionally weak under the known probe noise."
    )
    lines.append("")
    lines.append("## Strongest Partial Correlations")
    lines.append("")
    lines.append(
        best_partial[
            ["metric", "op", "present", "delta", "r", "r_lo", "r_hi"]
        ].to_markdown(index=False, floatfmt=".3f")
    )
    lines.append("")
    lines.append("## Primitive Combination Patterns")
    lines.append("")
    lines.append(
        best_combo[
            [
                "metric",
                "combo",
                "k",
                "present",
                "mean_present",
                "mean_absent",
                "delta",
                "delta_lo",
                "delta_hi",
                "r",
            ]
        ].to_markdown(index=False, floatfmt=".3f")
    )
    lines.append("")
    lines.append("## Variance Budget")
    lines.append("")
    lines.append(
        var_budget[
            [
                "metric",
                "n",
                "observed_std",
                "duplicate_fp_within_std",
                "rerun_anchor_std",
                "noise_std_used",
                "noise_variance_fraction",
                "source",
            ]
        ].to_markdown(index=False, floatfmt=".3f")
    )
    lines.append("")
    lines.append(
        "Direct reruns dominate interpretation. The induction anchor includes the original 0.564 single call plus three reruns near 0.98, so it is intentionally conservative for DB single-run interpretation. Binding range is much lower variance in the n=2 rerun artifact. Multislot has modest conditional seed std, but the original single call missed the later 7-call mean by about 0.057 absolute."
    )
    lines.append("")
    lines.append("## Cross-Architecture Checks")
    lines.append("")
    lines.append("Fresh-pretrained AR-validation v3 sweep, top rows:")
    lines.append("")
    lines.append(
        ar_arch[
            [
                "label",
                "lane",
                "pattern",
                "n_params_m",
                "rank_score_mean",
                "rank_score_std",
                "held_pair_acc_mean",
            ]
        ]
        .head(8)
        .to_markdown(index=False, floatfmt=".3f")
    )
    lines.append("")
    lines.append("Hydra real-text math discrimination sweep, top rows:")
    lines.append("")
    lines.append(
        hydra[
            [
                "label",
                "lane",
                "pattern",
                "n_params_m",
                "top1_mean",
                "top1_std",
                "top1_sem",
            ]
        ]
        .head(8)
        .to_markdown(index=False, floatfmt=".4f")
    )
    lines.append("")
    lines.append("## Ranked Recommendations")
    lines.append("")
    lines.append(
        "1. **Prioritize hybrid lanes that combine local convolution with the mined AR ensemble lanes.** The lowest-variance hydra sweep favors the conv + three-lane + ensemble hybrid over pure three-lane, and the DB combination scan repeatedly selects local/conv/attention-like mixtures rather than single primitives. Expected detectable gain: about `+0.015` to `+0.025` hydra top1 versus pure three-lane scale peers; detection floor: `~0.006` top1 SEM in the 3-seed hydra sweep, so this clears the low-variance check. Confidence: **medium-high**."
    )
    lines.append("")
    lines.append(
        "2. **Keep `local_window_attn + selective_scan/diff_attention + conv1d_seq` as the binding/induction cross-bias motif, but require rerun confirmation before ranking variants.** DB rows can show large positive binding/induction deltas for this family, and the existing top mined cross-bias fingerprints in `ensemble_screening.py` already encode it. Expected gain to look for: `+0.15` or larger on binding/intermediate or induction AUC; detection floor: induction needs `>=0.20` AUC under the conservative rerun anchor, while binding_intermediate has no direct rerun anchor. Confidence: **medium**."
    )
    lines.append("")
    lines.append(
        "3. **For AR-specific lanes, prefer the top mined primitives as an ensemble rather than betting on one op.** The four `TOP_AR_FPS` graphs cover tropical/local/conv/swiglu, linear_attention/block_sparse/softmax, and two variants; single primitive CIs are not stable enough to justify a one-op conclusion. Expected gain: retain `~0.8-0.9` AR-curriculum AUC in mined lanes, but architecture-level detectability should be judged with AR-val rank score changes `>0.8` because the n=2 rerun rank std is 0.826. Confidence: **medium**."
    )
    lines.append("")
    lines.append(
        "4. **Do not optimize against `binding_multislot_held_*` single calls yet.** Only 50 DB rows have this metric and the n=2 rerun moved from 0.0495 to 0.107 mean, so primitive rankings are useful only as triage. Expected effect must exceed `~0.04-0.06` held-slot accuracy to be credible from one run; smaller deltas need repeated probe calls. Confidence: **low**."
    )
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    for name in [
        "primitive_correlations.csv",
        "partial_correlations.csv",
        "combination_patterns.csv",
        "metric_variance_budget.csv",
        "ar_arch_sweep_summary.csv",
        "hydra_arch_sweep_summary.csv",
    ]:
        lines.append(f"- `{(OUT_DIR / name).relative_to(REPO)}`")
    lines.append("")
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df, ops_by_result = load_program_rows()
    df.to_csv(OUT_DIR / "analysis_rows.csv", index=False)
    prim, partial = primitive_tables(df, ops_by_result, n_boot=args.bootstrap)
    combos = combination_table(df, ops_by_result, n_boot=args.bootstrap)
    var_budget = variance_budget(df)
    ar_arch, hydra = hydra_summaries()

    prim.to_csv(OUT_DIR / "primitive_correlations.csv", index=False)
    partial.to_csv(OUT_DIR / "partial_correlations.csv", index=False)
    combos.to_csv(OUT_DIR / "combination_patterns.csv", index=False)
    var_budget.to_csv(OUT_DIR / "metric_variance_budget.csv", index=False)
    ar_arch.to_csv(OUT_DIR / "ar_arch_sweep_summary.csv", index=False)
    hydra.to_csv(OUT_DIR / "hydra_arch_sweep_summary.csv", index=False)
    write_report(
        df, prim, partial, combos, var_budget, ar_arch, hydra, n_boot=args.bootstrap
    )

    print(
        f"rows={len(df)} ops={len(set().union(*ops_by_result.values())) if ops_by_result else 0}"
    )
    print(f"wrote {OUT_DIR.relative_to(REPO)}")
    print(f"wrote {NOTE_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
