"""Extensions to Codex's trust-tier component analysis.

Adds three cuts the base report (research/reports/arch_component_trust_tier_2026-05-24/
report.md) does not cover:

  1. Dosage curves — does op COUNT (not just presence) matter? Bin 0/1/2/3+.
  2. Depth stratification — does per-op effect change across graph_depth tertiles?
  3. Per-template correlation — what's driving AR/induction at the template level
     (program_graph_features.template_name), not just per-op?

Primary signals only (ar_curriculum_auc_pair_final, induction_intermediate_auc).
No bootstrapping over noise here — Codex's base report covers that. CIs are
standard percentile-bootstrap on the row resample (1000 draws), no metric-noise
injection. The dose-vs-presence comparison and depth-stratum interaction are
the new contributions.

Run:
    source /home/tim/venvs/llm/bin/activate
    python -m research.tools.arch_component_extensions_20260524
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "research" / "runs.db"
OUT_DIR = REPO / "research" / "reports" / "arch_component_trust_tier_2026-05-24"
ADDENDUM_PATH = REPO / "research" / "notes" / "arch_component_extensions_2026-05-24.md"

RNG_SEED = 20260524
N_BOOT = 1000

LEAK_BUCKET_OPS = frozenset({"adjacent_token_merge", "token_merge"})
STRUCTURAL_OPS = frozenset(
    {"input", "output", "identity", "add", "mul", "multiply", "concat", "split"}
)

# Primary, leak-immune metrics only.
PRIMARY_METRICS = ("ar_curriculum_auc_pair_final", "induction_intermediate_auc")

# Ops whose dosage curves we care about. Picked from the base report's
# top-effect ops + the live-lead ops in MEMORY.md (selective_scan, conv1d_seq).
DOSAGE_OPS = (
    "selective_scan",
    "conv1d_seq",
    "tropical_attention",
    "softmax_attention",
    "rope_rotate",
    "swiglu_mlp",
    "sparsemax_attention",
    "multiscale_wavelet",
    "local_window_attn",
    "linear_attention",
)

SQL = """
SELECT
  pr.result_id,
  pr.graph_fingerprint,
  pr.graph_json,
  pr.param_count,
  pr.graph_depth,
  pr.graph_n_ops,
  pr.n_train_steps,
  pr.ar_curriculum_auc_pair_final,
  pr.induction_intermediate_auc,
  gf.template_name,
  gf.templates_json
FROM program_results pr
LEFT JOIN program_graph_features gf USING(result_id)
WHERE COALESCE(pr.graph_json, '') NOT IN ('', '{}')
  AND (pr.ar_curriculum_auc_pair_final IS NOT NULL
       OR pr.induction_intermediate_auc IS NOT NULL);
"""


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _json_loads(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
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


def _op_counts(graph: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for node in _iter_nodes(graph):
        op = node.get("op_name") or node.get("op") or node.get("type")
        if op and str(op) not in STRUCTURAL_OPS:
            counter[str(op)] += 1
    return counter


def _templates_used(graph: dict[str, Any]) -> list[str]:
    md = graph.get("metadata") or {}
    val = md.get("templates_used") or []
    if isinstance(val, list):
        return [str(t) for t in val]
    return []


def load_rows() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(SQL)
        cols = [d[0] for d in cur.description]
        records = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            g = _json_loads(d.pop("graph_json"))
            if not isinstance(g, dict):
                continue
            counts = _op_counts(g)
            ops_set = set(counts.keys())
            d["op_counts"] = counts
            d["ops_set"] = ops_set
            d["has_leak_op"] = bool(ops_set & LEAK_BUCKET_OPS)
            d["templates_used_meta"] = _templates_used(g)
            d["templates_db"] = _json_loads(d.pop("templates_json")) or []
            records.append(d)
    finally:
        conn.close()
    df = pd.DataFrame.from_records(records)
    for col in (
        "ar_curriculum_auc_pair_final",
        "induction_intermediate_auc",
        "param_count",
        "graph_depth",
        "graph_n_ops",
        "n_train_steps",
    ):
        df[col] = df[col].map(_finite)
    return df


# --------------------------- Bootstrap helpers ----------------------------- #


def _boot_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    if n < 2:
        return (mean, float("nan"), float("nan"))
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (mean, lo, hi)


def _boot_delta_ci(
    present: np.ndarray,
    absent: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if len(present) == 0 or len(absent) == 0:
        return (float("nan"), float("nan"), float("nan"))
    delta = float(present.mean() - absent.mean())
    p_draws = rng.integers(0, len(present), size=(n_boot, len(present)))
    a_draws = rng.integers(0, len(absent), size=(n_boot, len(absent)))
    deltas = present[p_draws].mean(axis=1) - absent[a_draws].mean(axis=1)
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return (delta, lo, hi)


# --------------------------- Cut 1: Dosage curves -------------------------- #


def _dose_bin(c: int) -> str:
    if c == 0:
        return "0"
    if c == 1:
        return "1"
    if c == 2:
        return "2"
    return "3+"


def dosage_curves(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for metric in PRIMARY_METRICS:
        sub = df.dropna(subset=[metric])
        for op in DOSAGE_OPS:
            counts = sub["op_counts"].map(lambda c, op=op: c.get(op, 0))
            for bucket in ("0", "1", "2", "3+"):
                bin_mask = counts.map(_dose_bin) == bucket
                values = sub.loc[bin_mask, metric].to_numpy()
                mean, lo, hi = _boot_mean_ci(values, rng)
                rows.append(
                    {
                        "metric": metric,
                        "op": op,
                        "dose_bin": bucket,
                        "n": int(bin_mask.sum()),
                        "mean": mean,
                        "mean_lo": lo,
                        "mean_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def dosage_top_vs_zero_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap CI on (count>=2) vs (count==0) — proper test for 'more is better'."""
    rng = np.random.default_rng(RNG_SEED + 2)
    rows = []
    for metric in PRIMARY_METRICS:
        sub = df.dropna(subset=[metric])
        for op in DOSAGE_OPS:
            counts = sub["op_counts"].map(lambda c, op=op: c.get(op, 0))
            high = sub.loc[counts >= 2, metric].to_numpy()
            low = sub.loc[counts == 1, metric].to_numpy()
            zero = sub.loc[counts == 0, metric].to_numpy()
            d_high_zero, lo_hz, hi_hz = _boot_delta_ci(high, zero, rng)
            d_high_low, lo_hl, hi_hl = _boot_delta_ci(high, low, rng)
            rows.append(
                {
                    "metric": metric,
                    "op": op,
                    "n_zero": len(zero),
                    "n_one": len(low),
                    "n_two_plus": len(high),
                    "delta_2plus_vs_0": d_high_zero,
                    "ci_lo_2plus_vs_0": lo_hz,
                    "ci_hi_2plus_vs_0": hi_hz,
                    "delta_2plus_vs_1": d_high_low,
                    "ci_lo_2plus_vs_1": lo_hl,
                    "ci_hi_2plus_vs_1": hi_hl,
                }
            )
    return pd.DataFrame(rows)


# ----------------------- Cut 2: Depth stratification ----------------------- #


def _depth_tertile(df: pd.DataFrame) -> pd.Series:
    d = df["graph_depth"].dropna()
    if d.empty:
        return pd.Series(["unknown"] * len(df), index=df.index)
    q1, q2 = np.nanquantile(d, [1 / 3, 2 / 3])

    def _bin(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "unknown"
        if v <= q1:
            return f"shallow_le_{q1:.0f}"
        if v <= q2:
            return f"mid_le_{q2:.0f}"
        return f"deep_gt_{q2:.0f}"

    return df["graph_depth"].map(_bin)


def depth_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """For top-effect ops, present/absent delta within each depth tertile."""
    rng = np.random.default_rng(RNG_SEED + 3)
    df = df.copy()
    df["depth_bin"] = _depth_tertile(df)
    target_ops = (
        "selective_scan",
        "conv1d_seq",
        "tropical_attention",
        "softmax_attention",
        "rope_rotate",
        "local_window_attn",
        "sparsemax_attention",
        "multiscale_wavelet",
    )
    rows = []
    for metric in PRIMARY_METRICS:
        sub = df.dropna(subset=[metric])
        for op in target_ops:
            for bin_label, group in sub.groupby("depth_bin"):
                present_mask = group["ops_set"].map(lambda s, op=op: op in s)
                present = group.loc[present_mask, metric].to_numpy()
                absent = group.loc[~present_mask, metric].to_numpy()
                if len(present) < 5 or len(absent) < 5:
                    continue
                delta, lo, hi = _boot_delta_ci(present, absent, rng)
                rows.append(
                    {
                        "metric": metric,
                        "op": op,
                        "depth_bin": bin_label,
                        "n_present": int(len(present)),
                        "n_absent": int(len(absent)),
                        "mean_present": float(present.mean()),
                        "mean_absent": float(absent.mean()),
                        "delta": delta,
                        "delta_lo": lo,
                        "delta_hi": hi,
                    }
                )
    return pd.DataFrame(rows).sort_values(["metric", "op", "depth_bin"])


# ----------------------- Cut 3: Per-template effects ----------------------- #


def template_effects(df: pd.DataFrame) -> pd.DataFrame:
    """For each template_name with n>=30, mean and bootstrap CI on primary
    metrics. Uses program_graph_features.template_name (single-template label
    per row); the multi-template templates_used list is reported as a separate
    Jaccard cut at the bottom if cardinality allows."""
    rng = np.random.default_rng(RNG_SEED + 4)
    rows = []
    for metric in PRIMARY_METRICS:
        sub = df.dropna(subset=[metric, "template_name"])
        global_mean = float(sub[metric].mean())
        for tmpl, group in sub.groupby("template_name"):
            if len(group) < 30:
                continue
            values = group[metric].to_numpy()
            other = sub.loc[sub["template_name"] != tmpl, metric].to_numpy()
            mean, lo, hi = _boot_mean_ci(values, rng)
            delta, dlo, dhi = _boot_delta_ci(values, other, rng)
            rows.append(
                {
                    "metric": metric,
                    "template_name": tmpl,
                    "n": int(len(group)),
                    "mean": mean,
                    "mean_lo": lo,
                    "mean_hi": hi,
                    "delta_vs_others": delta,
                    "delta_lo": dlo,
                    "delta_hi": dhi,
                    "global_mean": global_mean,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "delta_vs_others"], ascending=[True, False])


def template_pair_effects(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """For templates_used multi-template graphs: which 2-template pairs from
    metadata.templates_used correlate with primary signals."""
    rng = np.random.default_rng(RNG_SEED + 5)
    rows = []
    for metric in PRIMARY_METRICS:
        sub = df.dropna(subset=[metric])
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_vals: dict[tuple[str, str], list[float]] = {}
        for tmpls, val in zip(sub["templates_used_meta"], sub[metric]):
            uniq = sorted(set(tmpls))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    key = (uniq[i], uniq[j])
                    pair_counts[key] += 1
                    pair_vals.setdefault(key, []).append(val)
        all_vals = sub[metric].to_numpy()
        for key, vals in pair_vals.items():
            if pair_counts[key] < min_n:
                continue
            present = np.asarray(vals, dtype=float)
            sub[metric].notna()  # placeholder; just use complement
            absent = np.array([v for v in all_vals if v not in vals[:1]])  # cheap
            # absent is rough; we recompute properly:
            absent = sub[metric].to_numpy()
            delta, lo, hi = _boot_delta_ci(present, absent, rng)
            rows.append(
                {
                    "metric": metric,
                    "template_pair": " + ".join(key),
                    "n_present": int(len(present)),
                    "mean_present": float(present.mean()),
                    "mean_all": float(absent.mean()),
                    "delta_vs_all": delta,
                    "delta_lo": lo,
                    "delta_hi": hi,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["metric", "delta_vs_all"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ------------------------------- Reporting --------------------------------- #


def _md_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if float_cols:
        df = df.copy()
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return df.to_markdown(index=False)


def write_addendum(
    dosage: pd.DataFrame,
    dose_delta: pd.DataFrame,
    depth: pd.DataFrame,
    templates: pd.DataFrame,
    template_pairs: pd.DataFrame,
) -> None:
    float_curve = ["mean", "mean_lo", "mean_hi"]
    float_delta = [
        "delta_2plus_vs_0",
        "ci_lo_2plus_vs_0",
        "ci_hi_2plus_vs_0",
        "delta_2plus_vs_1",
        "ci_lo_2plus_vs_1",
        "ci_hi_2plus_vs_1",
    ]
    float_depth = ["mean_present", "mean_absent", "delta", "delta_lo", "delta_hi"]
    float_tmpl = [
        "mean",
        "mean_lo",
        "mean_hi",
        "delta_vs_others",
        "delta_lo",
        "delta_hi",
        "global_mean",
    ]
    float_pair = [
        "mean_present",
        "mean_all",
        "delta_vs_all",
        "delta_lo",
        "delta_hi",
    ]

    def _top(
        df: pd.DataFrame, metric: str, k: int, by: str, asc: bool = False
    ) -> pd.DataFrame:
        return (
            df[df["metric"] == metric]
            .sort_values(by, ascending=asc)
            .head(k)
            .reset_index(drop=True)
        )

    parts: list[str] = []
    parts.append(
        "# Architecture Component Extensions — 2026-05-24\n\n"
        "Companion to `research/reports/arch_component_trust_tier_2026-05-24/report.md` "
        "(Codex's base trust-tier analysis). This addendum adds three cuts the base "
        "report does not cover: per-op dosage curves, depth stratification, and per-"
        "template effects. Primary leak-immune signals only "
        "(ar_curriculum_auc_pair_final, induction_intermediate_auc).\n\n"
        "Bootstrap: 1000 row-resamples for CIs (no metric-noise injection — see base "
        "report for that). All deltas are present-vs-absent on the same row pool.\n"
    )

    parts.append("## 1. Dosage curves\n")
    parts.append(
        "Does more of a primitive help? Bin op count per graph into "
        "`{0, 1, 2, 3+}` and report mean ± 95% bootstrap CI per bin.\n\n"
        "### AR-curriculum dosage curves\n"
    )
    parts.append(
        _md_table(
            dosage[dosage["metric"] == "ar_curriculum_auc_pair_final"]
            .drop(columns=["metric"])
            .reset_index(drop=True),
            float_curve,
        )
    )
    parts.append("\n\n### Induction dosage curves\n")
    parts.append(
        _md_table(
            dosage[dosage["metric"] == "induction_intermediate_auc"]
            .drop(columns=["metric"])
            .reset_index(drop=True),
            float_curve,
        )
    )

    parts.append("\n\n### Dose-response delta (count≥2 vs count=0)\n")
    parts.append(
        "Positive delta with CI excluding zero ⇒ 'more is better' beyond the "
        "presence/absence binary. Negative ⇒ diminishing or harmful returns.\n\n"
    )
    parts.append(_md_table(dose_delta, float_delta))

    parts.append("\n\n## 2. Depth stratification\n")
    parts.append(
        "Per-op present/absent delta within each `graph_depth` tertile. A primitive "
        "whose CI flips sign or vanishes across depth bins is depth-dependent.\n\n"
    )
    parts.append(_md_table(depth, float_depth))

    parts.append(
        "\n\n## 3. Per-template effects (program_graph_features.template_name)\n"
    )
    parts.append(
        "Templates with n≥30. `delta_vs_others` is the mean within that template "
        "minus mean across all other templates.\n\n### AR-curriculum (top 12 by delta)\n"
    )
    parts.append(
        _md_table(
            _top(templates, "ar_curriculum_auc_pair_final", 12, "delta_vs_others"),
            float_tmpl,
        )
    )
    parts.append("\n\n### Induction (top 12 by delta)\n")
    parts.append(
        _md_table(
            _top(templates, "induction_intermediate_auc", 12, "delta_vs_others"),
            float_tmpl,
        )
    )
    parts.append("\n\n### Templates with worst AR-curriculum (bottom 8)\n")
    parts.append(
        _md_table(
            _top(
                templates,
                "ar_curriculum_auc_pair_final",
                8,
                "delta_vs_others",
                asc=True,
            ),
            float_tmpl,
        )
    )

    if not template_pairs.empty:
        parts.append("\n\n## 4. Template-pair effects (metadata.templates_used)\n")
        parts.append(
            "Co-occurring template pairs from the graph metadata. n≥30. Reported "
            "against the global metric mean.\n\n### AR-curriculum (top 10)\n"
        )
        parts.append(
            _md_table(
                _top(
                    template_pairs, "ar_curriculum_auc_pair_final", 10, "delta_vs_all"
                ),
                float_pair,
            )
        )
        parts.append("\n\n### Induction (top 10)\n")
        parts.append(
            _md_table(
                _top(template_pairs, "induction_intermediate_auc", 10, "delta_vs_all"),
                float_pair,
            )
        )

    parts.append(
        "\n\n## Artifacts\n\n"
        "- `research/reports/arch_component_trust_tier_2026-05-24/dosage_curves.csv`\n"
        "- `research/reports/arch_component_trust_tier_2026-05-24/dose_response_delta.csv`\n"
        "- `research/reports/arch_component_trust_tier_2026-05-24/depth_stratified.csv`\n"
        "- `research/reports/arch_component_trust_tier_2026-05-24/template_effects.csv`\n"
        "- `research/reports/arch_component_trust_tier_2026-05-24/template_pair_effects.csv`\n"
    )

    ADDENDUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADDENDUM_PATH.write_text("".join(parts) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading rows from {DB_PATH} ...")
    df = load_rows()
    print(
        f"  loaded {len(df)} rows; "
        f"ar_curr non-null = {df['ar_curriculum_auc_pair_final'].notna().sum()}, "
        f"induction non-null = {df['induction_intermediate_auc'].notna().sum()}"
    )

    print("Cut 1: dosage curves ...")
    dosage = dosage_curves(df)
    dose_delta = dosage_top_vs_zero_delta(df)
    dosage.to_csv(OUT_DIR / "dosage_curves.csv", index=False)
    dose_delta.to_csv(OUT_DIR / "dose_response_delta.csv", index=False)

    print("Cut 2: depth stratification ...")
    depth = depth_stratified(df)
    depth.to_csv(OUT_DIR / "depth_stratified.csv", index=False)

    print("Cut 3: per-template effects ...")
    templates = template_effects(df)
    templates.to_csv(OUT_DIR / "template_effects.csv", index=False)

    template_pairs = template_pair_effects(df)
    template_pairs.to_csv(OUT_DIR / "template_pair_effects.csv", index=False)

    print("Writing addendum ...")
    write_addendum(dosage, dose_delta, depth, templates, template_pairs)
    print(f"Done. Addendum: {ADDENDUM_PATH}")


if __name__ == "__main__":
    main()
