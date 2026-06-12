"""Read-only audit explaining why induction / binding / AR scores diverge.

This tool answers a single question for the user: *for the same
architecture row, why does induction score say one thing, binding say
another, and AR say a third?*

Approach (all read-only, no DB writes, no scoring-config edits):

1. Pull every row that has at least one capability-tier probe value from
   ``leaderboard`` joined to ``program_results`` (or
   ``program_results_compat`` when the view exists).
2. Compute the inter-probe Spearman correlation matrix across the
   canonical probe registry, with bootstrap 95% CIs.
3. Compute the partial Spearman of each capability probe against every
   other capability probe **controlling for** ``wikitext_perplexity``.
   The leading hypothesis is that scores partially track "did the model
   learn language at all", so the residual reveals what the probe
   actually adds beyond ppl.
4. Decompose each probe's variance into between-family vs within-family
   on the templated-family bucket. ``signal_ratio < 0.2`` at screening
   tier flags noise-dominated probes that should not be summed with the
   same weight as cleanly separating probes.
5. Cluster probes by ``1 - |rho|`` so redundant pairs are visible.
6. Emit a candidate ``weight_refit_proposal_<ts>.yaml`` keyed the same
   as ``research/scoring_config.yaml``. Weights are scaled by signal
   ratio divided by cluster size; anchors are recomputed as the cohort
   median. **Not applied** — proposal only, with a side-by-side delta.

Outputs land in ``research/reports/probe_normalization/``. The DB is
opened via the existing ``research.tools._db_maintenance.connect_readonly``
helper so PRAGMA query_only is set.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from research.defaults import RUNS_DB
from research.scientist.probe_normalization import (
    FAMILY_AR,
    FAMILY_BINDING,
    FAMILY_INDUCTION,
    PROBE_METRICS,
    PROBE_METRICS_BY_COLUMN,
    agglomerative_clusters,
    aligned_pairs,
    aligned_triples,
    partial_spearman,
    safe_float,
    spearman_ci,
    table_columns,
    template_family,
    variance_decomposition,
)

DEFAULT_OUT_DIR = Path("research/reports/probe_normalization")

# Columns we always pull from leaderboard (regardless of registry tier)
# so the audit can group / colour rows.
_LB_BASE_COLS = (
    "result_id",
    "entry_id",
    "tier",
    "composite_score",
    "template_name",
    "graph_fingerprint",
    "is_reference",
    "reference_name",
)


def _detect_program_results_table(conn: sqlite3.Connection) -> str:
    has_compat = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'program_results_compat' LIMIT 1"
    ).fetchone()
    return "program_results_compat" if has_compat else "program_results"


def _select_existing(
    conn: sqlite3.Connection, table: str, alias: str, candidates: Sequence[str]
) -> list[str]:
    cols = table_columns(conn, table)
    return [f"{alias}.{c}" for c in candidates if c in cols]


def _load_rows(db_path: Path) -> list[dict[str, Any]]:
    """Pull every row with at least one probe metric populated.

    Uses URI read-only mode so the DB cannot be mutated even if the
    process is interrupted mid-query.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        pr_table = _detect_program_results_table(conn)
        lb_cols = _select_existing(conn, "leaderboard", "l", _LB_BASE_COLS)
        # template_name may live in program_graph_features instead of
        # leaderboard depending on schema vintage; only join when needed.
        has_pgf = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'program_graph_features' LIMIT 1"
            ).fetchone()
        )
        pgf_join = (
            "LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id"
            if has_pgf
            else ""
        )
        pgf_template = ", pgf.template_name AS pgf_template_name" if has_pgf else ""
        # Pull every registered probe metric column that actually exists
        # on this DB's program_results table.
        pr_cols_available = table_columns(conn, pr_table)
        probe_cols = [
            pm.column for pm in PROBE_METRICS if pm.column in pr_cols_available
        ]
        if not probe_cols:
            return []
        probe_select = ", ".join(f"pr.{c}" for c in probe_cols)
        where_any = " OR ".join(f"pr.{c} IS NOT NULL" for c in probe_cols)
        sql = f"""
            SELECT {", ".join(lb_cols)},
                   {probe_select}
                   {pgf_template}
            FROM leaderboard l
            JOIN {pr_table} pr ON pr.result_id = l.result_id
            {pgf_join}
            WHERE {where_any}
        """
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        # Prefer leaderboard template_name; fall back to pgf.
        tmpl = row.get("template_name") or row.get("pgf_template_name")
        row["template_name"] = tmpl
        row["family"] = template_family(tmpl)
        # Reference family override — known-good arches get their own bucket.
        if row.get("is_reference"):
            ref = (row.get("reference_name") or "").lower()
            if ref:
                row["family"] = f"reference_{ref}"
        out.append(row)
    return out


# ── Correlation matrix ───────────────────────────────────────────────


def _pair_matrix(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    bootstrap: int,
) -> dict[str, Any]:
    """Build the full Spearman matrix with bootstrap CIs.

    Returns a dict with ``columns``, ``rho``, ``ci_low``, ``ci_high``,
    ``n_pairs``. Diagonal is 1.0 with zero-width CI.
    """
    cols = list(columns)
    rho: dict[tuple[str, str], float | None] = {}
    ci_low: dict[tuple[str, str], float | None] = {}
    ci_high: dict[tuple[str, str], float | None] = {}
    n: dict[tuple[str, str], int] = {}
    for i, a in enumerate(cols):
        for b in cols[i:]:
            if a == b:
                rho[(a, b)] = 1.0
                ci_low[(a, b)] = 1.0
                ci_high[(a, b)] = 1.0
                n[(a, b)] = sum(1 for row in rows if safe_float(row.get(a)) is not None)
                continue
            xs, ys = aligned_pairs(rows, a, b)
            n[(a, b)] = len(xs)
            if len(xs) < 3:
                rho[(a, b)] = None
                ci_low[(a, b)] = None
                ci_high[(a, b)] = None
                continue
            ci = spearman_ci(xs, ys, n_bootstrap=bootstrap)
            if ci is None:
                rho[(a, b)] = None
                ci_low[(a, b)] = None
                ci_high[(a, b)] = None
            else:
                r, lo, hi = ci
                rho[(a, b)] = r
                ci_low[(a, b)] = lo
                ci_high[(a, b)] = hi
    return {
        "columns": cols,
        "rho": {f"{a}|{b}": v for (a, b), v in rho.items()},
        "ci_low": {f"{a}|{b}": v for (a, b), v in ci_low.items()},
        "ci_high": {f"{a}|{b}": v for (a, b), v in ci_high.items()},
        "n_pairs": {f"{a}|{b}": v for (a, b), v in n.items()},
    }


def _partial_matrix(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    control_column: str,
) -> dict[str, dict[str, float | None]]:
    """Pairwise partial Spearman, controlling for ``control_column``."""
    out: dict[str, dict[str, float | None]] = {}
    for a in columns:
        out[a] = {}
        for b in columns:
            if a == b:
                out[a][b] = 1.0
                continue
            xs, ys, zs = aligned_triples(rows, a, b, control_column)
            if len(xs) < 4:
                out[a][b] = None
                continue
            out[a][b] = partial_spearman(xs, ys, zs)
    return out


# ── Variance + clustering ────────────────────────────────────────────


def _signal_ratios(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, dict[str, float] | None]:
    return {c: variance_decomposition(rows, c, "family") for c in columns}


def _redundancy_clusters(
    matrix: Mapping[str, Any], distance_threshold: float
) -> list[list[str]]:
    cols = list(matrix["columns"])
    rho_lookup: dict[tuple[str, str], float | None] = {}
    for key, val in matrix["rho"].items():
        a, _, b = key.partition("|")
        rho_lookup[(a, b)] = val
    return agglomerative_clusters(
        cols, rho_lookup, distance_threshold=distance_threshold
    )


# ── Weight refit proposal ────────────────────────────────────────────

# Map from capability registry column → scoring_config.yaml weight key.
# Probes without a direct weight knob are not refit (they enter the
# composite via a derived signal such as binding_screening_composite).
_WEIGHT_KEY_BY_COLUMN: Mapping[str, str] = {
    "induction_screening_auc": "w_cap_induction",
    "binding_screening_auc": "w_cap_binding",
    "ar_gate_score": "w_cap_ar",
    "ar_validation_rank_score": "w_cap_ar_validation_validation",
    "ar_legacy_auc": "w_legacy_ar",
    "blimp_overall_accuracy": "w_blimp",
    "hellaswag_acc": "w_hellaswag",
    "tinystories_score": "w_tinystories",
    "cross_task_score": "w_cross_task",
    "diagnostic_score": "w_diagnostic",
    "fp_hierarchy_fitness": "w_hierarchy",
    "fp_jacobian_erf_density": "w_cap_erf_density",
    "fp_id_collapse_rate": "w_cap_id_collapse",
    "fp_jacobian_erf_decay_slope": "w_cap_erf_decay",
    "fp_logit_margin_velocity": "w_cap_logit_margin",
}

# Same map for anchor knobs.
_ANCHOR_KEY_BY_COLUMN: Mapping[str, str] = {
    "induction_screening_auc": "cap_induction_anchor",
    "binding_screening_auc": "cap_binding_anchor",
    "ar_gate_score": "cap_ar_anchor",
    "ar_validation_rank_score": "cap_ar_validation_validation_anchor",
    "ar_legacy_auc": "legacy_ar_anchor",
    "blimp_overall_accuracy": "blimp",
    "hellaswag_acc": "hellaswag",
    "tinystories_score": "tinystories",
    "cross_task_score": "cross_task",
    "diagnostic_score": "diagnostic",
    "fp_hierarchy_fitness": "hierarchy",
    "fp_jacobian_erf_density": "cap_erf_density_anchor",
    "fp_id_collapse_rate": "cap_id_collapse_anchor",
    "fp_jacobian_erf_decay_slope": "cap_erf_decay_anchor",
    "fp_logit_margin_velocity": "cap_logit_margin_anchor",
}


def _cohort_median(rows: Sequence[Mapping[str, Any]], column: str) -> float | None:
    vals = [v for v in (safe_float(r.get(column)) for r in rows) if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _refit_weights(
    *,
    current_config: Mapping[str, Any],
    signal_ratios: Mapping[str, dict[str, float] | None],
    clusters: Sequence[Sequence[str]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a weight + anchor refit proposal.

    For each known weight key, ``w_proposed = w_current * signal_ratio /
    cluster_size``, capped at the current value (we only ever propose
    reducing redundant weight, never amplifying). Anchors are replaced
    by the cohort median when finite.
    """
    base = dict(current_config.get("base") or {})
    cluster_size: dict[str, int] = {}
    for grp in clusters:
        for col in grp:
            cluster_size[col] = len(grp)
    weights_out: dict[str, dict[str, float | None]] = {}
    anchors_out: dict[str, dict[str, float | None]] = {}
    for column, weight_key in _WEIGHT_KEY_BY_COLUMN.items():
        current = safe_float(base.get(weight_key))
        if current is None:
            continue
        sr = signal_ratios.get(column)
        sr_value = sr["signal_ratio"] if sr else 1.0
        size = cluster_size.get(column, 1)
        proposed = current * sr_value / max(1, size)
        if proposed > current:
            proposed = current
        weights_out[weight_key] = {
            "current": current,
            "proposed": round(proposed, 4),
            "signal_ratio": round(sr_value, 4),
            "cluster_size": size,
            "source_column": column,
        }
    for column, anchor_key in _ANCHOR_KEY_BY_COLUMN.items():
        current = safe_float(base.get(anchor_key))
        median = _cohort_median(rows, column)
        if median is None:
            continue
        anchors_out[anchor_key] = {
            "current": current,
            "proposed": round(median, 4),
            "source_column": column,
        }
    return {"weights": weights_out, "anchors": anchors_out}


def _load_scoring_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Report assembly ──────────────────────────────────────────────────


def _registry_columns_present(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Registry columns that have at least one non-null value in rows."""
    present: list[str] = []
    seen: set[str] = set()
    if not rows:
        return present
    sample = rows[0]
    for pm in PROBE_METRICS:
        if pm.column in sample and pm.column not in seen:
            # Confirm at least one row has it populated.
            for r in rows:
                if safe_float(r.get(pm.column)) is not None:
                    present.append(pm.column)
                    seen.add(pm.column)
                    break
    return present


def _capability_columns(present: Sequence[str]) -> list[str]:
    """Subset of present columns belonging to induction/binding/AR.

    Used for the partial-correlation block and the cluster headline.
    """
    caps = {FAMILY_INDUCTION, FAMILY_BINDING, FAMILY_AR}
    return [c for c in present if PROBE_METRICS_BY_COLUMN[c].family in caps]


def build_report(
    db_path: Path,
    *,
    scoring_yaml_path: Path,
    tiers: Sequence[str] = ("screening", "investigation", "validation"),
    bootstrap: int = 500,
    cluster_distance: float = 0.2,
) -> dict[str, Any]:
    rows = _load_rows(db_path)
    if not rows:
        return {
            "coverage": {"rows": 0, "tiers": {}, "families": {}},
            "tiers": {},
            "notes": ["no rows with probe metrics found"],
        }

    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_tier[str(r.get("tier") or "unknown")].append(r)

    config = _load_scoring_yaml(scoring_yaml_path)

    per_tier_reports: dict[str, Any] = {}
    for tier in tiers:
        tier_rows = by_tier.get(tier, [])
        if not tier_rows:
            per_tier_reports[tier] = {"n_rows": 0, "skipped": "no rows"}
            continue
        present = _registry_columns_present(tier_rows)
        if not present:
            per_tier_reports[tier] = {"n_rows": len(tier_rows), "skipped": "no probes"}
            continue
        # Make sure wikitext_perplexity is included for partial correlation.
        cap_cols = _capability_columns(present)
        partial_target = (
            "wikitext_perplexity" if "wikitext_perplexity" in present else None
        )

        matrix = _pair_matrix(tier_rows, present, bootstrap=bootstrap)
        signal_ratios = _signal_ratios(tier_rows, present)
        clusters = _redundancy_clusters(matrix, cluster_distance)
        partials: dict[str, Any] = {}
        if partial_target and cap_cols:
            partials = _partial_matrix(tier_rows, cap_cols, partial_target)
        refit = _refit_weights(
            current_config=config,
            signal_ratios=signal_ratios,
            clusters=clusters,
            rows=tier_rows,
        )
        per_tier_reports[tier] = {
            "n_rows": len(tier_rows),
            "columns": present,
            "capability_columns": cap_cols,
            "matrix": matrix,
            "partial_target": partial_target,
            "partial_matrix": partials,
            "signal_ratios": signal_ratios,
            "redundancy_clusters": clusters,
            "refit_proposal": refit,
        }

    coverage = {
        "rows": len(rows),
        "tiers": {tier: len(rows_t) for tier, rows_t in by_tier.items()},
        "families": dict(
            sorted(
                (f, sum(1 for r in rows if r.get("family") == f))
                for f in {r["family"] for r in rows}
            )
        ),
    }
    findings = _build_findings(per_tier_reports)
    return {
        "coverage": coverage,
        "tiers": per_tier_reports,
        "findings": findings,
        "scoring_config_path": str(scoring_yaml_path),
    }


def _build_findings(per_tier: Mapping[str, Any]) -> list[str]:
    """Plain-English summary of divergence between induction / binding / AR.

    Walks the screening tier first (where the three are co-evaluated)
    and emits at most one bullet per finding type.
    """
    findings: list[str] = []
    screening = per_tier.get("screening") or {}
    if not isinstance(screening, dict) or "matrix" not in screening:
        return findings
    cap_cols = screening.get("capability_columns") or []
    rho_lookup: dict[tuple[str, str], float | None] = {}
    for key, val in (screening.get("matrix") or {}).get("rho", {}).items():
        a, _, b = key.partition("|")
        rho_lookup[(a, b)] = val
        rho_lookup[(b, a)] = val
    # Direct pairwise rho between induction/binding/AR.
    if {"induction_screening_auc", "binding_screening_auc", "ar_gate_score"}.issubset(
        cap_cols
    ):
        r_ib = rho_lookup.get(("induction_screening_auc", "binding_screening_auc"))
        r_ia = rho_lookup.get(("induction_screening_auc", "ar_gate_score"))
        r_ba = rho_lookup.get(("binding_screening_auc", "ar_gate_score"))
        findings.append(
            "Screening-tier pairwise Spearman ρ: induction↔binding="
            f"{_fmt(r_ib)}, induction↔ar_gate={_fmt(r_ia)}, "
            f"binding↔ar_gate={_fmt(r_ba)}."
        )
        # Partial vs ppl
        partials = screening.get("partial_matrix") or {}
        if partials and screening.get("partial_target"):
            p_ib = partials.get("induction_screening_auc", {}).get(
                "binding_screening_auc"
            )
            p_ia = partials.get("induction_screening_auc", {}).get("ar_gate_score")
            p_ba = partials.get("binding_screening_auc", {}).get("ar_gate_score")
            findings.append(
                "Partial ρ controlling for wikitext_perplexity: induction↔binding="
                f"{_fmt(p_ib)}, induction↔ar_gate={_fmt(p_ia)}, "
                f"binding↔ar_gate={_fmt(p_ba)}. Values near zero mean the "
                "probe's signal vanishes once we account for general "
                "language-model competence."
            )
    # Signal ratios per capability probe
    sr = screening.get("signal_ratios") or {}
    for col in ("induction_screening_auc", "binding_screening_auc", "ar_gate_score"):
        sr_col = sr.get(col)
        if sr_col is None:
            continue
        ratio = sr_col.get("signal_ratio")
        if ratio is None:
            continue
        verdict = "noise-dominated" if ratio < 0.2 else "ok"
        findings.append(
            f"signal_ratio for {col}={ratio:.3f} ({verdict}). "
            "Below 0.2 means within-family variance exceeds between-family variance "
            "and a replication harness is warranted before trusting cross-arch ranking."
        )
    # Redundancy clusters
    clusters = screening.get("redundancy_clusters") or []
    for grp in clusters:
        if len(grp) > 1:
            findings.append(
                "Redundancy cluster (distance < 0.2 in 1-|rho|): "
                + ", ".join(grp)
                + ". These probes co-vary so strongly they double-count in any "
                "additive composite."
            )
    return findings


def _fmt(value: Any, digits: int = 3) -> str:
    val = safe_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


# ── Markdown rendering ───────────────────────────────────────────────


def _md_header_block(report: Mapping[str, Any]) -> list[str]:
    cov = report["coverage"]
    return [
        "# Probe Divergence Audit",
        "",
        "Read-only analysis of induction / binding / associative-recall / loss /",
        "perplexity / understanding probes across the live cohort. No DB writes,",
        "no scoring-config edits. Output companion JSON has the full numerical",
        "tables; this file is the executive summary.",
        "",
        "## Coverage",
        "",
        f"- Total rows with any probe: {cov['rows']}",
        "- Tier breakdown: " + ", ".join(f"{k}={v}" for k, v in cov["tiers"].items()),
        "- Family breakdown: "
        + ", ".join(f"{k}={v}" for k, v in cov["families"].items()),
        "",
        "## Headline findings",
        "",
    ]


def _md_findings_block(report: Mapping[str, Any]) -> list[str]:
    findings = report.get("findings") or []
    if not findings:
        return ["- (no screening tier rows with the full induction/binding/AR triple)"]
    return [f"- {f}" for f in findings]


def _md_matrix_block(
    columns: Sequence[str],
    rho_map: Mapping[str, Any],
    heading: str,
) -> list[str]:
    lines = [
        "",
        heading,
        "",
        "| | " + " | ".join(columns) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for a in columns:
        row_vals: list[str] = []
        for b in columns:
            key = f"{a}|{b}" if f"{a}|{b}" in rho_map else f"{b}|{a}"
            row_vals.append(_fmt(rho_map.get(key)))
        lines.append("| " + a + " | " + " | ".join(row_vals) + " |")
    return lines


def _md_partial_block(td: Mapping[str, Any]) -> list[str]:
    if not (td.get("partial_target") and td.get("capability_columns")):
        return []
    cap_cols = td["capability_columns"]
    pm = td["partial_matrix"]
    lines = [
        "",
        f"### Partial Spearman controlling for {td['partial_target']}",
        "",
        "| | " + " | ".join(cap_cols) + " |",
        "|" + "---|" * (len(cap_cols) + 1),
    ]
    for a in cap_cols:
        row_vals = [_fmt(pm.get(a, {}).get(b)) for b in cap_cols]
        lines.append("| " + a + " | " + " | ".join(row_vals) + " |")
    return lines


def _md_signal_ratio_block(signal_ratios: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "### Signal ratios (between-family / total variance)",
        "",
        "| column | signal_ratio | between | within |",
        "|---|---:|---:|---:|",
    ]
    for col, sr in signal_ratios.items():
        if sr is None:
            lines.append(f"| {col} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {col} | {sr['signal_ratio']:.3f} | {sr['between']:.2f} | "
            f"{sr['within']:.2f} |"
        )
    return lines


def _md_clusters_block(clusters: Sequence[Sequence[str]]) -> list[str]:
    multi = [g for g in clusters if len(g) > 1]
    if not multi:
        return []
    lines = ["", "### Redundancy clusters", ""]
    for grp in multi:
        lines.append("- " + ", ".join(grp))
    return lines


def _md_refit_block(refit: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if refit.get("weights"):
        lines.extend(
            [
                "",
                "### Weight refit proposal (dry-run; not applied)",
                "",
                "| key | current | proposed | signal_ratio | cluster_size |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, info in refit["weights"].items():
            lines.append(
                f"| {key} | {info['current']:.2f} | {info['proposed']:.2f} | "
                f"{info['signal_ratio']:.3f} | {info['cluster_size']} |"
            )
    if refit.get("anchors"):
        lines.extend(
            [
                "",
                "### Anchor refit proposal",
                "",
                "| key | current | proposed (cohort median) |",
                "|---|---:|---:|",
            ]
        )
        for key, info in refit["anchors"].items():
            cur = info["current"]
            lines.append(f"| {key} | {_fmt(cur)} | {info['proposed']:.4f} |")
    return lines


def _md_tier_block(tier: str, td: Mapping[str, Any]) -> list[str]:
    if not isinstance(td, dict) or td.get("skipped"):
        return []
    lines = ["", f"## Tier: {tier} ({td['n_rows']} rows)"]
    lines.extend(
        _md_matrix_block(
            td["columns"], td["matrix"]["rho"], "### Capability Spearman matrix"
        )
    )
    lines.extend(_md_partial_block(td))
    lines.extend(_md_signal_ratio_block(td["signal_ratios"]))
    lines.extend(_md_clusters_block(td.get("redundancy_clusters") or []))
    lines.extend(_md_refit_block(td.get("refit_proposal") or {}))
    return lines


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = _md_header_block(report)
    lines.extend(_md_findings_block(report))
    for tier, td in (report.get("tiers") or {}).items():
        lines.extend(_md_tier_block(tier, td))
    return "\n".join(lines) + "\n"


def _render_refit_yaml(report: Mapping[str, Any]) -> str:
    """Build a config-shaped YAML proposal aggregating across tiers.

    Conflicts (same key proposed by multiple tiers) are resolved by
    taking the minimum proposed weight (the more conservative reduction)
    and the screening-tier anchor when present.
    """
    weights_min: dict[str, float] = {}
    anchors: dict[str, float] = {}
    notes: dict[str, list[str]] = defaultdict(list)
    for tier, td in (report.get("tiers") or {}).items():
        if not isinstance(td, dict) or td.get("skipped"):
            continue
        refit = td.get("refit_proposal") or {}
        for key, info in (refit.get("weights") or {}).items():
            prop = float(info["proposed"])
            if key not in weights_min or prop < weights_min[key]:
                weights_min[key] = prop
            notes[key].append(
                f"{tier}: signal_ratio={info['signal_ratio']:.3f} "
                f"cluster_size={info['cluster_size']}"
            )
        if tier == "screening":
            for key, info in (refit.get("anchors") or {}).items():
                anchors[key] = float(info["proposed"])
    proposal = {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "applies_to": "research/scoring_config.yaml::base",
            "applied": False,
            "notes": dict(notes),
        },
        "base": {**weights_min, **anchors},
    }
    return yaml.safe_dump(proposal, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(RUNS_DB))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scoring-yaml",
        type=Path,
        default=Path("research/scoring_config.yaml"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--cluster-distance", type=float, default=0.2)
    parser.add_argument(
        "--tiers",
        type=str,
        default="screening,investigation,validation",
        help="Comma-separated leaderboard tier names to analyze.",
    )
    args = parser.parse_args()
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    report = build_report(
        args.db,
        scoring_yaml_path=args.scoring_yaml,
        tiers=tiers,
        bootstrap=args.bootstrap,
        cluster_distance=args.cluster_distance,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    json_path = args.out / f"divergence_audit_{ts}.json"
    md_path = args.out / f"divergence_audit_{ts}.md"
    yaml_path = args.out / f"weight_refit_proposal_{ts}.yaml"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    yaml_path.write_text(_render_refit_yaml(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
