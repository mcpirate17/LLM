#!/usr/bin/env python
"""Profile-grounded ML analysis over the standalone meta-analysis DB.

The analysis is read-only. It ranks profile, template, slot, routing,
compression, and external-prior descriptors against observed outcomes, then
emits reviewable recommendations without changing generation policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.meta_analysis.metadata_db import DEFAULT_META_ANALYSIS_DB


DEFAULT_REPORT_DIR = Path("research/reports")
DEFAULT_MIN_SUPPORT = 25


_BASE_FEATURE_COLUMNS = (
    "motif_count",
    "non_norm_motif_count",
    "norm_motif_count",
    "norm_dominance",
    "has_attention_motif",
    "has_ssm_motif",
    "has_conv_motif",
    "has_recurrent_motif",
    "has_routing_motif",
    "has_compression_motif",
    "has_effective_positional_mixer",
    "mixer_after_compression",
    "motif_thinness_score",
    "frequency_collapse_risk",
    "template_has_attention",
    "template_has_ssm",
    "template_has_conv",
    "template_has_routing",
    "template_has_compression",
    "template_has_memory",
    "template_has_math_space",
    "template_has_frequency_domain",
    "template_has_norm",
    "template_has_moe",
    "template_has_residual",
    "template_has_parallel_paths",
    "template_has_state",
    "template_routing_intensity",
    "template_memory_intensity",
    "template_compression_intensity",
    "template_local_global_mix",
    "template_math_space_intensity",
    "template_stabilization_need",
    "template_trainability_prior",
    "template_novelty_prior",
)

_GRAPH_PROFILE_FEATURE_COLUMNS = (
    "graph_profile_op_count",
    "profile_known_op_count",
    "profile_missing_op_count",
    "profile_coverage_rate",
    "profile_total_forward_time_us",
    "profile_total_backward_time_us",
    "profile_mean_forward_time_us",
    "profile_max_forward_time_us",
    "profile_total_peak_memory_bytes",
    "profile_total_flops_estimate",
    "profile_max_lipschitz_estimate",
    "profile_max_jacobian_condition_num",
    "profile_grad_vanishing_op_count",
    "profile_grad_exploding_op_count",
    "profile_output_nan_op_count",
    "profile_pair_count",
    "profile_pair_unstable_count",
    "profile_pair_grad_exploding_count",
    "profile_pair_max_lipschitz_estimate",
    "profile_triplet_count",
    "profile_triplet_unstable_count",
    "profile_triplet_divergent_count",
    "profile_triplet_grad_vanishing_count",
)

_EXTERNAL_TAG_COLUMNS = (
    "external_tag_routing_count",
    "external_tag_compression_count",
    "external_tag_math_count",
    "external_tag_sparse_learning_count",
    "external_tag_efficient_sequence_count",
    "external_tag_stability_count",
    "external_prior_hit_count",
)


@dataclass(frozen=True)
class AnalysisFrame:
    rows: list[dict[str, Any]]
    feature_names: list[str]


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db = Path(path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _num(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value)
    return default if parsed is None else parsed


def _rate(positive: int, total: int) -> float:
    return positive / total if total else 0.0


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        loaded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return loaded if loaded is not None else default


def _external_op_tags(conn: sqlite3.Connection) -> dict[str, set[str]]:
    if not _table_exists(conn, "external_component_prior_catalog"):
        return {}
    mapping: dict[str, set[str]] = {}
    rows = conn.execute(
        "SELECT mapped_ops_json, tags_json FROM external_component_prior_catalog"
    ).fetchall()
    for row in rows:
        tags = {str(tag) for tag in _json_loads(row["tags_json"], [])}
        for op_name in _json_loads(row["mapped_ops_json"], []):
            mapping.setdefault(str(op_name), set()).update(tags)
    return mapping


def _external_features_by_result(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    if not _table_exists(conn, "op_observations"):
        return {}
    op_tags = _external_op_tags(conn)
    features: dict[str, dict[str, int]] = {}
    rows = conn.execute("SELECT result_id, op_name FROM op_observations").fetchall()
    for row in rows:
        result_id = str(row["result_id"])
        tags = op_tags.get(str(row["op_name"]), set())
        dest = features.setdefault(
            result_id,
            {name: 0 for name in _EXTERNAL_TAG_COLUMNS},
        )
        if tags:
            dest["external_prior_hit_count"] += 1
        for tag in tags:
            key = f"external_tag_{tag}_count"
            if key in dest:
                dest[key] += 1
    return features


def load_analysis_frame(meta_db: str | Path) -> AnalysisFrame:
    conn = _connect_readonly(meta_db)
    try:
        graph_join = (
            "LEFT JOIN graph_profile_observations gp ON gp.result_id = ranked.result_id"
            if _table_exists(conn, "graph_profile_observations")
            else ""
        )
        gp_exprs = [
            f"gp.{name} AS {name}" if graph_join else f"0 AS {name}"
            for name in _GRAPH_PROFILE_FEATURE_COLUMNS
        ]
        select_cols = [
            "ranked.result_id",
            "ranked.template_name",
            "COALESCE(ranked.failure_op, '') AS failure_op",
            "ranked.stage1_passed",
            "ranked.wikitext_perplexity",
            "ranked.tinystories_score",
            "ranked.controlled_lang_s05_sa_score",
            "ranked.routing_fast_lane_ppl_improvement",
            "ranked.composite_score",
            *(f"ranked.{name} AS {name}" for name in _BASE_FEATURE_COLUMNS),
            *gp_exprs,
        ]
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY result_id
                        ORDER BY slot_count DESC, template_name ASC
                    ) AS rn
                FROM template_observations
            )
            SELECT {", ".join(select_cols)}
            FROM ranked
            {graph_join}
            WHERE ranked.rn = 1
            """
        ).fetchall()
        external_by_result = _external_features_by_result(conn)
    finally:
        conn.close()

    feature_names = [
        *_BASE_FEATURE_COLUMNS,
        *_GRAPH_PROFILE_FEATURE_COLUMNS,
        *_EXTERNAL_TAG_COLUMNS,
    ]
    out: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        external = external_by_result.get(
            str(record["result_id"]),
            {name: 0 for name in _EXTERNAL_TAG_COLUMNS},
        )
        record.update(external)
        record["target_nano_bind_failure"] = int(
            record.get("failure_op") == "nano_bind"
        )
        stage1 = _safe_float(record.get("stage1_passed"))
        record["target_stage1_passed"] = int(stage1 == 1.0)
        sa = _safe_float(record.get("controlled_lang_s05_sa_score"))
        record["target_controlled_lang_pass"] = int(
            sa is not None and sa >= 0.95 and record.get("failure_op") != "nano_bind"
        )
        ppl = _safe_float(record.get("wikitext_perplexity"))
        record["target_good_wikitext"] = int(ppl is not None and ppl < 200.0)
        tiny = _safe_float(record.get("tinystories_score"))
        record["target_good_tinystories"] = int(tiny is not None and tiny >= 0.55)
        improvement = _safe_float(record.get("routing_fast_lane_ppl_improvement"))
        record["target_routing_improved"] = int(
            improvement is not None and improvement > 0.0
        )
        for feature in feature_names:
            record[feature] = _num(record.get(feature), 0.0)
        out.append(record)
    return AnalysisFrame(rows=out, feature_names=feature_names)


def _univariate_lifts(
    rows: list[dict[str, Any]],
    feature_names: Iterable[str],
    target: str,
    *,
    min_support: int,
) -> list[dict[str, Any]]:
    positives = sum(int(row[target]) for row in rows)
    baseline = _rate(positives, len(rows))
    findings: list[dict[str, Any]] = []
    for feature in feature_names:
        values = sorted({float(row[feature]) for row in rows})
        if len(values) <= 1:
            continue
        thresholds = (
            [0.5] if set(values).issubset({0.0, 1.0}) else [values[len(values) // 2]]
        )
        for threshold in thresholds:
            for direction in (">=", "<"):
                if direction == ">=":
                    subset = [row for row in rows if float(row[feature]) >= threshold]
                else:
                    subset = [row for row in rows if float(row[feature]) < threshold]
                if len(subset) < min_support:
                    continue
                subset_pos = sum(int(row[target]) for row in subset)
                rate = _rate(subset_pos, len(subset))
                findings.append(
                    {
                        "target": target,
                        "feature": feature,
                        "direction": direction,
                        "threshold": round(threshold, 6),
                        "support": len(subset),
                        "positive_count": subset_pos,
                        "positive_rate": round(rate, 6),
                        "lift": round(rate / baseline, 6) if baseline else 0.0,
                    }
                )
    return sorted(
        findings,
        key=lambda row: (abs(row["lift"] - 1.0), row["support"]),
        reverse=True,
    )


def _sklearn_importances(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target: str,
    *,
    min_support: int,
) -> dict[str, Any]:
    positives = sum(int(row[target]) for row in rows)
    negatives = len(rows) - positives
    if positives < min_support or negatives < min_support:
        return {
            "available": False,
            "reason": "insufficient_class_support",
            "positive_count": positives,
            "negative_count": negatives,
        }
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover - exercised only without sklearn.
        return {"available": False, "reason": f"sklearn_unavailable:{exc}"}

    x = np.asarray(
        [[float(row[feature]) for feature in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row[target]) for row in rows], dtype=np.int32)
    stratify = y if positives >= 2 and negatives >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=7,
            min_samples_leaf=max(2, min_support // 10),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, scores) if len(set(y_test.tolist())) == 2 else None
    except Exception as exc:
        return {"available": False, "reason": f"fit_failed:{exc}"}

    importances = sorted(
        (
            {
                "target": target,
                "feature": feature,
                "importance": round(float(importance), 8),
            }
            for feature, importance in zip(feature_names, model.feature_importances_)
            if float(importance) > 0.0
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )
    return {
        "available": True,
        "positive_count": positives,
        "negative_count": negatives,
        "holdout_auc": round(float(auc), 6) if auc is not None else None,
        "top_importances": importances[:30],
    }


def _recommendations(
    target_payloads: dict[str, dict[str, Any]],
    lifts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    interesting_features = {
        "profile_pair_unstable_count": "test safer adjacent op pairings or insert normalization between unstable edges",
        "profile_triplet_divergent_count": "prioritize triplet ablations where pair predictions fail",
        "profile_grad_exploding_op_count": "profile stabilizers and lower-init variants for exploding ops",
        "profile_total_forward_time_us": "prefer cheaper equivalent components before routing/compression promotion",
        "frequency_collapse_risk": "add positional/effective mixers after compression-heavy motifs",
        "has_effective_positional_mixer": "preserve positional/content mixers when compressing tokens",
        "external_tag_compression_count": "validate compression with NanoBind and WikiText jointly",
        "external_tag_routing_count": "separate router quality from downstream LM quality in follow-up probes",
        "external_tag_math_count": "profile math-space candidates before expanding search weight",
        "external_tag_sparse_learning_count": "test sparse learning at fixed parameter and wall-clock budgets",
    }
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for target, payload in target_payloads.items():
        for row in payload.get("top_importances", [])[:12]:
            feature = row["feature"]
            if feature not in interesting_features or (target, feature) in seen:
                continue
            seen.add((target, feature))
            out.append(
                {
                    "target": target,
                    "feature": feature,
                    "evidence": f"importance={row['importance']}",
                    "recommendation": interesting_features[feature],
                }
            )
    for row in lifts[:30]:
        feature = row["feature"]
        target = row["target"]
        if feature not in interesting_features or (target, feature) in seen:
            continue
        seen.add((target, feature))
        out.append(
            {
                "target": target,
                "feature": feature,
                "evidence": f"lift={row['lift']} support={row['support']}",
                "recommendation": interesting_features[feature],
            }
        )
    return out[:40]


def build_payload(
    meta_db: str | Path,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> dict[str, Any]:
    frame = load_analysis_frame(meta_db)
    targets = [
        "target_nano_bind_failure",
        "target_controlled_lang_pass",
        "target_stage1_passed",
        "target_good_wikitext",
        "target_good_tinystories",
        "target_routing_improved",
    ]
    model_payloads = {
        target: _sklearn_importances(
            frame.rows,
            frame.feature_names,
            target,
            min_support=min_support,
        )
        for target in targets
    }
    all_lifts: list[dict[str, Any]] = []
    for target in targets:
        all_lifts.extend(
            _univariate_lifts(
                frame.rows,
                frame.feature_names,
                target,
                min_support=min_support,
            )[:30]
        )
    all_lifts.sort(
        key=lambda row: (abs(row["lift"] - 1.0), row["support"]), reverse=True
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta_db": str(meta_db),
        "n_graphs": len(frame.rows),
        "n_features": len(frame.feature_names),
        "min_support": min_support,
    }
    for target in targets:
        positives = sum(int(row[target]) for row in frame.rows)
        summary[f"{target}_positives"] = positives
        summary[f"{target}_rate"] = _rate(positives, len(frame.rows))
    return {
        "summary": summary,
        "targets": model_payloads,
        "univariate_lifts": all_lifts[:120],
        "recommendations": _recommendations(model_payloads, all_lifts),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(
    rows: list[dict[str, Any]], fields: list[str], *, limit: int
) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:limit]:
        lines.append(
            "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        f"# Profile-Grounded ML Analysis - {summary['created_utc']}",
        "",
        "Read-only analysis over `research/meta_analysis.db`; recommendations are advisory and do not change scoring, gates, or ablation policy.",
        "",
        "## Summary",
        "",
        f"- Graph rows analyzed: {summary['n_graphs']}",
        f"- Feature count: {summary['n_features']}",
        f"- Minimum support: {summary['min_support']}",
        "",
        "## Target Models",
        "",
    ]
    for target, model in payload["targets"].items():
        lines.extend(
            [
                f"### {target}",
                "",
                f"- Available: {model.get('available')}",
                f"- Positives: {model.get('positive_count')}",
                f"- Negatives: {model.get('negative_count')}",
                f"- Holdout AUC: {model.get('holdout_auc')}",
                f"- Reason: {model.get('reason', '')}",
                "",
                *_md_table(
                    model.get("top_importances", []),
                    ["feature", "importance"],
                    limit=10,
                ),
            ]
        )
    lines.extend(
        [
            "## Highest Lift Feature Rules",
            "",
            *_md_table(
                payload["univariate_lifts"],
                [
                    "target",
                    "feature",
                    "direction",
                    "threshold",
                    "support",
                    "positive_rate",
                    "lift",
                ],
                limit=30,
            ),
            "## Recommended Follow-Up Actions",
            "",
            *_md_table(
                payload["recommendations"],
                ["target", "feature", "evidence", "recommendation"],
                limit=40,
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=DEFAULT_META_ANALYSIS_DB)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prefix = args.output_prefix or f"meta_profile_ml_analysis_{stamp}"
    payload = build_payload(args.meta_db, min_support=args.min_support)

    json_path = report_dir / f"{prefix}.json"
    md_path = report_dir / f"{prefix}.md"
    lifts_csv = report_dir / f"{prefix}_univariate_lifts.csv"
    recommendations_csv = report_dir / f"{prefix}_recommendations.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(md_path, payload)
    write_csv(lifts_csv, payload["univariate_lifts"])
    write_csv(recommendations_csv, payload["recommendations"])

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "univariate_lifts_csv": str(lifts_csv),
                "recommendations_csv": str(recommendations_csv),
                "n_graphs": payload["summary"]["n_graphs"],
                "n_features": payload["summary"]["n_features"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
