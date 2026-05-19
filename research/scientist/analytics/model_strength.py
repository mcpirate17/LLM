from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm

from ..json_utils import fast_loads as _json_loads
from ..trust_policy import (
    PROMOTABLE_COMPARABILITY_LABELS,
    PROMOTABLE_TRUST_LABELS,
    TRUSTED_COMPARABILITY_LABELS,
    TRUSTED_TRUST_LABELS,
)
from .dynamic_component_features import dynamic_component_feature_summary
from .model_strength_schema import EMPTY_STRENGTH_COLUMNS

BASE_ANALYSIS_QUERY = """
SELECT
    pr.result_id,
    pr.experiment_id,
    pr.timestamp,
    pr.graph_fingerprint,
    pr.graph_json,
    pr.stage0_passed,
    pr.stage05_passed,
    pr.stage1_passed,
    pr.loss_ratio,
    pr.validation_loss_ratio,
    pr.discovery_loss_ratio,
    pr.induction_screening_auc,
    pr.binding_screening_auc,
    pr.ar_legacy_auc,
    pr.hellaswag_acc,
    -- Mask byte-era PPL: rows whose screening_wikitext_metric_version is
    -- not 'bpe_eval_v1' have wikitext_perplexity in different units
    -- (range 23 – 485M).  Returning NULL lets downstream filters (which
    -- already handle missing PPL) skip them cleanly without needing
    -- per-call guards everywhere analytics reads this column.
    CASE WHEN COALESCE(pr.screening_wikitext_metric_version, '') = 'bpe_eval_v1'
         THEN pr.wikitext_perplexity END AS wikitext_perplexity,
    pr.wikitext_score,
    pr.stability_score,
    pr.validation_robustness_score,
    pr.efficiency_multiple,
    pr.efficiency_wall_score,
    pr.param_count,
    pr.train_budget_steps,
    pr.n_train_steps,
    pr.total_train_time_ms,
    pr.graph_depth,
    pr.graph_n_ops,
    pr.graph_n_unique_ops,
    pr.graph_uses_math_spaces,
    pr.graph_uses_frequency_domain,
    pr.routing_savings_ratio,
    pr.compression_ratio,
    pr.activation_sparsity_score,
    pr.dead_neuron_ratio,
    pr.routing_collapse_score,
    pr.validation_is_unstable,
    pr.has_nan_output,
    pr.has_inf_output,
    pr.has_nan_grad,
    pr.has_zero_grad,
    pr.local_only,
    pr.novelty_score,
    pr.novelty_confidence,
    pr.result_cohort,
    pr.trust_label,
    pr.comparability_label,
    pr.evaluation_protocol_version,
    pr.init_regime,
    pr.model_source,
    pr.error_type,
    pr.stage_at_death,
    pr.data_provenance_json,
    exp.timestamp AS experiment_timestamp,
    exp.experiment_type,
    exp.config_json
FROM program_results_compat pr
LEFT JOIN experiments exp ON exp.experiment_id = pr.experiment_id
WHERE COALESCE(pr.graph_json, '') NOT IN ('', '{}')
"""

PRIMARY_METRICS: tuple[tuple[str, bool, float], ...] = (
    ("quality_metric", False, 0.34),
    ("induction_screening_auc", True, 0.18),
    ("binding_screening_auc", True, 0.14),
    ("hellaswag_acc", True, 0.14),
    ("wikitext_quality", True, 0.20),
)

SECONDARY_METRICS: tuple[tuple[str, bool, float], ...] = (
    ("stability_score", True, 0.30),
    ("validation_robustness_score", True, 0.20),
    ("efficiency_metric", True, 0.25),
    ("novelty_score", True, 0.10),
    ("stage1_passed", True, 0.15),
)

ANALYSIS_METRICS: dict[str, dict[str, Any]] = {
    "great_score": {"higher_is_better": True, "cohort": "promotable"},
    "quality_metric": {"higher_is_better": True, "cohort": "trusted"},
    "induction_screening_auc": {"higher_is_better": True, "cohort": "promotable"},
    "binding_screening_auc": {"higher_is_better": True, "cohort": "promotable"},
    "hellaswag_acc": {"higher_is_better": True, "cohort": "promotable"},
    "wikitext_quality": {"higher_is_better": True, "cohort": "promotable"},
    "stability_score": {"higher_is_better": True, "cohort": "trusted"},
    "efficiency_metric": {"higher_is_better": True, "cohort": "promotable"},
    "stage1_passed": {"higher_is_better": True, "cohort": "trusted"},
}


@dataclass(slots=True)
class StrengthDatasets:
    all_runs: pd.DataFrame
    trusted_runs: pd.DataFrame
    promotable_runs: pd.DataFrame
    dedup_all: pd.DataFrame
    dedup_trusted: pd.DataFrame
    dedup_promotable: pd.DataFrame


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = _json_loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _entropy_from_weights(weights: dict[str, Any]) -> float:
    vals = [
        max(float(value), 0.0)
        for value in weights.values()
        if isinstance(value, (int, float))
    ]
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    probs = [value / total for value in vals if value > 0.0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _parse_config_features(config_json: Any) -> dict[str, Any]:
    config = _safe_json_loads(config_json)
    out: dict[str, Any] = {
        "cfg_stage1_steps": config.get("stage1_steps"),
        "cfg_stage1_batch_size": config.get("stage1_batch_size"),
        "cfg_stage1_lr": config.get("stage1_lr"),
        "cfg_model_dim": config.get("model_dim"),
        "cfg_n_layers": config.get("n_layers"),
        "cfg_n_programs": config.get("n_programs"),
        "cfg_graphs_weighted": config.get("n_graphs_weighted"),
    }
    for prefix in ("category_weights", "op_weights", "template_weights"):
        weights = config.get(prefix)
        if not isinstance(weights, dict):
            continue
        out[f"{prefix}_entropy"] = _entropy_from_weights(weights)
        numeric_items = {
            str(key): float(value)
            for key, value in weights.items()
            if isinstance(value, (int, float))
        }
        if numeric_items:
            out[f"{prefix}_max_weight"] = max(numeric_items.values())
            out[f"{prefix}_min_weight"] = min(numeric_items.values())
        for key, value in numeric_items.items():
            out[f"{prefix}::{key}"] = value
    return out


def _iter_nodes(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nodes = graph.get("nodes") or {}
    if isinstance(nodes, dict):
        return [
            (str(node_id), node)
            for node_id, node in nodes.items()
            if isinstance(node, dict)
        ]
    if isinstance(nodes, list):
        return [
            (str(node.get("id", idx)), node)
            for idx, node in enumerate(nodes)
            if isinstance(node, dict)
        ]
    return []


def _graph_topology_features(
    nodes: list[tuple[str, dict[str, Any]]],
) -> tuple[list[str], list[str], list[str]]:
    by_id = {node_id: node for node_id, node in nodes}
    children = {node_id: [] for node_id, _ in nodes}
    indegree = {node_id: 0 for node_id, _ in nodes}
    depths = {node_id: 0 for node_id, _ in nodes}
    ops: list[str] = []
    depth_ops: list[tuple[str, str]] = []
    pairs: set[str] = set()
    for node_id, node in nodes:
        op = str(node.get("op_name") or "").strip()
        inputs = node.get("input_ids") or []
        if isinstance(inputs, list):
            valid_parents: list[str] = []
            for parent in inputs:
                parent_id = str(parent)
                if parent_id not in by_id:
                    continue
                valid_parents.append(parent_id)
                children[parent_id].append(node_id)
            indegree[node_id] = len(valid_parents)
            if op and op not in {"input", "output"}:
                for parent_id in valid_parents:
                    parent_node = by_id.get(parent_id) or {}
                    parent_op = str(parent_node.get("op_name") or "").strip()
                    if parent_op and parent_op not in {"input", "output"}:
                        a, b = sorted((parent_op, op))
                        pairs.add(f"{a}+{b}")
        if not op or op in {"input", "output"}:
            continue
        ops.append(op)
        depth_ops.append((node_id, op))
    op_depth_buckets = _op_depth_buckets(depth_ops, children, indegree, depths)
    return sorted(set(ops)), sorted(pairs), sorted(op_depth_buckets)


def _op_depth_buckets(
    depth_ops: list[tuple[str, str]],
    children: dict[str, list[str]],
    indegree: dict[str, int],
    depths: dict[str, int],
) -> set[str]:
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    while queue:
        node_id = queue.popleft()
        next_depth = depths[node_id] + 1
        for child_id in children.get(node_id, ()):
            if next_depth > depths.get(child_id, 0):
                depths[child_id] = next_depth
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)

    max_depth = max(depths.values(), default=0)
    op_depth_buckets: set[str] = set()
    for node_id, op in depth_ops:
        depth = depths.get(node_id, 0)
        if max_depth <= 1:
            bucket = "middle"
        else:
            rel = depth / max(max_depth, 1)
            if rel <= 0.33:
                bucket = "early"
            elif rel <= 0.66:
                bucket = "middle"
            else:
                bucket = "late"
        op_depth_buckets.add(f"{bucket}:{op}")
    return op_depth_buckets


def _metadata_sequence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _slot_usage_features(
    metadata: dict[str, Any],
    primary_template: str,
) -> tuple[list[str], list[str], list[str]]:
    slot_entries = metadata.get("template_slot_usage") or []
    slot_keys: list[str] = []
    slot_motifs: list[str] = []
    slot_components: list[str] = []
    if not isinstance(slot_entries, list):
        return slot_keys, slot_motifs, slot_components
    for slot in slot_entries:
        if not isinstance(slot, dict):
            continue
        slot_key = str(
            slot.get("slot_key")
            or f"{slot.get('template_name', primary_template or 'unknown')}.slot{slot.get('slot_index', 0)}"
        )
        slot_keys.append(slot_key)
        selected_motif = str(slot.get("selected_motif") or "").strip()
        selected_class = str(slot.get("selected_motif_class") or "").strip()
        if selected_motif:
            slot_motifs.append(selected_motif)
            slot_components.append(f"{slot_key}:{selected_motif}")
        elif selected_class:
            slot_components.append(f"{slot_key}:{selected_class}")
    return slot_keys, slot_motifs, slot_components


def _pattern_feature_flags(
    *,
    unique_ops: list[str],
    templates: list[str],
    slot_keys: list[str],
    dynamic_feature_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pattern_has_attention": int(any("attention" in op for op in unique_ops)),
        "pattern_has_moe": int(any("moe" in op or "router" in op for op in unique_ops)),
        "pattern_has_routing": int(
            any(
                token in op
                for op in unique_ops
                for token in ("route", "gate", "router")
            )
        ),
        "pattern_has_ssm": int(
            any(
                token in op
                for op in unique_ops
                for token in ("scan", "state_space", "rwkv")
            )
        ),
        "pattern_has_math_space": int(
            any(
                token in op
                for op in unique_ops
                for token in ("tropical", "padic", "clifford", "hyp_")
            )
        ),
        "pattern_has_residual": int("add" in unique_ops),
        "pattern_has_norm": int(any("norm" in op for op in unique_ops)),
        "pattern_multi_template": int(len(templates) > 1),
        "pattern_slot_telemetry": int(bool(slot_keys)),
        **dynamic_feature_flags,
    }


def _metadata_features(
    metadata: dict[str, Any], unique_ops: list[str]
) -> dict[str, Any]:
    templates = metadata.get("templates_used") or []
    motifs = metadata.get("motifs_used") or []
    dynamic_component_tokens, dynamic_feature_flags = dynamic_component_feature_summary(
        metadata
    )
    primary_template = str(
        metadata.get("primary_template")
        or (templates[0] if isinstance(templates, list) and templates else "")
    )
    template_names = _metadata_sequence(templates)
    motif_names = _metadata_sequence(motifs)
    slot_keys, slot_motifs, slot_components = _slot_usage_features(
        metadata,
        primary_template,
    )
    feature_flags = _pattern_feature_flags(
        unique_ops=unique_ops,
        templates=template_names,
        slot_keys=slot_keys,
        dynamic_feature_flags=dynamic_feature_flags,
    )
    return {
        "primary_template": primary_template,
        "templates_used": template_names,
        "motifs_used": motif_names,
        "dynamic_components": dynamic_component_tokens,
        "slot_keys": slot_keys,
        "slot_motifs": slot_motifs,
        "slot_components": slot_components,
        **feature_flags,
    }


def _graph_features(graph_json: Any) -> dict[str, Any]:
    graph = _safe_json_loads(graph_json)
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    unique_ops, op_pairs, depth_ops = _graph_topology_features(_iter_nodes(graph))
    return {
        **_metadata_features(metadata, unique_ops),
        "ops": unique_ops,
        "op_pairs": op_pairs,
        "depth_ops": depth_ops,
    }


def _cohort_mask(
    df: pd.DataFrame, trust_values: Iterable[str], comparability_values: Iterable[str]
) -> pd.Series:
    return df["trust_label"].isin(tuple(trust_values)) & df["comparability_label"].isin(
        tuple(comparability_values)
    )


def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    dedup = (
        df.sort_values(["timestamp", "result_id"])
        .groupby(["graph_fingerprint", "evaluation_protocol_version"], as_index=False)
        .tail(1)
        .copy()
    )
    return dedup.reset_index(drop=True)


def _winsorize(
    series: pd.Series, *, low: float = 0.02, high: float = 0.98
) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    lo = clean.quantile(low)
    hi = clean.quantile(high)
    return series.clip(lower=lo, upper=hi)


def _percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        numeric = -numeric
    mask = numeric.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if mask.any():
        out.loc[mask] = numeric.loc[mask].rank(pct=True, method="average")
    return out


def _compose_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quality_metric"] = -pd.to_numeric(
        out["validation_loss_ratio"].where(
            out["validation_loss_ratio"].notna(), out["loss_ratio"]
        ),
        errors="coerce",
    )
    out["wikitext_quality"] = pd.to_numeric(
        out["wikitext_score"].where(
            out["wikitext_score"].notna(), -out["wikitext_perplexity"]
        ),
        errors="coerce",
    )
    out["efficiency_metric"] = pd.to_numeric(
        out["efficiency_multiple"].where(
            out["efficiency_multiple"].notna(), out["efficiency_wall_score"]
        ),
        errors="coerce",
    )
    primary_num = pd.Series(0.0, index=out.index, dtype=float)
    primary_den = pd.Series(0.0, index=out.index, dtype=float)
    for metric_name, higher_is_better, weight in PRIMARY_METRICS:
        ranked = _percentile_rank(out[metric_name], higher_is_better=higher_is_better)
        out[f"{metric_name}__pct"] = ranked
        valid = ranked.notna().astype(float)
        primary_num = primary_num.add(ranked.fillna(0.0) * weight, fill_value=0.0)
        primary_den = primary_den.add(valid * weight, fill_value=0.0)
    secondary_num = pd.Series(0.0, index=out.index, dtype=float)
    secondary_den = pd.Series(0.0, index=out.index, dtype=float)
    for metric_name, higher_is_better, weight in SECONDARY_METRICS:
        ranked = _percentile_rank(out[metric_name], higher_is_better=higher_is_better)
        out[f"{metric_name}__pct"] = ranked
        valid = ranked.notna().astype(float)
        secondary_num = secondary_num.add(ranked.fillna(0.0) * weight, fill_value=0.0)
        secondary_den = secondary_den.add(valid * weight, fill_value=0.0)
    primary_total = primary_num / primary_den.replace(0.0, np.nan)
    secondary_total = secondary_num / secondary_den.replace(0.0, np.nan)
    penalty = (
        0.18 * pd.to_numeric(out["validation_is_unstable"], errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out["local_only"], errors="coerce").fillna(0.0)
        + 0.08
        * (
            pd.to_numeric(out["has_nan_output"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["has_inf_output"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["has_nan_grad"], errors="coerce").fillna(0.0)
        ).clip(upper=1.0)
        + 0.05
        * (
            pd.to_numeric(out["routing_collapse_score"], errors="coerce").fillna(0.0)
            > 0.3
        ).astype(float)
    )
    combined = (0.7 * primary_total.fillna(0.0)) + (0.3 * secondary_total.fillna(0.0))
    available = (primary_den.fillna(0.0) > 0.0) | (secondary_den.fillna(0.0) > 0.0)
    out["great_score"] = np.where(
        available,
        (combined - penalty).clip(0.0, 1.0) * 100.0,
        np.nan,
    )
    return out


def _load_strength_records(db_path: str | Path) -> list[dict[str, Any]]:
    from ..notebook.shared_conn import get_notebook_conn
    from ..notebook.graph_artifacts import resolve_graph_json_value

    conn = get_notebook_conn(str(db_path))
    rows = conn.execute(BASE_ANALYSIS_QUERY).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["graph_json"] = resolve_graph_json_value(
            conn,
            db_path,
            record.get("graph_json"),
        )
        record.update(_parse_config_features(record.get("config_json")))
        record.update(_graph_features(record.get("graph_json")))
        records.append(record)
    return records


def _strength_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    merged = pd.DataFrame.from_records(records)
    if merged.empty:
        merged = pd.DataFrame(columns=EMPTY_STRENGTH_COLUMNS)
    return merged


def _normalize_strength_frame(merged: pd.DataFrame) -> pd.DataFrame:
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged["log_param_count"] = np.log1p(
        pd.to_numeric(merged["param_count"], errors="coerce")
    )
    merged["log_train_budget"] = np.log1p(
        pd.to_numeric(
            merged["train_budget_steps"].where(
                merged["train_budget_steps"].notna(), merged["cfg_stage1_steps"]
            ),
            errors="coerce",
        )
    )
    merged["log_total_train_time_ms"] = np.log1p(
        pd.to_numeric(merged["total_train_time_ms"], errors="coerce")
    )
    merged["graph_depth"] = pd.to_numeric(merged["graph_depth"], errors="coerce")
    merged["graph_n_ops"] = pd.to_numeric(merged["graph_n_ops"], errors="coerce")
    merged["graph_n_unique_ops"] = pd.to_numeric(
        merged["graph_n_unique_ops"], errors="coerce"
    )
    merged["stage1_passed"] = pd.to_numeric(
        merged["stage1_passed"], errors="coerce"
    ).fillna(0.0)
    merged["loss_ratio"] = _winsorize(
        pd.to_numeric(merged["loss_ratio"], errors="coerce")
    )
    merged["validation_loss_ratio"] = _winsorize(
        pd.to_numeric(merged["validation_loss_ratio"], errors="coerce")
    )
    merged["wikitext_perplexity"] = _winsorize(
        pd.to_numeric(merged["wikitext_perplexity"], errors="coerce"),
        low=0.01,
        high=0.99,
    )
    return _compose_score(merged)


def load_strength_datasets(db_path: str | Path) -> StrengthDatasets:
    merged = _normalize_strength_frame(_strength_frame(_load_strength_records(db_path)))

    trusted_mask = _cohort_mask(
        merged, TRUSTED_TRUST_LABELS, TRUSTED_COMPARABILITY_LABELS
    )
    promotable_mask = _cohort_mask(
        merged, PROMOTABLE_TRUST_LABELS, PROMOTABLE_COMPARABILITY_LABELS
    )

    trusted = merged.loc[trusted_mask].copy()
    promotable = merged.loc[promotable_mask].copy()
    return StrengthDatasets(
        all_runs=merged,
        trusted_runs=trusted,
        promotable_runs=promotable,
        dedup_all=_dedupe_latest(merged),
        dedup_trusted=_dedupe_latest(trusted),
        dedup_promotable=_dedupe_latest(promotable),
    )


def _base_regression_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "target": pd.to_numeric(df[target], errors="coerce"),
            "timestamp": pd.to_numeric(df["timestamp"], errors="coerce"),
            "log_param_count": pd.to_numeric(df["log_param_count"], errors="coerce"),
            "log_train_budget": pd.to_numeric(df["log_train_budget"], errors="coerce"),
            "graph_depth": pd.to_numeric(df["graph_depth"], errors="coerce"),
            "graph_n_ops": pd.to_numeric(df["graph_n_ops"], errors="coerce"),
            "graph_n_unique_ops": pd.to_numeric(
                df["graph_n_unique_ops"], errors="coerce"
            ),
            "protocol": df["evaluation_protocol_version"].fillna("unknown"),
            "cohort": df["result_cohort"].fillna("unknown"),
            "primary_template": df["primary_template"].fillna("unknown"),
        }
    )
    work = work[work["target"].notna()].copy()
    return work


def _prepare_base_matrix(
    df: pd.DataFrame, target: str, *, include_template_fixed_effects: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    work = _base_regression_frame(df, target)
    design_parts = [
        work[
            [
                "timestamp",
                "log_param_count",
                "log_train_budget",
                "graph_depth",
                "graph_n_ops",
                "graph_n_unique_ops",
            ]
        ].fillna(0.0),
        pd.get_dummies(
            work["protocol"], prefix="protocol", drop_first=True, dtype=float
        ),
        pd.get_dummies(work["cohort"], prefix="cohort", drop_first=True, dtype=float),
    ]
    if include_template_fixed_effects:
        design_parts.append(
            pd.get_dummies(
                work["primary_template"], prefix="tpl", drop_first=True, dtype=float
            )
        )
    design = pd.concat(design_parts, axis=1).astype(float)
    design = sm.add_constant(design, has_constant="add").astype(float)
    return work, design.to_numpy(dtype=float)


def _fit_feature_effect_from_base(
    work: pd.DataFrame,
    base_x: np.ndarray,
    feature_values: pd.Series,
) -> dict[str, Any] | None:
    feature = pd.to_numeric(feature_values, errors="coerce").fillna(0.0)
    feature = feature.loc[work.index].to_numpy(dtype=float)
    if feature.sum() < 2 or (1.0 - feature).sum() < 2:
        return None
    x = np.column_stack([base_x, feature])
    y = work["target"].to_numpy(dtype=float)
    try:
        beta, _resid, _rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        beta = None
    if beta is None:
        pos = work.loc[feature > 0.5, "target"]
        neg = work.loc[feature <= 0.5, "target"]
        if pos.empty or neg.empty:
            return None
        effect = float(pos.mean() - neg.mean())
        return {
            "effect": effect,
            "p_value": 1.0,
            "ci_low": effect,
            "ci_high": effect,
            "n_obs": int(len(work)),
            "r2": 0.0,
        }
    y_hat = x @ beta
    resid = y - y_hat
    n_obs = int(len(y))
    n_params = int(x.shape[1])
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if sst <= 0.0 else 1.0 - (sse / sst)
    effect = float(beta[-1])
    dof = max(n_obs - n_params, 1)
    xtx_inv = np.linalg.pinv(x.T @ x)
    sigma2 = sse / dof
    se = math.sqrt(max(float(xtx_inv[-1, -1]) * sigma2, 0.0))
    if se <= 0.0:
        p_value = 1.0
        ci_low = effect
        ci_high = effect
    else:
        t_stat = effect / se
        p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), dof))
        t_crit = float(scipy_stats.t.ppf(0.975, dof))
        ci_low = effect - t_crit * se
        ci_high = effect + t_crit * se
    return {
        "effect": effect,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": n_obs,
        "r2": r2,
    }


def _within_template_delta(
    df: pd.DataFrame, feature: pd.Series, target: str, *, higher_is_better: bool
) -> float | None:
    feature_values = pd.to_numeric(feature, errors="coerce").fillna(0.0)
    target_values = pd.to_numeric(df[target], errors="coerce")
    valid_mask = target_values.notna()
    if not valid_mask.any():
        return None
    templates = df.loc[valid_mask, "primary_template"].fillna("unknown")
    feature_values = feature_values.loc[valid_mask]
    target_values = target_values.loc[valid_mask]
    deltas: list[float] = []
    weights: list[int] = []
    for _template, group_index in templates.groupby(
        templates, observed=False
    ).groups.items():
        group_feature = feature_values.loc[group_index]
        if group_feature.sum() < 2 or (1.0 - group_feature).sum() < 2:
            continue
        group_target = target_values.loc[group_index]
        pos_mask = group_feature > 0.5
        neg_mask = ~pos_mask
        pos = group_target.loc[pos_mask]
        neg = group_target.loc[neg_mask]
        if pos.empty or neg.empty:
            continue
        delta = float(pos.mean() - neg.mean())
        if not higher_is_better:
            delta = -delta
        deltas.append(delta)
        weights.append(int(len(group_index)))
    if not deltas:
        return None
    return float(np.average(np.asarray(deltas), weights=np.asarray(weights)))


def _feature_diagnostics(df: pd.DataFrame, feature: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(feature, errors="coerce").fillna(0.0)
    present_mask = values > 0.5
    if not present_mask.any():
        return {
            "template_count": 0,
            "dominant_template": None,
            "dominant_template_share": None,
            "protocol_count": 0,
            "dominant_protocol": None,
            "dominant_protocol_share": None,
            "experiment_count": 0,
            "matched_template_controls": 0,
        }
    templates = df["primary_template"].fillna("unknown")
    protocols = df["evaluation_protocol_version"].fillna("unknown")
    template_counts = templates.loc[present_mask].value_counts()
    protocol_counts = protocols.loc[present_mask].value_counts()
    total_by_template = templates.value_counts(sort=False)
    pos_by_template = templates.loc[present_mask].value_counts(sort=False)
    pos_by_template = pos_by_template.reindex(total_by_template.index, fill_value=0)
    neg_by_template = total_by_template - pos_by_template
    matched_template_controls = int(
        ((pos_by_template >= 2) & (neg_by_template >= 2)).sum()
    )
    present_count = int(present_mask.sum())
    return {
        "template_count": int(template_counts.size),
        "dominant_template": str(template_counts.index[0]),
        "dominant_template_share": float(template_counts.iloc[0] / present_count),
        "protocol_count": int(protocol_counts.size),
        "dominant_protocol": str(protocol_counts.index[0]),
        "dominant_protocol_share": float(protocol_counts.iloc[0] / present_count),
        "experiment_count": int(df.loc[present_mask, "experiment_id"].nunique()),
        "matched_template_controls": matched_template_controls,
    }


def _confidence_tier(
    *,
    support_graphs: int,
    dominant_template_share: float | None,
    protocol_count: int,
    matched_template_controls: int,
    within_template_delta: float | None,
) -> str:
    if support_graphs < 12 or matched_template_controls == 0:
        return "low"
    if dominant_template_share is not None and dominant_template_share >= 0.85:
        return "low"
    if protocol_count <= 1 and support_graphs < 25:
        return "low"
    if (
        support_graphs >= 100
        and matched_template_controls >= 3
        and (dominant_template_share or 0.0) < 0.75
        and within_template_delta is not None
    ):
        return "high"
    if support_graphs >= 25 and matched_template_controls >= 2:
        return "medium"
    return "low"


def _artifact_flags(
    *,
    support_graphs: int,
    dominant_template_share: float | None,
    protocol_count: int,
    matched_template_controls: int,
) -> list[str]:
    flags: list[str] = []
    if support_graphs < 12:
        flags.append("low_support")
    if dominant_template_share is not None and dominant_template_share >= 0.8:
        flags.append("template_coupled")
    if protocol_count <= 1:
        flags.append("single_protocol")
    if matched_template_controls == 0:
        flags.append("no_matched_template_control")
    return flags


def _rank_binary_features(
    df: pd.DataFrame,
    feature_map: dict[str, pd.Series],
    *,
    target: str,
    higher_is_better: bool,
    include_template_fixed_effects: bool,
    min_support: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_values = pd.to_numeric(df[target], errors="coerce")
    work, base_x = _prepare_base_matrix(
        df, target, include_template_fixed_effects=include_template_fixed_effects
    )
    if work.empty:
        return rows
    for name, feature in feature_map.items():
        values = pd.to_numeric(feature, errors="coerce").fillna(0.0)
        support = int((values > 0.5).sum())
        graphs = int(df.loc[values > 0.5, "graph_fingerprint"].nunique())
        if support < min_support or graphs < max(2, min_support // 2):
            continue
        metric = target_values.loc[values > 0.5]
        raw_mean = float(metric.dropna().mean()) if metric.notna().any() else None
        effect = _fit_feature_effect_from_base(work, base_x, values)
        if effect is None:
            continue
        within_template_delta = _within_template_delta(
            df, values, target, higher_is_better=higher_is_better
        )
        diagnostics = _feature_diagnostics(df, values)
        rows.append(
            {
                "name": name,
                "support_runs": support,
                "support_graphs": graphs,
                "raw_mean": raw_mean,
                "adjusted_effect": effect["effect"]
                if higher_is_better
                else -effect["effect"],
                "raw_effect_direction": effect["effect"],
                "p_value": effect["p_value"],
                "ci_low": effect["ci_low"],
                "ci_high": effect["ci_high"],
                "n_obs": effect["n_obs"],
                "r2": effect["r2"],
                "within_template_delta": within_template_delta,
                **diagnostics,
                "confidence_tier": _confidence_tier(
                    support_graphs=graphs,
                    dominant_template_share=diagnostics["dominant_template_share"],
                    protocol_count=diagnostics["protocol_count"],
                    matched_template_controls=diagnostics["matched_template_controls"],
                    within_template_delta=within_template_delta,
                ),
                "artifact_flags": _artifact_flags(
                    support_graphs=graphs,
                    dominant_template_share=diagnostics["dominant_template_share"],
                    protocol_count=diagnostics["protocol_count"],
                    matched_template_controls=diagnostics["matched_template_controls"],
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            -(item["adjusted_effect"]),
            item["p_value"],
            -(item["support_graphs"]),
        )
    )
    return rows


def _counter_feature_map(series: pd.Series) -> dict[str, pd.Series]:
    row_positions: dict[str, list[int]] = {}
    for pos, values in enumerate(series):
        if not isinstance(values, list):
            continue
        valid_values = {value for value in values if isinstance(value, str) and value}
        for value in valid_values:
            row_positions.setdefault(value, []).append(pos)
    columns: dict[str, pd.Series] = {}
    index = series.index
    size = len(series)
    for item, positions in row_positions.items():
        data = np.zeros(size, dtype=float)
        data[positions] = 1.0
        columns[item] = pd.Series(data, index=index)
    return columns


def _structural_feature_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    features: dict[str, pd.Series] = {}
    for column in df.columns:
        if column.startswith("pattern_"):
            features[column.removeprefix("pattern_")] = pd.to_numeric(
                df[column], errors="coerce"
            ).fillna(0.0)
    depth = pd.to_numeric(df["graph_depth"], errors="coerce")
    features["depth_shallow"] = (depth <= 4).astype(float)
    features["depth_medium"] = ((depth > 4) & (depth <= 10)).astype(float)
    features["depth_deep"] = (depth > 10).astype(float)
    width = pd.to_numeric(df["graph_n_unique_ops"], errors="coerce")
    features["op_palette_compact"] = (width <= 6).astype(float)
    features["op_palette_broad"] = (width >= 10).astype(float)
    return features


def _metric_rankings(
    datasets: StrengthDatasets, *, min_support: int, top_k: int
) -> dict[str, Any]:
    trusted = datasets.dedup_trusted
    promotable = datasets.dedup_promotable

    output: dict[str, Any] = {}
    for metric_name, spec in ANALYSIS_METRICS.items():
        source = promotable if spec["cohort"] == "promotable" else trusted
        output[metric_name] = {
            "components": _rank_binary_features(
                source,
                _counter_feature_map(source["ops"]),
                target=metric_name,
                higher_is_better=bool(spec["higher_is_better"]),
                include_template_fixed_effects=True,
                min_support=min_support,
            )[:top_k],
            "pairs": _rank_binary_features(
                source,
                _counter_feature_map(source["op_pairs"]),
                target=metric_name,
                higher_is_better=bool(spec["higher_is_better"]),
                include_template_fixed_effects=True,
                min_support=min_support,
            )[:top_k],
            "slot_components": _rank_binary_features(
                source,
                _counter_feature_map(source["slot_components"]),
                target=metric_name,
                higher_is_better=bool(spec["higher_is_better"]),
                include_template_fixed_effects=False,
                min_support=max(2, min_support // 2),
            )[:top_k],
            "templates": _rank_binary_features(
                source,
                _counter_feature_map(source["templates_used"]),
                target=metric_name,
                higher_is_better=bool(spec["higher_is_better"]),
                include_template_fixed_effects=False,
                min_support=max(2, min_support // 2),
            )[:top_k],
            "structural_patterns": _rank_binary_features(
                source,
                _structural_feature_map(source),
                target=metric_name,
                higher_is_better=bool(spec["higher_is_better"]),
                include_template_fixed_effects=False,
                min_support=min_support,
            )[:top_k],
        }
    output["best_components_overall"] = output["great_score"]["components"]
    output["best_pairs_overall"] = output["great_score"]["pairs"]
    output["best_slot_components_overall"] = output["great_score"]["slot_components"]
    output["best_templates_overall"] = output["great_score"]["templates"]
    output["best_structural_patterns_overall"] = output["great_score"][
        "structural_patterns"
    ]
    return output


def _drift_report(datasets: StrengthDatasets) -> dict[str, Any]:
    trusted = datasets.dedup_trusted
    promotable = datasets.dedup_promotable
    search_all = datasets.all_runs[
        (datasets.all_runs["result_cohort"] == "search")
        & datasets.all_runs["loss_ratio"].notna()
    ].copy()
    search_promotable = promotable[
        (promotable["result_cohort"] == "search") & promotable["loss_ratio"].notna()
    ].copy()
    for frame in (search_all, search_promotable):
        if not frame.empty:
            frame["time_z"] = (frame["timestamp"] - frame["timestamp"].mean()) / max(
                frame["timestamp"].std(ddof=0), 1e-9
            )

    def fit_time_model(
        frame: pd.DataFrame, *, include_weights: bool, include_arch: bool
    ) -> dict[str, Any] | None:
        if frame.empty or frame["loss_ratio"].notna().sum() < 20:
            return None
        design_parts = [
            frame[["time_z", "log_param_count", "log_train_budget"]].fillna(0.0),
            pd.get_dummies(
                frame["evaluation_protocol_version"].fillna("unknown"),
                prefix="protocol",
                drop_first=True,
                dtype=float,
            ),
        ]
        if include_weights:
            weight_cols = [
                column
                for column in frame.columns
                if column.startswith("category_weights::")
            ]
            if weight_cols:
                design_parts.append(frame[weight_cols].fillna(0.0))
            for column in (
                "category_weights_entropy",
                "template_weights_entropy",
                "category_weights_max_weight",
                "template_weights_max_weight",
            ):
                if column in frame.columns:
                    design_parts.append(frame[[column]].fillna(0.0))
        if include_arch:
            design_parts.append(
                frame[
                    [
                        "graph_depth",
                        "graph_n_ops",
                        "graph_n_unique_ops",
                        "routing_savings_ratio",
                        "compression_ratio",
                    ]
                ].fillna(0.0)
            )
            design_parts.append(
                pd.get_dummies(
                    frame["primary_template"].fillna("unknown"),
                    prefix="tpl",
                    drop_first=True,
                    dtype=float,
                )
            )
        design = pd.concat(design_parts, axis=1)
        design = sm.add_constant(design, has_constant="add")
        try:
            result = sm.OLS(frame["loss_ratio"], design, missing="drop").fit(
                cov_type="HC3"
            )
        except Exception:
            return None
        return {
            "time_coef": float(result.params.get("time_z", np.nan)),
            "time_p_value": float(result.pvalues.get("time_z", np.nan)),
            "r2": float(getattr(result, "rsquared", np.nan)),
            "n_obs": int(result.nobs),
        }

    def summarize_bins(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        work = frame[["timestamp", "loss_ratio"]].dropna().copy()
        work["time_bin"] = pd.qcut(work["timestamp"], 6, duplicates="drop")
        rows: list[dict[str, Any]] = []
        for key, group in work.groupby("time_bin", observed=False):
            rows.append(
                {
                    "time_bin": str(key),
                    "count": int(len(group)),
                    "mean_loss_ratio": float(group["loss_ratio"].mean()),
                    "median_loss_ratio": float(group["loss_ratio"].median()),
                }
            )
        return rows

    return {
        "all_search_bins": summarize_bins(search_all),
        "promotable_search_bins": summarize_bins(search_promotable),
        "models": {
            "all_search_time_only": fit_time_model(
                search_all, include_weights=False, include_arch=False
            ),
            "all_search_plus_weights": fit_time_model(
                search_all, include_weights=True, include_arch=False
            ),
            "all_search_plus_weights_and_arch": fit_time_model(
                search_all, include_weights=True, include_arch=True
            ),
            "promotable_time_only": fit_time_model(
                search_promotable, include_weights=False, include_arch=False
            ),
            "promotable_plus_weights": fit_time_model(
                search_promotable, include_weights=True, include_arch=False
            ),
            "promotable_plus_weights_and_arch": fit_time_model(
                search_promotable, include_weights=True, include_arch=True
            ),
        },
        "distribution_checks": {
            "trusted_loss_ratio_median": float(trusted["loss_ratio"].median())
            if not trusted.empty
            else None,
            "promotable_loss_ratio_median": float(promotable["loss_ratio"].median())
            if not promotable.empty
            else None,
            "search_all_loss_ratio_median": float(search_all["loss_ratio"].median())
            if not search_all.empty
            else None,
            "search_promotable_loss_ratio_median": float(
                search_promotable["loss_ratio"].median()
            )
            if not search_promotable.empty
            else None,
        },
    }


def _weight_bias_summary(datasets: StrengthDatasets) -> dict[str, Any]:
    search = datasets.all_runs[datasets.all_runs["result_cohort"] == "search"]
    if search.empty:
        return {"top_weighted_categories": [], "category_weight_vs_loss": []}
    weight_columns = sorted(
        column for column in search.columns if column.startswith("category_weights::")
    )
    top_weighted_categories = []
    correlations = []
    for column in weight_columns:
        series = pd.to_numeric(search[column], errors="coerce")
        if series.notna().sum() < 20:
            continue
        correlations.append(
            {
                "category": column.split("::", 1)[1],
                "corr_with_loss_ratio": float(
                    series.corr(pd.to_numeric(search["loss_ratio"], errors="coerce"))
                ),
                "corr_with_stage1_passed": float(
                    series.corr(pd.to_numeric(search["stage1_passed"], errors="coerce"))
                ),
                "mean_weight": float(series.mean()),
            }
        )
    mean_weights = {
        column.split("::", 1)[1]: float(
            pd.to_numeric(search[column], errors="coerce").mean()
        )
        for column in weight_columns
        if pd.to_numeric(search[column], errors="coerce").notna().sum() >= 5
    }
    top_weighted_categories = [
        {"category": key, "mean_weight": value}
        for key, value in sorted(
            mean_weights.items(), key=lambda item: item[1], reverse=True
        )[:8]
    ]
    correlations.sort(
        key=lambda item: (
            abs(item["corr_with_loss_ratio"])
            if not math.isnan(item["corr_with_loss_ratio"])
            else -1.0
        ),
        reverse=True,
    )
    return {
        "top_weighted_categories": top_weighted_categories,
        "category_weight_vs_loss": correlations[:10],
    }


def _support_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "runs": int(len(df)),
        "unique_graphs": int(df["graph_fingerprint"].nunique()) if not df.empty else 0,
        "loss_ratio_coverage": int(df["loss_ratio"].notna().sum())
        if "loss_ratio" in df
        else 0,
        "induction_coverage": int(df["induction_screening_auc"].notna().sum())
        if "induction_screening_auc" in df
        else 0,
        "binding_coverage": int(df["binding_screening_auc"].notna().sum())
        if "binding_screening_auc" in df
        else 0,
        "hellaswag_coverage": int(df["hellaswag_acc"].notna().sum())
        if "hellaswag_acc" in df
        else 0,
        "wikitext_coverage": int(df["wikitext_perplexity"].notna().sum())
        if "wikitext_perplexity" in df
        else 0,
    }


def build_model_strength_report(
    db_path: str | Path,
    *,
    min_support: int = 12,
    top_k: int = 20,
) -> dict[str, Any]:
    datasets = load_strength_datasets(db_path)
    rankings = _metric_rankings(datasets, min_support=min_support, top_k=top_k)
    return {
        "metadata": {
            "db_path": str(Path(db_path)),
            "base_query": BASE_ANALYSIS_QUERY.strip(),
            "dedupe_policy": "latest row per (graph_fingerprint, evaluation_protocol_version)",
            "min_support": min_support,
            "top_k": top_k,
        },
        "support": {
            "all_runs": _support_summary(datasets.all_runs),
            "trusted_runs": _support_summary(datasets.trusted_runs),
            "promotable_runs": _support_summary(datasets.promotable_runs),
            "dedup_trusted": _support_summary(datasets.dedup_trusted),
            "dedup_promotable": _support_summary(datasets.dedup_promotable),
        },
        "great_model_definition": {
            "primary_metrics": [
                {
                    "metric": metric_name,
                    "weight": weight,
                    "direction": "higher" if higher_is_better else "lower",
                }
                for metric_name, higher_is_better, weight in PRIMARY_METRICS
            ],
            "secondary_metrics": [
                {
                    "metric": metric_name,
                    "weight": weight,
                    "direction": "higher" if higher_is_better else "lower",
                }
                for metric_name, higher_is_better, weight in SECONDARY_METRICS
            ],
            "penalties": [
                "validation_is_unstable",
                "local_only",
                "has_nan_output / has_inf_output / has_nan_grad",
                "routing_collapse_score > 0.3",
            ],
            "score_name": "great_score",
            "score_scale": "0-100 percentile composite with explicit penalties",
        },
        "rankings": rankings,
        "drift_analysis": _drift_report(datasets),
        "weight_bias": _weight_bias_summary(datasets),
        "query_provenance": {
            "source_functions": [
                "research.scientist.analytics.model_strength.load_strength_datasets",
                "research.scientist.analytics.model_strength._graph_features",
                "research.scientist.analytics.model_strength._rank_binary_features",
                "research.scientist.analytics.model_strength._drift_report",
            ],
        },
    }
