from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as scipy_stats
from scipy.stats import rankdata

from ..trust_policy import (
    PROMOTABLE_COMPARABILITY_LABELS,
    PROMOTABLE_TRUST_LABELS,
    TRUSTED_COMPARABILITY_LABELS,
    TRUSTED_TRUST_LABELS,
)
from .model_strength_features import (
    graph_features as _graph_features,
    parse_config_features as _parse_config_features,
)
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
    all_runs: pl.DataFrame
    trusted_runs: pl.DataFrame
    promotable_runs: pl.DataFrame
    dedup_all: pl.DataFrame
    dedup_trusted: pl.DataFrame
    dedup_promotable: pl.DataFrame


# --------------------------------------------------------------------------- #
# Column extraction helpers (Polars column -> numpy, null/NaN unified)         #
# --------------------------------------------------------------------------- #


def _fcol(df: pl.DataFrame, name: str) -> np.ndarray:
    """Numeric column as float64 numpy with NaN for null/unparseable values."""
    return (
        df.get_column(name)
        .cast(pl.Float64, strict=False)
        .to_numpy()
        .astype(float, copy=False)
    )


def _scol(df: pl.DataFrame, name: str, fill: str) -> np.ndarray:
    """String column as an object numpy array with nulls replaced by ``fill``."""
    values = df.get_column(name).cast(pl.Utf8, strict=False).fill_null(fill).to_list()
    return np.array(values, dtype=object)


def _dummies(values: np.ndarray) -> np.ndarray:
    """One-hot matrix matching ``pd.get_dummies(..., drop_first=True, dtype=float)``.

    Categories are sorted ascending and the first is dropped, so the column
    count (and therefore the regression degrees of freedom) matches the pandas
    baseline exactly.
    """
    categories = sorted(set(values.tolist()))[1:]
    if not categories:
        return np.zeros((len(values), 0), dtype=float)
    matrix = np.zeros((len(values), len(categories)), dtype=float)
    for col, category in enumerate(categories):
        matrix[:, col] = (values == category).astype(float)
    return matrix


def _value_counts_desc(values: np.ndarray) -> list[tuple[Any, int]]:
    """Replicate ``pd.Series.value_counts`` order: count desc, ties by first seen."""
    counts: dict[Any, int] = {}
    first_seen: dict[Any, int] = {}
    for pos, value in enumerate(values.tolist()):
        if value not in counts:
            counts[value] = 0
            first_seen[value] = pos
        counts[value] += 1
    ordered = sorted(counts, key=lambda v: (-counts[v], first_seen[v]))
    return [(value, counts[value]) for value in ordered]


# --------------------------------------------------------------------------- #
# Frame construction, normalisation, scoring                                   #
# --------------------------------------------------------------------------- #


def _cohort_filter(
    df: pl.DataFrame,
    trust_values: Iterable[str],
    comparability_values: Iterable[str],
) -> pl.DataFrame:
    return df.filter(
        pl.col("trust_label").is_in(list(trust_values))
        & pl.col("comparability_label").is_in(list(comparability_values))
    )


def _dedupe_latest(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df.clone()
    # pandas groupby drops rows with a null group key (dropna=True) and sorts by
    # key; replicate both so downstream value_counts tie-breaks match.
    return (
        df.filter(
            pl.col("graph_fingerprint").is_not_null()
            & pl.col("evaluation_protocol_version").is_not_null()
        )
        .sort(["timestamp", "result_id"])
        .group_by(
            ["graph_fingerprint", "evaluation_protocol_version"], maintain_order=True
        )
        .last()
        .sort(["graph_fingerprint", "evaluation_protocol_version"])
    )


def _winsorize(
    values: np.ndarray, *, low: float = 0.02, high: float = 0.98
) -> np.ndarray:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return values
    lo = float(np.percentile(clean, low * 100.0))
    hi = float(np.percentile(clean, high * 100.0))
    return np.clip(values, lo, hi)


def _percentile_rank(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    numeric = values.astype(float, copy=True)
    if not higher_is_better:
        numeric = -numeric
    out = np.full(numeric.shape, np.nan)
    mask = ~np.isnan(numeric)
    count = int(mask.sum())
    if count:
        out[mask] = rankdata(numeric[mask], method="average") / count
    return out


def _compose_score(df: pl.DataFrame) -> pl.DataFrame:
    validation_loss = _fcol(df, "validation_loss_ratio")
    loss = _fcol(df, "loss_ratio")
    quality = -np.where(np.isnan(validation_loss), loss, validation_loss)
    wikitext_score = _fcol(df, "wikitext_score")
    wikitext_perplexity = _fcol(df, "wikitext_perplexity")
    wikitext_quality = np.where(
        np.isnan(wikitext_score), -wikitext_perplexity, wikitext_score
    )
    efficiency_multiple = _fcol(df, "efficiency_multiple")
    efficiency_wall = _fcol(df, "efficiency_wall_score")
    efficiency = np.where(
        np.isnan(efficiency_multiple), efficiency_wall, efficiency_multiple
    )

    metric_arrays = {
        "quality_metric": quality,
        "wikitext_quality": wikitext_quality,
        "efficiency_metric": efficiency,
        "induction_screening_auc": _fcol(df, "induction_screening_auc"),
        "binding_screening_auc": _fcol(df, "binding_screening_auc"),
        "hellaswag_acc": _fcol(df, "hellaswag_acc"),
        "stability_score": _fcol(df, "stability_score"),
        "validation_robustness_score": _fcol(df, "validation_robustness_score"),
        "novelty_score": _fcol(df, "novelty_score"),
        "stage1_passed": _fcol(df, "stage1_passed"),
    }
    rows = df.height

    def accumulate(
        specs: tuple[tuple[str, bool, float], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        num = np.zeros(rows)
        den = np.zeros(rows)
        for metric_name, higher_is_better, weight in specs:
            ranked = _percentile_rank(
                metric_arrays[metric_name], higher_is_better=higher_is_better
            )
            valid = (~np.isnan(ranked)).astype(float)
            num += np.nan_to_num(ranked) * weight
            den += valid * weight
        return num, den

    primary_num, primary_den = accumulate(PRIMARY_METRICS)
    secondary_num, secondary_den = accumulate(SECONDARY_METRICS)
    with np.errstate(invalid="ignore", divide="ignore"):
        primary_total = primary_num / np.where(primary_den == 0.0, np.nan, primary_den)
        secondary_total = secondary_num / np.where(
            secondary_den == 0.0, np.nan, secondary_den
        )

    def penalty_term(name: str) -> np.ndarray:
        return np.nan_to_num(_fcol(df, name))

    penalty = (
        0.18 * penalty_term("validation_is_unstable")
        + 0.10 * penalty_term("local_only")
        + 0.08
        * np.minimum(
            penalty_term("has_nan_output")
            + penalty_term("has_inf_output")
            + penalty_term("has_nan_grad"),
            1.0,
        )
        + 0.05 * (penalty_term("routing_collapse_score") > 0.3).astype(float)
    )
    combined = 0.7 * np.nan_to_num(primary_total) + 0.3 * np.nan_to_num(secondary_total)
    available = (primary_den > 0.0) | (secondary_den > 0.0)
    great_score = np.where(
        available, np.clip(combined - penalty, 0.0, 1.0) * 100.0, np.nan
    )
    return df.with_columns(
        pl.Series("quality_metric", quality),
        pl.Series("wikitext_quality", wikitext_quality),
        pl.Series("efficiency_metric", efficiency),
        pl.Series("great_score", great_score),
    )


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


def _strength_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame({column: [] for column in EMPTY_STRENGTH_COLUMNS})
    # Union of keys preserving first-seen order (pandas from_records semantics).
    columns = list(dict.fromkeys(key for record in records for key in record))
    data = {column: [record.get(column) for record in records] for column in columns}
    return pl.DataFrame(data, strict=False)


def _log1p_numeric(df: pl.DataFrame, name: str) -> np.ndarray:
    return np.log1p(_fcol(df, name))


def _normalize_strength_frame(merged: pl.DataFrame) -> pl.DataFrame:
    train_budget = _fcol(merged, "train_budget_steps")
    cfg_steps = _fcol(merged, "cfg_stage1_steps")
    log_train_budget = np.log1p(
        np.where(np.isnan(train_budget), cfg_steps, train_budget)
    )
    merged = merged.with_columns(
        pl.Series("timestamp", _fcol(merged, "timestamp")),
        pl.Series("log_param_count", _log1p_numeric(merged, "param_count")),
        pl.Series("log_train_budget", log_train_budget),
        pl.Series(
            "log_total_train_time_ms", _log1p_numeric(merged, "total_train_time_ms")
        ),
        pl.Series("graph_depth", _fcol(merged, "graph_depth")),
        pl.Series("graph_n_ops", _fcol(merged, "graph_n_ops")),
        pl.Series("graph_n_unique_ops", _fcol(merged, "graph_n_unique_ops")),
        pl.Series("stage1_passed", np.nan_to_num(_fcol(merged, "stage1_passed"))),
        pl.Series("loss_ratio", _winsorize(_fcol(merged, "loss_ratio"))),
        pl.Series(
            "validation_loss_ratio", _winsorize(_fcol(merged, "validation_loss_ratio"))
        ),
        pl.Series(
            "wikitext_perplexity",
            _winsorize(_fcol(merged, "wikitext_perplexity"), low=0.01, high=0.99),
        ),
    )
    return _compose_score(merged)


def load_strength_datasets(db_path: str | Path) -> StrengthDatasets:
    merged = _normalize_strength_frame(_strength_frame(_load_strength_records(db_path)))

    trusted = _cohort_filter(merged, TRUSTED_TRUST_LABELS, TRUSTED_COMPARABILITY_LABELS)
    promotable = _cohort_filter(
        merged, PROMOTABLE_TRUST_LABELS, PROMOTABLE_COMPARABILITY_LABELS
    )
    return StrengthDatasets(
        all_runs=merged,
        trusted_runs=trusted,
        promotable_runs=promotable,
        dedup_all=_dedupe_latest(merged),
        dedup_trusted=_dedupe_latest(trusted),
        dedup_promotable=_dedupe_latest(promotable),
    )


# --------------------------------------------------------------------------- #
# Regression + feature ranking (numpy design matrices, statsmodels/scipy math) #
# --------------------------------------------------------------------------- #


def _num_block(
    df: pl.DataFrame, columns: Sequence[str], mask: np.ndarray
) -> np.ndarray:
    if not columns:
        return np.zeros((int(mask.sum()), 0), dtype=float)
    return np.column_stack(
        [np.nan_to_num(_fcol(df, column)[mask]) for column in columns]
    )


_BASE_NUMERIC_COLUMNS = (
    "timestamp",
    "log_param_count",
    "log_train_budget",
    "graph_depth",
    "graph_n_ops",
    "graph_n_unique_ops",
)


def _prepare_base_matrix(
    df: pl.DataFrame, target: str, *, include_template_fixed_effects: bool
) -> tuple[np.ndarray, np.ndarray]:
    target_values = _fcol(df, target)
    valid = ~np.isnan(target_values)
    numeric = _num_block(df, _BASE_NUMERIC_COLUMNS, valid)
    parts = [
        numeric,
        _dummies(_scol(df, "evaluation_protocol_version", "unknown")[valid]),
        _dummies(_scol(df, "result_cohort", "unknown")[valid]),
    ]
    if include_template_fixed_effects:
        parts.append(_dummies(_scol(df, "primary_template", "unknown")[valid]))
    design = np.column_stack([part for part in parts if part.shape[1] or part.size])
    design = sm.add_constant(design, has_constant="add").astype(float)
    return valid, design


def _fit_feature_effect_from_base(
    valid: np.ndarray,
    base_x: np.ndarray,
    y: np.ndarray,
    feature_values: np.ndarray,
) -> dict[str, Any] | None:
    feature = np.nan_to_num(feature_values)[valid]
    if feature.sum() < 2 or (1.0 - feature).sum() < 2:
        return None
    x = np.column_stack([base_x, feature])
    try:
        beta, _resid, _rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        beta = None
    if beta is None:
        pos = y[feature > 0.5]
        neg = y[feature <= 0.5]
        if pos.size == 0 or neg.size == 0:
            return None
        effect = float(pos.mean() - neg.mean())
        return {
            "effect": effect,
            "p_value": 1.0,
            "ci_low": effect,
            "ci_high": effect,
            "n_obs": int(len(y)),
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
    df: pl.DataFrame, feature: np.ndarray, target: str, *, higher_is_better: bool
) -> float | None:
    feature_values = np.nan_to_num(feature)
    target_values = _fcol(df, target)
    valid_mask = ~np.isnan(target_values)
    if not valid_mask.any():
        return None
    templates = _scol(df, "primary_template", "unknown")[valid_mask]
    feature_values = feature_values[valid_mask]
    target_values = target_values[valid_mask]
    deltas: list[float] = []
    weights: list[int] = []
    for template in sorted(set(templates.tolist())):
        group_mask = templates == template
        group_feature = feature_values[group_mask]
        if group_feature.sum() < 2 or (1.0 - group_feature).sum() < 2:
            continue
        group_target = target_values[group_mask]
        pos = group_target[group_feature > 0.5]
        neg = group_target[group_feature <= 0.5]
        if pos.size == 0 or neg.size == 0:
            continue
        delta = float(pos.mean() - neg.mean())
        if not higher_is_better:
            delta = -delta
        deltas.append(delta)
        weights.append(int(group_mask.sum()))
    if not deltas:
        return None
    return float(np.average(np.asarray(deltas), weights=np.asarray(weights)))


def _feature_diagnostics(df: pl.DataFrame, feature: np.ndarray) -> dict[str, Any]:
    values = np.nan_to_num(feature)
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
    templates = _scol(df, "primary_template", "unknown")
    protocols = _scol(df, "evaluation_protocol_version", "unknown")
    template_counts = _value_counts_desc(templates[present_mask])
    protocol_counts = _value_counts_desc(protocols[present_mask])

    present_by_template: dict[Any, int] = dict(
        _value_counts_desc(templates[present_mask])
    )
    matched_template_controls = 0
    for template, total in _value_counts_desc(templates):
        pos = present_by_template.get(template, 0)
        neg = total - pos
        if pos >= 2 and neg >= 2:
            matched_template_controls += 1

    present_count = int(present_mask.sum())
    experiment_ids = df.get_column("experiment_id").to_list()
    experiment_count = len(
        {
            experiment_ids[pos]
            for pos in np.flatnonzero(present_mask)
            if experiment_ids[pos] is not None
        }
    )
    return {
        "template_count": len(template_counts),
        "dominant_template": str(template_counts[0][0]),
        "dominant_template_share": float(template_counts[0][1] / present_count),
        "protocol_count": len(protocol_counts),
        "dominant_protocol": str(protocol_counts[0][0]),
        "dominant_protocol_share": float(protocol_counts[0][1] / present_count),
        "experiment_count": experiment_count,
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
    df: pl.DataFrame,
    feature_map: dict[str, np.ndarray],
    *,
    target: str,
    higher_is_better: bool,
    include_template_fixed_effects: bool,
    min_support: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.height == 0:
        return rows
    target_values = _fcol(df, target)
    valid, base_x = _prepare_base_matrix(
        df, target, include_template_fixed_effects=include_template_fixed_effects
    )
    if not valid.any():
        return rows
    y = target_values[valid]
    fingerprints = _scol(df, "graph_fingerprint", "")
    for name, feature in feature_map.items():
        values = np.nan_to_num(feature)
        present = values > 0.5
        support = int(present.sum())
        graphs = len({fp for fp in fingerprints[present].tolist()})
        if support < min_support or graphs < max(2, min_support // 2):
            continue
        metric = target_values[present]
        finite_metric = metric[~np.isnan(metric)]
        raw_mean = float(finite_metric.mean()) if finite_metric.size else None
        effect = _fit_feature_effect_from_base(valid, base_x, y, values)
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


def _counter_feature_map(series: Iterable[Any]) -> dict[str, np.ndarray]:
    """Build per-token binary indicator vectors from a column of string lists.

    Returns positional float64 arrays (1.0 where the row's list contains the
    token). Non-list entries contribute nothing.
    """
    materialised = list(series)
    row_positions: dict[str, list[int]] = {}
    for pos, values in enumerate(materialised):
        if not isinstance(values, list):
            continue
        valid_values = {value for value in values if isinstance(value, str) and value}
        for value in valid_values:
            row_positions.setdefault(value, []).append(pos)
    size = len(materialised)
    columns: dict[str, np.ndarray] = {}
    for item, positions in row_positions.items():
        data = np.zeros(size, dtype=float)
        data[positions] = 1.0
        columns[item] = data
    return columns


def _structural_feature_map(df: pl.DataFrame) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for column in df.columns:
        if column.startswith("pattern_"):
            features[column.removeprefix("pattern_")] = np.nan_to_num(_fcol(df, column))
    depth = _fcol(df, "graph_depth")
    features["depth_shallow"] = (depth <= 4).astype(float)
    features["depth_medium"] = ((depth > 4) & (depth <= 10)).astype(float)
    features["depth_deep"] = (depth > 10).astype(float)
    width = _fcol(df, "graph_n_unique_ops")
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
        higher_is_better = bool(spec["higher_is_better"])
        output[metric_name] = {
            "components": _rank_binary_features(
                source,
                _counter_feature_map(source.get_column("ops").to_list()),
                target=metric_name,
                higher_is_better=higher_is_better,
                include_template_fixed_effects=True,
                min_support=min_support,
            )[:top_k],
            "pairs": _rank_binary_features(
                source,
                _counter_feature_map(source.get_column("op_pairs").to_list()),
                target=metric_name,
                higher_is_better=higher_is_better,
                include_template_fixed_effects=True,
                min_support=min_support,
            )[:top_k],
            "slot_components": _rank_binary_features(
                source,
                _counter_feature_map(source.get_column("slot_components").to_list()),
                target=metric_name,
                higher_is_better=higher_is_better,
                include_template_fixed_effects=False,
                min_support=max(2, min_support // 2),
            )[:top_k],
            "templates": _rank_binary_features(
                source,
                _counter_feature_map(source.get_column("templates_used").to_list()),
                target=metric_name,
                higher_is_better=higher_is_better,
                include_template_fixed_effects=False,
                min_support=max(2, min_support // 2),
            )[:top_k],
            "structural_patterns": _rank_binary_features(
                source,
                _structural_feature_map(source),
                target=metric_name,
                higher_is_better=higher_is_better,
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


# --------------------------------------------------------------------------- #
# Drift + weight-bias reporting                                                #
# --------------------------------------------------------------------------- #


def _search_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df.clone()
    cohort = _scol(df, "result_cohort", "")
    loss = _fcol(df, "loss_ratio")
    mask = (cohort == "search") & (~np.isnan(loss))
    return df.filter(pl.Series(mask))


def _time_z(frame: pl.DataFrame) -> np.ndarray:
    timestamp = _fcol(frame, "timestamp")
    mean = float(np.nanmean(timestamp)) if timestamp.size else 0.0
    std = float(np.nanstd(timestamp)) if timestamp.size else 0.0
    return (timestamp - mean) / max(std, 1e-9)


def _fit_time_model(
    frame: pl.DataFrame, *, include_weights: bool, include_arch: bool
) -> dict[str, Any] | None:
    loss = _fcol(frame, "loss_ratio")
    if frame.height == 0 or int((~np.isnan(loss)).sum()) < 20:
        return None
    keep = np.ones(frame.height, dtype=bool)
    time_z = _time_z(frame)
    parts: list[np.ndarray] = [
        time_z.reshape(-1, 1),
        _num_block(frame, ("log_param_count", "log_train_budget"), keep),
        _dummies(_scol(frame, "evaluation_protocol_version", "unknown")),
    ]
    if include_weights:
        weight_columns = [
            column
            for column in frame.columns
            if column.startswith("category_weights::")
        ]
        if weight_columns:
            parts.append(_num_block(frame, weight_columns, keep))
        for column in (
            "category_weights_entropy",
            "template_weights_entropy",
            "category_weights_max_weight",
            "template_weights_max_weight",
        ):
            if column in frame.columns:
                parts.append(_num_block(frame, (column,), keep))
    if include_arch:
        parts.append(
            _num_block(
                frame,
                (
                    "graph_depth",
                    "graph_n_ops",
                    "graph_n_unique_ops",
                    "routing_savings_ratio",
                    "compression_ratio",
                ),
                keep,
            )
        )
        parts.append(_dummies(_scol(frame, "primary_template", "unknown")))
    design = np.column_stack([part for part in parts if part.shape[1] or part.size])
    design = sm.add_constant(design, has_constant="add", prepend=True).astype(float)
    try:
        result = sm.OLS(loss, design, missing="drop").fit(cov_type="HC3")
    except Exception:
        return None
    # add_constant prepends the intercept, so time_z sits at design column 1.
    params = np.asarray(result.params, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    return {
        "time_coef": float(params[1]) if params.size > 1 else float("nan"),
        "time_p_value": float(pvalues[1]) if pvalues.size > 1 else float("nan"),
        "r2": float(getattr(result, "rsquared", float("nan"))),
        "n_obs": int(result.nobs),
    }


def _round_frac(x: float, precision: int) -> float:
    if not np.isfinite(x) or x == 0:
        return x
    frac, whole = np.modf(x)
    if whole == 0:
        digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
    else:
        digits = precision
    return float(np.around(x, digits))


def _infer_precision(base_precision: int, bins: np.ndarray) -> int:
    for precision in range(base_precision, 20):
        levels = np.asarray([_round_frac(b, precision) for b in bins])
        if np.unique(levels).size == bins.size:
            return precision
    return base_precision


def _qcut_bins(values: np.ndarray, q: int) -> list[tuple[str, np.ndarray]]:
    """Equal-frequency bins matching ``pd.qcut(values, q, duplicates='drop')``.

    Returns ``(interval_label, membership_mask)`` per bin, reproducing pandas'
    right-closed / include-lowest assignment and Interval label formatting.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return []
    raw = np.unique(np.percentile(finite, np.linspace(0, 1, q + 1) * 100.0))
    if raw.size < 2:
        return []
    ids = raw.searchsorted(arr, side="left")
    ids[arr == raw[0]] = 1  # include_lowest
    precision = _infer_precision(3, raw)
    breaks = [_round_frac(b, precision) for b in raw]
    breaks[0] = breaks[0] - 10 ** (-precision)
    return [
        (f"({breaks[i]}, {breaks[i + 1]}]", ids == (i + 1)) for i in range(raw.size - 1)
    ]


def _summarize_bins(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.height == 0:
        return []
    timestamp = _fcol(frame, "timestamp")
    loss = _fcol(frame, "loss_ratio")
    keep = (~np.isnan(timestamp)) & (~np.isnan(loss))
    timestamp = timestamp[keep]
    loss = loss[keep]
    if timestamp.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for label, mask in _qcut_bins(timestamp, 6):
        group = loss[mask]
        rows.append(
            {
                "time_bin": label,
                "count": int(mask.sum()),
                "mean_loss_ratio": float(np.mean(group))
                if group.size
                else float("nan"),
                "median_loss_ratio": float(np.median(group))
                if group.size
                else float("nan"),
            }
        )
    return rows


def _median_loss(df: pl.DataFrame) -> float | None:
    if df.height == 0:
        return None
    loss = _fcol(df, "loss_ratio")
    finite = loss[~np.isnan(loss)]
    return float(np.median(finite)) if finite.size else float("nan")


def _drift_report(datasets: StrengthDatasets) -> dict[str, Any]:
    trusted = datasets.dedup_trusted
    promotable = datasets.dedup_promotable
    search_all = _search_frame(datasets.all_runs)
    search_promotable = _search_frame(promotable)

    return {
        "all_search_bins": _summarize_bins(search_all),
        "promotable_search_bins": _summarize_bins(search_promotable),
        "models": {
            "all_search_time_only": _fit_time_model(
                search_all, include_weights=False, include_arch=False
            ),
            "all_search_plus_weights": _fit_time_model(
                search_all, include_weights=True, include_arch=False
            ),
            "all_search_plus_weights_and_arch": _fit_time_model(
                search_all, include_weights=True, include_arch=True
            ),
            "promotable_time_only": _fit_time_model(
                search_promotable, include_weights=False, include_arch=False
            ),
            "promotable_plus_weights": _fit_time_model(
                search_promotable, include_weights=True, include_arch=False
            ),
            "promotable_plus_weights_and_arch": _fit_time_model(
                search_promotable, include_weights=True, include_arch=True
            ),
        },
        "distribution_checks": {
            "trusted_loss_ratio_median": _median_loss(trusted),
            "promotable_loss_ratio_median": _median_loss(promotable),
            "search_all_loss_ratio_median": _median_loss(search_all),
            "search_promotable_loss_ratio_median": _median_loss(search_promotable),
        },
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if int(mask.sum()) < 2:
        return float("nan")
    x = a[mask]
    y = b[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _weight_bias_summary(datasets: StrengthDatasets) -> dict[str, Any]:
    search = datasets.all_runs.filter(pl.col("result_cohort") == "search")
    if search.height == 0:
        return {"top_weighted_categories": [], "category_weight_vs_loss": []}
    weight_columns = sorted(
        column for column in search.columns if column.startswith("category_weights::")
    )
    loss = _fcol(search, "loss_ratio")
    stage1 = _fcol(search, "stage1_passed")
    correlations = []
    mean_weights: dict[str, float] = {}
    for column in weight_columns:
        series = _fcol(search, column)
        coverage = int((~np.isnan(series)).sum())
        if coverage >= 5:
            mean_weights[column.split("::", 1)[1]] = float(np.nanmean(series))
        if coverage < 20:
            continue
        correlations.append(
            {
                "category": column.split("::", 1)[1],
                "corr_with_loss_ratio": _corr(series, loss),
                "corr_with_stage1_passed": _corr(series, stage1),
                "mean_weight": float(np.nanmean(series)),
            }
        )
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


def _coverage(df: pl.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int((~np.isnan(_fcol(df, column))).sum())


def _support_summary(df: pl.DataFrame) -> dict[str, Any]:
    unique_graphs = 0
    if df.height and "graph_fingerprint" in df.columns:
        unique_graphs = df.get_column("graph_fingerprint").drop_nulls().n_unique()
    return {
        "runs": int(df.height),
        "unique_graphs": int(unique_graphs),
        "loss_ratio_coverage": _coverage(df, "loss_ratio"),
        "induction_coverage": _coverage(df, "induction_screening_auc"),
        "binding_coverage": _coverage(df, "binding_screening_auc"),
        "hellaswag_coverage": _coverage(df, "hellaswag_acc"),
        "wikitext_coverage": _coverage(df, "wikitext_perplexity"),
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
