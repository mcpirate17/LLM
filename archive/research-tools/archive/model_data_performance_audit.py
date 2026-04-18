#!/usr/bin/env python3
"""End-to-end data + model performance audit.

Reasonable assumptions:
- Training logs contain one row per step/epoch/update and expose at least a step-like
  column plus train/eval loss columns, either as wide columns or metric/value rows.
- Evaluation outputs contain one row per evaluated sample with targets plus either
  predictions, probabilities, logits, scores, or enough columns to derive them.
- Dataset samples contain train/eval rows or at least one table with targets and
  raw inputs or engineered features. If `sample_id` exists, eval rows are joined to
  dataset rows on that key; otherwise the audit falls back to aligned row order.
- Input files can be CSV, JSON, JSONL/NDJSON, or Parquet.

Outputs:
- Console summary
- JSON report
- PNG plots: loss curves, calibration plot, error distribution, data distribution
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    top_k_accuracy_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
RANDOM_STATE = 42
MAX_NEAR_DUP_SAMPLES = 4000
MAX_TEXT_FEATURES = 3000
MAX_CLUSTER_ROWS = 5000
CALIBRATION_BINS = 10
DRIFT_BUCKETS = 10

TARGET_ALIASES = (
    "target",
    "label",
    "y",
    "ground_truth",
    "truth",
    "expected",
)
PREDICTION_ALIASES = (
    "prediction",
    "pred",
    "predicted",
    "output",
    "class_pred",
)
CONFIDENCE_ALIASES = (
    "confidence",
    "conf",
    "max_prob",
    "pred_confidence",
)
LOSS_ALIASES = ("loss", "nll", "cross_entropy", "ce_loss")
STEP_ALIASES = ("step", "global_step", "iteration", "iter", "update", "epoch")
TIME_ALIASES = ("timestamp", "time", "datetime", "date", "created_at", "ts")
SPLIT_ALIASES = ("split", "partition", "subset")
ID_ALIASES = ("sample_id", "id", "row_id", "example_id", "uid")
TEXT_ALIASES = (
    "text",
    "input",
    "prompt",
    "content",
    "sequence",
    "source_text",
    "input_text",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="Training logs file")
    parser.add_argument(
        "--eval",
        dest="eval_path",
        type=Path,
        required=True,
        help="Evaluation outputs file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset samples file",
    )
    parser.add_argument(
        "--components",
        type=Path,
        default=None,
        help="Optional component-level outputs file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON report and plots",
    )
    parser.add_argument(
        "--error-clusters",
        type=int,
        default=4,
        help="Number of error clusters for failed predictions",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k to report when class probabilities/logits are available",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for near-duplicate text samples",
    )
    return parser.parse_args()


def _canonical_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, orient="records", lines=True)
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return pd.read_json(path)
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"])
            for value in data.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return pd.DataFrame(value)
            return pd.json_normalize(data)
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError(f"Unsupported JSON payload in {path}")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_canonical_name(col) for col in out.columns]
    return out


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    alias_set = {_canonical_name(alias) for alias in aliases}
    for column in columns:
        if column in alias_set:
            return column
    for alias in alias_set:
        for column in columns:
            if column.endswith(alias) or alias in column:
                return column
    return None


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    wanted = {_canonical_name(x) for x in candidates}
    return [col for col in columns if col in wanted]


def _maybe_parse_json_cell(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _is_prob_vector_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    sample = series.dropna().head(5).map(_maybe_parse_json_cell)
    if sample.empty:
        return False
    return all(isinstance(x, (list, dict)) for x in sample)


def _prepare_logs(logs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    df = _normalize_columns(logs)
    if df.empty:
        raise ValueError("Training log table is empty")

    metric_col = _find_column(df.columns, ("metric", "name", "key"))
    value_col = _find_column(df.columns, ("value", "metric_value"))
    if metric_col and value_col:
        step_col = _find_column(df.columns, STEP_ALIASES) or df.columns[0]
        pivot = (
            df.assign(metric_name=df[metric_col].astype(str).map(_canonical_name))
            .pivot_table(
                index=step_col,
                columns="metric_name",
                values=value_col,
                aggfunc="last",
            )
            .reset_index()
        )
        df = _normalize_columns(pivot)

    step_col = _find_column(df.columns, STEP_ALIASES)
    if step_col is None:
        step_col = "step"
        df.insert(0, step_col, np.arange(len(df), dtype=np.int64))
        warnings_out.append("Logs missing step column; generated sequential steps.")

    train_loss_col = None
    eval_loss_col = None
    for col in df.columns:
        if "train" in col and "loss" in col:
            train_loss_col = col
        if ("eval" in col or "val" in col or "validation" in col) and "loss" in col:
            eval_loss_col = col
    if train_loss_col is None:
        train_loss_col = _find_column(df.columns, LOSS_ALIASES)
        if train_loss_col:
            warnings_out.append(
                f"Using `{train_loss_col}` as train loss because no explicit train loss was found."
            )

    if train_loss_col is None or eval_loss_col is None:
        long_loss_cols = [c for c in df.columns if "loss" in c]
        if len(long_loss_cols) >= 2:
            train_loss_col = train_loss_col or long_loss_cols[0]
            eval_loss_col = eval_loss_col or long_loss_cols[1]

    if train_loss_col is None:
        raise ValueError("Could not identify training loss column in logs")
    if eval_loss_col is None:
        warnings_out.append(
            "Eval loss column not found; some overfit diagnostics will be partial."
        )

    grad_col = _find_column(
        df.columns, ("grad_norm", "gradient_norm", "grad", "grad_l2")
    )
    lr_col = _find_column(df.columns, ("lr", "learning_rate"))

    out = pd.DataFrame(
        {
            "step": pd.to_numeric(df[step_col], errors="coerce"),
            "train_loss": pd.to_numeric(df[train_loss_col], errors="coerce"),
        }
    )
    if eval_loss_col is not None:
        out["eval_loss"] = pd.to_numeric(df[eval_loss_col], errors="coerce")
    if grad_col is not None:
        out["grad_norm"] = pd.to_numeric(df[grad_col], errors="coerce")
    if lr_col is not None:
        out["lr"] = pd.to_numeric(df[lr_col], errors="coerce")
    out = (
        out.dropna(subset=["step", "train_loss"])
        .sort_values("step")
        .reset_index(drop=True)
    )
    return out, warnings_out


def _extract_probability_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray | None, list[str] | None]:
    vector_col = _find_column(
        df.columns, ("probabilities", "probs", "logits", "scores")
    )
    if vector_col and _is_prob_vector_series(df[vector_col]):
        parsed = df[vector_col].map(_maybe_parse_json_cell)
        if isinstance(parsed.dropna().iloc[0], dict):
            keys = sorted({str(k) for row in parsed.dropna() for k in row.keys()})
            matrix = np.vstack(
                [
                    np.array(
                        [float(row.get(key, np.nan)) for key in keys], dtype=np.float64
                    )
                    if isinstance(row, dict)
                    else np.full(len(keys), np.nan, dtype=np.float64)
                    for row in parsed
                ]
            )
            return matrix, keys
        rows = []
        width = max(len(row) if isinstance(row, list) else 0 for row in parsed.dropna())
        for row in parsed:
            if isinstance(row, list):
                arr = np.asarray(row, dtype=np.float64)
                if len(arr) < width:
                    arr = np.pad(arr, (0, width - len(arr)), constant_values=np.nan)
                rows.append(arr)
            else:
                rows.append(np.full(width, np.nan, dtype=np.float64))
        return np.vstack(rows), [str(i) for i in range(width)]

    prob_cols = []
    for col in df.columns:
        if re.match(r"^(prob|score|logit|p)_[a-z0-9_.-]+$", col):
            prob_cols.append(col)
    if len(prob_cols) >= 2:
        prob_cols = sorted(prob_cols)
        matrix = (
            df[prob_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        class_names = [re.sub(r"^(prob|score|logit|p)_", "", col) for col in prob_cols]
        return matrix, class_names
    return None, None


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.nansum(exp, axis=1, keepdims=True)
    return exp / np.clip(denom, EPS, None)


def _safe_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _prepare_eval(
    eval_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    warnings_out: list[str] = []
    df = _normalize_columns(eval_df)
    if df.empty:
        raise ValueError("Evaluation table is empty")

    target_col = _find_column(df.columns, TARGET_ALIASES)
    if target_col is None:
        raise ValueError("Could not identify target column in evaluation outputs")

    pred_col = _find_column(df.columns, PREDICTION_ALIASES)
    conf_col = _find_column(df.columns, CONFIDENCE_ALIASES)
    id_col = _find_column(df.columns, ID_ALIASES)
    split_col = _find_column(df.columns, SPLIT_ALIASES)
    time_col = _find_column(df.columns, TIME_ALIASES)
    batch_col = _find_column(
        df.columns, ("batch", "batch_id", "chunk", "window", "shard")
    )

    prob_matrix, class_names = _extract_probability_matrix(df)
    y_true_raw = df[target_col]

    if prob_matrix is not None:
        valid_rows = np.isfinite(prob_matrix).all(axis=1)
        prob_matrix = prob_matrix.astype(np.float64, copy=False)
        if np.nanmax(prob_matrix) > 1.0 or np.nanmin(prob_matrix) < 0.0:
            prob_matrix = _safe_softmax(prob_matrix)
        else:
            row_sums = np.nansum(prob_matrix, axis=1, keepdims=True)
            prob_matrix = prob_matrix / np.clip(row_sums, EPS, None)
    else:
        valid_rows = np.ones(len(df), dtype=bool)

    if pred_col is not None:
        y_pred_raw = df[pred_col].astype(str)
    elif prob_matrix is not None and class_names is not None:
        y_pred_raw = pd.Series(
            np.array(class_names, dtype=object)[np.nanargmax(prob_matrix, axis=1)]
        )
    else:
        raise ValueError(
            "Could not identify prediction or probability columns in evaluation outputs"
        )

    y_true_raw = y_true_raw.astype(str)
    label_values = sorted(
        set(y_true_raw.dropna().tolist()) | set(y_pred_raw.dropna().tolist())
    )
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}
    y_true = y_true_raw.map(label_to_idx).to_numpy(dtype=np.int64)
    y_pred = y_pred_raw.map(label_to_idx).fillna(-1).to_numpy(dtype=np.int64)

    if prob_matrix is not None and class_names is not None:
        reorder = []
        class_lookup = {
            _canonical_name(name): idx for idx, name in enumerate(class_names)
        }
        aligned = True
        for label in label_values:
            idx = class_lookup.get(_canonical_name(label))
            if idx is None:
                aligned = False
                break
            reorder.append(idx)
        if aligned:
            prob_matrix = prob_matrix[:, reorder]
        else:
            warnings_out.append(
                "Probability columns do not align with label names; top-k/log-loss use raw class order."
            )
    elif prob_matrix is None:
        if conf_col is not None:
            confidence = (
                pd.to_numeric(df[conf_col], errors="coerce")
                .fillna(0.5)
                .to_numpy(dtype=np.float64)
            )
        else:
            confidence = np.where(y_pred == y_true, 0.75, 0.55).astype(np.float64)
            warnings_out.append(
                "Confidence column missing; confidence-based diagnostics fall back to heuristic values."
            )
        n_classes = max(len(label_values), 2)
        prob_matrix = np.full((len(df), n_classes), np.nan, dtype=np.float64)
        incorrect_prob = (1.0 - confidence) / max(n_classes - 1, 1)
        prob_matrix[:] = incorrect_prob[:, None]
        rows = np.arange(len(df))
        valid_pred_rows = y_pred >= 0
        prob_matrix[rows[valid_pred_rows], y_pred[valid_pred_rows]] = confidence[
            valid_pred_rows
        ]

    confidence = np.nanmax(prob_matrix, axis=1)
    correct = (y_true == y_pred).astype(np.int8)
    sample_loss = -np.log(np.clip(prob_matrix[np.arange(len(df)), y_true], EPS, 1.0))

    out = pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=np.int64),
            "target": y_true,
            "target_label": y_true_raw.to_numpy(dtype=object),
            "prediction": y_pred,
            "prediction_label": y_pred_raw.to_numpy(dtype=object),
            "confidence": confidence,
            "correct": correct,
            "sample_loss": sample_loss,
        }
    )
    if id_col is not None:
        out["sample_id"] = df[id_col]
    if split_col is not None:
        out["split"] = df[split_col].astype(str).str.lower()
    if batch_col is not None:
        out["batch"] = pd.to_numeric(df[batch_col], errors="coerce")
    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        out["time"] = parsed_time

    metadata = {
        "label_values": label_values,
        "probability_matrix": prob_matrix,
        "n_classes": len(label_values),
        "topk_requested": None,
        "valid_probability_rows": int(valid_rows.sum()),
    }
    return out, metadata, warnings_out


def _prepare_dataset(dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    df = _normalize_columns(dataset_df)
    if df.empty:
        raise ValueError("Dataset table is empty")

    id_col = _find_column(df.columns, ID_ALIASES)
    target_col = _find_column(df.columns, TARGET_ALIASES)
    split_col = _find_column(df.columns, SPLIT_ALIASES)
    time_col = _find_column(df.columns, TIME_ALIASES)
    text_col = _find_column(df.columns, TEXT_ALIASES)

    out = df.copy()
    if id_col is None:
        out["sample_id"] = np.arange(len(out), dtype=np.int64)
        warnings_out.append(
            "Dataset missing sample identifier; generated sequential sample_id values."
        )
    elif id_col != "sample_id":
        out = out.rename(columns={id_col: "sample_id"})
    if split_col and split_col != "split":
        out = out.rename(columns={split_col: "split"})
    if target_col and target_col != "target_raw":
        out = out.rename(columns={target_col: "target_raw"})
    if time_col and time_col != "time":
        out = out.rename(columns={time_col: "time"})
        out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
    if text_col and text_col != "input_text":
        out = out.rename(columns={text_col: "input_text"})
    elif text_col is None:
        warnings_out.append(
            "Dataset text/input column not found; token and redundancy analysis use non-text features only."
        )

    return out, warnings_out


def _join_eval_dataset(
    eval_frame: pd.DataFrame, dataset_frame: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    if "sample_id" in eval_frame.columns and "sample_id" in dataset_frame.columns:
        merged = eval_frame.merge(
            dataset_frame,
            on="sample_id",
            how="left",
            suffixes=("", "_data"),
        )
        coverage = float(merged.notna().any(axis=1).mean())
        if coverage < 0.8:
            warnings_out.append(
                f"Dataset join coverage is low ({coverage:.1%}); consider aligning sample_id semantics."
            )
        return merged, warnings_out

    merged = eval_frame.copy()
    data_subset = dataset_frame.reset_index(drop=True).copy()
    merged = pd.concat([merged.reset_index(drop=True), data_subset], axis=1)
    warnings_out.append(
        "Joined evaluation rows to dataset rows by order because sample_id was unavailable."
    )
    return merged, warnings_out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if window <= 1:
        return values.copy()
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)


def _series_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True, dropna=False).to_numpy(dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log2(np.clip(probs, EPS, 1.0))).sum())


def _normalized_text_entropy(text: str) -> float:
    tokens = re.findall(r"\w+", str(text).lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    probs = np.array(list(counts.values()), dtype=np.float64) / len(tokens)
    entropy = -(probs * np.log2(np.clip(probs, EPS, 1.0))).sum()
    norm = math.log2(max(len(counts), 2))
    return float(entropy / max(norm, EPS))


def _quantile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    if values.nunique(dropna=True) < len(labels):
        return pd.Series(
            np.repeat(labels[min(1, len(labels) - 1)], len(values)), index=values.index
        )
    q = pd.qcut(values.rank(method="first"), q=len(labels), labels=labels)
    return q.astype(str)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if pd.isna(value):
        return None
    return value


def _loss_trends(logs: pd.DataFrame) -> dict[str, Any]:
    train = logs["train_loss"].to_numpy(dtype=np.float64)
    steps = logs["step"].to_numpy(dtype=np.float64)
    eval_values = (
        logs["eval_loss"].to_numpy(dtype=np.float64)
        if "eval_loss" in logs.columns
        else None
    )

    train_diff = np.diff(train)
    early_window = max(2, int(len(train) * 0.2))
    late_window = max(2, int(len(train) * 0.2))
    early_speed = float(
        (train[0] - train[min(len(train) - 1, early_window - 1)])
        / max(early_window - 1, 1)
    )
    late_speed = float(
        (train[max(0, len(train) - late_window)] - train[-1]) / max(late_window - 1, 1)
    )

    instability_threshold = np.nanmedian(np.abs(train_diff)) + 2.5 * np.nanstd(
        train_diff
    )
    instability_events = int(
        np.sum(np.abs(train_diff) > max(instability_threshold, EPS))
    )
    plateau_window = max(5, len(train) // 12)
    rolling_improvement = -_rolling_mean(
        np.diff(train, prepend=train[0]), plateau_window
    )
    plateau_step = None
    plateau_mask = rolling_improvement < max(np.nanstd(train_diff) * 0.05, 1e-4)
    plateau_indices = np.flatnonzero(plateau_mask)
    if len(plateau_indices) > 0:
        plateau_step = float(steps[int(plateau_indices[0])])

    result = {
        "train_start": float(train[0]),
        "train_end": float(train[-1]),
        "train_best": float(np.nanmin(train)),
        "train_reduction_pct": float((train[0] - train[-1]) / max(abs(train[0]), EPS)),
        "convergence_speed_early": early_speed,
        "convergence_speed_late": late_speed,
        "instability_events": instability_events,
        "instability_ratio": float(instability_events / max(len(train_diff), 1)),
        "plateau_step": plateau_step,
    }
    if eval_values is not None:
        result.update(
            {
                "eval_start": float(eval_values[0]),
                "eval_end": float(eval_values[-1]),
                "eval_best": float(np.nanmin(eval_values)),
                "generalization_gap_end": float(eval_values[-1] - train[-1]),
                "generalization_gap_best_eval": float(
                    np.nanmin(eval_values) - train[np.nanargmin(eval_values)]
                ),
            }
        )
    return result


def _calibration_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray
) -> dict[str, Any]:
    confidence = np.max(prob, axis=1)
    correctness = (y_true == y_pred).astype(np.float64)
    bins = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, CALIBRATION_BINS - 1)
    ece = 0.0
    per_bin = []
    for idx in range(CALIBRATION_BINS):
        mask = bin_idx == idx
        if not np.any(mask):
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correctness[mask].mean())
        frac = float(mask.mean())
        ece += abs(avg_conf - avg_acc) * frac
        per_bin.append(
            {
                "bin": idx,
                "count": int(mask.sum()),
                "avg_confidence": avg_conf,
                "accuracy": avg_acc,
            }
        )
    return {
        "expected_calibration_error": float(ece),
        "average_confidence": float(confidence.mean()),
        "accuracy": float(correctness.mean()),
        "overconfidence_gap": float(confidence.mean() - correctness.mean()),
        "underconfidence_gap": float(correctness.mean() - confidence.mean()),
        "per_bin": per_bin,
    }


def _difficulty_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["difficulty"] = _quantile_labels(
        work["sample_loss"], ["easy", "medium", "hard"]
    )
    grouped = (
        work.groupby("difficulty", dropna=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_loss=("sample_loss", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("difficulty")
    )
    return {
        "segments": grouped.to_dict(orient="records"),
        "hard_fraction": float((work["difficulty"] == "hard").mean()),
    }


def _drift_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    axis_name = None
    if "time" in work.columns and work["time"].notna().sum() >= max(10, DRIFT_BUCKETS):
        work = work.sort_values("time")
        work["drift_bucket"] = pd.qcut(
            work["time"].rank(method="first"),
            q=min(DRIFT_BUCKETS, work["time"].notna().sum()),
            duplicates="drop",
        )
        axis_name = "time"
    elif "batch" in work.columns and work["batch"].notna().sum() >= max(
        10, DRIFT_BUCKETS
    ):
        work = work.sort_values("batch")
        work["drift_bucket"] = pd.qcut(
            work["batch"].rank(method="first"),
            q=min(DRIFT_BUCKETS, work["batch"].notna().sum()),
            duplicates="drop",
        )
        axis_name = "batch"
    else:
        work = work.sort_values("row_index")
        work["drift_bucket"] = pd.qcut(
            work["row_index"].rank(method="first"),
            q=min(DRIFT_BUCKETS, len(work)),
            duplicates="drop",
        )
        axis_name = "eval_order"

    by_bucket = (
        work.groupby("drift_bucket", dropna=False, observed=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_loss=("sample_loss", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    acc_values = by_bucket["accuracy"].to_numpy(dtype=np.float64)
    loss_values = by_bucket["mean_loss"].to_numpy(dtype=np.float64)
    return {
        "axis": axis_name,
        "bucket_metrics": [
            {
                "bucket": str(row["drift_bucket"]),
                "n": int(row["n"]),
                "accuracy": float(row["accuracy"]),
                "mean_loss": float(row["mean_loss"]),
                "mean_confidence": float(row["mean_confidence"]),
            }
            for _, row in by_bucket.iterrows()
        ],
        "accuracy_range": float(np.nanmax(acc_values) - np.nanmin(acc_values)),
        "loss_range": float(np.nanmax(loss_values) - np.nanmin(loss_values)),
    }


def _error_features(
    frame: pd.DataFrame, prob_matrix: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    features = [
        frame["confidence"].to_numpy(dtype=np.float64),
        frame["sample_loss"].to_numpy(dtype=np.float64),
        (frame["prediction"] == frame["target"]).astype(np.float64).to_numpy(),
        np.max(prob_matrix, axis=1),
        _safe_entropy(prob_matrix) / max(math.log(max(prob_matrix.shape[1], 2)), EPS),
    ]
    names = [
        "confidence",
        "sample_loss",
        "correct_flag",
        "max_probability",
        "normalized_prediction_entropy",
    ]
    if "input_text" in frame.columns:
        token_lengths = (
            frame["input_text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
            .to_numpy(dtype=np.float64)
        )
        features.append(token_lengths)
        names.append("token_length")
    return np.column_stack(features), names


def _error_clustering(
    frame: pd.DataFrame,
    prob_matrix: np.ndarray,
    n_clusters: int,
) -> list[dict[str, Any]]:
    errors = frame.loc[frame["correct"] == 0].copy()
    if errors.empty:
        return []
    if len(errors) == 1:
        row = errors.iloc[0]
        return [
            {
                "cluster_id": 0,
                "n": 1,
                "mean_confidence": float(row["confidence"]),
                "mean_loss": float(row["sample_loss"]),
                "top_target_labels": [str(row["target_label"])],
                "top_prediction_labels": [str(row["prediction_label"])],
            }
        ]

    error_prob = prob_matrix[errors.index.to_numpy()]
    X, feature_names = _error_features(errors, error_prob)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(errors) > MAX_CLUSTER_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        keep = np.sort(rng.choice(len(errors), size=MAX_CLUSTER_ROWS, replace=False))
        errors = errors.iloc[keep].copy()
        X = X[keep]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clusters = min(max(2, n_clusters), len(errors))
    model = KMeans(n_clusters=clusters, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(Xs)
    errors["cluster_id"] = labels

    results = []
    for cluster_id, group in errors.groupby("cluster_id"):
        center = model.cluster_centers_[int(cluster_id)]
        center_raw = scaler.inverse_transform([center])[0]
        results.append(
            {
                "cluster_id": int(cluster_id),
                "n": int(len(group)),
                "mean_confidence": float(group["confidence"].mean()),
                "mean_loss": float(group["sample_loss"].mean()),
                "top_target_labels": group["target_label"]
                .astype(str)
                .value_counts()
                .head(3)
                .index.tolist(),
                "top_prediction_labels": group["prediction_label"]
                .astype(str)
                .value_counts()
                .head(3)
                .index.tolist(),
                "feature_center": {
                    feature_names[i]: float(center_raw[i])
                    for i in range(len(feature_names))
                },
            }
        )
    return sorted(results, key=lambda x: x["n"], reverse=True)


def _repeated_failure_patterns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    errors = frame.loc[frame["correct"] == 0].copy()
    if errors.empty:
        return []
    errors["confidence_band"] = pd.cut(
        errors["confidence"],
        bins=[0.0, 0.5, 0.7, 0.85, 1.0],
        include_lowest=True,
    ).astype(str)
    if "input_text" in errors.columns:
        token_len = errors["input_text"].fillna("").astype(str).str.split().str.len()
        errors["length_band"] = pd.cut(
            token_len,
            bins=[-1, 16, 64, 256, np.inf],
            labels=["short", "medium", "long", "very_long"],
        ).astype(str)
    else:
        errors["length_band"] = "unknown"
    grouped = (
        errors.groupby(
            ["target_label", "prediction_label", "confidence_band", "length_band"],
            dropna=False,
            observed=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(10)
    )
    return grouped.to_dict(orient="records")


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    reserved = {
        "sample_id",
        "split",
        "target_raw",
        "input_text",
        "time",
        "target",
        "prediction",
        "prediction_label",
        "target_label",
        "row_index",
        "confidence",
        "sample_loss",
        "correct",
        "batch",
    }
    numeric_cols = [
        col
        for col in dataset.columns
        if col not in reserved and pd.api.types.is_numeric_dtype(dataset[col])
    ]
    categorical_cols = [
        col
        for col in dataset.columns
        if col not in reserved
        and (
            dataset[col].dtype == "object"
            or str(dataset[col].dtype).startswith("string")
        )
    ]
    return numeric_cols, categorical_cols


def _data_distribution(
    dataset: pd.DataFrame, audit_frame: pd.DataFrame
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_rows": int(len(dataset)),
    }
    if "target_raw" in dataset.columns:
        target_counts = dataset["target_raw"].astype(str).value_counts(dropna=False)
        probs = target_counts.to_numpy(dtype=np.float64) / max(target_counts.sum(), 1)
        imbalance_ratio = float(target_counts.max() / max(target_counts.min(), 1))
        result["target_distribution"] = target_counts.to_dict()
        result["target_entropy_bits"] = float(
            -(probs * np.log2(np.clip(probs, EPS, 1.0))).sum()
        )
        result["imbalance_ratio"] = imbalance_ratio

    if "input_text" in dataset.columns:
        text = dataset["input_text"].fillna("").astype(str)
        token_lengths = text.str.split().str.len()
        sample_entropy = text.map(_normalized_text_entropy)
        result["token_length"] = {
            "mean": float(token_lengths.mean()),
            "median": float(token_lengths.median()),
            "p95": float(token_lengths.quantile(0.95)),
        }
        result["sample_entropy"] = {
            "mean": float(sample_entropy.mean()),
            "low_entropy_fraction": float(
                (sample_entropy < sample_entropy.quantile(0.1)).mean()
            ),
            "high_entropy_fraction": float(
                (sample_entropy > sample_entropy.quantile(0.9)).mean()
            ),
        }

        all_tokens = re.findall(r"\w+", " ".join(text.head(20000).str.lower().tolist()))
        token_counter = Counter(all_tokens)
        result["top_tokens"] = token_counter.most_common(20)

    numeric_cols, categorical_cols = _feature_columns(dataset)
    if numeric_cols:
        describe = (
            dataset[numeric_cols].describe(percentiles=[0.1, 0.5, 0.9]).transpose()
        )
        result["numeric_feature_summary"] = describe.round(6).to_dict(orient="index")
    if categorical_cols:
        result["categorical_cardinality"] = {
            col: int(dataset[col].nunique(dropna=False))
            for col in categorical_cols[:20]
        }

    if "correct" in audit_frame.columns:
        result["eval_coverage"] = int(len(audit_frame))
    return result


def _redundancy_analysis(
    dataset: pd.DataFrame, similarity_threshold: float
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "examples": [],
    }
    if "input_text" in dataset.columns:
        text = dataset["input_text"].fillna("").astype(str)
        exact_dup_mask = text.duplicated(keep=False) & text.ne("")
        result["exact_duplicates"] = int(exact_dup_mask.sum())
        dup_examples = text[exact_dup_mask].drop_duplicates().head(5).tolist()
        result["examples"].extend(dup_examples)

        unique_text = text[text.ne("")].drop_duplicates().reset_index(drop=True)
        if len(unique_text) >= 5:
            sample = unique_text.head(MAX_NEAR_DUP_SAMPLES)
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=MAX_TEXT_FEATURES,
            )
            X = vectorizer.fit_transform(sample)
            if X.shape[0] >= 2 and X.shape[1] > 0:
                nn = NearestNeighbors(metric="cosine", n_neighbors=2)
                nn.fit(X)
                distances, indices = nn.kneighbors(X)
                similarities = 1.0 - distances[:, 1]
                near_mask = similarities >= similarity_threshold
                result["near_duplicates"] = int(near_mask.sum())
                for row_idx in np.flatnonzero(near_mask)[:5]:
                    other_idx = int(indices[row_idx, 1])
                    result["examples"].append(
                        {
                            "similarity": float(similarities[row_idx]),
                            "sample_a": sample.iloc[int(row_idx)][:200],
                            "sample_b": sample.iloc[other_idx][:200],
                        }
                    )
    else:
        numeric_cols, _ = _feature_columns(dataset)
        if numeric_cols:
            rounded = dataset[numeric_cols].round(6).astype(str).agg("|".join, axis=1)
            exact_dup_mask = rounded.duplicated(keep=False)
            result["exact_duplicates"] = int(exact_dup_mask.sum())
    return result


def _signal_strength(
    dataset: pd.DataFrame, audit_frame: pd.DataFrame
) -> dict[str, Any]:
    result: dict[str, Any] = {"feature_importance": [], "weak_signal_regions": []}
    if "target_raw" not in dataset.columns:
        return result

    numeric_cols, categorical_cols = _feature_columns(dataset)
    if not numeric_cols and "input_text" not in dataset.columns:
        return result

    work = dataset.copy()
    target_series = work["target_raw"].astype(str)
    classification = target_series.nunique(dropna=True) <= 30
    if classification:
        classes = {
            label: idx
            for idx, label in enumerate(
                sorted(target_series.dropna().unique().tolist())
            )
        }
        y = target_series.map(classes).to_numpy(dtype=np.int64)
    else:
        y = (
            pd.to_numeric(work["target_raw"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    feature_scores: list[dict[str, Any]] = []
    if numeric_cols:
        X_num = (
            work[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(work[numeric_cols].median(numeric_only=True))
            .to_numpy(dtype=np.float64)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if classification:
                f_scores, p_values = f_classif(X_num, y)
                mi_scores = mutual_info_classif(X_num, y, random_state=RANDOM_STATE)
            else:
                f_scores, p_values = f_regression(X_num, y)
                mi_scores = mutual_info_regression(X_num, y, random_state=RANDOM_STATE)
        for idx, col in enumerate(numeric_cols):
            feature_scores.append(
                {
                    "feature": col,
                    "f_score": float(np.nan_to_num(f_scores[idx])),
                    "p_value": float(np.nan_to_num(p_values[idx], nan=1.0)),
                    "mutual_information": float(np.nan_to_num(mi_scores[idx])),
                }
            )

    if "input_text" in work.columns:
        token_lengths = (
            work["input_text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
            .to_numpy(dtype=np.float64)
        )
        if len(token_lengths) == len(y):
            if classification:
                corr = (
                    np.corrcoef(token_lengths, y)[0, 1]
                    if np.std(token_lengths) > 0 and np.std(y) > 0
                    else 0.0
                )
            else:
                corr = (
                    np.corrcoef(token_lengths, y)[0, 1]
                    if np.std(token_lengths) > 0 and np.std(y) > 0
                    else 0.0
                )
            feature_scores.append(
                {
                    "feature": "token_length",
                    "correlation": float(np.nan_to_num(corr)),
                    "mean": float(token_lengths.mean()),
                }
            )

    feature_scores = sorted(
        feature_scores,
        key=lambda x: abs(
            float(
                x.get("mutual_information", x.get("f_score", x.get("correlation", 0.0)))
            )
        ),
        reverse=True,
    )
    result["feature_importance"] = feature_scores[:20]

    if "sample_loss" in audit_frame.columns:
        joined = audit_frame.copy()
        for col in numeric_cols[:10]:
            series = pd.to_numeric(joined.get(col), errors="coerce")
            if (
                series is None
                or series.notna().sum() < 20
                or series.nunique(dropna=True) < 4
            ):
                continue
            buckets = pd.qcut(
                series.rank(method="first"),
                q=min(5, series.notna().sum()),
                duplicates="drop",
            )
            grouped = (
                joined.assign(_bucket=buckets)
                .groupby("_bucket", dropna=False, observed=False)
                .agg(
                    n=("correct", "size"),
                    accuracy=("correct", "mean"),
                    mean_loss=("sample_loss", "mean"),
                )
                .reset_index()
            )
            if grouped.empty:
                continue
            weak = grouped.sort_values(
                ["accuracy", "mean_loss"], ascending=[True, False]
            ).iloc[0]
            if weak["n"] >= 10:
                result["weak_signal_regions"].append(
                    {
                        "feature": col,
                        "bucket": str(weak["_bucket"]),
                        "n": int(weak["n"]),
                        "accuracy": float(weak["accuracy"]),
                        "mean_loss": float(weak["mean_loss"]),
                    }
                )
    result["weak_signal_regions"] = sorted(
        result["weak_signal_regions"], key=lambda x: (x["accuracy"], -x["mean_loss"])
    )[:10]
    return result


def _temporal_trends(
    dataset: pd.DataFrame, audit_frame: pd.DataFrame
) -> dict[str, Any]:
    if "time" not in dataset.columns and "time" not in audit_frame.columns:
        return {"available": False}

    time_source = audit_frame if "time" in audit_frame.columns else dataset
    work = time_source.loc[time_source["time"].notna()].copy()
    if len(work) < 10:
        return {"available": False}

    work = work.sort_values("time")
    work["bucket"] = pd.qcut(
        work["time"].rank(method="first"),
        q=min(DRIFT_BUCKETS, len(work)),
        duplicates="drop",
    )
    grouped = work.groupby("bucket", dropna=False, observed=False)
    output = {
        "available": True,
        "bucket_summary": [],
    }
    for bucket, group in grouped:
        row = {
            "bucket": str(bucket),
            "n": int(len(group)),
        }
        if "target_raw" in group.columns:
            row["target_entropy_bits"] = _series_entropy(
                group["target_raw"].astype(str)
            )
        if "correct" in group.columns:
            row["accuracy"] = float(group["correct"].mean())
            row["mean_loss"] = float(group["sample_loss"].mean())
        if "input_text" in group.columns:
            token_lengths = (
                group["input_text"].fillna("").astype(str).str.split().str.len()
            )
            row["mean_token_length"] = float(token_lengths.mean())
        output["bucket_summary"].append(row)
    return output


def _learning_effectiveness(logs: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    train = logs["train_loss"].to_numpy(dtype=np.float64)
    loss_delta = np.diff(train)
    result["mean_improvement_per_step"] = (
        float((-loss_delta).mean()) if len(loss_delta) else 0.0
    )
    result["median_improvement_per_step"] = (
        float(np.median(-loss_delta)) if len(loss_delta) else 0.0
    )
    result["positive_update_fraction"] = (
        float((loss_delta < 0).mean()) if len(loss_delta) else 0.0
    )

    if "grad_norm" in logs.columns and logs["grad_norm"].notna().sum() >= 3:
        grad = logs["grad_norm"].to_numpy(dtype=np.float64)
        future_gain = -np.diff(train, append=train[-1])
        corr = (
            np.corrcoef(grad, future_gain)[0, 1]
            if np.std(grad) > 0 and np.std(future_gain) > 0
            else 0.0
        )
        meaningful_updates = (grad > np.nanmedian(grad)) & (future_gain > 0)
        result["gradient_loss_correlation"] = float(np.nan_to_num(corr))
        result["meaningful_update_fraction"] = float(meaningful_updates.mean())
    else:
        result["gradient_loss_correlation"] = None
        result["meaningful_update_fraction"] = None

    plateau_window = max(5, len(train) // 10)
    rolling_gain = _rolling_mean(-np.diff(train, prepend=train[0]), plateau_window)
    result["plateau_fraction"] = float(
        (rolling_gain < max(np.nanstd(loss_delta) * 0.05, 1e-4)).mean()
    )
    if "eval_loss" in logs.columns:
        train_end = float(train[-1])
        eval_end = float(logs["eval_loss"].iloc[-1])
        gap = eval_end - train_end
        result["underfit_signal"] = bool(
            train_end > np.nanmedian(train) * 0.8 and gap < 0.05 * max(eval_end, EPS)
        )
        result["overfit_signal"] = bool(gap > max(0.1, 0.1 * abs(eval_end)))
    else:
        result["underfit_signal"] = None
        result["overfit_signal"] = None
    return result


def _blind_spots(frame: pd.DataFrame) -> list[dict[str, Any]]:
    work = frame.copy()
    blind_spots: list[dict[str, Any]] = []

    grouped = (
        work.groupby(["target_label", "prediction_label"], dropna=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_loss=("sample_loss", "mean"),
        )
        .reset_index()
        .sort_values(
            ["accuracy", "mean_confidence", "mean_loss"], ascending=[True, False, False]
        )
    )
    for _, row in grouped.head(10).iterrows():
        if row["target_label"] == row["prediction_label"]:
            continue
        blind_spots.append(
            {
                "segment_type": "confusion_pair",
                "target_label": str(row["target_label"]),
                "prediction_label": str(row["prediction_label"]),
                "n": int(row["n"]),
                "accuracy": float(row["accuracy"]),
                "mean_confidence": float(row["mean_confidence"]),
                "mean_loss": float(row["mean_loss"]),
            }
        )

    if "input_text" in work.columns:
        token_len = work["input_text"].fillna("").astype(str).str.split().str.len()
        work["length_bucket"] = pd.cut(
            token_len,
            bins=[-1, 16, 64, 256, np.inf],
            labels=["short", "medium", "long", "very_long"],
        )
        grouped_len = (
            work.groupby("length_bucket", dropna=False, observed=False)
            .agg(
                n=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
                mean_loss=("sample_loss", "mean"),
            )
            .reset_index()
            .sort_values(["accuracy", "mean_loss"], ascending=[True, False])
        )
        for _, row in grouped_len.head(4).iterrows():
            if row["n"] >= 10:
                blind_spots.append(
                    {
                        "segment_type": "length_bucket",
                        "bucket": str(row["length_bucket"]),
                        "n": int(row["n"]),
                        "accuracy": float(row["accuracy"]),
                        "mean_confidence": float(row["mean_confidence"]),
                        "mean_loss": float(row["mean_loss"]),
                    }
                )
    return blind_spots[:12]


def _data_dead_zones(
    signal: dict[str, Any], drift: dict[str, Any]
) -> list[dict[str, Any]]:
    dead_zones = []
    for region in signal.get("weak_signal_regions", []):
        if region["accuracy"] < 0.5:
            dead_zones.append(
                {
                    "type": "weak_signal_region",
                    "feature": region["feature"],
                    "bucket": region["bucket"],
                    "n": region["n"],
                    "accuracy": region["accuracy"],
                    "mean_loss": region["mean_loss"],
                }
            )
    for bucket in drift.get("bucket_metrics", []):
        if bucket["accuracy"] < 0.5 and bucket["mean_loss"] > 1.0:
            dead_zones.append(
                {
                    "type": "drift_bucket",
                    "bucket": bucket["bucket"],
                    "n": bucket["n"],
                    "accuracy": bucket["accuracy"],
                    "mean_loss": bucket["mean_loss"],
                }
            )
    return dead_zones[:10]


def _component_analysis(component_path: Path | None) -> dict[str, Any] | None:
    if component_path is None:
        return None
    frame = _normalize_columns(_load_table(component_path))
    if frame.empty:
        return {"available": False}
    numeric_cols = [
        col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])
    ]
    summary = {}
    if numeric_cols:
        summary["numeric_summary"] = (
            frame[numeric_cols].describe().round(6).to_dict(orient="index")
        )
    if "component" in frame.columns:
        summary["component_counts"] = (
            frame["component"].astype(str).value_counts().head(20).to_dict()
        )
    summary["available"] = True
    return summary


def _performance_metrics(
    frame: pd.DataFrame,
    prob_matrix: np.ndarray,
    label_values: list[str],
    top_k: int,
) -> dict[str, Any]:
    y_true = frame["target"].to_numpy(dtype=np.int64)
    y_pred = frame["prediction"].to_numpy(dtype=np.int64)
    n_classes = len(label_values)
    result = {
        "n_samples": int(len(frame)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(
            log_loss(
                y_true, np.clip(prob_matrix, EPS, 1.0), labels=np.arange(n_classes)
            )
        ),
        "brier_score": float(
            np.mean(np.sum((prob_matrix - np.eye(n_classes)[y_true]) ** 2, axis=1))
        ),
        "mean_sample_loss": float(frame["sample_loss"].mean()),
        "calibration": _calibration_metrics(y_true, y_pred, prob_matrix),
    }
    effective_top_k = min(top_k, max(n_classes - 1, 1))
    if n_classes > 2 and effective_top_k >= 2:
        result["top_k_accuracy"] = float(
            top_k_accuracy_score(
                y_true,
                prob_matrix,
                k=effective_top_k,
                labels=np.arange(n_classes),
            )
        )
        result["top_k"] = int(effective_top_k)
    else:
        positive_index = 1 if n_classes > 1 else 0
        result["rmse_probability"] = float(
            math.sqrt(
                mean_squared_error(
                    (y_true == positive_index).astype(np.float64),
                    prob_matrix[:, positive_index],
                )
            )
        )
    return result


def _plot_loss_curves(logs: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(logs["step"], logs["train_loss"], label="train_loss")
    if "eval_loss" in logs.columns:
        ax.plot(logs["step"], logs["eval_loss"], label="eval_loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.2)
    path = output_dir / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_calibration(
    frame: pd.DataFrame, prob_matrix: np.ndarray, output_dir: Path
) -> Path:
    y_true = frame["target"].to_numpy(dtype=np.int64)
    y_pred = frame["prediction"].to_numpy(dtype=np.int64)
    confidence = np.max(prob_matrix, axis=1)
    correctness = (y_true == y_pred).astype(np.int64)
    frac_pos, mean_pred = calibration_curve(
        correctness, confidence, n_bins=CALIBRATION_BINS, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Calibration")
    ax.grid(True, alpha=0.2)
    path = output_dir / "calibration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_error_distribution(frame: pd.DataFrame, output_dir: Path) -> Path:
    errors = frame.loc[frame["correct"] == 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(errors["confidence"], bins=20, color="tab:red", alpha=0.8)
    axes[0].set_title("Error Confidence")
    axes[0].set_xlabel("confidence")
    axes[0].set_ylabel("count")

    confusion = (
        errors.groupby(["target_label", "prediction_label"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    labels = [f"{t}->{p}" for t, p in confusion.index]
    axes[1].barh(labels, confusion.to_numpy(), color="tab:orange")
    axes[1].set_title("Top Error Pairs")
    axes[1].set_xlabel("count")
    axes[1].invert_yaxis()
    path = output_dir / "error_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_data_distribution(dataset: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if "input_text" in dataset.columns:
        token_lengths = (
            dataset["input_text"].fillna("").astype(str).str.split().str.len()
        )
        axes[0].hist(token_lengths, bins=30, color="tab:blue", alpha=0.8)
        axes[0].set_title("Token Length")
        axes[0].set_xlabel("tokens")
        axes[0].set_ylabel("count")
    else:
        numeric_cols, _ = _feature_columns(dataset)
        if numeric_cols:
            first = numeric_cols[0]
            axes[0].hist(
                pd.to_numeric(dataset[first], errors="coerce").dropna(),
                bins=30,
                color="tab:blue",
                alpha=0.8,
            )
            axes[0].set_title(f"Feature Distribution: {first}")
            axes[0].set_xlabel(first)
            axes[0].set_ylabel("count")
        else:
            axes[0].text(
                0.5, 0.5, "No feature distribution available", ha="center", va="center"
            )
            axes[0].set_axis_off()

    if "target_raw" in dataset.columns:
        counts = dataset["target_raw"].astype(str).value_counts().head(20)
        axes[1].bar(counts.index.astype(str), counts.to_numpy(), color="tab:green")
        axes[1].set_title("Target Distribution")
        axes[1].set_ylabel("count")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].text(
            0.5, 0.5, "No target distribution available", ha="center", va="center"
        )
        axes[1].set_axis_off()

    path = output_dir / "data_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _console_summary(report: dict[str, Any]) -> str:
    perf = report["performance_metrics"]
    learn = report["learning_effectiveness"]
    drift = report["drift_analysis"]
    dist = report["data_distribution"]
    blind_spots = report["advanced_diagnostics"]["blind_spots"]

    lines = [
        f"samples={perf['n_samples']} classes={perf['n_classes']} accuracy={perf['accuracy']:.4f} log_loss={perf['log_loss']:.4f}",
        f"calibration_ece={perf['calibration']['expected_calibration_error']:.4f} overconfidence_gap={perf['calibration']['overconfidence_gap']:.4f}",
        f"mean_improvement_per_step={learn['mean_improvement_per_step']:.6f} positive_update_fraction={learn['positive_update_fraction']:.4f}",
        f"drift_axis={drift['axis']} accuracy_range={drift['accuracy_range']:.4f} loss_range={drift['loss_range']:.4f}",
    ]
    if "imbalance_ratio" in dist:
        lines.append(
            f"target_entropy_bits={dist.get('target_entropy_bits', 0.0):.4f} imbalance_ratio={dist['imbalance_ratio']:.4f}"
        )
    lines.append(
        f"blind_spots={len(blind_spots)} error_clusters={len(report['error_clusters'])}"
    )
    if report.get("warnings"):
        lines.append(f"warnings={len(report['warnings'])}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logs_raw = _load_table(args.logs)
    eval_raw = _load_table(args.eval_path)
    dataset_raw = _load_table(args.dataset)

    logs, warnings_logs = _prepare_logs(logs_raw)
    eval_frame, eval_meta, warnings_eval = _prepare_eval(eval_raw)
    dataset, warnings_dataset = _prepare_dataset(dataset_raw)
    audit_frame, warnings_join = _join_eval_dataset(eval_frame, dataset)

    loss_report = _loss_trends(logs)
    perf_report = _performance_metrics(
        audit_frame,
        eval_meta["probability_matrix"],
        eval_meta["label_values"],
        args.top_k,
    )
    difficulty = _difficulty_analysis(audit_frame)
    drift = _drift_analysis(audit_frame)
    errors = {
        "repeated_failure_patterns": _repeated_failure_patterns(audit_frame),
    }
    error_clusters = _error_clustering(
        audit_frame,
        eval_meta["probability_matrix"],
        args.error_clusters,
    )
    distribution = _data_distribution(dataset, audit_frame)
    redundancy = _redundancy_analysis(dataset, args.near_dup_threshold)
    signal = _signal_strength(dataset, audit_frame)
    temporal = _temporal_trends(dataset, audit_frame)
    learning = _learning_effectiveness(logs)
    blind_spots = _blind_spots(audit_frame)
    dead_zones = _data_dead_zones(signal, drift)
    component_analysis = _component_analysis(args.components)

    figures = {
        "loss_curves": str(_plot_loss_curves(logs, args.output_dir)),
        "calibration": str(
            _plot_calibration(
                audit_frame, eval_meta["probability_matrix"], args.output_dir
            )
        ),
        "error_distribution": str(
            _plot_error_distribution(audit_frame, args.output_dir)
        ),
        "data_distribution": str(_plot_data_distribution(dataset, args.output_dir)),
    }

    report = {
        "assumptions": [
            "Logs expose step-like progress plus train/eval loss columns or metric/value rows.",
            "Evaluation rows contain targets plus predictions, probabilities, logits, or scores.",
            "Dataset rows include either sample_id for joining or row-order alignment is acceptable.",
        ],
        "inputs": {
            "logs": str(args.logs),
            "eval": str(args.eval_path),
            "dataset": str(args.dataset),
            "components": str(args.components) if args.components else None,
        },
        "performance_metrics": {
            **perf_report,
            "loss_trends": loss_report,
            "difficulty_analysis": difficulty,
            "error_distribution": errors,
        },
        "error_clusters": error_clusters,
        "data_distribution": {
            **distribution,
            "redundancy": redundancy,
            "signal_strength": signal,
            "temporal_trends": temporal,
        },
        "drift_analysis": drift,
        "learning_effectiveness": learning,
        "advanced_diagnostics": {
            "blind_spots": blind_spots,
            "data_dead_zones": dead_zones,
            "component_analysis": component_analysis,
        },
        "artifacts": figures,
        "warnings": warnings_logs + warnings_eval + warnings_dataset + warnings_join,
    }

    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    print(_console_summary(report))
    print(f"json_report={report_path}")
    for name, path in figures.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
