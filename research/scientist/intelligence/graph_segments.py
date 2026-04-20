from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from research.synthesis.primitives import canonicalize_op_name as _canonicalize_op_name
from research.scientist.intelligence.metrics_utils import (
    binary_classification_metrics,
    operating_point_profiles,
    safe_binary_roc_auc,
)
from research.scientist.intelligence.ml_corpus import (
    _graph_fingerprint,
    build_dense_feature_matrix,
)
from research.synthesis.context_rules import find_byte_safety_violations
from research.synthesis.graph_features import (
    _build_adjacency,
    enrich_with_op_stats,
    extract_graph_features_bundle,
    load_op_stats,
)
from research.synthesis.serializer import graph_from_json
from research.scientist.native.core import _try_import_rust_scheduler

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphSegmentExtraction:
    presence_set: frozenset[str]
    count_map: Dict[str, int]


@dataclass(frozen=True, slots=True)
class SegmentCorpusRow:
    canonical_fingerprint: str
    graph_json: str
    n_rows: int
    latest_timestamp: float
    stage1_any_passed: bool
    stage1_pass_rate: float
    loss_ratio_best: float | None
    wikitext_perplexity_best: float | None
    binding_auc: float | None
    induction_auc: float | None
    hellaswag_acc: float | None
    binding_positive: bool
    induction_positive: bool
    hellaswag_positive: bool
    all_three_positive: bool


@dataclass(frozen=True, slots=True)
class SegmentAssociation:
    fragment_id: str
    path_len: int
    support_graphs: int
    support_total_count: int
    present_rate: float
    absent_rate: float
    rate_lift: float
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_low: float
    posterior_high: float


def _loads_graph_json(graph_json: Any) -> Dict[str, Any]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return graph_json if isinstance(graph_json, dict) else {}


def _non_input_op_nodes(nodes: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for node_id, node in nodes.items():
        op_name = _canonicalize_op_name(str(node.get("op_name", "")))
        if op_name and op_name != "input":
            out[str(node_id)] = op_name
    return out


def _extract_graph_segments_native(
    graph_json: Any,
    *,
    min_len: int,
    max_len: int,
) -> GraphSegmentExtraction | None:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "extract_graph_segments_native"):
        return None
    if isinstance(graph_json, str):
        payload = graph_json
    else:
        try:
            payload = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
    try:
        raw = rust.extract_graph_segments_native(payload, int(min_len), int(max_len))
    except Exception:
        return None
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    count_map = {
        str(key): int(value) for key, value in loaded.items() if isinstance(key, str)
    }
    return GraphSegmentExtraction(frozenset(count_map.keys()), count_map)


def extract_graph_segments(
    graph_json: Any,
    *,
    min_len: int = 3,
    max_len: int = 6,
) -> GraphSegmentExtraction:
    native = _extract_graph_segments_native(
        graph_json,
        min_len=min_len,
        max_len=max_len,
    )
    if native is not None:
        return native

    graph_dict = _loads_graph_json(graph_json)
    nodes = graph_dict.get("nodes") or {}
    if not isinstance(nodes, dict) or not nodes:
        return GraphSegmentExtraction(frozenset(), {})

    fwd, _rev = _build_adjacency(nodes)
    op_nodes = _non_input_op_nodes(nodes)
    if not op_nodes:
        return GraphSegmentExtraction(frozenset(), {})

    counts: Counter[str] = Counter()

    def _visit(path_nodes: List[str]) -> None:
        op_path = [op_nodes[node_id] for node_id in path_nodes]
        path_len = len(op_path)
        if path_len >= min_len:
            fragment_id = f"seg_p{path_len}:{' > '.join(op_path)}".replace(" > ", ">")
            counts[fragment_id] += 1
        if path_len >= max_len:
            return
        last = path_nodes[-1]
        for child in sorted(fwd.get(last, [])):
            child_id = str(child)
            if child_id not in op_nodes:
                continue
            if child_id in path_nodes:
                continue
            path_nodes.append(child_id)
            _visit(path_nodes)
            path_nodes.pop()

    for node_id in sorted(op_nodes):
        _visit([node_id])

    return GraphSegmentExtraction(frozenset(counts.keys()), dict(counts))


def _is_native_safe_graph(graph_json: str) -> bool:
    payload = str(graph_json or "").strip()
    if not payload or payload == "{}":
        return False
    try:
        graph = graph_from_json(payload)
    except Exception:
        return False
    return not find_byte_safety_violations(graph)


def _min_opt(current: float | None, candidate: Any) -> float | None:
    if candidate is None:
        return current
    value = float(candidate)
    if not math.isfinite(value):
        return current
    if current is None:
        return value
    return value if value < current else current


def _max_opt(current: float | None, candidate: Any) -> float | None:
    if candidate is None:
        return current
    value = float(candidate)
    if not math.isfinite(value):
        return current
    if current is None:
        return value
    return value if value > current else current


def load_stage05_native_segment_corpus(
    db_path: str | Path,
) -> List[SegmentCorpusRow]:
    db_path = str(Path(db_path))
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    rows = conn.execute(
        """
        SELECT graph_json, graph_fingerprint, stage0_passed, stage05_passed, stage1_passed,
               loss_ratio, wikitext_perplexity, binding_auc, induction_auc, hellaswag_acc,
               timestamp
        FROM program_results
        WHERE TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{}'
          AND COALESCE(stage0_passed, 0) = 1
          AND COALESCE(stage05_passed, 0) = 1
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = str(row["graph_json"])
        if not _is_native_safe_graph(graph_json):
            continue
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            {
                "canonical_fingerprint": canonical,
                "graph_json": graph_json,
                "n_rows": 0,
                "latest_timestamp": 0.0,
                "stage1_any_passed": False,
                "_n_stage1_passed": 0,
                "loss_ratio_best": None,
                "wikitext_perplexity_best": None,
                "binding_auc": None,
                "induction_auc": None,
                "hellaswag_acc": None,
                "_best_rank": None,
            },
        )
        stage1 = bool(row["stage1_passed"])
        timestamp = float(row["timestamp"] or 0.0)
        group["n_rows"] += 1
        group["_n_stage1_passed"] += int(stage1)
        group["stage1_any_passed"] = bool(group["stage1_any_passed"] or stage1)
        group["latest_timestamp"] = max(float(group["latest_timestamp"]), timestamp)
        group["loss_ratio_best"] = _min_opt(group["loss_ratio_best"], row["loss_ratio"])
        group["wikitext_perplexity_best"] = _min_opt(
            group["wikitext_perplexity_best"], row["wikitext_perplexity"]
        )
        group["binding_auc"] = _max_opt(group["binding_auc"], row["binding_auc"])
        group["induction_auc"] = _max_opt(group["induction_auc"], row["induction_auc"])
        group["hellaswag_acc"] = _max_opt(group["hellaswag_acc"], row["hellaswag_acc"])

        rank = (
            0 if stage1 else 1,
            row["loss_ratio"] is None,
            float(row["loss_ratio"]) if row["loss_ratio"] is not None else float("inf"),
            -timestamp,
        )
        if group["_best_rank"] is None or rank < group["_best_rank"]:
            group["_best_rank"] = rank
            group["graph_json"] = graph_json

    out: List[SegmentCorpusRow] = []
    for group in grouped.values():
        n_rows = int(group["n_rows"])
        binding_auc = group["binding_auc"]
        induction_auc = group["induction_auc"]
        hellaswag_acc = group["hellaswag_acc"]
        binding_positive = bool(binding_auc is not None and binding_auc > 0.0)
        induction_positive = bool(induction_auc is not None and induction_auc > 0.0)
        hellaswag_positive = bool(hellaswag_acc is not None and hellaswag_acc > 0.0)
        out.append(
            SegmentCorpusRow(
                canonical_fingerprint=str(group["canonical_fingerprint"]),
                graph_json=str(group["graph_json"]),
                n_rows=n_rows,
                latest_timestamp=float(group["latest_timestamp"]),
                stage1_any_passed=bool(group["stage1_any_passed"]),
                stage1_pass_rate=float(group["_n_stage1_passed"]) / max(n_rows, 1),
                loss_ratio_best=group["loss_ratio_best"],
                wikitext_perplexity_best=group["wikitext_perplexity_best"],
                binding_auc=binding_auc,
                induction_auc=induction_auc,
                hellaswag_acc=hellaswag_acc,
                binding_positive=binding_positive,
                induction_positive=induction_positive,
                hellaswag_positive=hellaswag_positive,
                all_three_positive=(
                    binding_positive and induction_positive and hellaswag_positive
                ),
            )
        )
    out.sort(key=lambda row: row.canonical_fingerprint)
    return out


def _build_fragment_feature_names(
    extractions: Sequence[GraphSegmentExtraction],
    min_support: int,
) -> List[str]:
    support: Counter[str] = Counter()
    for extraction in extractions:
        support.update(extraction.presence_set)
    return sorted(
        fragment_id
        for fragment_id, n_graphs in support.items()
        if n_graphs >= min_support
    )


def build_feature_matrices(
    rows: Sequence[SegmentCorpusRow],
    *,
    min_support: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[GraphSegmentExtraction]]:
    extractions = [extract_graph_segments(row.graph_json) for row in rows]
    fragment_names = _build_fragment_feature_names(extractions, min_support)

    op_stats_cache = load_op_stats()
    baseline_feature_dicts: List[Dict[str, float]] = []
    for row in rows:
        graph_dict = _loads_graph_json(row.graph_json)
        feats, ops = extract_graph_features_bundle(graph_dict)
        for op in ops:
            if op:
                feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
        enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
        baseline_feature_dicts.append(feats)

    X_baseline, baseline_names = build_dense_feature_matrix(baseline_feature_dicts)
    X_frag, fragment_names = build_dense_feature_matrix(
        [extraction.count_map for extraction in extractions],
        feature_names=fragment_names,
    )

    return X_baseline, X_frag, baseline_names, fragment_names, extractions


def summarize_binary_fragment_associations(
    rows: Sequence[SegmentCorpusRow],
    extractions: Sequence[GraphSegmentExtraction],
    *,
    target_name: str,
    min_support: int = 20,
    prior_strength: float = 20.0,
) -> List[SegmentAssociation]:
    y = np.array(
        [1 if getattr(row, target_name) else 0 for row in rows], dtype=np.int32
    )
    if y.size == 0:
        return []
    base_rate = float(np.mean(y))
    alpha0 = max(base_rate * prior_strength, 1e-6)
    beta0 = max((1.0 - base_rate) * prior_strength, 1e-6)

    support_graphs: Counter[str] = Counter()
    support_total_count: Counter[str] = Counter()
    positives_when_present: Counter[str] = Counter()
    for row, extraction in zip(rows, extractions):
        label = 1 if getattr(row, target_name) else 0
        for fragment_id in extraction.presence_set:
            support_graphs[fragment_id] += 1
            positives_when_present[fragment_id] += label
        for fragment_id, count in extraction.count_map.items():
            support_total_count[fragment_id] += int(count)

    out: List[SegmentAssociation] = []
    n_total = len(rows)
    total_positive = int(np.sum(y))
    for fragment_id, present_n in support_graphs.items():
        if present_n < min_support:
            continue
        pos_present = positives_when_present[fragment_id]
        absent_n = n_total - present_n
        pos_absent = total_positive - pos_present
        alpha = alpha0 + pos_present
        beta = beta0 + (present_n - pos_present)
        posterior_mean = alpha / (alpha + beta)
        posterior_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        posterior_std = math.sqrt(max(posterior_var, 0.0))
        present_rate = float(pos_present / present_n)
        absent_rate = float(pos_absent / absent_n) if absent_n > 0 else base_rate
        out.append(
            SegmentAssociation(
                fragment_id=fragment_id,
                path_len=int(fragment_id.split(":", 1)[0].replace("seg_p", "")),
                support_graphs=present_n,
                support_total_count=int(support_total_count.get(fragment_id, 0)),
                present_rate=present_rate,
                absent_rate=absent_rate,
                rate_lift=present_rate - absent_rate,
                posterior_alpha=float(alpha),
                posterior_beta=float(beta),
                posterior_mean=float(posterior_mean),
                posterior_low=max(0.0, float(posterior_mean - 1.96 * posterior_std)),
                posterior_high=min(1.0, float(posterior_mean + 1.96 * posterior_std)),
            )
        )
    out.sort(
        key=lambda item: (
            item.posterior_mean - base_rate,
            item.support_graphs,
            item.support_total_count,
        ),
        reverse=True,
    )
    return out


def _stratified_split_indices(
    y: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = np.asarray(y, dtype=np.int32)
    if labels.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    unique = np.unique(labels)
    if unique.size < 2:
        idx = np.arange(labels.size, dtype=np.int32)
        split = int(round(labels.size * train_fraction))
        return idx[:split], idx[split:]

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for label in unique:
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        split = max(1, int(round(len(idx) * train_fraction)))
        split = min(split, max(len(idx) - 1, 1))
        train_parts.append(idx[:split])
        val_parts.append(idx[split:])
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int32)
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int32)
    return train_idx, val_idx


def _evaluate_binary_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    train_idx, val_idx = _stratified_split_indices(y, seed=seed)
    if len(train_idx) < 5 or len(val_idx) < 2 or np.unique(y[train_idx]).size < 2:
        return {"error": "insufficient_data"}
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        y_score = model.predict_proba(X[val_idx])[:, 1]
    except Exception as exc:
        return {"error": f"logistic_fit_failed: {exc}"}
    operating_points = operating_point_profiles(y[val_idx], y_score)
    selected = operating_points["f1"]
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "roc_auc": float(safe_binary_roc_auc(y[val_idx], y_score)),
        "selected_metrics": selected,
        "threshold_0_5_metrics": binary_classification_metrics(
            y[val_idx], y_score, 0.5
        ),
    }


def _evaluate_regression_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    mask = np.isfinite(y)
    if int(np.sum(mask)) < 20:
        return {"error": "insufficient_data"}
    Xf = X[mask]
    yf = y[mask]
    pseudo_labels = (yf >= np.median(yf)).astype(np.int32)
    train_idx, val_idx = _stratified_split_indices(pseudo_labels, seed=seed)
    if len(train_idx) < 10 or len(val_idx) < 5:
        return {"error": "insufficient_split"}
    try:
        from sklearn.linear_model import Ridge
        from scipy.stats import spearmanr

        model = Ridge(alpha=1.0)
        model.fit(Xf[train_idx], yf[train_idx])
        pred = model.predict(Xf[val_idx])
        mae = float(np.mean(np.abs(pred - yf[val_idx])))
        rho, _ = spearmanr(yf[val_idx], pred)
        return {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "mae": mae,
            "spearman": float(rho) if np.isfinite(rho) else 0.0,
        }
    except Exception as exc:
        return {"error": f"ridge_fit_failed: {exc}"}


def evaluate_feature_families(
    rows: Sequence[SegmentCorpusRow],
    *,
    min_support: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    X_baseline, X_frag, baseline_names, fragment_names, _extractions = (
        build_feature_matrices(rows, min_support=min_support)
    )
    X_hybrid = (
        np.concatenate([X_baseline, X_frag], axis=1)
        if X_frag.shape[1]
        else X_baseline.copy()
    )

    families = {
        "baseline": X_baseline,
        "fragment_only": X_frag,
        "hybrid": X_hybrid,
    }
    binary_targets = {
        "stage1_any_passed": np.array(
            [int(row.stage1_any_passed) for row in rows], dtype=np.int32
        ),
        "binding_positive": np.array(
            [int(row.binding_positive) for row in rows], dtype=np.int32
        ),
        "induction_positive": np.array(
            [int(row.induction_positive) for row in rows], dtype=np.int32
        ),
        "hellaswag_positive": np.array(
            [int(row.hellaswag_positive) for row in rows], dtype=np.int32
        ),
        "all_three_positive": np.array(
            [int(row.all_three_positive) for row in rows], dtype=np.int32
        ),
    }
    continuous_targets = {
        "loss_ratio_best": np.array(
            [
                float(row.loss_ratio_best)
                if row.loss_ratio_best is not None
                else np.nan
                for row in rows
            ],
            dtype=np.float64,
        ),
        "wikitext_perplexity_best": np.array(
            [
                float(row.wikitext_perplexity_best)
                if row.wikitext_perplexity_best is not None
                else np.nan
                for row in rows
            ],
            dtype=np.float64,
        ),
        "binding_auc": np.array(
            [
                float(row.binding_auc) if row.binding_auc is not None else np.nan
                for row in rows
            ],
            dtype=np.float64,
        ),
        "induction_auc": np.array(
            [
                float(row.induction_auc) if row.induction_auc is not None else np.nan
                for row in rows
            ],
            dtype=np.float64,
        ),
        "hellaswag_acc": np.array(
            [
                float(row.hellaswag_acc) if row.hellaswag_acc is not None else np.nan
                for row in rows
            ],
            dtype=np.float64,
        ),
    }

    report: Dict[str, Any] = {
        "n_graphs": int(len(rows)),
        "n_baseline_features": int(len(baseline_names)),
        "n_fragment_features": int(len(fragment_names)),
        "binary": {},
        "continuous": {},
    }
    for target_name, y in binary_targets.items():
        report["binary"][target_name] = {
            family_name: _evaluate_binary_model(X, y, seed=seed)
            for family_name, X in families.items()
            if X.shape[1] > 0
        }
    for target_name, y in continuous_targets.items():
        report["continuous"][target_name] = {
            family_name: _evaluate_regression_model(X, y, seed=seed)
            for family_name, X in families.items()
            if X.shape[1] > 0
        }
    return report
