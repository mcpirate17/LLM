from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..native.core import _try_import_rust_scheduler
from ..trust_policy import (
    PROMOTABLE_COMPARABILITY_LABELS,
    PROMOTABLE_TRUST_LABELS,
    TRUSTED_COMPARABILITY_LABELS,
    TRUSTED_TRUST_LABELS,
)
from ..notebook.graph_artifacts import resolve_graph_json_value

logger = logging.getLogger(__name__)


class CorpusIntegrityError(ValueError):
    """Raised when the deduped ML corpus violates invariants."""


@dataclass(slots=True)
class _CorpusCacheEntry:
    rows: List[Dict[str, Any]]
    validated: bool = False


_CORPUS_CACHE_LOCK = Lock()
_CORPUS_CACHE: Dict[
    Tuple[str, str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    _CorpusCacheEntry,
] = {}
_CORPUS_CACHE_STATS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "validations": 0,
    "clears": 0,
}


def load_deduped_graph_training_rows(
    db_path: str | Path, *, validate: bool = False
) -> List[Dict[str, Any]]:
    path = str(Path(db_path))
    return _load_cached_corpus_rows(
        cache_kind="graph_training",
        db_path=path,
        builder=lambda: _build_graph_training_rows(path),
        validator=validate_graph_training_rows if validate else None,
    )


def load_deduped_screening_predictor_rows(
    db_path: str | Path, *, validate: bool = False
) -> List[Dict[str, Any]]:
    path = str(Path(db_path))
    return _load_cached_corpus_rows(
        cache_kind="screening_predictor",
        db_path=path,
        builder=lambda: _build_screening_predictor_rows(path),
        validator=validate_graph_training_rows if validate else None,
    )


def load_deduped_graph_analysis_rows(db_path: str | Path) -> List[Dict[str, Any]]:
    path = str(Path(db_path))
    return _load_cached_corpus_rows(
        cache_kind="graph_analysis",
        db_path=path,
        builder=lambda: _fallback_graph_analysis_rows(path),
        validator=None,
    )


def load_deduped_predictor_training_rows(
    db_path: str | Path, *, validate: bool = False
) -> List[Dict[str, Any]]:
    path = str(Path(db_path))
    return _load_cached_corpus_rows(
        cache_kind="predictor_training",
        db_path=path,
        builder=lambda: _build_predictor_training_rows(path),
        validator=validate_predictor_training_rows if validate else None,
    )


def load_screening_predictor_corpus_rows(
    db_path: str | Path, *, validate: bool = False
) -> List[Dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return load_deduped_graph_training_rows(path, validate=validate)
    cols = _program_results_columns(str(path))
    required = {
        "result_cohort",
        "data_provenance_json",
        "trust_label",
        "comparability_label",
    }
    if not required.issubset(cols):
        return load_deduped_graph_training_rows(path, validate=validate)
    return load_deduped_screening_predictor_rows(path, validate=validate)


def rerun_confidence_weight(n_rows: int) -> float:
    return min(math.sqrt(max(int(n_rows or 1), 1)), 3.0)


def _clear_corpus_cache() -> None:
    with _CORPUS_CACHE_LOCK:
        _CORPUS_CACHE.clear()
        _CORPUS_CACHE_STATS["clears"] += 1


def get_corpus_cache_stats() -> Dict[str, int]:
    with _CORPUS_CACHE_LOCK:
        return dict(_CORPUS_CACHE_STATS)


def grouped_stratified_split(
    signatures: List[str], labels: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Split rows by exact-graph signature to avoid duplicate leakage."""
    groups: Dict[str, List[int]] = {}
    for idx, sig in enumerate(signatures):
        groups.setdefault(sig, []).append(idx)

    rng = np.random.RandomState(seed)
    pos_groups: List[str] = []
    neg_groups: List[str] = []
    ambiguous_groups = 0
    for sig, idxs in groups.items():
        rate = float(np.mean(labels[idxs]))
        if 0.0 < rate < 1.0:
            ambiguous_groups += 1
        if rate >= 0.5:
            pos_groups.append(sig)
        else:
            neg_groups.append(sig)

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)
    pos_split = int(len(pos_groups) * 0.8)
    neg_split = int(len(neg_groups) * 0.8)
    train_groups = set(pos_groups[:pos_split]) | set(neg_groups[:neg_split])
    val_groups = set(pos_groups[pos_split:]) | set(neg_groups[neg_split:])

    train_idx = np.fromiter(
        (idx for sig, idxs in groups.items() if sig in train_groups for idx in idxs),
        dtype=np.int32,
    )
    val_idx = np.fromiter(
        (idx for sig, idxs in groups.items() if sig in val_groups for idx in idxs),
        dtype=np.int32,
    )
    stats = {
        "n_unique_graphs": len(groups),
        "n_duplicate_groups": int(sum(1 for idxs in groups.values() if len(idxs) > 1)),
        "n_ambiguous_duplicate_groups": int(ambiguous_groups),
    }
    return train_idx, val_idx, stats


def grouped_temporal_split(
    signatures: List[str],
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | float | str]]:
    """Chronological grouped split using each signature group's latest timestamp."""
    groups: Dict[str, List[int]] = {}
    for idx, sig in enumerate(signatures):
        groups.setdefault(sig, []).append(idx)

    if not groups:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            {"split_strategy": "temporal_grouped", "error": "no_groups"},
        )

    group_order: List[Tuple[float, str, float]] = []
    ambiguous_groups = 0
    for sig, idxs in groups.items():
        rate = float(np.mean(labels[idxs]))
        if 0.0 < rate < 1.0:
            ambiguous_groups += 1
        latest_ts = float(np.max(np.asarray(timestamps[idxs], dtype=np.float64)))
        group_order.append((latest_ts, sig, rate))

    group_order.sort(key=lambda item: (item[0], item[1]))
    n_groups = len(group_order)
    split_at = int(np.floor(n_groups * float(train_fraction)))
    split_at = max(1, min(n_groups - 1, split_at))
    train_order = list(group_order[:split_at])
    val_order = list(group_order[split_at:])

    def _ensure_class_presence(
        train_part: List[Tuple[float, str, float]],
        val_part: List[Tuple[float, str, float]],
        *,
        want_positive: bool,
    ) -> None:
        predicate = (
            (lambda rate: rate >= 0.5) if want_positive else (lambda rate: rate < 0.5)
        )
        if any(predicate(rate) for _, _, rate in val_part):
            return
        for idx in range(len(train_part) - 1, -1, -1):
            ts, sig, rate = train_part[idx]
            if predicate(rate):
                val_part.insert(0, train_part.pop(idx))
                return

    _ensure_class_presence(train_order, val_order, want_positive=True)
    _ensure_class_presence(train_order, val_order, want_positive=False)

    if not train_order:
        train_order.append(val_order.pop(0))
    if not val_order:
        val_order.append(train_order.pop())

    train_groups = {sig for _, sig, _ in train_order}
    val_groups = {sig for _, sig, _ in val_order}
    train_idx = np.fromiter(
        (idx for sig, idxs in groups.items() if sig in train_groups for idx in idxs),
        dtype=np.int32,
    )
    val_idx = np.fromiter(
        (idx for sig, idxs in groups.items() if sig in val_groups for idx in idxs),
        dtype=np.int32,
    )
    stats: Dict[str, int | float | str] = {
        "split_strategy": "temporal_grouped",
        "n_unique_graphs": len(groups),
        "n_duplicate_groups": int(sum(1 for idxs in groups.values() if len(idxs) > 1)),
        "n_ambiguous_duplicate_groups": int(ambiguous_groups),
        "n_train_groups": int(len(train_groups)),
        "n_val_groups": int(len(val_groups)),
        "temporal_train_fraction": float(train_fraction),
        "temporal_cutoff_timestamp": float(train_order[-1][0]),
        "temporal_val_start_timestamp": float(val_order[0][0]),
    }
    return train_idx, val_idx, stats


def build_dense_feature_matrix(
    feat_dicts: List[Dict[str, float]],
    *,
    feature_names: List[str] | None = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, List[str]]:
    """Materialize sparse feature dicts into a dense matrix in one pass."""
    if not feat_dicts:
        return np.zeros((0, 0), dtype=dtype), []
    if feature_names is None:
        feature_names = sorted({key for feats in feat_dicts for key in feats.keys()})
    col_idx = {name: idx for idx, name in enumerate(feature_names)}
    X = np.zeros((len(feat_dicts), len(feature_names)), dtype=dtype)
    for row_idx, feats in enumerate(feat_dicts):
        row = X[row_idx]
        for key, value in feats.items():
            col = col_idx.get(key)
            if col is not None:
                row[col] = float(value)
    return X, feature_names


def validate_graph_training_rows(rows: List[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        fp = str(row.get("canonical_fingerprint") or "").strip()
        graph_json = row.get("graph_json")
        if not fp:
            raise CorpusIntegrityError(
                f"graph corpus row {idx} is missing canonical_fingerprint"
            )
        if fp in seen:
            raise CorpusIntegrityError(
                f"graph corpus contains duplicate canonical_fingerprint {fp}"
            )
        seen.add(fp)
        if not isinstance(graph_json, str) or not graph_json.strip():
            raise CorpusIntegrityError(f"graph corpus row {idx} is missing graph_json")
        recomputed = _graph_fingerprint(graph_json)
        if recomputed != fp:
            raise CorpusIntegrityError(
                f"graph corpus fingerprint mismatch for {fp}: recomputed {recomputed}"
            )
        n_rows = int(row.get("n_rows", 0) or 0)
        if n_rows < 1:
            raise CorpusIntegrityError(
                f"graph corpus row {fp} has invalid n_rows={n_rows}"
            )
        pass_rate = float(row.get("stage1_pass_rate", 0.0) or 0.0)
        if pass_rate < 0.0 or pass_rate > 1.0:
            raise CorpusIntegrityError(
                f"graph corpus row {fp} has invalid stage1_pass_rate={pass_rate}"
            )
        if bool(row.get("stage1_any_passed")) and pass_rate <= 0.0:
            raise CorpusIntegrityError(
                f"graph corpus row {fp} marks stage1_any_passed with zero pass rate"
            )
        induction_screening_auc = row.get("induction_screening_auc_500")
        if induction_screening_auc is not None and not math.isfinite(
            float(induction_screening_auc)
        ):
            raise CorpusIntegrityError(
                f"graph corpus row {fp} has invalid induction_screening_auc_500={induction_screening_auc}"
            )


def validate_predictor_training_rows(rows: List[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        fp = str(row.get("canonical_fingerprint") or "").strip()
        if not fp:
            raise CorpusIntegrityError(
                f"predictor corpus row {idx} is missing canonical_fingerprint"
            )
        if fp in seen:
            raise CorpusIntegrityError(
                f"predictor corpus contains duplicate canonical_fingerprint {fp}"
            )
        seen.add(fp)
        fp_json = row.get("fingerprint_json")
        if not isinstance(fp_json, str) or not fp_json.strip():
            raise CorpusIntegrityError(
                f"predictor corpus row {fp} is missing fingerprint_json"
            )
        target = row.get("target_loss_ratio")
        if target is None or not math.isfinite(float(target)):
            raise CorpusIntegrityError(
                f"predictor corpus row {fp} has invalid target_loss_ratio={target}"
            )
        n_rows = int(row.get("n_rows", 0) or 0)
        if n_rows < 1:
            raise CorpusIntegrityError(
                f"predictor corpus row {fp} has invalid n_rows={n_rows}"
            )


def _load_cached_corpus_rows(
    *,
    cache_kind: str,
    db_path: str,
    builder: Callable[[], List[Dict[str, Any]]],
    validator: Callable[[List[Dict[str, Any]]], None] | None,
) -> List[Dict[str, Any]]:
    key = (cache_kind, str(Path(db_path).resolve()), _db_cache_signature(db_path))
    with _CORPUS_CACHE_LOCK:
        entry = _CORPUS_CACHE.get(key)
    if entry is None:
        with _CORPUS_CACHE_LOCK:
            _CORPUS_CACHE_STATS["misses"] += 1
        rows = builder()
        entry = _CorpusCacheEntry(rows=rows, validated=False)
        with _CORPUS_CACHE_LOCK:
            _CORPUS_CACHE[key] = entry
        logger.debug(
            "ML corpus cache miss kind=%s db=%s rows=%d",
            cache_kind,
            db_path,
            len(rows),
        )
    else:
        with _CORPUS_CACHE_LOCK:
            _CORPUS_CACHE_STATS["hits"] += 1
        logger.debug(
            "ML corpus cache hit kind=%s db=%s rows=%d validated=%s",
            cache_kind,
            db_path,
            len(entry.rows),
            entry.validated,
        )
    if validator is not None and not entry.validated:
        validator(entry.rows)
        entry.validated = True
        with _CORPUS_CACHE_LOCK:
            _CORPUS_CACHE_STATS["validations"] += 1
        logger.debug(
            "ML corpus cache validated kind=%s db=%s rows=%d",
            cache_kind,
            db_path,
            len(entry.rows),
        )
    return entry.rows


def _build_graph_training_rows(db_path: str) -> List[Dict[str, Any]]:
    rows = _load_rust_corpus_rows(
        db_path,
        builder_name="build_graph_training_corpus",
        fallback_builder=_fallback_graph_training_rows,
    )
    return _attach_induction_metrics(rows, db_path)


def _build_screening_predictor_rows(db_path: str) -> List[Dict[str, Any]]:
    return _attach_induction_metrics(
        _fallback_screening_predictor_rows(db_path), db_path
    )


def _build_predictor_training_rows(db_path: str) -> List[Dict[str, Any]]:
    rows = _load_rust_corpus_rows(
        db_path,
        builder_name="build_predictor_training_corpus",
        fallback_builder=_fallback_predictor_training_rows,
    )
    return _attach_induction_metrics(rows, db_path)


def _db_cache_signature(
    db_path: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    db_file = Path(db_path)
    wal_file = db_file.with_name(db_file.name + "-wal")
    return (_path_signature(db_file), _path_signature(wal_file))


def _attach_induction_metrics(
    rows: List[Dict[str, Any]], db_path: str
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    fingerprints = [
        str(row.get("canonical_fingerprint") or "").strip()
        for row in rows
        if str(row.get("canonical_fingerprint") or "").strip()
    ]
    if not fingerprints:
        return rows
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    tables = {
        str(r["name"])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "induction_metrics_v2" not in tables:
        return rows
    metric_rows = []
    chunk_size = 500
    for start in range(0, len(fingerprints), chunk_size):
        chunk = fingerprints[start : start + chunk_size]
        placeholders = ", ".join("?" for _ in chunk)
        metric_rows.extend(
            conn.execute(
                f"""
                SELECT graph_fingerprint, auc, gap_4, gap_8, gap_16, gap_32, gap_64,
                       wall_ms, metric_version, speed_mode, train_steps, eval_examples,
                       batch_size, pool_size, source_cohort
                FROM induction_metrics_v2
                WHERE graph_fingerprint IN ({placeholders})
                """,
                chunk,
            ).fetchall()
        )

    by_fp = {
        str(row["graph_fingerprint"]): {
            "induction_screening_auc_500": (
                float(row["auc"]) if row["auc"] is not None else None
            ),
            "induction_gap_4": (
                float(row["gap_4"]) if row["gap_4"] is not None else None
            ),
            "induction_gap_8": (
                float(row["gap_8"]) if row["gap_8"] is not None else None
            ),
            "induction_gap_16": (
                float(row["gap_16"]) if row["gap_16"] is not None else None
            ),
            "induction_gap_32": (
                float(row["gap_32"]) if row["gap_32"] is not None else None
            ),
            "induction_gap_64": (
                float(row["gap_64"]) if row["gap_64"] is not None else None
            ),
            "induction_wall_ms_500": (
                float(row["wall_ms"]) if row["wall_ms"] is not None else None
            ),
            "induction_metric_version": str(row["metric_version"] or ""),
            "induction_speed_mode": str(row["speed_mode"] or ""),
            "induction_train_steps": (
                int(row["train_steps"]) if row["train_steps"] is not None else None
            ),
            "induction_eval_examples": (
                int(row["eval_examples"]) if row["eval_examples"] is not None else None
            ),
            "induction_batch_size": (
                int(row["batch_size"]) if row["batch_size"] is not None else None
            ),
            "induction_pool_size": (
                int(row["pool_size"]) if row["pool_size"] is not None else None
            ),
            "induction_source_cohort": str(row["source_cohort"] or ""),
        }
        for row in metric_rows
    }

    for row in rows:
        fp = str(row.get("canonical_fingerprint") or "").strip()
        if not fp:
            continue
        induction = by_fp.get(fp)
        if induction:
            row.update(induction)
    return rows


def _path_signature(path: Path) -> Tuple[int, int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0, 0)
    return (int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ino))


def _graph_fingerprint(graph_json: str) -> str:
    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, "fingerprint_notebook_graph"):
        try:
            return str(rust.fingerprint_notebook_graph(graph_json))
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning(
                "Rust graph fingerprinting failed; using Python fallback: %s", exc
            )

    from research.synthesis.serializer import graph_from_json

    try:
        return str(graph_from_json(graph_json).fingerprint())
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        import hashlib

        return hashlib.sha256(graph_json.encode()).hexdigest()[:16]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'program_results_compat' LIMIT 1"
    ).fetchone()
    return "program_results_compat" if row else "program_results"


def _has_trust_columns(conn: sqlite3.Connection, table: str) -> bool:
    return {"trust_label", "comparability_label"}.issubset(_table_columns(conn, table))


def _sql_membership_clause(column: str, values: tuple[str, ...]) -> str:
    quoted_values = ", ".join(f"'{value}'" for value in values)
    return f"COALESCE({column}, '') IN ({quoted_values})"


_BYTE_TRAINING_TOKENIZER_VALUES = (
    "byte",
    "bytes",
    "raw_byte",
    "raw_bytes",
)
# Marker substrings that flag byte-era / pre-BPE rows whose
# wikitext_perplexity is in the wrong units.  IMPORTANT: do NOT add
# "bpe" here — `bpe_eval_v1` is the GOOD post-backfill version and
# matching it as a "byte" marker was previously excluding the rows we
# wanted to keep while passing the byte-era ones (which have no marker
# substring in `screening_wikitext_v1`).
_BYTE_TRAINING_VERSION_MARKERS = ("byte",)
# Positive identifier for the post-BPE-backfill metric version.  Used
# to conditionally aggregate wikitext_perplexity in the predictor /
# analytics paths so only comparable PPLs feed the labels.
BPE_EVAL_METRIC_VERSION = "bpe_eval_v1"
_BYTE_TRAINING_PROVENANCE_PATHS = (
    "$.tokenizer_mode",
    "$.tokenizer_id",
    "$.tokenizer_version",
    "$.screening_wikitext_metric_version",
    "$.metric_version",
    "$.wikitext_metric_version",
)


def _qualified_col(name: str, alias: str | None) -> str:
    return f"{alias}.{name}" if alias else name


def _lower_coalesce_expr(expr: str) -> str:
    return f"LOWER(COALESCE(CAST({expr} AS TEXT), ''))"


def _json_text_expr(json_col: str, path: str) -> str:
    return (
        f"CASE WHEN json_valid(COALESCE({json_col}, '{{}}')) "
        f"THEN json_extract({json_col}, '{path}') ELSE NULL END"
    )


def _sql_not_contains_markers(expr: str) -> str:
    lowered = _lower_coalesce_expr(expr)
    return (
        "("
        + " AND ".join(
            f"{lowered} NOT LIKE '%{marker}%'"
            for marker in _BYTE_TRAINING_VERSION_MARKERS
        )
        + ")"
    )


def _non_byte_training_data_clauses(
    available: set[str], *, alias: str | None = None
) -> List[str]:
    """SQL predicates that keep byte-era evaluation rows out of ML corpora."""
    clauses: List[str] = []
    if "tokenizer_mode" in available:
        tokenizer_values = ", ".join(
            f"'{value}'" for value in _BYTE_TRAINING_TOKENIZER_VALUES
        )
        clauses.append(
            f"{_lower_coalesce_expr(_qualified_col('tokenizer_mode', alias))} "
            f"NOT IN ({tokenizer_values})"
        )
    if "screening_wikitext_metric_version" in available:
        clauses.append(
            _sql_not_contains_markers(
                _qualified_col("screening_wikitext_metric_version", alias)
            )
        )
    if "data_provenance_json" in available:
        json_col = _qualified_col("data_provenance_json", alias)
        for path in _BYTE_TRAINING_PROVENANCE_PATHS:
            clauses.append(_sql_not_contains_markers(_json_text_expr(json_col, path)))
    return clauses


def _ppl_metric_is_comparable(metric_version: str) -> bool:
    """Rows surviving byte filters are comparable when BPE-tagged or legacy-unversioned."""
    return metric_version == BPE_EVAL_METRIC_VERSION or metric_version == ""


def _program_results_columns(db_path: str) -> set[str]:
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    return _table_columns(conn, "program_results")


def _load_rust_corpus_rows(
    db_path: str,
    *,
    builder_name: str,
    fallback_builder: Callable[[str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, builder_name):
        builder = getattr(rust, builder_name)
        try:
            payload = builder(db_path)
            rows = json.loads(payload)
        except (TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Rust corpus builder %s failed for %s; using SQL fallback: %s",
                builder_name,
                db_path,
                exc,
            )
        else:
            if isinstance(rows, list):
                return rows
            logger.warning(
                "Rust corpus builder %s returned non-list payload for %s; using SQL fallback",
                builder_name,
                db_path,
            )
    return fallback_builder(db_path)


def _fallback_graph_training_rows(db_path: str) -> List[Dict[str, Any]]:
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    pr_cols = _table_columns(conn, "program_results")
    pr_table = _program_results_read_table(conn)
    where = [
        "TRIM(COALESCE(graph_json, '')) <> ''",
        "graph_json <> '{}'",
    ]
    where.extend(_non_byte_training_data_clauses(pr_cols))
    if _has_trust_columns(conn, "program_results"):
        where.extend(
            [
                _sql_membership_clause("trust_label", TRUSTED_TRUST_LABELS),
                _sql_membership_clause(
                    "comparability_label", TRUSTED_COMPARABILITY_LABELS
                ),
            ]
        )
    # Pull screening_wikitext_metric_version so the PPL aggregation
    # below can skip byte-era rows whose units differ from BPE rows.
    metric_version_select = (
        ", screening_wikitext_metric_version"
        if "screening_wikitext_metric_version" in pr_cols
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT graph_json, stage1_passed, wikitext_perplexity, loss_ratio,
               stage0_passed, stage05_passed, timestamp{metric_version_select}
        FROM {pr_table}
        WHERE {" AND ".join(where)}
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = resolve_graph_json_value(conn, db_path, row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            {
                "canonical_fingerprint": canonical,
                "graph_json": graph_json,
                "stage1_any_passed": False,
                "stage1_pass_rate": 0.0,
                "stage0_any_passed": False,
                "stage05_any_passed": False,
                "wikitext_perplexity_best": None,
                "loss_ratio_best": None,
                "n_rows": 0,
                "latest_timestamp": 0.0,
                "_best_rank": None,
                "_n_stage1_passed": 0,
            },
        )
        stage1_passed = bool(row["stage1_passed"])
        group["n_rows"] += 1
        group["_n_stage1_passed"] += int(stage1_passed)
        group["stage1_any_passed"] = bool(group["stage1_any_passed"] or stage1_passed)
        group["stage0_any_passed"] = bool(
            group["stage0_any_passed"] or bool(row["stage0_passed"])
        )
        group["stage05_any_passed"] = bool(
            group["stage05_any_passed"] or bool(row["stage05_passed"])
        )
        # Only fold this row's PPL into the per-fingerprint best PPL
        # when its metric version is the BPE backfilled one.  Byte-era
        # rows have PPL in different units (range 23 – 485M) and
        # silently corrupted the predictor's labels here for months.
        ppl_keys = row.keys() if hasattr(row, "keys") else None
        metric_version = (
            str(row["screening_wikitext_metric_version"] or "").strip()
            if ppl_keys is not None and "screening_wikitext_metric_version" in ppl_keys
            else ""
        )
        if _ppl_metric_is_comparable(metric_version):
            group["wikitext_perplexity_best"] = _min_opt(
                group["wikitext_perplexity_best"], row["wikitext_perplexity"]
            )
        group["loss_ratio_best"] = _min_opt(group["loss_ratio_best"], row["loss_ratio"])
        group["latest_timestamp"] = max(
            float(group["latest_timestamp"]), float(row["timestamp"] or 0.0)
        )

        rank = (
            0 if stage1_passed else 1,
            row["loss_ratio"] is None,
            float(row["loss_ratio"]) if row["loss_ratio"] is not None else float("inf"),
            float(row["timestamp"] or 0.0),
        )
        if group["_best_rank"] is None or rank < group["_best_rank"]:
            group["_best_rank"] = rank
            group["graph_json"] = graph_json

    out: List[Dict[str, Any]] = []
    for group in grouped.values():
        n_rows = int(group["n_rows"])
        group["stage1_pass_rate"] = float(group.pop("_n_stage1_passed")) / max(
            n_rows, 1
        )
        group.pop("_best_rank", None)
        out.append(group)
    out.sort(key=lambda row: str(row["canonical_fingerprint"]))
    return out


def _fallback_predictor_training_rows(db_path: str) -> List[Dict[str, Any]]:
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    pr_cols = _table_columns(conn, "program_results")
    pr_table = _program_results_read_table(conn)
    where = [
        "TRIM(COALESCE(pr.graph_json, '')) <> ''",
        "pr.graph_json <> '{}'",
        "pr.fingerprint_json IS NOT NULL",
        "COALESCE(l.investigation_loss_ratio, pr.loss_ratio) IS NOT NULL",
    ]
    where.extend(_non_byte_training_data_clauses(pr_cols, alias="pr"))
    if _has_trust_columns(conn, "program_results"):
        where.extend(
            [
                _sql_membership_clause("pr.trust_label", PROMOTABLE_TRUST_LABELS),
                _sql_membership_clause(
                    "pr.comparability_label", PROMOTABLE_COMPARABILITY_LABELS
                ),
            ]
        )
    rows = conn.execute(
        f"""
        SELECT pr.graph_json, pr.fingerprint_json, pr.novelty_score,
               pr.structural_novelty,
               COALESCE(l.investigation_loss_ratio, pr.loss_ratio) AS target_loss_ratio,
               COALESCE(l.tier, 'screening') AS tier,
               pr.timestamp
        FROM {pr_table} pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE {" AND ".join(where)}
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = resolve_graph_json_value(conn, db_path, row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            {
                "canonical_fingerprint": canonical,
                "fingerprint_json": str(row["fingerprint_json"]),
                "novelty_score": row["novelty_score"],
                "structural_novelty": row["structural_novelty"],
                "target_loss_ratio": float(row["target_loss_ratio"]),
                "tier": str(row["tier"] or "screening"),
                "n_rows": 0,
                "_best_rank": None,
            },
        )
        group["n_rows"] += 1
        rank = (
            _tier_rank(str(row["tier"] or "screening")),
            float(row["target_loss_ratio"]),
            float(row["timestamp"] or 0.0),
        )
        if group["_best_rank"] is None or rank < group["_best_rank"]:
            group["_best_rank"] = rank
            group["fingerprint_json"] = str(row["fingerprint_json"])
            group["novelty_score"] = row["novelty_score"]
            group["structural_novelty"] = row["structural_novelty"]
            group["target_loss_ratio"] = float(row["target_loss_ratio"])
            group["tier"] = str(row["tier"] or "screening")

    out: List[Dict[str, Any]] = []
    for group in grouped.values():
        group.pop("_best_rank", None)
        out.append(group)
    out.sort(key=lambda row: str(row["canonical_fingerprint"]))
    return out


def _fallback_screening_predictor_rows(db_path: str) -> List[Dict[str, Any]]:
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    pr_cols = _table_columns(conn, "program_results")
    pr_table = _program_results_read_table(conn)
    use_explicit_flags = "data_provenance_json" in pr_cols
    where = [
        "TRIM(COALESCE(pr.graph_json, '')) <> ''",
        "pr.graph_json <> '{}'",
        # Aria-scheduler's IR converter rejects graphs with <2 nodes — filter
        # at the corpus layer so GraphPredictor's batched IR call doesn't fail
        # for every caller. Rows with NULL graph_n_ops (older schema) pass.
        "(pr.graph_n_ops IS NULL OR pr.graph_n_ops >= 2)",
    ]
    where.extend(_non_byte_training_data_clauses(pr_cols, alias="pr"))
    if use_explicit_flags:
        where.append(
            "("
            "json_extract(pr.data_provenance_json, '$.eligible_for_screening_model_training') = 1"
            " OR ("
            + " AND ".join(
                [
                    _sql_membership_clause("pr.trust_label", TRUSTED_TRUST_LABELS),
                    _sql_membership_clause(
                        "pr.comparability_label", TRUSTED_COMPARABILITY_LABELS
                    ),
                ]
            )
            + ")"
            ")"
        )
    else:
        where.append(
            "("
            "("
            + " AND ".join(
                [
                    _sql_membership_clause("pr.trust_label", TRUSTED_TRUST_LABELS),
                    _sql_membership_clause(
                        "pr.comparability_label", TRUSTED_COMPARABILITY_LABELS
                    ),
                ]
            )
            + ") OR ("
            + " AND ".join(
                [
                    "COALESCE(pr.trust_label, '') = 'runtime_observation'",
                    "COALESCE(pr.result_cohort, '') = 'search'",
                    "COALESCE(pr.stage0_passed, 0) = 1",
                    "COALESCE(pr.stage05_passed, 0) = 1",
                    "COALESCE(pr.stage1_passed, 0) = 0",
                    "json_extract(pr.data_provenance_json, '$.provenance_complete') = 1",
                ]
            )
            + "))"
        )
    metric_version_select = (
        ", pr.screening_wikitext_metric_version"
        if "screening_wikitext_metric_version" in pr_cols
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT pr.graph_json, pr.stage1_passed, pr.wikitext_perplexity, pr.loss_ratio,
               pr.stage0_passed, pr.stage05_passed, pr.timestamp, pr.trust_label,
               pr.comparability_label, pr.result_cohort, pr.data_provenance_json,
               pr.hellaswag_acc, pr.induction_screening_auc, pr.ar_legacy_auc,
               pr.blimp_overall_accuracy, pr.binding_screening_composite,
               pr.induction_intermediate_auc, pr.binding_intermediate_auc,
               pr.validation_loss_ratio, pr.rapid_screening_passed,
               pr.initial_loss, pr.mean_grad_norm, pr.max_grad_norm,
               pr.grad_norm_std,
               -- Gemini trajectory metrics (v9 scoring + ML predictor features)
               pr.fp_jacobian_erf_density, pr.fp_jacobian_erf_variance,
               pr.fp_icld_velocity, pr.fp_logit_margin_velocity,
               pr.fp_id_collapse_rate, pr.fp_jacobian_spectral_norm,
               pr.diagnostic_score, pr.cross_task_score,
               l.composite_score{metric_version_select}
        FROM {pr_table} pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE {" AND ".join(where)}
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = resolve_graph_json_value(conn, db_path, row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            {
                "canonical_fingerprint": canonical,
                "graph_json": graph_json,
                "stage1_any_passed": False,
                "stage1_pass_rate": 0.0,
                "stage0_any_passed": False,
                "stage05_any_passed": False,
                "wikitext_perplexity_best": None,
                "loss_ratio_best": None,
                "hellaswag_acc_best": None,
                "induction_screening_auc_best": None,
                "ar_legacy_auc_best": None,
                "blimp_overall_accuracy_best": None,
                "binding_screening_composite_best": None,
                "induction_intermediate_auc_best": None,
                "binding_intermediate_auc_best": None,
                "validation_loss_ratio_best": None,
                "rapid_screening_passed_best": None,
                "composite_score_best": None,
                "initial_loss_best": None,
                "mean_grad_norm_best": None,
                "max_grad_norm_best": None,
                "grad_norm_std_best": None,
                # Gemini trajectory metrics — feature inputs for v9+ ML
                # predictor. Aggregator picks the most-informative
                # value across rows sharing this fingerprint.
                "fp_jacobian_erf_density_best": None,
                "fp_jacobian_erf_variance_best": None,
                "fp_icld_velocity_best": None,
                "fp_logit_margin_velocity_best": None,
                "fp_id_collapse_rate_best": None,
                "fp_jacobian_spectral_norm_best": None,
                # Understanding-tier scoring features (added 2026-04-26).
                "diagnostic_score_best": None,
                "cross_task_score_best": None,
                "n_rows": 0,
                "latest_timestamp": 0.0,
                "has_trusted_positive": False,
                "has_runtime_negative": False,
                "_best_rank": None,
                "_n_stage1_passed": 0,
            },
        )
        stage1_passed = bool(row["stage1_passed"])
        is_runtime_negative = (
            str(row["trust_label"] or "") == "runtime_observation" and not stage1_passed
        )
        is_trusted_positive = stage1_passed
        raw_payload = (
            row["data_provenance_json"]
            if "data_provenance_json" in row.keys()
            else None
        )
        screening_role = ""
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                payload = json.loads(raw_payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            if isinstance(payload, dict):
                screening_role = str(
                    payload.get("screening_model_training_role") or ""
                ).strip()
        if screening_role == "negative":
            is_runtime_negative = True
        elif screening_role == "positive":
            is_trusted_positive = True
        group["n_rows"] += 1
        group["_n_stage1_passed"] += int(stage1_passed)
        group["stage1_any_passed"] = bool(group["stage1_any_passed"] or stage1_passed)
        group["stage0_any_passed"] = bool(
            group["stage0_any_passed"] or bool(row["stage0_passed"])
        )
        group["stage05_any_passed"] = bool(
            group["stage05_any_passed"] or bool(row["stage05_passed"])
        )
        group["has_trusted_positive"] = bool(
            group["has_trusted_positive"] or is_trusted_positive
        )
        group["has_runtime_negative"] = bool(
            group["has_runtime_negative"] or is_runtime_negative
        )
        # Skip byte-era PPL: row in different units would corrupt the
        # min() across runs.  See _BYTE_TRAINING_VERSION_MARKERS notes.
        _row_keys = row.keys() if hasattr(row, "keys") else ()
        _metric_version = (
            str(row["screening_wikitext_metric_version"] or "").strip()
            if "screening_wikitext_metric_version" in _row_keys
            else ""
        )
        if _ppl_metric_is_comparable(_metric_version):
            group["wikitext_perplexity_best"] = _min_opt(
                group["wikitext_perplexity_best"], row["wikitext_perplexity"]
            )
        group["loss_ratio_best"] = _min_opt(group["loss_ratio_best"], row["loss_ratio"])
        group["validation_loss_ratio_best"] = _min_opt(
            group["validation_loss_ratio_best"], row["validation_loss_ratio"]
        )
        for probe_key, row_key in (
            ("hellaswag_acc_best", "hellaswag_acc"),
            ("induction_screening_auc_best", "induction_screening_auc"),
            ("ar_legacy_auc_best", "ar_legacy_auc"),
            ("blimp_overall_accuracy_best", "blimp_overall_accuracy"),
            ("binding_screening_composite_best", "binding_screening_composite"),
            (
                "induction_intermediate_auc_best",
                "induction_intermediate_auc",
            ),
            (
                "binding_intermediate_auc_best",
                "binding_intermediate_auc",
            ),
            # 0/1 int → max() == OR
            ("rapid_screening_passed_best", "rapid_screening_passed"),
            ("composite_score_best", "composite_score"),
            # Gemini trajectory metrics where higher = better.
            ("fp_jacobian_erf_density_best", "fp_jacobian_erf_density"),
            ("fp_jacobian_erf_variance_best", "fp_jacobian_erf_variance"),
            ("fp_logit_margin_velocity_best", "fp_logit_margin_velocity"),
            ("fp_jacobian_spectral_norm_best", "fp_jacobian_spectral_norm"),
            # Understanding-tier (higher = better)
            ("diagnostic_score_best", "diagnostic_score"),
            ("cross_task_score_best", "cross_task_score"),
        ):
            group[probe_key] = _max_opt(group[probe_key], row[row_key])
        # Training dynamics: take min (best-behaved run).
        # ICLD velocity and ID Collapse rate are also "min is better"
        # (more negative = stronger signal).
        for dyn_key, row_key in (
            ("initial_loss_best", "initial_loss"),
            ("mean_grad_norm_best", "mean_grad_norm"),
            ("max_grad_norm_best", "max_grad_norm"),
            ("grad_norm_std_best", "grad_norm_std"),
            ("fp_icld_velocity_best", "fp_icld_velocity"),
            ("fp_id_collapse_rate_best", "fp_id_collapse_rate"),
        ):
            group[dyn_key] = _min_opt(group[dyn_key], row[row_key])
        group["latest_timestamp"] = max(
            float(group["latest_timestamp"]), float(row["timestamp"] or 0.0)
        )
        rank = (
            0 if is_trusted_positive else 1,
            0 if is_runtime_negative else 1,
            row["loss_ratio"] is None,
            float(row["loss_ratio"]) if row["loss_ratio"] is not None else float("inf"),
            float(row["timestamp"] or 0.0),
        )
        if group["_best_rank"] is None or rank < group["_best_rank"]:
            group["_best_rank"] = rank
            group["graph_json"] = graph_json

    out: List[Dict[str, Any]] = []
    for group in grouped.values():
        n_rows = int(group["n_rows"])
        group["stage1_pass_rate"] = float(group.pop("_n_stage1_passed")) / max(
            n_rows, 1
        )
        group.pop("_best_rank", None)
        out.append(group)
    out.sort(key=lambda row: str(row["canonical_fingerprint"]))
    return out


def _select_program_result_col(available: set[str], name: str) -> str:
    return name if name in available else f"NULL AS {name}"


def _select_first_program_result_col(
    available: set[str], alias: str, *candidates: str
) -> str:
    existing = [name for name in candidates if name in available]
    if not existing:
        return f"NULL AS {alias}"
    if len(existing) == 1:
        source = existing[0]
        return source if source == alias else f"{source} AS {alias}"
    return f"COALESCE({', '.join(existing)}) AS {alias}"


def _graph_analysis_select_cols(available: set[str]) -> List[str]:
    def col(name: str) -> str:
        return _select_program_result_col(available, name)

    def first(alias: str, *names: str) -> str:
        return _select_first_program_result_col(available, alias, *names)

    return [
        "result_id",
        "experiment_id",
        "graph_json",
        col("novelty_score"),
        col("loss_ratio"),
        col("param_count"),
        col("graph_n_params_estimate"),
        col("graph_depth"),
        col("graph_uses_math_spaces"),
        col("stage0_passed"),
        col("stage05_passed"),
        col("stage1_passed"),
        col("timestamp"),
        col("induction_screening_auc"),
        first(
            "binding_screening_auc", "binding_curriculum_auc", "binding_screening_auc"
        ),
        col("binding_screening_composite"),
        col("ar_legacy_auc"),
        col("hellaswag_acc"),
        col("blimp_overall_accuracy"),
        col("induction_intermediate_auc"),
        col("binding_intermediate_auc"),
    ]


def _initial_graph_analysis_group(
    canonical: str, row: sqlite3.Row, graph_json: str
) -> Dict[str, Any]:
    return {
        "canonical_fingerprint": canonical,
        "result_id": str(row["result_id"] or ""),
        "experiment_id": str(row["experiment_id"] or ""),
        "graph_json": graph_json,
        "novelty_score": row["novelty_score"],
        "loss_ratio": row["loss_ratio"],
        "param_count": row["param_count"],
        "graph_n_params_estimate": row["graph_n_params_estimate"],
        "graph_depth": row["graph_depth"],
        "graph_uses_math_spaces": bool(row["graph_uses_math_spaces"]),
        "induction_screening_auc": row["induction_screening_auc"],
        "binding_screening_auc": row["binding_screening_auc"],
        "binding_screening_composite": row["binding_screening_composite"],
        "ar_legacy_auc": row["ar_legacy_auc"],
        "hellaswag_acc": row["hellaswag_acc"],
        "blimp_overall_accuracy": row["blimp_overall_accuracy"],
        "induction_intermediate_auc": row["induction_intermediate_auc"],
        "binding_intermediate_auc": row["binding_intermediate_auc"],
        "stage0_any_passed": False,
        "stage05_any_passed": False,
        "stage1_any_passed": False,
        "n_rows": 0,
        "latest_timestamp": 0.0,
        "_best_rank": None,
    }


def _update_graph_analysis_group(
    group: Dict[str, Any],
    row: sqlite3.Row,
    graph_json: str,
) -> None:
    stage1_passed = bool(row["stage1_passed"])
    group["n_rows"] += 1
    group["stage0_any_passed"] = bool(
        group["stage0_any_passed"] or bool(row["stage0_passed"])
    )
    group["stage05_any_passed"] = bool(
        group["stage05_any_passed"] or bool(row["stage05_passed"])
    )
    group["stage1_any_passed"] = bool(group["stage1_any_passed"] or stage1_passed)
    group["latest_timestamp"] = max(
        float(group["latest_timestamp"]), float(row["timestamp"] or 0.0)
    )
    for metric_key in (
        "induction_screening_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "ar_legacy_auc",
        "hellaswag_acc",
        "blimp_overall_accuracy",
        "induction_intermediate_auc",
        "binding_intermediate_auc",
    ):
        group[metric_key] = _max_opt(group.get(metric_key), row[metric_key])

    rank = (
        0 if stage1_passed else 1,
        row["loss_ratio"] is None,
        float(row["loss_ratio"]) if row["loss_ratio"] is not None else float("inf"),
        -float(row["novelty_score"]) if row["novelty_score"] is not None else 0.0,
        float(row["timestamp"] or 0.0),
    )
    if group["_best_rank"] is None or rank < group["_best_rank"]:
        group["_best_rank"] = rank
        group["result_id"] = str(row["result_id"] or "")
        group["experiment_id"] = str(row["experiment_id"] or "")
        group["graph_json"] = graph_json
        group["novelty_score"] = row["novelty_score"]
        group["loss_ratio"] = row["loss_ratio"]
        group["param_count"] = row["param_count"]
        group["graph_n_params_estimate"] = row["graph_n_params_estimate"]
        group["graph_depth"] = row["graph_depth"]
        group["graph_uses_math_spaces"] = bool(row["graph_uses_math_spaces"])


def _finalize_graph_analysis_groups(
    grouped: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group in grouped.values():
        group.pop("_best_rank", None)
        out.append(group)
    out.sort(key=lambda row: str(row["canonical_fingerprint"]))
    return out


def _fallback_graph_analysis_rows(db_path: str) -> List[Dict[str, Any]]:
    from ..notebook.shared_conn import get_notebook_conn

    conn = get_notebook_conn(db_path)
    pr_cols = _table_columns(conn, "program_results")
    pr_table = _program_results_read_table(conn)
    select_cols = _graph_analysis_select_cols(pr_cols)
    where = [
        "TRIM(COALESCE(graph_json, '')) <> ''",
        "graph_json <> '{}'",
        *_non_byte_training_data_clauses(pr_cols),
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM {pr_table}
        WHERE {" AND ".join(where)}
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = resolve_graph_json_value(conn, db_path, row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            _initial_graph_analysis_group(canonical, row, graph_json),
        )
        _update_graph_analysis_group(group, row, graph_json)
    return _finalize_graph_analysis_groups(grouped)


def _min_opt(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if float(candidate) < float(current) else current


def _max_opt(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if float(candidate) > float(current) else current


def _tier_rank(tier: str) -> int:
    return {
        "breakthrough": 0,
        "validation": 1,
        "investigation": 2,
        "investigation_failed": 3,
        "investigation_fingerprint_incomplete": 4,
        "screening": 5,
        "screened_out": 6,
    }.get(tier, 7)
