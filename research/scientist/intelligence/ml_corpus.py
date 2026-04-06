from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

from ..native.core import _try_import_rust_scheduler

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


def rerun_confidence_weight(n_rows: int) -> float:
    return min(math.sqrt(max(int(n_rows or 1), 1)), 3.0)


def _clear_corpus_cache() -> None:
    with _CORPUS_CACHE_LOCK:
        _CORPUS_CACHE.clear()
        _CORPUS_CACHE_STATS["clears"] += 1


def get_corpus_cache_stats() -> Dict[str, int]:
    with _CORPUS_CACHE_LOCK:
        return dict(_CORPUS_CACHE_STATS)


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
        induction_auc = row.get("induction_auc_500")
        if induction_auc is not None and not math.isfinite(float(induction_auc)):
            raise CorpusIntegrityError(
                f"graph corpus row {fp} has invalid induction_auc_500={induction_auc}"
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
    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, "build_graph_training_corpus"):
        rows = json.loads(rust.build_graph_training_corpus(db_path))
    else:
        rows = _fallback_graph_training_rows(db_path)
    return _attach_induction_metrics(rows, db_path)


def _build_predictor_training_rows(db_path: str) -> List[Dict[str, Any]]:
    rust = _try_import_rust_scheduler()
    if rust is not None and hasattr(rust, "build_predictor_training_corpus"):
        rows = json.loads(rust.build_predictor_training_corpus(db_path))
    else:
        rows = _fallback_predictor_training_rows(db_path)
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
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            str(r["name"])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "induction_metrics_v2" not in tables:
            return rows
        metric_rows = conn.execute(
            """
            SELECT graph_fingerprint, auc, gap_4, gap_8, gap_16, gap_32, gap_64,
                   wall_ms, metric_version, speed_mode, train_steps, eval_examples,
                   batch_size, pool_size, source_cohort
            FROM induction_metrics_v2
            """
        ).fetchall()
    finally:
        conn.close()

    by_fp = {
        str(row["graph_fingerprint"]): {
            "induction_auc_500": (
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
        return str(rust.fingerprint_notebook_graph(graph_json))

    from research.synthesis.graph import ComputationGraph

    return str(ComputationGraph.from_dict(json.loads(graph_json)).fingerprint())


def _fallback_graph_training_rows(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT graph_json, stage1_passed, wikitext_perplexity, loss_ratio,
               stage0_passed, stage05_passed, timestamp
        FROM program_results
        WHERE TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{}'
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = str(row["graph_json"])
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
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pr.graph_json, pr.fingerprint_json, pr.novelty_score,
               pr.structural_novelty,
               COALESCE(l.investigation_loss_ratio, pr.loss_ratio) AS target_loss_ratio,
               COALESCE(l.tier, 'screening') AS tier,
               pr.timestamp
        FROM program_results pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
          AND pr.graph_json <> '{}'
          AND pr.fingerprint_json IS NOT NULL
          AND COALESCE(l.investigation_loss_ratio, pr.loss_ratio) IS NOT NULL
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = str(row["graph_json"])
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


def _fallback_graph_analysis_rows(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT result_id, experiment_id, graph_json, novelty_score, loss_ratio,
               param_count, graph_n_params_estimate, graph_depth,
               graph_uses_math_spaces, stage0_passed, stage05_passed,
               stage1_passed, timestamp
        FROM program_results
        WHERE TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{}'
        """
    ).fetchall()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = str(row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        group = grouped.setdefault(
            canonical,
            {
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
                "stage0_any_passed": False,
                "stage05_any_passed": False,
                "stage1_any_passed": False,
                "n_rows": 0,
                "latest_timestamp": 0.0,
                "_best_rank": None,
            },
        )
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

    out: List[Dict[str, Any]] = []
    for group in grouped.values():
        group.pop("_best_rank", None)
        out.append(group)
    out.sort(key=lambda row: str(row["canonical_fingerprint"]))
    return out


def _min_opt(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if float(candidate) < float(current) else current


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
