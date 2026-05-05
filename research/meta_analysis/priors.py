"""Build and apply meta-analysis grammar priors.

The meta-analysis database is an offline, derived dataset. Generation reads a
small JSON artifact from this module so normal candidate synthesis does not run
large SQLite scans on the hot path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from .metadata_db import DEFAULT_META_ANALYSIS_DB


DEFAULT_PRIOR_DIR = Path("research/artifacts/meta_analysis_priors")
PRIOR_SCHEMA_VERSION = "meta_analysis_prior_v1"
VALID_TARGETS = frozenset({"induction", "induction_v2", "composite", "balanced"})

_TEXT_COLUMNS = {
    "op_name",
    "template_name",
    "op_category",
    "op_algebraic_space",
}

_CATEGORY_POLICY: dict[str, dict[str, float]] = {
    "induction": {
        "frequency": 1.80,
        "functional": 1.70,
        "mixing": 1.25,
        "linear_algebra": 1.10,
        "sequence": 0.90,
        "math_space": 0.85,
        "elementwise_unary": 0.85,
        "reduction": 0.75,
    },
    "induction_v2": {
        "frequency": 1.95,
        "functional": 1.80,
        "mixing": 1.30,
        "linear_algebra": 1.10,
        "sequence": 0.90,
        "math_space": 0.80,
        "elementwise_unary": 0.80,
        "reduction": 0.70,
    },
    "composite": {
        "mixing": 1.50,
        "linear_algebra": 1.35,
        "sequence": 1.25,
        "frequency": 1.10,
        "functional": 1.10,
        "math_space": 1.10,
        "elementwise_unary": 0.85,
        "reduction": 0.75,
    },
    "balanced": {
        "frequency": 1.55,
        "functional": 1.45,
        "mixing": 1.35,
        "linear_algebra": 1.15,
        "sequence": 1.05,
        "math_space": 0.90,
        "elementwise_unary": 0.85,
        "reduction": 0.75,
    },
}

_SEEDED_OP_WEIGHTS: dict[str, dict[str, float]] = {
    "induction": {
        "rope_rotate": 1.55,
        "token_entropy": 1.40,
        "entropy_score": 1.35,
        "spectral_filter": 1.30,
    },
    "induction_v2": {
        "rope_rotate": 1.75,
        "token_entropy": 1.55,
        "entropy_score": 1.45,
        "spectral_filter": 1.40,
    },
    "composite": {
        "spectral_filter": 1.35,
        "rope_rotate": 1.25,
        "adjacent_token_merge": 1.15,
        "token_entropy": 1.10,
    },
    "balanced": {
        "rope_rotate": 1.45,
        "token_entropy": 1.30,
        "entropy_score": 1.25,
        "spectral_filter": 1.30,
        "adjacent_token_merge": 1.10,
    },
}


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db = Path(path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.DatabaseError:
        return set()


def _select_column(columns: set[str], name: str, alias: str | None = None) -> str:
    out_name = alias or name
    if name in columns:
        return f'"{name}" AS "{out_name}"'
    sql_type = "TEXT" if out_name in _TEXT_COLUMNS else "REAL"
    return f'CAST(NULL AS {sql_type}) AS "{out_name}"'


def _metric_expr(columns: set[str], preferred: str, fallback: str | None = None) -> str:
    if preferred in columns and fallback and fallback in columns:
        return f'COALESCE("{preferred}", "{fallback}")'
    if preferred in columns:
        return f'"{preferred}"'
    if fallback and fallback in columns:
        return f'"{fallback}"'
    return "CAST(NULL AS REAL)"


def _induction_metric_expr(columns: set[str], target: str) -> str:
    if target == "induction_v2":
        return _metric_expr(columns, "induction_v2_investigation_auc")
    return _metric_expr(columns, "induction_v2_investigation_auc", "induction_auc")


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


def _relative_lift(value: float | None, baseline: float | None, floor: float) -> float:
    if value is None or baseline is None:
        return 0.0
    denom = max(abs(float(baseline)), floor)
    return (float(value) - float(baseline)) / denom


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_weight(value: float) -> float:
    return round(float(value), 4)


def _score_row(
    row: dict[str, Any], global_stats: dict[str, float | None], target: str
) -> float:
    ind_lift = _relative_lift(
        row.get("mean_induction"), global_stats["mean_induction"], 0.02
    )
    comp_lift = _relative_lift(
        row.get("mean_composite"), global_stats["mean_composite"], 1.0
    )
    s1_lift = _relative_lift(row.get("s1_rate"), global_stats["s1_rate"], 0.05)
    if target == "induction":
        return ind_lift + 0.25 * s1_lift
    if target == "composite":
        return 0.85 * comp_lift + 0.20 * ind_lift + 0.15 * s1_lift
    return 0.50 * ind_lift + 0.35 * comp_lift + 0.15 * s1_lift


def _multiplier_from_score(
    score: float, support: int, min_support: int, scale: float
) -> float:
    shrink = min(
        1.0, math.sqrt(max(0, int(support)) / max(1.0, float(min_support) * 4.0))
    )
    return 1.0 + scale * float(score) * shrink


def _global_stats(conn: sqlite3.Connection, target: str) -> dict[str, float | None]:
    cols = _table_columns(conn, "op_observations")
    if not cols:
        return {"mean_induction": None, "mean_composite": None, "s1_rate": None}
    induction_expr = _induction_metric_expr(cols, target)
    row = conn.execute(
        f"""
        SELECT
            AVG({induction_expr}) AS mean_induction,
            AVG({_metric_expr(cols, "composite_score")}) AS mean_composite,
            AVG({_metric_expr(cols, "stage1_passed")}) AS s1_rate
        FROM op_observations
        """
    ).fetchone()
    return {
        "mean_induction": _safe_float(row["mean_induction"] if row else None),
        "mean_composite": _safe_float(row["mean_composite"] if row else None),
        "s1_rate": _safe_float(row["s1_rate"] if row else None),
    }


def _aggregate_table(
    conn: sqlite3.Connection,
    table: str,
    key_column: str,
    *,
    extra_columns: tuple[str, ...] = (),
    target: str = "balanced",
) -> list[dict[str, Any]]:
    cols = _table_columns(conn, table)
    if key_column not in cols:
        return []
    induction_expr = _induction_metric_expr(cols, target)
    select_parts = [
        f'"{key_column}" AS key',
        f"COUNT({induction_expr}) AS induction_support",
        "COUNT(*) AS total_support",
        f"AVG({induction_expr}) AS mean_induction",
        f"AVG({_metric_expr(cols, 'composite_score')}) AS mean_composite",
        f"AVG({_metric_expr(cols, 'stage1_passed')}) AS s1_rate",
    ]
    for col in extra_columns:
        if col in cols:
            select_parts.append(f'MAX("{col}") AS "{col}"')
        else:
            sql_type = "TEXT" if col in _TEXT_COLUMNS else "REAL"
            select_parts.append(f'CAST(NULL AS {sql_type}) AS "{col}"')
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_parts)}
        FROM "{table}"
        GROUP BY "{key_column}"
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        induction_support = int(payload.get("induction_support") or 0)
        total_support = int(payload.get("total_support") or 0)
        payload["support"] = (
            induction_support if target == "induction_v2" else total_support
        )
        for metric in ("mean_induction", "mean_composite", "s1_rate"):
            payload[metric] = _safe_float(payload.get(metric))
        out.append(payload)
    return out


def _aggregate_categories(
    conn: sqlite3.Connection, target: str
) -> list[dict[str, Any]]:
    cols = _table_columns(conn, "op_observations")
    if "op_category" not in cols:
        return []
    induction_expr = _induction_metric_expr(cols, target)
    rows = conn.execute(
        f"""
        SELECT
            op_category AS key,
            COUNT({induction_expr}) AS induction_support,
            COUNT(*) AS total_support,
            AVG({induction_expr}) AS mean_induction,
            AVG({_metric_expr(cols, "composite_score")}) AS mean_composite,
            AVG({_metric_expr(cols, "stage1_passed")}) AS s1_rate
        FROM op_observations
        WHERE COALESCE(op_category, '') <> ''
        GROUP BY op_category
        """
    ).fetchall()
    return [
        {
            "key": str(row["key"]),
            "support": int(
                (
                    row["induction_support"]
                    if target == "induction_v2"
                    else row["total_support"]
                )
                or 0
            ),
            "mean_induction": _safe_float(row["mean_induction"]),
            "mean_composite": _safe_float(row["mean_composite"]),
            "s1_rate": _safe_float(row["s1_rate"]),
        }
        for row in rows
    ]


def _build_category_weights(
    category_rows: list[dict[str, Any]],
    global_stats: dict[str, float | None],
    target: str,
    min_support: int,
) -> dict[str, float]:
    weights = dict(_CATEGORY_POLICY[target])
    for row in category_rows:
        category = str(row.get("key") or "").strip()
        support = int(row.get("support") or 0)
        if not category or support < min_support:
            continue
        score = _score_row(row, global_stats, target)
        learned = _clamp(
            _multiplier_from_score(score, support, min_support, 0.50), 0.50, 2.50
        )
        current = weights.get(category)
        merged = learned if current is None else math.sqrt(float(current) * learned)
        if category == "math_space" and target in {"induction", "induction_v2"}:
            merged = min(merged, 1.00)
        elif category == "math_space" and target == "balanced":
            merged = min(merged, 1.05)
        weights[category] = _round_weight(merged)
    return {k: _round_weight(v) for k, v in sorted(weights.items())}


def _build_op_weights(
    op_rows: list[dict[str, Any]],
    global_stats: dict[str, float | None],
    target: str,
    min_support: int,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in op_rows:
        op_name = str(row.get("key") or "").strip()
        support = int(row.get("support") or 0)
        if not op_name or support < min_support:
            continue
        score = _score_row(row, global_stats, target)
        learned = _clamp(
            _multiplier_from_score(score, support, min_support, 0.75), 0.35, 3.00
        )
        if abs(learned - 1.0) >= 0.08:
            weights[op_name] = _round_weight(learned)
    for op_name, seeded in _SEEDED_OP_WEIGHTS[target].items():
        weights[op_name] = _round_weight(max(float(weights.get(op_name, 1.0)), seeded))
    return dict(sorted(weights.items()))


def _build_template_weights(
    template_rows: list[dict[str, Any]],
    global_stats: dict[str, float | None],
    target: str,
    min_support: int,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    template_min_support = max(3, min_support // 2)
    for row in template_rows:
        template_name = str(row.get("key") or "").strip()
        support = int(row.get("support") or 0)
        if not template_name or support < template_min_support:
            continue
        score = _score_row(row, global_stats, target)
        learned = _clamp(
            _multiplier_from_score(score, support, template_min_support, 0.50),
            0.50,
            2.00,
        )
        if abs(learned - 1.0) >= 0.06:
            weights[template_name] = _round_weight(learned)
    return dict(sorted(weights.items()))


def _build_probe_queue(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    cols = _table_columns(conn, "op_property_catalog")
    if "op_name" not in cols:
        return []
    needed_expr = (
        "op_empirical_probe_needed" if "op_empirical_probe_needed" in cols else "0"
    )
    category_expr = "op_category" if "op_category" in cols else "NULL"
    rows = conn.execute(
        f"""
        SELECT
            op_name,
            observed_count,
            eval_count,
            {category_expr} AS op_category,
            {needed_expr} AS probe_needed
        FROM op_property_catalog
        WHERE CAST({needed_expr} AS INTEGER) = 1
        ORDER BY observed_count DESC, eval_count ASC, op_name ASC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [
        {
            "op_name": str(row["op_name"]),
            "op_category": str(row["op_category"] or "unknown"),
            "observed_count": int(row["observed_count"] or 0),
            "eval_count": int(row["eval_count"] or 0),
            "reason": "static metadata marks this op as needing empirical numerical/Jacobian probes",
        }
        for row in rows
    ]


def build_meta_analysis_prior(
    *,
    meta_db_path: str | Path = DEFAULT_META_ANALYSIS_DB,
    target: str = "balanced",
    min_support: int = 100,
    probe_queue_limit: int = 32,
    created_at: float | None = None,
) -> dict[str, Any]:
    """Build a compact generation prior from the standalone meta-analysis DB."""

    if target not in VALID_TARGETS:
        raise ValueError(
            f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}"
        )
    created = float(time.time() if created_at is None else created_at)
    meta_path = Path(meta_db_path)
    conn = _connect_readonly(meta_path)
    try:
        global_stats = _global_stats(conn, target)
        op_rows = _aggregate_table(
            conn,
            "op_observations",
            "op_name",
            extra_columns=(
                "op_category",
                "op_lambda_calculus_affinity",
                "op_alternative_math_affinity",
            ),
            target=target,
        )
        template_rows = _aggregate_table(
            conn,
            "template_observations",
            "template_name",
            target=target,
        )
        category_rows = _aggregate_categories(conn, target)
        category_weights = _build_category_weights(
            category_rows,
            global_stats,
            target,
            min_support,
        )
        op_weights = _build_op_weights(op_rows, global_stats, target, min_support)
        template_weights = _build_template_weights(
            template_rows,
            global_stats,
            target,
            min_support,
        )
        probe_queue = _build_probe_queue(conn, probe_queue_limit)
    finally:
        conn.close()

    high_lambda = [
        row["key"]
        for row in sorted(
            op_rows,
            key=lambda r: (
                _safe_float(r.get("mean_induction")) or 0.0,
                int(r.get("support") or 0),
            ),
            reverse=True,
        )
        if int(row.get("support") or 0) >= min_support
        and _safe_float(row.get("op_lambda_calculus_affinity")) is not None
        and float(row.get("op_lambda_calculus_affinity") or 0.0) >= 0.5
    ][:12]

    return {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "version": f"meta_prior_{target}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime(created))}",
        "created_at": created,
        "target": target,
        "source_db": str(meta_path),
        "min_support": int(min_support),
        "global_stats": global_stats,
        "category_weights": category_weights,
        "op_weights": op_weights,
        "template_weights": template_weights,
        "slot_motif_weight_multipliers": {},
        "slot_motif_denylist": {},
        "probe_queue": probe_queue,
        "rationale": [
            "Boost frequency, functional, and mixing primitives for induction-oriented search.",
            "Keep broad math_space restrained unless a specific op has observed support.",
            "Use selected op boosts for rope/spectral/entropy primitives that over-indexed in meta-analysis.",
            "Leave slot motif priors empty in v1; existing construction priors still own slot-level motif evidence.",
        ],
        "signals": {
            "n_op_rows": len(op_rows),
            "n_template_rows": len(template_rows),
            "n_category_rows": len(category_rows),
            "target_metric": (
                "induction_v2_investigation_auc"
                if target == "induction_v2"
                else "coalesce(induction_v2_investigation_auc, induction_auc)"
                if target in {"induction", "balanced"}
                else "composite_score"
            ),
            "high_lambda_affinity_supported_ops": high_lambda,
        },
    }


def write_meta_analysis_prior(
    prior: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_PRIOR_DIR,
) -> Path:
    """Write a versioned prior artifact and update the target-specific latest file."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = str(prior.get("target") or "balanced")
    version = str(prior.get("version") or f"meta_prior_{target}_{int(time.time())}")
    path = out_dir / f"{version}.json"
    payload = json.dumps(prior, indent=2, sort_keys=True)
    path.write_text(payload + "\n")
    (out_dir / f"latest_{target}.json").write_text(payload + "\n")
    return path


def load_latest_meta_analysis_prior(
    path_or_dir: str | Path = DEFAULT_PRIOR_DIR,
    *,
    target: str = "balanced",
) -> dict[str, Any] | None:
    """Load a prior JSON file or the latest artifact in a prior directory."""

    path = Path(path_or_dir)
    if path.is_file():
        return json.loads(path.read_text())
    latest = path / f"latest_{target}.json"
    if latest.exists():
        return json.loads(latest.read_text())
    candidates = sorted(
        path.glob(f"meta_prior_{target}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


def meta_analysis_prior_as_grammar_adjustments(prior: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of a prior that grammar generation consumes."""

    return {
        "version": prior.get("version"),
        "target": prior.get("target"),
        "category_weights": dict(prior.get("category_weights") or {}),
        "op_weights": dict(prior.get("op_weights") or {}),
        "template_weights": dict(prior.get("template_weights") or {}),
        "slot_motif_multipliers": dict(
            prior.get("slot_motif_weight_multipliers") or {}
        ),
        "slot_motif_denylist": dict(prior.get("slot_motif_denylist") or {}),
    }


def apply_meta_analysis_prior_to_grammar(
    grammar: Any, prior: dict[str, Any]
) -> dict[str, int]:
    """Apply grammar prior multipliers in place and return application counts."""

    adjustments = meta_analysis_prior_as_grammar_adjustments(prior)
    counts = {
        "category_weights": 0,
        "op_weights": 0,
        "template_weights": 0,
        "slot_motif_multipliers": 0,
        "slot_motif_denylist": 0,
    }
    for category, multiplier in adjustments["category_weights"].items():
        try:
            mult = float(multiplier)
        except (TypeError, ValueError):
            continue
        base = float(grammar.category_weights.get(str(category), 1.0))
        grammar.category_weights[str(category)] = _round_weight(
            _clamp(base * mult, 0.05, 8.0)
        )
        counts["category_weights"] += 1

    for op_name, multiplier in adjustments["op_weights"].items():
        try:
            mult = float(multiplier)
        except (TypeError, ValueError):
            continue
        base = float(grammar.op_weights.get(str(op_name), 1.0))
        grammar.op_weights[str(op_name)] = _round_weight(_clamp(base * mult, 0.05, 8.0))
        counts["op_weights"] += 1

    for template_name, multiplier in adjustments["template_weights"].items():
        try:
            mult = float(multiplier)
        except (TypeError, ValueError):
            continue
        base = float(grammar.template_weights.get(str(template_name), 1.0))
        grammar.template_weights[str(template_name)] = _round_weight(
            _clamp(base * mult, 0.05, 5.0)
        )
        counts["template_weights"] += 1

    for slot_key, weights in adjustments["slot_motif_multipliers"].items():
        merged = dict(grammar.slot_motif_weight_multipliers.get(str(slot_key), {}))
        for motif_name, multiplier in (weights or {}).items():
            try:
                mult = float(multiplier)
            except (TypeError, ValueError):
                continue
            current = float(merged.get(str(motif_name), 1.0))
            merged[str(motif_name)] = _round_weight(_clamp(current * mult, 0.05, 5.0))
            counts["slot_motif_multipliers"] += 1
        grammar.slot_motif_weight_multipliers[str(slot_key)] = merged

    for slot_key, denied in adjustments["slot_motif_denylist"].items():
        existing = set(grammar.slot_motif_denylist.get(str(slot_key), frozenset()))
        before = len(existing)
        existing.update(str(name) for name in (denied or []) if str(name).strip())
        if existing:
            grammar.slot_motif_denylist[str(slot_key)] = frozenset(existing)
        counts["slot_motif_denylist"] += max(0, len(existing) - before)
    return counts
