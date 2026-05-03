"""Shared observability computations used by the Flask blueprint."""

from __future__ import annotations

import logging
import math as _math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import fast_dumps, fast_loads
from ..native.core import _try_import_rust_scheduler
from .deps import get_notebook

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "s0_pass_rate_min": 0.30,
    "s1_pass_rate_min": 0.05,
    "grad_norm_max": 50000.0,
    "routing_collapse_score_min": 0.3,
    "op_failure_rate_max": 0.90,
    "stale_experiment_hours": 6,
}

_WINDOW_SECONDS: Dict[str, Optional[int]] = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
    "all": None,
}

_CONTROLLED_LANG_AVG_FIELDS = (
    "avg_controlled_lang_s05_sa_score",
    "avg_controlled_lang_s05_nb_order_acc",
    "avg_controlled_lang_s05_nb_score",
    "avg_controlled_lang_s10_sa_score",
    "avg_controlled_lang_s10_nb_order_acc",
    "avg_controlled_lang_s10_nb_score",
    "avg_controlled_lang_inv_sa_score",
    "avg_controlled_lang_inv_nb_order_acc",
    "avg_controlled_lang_inv_nb_score",
)

_CONTROLLED_LANG_SCORE_FIELDS = (
    "avg_controlled_lang_s05_score",
    "avg_controlled_lang_s10_score",
    "avg_controlled_lang_inv_score",
)

_health_cache: Dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 120.0
_op_index_caches: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_op_index_lock = threading.Lock()
_OP_INDEX_TTL = 300.0
_throughput_cache: Optional[Dict[str, Any]] = None
_throughput_cache_ts: float = 0.0
_THROUGHPUT_TTL = 60.0
_alerts_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}
_ALERTS_TTL = 5.0
_alerts_lock = threading.Lock()


def refresh_observability_caches() -> None:
    global _health_cache_ts, _throughput_cache_ts, _throughput_cache
    _health_cache_ts = 0.0
    _throughput_cache_ts = 0.0
    _throughput_cache = None
    _op_index_caches.clear()
    with _alerts_lock:
        _alerts_cache.clear()


def _build_op_index_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_counts": {
            (entry["op_a"], entry["op_b"]): {
                "n": int(entry["n"]),
                "s0": int(entry["s0"]),
                "s1": int(entry["s1"]),
            }
            for entry in payload.get("pair_counts", [])
        },
        "loss_by_op": {
            entry["op"]: [float(value) for value in entry.get("values", [])]
            for entry in payload.get("loss_by_op", [])
        },
        "failure_groups": {
            entry["name"]: {
                "ops": {
                    op_entry["op"]: int(op_entry["count"])
                    for op_entry in entry.get("ops", [])
                },
                "count": int(entry["count"]),
            }
            for entry in payload.get("failure_groups", [])
        },
        "stored_rates": {
            entry["op"]: {
                "n": int(entry["n"]),
                "s0": int(entry["s0"]),
                "s1": int(entry["s1"]),
            }
            for entry in payload.get("stored_rates", [])
        },
        "corrected_rates": {
            entry["op"]: {
                "n": int(entry["n"]),
                "s0": int(entry["s0"]),
                "s1": int(entry["s1"]),
                "excluded": int(entry["excluded"]),
            }
            for entry in payload.get("corrected_rates", [])
        },
    }


def _load_program_rows(nb, window: str) -> list[dict[str, Any]]:
    cutoff = None
    window_seconds = _WINDOW_SECONDS.get(window)
    if window_seconds is not None:
        cutoff = time.time() - window_seconds

    query = (
        "SELECT graph_json, stage0_passed, stage1_passed, loss_ratio, "
        "error_type, failure_op, failure_details_json "
        "FROM program_results "
        "WHERE graph_json IS NOT NULL AND length(graph_json) > 0"
    )
    params: tuple[Any, ...] = ()
    if cutoff is not None:
        query += " AND timestamp > ?"
        params = (cutoff,)
    try:
        cursor = nb.conn.execute(query, params)
    except sqlite3.OperationalError as exc:
        logger.error("Failed to load observability program rows: %s", exc)
        return []
    payload_rows: list[dict[str, Any]] = []
    for row in cursor:
        graph_json = row["graph_json"]
        if not isinstance(graph_json, str) or not graph_json:
            continue
        loss_ratio = row["loss_ratio"]
        payload_rows.append(
            {
                "graph_json": graph_json,
                "stage0_passed": bool(row["stage0_passed"]),
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": float(loss_ratio) if loss_ratio is not None else None,
                "error_type": row["error_type"],
                "failure_op": row["failure_op"],
                "failure_details_json": row["failure_details_json"],
            }
        )
    return payload_rows


def build_op_index(notebook_path: str, window: str = "all") -> Dict[str, Any]:
    now = time.monotonic()
    cache_key = (notebook_path, window)
    cached = _op_index_caches.get(cache_key)
    if cached and (now - cached[0]) < _OP_INDEX_TTL:
        return cached[1]
    with _op_index_lock:
        now = time.monotonic()
        cached = _op_index_caches.get(cache_key)
        if cached and (now - cached[0]) < _OP_INDEX_TTL:
            return cached[1]

        rust = _try_import_rust_scheduler()
        if rust is None or not hasattr(rust, "build_op_index_from_rows"):
            raise RuntimeError("aria_scheduler.build_op_index_from_rows is required")

        nb = get_notebook(notebook_path, read_only=True)
        payload_rows = _load_program_rows(nb, window)
        payload = fast_loads(rust.build_op_index_from_rows(fast_dumps(payload_rows)))
        result = _build_op_index_result(payload)
        _op_index_caches[cache_key] = (now, result)
        return result


def get_throughput(notebook_path: str) -> Dict[str, Any]:
    global _throughput_cache, _throughput_cache_ts
    now_mono = time.monotonic()
    if _throughput_cache and (now_mono - _throughput_cache_ts) < _THROUGHPUT_TTL:
        return _throughput_cache

    now = time.time()
    windows = {"1h": 3600, "6h": 21600, "24h": 86400}
    result: Dict[str, Any] = {}
    nb = get_notebook(notebook_path, read_only=True)
    try:
        for label, seconds in windows.items():
            cutoff = now - seconds
            row = nb.conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) as s0, "
                "SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1 "
                "FROM program_results WHERE timestamp > ?",
                (cutoff,),
            ).fetchone()
            result[label] = {
                "total": row["total"] or 0,
                "s0_passed": row["s0"] or 0,
                "s1_passed": row["s1"] or 0,
                "s0_rate": round((row["s0"] or 0) / max(row["total"] or 1, 1), 3),
                "s1_rate": round((row["s1"] or 0) / max(row["s0"] or 1, 1), 3)
                if (row["s0"] or 0) > 0
                else 0.0,
            }
    except sqlite3.OperationalError as exc:
        logger.error("Throughput query error: %s", exc)

    result["computed_at"] = now
    _throughput_cache = result
    _throughput_cache_ts = now_mono
    return result


def _load_op_rates(nb, window: str) -> list[dict[str, Any]]:
    try:
        window_seconds = _WINDOW_SECONDS.get(window)
        if window_seconds is not None:
            return nb.get_op_success_rates_windowed(time.time() - window_seconds)
        return nb.get_op_success_rates()
    except sqlite3.OperationalError as exc:
        logger.debug("op_success_rates query failed: %s", exc)
        return []


def _load_grad_health() -> Dict[str, Dict[str, Any]]:
    grad_health: Dict[str, Dict[str, Any]] = {}
    try:
        profiling_db = Path("research/profiling/component_profiles.db")
        if not profiling_db.exists():
            return grad_health
        conn = sqlite3.connect(str(profiling_db), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT op_name, grad_norm, grad_exploding, grad_vanishing, "
                "output_has_nan, output_has_inf, forward_time_us, backward_time_us, "
                "lipschitz_estimate, error FROM op_profiles"
            )
            for row in cursor:
                grad_health[row["op_name"]] = {
                    "grad_norm": float(row["grad_norm"])
                    if row["grad_norm"] is not None
                    else None,
                    "grad_exploding": bool(row["grad_exploding"])
                    if row["grad_exploding"] is not None
                    else False,
                    "grad_vanishing": bool(row["grad_vanishing"])
                    if row["grad_vanishing"] is not None
                    else False,
                    "has_nan": bool(row["output_has_nan"])
                    if row["output_has_nan"] is not None
                    else False,
                    "has_inf": bool(row["output_has_inf"])
                    if row["output_has_inf"] is not None
                    else False,
                    "fwd_us": float(row["forward_time_us"])
                    if row["forward_time_us"] is not None
                    else None,
                    "bwd_us": float(row["backward_time_us"])
                    if row["backward_time_us"] is not None
                    else None,
                    "lipschitz": float(row["lipschitz_estimate"])
                    if row["lipschitz_estimate"] is not None
                    else None,
                    "profile_error": row["error"],
                }
        finally:
            conn.close()
    except (sqlite3.OperationalError, KeyError, TypeError, OSError) as exc:
        logger.debug(
            "Failed to load component profiling health data: %s", exc, exc_info=True
        )
    return grad_health


def _compute_blame(
    max_n_used: int, n_used: int, n_s0: int
) -> Tuple[float, float, float]:
    if n_used == 0 or max_n_used == 0:
        return 0.0, 0.0, 0.0
    tf = 1.0 - (n_s0 / n_used)
    idf = _math.log(max(max_n_used, 1) / n_used) if n_used < max_n_used else 0.0
    return tf * idf, tf, idf


def _build_structural_component(
    op: str,
    reasons: list[str],
    n_used: int,
    n_s0: int,
    n_s05: int,
    n_s1: int,
    s0_rate: float,
    s1_rate: float,
    blame: float,
    raw_blame: float,
    n_excluded: int,
    grad_norm: Any,
    lipschitz: float,
    prof: dict[str, Any],
) -> dict[str, Any]:
    return {
        "op": op,
        "status": "structural",
        "reasons": reasons,
        "n_used": n_used,
        "n_s0": n_s0,
        "n_s05": n_s05,
        "n_s1": n_s1,
        "s0_rate": round(s0_rate, 4),
        "s1_rate": round(s1_rate, 4),
        "blame": round(blame, 3),
        "raw_blame": round(raw_blame, 3),
        "n_excluded": n_excluded,
        "grad_norm": grad_norm,
        "lipschitz": lipschitz,
        "data_source": "search+profiling" if prof else "search",
    }


def _classify_component_status(
    s1_rate: float,
    n_s0: int,
    raw_n: int,
    blame: float,
    tf: float,
    idf: float,
    prof: dict[str, Any],
) -> tuple[str, list[str], Any]:
    grad_norm = prof.get("grad_norm")
    has_nan = prof.get("has_nan", False)
    lipschitz = prof.get("lipschitz") or 0.0
    profile_clean = (
        bool(prof)
        and not has_nan
        and not prof.get("has_inf", False)
        and not prof.get("profile_error")
        and (grad_norm is None or grad_norm < 5000)
        and lipschitz <= 1.01
    )
    redeemed = s1_rate > 0.5 or profile_clean
    confidence = min(raw_n / 50.0, 1.0)
    broken_threshold = 2.0 + 2.0 * (1.0 - confidence)
    degraded_threshold = broken_threshold / 2.0
    status = "healthy"
    reasons: list[str] = []

    if has_nan or prof.get("has_inf", False):
        return "broken", ["NaN/Inf in output"], grad_norm
    if prof.get("profile_error"):
        return "broken", [f"profile error: {prof['profile_error'][:60]}"], grad_norm
    if blame > broken_threshold and raw_n >= 20 and not redeemed:
        return (
            "broken",
            [
                f"TF-IDF blame={blame:.2f} (fail_rate={tf:.0%}, rarity={idf:.1f}, n={raw_n})"
            ],
            grad_norm,
        )
    if grad_norm is not None and grad_norm > 50000:
        status = "degraded"
        reasons.append(f"grad_norm={grad_norm:.0f}")
    elif blame > degraded_threshold and raw_n >= 10 and not redeemed:
        status = "degraded"
        reasons.append(
            f"TF-IDF blame={blame:.2f} (fail_rate={tf:.0%}, rarity={idf:.1f}, n={raw_n})"
        )
    elif lipschitz > 2.0:
        status = "degraded"
        reasons.append(f"gradient amplifier (lipschitz={lipschitz:.1f})")
    elif s1_rate < 0.05 and n_s0 >= 10:
        status = "degraded"
        reasons.append(f"S1 pass rate {s1_rate:.0%}")
    return status, reasons, grad_norm


def _component_raw_counts(
    row: dict[str, Any], raw: dict[str, int] | None
) -> tuple[int, int, int]:
    has_stored_rates = raw is not None and raw["n"] > 0
    return (
        int(raw["n"]) if has_stored_rates else int(row.get("n_used") or 0),
        int(raw["s0"]) if has_stored_rates else int(row.get("n_stage0_passed") or 0),
        int(raw["s1"]) if has_stored_rates else int(row.get("n_stage1_passed") or 0),
    )


def _component_display_counts(
    corrected: dict[str, int] | None,
    raw_n: int,
    raw_s0: int,
    raw_s1: int,
) -> tuple[int, int, int]:
    if corrected is None:
        return raw_n, raw_s0, raw_s1
    return int(corrected["n"]), int(corrected["s0"]), int(corrected["s1"])


def _component_exclusion_reason(
    corrected: dict[str, int] | None, n_used: int
) -> str | None:
    if corrected is None or corrected["excluded"] <= 0:
        return None
    n_excluded = corrected["excluded"]
    if n_used == 0:
        return f"excluded {n_excluded} runtime-only failures"
    return f"excluded {n_excluded} runtime-only failures from displayed rates"


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if _math.isfinite(number) else None


def _rounded_metric(value: Any, digits: int = 3) -> float | None:
    number = _finite_float_or_none(value)
    return round(number, digits) if number is not None else None


def _controlled_lang_display_score(
    sa_score: Any,
    nb_order_acc: Any,
    nb_score: Any,
) -> float | None:
    values = [
        value
        for value in (
            _finite_float_or_none(sa_score),
            _finite_float_or_none(nb_score)
            if _finite_float_or_none(nb_score) is not None
            else _finite_float_or_none(nb_order_acc),
        )
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def _component_controlled_lang_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = {field: row.get(field) for field in _CONTROLLED_LANG_AVG_FIELDS}
    metrics["avg_controlled_lang_s05_score"] = _controlled_lang_display_score(
        metrics.get("avg_controlled_lang_s05_sa_score"),
        metrics.get("avg_controlled_lang_s05_nb_order_acc"),
        metrics.get("avg_controlled_lang_s05_nb_score"),
    )
    metrics["avg_controlled_lang_s10_score"] = _controlled_lang_display_score(
        metrics.get("avg_controlled_lang_s10_sa_score"),
        metrics.get("avg_controlled_lang_s10_nb_order_acc"),
        metrics.get("avg_controlled_lang_s10_nb_score"),
    )
    metrics["avg_controlled_lang_inv_score"] = _controlled_lang_display_score(
        metrics.get("avg_controlled_lang_inv_sa_score"),
        metrics.get("avg_controlled_lang_inv_nb_order_acc"),
        metrics.get("avg_controlled_lang_inv_nb_score"),
    )
    return metrics


def _attach_component_observability_metrics(
    component: dict[str, Any],
    row: dict[str, Any],
    overlay: dict[str, Any] | None,
    *,
    n_used: int,
    n_s05: int,
) -> dict[str, Any]:
    overlay = overlay or {}
    component["n_s05"] = int(n_s05 or 0)
    component["s05_rate"] = round((n_s05 / n_used), 3) if n_used > 0 else None
    component["avg_loss_ratio"] = _rounded_metric(
        overlay.get("avg_loss_ratio", row.get("avg_loss_ratio"))
    )
    component["avg_validation_loss_ratio"] = _rounded_metric(
        overlay.get("avg_validation_loss_ratio")
    )
    component["avg_induction_auc"] = _rounded_metric(overlay.get("avg_induction_auc"))
    component["avg_binding_auc"] = _rounded_metric(overlay.get("avg_binding_auc"))
    component["avg_induction_v2_auc"] = _rounded_metric(
        overlay.get("avg_induction_v2_auc")
    )
    component["avg_binding_v2_auc"] = _rounded_metric(overlay.get("avg_binding_v2_auc"))
    component["avg_hellaswag_acc"] = _rounded_metric(overlay.get("avg_hellaswag_acc"))
    component["avg_blimp_overall_accuracy"] = _rounded_metric(
        overlay.get("avg_blimp_overall_accuracy")
    )
    for key, value in _component_controlled_lang_metrics(overlay).items():
        component[key] = _rounded_metric(value)
    component["avg_composite_score"] = _rounded_metric(
        overlay.get("avg_composite_score")
    )
    component["avg_erf_density"] = _rounded_metric(overlay.get("avg_erf_density"))
    component["avg_id_collapse_rate"] = _rounded_metric(
        overlay.get("avg_id_collapse_rate")
    )
    component["avg_id_collapse_rate_normalized"] = _rounded_metric(
        overlay.get("avg_id_collapse_rate_normalized")
    )
    component["avg_erf_decay_slope"] = _rounded_metric(
        overlay.get("avg_erf_decay_slope")
    )
    component["avg_erf_first_norm"] = _rounded_metric(overlay.get("avg_erf_first_norm"))
    component["avg_erf_last_norm"] = _rounded_metric(overlay.get("avg_erf_last_norm"))
    component["avg_logit_margin_velocity"] = _rounded_metric(
        overlay.get("avg_logit_margin_velocity")
    )
    component["avg_logit_margin_delta"] = _rounded_metric(
        overlay.get("avg_logit_margin_delta")
    )
    component["avg_erf_variance_log"] = _rounded_metric(
        overlay.get("avg_erf_variance_log")
    )
    component["avg_spec_norm_log"] = _rounded_metric(overlay.get("avg_spec_norm_log"))
    component["avg_icld_velocity"] = _rounded_metric(overlay.get("avg_icld_velocity"))
    component["avg_icld_delta_loss"] = _rounded_metric(
        overlay.get("avg_icld_delta_loss")
    )
    component["avg_jacobian_effective_rank"] = _rounded_metric(
        overlay.get("avg_jacobian_effective_rank")
    )
    component["avg_sensitivity_uniformity"] = _rounded_metric(
        overlay.get("avg_sensitivity_uniformity")
    )
    component["top_failure_reason"] = overlay.get("top_failure_reason")
    return component


def _build_search_component_entry(
    op: str,
    status: str,
    reasons: list[str],
    counts: dict[str, Any],
    metrics: dict[str, Any],
    prof: dict[str, Any],
) -> dict[str, Any]:
    grad_norm = metrics["grad_norm"]
    lipschitz = metrics["lipschitz"]
    return {
        "op": op,
        "status": status,
        "reasons": reasons,
        "n_used": counts["n_used"],
        "s0_rate": round(metrics["s0_rate"], 3)
        if metrics["s0_rate"] is not None
        else None,
        "s1_rate": round(metrics["s1_rate"], 3)
        if metrics["s1_rate"] is not None
        else None,
        "blame": round(metrics["blame"], 3),
        "fail_rate": round(metrics["tf"], 3),
        "rarity": round(metrics["idf"], 3),
        "raw_blame": round(metrics["raw_blame"], 3),
        "raw_fail_rate": round(metrics["raw_tf"], 3),
        "raw_n_used": counts["raw_n"],
        "raw_s0_rate": round(counts["raw_s0"] / counts["raw_n"], 3)
        if counts["raw_n"] > 0
        else None,
        "raw_s1_rate": round(counts["raw_s1"] / counts["raw_s0"], 3)
        if counts["raw_s0"] > 0
        else None,
        "n_excluded": counts["n_excluded"],
        "lipschitz": round(lipschitz, 2) if lipschitz else None,
        "grad_norm": round(grad_norm, 1) if grad_norm is not None else None,
        "has_nan": prof.get("has_nan", False),
        "fwd_us": prof.get("fwd_us"),
        "bwd_us": prof.get("bwd_us"),
        "data_source": "search+profiling" if prof else "search",
    }


def _build_component_entry(
    row: dict[str, Any],
    stored_rates: Dict[str, Dict[str, int]],
    corrected_rates: Dict[str, Dict[str, int]],
    grad_health: Dict[str, Dict[str, Any]],
    metric_overlays: Dict[str, Dict[str, Any]],
    max_n_used: int,
) -> dict[str, Any]:
    from research.synthesis.context_rules import S1_EXEMPT_OPS

    op = row["op_name"]
    overlay = metric_overlays.get(op, {})
    prof = grad_health.get(op, {})
    raw = stored_rates.get(op)
    raw_n, raw_s0, raw_s1 = _component_raw_counts(row, raw)
    raw_blame, raw_tf, _ = _compute_blame(max_n_used, raw_n, raw_s0)
    corrected = corrected_rates.get(op)
    if corrected and corrected["n"] > 0:
        blame, tf, idf = _compute_blame(max_n_used, corrected["n"], corrected["s0"])
    else:
        blame, tf, idf = _compute_blame(max_n_used, raw_n, raw_s0)
    n_used, n_s0, n_s1 = _component_display_counts(corrected, raw_n, raw_s0, raw_s1)
    raw_s05 = int(row.get("n_stage05_passed") or 0)
    n_s05 = min(raw_s05, n_used)
    s0_rate = (n_s0 / n_used) if n_used > 0 else None
    s1_rate = (n_s1 / n_s0) if n_s0 > 0 else None
    n_excluded = corrected["excluded"] if corrected else 0
    lipschitz = prof.get("lipschitz") or 0.0
    exclusion_reason = _component_exclusion_reason(corrected, n_used)

    if op in S1_EXEMPT_OPS:
        reasons = ["scaffolding op — not a standalone learner"]
        if exclusion_reason:
            reasons.append(exclusion_reason)
        entry = _build_structural_component(
            op,
            reasons,
            n_used,
            n_s0,
            n_s05,
            n_s1,
            float(s0_rate or 0.0),
            float(s1_rate or 0.0),
            blame,
            raw_blame,
            n_excluded,
            prof.get("grad_norm"),
            lipschitz,
            prof,
        )
        return _attach_component_observability_metrics(
            entry,
            row,
            overlay,
            n_used=n_used,
            n_s05=n_s05,
        )

    status, reasons, grad_norm = _classify_component_status(
        float(s1_rate or 0.0),
        n_s0,
        raw_n,
        blame,
        tf,
        idf,
        prof,
    )
    if exclusion_reason:
        reasons = [*reasons, exclusion_reason]
    entry = _build_search_component_entry(
        op,
        status,
        reasons,
        {
            "n_used": n_used,
            "raw_n": raw_n,
            "raw_s0": raw_s0,
            "raw_s1": raw_s1,
            "n_excluded": n_excluded,
        },
        {
            "s0_rate": s0_rate,
            "s1_rate": s1_rate,
            "blame": blame,
            "tf": tf,
            "idf": idf,
            "raw_blame": raw_blame,
            "raw_tf": raw_tf,
            "lipschitz": lipschitz,
            "grad_norm": grad_norm,
        },
        prof,
    )
    return _attach_component_observability_metrics(
        entry,
        row,
        overlay,
        n_used=n_used,
        n_s05=n_s05,
    )


def _build_profile_only_component(op_name: str, prof: dict[str, Any]) -> dict[str, Any]:
    status = "healthy"
    reasons: list[str] = []
    if prof.get("has_nan") or prof.get("has_inf"):
        status = "broken"
        reasons.append("NaN/Inf in profiling")
    elif prof.get("profile_error"):
        status = "broken"
        reasons.append("profile error")
    elif prof.get("grad_norm") and prof["grad_norm"] > 50000:
        status = "degraded"
        reasons.append(f"grad_norm={prof['grad_norm']:.0f}")
    return {
        "op": op_name,
        "status": status,
        "reasons": reasons,
        "n_used": 0,
        "s0_rate": None,
        "s05_rate": None,
        "s1_rate": None,
        "avg_loss_ratio": None,
        "avg_validation_loss_ratio": None,
        "avg_induction_auc": None,
        "avg_binding_auc": None,
        "avg_induction_v2_auc": None,
        "avg_binding_v2_auc": None,
        "avg_hellaswag_acc": None,
        "avg_blimp_overall_accuracy": None,
        **{key: None for key in _CONTROLLED_LANG_AVG_FIELDS},
        **{key: None for key in _CONTROLLED_LANG_SCORE_FIELDS},
        "avg_composite_score": None,
        "avg_erf_density": None,
        "avg_id_collapse_rate": None,
        "avg_id_collapse_rate_normalized": None,
        "avg_erf_decay_slope": None,
        "avg_erf_first_norm": None,
        "avg_erf_last_norm": None,
        "avg_logit_margin_velocity": None,
        "avg_logit_margin_delta": None,
        "avg_erf_variance_log": None,
        "avg_spec_norm_log": None,
        "avg_icld_velocity": None,
        "avg_icld_delta_loss": None,
        "avg_jacobian_effective_rank": None,
        "avg_sensitivity_uniformity": None,
        "top_failure_reason": None,
        "grad_norm": round(prof["grad_norm"], 1)
        if prof.get("grad_norm") is not None
        else None,
        "grad_exploding": prof.get("grad_exploding", False),
        "has_nan": prof.get("has_nan", False),
        "fwd_us": prof.get("fwd_us"),
        "bwd_us": prof.get("bwd_us"),
        "data_source": "profiling_only",
    }


def _component_metric_where(window: str) -> tuple[str, tuple[Any, ...]]:
    where = "gpo.op_name IS NOT NULL AND gpo.op_name <> '' AND gpo.op_name <> 'input'"
    params: tuple[Any, ...] = ()
    window_seconds = _WINDOW_SECONDS.get(window)
    if window_seconds is not None:
        where += " AND pr.timestamp > ?"
        params = (time.time() - window_seconds,)
    return where, params


def _load_component_metric_overlays(nb, window: str) -> Dict[str, Dict[str, Any]]:
    where, params = _component_metric_where(window)
    overlays: Dict[str, Dict[str, Any]] = {}
    try:
        rows = nb.conn.execute(
            f"""
            WITH op_rows AS (
                SELECT DISTINCT
                    pr.result_id AS result_id,
                    gpo.op_name AS op_name,
                    pr.loss_ratio AS loss_ratio,
                    pr.validation_loss_ratio AS validation_loss_ratio,
                    pr.induction_auc AS induction_auc,
                    pr.induction_v2_investigation_auc AS induction_v2_auc,
                    COALESCE(pr.binding_auc_curriculum, pr.binding_auc) AS binding_auc,
                    pr.binding_v2_investigation_auc AS binding_v2_auc,
                    COALESCE(
                        pr.hellaswag_acc,
                        CASE
                            WHEN pr.screening_hellaswag_total > 0
                            THEN CAST(pr.screening_hellaswag_correct AS REAL)
                                 / pr.screening_hellaswag_total
                            ELSE NULL
                        END
                    ) AS hellaswag_acc,
                    pr.blimp_overall_accuracy AS blimp_overall_accuracy,
                    pr.controlled_lang_s05_sa_score AS controlled_lang_s05_sa_score,
                    pr.controlled_lang_s05_nb_order_acc AS controlled_lang_s05_nb_order_acc,
                    pr.controlled_lang_s05_nb_score AS controlled_lang_s05_nb_score,
                    pr.controlled_lang_s10_sa_score AS controlled_lang_s10_sa_score,
                    pr.controlled_lang_s10_nb_order_acc AS controlled_lang_s10_nb_order_acc,
                    pr.controlled_lang_s10_nb_score AS controlled_lang_s10_nb_score,
                    pr.controlled_lang_inv_sa_score AS controlled_lang_inv_sa_score,
                    pr.controlled_lang_inv_nb_order_acc AS controlled_lang_inv_nb_order_acc,
                    pr.controlled_lang_inv_nb_score AS controlled_lang_inv_nb_score,
                    l.composite_score AS composite_score,
                    pr.fp_jacobian_effective_rank AS jacobian_effective_rank,
                    pr.fp_sensitivity_uniformity AS sensitivity_uniformity,
                    pr.fp_jacobian_erf_density AS erf_density,
                    pr.fp_id_collapse_rate AS id_collapse_rate,
                    pr.fp_id_collapse_rate_normalized AS id_collapse_rate_normalized,
                    pr.fp_jacobian_erf_decay_slope AS erf_decay_slope,
                    pr.fp_jacobian_erf_first_norm AS erf_first_norm,
                    pr.fp_jacobian_erf_last_norm AS erf_last_norm,
                    pr.fp_logit_margin_velocity AS logit_margin_velocity,
                    pr.fp_logit_margin_delta AS logit_margin_delta,
                    CASE WHEN pr.fp_jacobian_erf_variance IS NOT NULL
                         THEN log(abs(pr.fp_jacobian_erf_variance) + 0.000000001)
                         ELSE NULL
                    END AS erf_variance_log,
                    CASE WHEN pr.fp_jacobian_spectral_norm IS NOT NULL
                         THEN log(abs(pr.fp_jacobian_spectral_norm) + 0.000000001)
                         ELSE NULL
                    END AS spec_norm_log,
                    pr.fp_icld_velocity AS icld_velocity,
                    pr.fp_icld_delta_loss AS icld_delta_loss
                FROM program_results pr
                JOIN program_graph_ops gpo ON gpo.result_id = pr.result_id
                LEFT JOIN leaderboard l ON l.result_id = pr.result_id
                WHERE {where}
            )
            SELECT
                op_name,
                AVG(loss_ratio) AS avg_loss_ratio,
                AVG(validation_loss_ratio) AS avg_validation_loss_ratio,
                AVG(induction_auc) AS avg_induction_auc,
                AVG(binding_auc) AS avg_binding_auc,
                AVG(induction_v2_auc) AS avg_induction_v2_auc,
                AVG(binding_v2_auc) AS avg_binding_v2_auc,
                AVG(hellaswag_acc) AS avg_hellaswag_acc,
                AVG(blimp_overall_accuracy) AS avg_blimp_overall_accuracy,
                AVG(controlled_lang_s05_sa_score) AS avg_controlled_lang_s05_sa_score,
                AVG(controlled_lang_s05_nb_order_acc) AS avg_controlled_lang_s05_nb_order_acc,
                AVG(controlled_lang_s05_nb_score) AS avg_controlled_lang_s05_nb_score,
                AVG(controlled_lang_s10_sa_score) AS avg_controlled_lang_s10_sa_score,
                AVG(controlled_lang_s10_nb_order_acc) AS avg_controlled_lang_s10_nb_order_acc,
                AVG(controlled_lang_s10_nb_score) AS avg_controlled_lang_s10_nb_score,
                AVG(controlled_lang_inv_sa_score) AS avg_controlled_lang_inv_sa_score,
                AVG(controlled_lang_inv_nb_order_acc) AS avg_controlled_lang_inv_nb_order_acc,
                AVG(controlled_lang_inv_nb_score) AS avg_controlled_lang_inv_nb_score,
                AVG(composite_score) AS avg_composite_score,
                AVG(erf_density) AS avg_erf_density,
                AVG(id_collapse_rate) AS avg_id_collapse_rate,
                AVG(id_collapse_rate_normalized) AS avg_id_collapse_rate_normalized,
                AVG(erf_decay_slope) AS avg_erf_decay_slope,
                AVG(erf_first_norm) AS avg_erf_first_norm,
                AVG(erf_last_norm) AS avg_erf_last_norm,
                AVG(logit_margin_velocity) AS avg_logit_margin_velocity,
                AVG(logit_margin_delta) AS avg_logit_margin_delta,
                AVG(erf_variance_log) AS avg_erf_variance_log,
                AVG(spec_norm_log) AS avg_spec_norm_log,
                AVG(icld_velocity) AS avg_icld_velocity,
                AVG(icld_delta_loss) AS avg_icld_delta_loss,
                AVG(jacobian_effective_rank) AS avg_jacobian_effective_rank,
                AVG(sensitivity_uniformity) AS avg_sensitivity_uniformity
            FROM op_rows
            GROUP BY op_name
            """,
            params,
        ).fetchall()
        for row in rows:
            overlays[row["op_name"]] = {
                "avg_loss_ratio": row["avg_loss_ratio"],
                "avg_validation_loss_ratio": row["avg_validation_loss_ratio"],
                "avg_induction_auc": row["avg_induction_auc"],
                "avg_binding_auc": row["avg_binding_auc"],
                "avg_induction_v2_auc": row["avg_induction_v2_auc"],
                "avg_binding_v2_auc": row["avg_binding_v2_auc"],
                "avg_hellaswag_acc": row["avg_hellaswag_acc"],
                "avg_blimp_overall_accuracy": row["avg_blimp_overall_accuracy"],
                "avg_controlled_lang_s05_sa_score": row[
                    "avg_controlled_lang_s05_sa_score"
                ],
                "avg_controlled_lang_s05_nb_order_acc": row[
                    "avg_controlled_lang_s05_nb_order_acc"
                ],
                "avg_controlled_lang_s05_nb_score": row[
                    "avg_controlled_lang_s05_nb_score"
                ],
                "avg_controlled_lang_s10_sa_score": row[
                    "avg_controlled_lang_s10_sa_score"
                ],
                "avg_controlled_lang_s10_nb_order_acc": row[
                    "avg_controlled_lang_s10_nb_order_acc"
                ],
                "avg_controlled_lang_s10_nb_score": row[
                    "avg_controlled_lang_s10_nb_score"
                ],
                "avg_controlled_lang_inv_sa_score": row[
                    "avg_controlled_lang_inv_sa_score"
                ],
                "avg_controlled_lang_inv_nb_order_acc": row[
                    "avg_controlled_lang_inv_nb_order_acc"
                ],
                "avg_controlled_lang_inv_nb_score": row[
                    "avg_controlled_lang_inv_nb_score"
                ],
                "avg_composite_score": row["avg_composite_score"],
                "avg_erf_density": row["avg_erf_density"],
                "avg_id_collapse_rate": row["avg_id_collapse_rate"],
                "avg_id_collapse_rate_normalized": row[
                    "avg_id_collapse_rate_normalized"
                ],
                "avg_erf_decay_slope": row["avg_erf_decay_slope"],
                "avg_erf_first_norm": row["avg_erf_first_norm"],
                "avg_erf_last_norm": row["avg_erf_last_norm"],
                "avg_logit_margin_velocity": row["avg_logit_margin_velocity"],
                "avg_logit_margin_delta": row["avg_logit_margin_delta"],
                "avg_erf_variance_log": row["avg_erf_variance_log"],
                "avg_spec_norm_log": row["avg_spec_norm_log"],
                "avg_icld_velocity": row["avg_icld_velocity"],
                "avg_icld_delta_loss": row["avg_icld_delta_loss"],
                "avg_jacobian_effective_rank": row["avg_jacobian_effective_rank"],
                "avg_sensitivity_uniformity": row["avg_sensitivity_uniformity"],
            }

        reason_rows = nb.conn.execute(
            f"""
            WITH reason_rows AS (
                SELECT DISTINCT
                    pr.result_id AS result_id,
                    gpo.op_name AS op_name,
                    COALESCE(NULLIF(pr.error_type, ''), NULLIF(pr.stage_at_death, '')) AS reason
                FROM program_results pr
                JOIN program_graph_ops gpo ON gpo.result_id = pr.result_id
                WHERE {where} AND pr.stage1_passed = 0
            ),
            reason_counts AS (
                SELECT op_name, reason, COUNT(*) AS n
                FROM reason_rows
                WHERE reason IS NOT NULL AND reason <> ''
                GROUP BY op_name, reason
            ),
            ranked AS (
                SELECT
                    op_name,
                    reason,
                    ROW_NUMBER() OVER (
                        PARTITION BY op_name
                        ORDER BY n DESC, reason ASC
                    ) AS rn
                FROM reason_counts
            )
            SELECT op_name, reason
            FROM ranked
            WHERE rn = 1
            """,
            params,
        ).fetchall()
        for row in reason_rows:
            overlays.setdefault(row["op_name"], {})["top_failure_reason"] = row[
                "reason"
            ]
    except sqlite3.OperationalError as exc:
        logger.debug("component metric overlay query failed: %s", exc)
    return overlays


def get_component_health(notebook_path: str, window: str = "all") -> Dict[str, Any]:
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    if (
        _health_cache
        and (now - _health_cache_ts) < _HEALTH_CACHE_TTL
        and _health_cache.get("_window") == window
    ):
        return _health_cache

    nb = get_notebook(notebook_path, read_only=True)
    op_rates = _load_op_rates(nb, window)
    grad_health = _load_grad_health()
    metric_overlays = _load_component_metric_overlays(nb, window)
    idx = build_op_index(notebook_path, window=window)
    stored_rates = idx.get("stored_rates", {})
    corrected_rates = idx.get("corrected_rates", {})
    max_n_used = max((row.get("n_used") or 0 for row in op_rates), default=0)

    components = [
        _build_component_entry(
            row,
            stored_rates,
            corrected_rates,
            grad_health,
            metric_overlays,
            max_n_used,
        )
        for row in op_rates
    ]
    rated_ops = {row["op_name"] for row in op_rates}
    components.extend(
        _build_profile_only_component(op_name, prof)
        for op_name, prof in grad_health.items()
        if op_name not in rated_ops
    )
    components.sort(
        key=lambda component: (
            {"broken": 0, "degraded": 1, "structural": 2, "healthy": 3}.get(
                component["status"],
                2,
            ),
            -(component["n_used"] or 0),
        )
    )
    result = {
        "components": components,
        "total": len(components),
        "healthy": sum(
            component["status"] in {"healthy", "structural"} for component in components
        ),
        "degraded": sum(component["status"] == "degraded" for component in components),
        "broken": sum(component["status"] == "broken" for component in components),
        "cached_at": time.time(),
        "window": window,
        "_window": window,
    }
    _health_cache = result
    _health_cache_ts = now
    return result


def _append_rate_alert(
    alerts: List[Dict[str, Any]],
    *,
    notebook_path: str,
    now: float,
    query: str,
    params: tuple[Any, ...],
    min_total: int,
    threshold_key: str,
    alert_id: str,
    severity: str,
    title: str,
    message_window: str,
) -> None:
    nb = get_notebook(notebook_path, read_only=True)
    try:
        row = nb.conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        logger.debug("%s alert evaluation failed", alert_id, exc_info=True)
        return
    if not row or row["total"] < min_total:
        return
    rate = row["passed"] / row["total"]
    threshold = _DEFAULT_THRESHOLDS[threshold_key]
    if rate >= threshold:
        return
    alerts.append(
        {
            "id": alert_id,
            "severity": severity,
            "title": title,
            "message": f"{title} is {rate:.0%} in the last {message_window} ({row['passed']}/{row['total']})",
            "value": round(rate, 3),
            "threshold": threshold,
            "timestamp": now,
        }
    )


def _append_routing_collapse_alert(
    alerts: List[Dict[str, Any]],
    notebook_path: str,
    thresholds: Dict[str, Any],
    now: float,
) -> None:
    nb = get_notebook(notebook_path, read_only=True)
    try:
        row = nb.conn.execute(
            "SELECT AVG(CAST(json_extract(starvation_report_json, '$.collapse_score') AS REAL)) as avg_collapse "
            "FROM program_results "
            "WHERE starvation_report_json IS NOT NULL AND timestamp > ?",
            (now - 7200,),
        ).fetchone()
    except sqlite3.OperationalError:
        logger.debug("Routing collapse alert evaluation failed", exc_info=True)
        return
    if not row or row["avg_collapse"] is None:
        return
    score = float(row["avg_collapse"])
    threshold = thresholds.get("routing_collapse_score_min", 0.3)
    if score >= threshold:
        return
    alerts.append(
        {
            "id": "routing_collapse",
            "severity": "warning",
            "title": "Routing collapse detected",
            "message": f"Average routing health score is {score:.2f} (threshold: {threshold})",
            "value": round(score, 3),
            "threshold": threshold,
            "timestamp": now,
        }
    )


def _append_broken_component_alert(
    alerts: List[Dict[str, Any]],
    notebook_path: str,
    now: float,
) -> None:
    health = get_component_health(notebook_path)
    if health["broken"] <= 0:
        return
    broken_ops = [
        component["op"]
        for component in health["components"]
        if component["status"] == "broken"
    ][:5]
    alerts.append(
        {
            "id": "broken_components",
            "severity": "warning" if health["broken"] <= 3 else "critical",
            "title": f"{health['broken']} broken component(s)",
            "message": f"Broken ops: {', '.join(broken_ops)}"
            + (" ..." if health["broken"] > 5 else ""),
            "value": health["broken"],
            "threshold": 0,
            "timestamp": now,
        }
    )


def _append_stale_pipeline_alert(
    alerts: List[Dict[str, Any]],
    notebook_path: str,
    thresholds: Dict[str, Any],
    now: float,
) -> None:
    nb = get_notebook(notebook_path, read_only=True)
    try:
        row = nb.conn.execute(
            "SELECT MAX(timestamp) as last_ts FROM program_results"
        ).fetchone()
    except sqlite3.OperationalError:
        logger.debug("Stale pipeline alert evaluation failed", exc_info=True)
        return
    if not row or not row["last_ts"]:
        return
    hours_since = (now - row["last_ts"]) / 3600
    threshold_hours = thresholds.get("stale_experiment_hours", 6)
    if hours_since <= threshold_hours:
        return
    alerts.append(
        {
            "id": "stale_pipeline",
            "severity": "info",
            "title": "Pipeline idle",
            "message": f"No new results in {hours_since:.1f} hours",
            "value": round(hours_since, 1),
            "threshold": threshold_hours,
            "timestamp": now,
        }
    )


def evaluate_alerts(
    notebook_path: str, thresholds: Dict[str, Any]
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    now = time.time()

    _append_rate_alert(
        alerts,
        notebook_path=notebook_path,
        now=now,
        query=(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) as passed "
            "FROM program_results WHERE timestamp > ?"
        ),
        params=(now - 3600,),
        min_total=10,
        threshold_key="s0_pass_rate_min",
        alert_id="low_s0_rate",
        severity="critical",
        title="Low S0 pass rate",
        message_window="hour",
    )
    _append_rate_alert(
        alerts,
        notebook_path=notebook_path,
        now=now,
        query=(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as passed "
            "FROM program_results WHERE stage0_passed = 1 AND timestamp > ?"
        ),
        params=(now - 7200,),
        min_total=20,
        threshold_key="s1_pass_rate_min",
        alert_id="low_s1_rate",
        severity="critical",
        title="Low S1 pass rate",
        message_window="2 hours",
    )
    _append_routing_collapse_alert(alerts, notebook_path, thresholds, now)
    _append_broken_component_alert(alerts, notebook_path, now)
    _append_stale_pipeline_alert(alerts, notebook_path, thresholds, now)
    alerts.sort(
        key=lambda alert: {"critical": 0, "warning": 1, "info": 2}[alert["severity"]]
    )
    return alerts


def get_cached_alerts(
    notebook_path: str, thresholds: Dict[str, Any]
) -> List[Dict[str, Any]]:
    now = time.monotonic()
    with _alerts_lock:
        cached = _alerts_cache.get(notebook_path)
        if cached and (now - cached[0]) < _ALERTS_TTL:
            return cached[1]
    alerts = evaluate_alerts(notebook_path, thresholds)
    with _alerts_lock:
        _alerts_cache[notebook_path] = (now, alerts)
    return alerts
