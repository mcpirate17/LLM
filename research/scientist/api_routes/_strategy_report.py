"""Report query/filter helpers and report-specific utilities."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ._strategy_preflight import build_start_mode_eligibility


def parse_report_date(value: Optional[str], end_of_day: bool = False) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.timestamp()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def report_program_matches_theme(program: Dict[str, Any], theme: str) -> bool:
    normalized = str(theme or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    graph_json = str(program.get("graph_json") or "").lower()
    arch_spec = str(program.get("arch_spec_json") or "").lower()
    pruning_method = str(program.get("pruning_method") or "").lower()
    if normalized == "sparsity":
        return (
            program.get("sparse_density_mean") is not None
            or "sparse" in graph_json
            or "sparse" in arch_spec
            or bool(pruning_method)
        )
    if normalized == "compression":
        compression_markers = (
            "low_rank", "shared_basis", "tied_proj", "grouped_linear", "bottleneck", "quant", "compressed"
        )
        return any(marker in graph_json or marker in arch_spec for marker in compression_markers)
    if normalized == "routing":
        return (
            program.get("routing_confidence_mean") is not None
            or "routing" in graph_json
            or "moe" in graph_json
            or "gate" in graph_json
        )
    if normalized == "mathspace":
        return (
            bool(program.get("graph_uses_math_spaces"))
            or "mathspace" in graph_json
            or "clifford" in graph_json
            or "hyperbolic" in graph_json
            or "padic" in graph_json
            or "tropical" in graph_json
        )
    if normalized == "failure_modes":
        return (program.get("stage1_passed") or 0) == 0 or bool(program.get("error_type"))
    return True


def experiment_s1_rate(exp: Dict[str, Any]) -> Optional[float]:
    generated = exp.get("n_programs_generated")
    if generated is None:
        generated = exp.get("n_programs")
    passed = exp.get("n_stage1_passed")
    if passed is None:
        passed = exp.get("s1_passed")
    try:
        gen = float(generated or 0)
        s1 = float(passed or 0)
    except Exception:
        return None
    if gen <= 0:
        return None
    return s1 / gen


def report_experiment_matches_trend(exp: Dict[str, Any], trend: str) -> bool:
    normalized = str(trend or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    rate = experiment_s1_rate(exp)
    novelty = exp.get("best_novelty_score")
    if normalized == "high_novelty":
        return isinstance(novelty, (int, float)) and float(novelty) >= 0.5
    if rate is None:
        return False
    if normalized in {"improving", "high_survival"}:
        return rate >= 0.08
    if normalized == "declining":
        return rate < 0.03
    if normalized == "plateaued":
        return 0.03 <= rate < 0.08
    return True


def build_filtered_report_summary(
    base_summary: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not experiments:
        return dict(base_summary or {})
    total_programs = 0
    total_survivors = 0
    for exp in experiments:
        total_programs += int(exp.get("n_programs_generated") or 0)
        total_survivors += int(exp.get("n_stage1_passed") or 0)
    out = dict(base_summary or {})
    out["total_experiments"] = len(experiments)
    out["total_programs_evaluated"] = total_programs
    out["stage1_survivors"] = total_survivors
    return out


def build_report_snapshot_key(scope: str, query_payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        {"scope": scope, "query": query_payload or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_report_action_eligibility(
    nb,
    result_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-result report action eligibility aligned with start guardrails."""
    from ._helpers import normalize_result_ids

    normalized_ids = normalize_result_ids(result_ids)
    if not normalized_ids:
        return {}

    inv = build_start_mode_eligibility(nb, "investigation", normalized_ids)
    val = build_start_mode_eligibility(nb, "validation", normalized_ids)

    inv_eligible = set(inv.get("eligible_result_ids") or [])
    val_eligible = set(val.get("eligible_result_ids") or [])
    inv_reason = {
        row.get("result_id"): row.get("reason")
        for row in (inv.get("ineligible") or [])
        if row.get("result_id")
    }
    val_reason = {
        row.get("result_id"): row.get("reason")
        for row in (val.get("ineligible") or [])
        if row.get("result_id")
    }

    eligibility_by_id: Dict[str, Dict[str, Any]] = {}
    for result_id in normalized_ids:
        investigation_eligible = result_id in inv_eligible
        validation_eligible = result_id in val_eligible
        queue_eligible = investigation_eligible or validation_eligible
        queue_reason = None
        if not queue_eligible:
            queue_reason = inv_reason.get(result_id) or val_reason.get(result_id) or "not_progression_eligible"

        eligibility_by_id[result_id] = {
            "investigationEligible": investigation_eligible,
            "validationEligible": validation_eligible,
            "queueEligible": queue_eligible,
            "queueReason": queue_reason,
            "investigationReason": inv_reason.get(result_id),
            "validationReason": val_reason.get(result_id),
        }

    return eligibility_by_id


def normalize_entries(entries: list) -> list:
    """Normalize notebook entry dicts for API consumption.

    Parses JSON metadata strings, ensures consistent field types.
    """
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        d = dict(entry)
        meta = d.get("metadata") or d.get("metadata_json", "{}") or d.get("metadata_json", "{}")
        if isinstance(meta, str):
            try:
                d["metadata"] = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        elif meta is None:
            d["metadata"] = {}
        result.append(d)
    return result


def parse_bool_query(value: Optional[str], default: bool = False) -> bool:
    """Parse a query parameter as a boolean."""
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def build_full_report_data(nb: LabNotebook, analytics, fast_mode: bool, include_heavy: bool) -> Dict[str, Any]:
    """Build the complete data payload for the full research report."""
    top_limit = 20 if not fast_mode else 12
    expanded_limit = 80 if include_heavy else 0
    recent_limit = 100 if include_heavy else 30

    data = {
        "summary": nb.get_dashboard_summary(),
        "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(top_limit, sort_by="loss_ratio"),
        "top_programs_expanded": nb.get_top_programs(expanded_limit, sort_by="loss_ratio") if include_heavy else [],
        "recent_experiments": nb.get_recent_experiments(recent_limit),
        "op_success_rates": analytics.op_success_rates(),
        "failure_patterns": analytics.failure_patterns(),
        "grammar_weights": {
            "learned": analytics.compute_grammar_weights(),
            "default": analytics.get_current_grammar_weights(),
        },
        "insights": nb.get_insights(),
    }
    
    if include_heavy:
        data.update({
            "math_family_coverage": analytics.math_family_coverage(),
            "efficiency_frontier": analytics.efficiency_frontier(),
        })
    
    return data


def build_scoped_report_query(nb: LabNotebook, analytics, start_ts, end_ts, theme, trend, limit):
    """Filter experiments and programs for a scoped report query."""
    experiments = nb.get_recent_experiments(500)
    filtered_exps = [e for e in experiments if _matches_report_filters(e, start_ts, end_ts, trend)]
    
    sort_by = "novelty_score" if trend == "high_novelty" else "loss_ratio"
    programs = nb.get_top_programs(max(limit * 3, 120), sort_by=sort_by)
    filtered_progs = [p for p in programs if _matches_program_filters(p, start_ts, end_ts, theme)]
    
    # Deduplicate by fingerprint
    grouped = []
    seen = set()
    for p in filtered_progs:
        fp = p.get("graph_fingerprint")
        if fp and fp not in seen:
            seen.add(fp)
            grouped.append(p)
            if len(grouped) >= limit: break
            
    return {
        "summary": build_filtered_report_summary(nb.get_dashboard_summary(), filtered_exps),
        "top_programs": grouped,
        "top_programs_expanded": filtered_progs[:max(limit*2, 40)],
        "recent_experiments": filtered_exps[:max(limit*5, 40)],
    }


def _matches_report_filters(exp, start, end, trend):
    ts = exp.get("timestamp")
    if start and ts < start: return False
    if end and ts > end: return False
    return report_experiment_matches_trend(exp, trend)


def _matches_program_filters(prog, start, end, theme):
    ts = prog.get("timestamp")
    if start and ts < start: return False
    if end and ts > end: return False
    return report_program_matches_theme(prog, theme)
