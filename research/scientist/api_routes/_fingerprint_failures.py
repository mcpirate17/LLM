"""Fingerprint failure metadata shared by dashboard API routes."""

from __future__ import annotations

import json
from typing import Any, Dict, List


FINGERPRINT_STATUS_FIELDS = (
    ("fp_spec_norm_status", "Spectral norm"),
    ("fp_jacobian_erf_status", "Jacobian ERF"),
    ("fp_icld_status", "ICLD velocity"),
    ("fp_id_collapse_status", "ID collapse"),
    ("fp_logit_margin_status", "Logit margin"),
)

POST_INVESTIGATION_FINGERPRINT_REQUIRED_TIERS = {
    "investigation",
    "validation",
    "breakthrough",
}

NON_FAILURE_STATUSES = {
    "",
    "init",
    "ok",
    "pass",
    "passed",
    "skipped",
    "not_run",
    "not_applicable",
    "unavailable",
}


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _fingerprint_json_meta(entry: Dict[str, Any]) -> Dict[str, Any]:
    raw = entry.get("fingerprint_json")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"parse_error": True}
    return parsed if isinstance(parsed, dict) else {}


def _is_screening_replay_entry(entry: Dict[str, Any]) -> bool:
    return _normalize_status(entry.get("model_source")) == "exact_graph_replay"


def _is_deferred_observation_entry(entry: Dict[str, Any]) -> bool:
    trust_label = _normalize_status(entry.get("trust_label"))
    result_cohort = _normalize_status(entry.get("result_cohort") or entry.get("cohort"))
    model_source = _normalize_status(entry.get("model_source"))
    return (
        trust_label == "backfill_observation"
        or result_cohort == "backfill"
        or model_source == "exact_graph_replay"
    )


def fingerprint_failure_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized UI metadata for incomplete/failed fingerprint checks."""
    failed_checks: List[Dict[str, str]] = []
    ok_checks: List[Dict[str, str]] = []
    pending_checks: List[Dict[str, str]] = []

    for field, label in FINGERPRINT_STATUS_FIELDS:
        raw_status = entry.get(field)
        status = _normalize_status(raw_status)
        item = {"field": field, "label": label, "status": status or "missing"}
        if status in {"", "init", "not_run"}:
            pending_checks.append(item)
        elif status in NON_FAILURE_STATUSES:
            ok_checks.append(item)
        else:
            failed_checks.append(item)

    meta = _fingerprint_json_meta(entry)
    completed_marker_present = "fingerprint_completed_post_investigation" in meta
    completed = meta.get("fingerprint_completed_post_investigation")
    json_parse_error = bool(meta.get("parse_error"))
    if json_parse_error:
        failed_checks.append(
            {
                "field": "fingerprint_json",
                "label": "Fingerprint JSON",
                "status": "parse_error",
            }
        )

    tier = _normalize_status(entry.get("tier") or entry.get("leaderboard_tier"))
    incomplete_tier = tier == "investigation_fingerprint_incomplete"
    post_investigation_required = not _is_deferred_observation_entry(entry) and (
        incomplete_tier or tier in POST_INVESTIGATION_FINGERPRINT_REQUIRED_TIERS
    )
    if completed_marker_present and completed is False and post_investigation_required:
        failed_checks.append(
            {
                "field": "fingerprint_completed_post_investigation",
                "label": "Post-investigation fingerprint",
                "status": "incomplete",
            }
        )
    elif incomplete_tier and not failed_checks:
        failed_checks.append(
            {
                "field": "tier",
                "label": "Post-investigation fingerprint",
                "status": "incomplete",
            }
        )

    is_failed = bool(failed_checks or incomplete_tier)
    return {
        "failed": is_failed,
        "label": "Fingerprint failed" if is_failed else "Fingerprint complete",
        "failed_count": len(failed_checks),
        "ok_count": len(ok_checks),
        "pending_count": len(pending_checks),
        "failed_checks": failed_checks,
        "ok_checks": ok_checks,
        "pending_checks": pending_checks,
        "completed_post_investigation": bool(completed)
        if completed_marker_present
        else None,
    }


def attach_fingerprint_failure_metadata(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary = fingerprint_failure_summary(entry)
        entry["fingerprint_failure_summary"] = summary
        entry["fingerprint_failed"] = summary["failed"]
        entry["fingerprint_failure_count"] = summary["failed_count"]
