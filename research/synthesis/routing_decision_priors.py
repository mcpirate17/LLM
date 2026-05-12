"""Safe advisory loader for routing-decision prior artifacts.

Generation does not consume this module yet. It exists as a small, fail-closed
surface for future routing policies to read offline evidence without scanning
the runs database on the hot path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_ROUTING_DECISION_PRIOR_PATH = Path(
    "research/artifacts/routing_decision_priors/latest.json"
)
ROUTING_DECISION_PRIOR_SCHEMA_VERSION = "routing_decision_prior_v1"


def canonical_routing_decision_value(value: Any) -> str:
    """Return the stable lookup key used for routing-decision values."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(value), separators=(",", ":"))


def empty_routing_decision_priors(reason: str = "empty") -> dict[str, Any]:
    """Return a valid empty prior payload.

    Callers can treat this exactly like a loaded artifact: the nested priors map
    is empty and every lookup returns the neutral/default answer.
    """

    return {
        "schema_version": ROUTING_DECISION_PRIOR_SCHEMA_VERSION,
        "version": None,
        "created_at": None,
        "records": [],
        "priors": {},
        "loaded": False,
        "load_reason": reason,
    }


def load_routing_decision_priors(
    path_or_dir: str | Path = DEFAULT_ROUTING_DECISION_PRIOR_PATH,
    *,
    max_age_seconds: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Load routing-decision priors, returning empty priors on any problem."""

    path = Path(path_or_dir)
    artifact = path / "latest.json" if path.is_dir() else path
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return empty_routing_decision_priors("missing_or_invalid")

    if not isinstance(payload, dict):
        return empty_routing_decision_priors("invalid_payload")
    if payload.get("schema_version") != ROUTING_DECISION_PRIOR_SCHEMA_VERSION:
        return empty_routing_decision_priors("schema_mismatch")

    created_at = _safe_float(payload.get("created_at"))
    if max_age_seconds is not None and created_at is not None:
        current = float(time.time() if now is None else now)
        if current - created_at > float(max_age_seconds):
            return empty_routing_decision_priors("stale")

    records = payload.get("records")
    if not isinstance(records, list):
        return empty_routing_decision_priors("invalid_records")
    priors = payload.get("priors")
    if not isinstance(priors, dict):
        priors = build_routing_decision_prior_index(records)

    loaded = dict(payload)
    loaded["records"] = records
    loaded["priors"] = priors
    loaded["loaded"] = True
    loaded["load_reason"] = "loaded"
    return loaded


def build_routing_decision_prior_index(records: list[Any]) -> dict[str, Any]:
    """Build nested template -> decision -> value lookup from record rows."""

    index: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        template_name = str(record.get("template_name") or "")
        decision_key = str(record.get("decision_key") or "")
        if not template_name or not decision_key:
            continue
        value_key = str(
            record.get("value_key")
            or canonical_routing_decision_value(record.get("value"))
        )
        index.setdefault(template_name, {}).setdefault(decision_key, {})[value_key] = (
            record
        )
    return index


def routing_decision_prior_for(
    prior: dict[str, Any] | None,
    template_name: str,
    decision_key: str,
    value: Any,
) -> dict[str, Any] | None:
    """Return the advisory prior record for a routing value, if present."""

    if not isinstance(prior, dict):
        return None
    priors = prior.get("priors")
    if not isinstance(priors, dict):
        return None
    by_template = priors.get(str(template_name))
    if not isinstance(by_template, dict):
        return None
    by_decision = by_template.get(str(decision_key))
    if not isinstance(by_decision, dict):
        return None
    record = by_decision.get(canonical_routing_decision_value(value))
    return record if isinstance(record, dict) else None


def routing_decision_prior_weight(
    prior: dict[str, Any] | None,
    template_name: str,
    decision_key: str,
    value: Any,
    *,
    default: float = 1.0,
) -> float:
    """Return the advisory multiplier for a routing value."""

    record = routing_decision_prior_for(prior, template_name, decision_key, value)
    if not record:
        return float(default)
    weight = _safe_float(record.get("advisory_weight"))
    return float(default) if weight is None else weight


def _safe_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return (
        converted if converted == converted and abs(converted) != float("inf") else None
    )
