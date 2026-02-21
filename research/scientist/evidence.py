"""
Evidence Pack utilities.

Enforces that recommendations and decisions are backed by measurable,
queryable metrics from the lab notebook.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


class EvidencePackError(ValueError):
    """Raised when an Evidence Pack is missing required fields."""


SELECTION_DECISION_LOG_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ExperimentSelectionDecision",
    "type": "object",
    "required": [
        "decision_id",
        "timestamp",
        "context",
        "candidate_pool_summary",
        "score_breakdown",
        "policy",
        "reason",
        "chosen_experiments",
    ],
    "properties": {
        "decision_id": {"type": "string"},
        "timestamp": {"type": "number"},
        "context": {"type": "string"},
        "experiment_id": {"type": ["string", "null"]},
        "candidate_pool_summary": {"type": "object"},
        "score_breakdown": {"type": "array", "items": {"type": "object"}},
        "policy": {"type": "object"},
        "reason": {"type": "string"},
        "chosen_experiments": {"type": "array", "items": {"type": "object"}},
        "trigger": {"type": ["object", "null"]},
    },
}


def _median(values: Iterable[float]) -> Optional[float]:
    data = [v for v in values if isinstance(v, (int, float))]
    if not data:
        return None
    data.sort()
    mid = len(data) // 2
    if len(data) % 2 == 1:
        return float(data[mid])
    return float((data[mid - 1] + data[mid]) / 2.0)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class EvidencePack:
    """Structured evidence pack for a recommendation/decision."""
    hypothesis: str
    supporting_metrics: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    confounders: List[str] = field(default_factory=list)
    falsification: List[str] = field(default_factory=list)
    novelty_reference: Optional[Dict[str, Any]] = None
    audit_queries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "supporting_metrics": self.supporting_metrics,
            "uncertainty": self.uncertainty,
            "confounders": self.confounders,
            "falsification": self.falsification,
            "novelty_reference": self.novelty_reference,
            "audit_queries": self.audit_queries,
        }


def validate_evidence_pack(pack: Dict[str, Any]) -> None:
    """Raise EvidencePackError if the pack is missing required fields."""
    if not isinstance(pack, dict):
        raise EvidencePackError("Evidence pack must be a dict.")

    required = ["hypothesis", "supporting_metrics", "uncertainty", "confounders", "falsification"]
    for field_name in required:
        if field_name not in pack:
            raise EvidencePackError(f"Missing evidence pack field: {field_name}")

    if not isinstance(pack.get("supporting_metrics"), list) or not pack["supporting_metrics"]:
        raise EvidencePackError("Evidence pack must include supporting_metrics list.")

    for metric in pack["supporting_metrics"]:
        if "name" not in metric or "value" not in metric:
            raise EvidencePackError("Each supporting metric must include name and value.")
        if "baseline" not in metric or "delta_vs_baseline" not in metric:
            raise EvidencePackError("Each supporting metric must include baseline and delta_vs_baseline.")

    has_novelty_metric = any(
        isinstance(m, dict) and "novelty" in str(m.get("name", "")).lower()
        for m in pack.get("supporting_metrics", [])
    )
    if has_novelty_metric and not pack.get("novelty_reference"):
        raise EvidencePackError(
            "Novelty metric present but novelty_reference is missing."
        )

    if pack.get("novelty_reference") is not None:
        novelty_ref = pack["novelty_reference"]
        for key in ("cka_source", "cka_artifact_version", "similarity_path"):
            if not novelty_ref.get(key):
                raise EvidencePackError(
                    f"Novelty reference missing required field: {key}"
                )


def ensure_evidence_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return the evidence pack."""
    validate_evidence_pack(pack)
    return pack


def validate_selection_decision_log(log: Dict[str, Any]) -> None:
    """Lightweight structural validation for selection decision logs."""
    if not isinstance(log, dict):
        raise EvidencePackError("Selection decision log must be a dict.")
    for key in SELECTION_DECISION_LOG_SCHEMA["required"]:
        if key not in log:
            raise EvidencePackError(f"Selection decision log missing field: {key}")
    if not isinstance(log.get("candidate_pool_summary"), dict):
        raise EvidencePackError("candidate_pool_summary must be an object.")
    if not isinstance(log.get("score_breakdown"), list):
        raise EvidencePackError("score_breakdown must be a list.")
    if not isinstance(log.get("policy"), dict):
        raise EvidencePackError("policy must be an object.")
    if not isinstance(log.get("chosen_experiments"), list):
        raise EvidencePackError("chosen_experiments must be a list.")


def _extract_evidence_payload(evidence: Optional[str]) -> Dict[str, Any]:
    if not evidence:
        return {}
    text = str(evidence)
    if "meta=" in text:
        text = text.split("meta=", 1)[-1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def validate_learning_log_entry(entry: Dict[str, Any]) -> None:
    """Validate audit evidence for learning-log entries."""
    if not entry:
        return
    if entry.get("event_type") != "grammar_weights_applied":
        return
    payload = _extract_evidence_payload(entry.get("evidence"))
    audit = payload.get("audit_query") if isinstance(payload, dict) else None
    if not audit or not isinstance(audit, dict):
        raise EvidencePackError("grammar_weights_applied missing audit_query evidence.")
    if not audit.get("query"):
        raise EvidencePackError("grammar_weights_applied audit_query missing query.")


def _query_scalar(nb, query: str, params: Optional[tuple] = None) -> Optional[float]:
    row = nb.conn.execute(query, params or ()).fetchone()
    if not row:
        return None
    # First column
    return _safe_float(row[0])


def _query_pair(nb, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
    row = nb.conn.execute(query, params or ()).fetchone()
    if not row:
        return None
    return tuple(row)


def build_evidence_pack(
    nb,
    analytics=None,
    recommendation: Optional[Dict[str, Any]] = None,
    decision_type: str = "recommendation",
    recent_experiments: Optional[List[Dict[str, Any]]] = None,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """Build a minimal evidence pack backed by SQLite metrics."""
    if not hasattr(nb, "conn"):
        hypothesis = "Run the recommended experiment to gather measurable evidence."
        if recommendation:
            mode = recommendation.get("mode") or recommendation.get("config", {}).get("mode")
            if mode:
                hypothesis = f"Switch to {mode} to gather measurable evidence."
        pack = EvidencePack(
            hypothesis=hypothesis,
            supporting_metrics=[{
                "name": "evidence_unavailable",
                "value": 0.0,
                "baseline": 0.0,
                "delta_vs_baseline": 0.0,
                "source": "notebook_missing",
            }],
            uncertainty={"note": "Notebook connection unavailable; metrics not queried."},
            confounders=["Notebook connection unavailable."],
            falsification=["If metrics remain unavailable after next cycle, halt decisions."],
            novelty_reference=None,
            audit_queries=[],
        ).to_dict()
        return ensure_evidence_pack(pack)
    if recent_experiments is None:
        recent_experiments = nb.get_recent_experiments(sample_size)

    completed = [e for e in recent_experiments if e.get("status") == "completed"]
    latest = completed[0] if completed else (recent_experiments[0] if recent_experiments else {})
    latest_id = latest.get("experiment_id")

    exp_ids = [e.get("experiment_id") for e in completed if e.get("experiment_id")]
    placeholders = ",".join("?" * len(exp_ids)) if exp_ids else None

    def _s1_rate_for_ids(ids: List[str]) -> Optional[float]:
        if not ids:
            return None
        ph = ",".join("?" * len(ids))
        row = nb.conn.execute(
            f"""SELECT COUNT(*) as total,
                       SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1
                FROM program_results WHERE experiment_id IN ({ph})""",
            tuple(ids),
        ).fetchone()
        total = float(row["total"] or 0)
        s1 = float(row["s1"] or 0)
        return s1 / max(total, 1.0)

    recent_s1 = _s1_rate_for_ids(exp_ids) if exp_ids else None
    overall_s1 = _query_scalar(
        nb,
        """SELECT SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) * 1.0
                  / MAX(COUNT(*), 1)
           FROM program_results""",
    )

    recent_best_loss = _safe_float(latest.get("best_loss_ratio"))
    overall_loss_values = [
        _safe_float(r["best_loss_ratio"])
        for r in nb.conn.execute(
            "SELECT best_loss_ratio FROM experiments WHERE best_loss_ratio IS NOT NULL"
        ).fetchall()
    ]
    overall_loss_median = _median([v for v in overall_loss_values if v is not None])

    recent_best_novelty = _safe_float(latest.get("best_novelty_score"))
    overall_novelty_values = [
        _safe_float(r["best_novelty_score"])
        for r in nb.conn.execute(
            "SELECT best_novelty_score FROM experiments WHERE best_novelty_score IS NOT NULL"
        ).fetchall()
    ]
    overall_novelty_median = _median([v for v in overall_novelty_values if v is not None])

    novelty_reference = None
    if latest_id:
        row = nb.conn.execute(
            """SELECT cka_source, cka_artifact_version, fingerprint_json
               FROM program_results
               WHERE experiment_id = ? AND novelty_score IS NOT NULL
               ORDER BY novelty_score DESC LIMIT 1""",
            (latest_id,),
        ).fetchone()
        if row:
            similarity_path = None
            if row["fingerprint_json"]:
                try:
                    fp = json.loads(row["fingerprint_json"])
                    similarity_path = fp.get("similarity_path")
                except json.JSONDecodeError:
                    similarity_path = None
            if row["cka_source"] and similarity_path:
                novelty_reference = {
                    "cka_source": row["cka_source"],
                    "cka_artifact_version": row["cka_artifact_version"],
                    "similarity_path": similarity_path,
                }

    supporting_metrics = []
    if recent_s1 is not None:
        supporting_metrics.append({
            "name": "s1_pass_rate",
            "value": round(recent_s1, 4),
            "baseline": round(overall_s1 or 0.0, 4),
            "delta_vs_baseline": round(recent_s1 - (overall_s1 or 0.0), 4),
            "source": "program_results.stage1_passed",
        })
    if recent_best_loss is not None:
        supporting_metrics.append({
            "name": "best_loss_ratio",
            "value": round(recent_best_loss, 6),
            "baseline": round(overall_loss_median or 0.0, 6),
            "delta_vs_baseline": round((recent_best_loss - (overall_loss_median or 0.0)), 6),
            "source": "experiments.best_loss_ratio",
        })
    if recent_best_novelty is not None and novelty_reference:
        supporting_metrics.append({
            "name": "best_novelty_score",
            "value": round(recent_best_novelty, 4),
            "baseline": round(overall_novelty_median or 0.0, 4),
            "delta_vs_baseline": round((recent_best_novelty - (overall_novelty_median or 0.0)), 4),
            "source": "experiments.best_novelty_score",
        })

    uncertainty = {
        "sample_size_experiments": len(exp_ids),
        "sample_size_programs": _query_scalar(
            nb,
            f"SELECT COUNT(*) FROM program_results WHERE experiment_id IN ({placeholders})"
            if placeholders else "SELECT COUNT(*) FROM program_results",
            tuple(exp_ids) if exp_ids else None,
        ) or 0,
    }
    if analytics:
        try:
            control = analytics.control_experiment_comparison()
        except Exception:
            control = None
        if control:
            uncertainty["control_comparison"] = control

    confounders = []
    if uncertainty["sample_size_experiments"] < 3:
        confounders.append("Low experiment count; results may be noisy.")
    if novelty_reference and (novelty_reference.get("cka_source") in (None, "none")):
        confounders.append("Novelty reference missing; novelty scores may be structural-only.")

    falsification = [
        "If the next experiment fails to match or exceed recent S1 pass rate.",
        "If best loss ratio regresses relative to the median baseline.",
    ]
    if recent_best_novelty is not None:
        falsification.append("If novelty scores drop below recent median for the next cycle.")

    hypothesis = "Run the recommended experiment to improve S1 pass rate and novelty."
    if recommendation:
        mode = recommendation.get("mode") or recommendation.get("config", {}).get("mode")
        if mode:
            hypothesis = f"Switch to {mode} to improve evidence-backed metrics."

    pack = EvidencePack(
        hypothesis=hypothesis,
        supporting_metrics=supporting_metrics,
        uncertainty=uncertainty,
        confounders=confounders,
        falsification=falsification,
        novelty_reference=novelty_reference,
        audit_queries=[
            {
                "query": "SELECT * FROM program_results WHERE experiment_id = ?",
                "params": [latest_id] if latest_id else [],
            },
        ],
    ).to_dict()

    return ensure_evidence_pack(pack)
