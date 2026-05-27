"""Durable observability for measured-rescue candidates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


RECORDS_KEY = "measured_rescue_records"


def _records(results: Dict[str, Any]) -> list[Dict[str, Any]]:
    records = results.setdefault(RECORDS_KEY, [])
    if isinstance(records, list):
        return records
    results[RECORDS_KEY] = []
    return results[RECORDS_KEY]


def _record_for_fingerprint(
    results: Dict[str, Any],
    graph_fingerprint: str,
) -> Optional[Dict[str, Any]]:
    if not graph_fingerprint:
        return None
    for record in _records(results):
        if str(record.get("graph_fingerprint") or "") == graph_fingerprint:
            return record
    return None


def initialize_measured_rescue_records(
    results: Dict[str, Any],
    rescue_records: Iterable[Dict[str, Any]],
    *,
    experiment_id: str,
    tau: Any = None,
    max_rescue: Any = None,
    probe_budget: Any = None,
) -> list[Dict[str, Any]]:
    """Persist rescue records with default downstream-outcome fields."""
    stored = _records(results)
    by_fp = {
        str(record.get("graph_fingerprint") or ""): record
        for record in stored
        if record.get("graph_fingerprint")
    }
    for source in rescue_records:
        fp = str(source.get("graph_fingerprint") or "")
        if not fp:
            continue
        record = by_fp.get(fp)
        if record is None:
            record = {"graph_fingerprint": fp}
            stored.append(record)
            by_fp[fp] = record
        record.update(dict(source))
        record.update(
            {
                "experiment_id": experiment_id,
                "rescue_reason": source.get("rescue_reason")
                or "measured_long_range_reach_ge_tau",
                "measured_probe_passed": bool(
                    source.get("measured_probe_passed", True)
                ),
                "structural_induction_signal": source.get(
                    "structural_induction_signal", "long_range_reach"
                ),
                "measured_rescue_tau": tau,
                "measured_rescue_max": max_rescue,
                "measured_rescue_probe_budget": probe_budget,
                "rescued_at": "gbm_prescreener",
                "reached_screening": False,
                "screening_index": None,
                "gate_status": None,
                "stage_at_death": None,
                "reached_stage0": False,
                "stage0_passed": None,
                "stage05_passed": None,
                "stability_score": None,
                "reached_rapid_screening": False,
                "rapid_screening_passed": None,
                "rapid_screening_kill_reason": None,
                "stage1_queued": False,
                "stage1_completed": False,
                "stage1_passed": None,
                "result_id": None,
                "loss_ratio": None,
                "final_loss": None,
            }
        )
    return stored


def measured_rescue_metrics_for_fingerprint(
    results: Dict[str, Any],
    graph_fingerprint: str,
) -> Dict[str, Any]:
    """Return DB-row metrics for a rescued candidate, or empty dict."""
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return {}
    metrics: Dict[str, Any] = {
        "measured_rescue_candidate": 1,
        "measured_rescue_reason": record.get("rescue_reason"),
    }
    for key in (
        "measured_long_range_reach",
        "measured_content_dependence",
        "predicted_p_s1",
        "predicted_induction_screening_auc",
        "predicted_p_induction_learner",
        "predictor_planning_score",
        "predicted_rank_composite",
        "screening_ensemble_p_pass_floor",
        "screening_ensemble_p_pass_floor_source",
    ):
        if record.get(key) is not None:
            metrics[key] = record.get(key)
    return metrics


def mark_measured_rescue_screening(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    index: int,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["reached_screening"] = True
    record["screening_index"] = index
    record["gate_status"] = "screening_considered"


def mark_measured_rescue_gate_drop(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    reason: str,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["gate_status"] = f"dropped_{reason}"
    record["stage_at_death"] = reason


def mark_measured_rescue_stage0_attempted(
    results: Dict[str, Any],
    graph_fingerprint: str,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["reached_stage0"] = True
    record["gate_status"] = "stage0_attempted"


def mark_measured_rescue_stage0_result(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    stage0_passed: bool,
    stage05_passed: bool,
    stability_score: Any = None,
    error_type: Any = None,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["stage0_passed"] = bool(stage0_passed)
    record["stage05_passed"] = bool(stage05_passed)
    record["stability_score"] = stability_score
    if not stage0_passed:
        record["stage_at_death"] = "stage0"
        record["gate_status"] = "dropped_stage0"
    elif not stage05_passed:
        record["stage_at_death"] = "stage05"
        record["gate_status"] = "dropped_stage05"
    else:
        record["gate_status"] = "stage05_passed"
    if error_type:
        record["error_type"] = error_type


def mark_measured_rescue_s075_drop(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    initial_loss: Any,
    threshold: Any,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["stage_at_death"] = "stage075"
    record["gate_status"] = "dropped_s075_high_init"
    record["s075_initial_loss"] = initial_loss
    record["s075_threshold"] = threshold


def mark_measured_rescue_rapid_result(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    passed: bool,
    kill_reason: Any = None,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["reached_rapid_screening"] = True
    record["rapid_screening_passed"] = bool(passed)
    record["rapid_screening_kill_reason"] = kill_reason
    if passed:
        record["gate_status"] = "rapid_screening_passed"
    else:
        record["gate_status"] = "dropped_rapid_screening"
        record["stage_at_death"] = "rapid_screening"


def mark_measured_rescue_stage1_queued(
    results: Dict[str, Any],
    graph_fingerprint: str,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["stage1_queued"] = True
    record["gate_status"] = "stage1_queued"


def mark_measured_rescue_stage1_result(
    results: Dict[str, Any],
    graph_fingerprint: str,
    *,
    completed: bool,
    passed: bool,
    result_id: Any = None,
    loss_ratio: Any = None,
    final_loss: Any = None,
) -> None:
    record = _record_for_fingerprint(results, graph_fingerprint)
    if record is None:
        return
    record["stage1_completed"] = bool(completed)
    record["stage1_passed"] = bool(passed) if completed else None
    record["result_id"] = result_id
    record["loss_ratio"] = loss_ratio
    record["final_loss"] = final_loss
    if completed:
        record["gate_status"] = "stage1_passed" if passed else "stage1_failed"
        if not passed:
            record["stage_at_death"] = "stage1"
