"""Active-learning ranking and replay target helpers for phase-7 escalation."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

from .auto_escalate_data import effective_validation_threshold
from ._types import RunConfig
from ..notebook import LabNotebook


def active_learning_screening_rank(
    nb: LabNotebook,
    rows: List[Dict[str, Any]],
    score_map: Dict[str, float],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Prioritize backfill/replay candidates by expected information gain."""
    if not rows:
        return []
    fingerprints = [
        str(row.get("graph_fingerprint") or "").strip()
        for row in rows
        if str(row.get("graph_fingerprint") or "").strip()
    ]
    agg_by_fp = nb.get_fingerprint_aggregates_batch(fingerprints)
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        result_id = str(row.get("result_id") or "")
        fp = str(row.get("graph_fingerprint") or "").strip()
        agg = agg_by_fp.get(fp, {})
        n_runs = int(agg.get("n_runs") or 0)
        n_s1 = int(agg.get("n_s1_passed") or 0)
        s1_rate = n_s1 / max(n_runs, 1)
        ambiguity = 1.0 - abs((2.0 * s1_rate) - 1.0)
        instability = min(1.0, float(agg.get("loss_std") or 0.0) / 0.15)
        threshold_distance = abs(float(score_map.get(result_id, threshold)) - threshold)
        threshold_proximity = max(0.0, 1.0 - min(1.0, threshold_distance / 12.0))
        novelty = float(row.get("novelty_score") or 0.0)
        info_gain = (
            0.35 * threshold_proximity
            + 0.25 * ambiguity
            + 0.20 * instability
            + 0.20 * novelty
        )
        enriched = dict(row)
        enriched["_active_learning"] = {
            "info_gain": round(info_gain, 6),
            "ambiguity": round(ambiguity, 6),
            "instability": round(instability, 6),
            "threshold_proximity": round(threshold_proximity, 6),
            "n_runs": n_runs,
            "s1_rate": round(s1_rate, 6),
        }
        ranked.append(enriched)
    ranked.sort(
        key=lambda row: (
            float((row.get("_active_learning") or {}).get("info_gain") or 0.0),
            float(score_map.get(str(row.get("result_id") or ""), threshold)),
        ),
        reverse=True,
    )
    return ranked


def followup_priority_summary(
    rows: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    if not rows:
        return 0.0, {"policy": "expected_information_gain", "per_result": {}}
    per_result: Dict[str, Any] = {}
    scores: List[float] = []
    for row in rows:
        result_id = str(row.get("result_id") or "")
        details = dict((row.get("_active_learning") or {}))
        if not result_id or not details:
            continue
        per_result[result_id] = details
        scores.append(float(details.get("info_gain") or 0.0))
    priority_score = (sum(scores) / len(scores)) if scores else 0.0
    return priority_score, {
        "policy": "expected_information_gain",
        "selected_count": len(per_result),
        "per_result": per_result,
    }


def active_replay_suppressed_result_ids(nb: LabNotebook) -> set[str]:
    suppressed_ids: set[str] = set()
    for status in ("queued", "running"):
        for task in nb.get_followup_tasks(stage="replay", status=status, limit=200):
            suppressed_ids.update(
                str(rid).strip()
                for rid in (task.get("result_ids_json") or [])
                if str(rid).strip()
            )

    recent_cutoff = time.time() - (12.0 * 3600.0)
    for task in nb.get_followup_tasks(stage="replay", status="completed", limit=200):
        completed_ts = float(task.get("completed_timestamp") or 0.0)
        if completed_ts < recent_cutoff:
            continue
        suppressed_ids.update(
            str(rid).strip()
            for rid in (task.get("result_ids_json") or [])
            if str(rid).strip()
        )
    return suppressed_ids


def active_replay_row_is_eligible(row: Dict[str, Any]) -> bool:
    details = row.get("_active_learning") or {}
    info_gain = float(details.get("info_gain") or 0.0)
    threshold_proximity = float(details.get("threshold_proximity") or 0.0)
    ambiguity = float(details.get("ambiguity") or 0.0)
    instability = float(details.get("instability") or 0.0)
    n_runs = int(details.get("n_runs") or 0)
    return (
        info_gain >= 0.60
        and threshold_proximity >= 0.80
        and (ambiguity >= 0.30 or instability >= 0.20)
        and n_runs >= 1
    )


def active_replay_targets(
    nb: LabNotebook,
    config: RunConfig,
    rows: List[Dict[str, Any]],
    suppressed_ids: set[str],
) -> List[Dict[str, Any]]:
    replay_targets: List[Dict[str, Any]] = []
    seen_canonical_ids: set[str] = set()
    max_targets = max(1, min(2, int(config.auto_investigate_top_n or 1)))
    for row in rows:
        if not active_replay_row_is_eligible(row):
            continue
        canonical_id = str(
            nb.resolve_canonical_result_id(str(row.get("result_id") or "").strip())
            or row.get("result_id")
            or ""
        ).strip()
        if not canonical_id:
            continue
        if canonical_id in seen_canonical_ids or canonical_id in suppressed_ids:
            continue
        replay_target = dict(row)
        replay_target["result_id"] = canonical_id
        replay_targets.append(replay_target)
        seen_canonical_ids.add(canonical_id)
        if len(replay_targets) >= max_targets:
            break
    return replay_targets


def active_learning_validation_rank(
    rows: List[Dict[str, Any]],
    composite_scores: Dict[str, float],
    replication_info: Dict[str, Dict[str, Any]],
    min_score: float,
) -> List[Dict[str, Any]]:
    """Prioritize validation follow-up by uncertainty and decision value."""
    if not rows:
        return []
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        result_id = str(row.get("result_id") or "")
        if not result_id:
            continue
        score = float(composite_scores.get(result_id, min_score) or min_score)
        replication = replication_info.get(
            result_id,
            {"n": 1, "loss_std": 0.0},
        )
        n_rep = max(1, int(replication.get("n") or 1))
        loss_std = float(replication.get("loss_std") or 0.0)
        effective_threshold = effective_validation_threshold(
            min_score=min_score,
            replication_n=n_rep,
            loss_std=loss_std,
        )
        threshold_distance = abs(score - effective_threshold)
        threshold_proximity = max(0.0, 1.0 - min(1.0, threshold_distance / 10.0))
        uncertainty = min(1.0, 1.0 / math.sqrt(float(n_rep)))
        instability = min(1.0, loss_std / 0.12)
        novelty = float(row.get("novelty_score") or 0.0)
        info_gain = (
            0.35 * threshold_proximity
            + 0.25 * uncertainty
            + 0.20 * instability
            + 0.20 * novelty
        )
        enriched = dict(row)
        enriched["_active_learning"] = {
            "info_gain": round(info_gain, 6),
            "threshold_proximity": round(threshold_proximity, 6),
            "uncertainty": round(uncertainty, 6),
            "instability": round(instability, 6),
            "replication_n": n_rep,
            "loss_std": round(loss_std, 6),
            "effective_threshold": round(effective_threshold, 6),
        }
        ranked.append(enriched)
    ranked.sort(
        key=lambda row: (
            float((row.get("_active_learning") or {}).get("info_gain") or 0.0),
            float(composite_scores.get(str(row.get("result_id") or ""), min_score)),
        ),
        reverse=True,
    )
    return ranked


class _ActiveLearningFollowupMixin:
    def _active_learning_screening_rank(
        self,
        nb: LabNotebook,
        rows: List[Dict[str, Any]],
        score_map: Dict[str, float],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        return active_learning_screening_rank(nb, rows, score_map, threshold)

    @staticmethod
    def _followup_priority_summary(
        rows: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        return followup_priority_summary(rows)

    @staticmethod
    def _active_replay_suppressed_result_ids(nb: LabNotebook) -> set[str]:
        return active_replay_suppressed_result_ids(nb)

    def _active_replay_targets(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        rows: List[Dict[str, Any]],
        suppressed_ids: set[str],
    ) -> List[Dict[str, Any]]:
        return active_replay_targets(nb, config, rows, suppressed_ids)

    def _enqueue_active_learning_replay(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        replay_targets: List[Dict[str, Any]],
        source_context: str,
        source_experiment_id: str | None,
        suppressed_count: int,
    ) -> str | None:
        priority_score, priority_reasons = self._followup_priority_summary(
            replay_targets
        )
        result_ids = [
            str(row.get("result_id") or "").strip()
            for row in replay_targets
            if str(row.get("result_id") or "").strip()
        ]
        if not result_ids:
            return None
        evidence_pack = self._safe_build_evidence_pack(
            nb,
            recommendation={"mode": "exact_graph_replay"},
            decision_type="active_learning_replay",
        )
        hypothesis = (
            "Active-learning replay: re-measure ambiguous or unstable frontier "
            f"candidates before downstream promotion ({len(result_ids)} canonical graphs)."
        )
        task_id = nb.enqueue_followup_task(
            stage="replay",
            result_ids=result_ids,
            hypothesis=hypothesis,
            config={"device": config.device, "repeat_per_source": 2, "fast": True},
            evidence_pack=evidence_pack,
            source_context=source_context,
            source_experiment_id=source_experiment_id,
            priority_score=priority_score,
            priority_reasons=priority_reasons,
            metadata={
                "policy": "exact_graph_replay",
                "target_type": "ambiguous_unstable_frontier",
            },
        )
        self._emit_event(
            "active_learning_replay_queued",
            {
                "task_id": task_id,
                "result_ids": result_ids,
                "n_candidates": len(result_ids),
                "suppressed_candidate_count": suppressed_count,
                "priority_score": round(float(priority_score or 0.0), 6),
                "priority_reasons": priority_reasons,
                "evidence_pack": evidence_pack,
            },
        )
        return task_id

    def _active_learning_validation_rank(
        self,
        rows: List[Dict[str, Any]],
        composite_scores: Dict[str, float],
        replication_info: Dict[str, Dict[str, Any]],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        return active_learning_validation_rank(
            rows,
            composite_scores,
            replication_info,
            min_score,
        )

    def _queue_active_learning_replays(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        rows: List[Dict[str, Any]],
        source_context: str,
        source_experiment_id: str | None = None,
    ) -> str | None:
        """Queue exact replays for ambiguous, unstable frontier cases."""
        suppressed_ids = self._active_replay_suppressed_result_ids(nb)
        replay_targets = self._active_replay_targets(
            nb=nb,
            config=config,
            rows=rows,
            suppressed_ids=suppressed_ids,
        )
        if not replay_targets:
            return None

        return self._enqueue_active_learning_replay(
            nb=nb,
            config=config,
            replay_targets=replay_targets,
            source_context=source_context,
            source_experiment_id=source_experiment_id,
            suppressed_count=len(suppressed_ids),
        )


class _ActiveLearningFollowupMixin:
    def _active_learning_screening_rank(
        self,
        nb: LabNotebook,
        rows: List[Dict[str, Any]],
        score_map: Dict[str, float],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        return active_learning_screening_rank(nb, rows, score_map, threshold)

    @staticmethod
    def _followup_priority_summary(
        rows: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        return followup_priority_summary(rows)

    def _active_learning_validation_rank(
        self,
        rows: List[Dict[str, Any]],
        composite_scores: Dict[str, float],
        replication_info: Dict[str, Dict[str, Any]],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        return active_learning_validation_rank(
            rows,
            composite_scores,
            replication_info,
            min_score,
        )

    def _queue_active_learning_replays(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        rows: List[Dict[str, Any]],
        source_context: str,
        source_experiment_id: str | None = None,
    ) -> str | None:
        """Queue exact replays for ambiguous, unstable frontier cases."""
        suppressed_ids = active_replay_suppressed_result_ids(nb)
        replay_targets = active_replay_targets(nb, config, rows, suppressed_ids)
        if not replay_targets:
            return None
        return self._enqueue_active_learning_replay(
            nb=nb,
            config=config,
            replay_targets=replay_targets,
            source_context=source_context,
            source_experiment_id=source_experiment_id,
            suppressed_count=len(suppressed_ids),
        )

    def _enqueue_active_learning_replay(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        replay_targets: List[Dict[str, Any]],
        source_context: str,
        source_experiment_id: str | None,
        suppressed_count: int,
    ) -> str | None:
        priority_score, priority_reasons = followup_priority_summary(replay_targets)
        result_ids = [
            str(row.get("result_id") or "").strip()
            for row in replay_targets
            if str(row.get("result_id") or "").strip()
        ]
        if not result_ids:
            return None
        evidence_pack = self._safe_build_evidence_pack(
            nb,
            recommendation={"mode": "exact_graph_replay"},
            decision_type="active_learning_replay",
        )
        hypothesis = (
            "Active-learning replay: re-measure ambiguous or unstable frontier "
            f"candidates before downstream promotion ({len(result_ids)} canonical graphs)."
        )
        task_id = nb.enqueue_followup_task(
            stage="replay",
            result_ids=result_ids,
            hypothesis=hypothesis,
            config={"device": config.device, "repeat_per_source": 2, "fast": True},
            evidence_pack=evidence_pack,
            source_context=source_context,
            source_experiment_id=source_experiment_id,
            priority_score=priority_score,
            priority_reasons=priority_reasons,
            metadata={
                "policy": "exact_graph_replay",
                "target_type": "ambiguous_unstable_frontier",
            },
        )
        self._emit_event(
            "active_learning_replay_queued",
            {
                "task_id": task_id,
                "result_ids": result_ids,
                "n_candidates": len(result_ids),
                "suppressed_candidate_count": suppressed_count,
                "priority_score": round(float(priority_score or 0.0), 6),
                "priority_reasons": priority_reasons,
                "evidence_pack": evidence_pack,
            },
        )
        return task_id
