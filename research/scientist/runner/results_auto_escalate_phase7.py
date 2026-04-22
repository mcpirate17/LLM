"""Phase 7 split helpers for results._auto_escalate.

MIGRATION NOTE — loss_ratio formula (2026-03-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
program_results.loss_ratio has two historical formulas:

  RAW  = final_loss / initial_loss   (relative improvement, range 0–1+)
  NORM = final_loss / ln(vocab_size) (absolute position,    range 0–1+)

The auto-escalation threshold 0.18 was calibrated against RAW values.
Under NORM, a model with final_loss=2.0 scores 0.174 — the threshold is
nearly unreachable.

As of this commit, execution_training.py stores:
  loss_ratio      = RAW  (backward compatible)
  loss_ratio_raw  = RAW  (explicit)
  loss_ratio_norm = NORM (explicit)

All threshold comparisons in this file use loss_ratio (= RAW).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from typing import Any, Dict, List, Tuple

from .auto_escalate_data import (
    composite_score_map,
    effective_validation_threshold,
    graph_meta_by_result_id,
    investigation_support_data,
    novelty_metadata,
    trusted_global_screening_candidates,
    trusted_screening_candidates,
)
from .auto_escalate_flow import (
    build_selected_screening_ids,
    build_selection_decision_payload,
    filter_uninvestigated_rows,
    merge_unique_result_rows,
    prepare_validation_candidates,
    screening_candidates_above_threshold,
    screening_understanding_filter,
    sparse_dense_learning_signal,
    strong_investigation_candidates,
)
from ..evidence import validate_selection_decision_log
from ..llm.context_experiment import build_go_no_go_context
from ..notebook import ExperimentEntry, LabNotebook
from ..thresholds import (
    EMPIRICAL_OVERRIDE_BASELINE_LR,
    EMPIRICAL_OVERRIDE_BEST_LR,
    EMPIRICAL_OVERRIDE_ROBUSTNESS,
    EMPIRICAL_OVERRIDE_SCORE_MULT,
    V7_INVESTIGATION_THRESHOLD,
    V7_SCREENING_THRESHOLD,
)
from ..trust_policy import sql_trusted_clause
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ResultsAutoEscalatePhase7Mixin:
    """Branch helpers extracted from _auto_escalate orchestration."""

    @staticmethod
    def _meets_empirical_validation_override(
        candidate: Dict[str, Any],
        candidate_score: float,
        min_score: float,
    ) -> bool:
        robustness = float(candidate.get("robustness") or 0.0)
        best_loss_ratio = float(candidate.get("best_loss_ratio") or 1.0)
        baseline_loss_ratio = candidate.get("baseline_loss_ratio")
        baseline_value = (
            float(baseline_loss_ratio) if baseline_loss_ratio is not None else None
        )
        novelty_confidence = candidate.get("novelty_confidence")
        # Allow near-known-family architectures through when investigation-time
        # evidence is dominant enough that novelty-confidence and missing/noisy
        # baseline comparisons should affect ranking, not veto progression.
        # Still require novelty_confidence to exist — completely missing evidence
        # should not be overridden.
        if novelty_confidence is None:
            return False
        if robustness < EMPIRICAL_OVERRIDE_ROBUSTNESS:
            return False
        if best_loss_ratio >= EMPIRICAL_OVERRIDE_BEST_LR:
            return False
        if (
            baseline_value is not None
            and baseline_value >= EMPIRICAL_OVERRIDE_BASELINE_LR
        ):
            return False
        if min_score > 0 and candidate_score < (
            min_score * EMPIRICAL_OVERRIDE_SCORE_MULT
        ):
            return False
        return True

    def _active_learning_screening_rank(
        self,
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
            threshold_distance = abs(
                float(score_map.get(result_id, threshold)) - threshold
            )
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

    @staticmethod
    def _followup_priority_summary(
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

    def _active_learning_validation_rank(
        self,
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

    def _queue_active_learning_replays(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        rows: List[Dict[str, Any]],
        source_context: str,
        source_experiment_id: str | None = None,
    ) -> str | None:
        """Queue exact replays for ambiguous, unstable frontier cases.

        This does not invent a new evaluation path. It routes canonical
        result_ids through the existing exact_graph_replay pipeline.
        """
        suppressed_ids: set[str] = set()
        active_statuses = ("queued", "running")
        for status in active_statuses:
            for task in nb.get_followup_tasks(stage="replay", status=status, limit=200):
                suppressed_ids.update(
                    str(rid).strip()
                    for rid in (task.get("result_ids_json") or [])
                    if str(rid).strip()
                )
        recent_cutoff = time.time() - (12.0 * 3600.0)
        for task in nb.get_followup_tasks(
            stage="replay", status="completed", limit=200
        ):
            completed_ts = float(task.get("completed_timestamp") or 0.0)
            if completed_ts < recent_cutoff:
                continue
            suppressed_ids.update(
                str(rid).strip()
                for rid in (task.get("result_ids_json") or [])
                if str(rid).strip()
            )

        replay_targets: List[Dict[str, Any]] = []
        seen_canonical_ids: set[str] = set()
        max_targets = max(1, min(2, int(config.auto_investigate_top_n or 1)))
        for row in rows:
            details = row.get("_active_learning") or {}
            info_gain = float(details.get("info_gain") or 0.0)
            threshold_proximity = float(details.get("threshold_proximity") or 0.0)
            ambiguity = float(details.get("ambiguity") or 0.0)
            instability = float(details.get("instability") or 0.0)
            n_runs = int(details.get("n_runs") or 0)
            canonical_id = str(
                nb.resolve_canonical_result_id(str(row.get("result_id") or "").strip())
                or row.get("result_id")
                or ""
            ).strip()
            if not canonical_id:
                continue
            if canonical_id in seen_canonical_ids or canonical_id in suppressed_ids:
                continue
            if info_gain < 0.60:
                continue
            if threshold_proximity < 0.80:
                continue
            if ambiguity < 0.30 and instability < 0.20:
                continue
            if n_runs < 1:
                continue
            replay_target = dict(row)
            replay_target["result_id"] = canonical_id
            replay_targets.append(replay_target)
            seen_canonical_ids.add(canonical_id)
            if len(replay_targets) >= max_targets:
                break
        if not replay_targets:
            return None

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
            config={
                "device": config.device,
                "repeat_per_source": 2,
                "fast": True,
            },
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
                "suppressed_candidate_count": len(suppressed_ids),
                "priority_score": round(float(priority_score or 0.0), 6),
                "priority_reasons": priority_reasons,
                "evidence_pack": evidence_pack,
            },
        )
        return task_id

    def sweep_backfill_candidates(self, config: RunConfig, nb: LabNotebook) -> int:
        """Standalone global sweep for accumulated backfill survivors.

        Runs independently of any experiment.  Finds screening-tier entries
        above the composite-score threshold and queues them for investigation.
        Called at continuous-mode startup and periodically between cycles.

        Returns number of candidates queued for investigation.
        """
        if not config.auto_investigate:
            return 0

        top = trusted_global_screening_candidates(
            nb,
            limit=config.auto_investigate_top_n * 3,  # wider net
        )
        if not top:
            return 0

        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            top, _skipped = filter_uninvestigated_rows(top, investigated_fps)

        _screening_threshold = V7_SCREENING_THRESHOLD
        if config.adaptive_thresholds_enabled:
            _screening_threshold = self._adaptive_screening_threshold(
                nb, config, _screening_threshold
            )

        _cs_map = composite_score_map(nb, (p.get("result_id") for p in top))
        replay_ranked = self._active_learning_screening_rank(
            nb,
            top,
            _cs_map,
            _screening_threshold,
        )
        self._queue_active_learning_replays(
            nb=nb,
            config=config,
            rows=replay_ranked,
            source_context="backfill_sweep_replay",
        )
        qualified = screening_candidates_above_threshold(
            top, _cs_map, _screening_threshold
        )
        if not qualified:
            return 0

        qualified = self._active_learning_screening_rank(
            nb,
            qualified,
            _cs_map,
            _screening_threshold,
        )

        top_by_id = {row["result_id"]: row for row in qualified if row.get("result_id")}
        selected_ids = build_selected_screening_ids(
            qualified, top_by_id, limit=config.auto_investigate_top_n
        )
        if not selected_ids:
            return 0
        selected_rows = [top_by_id[rid] for rid in selected_ids if rid in top_by_id]
        priority_score, priority_reasons = self._followup_priority_summary(
            selected_rows
        )

        logger.info(
            "Backfill sweep: %d candidates above %.1f — queuing %d for investigation",
            len(qualified),
            _screening_threshold,
            len(selected_ids),
        )
        self._queue_pending_followup(
            nb=nb,
            stage="investigation",
            result_ids=selected_ids,
            config=config,
            survivor_count=len(qualified),
            qualifying_count=len(selected_ids),
            source_context="backfill_sweep_screening",
            priority_score=priority_score,
            priority_reasons=priority_reasons,
        )
        return len(selected_ids)

    def _auto_escalate(
        self,
        results: Dict,
        config: RunConfig,
        nb: LabNotebook,
        phase: str = "screening",
    ) -> None:
        """Auto-escalate candidates through the research pipeline."""
        if phase in ("screening", "experiment"):
            self._auto_escalate_screening(results, config, nb)
        elif phase == "investigation":
            self._auto_escalate_investigation(results, config, nb)

    @staticmethod
    def _record_selection_decision(
        nb: LabNotebook,
        *,
        decision_payload: Dict[str, Any],
        candidate_ids: List[str],
        supporting_insight_ids: List[str],
        source_experiment_id: str | None,
        failure_log_label: str,
    ) -> str | None:
        try:
            validate_selection_decision_log(decision_payload)
            decision_id = nb.record_selection_decision(
                context=decision_payload["context"],
                experiment_id=decision_payload["experiment_id"],
                candidate_pool_summary=decision_payload["candidate_pool_summary"],
                score_breakdown=decision_payload["score_breakdown"],
                policy=decision_payload["policy"],
                reason=decision_payload["reason"],
                chosen_experiments=decision_payload["chosen_experiments"],
                trigger=None,
            )
            if supporting_insight_ids:
                nb.record_selection_insight_trial(
                    decision_id=decision_id,
                    context=decision_payload["context"],
                    insight_ids=supporting_insight_ids,
                    chosen_result_ids=candidate_ids,
                    source_experiment_id=str(source_experiment_id or ""),
                )
            return decision_id
        except (ValueError, sqlite3.OperationalError) as error:
            logger.debug("%s: %s", failure_log_label, error)
        return None

    def _approved_screening_candidate_ids(
        self,
        *,
        nb: LabNotebook,
        config: RunConfig,
        selected_rows: List[Dict[str, Any]],
    ) -> List[str]:
        if not (config.auto_go_no_go and config.enable_campaigns):
            return [row["result_id"] for row in selected_rows if row.get("result_id")]

        try:
            existing_decisions = nb.get_decisions(campaign_id=self._active_campaign_id)
        except (sqlite3.OperationalError, KeyError) as error:
            logger.debug("Failed to fetch existing decisions: %s", error)
            existing_decisions = []
        already_decided = {
            result_id
            for decision in existing_decisions
            for result_id in (decision.get("evidence_ids") or [])
        }
        approved_ids: List[str] = []
        campaign = nb.get_campaign(self._active_campaign_id or "") or {}
        campaign_criteria = campaign.get("success_criteria", "")

        for row in selected_rows:
            result_id = row.get("result_id")
            if not result_id:
                continue
            if result_id in already_decided:
                approved_ids.append(result_id)
                continue
            try:
                go_context = build_go_no_go_context(
                    candidate=row,
                    campaign_criteria=campaign_criteria,
                )
                decision = self.aria.generate_go_no_go(
                    subject=f"Promote {result_id[:8]} to investigation",
                    evidence=f"loss_ratio={row.get('loss_ratio', '?')}, novelty={row.get('novelty_score', '?')}",
                    context=go_context,
                )
                evidence_pack = self._safe_build_evidence_pack(
                    nb,
                    recommendation={"mode": "investigation"},
                    decision_type="go_no_go",
                )
                nb.record_decision(
                    campaign_id=self._active_campaign_id,
                    decision_type=decision["decision"],
                    subject=f"Promote {result_id[:8]} to investigation",
                    rationale=decision["rationale"],
                    evidence_ids=[result_id],
                    alternatives=[{"considered": decision.get("alternatives", "")}],
                    evidence_pack=evidence_pack,
                )
                self._emit_event(
                    "decision_recorded",
                    {
                        "decision_type": decision["decision"],
                        "subject": result_id[:8],
                        "rationale": decision["rationale"][:200],
                        "evidence_pack": evidence_pack,
                    },
                )
                if decision["decision"] in ("go", "pivot"):
                    approved_ids.append(result_id)
            except (RuntimeError, ValueError, KeyError) as error:
                logger.debug("Go/no-go failed for %s: %s", result_id, error)
        return approved_ids

    def _queue_pending_followup(
        self,
        *,
        nb: LabNotebook,
        stage: str,
        result_ids: List[str],
        config: RunConfig,
        blocked_incomplete_fingerprint: int | None = None,
        survivor_count: int | None = None,
        qualifying_count: int | None = None,
        source_context: str | None = None,
        source_decision_id: str | None = None,
        source_experiment_id: str | None = None,
        priority_score: float = 0.0,
        priority_reasons: Dict[str, Any] | None = None,
    ) -> str | None:
        if stage == "investigation":
            hypothesis = (
                f"Auto-investigation: testing robustness of top "
                f"{len(result_ids)} screening survivors with "
                f"{config.n_training_programs} training programs each."
            )
            evidence_pack = self._safe_build_evidence_pack(
                nb,
                recommendation={"mode": "investigation"},
                decision_type="auto_investigate",
            )
            task_id = nb.enqueue_followup_task(
                stage="investigation",
                result_ids=result_ids,
                hypothesis=hypothesis,
                config=config.to_dict(),
                evidence_pack=evidence_pack,
                source_context=source_context or "auto_investigate_screening",
                source_decision_id=source_decision_id,
                source_experiment_id=source_experiment_id,
                priority_score=priority_score,
                priority_reasons=priority_reasons,
                metadata={
                    "survivor_count": survivor_count,
                    "qualifying_count": qualifying_count,
                    "blocked_incomplete_fingerprint": blocked_incomplete_fingerprint,
                },
            )
            self._pending_investigation = {
                "result_ids": result_ids,
                "config": config,
                "hypothesis": hypothesis,
                "task_id": task_id,
            }
            self._pending_investigation["evidence_pack"] = evidence_pack
            self._emit_event(
                "auto_investigate_queued",
                {
                    "task_id": task_id,
                    "result_ids": result_ids,
                    "n_candidates": len(result_ids),
                    "reason": f"{survivor_count} S1 survivors with loss_ratio < 0.5",
                    "priority_score": round(float(priority_score or 0.0), 6),
                    "priority_reasons": priority_reasons or {},
                    "evidence_pack": evidence_pack,
                },
            )
            nb.add_entry(
                ExperimentEntry(
                    entry_type="decision",
                    title="Auto-Investigation Triggered",
                    content=(
                        f"Automatically queuing investigation for {len(result_ids)} "
                        f"top performers. Criteria: {survivor_count} S1 survivors."
                    ),
                    metadata={
                        "task_id": task_id,
                        "result_ids": result_ids,
                        "priority_score": priority_score,
                        "priority_reasons": priority_reasons or {},
                        "evidence_pack": evidence_pack,
                    },
                )
            )
            return task_id

        hypothesis = (
            f"Auto-validation: publication-grade testing of "
            f"{len(result_ids)} robust investigation survivors."
        )
        evidence_pack = self._safe_build_evidence_pack(
            nb,
            recommendation={"mode": "validation"},
            decision_type="auto_validate",
        )
        task_id = nb.enqueue_followup_task(
            stage="validation",
            result_ids=result_ids,
            hypothesis=hypothesis,
            config=config.to_dict(),
            evidence_pack=evidence_pack,
            source_context=source_context or "auto_validate_investigation",
            source_decision_id=source_decision_id,
            source_experiment_id=source_experiment_id,
            priority_score=priority_score,
            priority_reasons=priority_reasons,
            metadata={
                "survivor_count": survivor_count,
                "qualifying_count": qualifying_count,
                "blocked_incomplete_fingerprint": blocked_incomplete_fingerprint,
            },
        )
        self._pending_validation = {
            "result_ids": result_ids,
            "config": config,
            "hypothesis": hypothesis,
            "task_id": task_id,
        }
        self._pending_validation["evidence_pack"] = evidence_pack
        self._emit_event(
            "auto_validate_queued",
            {
                "task_id": task_id,
                "result_ids": result_ids,
                "n_candidates": len(result_ids),
                "blocked_incomplete_fingerprint": blocked_incomplete_fingerprint,
                "reason": f"{qualifying_count} candidates passed fingerprint + novelty + "
                f"robustness >= {config.auto_validate_min_robustness} gates",
                "priority_score": round(float(priority_score or 0.0), 6),
                "priority_reasons": priority_reasons or {},
                "evidence_pack": evidence_pack,
            },
        )
        nb.add_entry(
            ExperimentEntry(
                entry_type="decision",
                title="Auto-Validation Triggered",
                content=(
                    f"Automatically queuing validation for {len(result_ids)} "
                    f"robust investigation survivors."
                ),
                metadata={
                    "task_id": task_id,
                    "result_ids": result_ids,
                    "priority_score": priority_score,
                    "priority_reasons": priority_reasons or {},
                    "evidence_pack": evidence_pack,
                },
            )
        )
        return task_id

    @staticmethod
    def _apply_sparse_learning_signal(
        nb: LabNotebook,
        config: RunConfig,
        top_rows: List[Dict[str, Any]],
    ) -> None:
        learning_signal = sparse_dense_learning_signal(top_rows)
        if learning_signal is None:
            return
        avg_sparse_loss, avg_dense_loss = learning_signal
        if avg_sparse_loss >= avg_dense_loss * 0.95:
            return
        delta = 0.1
        old_bias = config.grammar_config.structured_sparsity_bias
        config.grammar_config.update_bias(delta)
        nb.log_learning_event(
            event_type="grammar_adjustment",
            description=f"Boosted structured_sparsity_bias by {delta} due to sparse dominance.",
            old_weights={"bias": old_bias},
            new_weights={"bias": config.grammar_config.structured_sparsity_bias},
            evidence=f"avg_sparse_loss={avg_sparse_loss:.4f}, avg_dense_loss={avg_dense_loss:.4f}",
        )

    def _auto_escalate_screening(
        self, results: Dict, config: RunConfig, nb: LabNotebook
    ) -> None:
        if not config.auto_investigate:
            return

        exp_id = results.get("experiment_id")
        s1_count = results.get("stage1_passed", 0)

        # Gather experiment-local candidates (if enough S1 passers)
        top: list = []
        if s1_count >= config.auto_investigate_min_survivors:
            top = trusted_screening_candidates(
                nb,
                experiment_id=exp_id,
                limit=config.auto_investigate_top_n,
            )

        # Always run global sweep — backfill experiments may produce
        # fewer than min_survivors but still push prior entries above
        # the composite threshold.
        try:
            global_rows = trusted_global_screening_candidates(
                nb, limit=config.auto_investigate_top_n
            )
            top = merge_unique_result_rows(top, global_rows)
            if global_rows:
                logger.info(
                    "Auto-escalate: global sweep found %d leaderboard candidates",
                    len(global_rows),
                )
        except sqlite3.OperationalError as e:
            logger.warning("Auto-escalate global sweep failed: %s", e)

        if not top:
            logger.info("Auto-escalate: no candidates from experiment or global sweep")
            return

        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            top, skipped = filter_uninvestigated_rows(top, investigated_fps)
            if skipped:
                logger.info(
                    "Auto-escalate: skipped %d already-investigated archs", skipped
                )

        # Content-addressed gate: reject candidates whose graph has no
        # attention-class op. Without Q·K content-based routing, a model
        # cannot develop induction heads or binding — it will score well
        # on loss but fail every understanding probe at investigation.
        from ..runner.execution_screening_graphs import CONTENT_ADDRESSED_OPS

        def _has_content_addressing(row: Dict) -> bool:
            gj = row.get("graph_json")
            if not gj:
                # No graph_json on the row at all — can't audit; allow but log.
                # This is rare; usually a fingerprint-only row from a backfill.
                logger.debug(
                    "Auto-escalate content-addressing: no graph_json for %s; allowing",
                    str(row.get("result_id", ""))[:12],
                )
                return True
            try:
                data = json.loads(gj)
                nodes = data.get("nodes", {})
                ops = {
                    n.get("op_name")
                    for n in nodes.values()
                    if isinstance(n, dict) and n.get("op_name")
                }
                return bool(ops & CONTENT_ADDRESSED_OPS)
            except (json.JSONDecodeError, TypeError) as e:
                # Malformed graph_json should not be a free pass: a model that
                # cannot be audited cannot be promoted. Reject and log.
                logger.warning(
                    "Auto-escalate content-addressing: graph_json unparseable for %s "
                    "(%s); rejecting candidate",
                    str(row.get("result_id", ""))[:12],
                    e,
                )
                return False

        before_ca = len(top)
        top = [r for r in top if _has_content_addressing(r)]
        ca_rejected = before_ca - len(top)
        if ca_rejected:
            logger.info(
                "Auto-escalate: rejected %d candidates with no content-addressed ops "
                "(no attention/matmul → cannot develop induction/binding)",
                ca_rejected,
            )

        # Capability filter: drop candidates whose probes have already been
        # measured and are uniformly near-zero. Most candidates pass this
        # vacuously (no probe data yet); the filter only fires for
        # re-screened rows whose investigation tier proved them incapable.
        try:
            _top_ids = [str(r.get("result_id")) for r in top if r.get("result_id")]
            _, _, _understanding_pre = investigation_support_data(nb, _top_ids)
            before_uf = len(top)
            kept: list = []
            blocked = 0
            for r in top:
                rid = str(r.get("result_id") or "")
                allow, reason = screening_understanding_filter(
                    _understanding_pre.get(rid, {})
                )
                if allow:
                    kept.append(r)
                else:
                    blocked += 1
                    logger.info(
                        "Auto-escalate: rejected %s at screening understanding filter (%s)",
                        rid[:12],
                        reason,
                    )
            top = kept
            if blocked:
                logger.info(
                    "Auto-escalate: %d/%d candidates blocked by screening understanding filter",
                    blocked,
                    before_uf,
                )
        except (sqlite3.OperationalError, ValueError, TypeError) as _uf_err:
            logger.debug("Screening understanding filter skipped: %s", _uf_err)

        # v7 screening → investigation threshold: see thresholds.py for calibration.
        _screening_floor = V7_SCREENING_THRESHOLD
        _screening_threshold = _screening_floor
        if config.adaptive_thresholds_enabled:
            _screening_threshold = self._adaptive_screening_threshold(
                nb, config, _screening_floor
            )
        try:
            before = len(top)
            _cs_map = composite_score_map(nb, (p.get("result_id") for p in top))
            replay_ranked = self._active_learning_screening_rank(
                nb,
                top,
                _cs_map,
                _screening_threshold,
            )
            self._queue_active_learning_replays(
                nb=nb,
                config=config,
                rows=replay_ranked,
                source_context="auto_investigate_replay",
                source_experiment_id=str(exp_id or ""),
            )
            qualified = screening_candidates_above_threshold(
                top,
                _cs_map,
                _screening_threshold,
            )
            if qualified:
                top = qualified
                logger.info(
                    "Auto-escalate: v7 screening threshold %.1f "
                    "(floor=%.1f, adaptive=%s), %d/%d candidates qualify",
                    _screening_threshold,
                    _screening_floor,
                    config.adaptive_thresholds_enabled,
                    len(top),
                    before,
                )
            else:
                logger.info(
                    "Auto-escalate: no candidates meet v7 threshold %.1f, "
                    "skipping investigation",
                    _screening_threshold,
                )
                return
        except (sqlite3.OperationalError, ValueError, TypeError) as e:
            logger.debug("Auto-escalate score floor check failed: %s", e)

        selection = self._score_candidate_pool(
            candidates=top,
            config=config,
            nb=nb,
            context="auto_investigate_screening",
            experiment_id=exp_id,
        )
        scored_by_id = {s["result_id"]: s for s in selection.get("scored", [])}
        ranked = selection.get("selected", [])
        top_by_id = {row["result_id"]: row for row in top if row.get("result_id")}
        candidate_ids = build_selected_screening_ids(
            ranked,
            top_by_id,
            limit=config.auto_investigate_top_n,
        )

        if len(candidate_ids) < config.auto_investigate_min_survivors:
            return
        selected_rows = [top_by_id[rid] for rid in candidate_ids if rid in top_by_id]
        decision_payload = build_selection_decision_payload(
            context="auto_investigate_screening",
            experiment_id=exp_id,
            selection=selection,
            candidate_ids=candidate_ids,
            scored_by_id=scored_by_id,
        )
        decision_id = self._record_selection_decision(
            nb,
            decision_payload=decision_payload,
            candidate_ids=candidate_ids,
            supporting_insight_ids=selection.get("supporting_insight_ids") or [],
            source_experiment_id=exp_id,
            failure_log_label="Auto-investigate selection logging failed",
        )

        candidate_ids = self._approved_screening_candidate_ids(
            nb=nb,
            config=config,
            selected_rows=selected_rows,
        )
        selected_rows = [
            row for row in selected_rows if row.get("result_id") in candidate_ids
        ]
        if not candidate_ids:
            return

        for rid in candidate_ids:
            score_row = scored_by_id.get(rid)
            if not score_row or not decision_id:
                continue
            nb.record_selection_family_trial(
                decision_id=decision_id,
                context="auto_investigate_screening",
                family=str(score_row.get("family") or "Unknown"),
                chosen_result_ids=[rid],
                source_experiment_id=str(exp_id or ""),
            )
        selected_row_map = {
            str(row.get("result_id") or ""): row for row in selected_rows
        }
        queue_rows = self._active_learning_screening_rank(
            nb,
            [selected_row_map[rid] for rid in candidate_ids if rid in selected_row_map],
            _cs_map,
            _screening_threshold,
        )
        priority_score, priority_reasons = self._followup_priority_summary(queue_rows)

        # Leaderboard entries are created at S1-pass time in dashboard.py
        # via _upsert_screening_entry(). No need to duplicate here.
        self._queue_pending_followup(
            nb=nb,
            stage="investigation",
            result_ids=candidate_ids,
            config=config,
            survivor_count=s1_count,
            source_context="auto_investigate_screening",
            source_decision_id=decision_id,
            source_experiment_id=str(exp_id or ""),
            priority_score=priority_score,
            priority_reasons=priority_reasons,
        )

        try:
            self._apply_sparse_learning_signal(nb, config, top)
        except (
            ValueError,
            TypeError,
            AttributeError,
            sqlite3.OperationalError,
        ) as z7_err:
            logger.debug("Z7 learning logic failed: %s", z7_err)

    def _auto_escalate_investigation(
        self, results: Dict, config: RunConfig, nb: LabNotebook
    ) -> None:
        if not config.auto_validate:
            return

        inv_results = results.get("investigation_results", [])
        inv_ids = [r.get("result_id") for r in inv_results if r.get("result_id")]
        novelty_meta = novelty_metadata(nb, inv_ids)

        # v7 investigation → validation threshold: see thresholds.py for calibration.
        _inv_floor = max(
            config.auto_validate_min_composite_score,
            V7_INVESTIGATION_THRESHOLD,
        )
        min_score = _inv_floor
        if config.adaptive_thresholds_enabled:
            min_score = self._adaptive_investigation_threshold(nb, config, _inv_floor)

        inv_id_list = [r.get("result_id") for r in inv_results if r.get("result_id")]
        composite_scores, replication_info, _understanding_data = (
            investigation_support_data(nb, inv_id_list)
        )

        strong, blocked_incomplete_fingerprint = strong_investigation_candidates(
            inv_results=inv_results,
            novelty_meta=novelty_meta,
            composite_scores=composite_scores,
            replication_info=replication_info,
            understanding_data=_understanding_data,
            min_score=min_score,
            config=config,
            threshold_for_replication=effective_validation_threshold,
            meets_empirical_override=self._meets_empirical_validation_override,
            logger=logger,
        )

        if blocked_incomplete_fingerprint:
            logger.info(
                "Auto-validate: blocked %d candidates with incomplete fingerprints",
                blocked_incomplete_fingerprint,
            )

        if not strong:
            return

        ranked_strong = self._active_learning_validation_rank(
            strong,
            composite_scores,
            replication_info,
            min_score,
        )
        result_ids_all = [r.get("result_id") for r in strong if r.get("result_id")]
        graph_meta = graph_meta_by_result_id(nb, result_ids_all)

        prepared_candidates = prepare_validation_candidates(ranked_strong, graph_meta)

        selection = self._score_candidate_pool(
            candidates=prepared_candidates,
            config=config,
            nb=nb,
            context="auto_validate_investigation",
            experiment_id=results.get("experiment_id"),
        )
        scored_by_id = {s["result_id"]: s for s in selection.get("scored", [])}
        ranked = selection.get("selected", [])
        candidate_ids = [
            item["result_id"] for item in ranked[: config.auto_validate_top_n]
        ]
        decision_payload = build_selection_decision_payload(
            context="auto_validate_investigation",
            experiment_id=results.get("experiment_id"),
            selection=selection,
            candidate_ids=candidate_ids,
            scored_by_id=scored_by_id,
        )
        decision_id = self._record_selection_decision(
            nb,
            decision_payload=decision_payload,
            candidate_ids=candidate_ids,
            supporting_insight_ids=selection.get("supporting_insight_ids") or [],
            source_experiment_id=results.get("experiment_id"),
            failure_log_label="Auto-validate selection logging failed",
        )

        for rid in candidate_ids:
            score_row = scored_by_id.get(rid)
            if not score_row or not decision_id:
                continue
            nb.record_selection_family_trial(
                decision_id=decision_id,
                context="auto_validate_investigation",
                family=str(score_row.get("family") or "Unknown"),
                chosen_result_ids=[rid],
                source_experiment_id=str(results.get("experiment_id") or ""),
            )
        ranked_strong_by_id = {
            str(row.get("result_id") or ""): row for row in ranked_strong
        }
        queue_rows = [
            ranked_strong_by_id[rid]
            for rid in candidate_ids
            if rid in ranked_strong_by_id
        ]
        priority_score, priority_reasons = self._followup_priority_summary(queue_rows)

        self._queue_pending_followup(
            nb=nb,
            stage="validation",
            result_ids=candidate_ids,
            config=config,
            blocked_incomplete_fingerprint=blocked_incomplete_fingerprint,
            qualifying_count=len(strong),
            source_context="auto_validate_investigation",
            source_decision_id=decision_id,
            source_experiment_id=str(results.get("experiment_id") or ""),
            priority_score=priority_score,
            priority_reasons=priority_reasons,
        )

    @staticmethod
    def _recent_threshold_scores(
        nb: LabNotebook,
        *,
        tier_clause: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        rows = nb.conn.execute(
            f"""SELECT l.result_id, l.composite_score,
                       l.investigation_passed, l.investigation_loss_ratio,
                       l.investigation_robustness, l.validation_passed,
                       l.validation_loss_ratio, l.validation_baseline_ratio,
                       l.validation_multi_seed_std
                FROM leaderboard l
                WHERE {tier_clause}
                  AND l.composite_score IS NOT NULL
                  AND COALESCE(l.is_reference, 0) = 0
                  AND {sql_trusted_clause(table_alias="l")}
                ORDER BY l.rowid DESC
                LIMIT ?""",
            (max(20, int(limit)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def _calibrated_promotion_threshold(
        self,
        nb: LabNotebook,
        *,
        tier_clause: str,
        floor: float,
        percentile: float,
        context: str,
    ) -> float:
        """Pick a stage threshold from recent realized downstream outcomes."""
        try:
            import numpy as np

            rows = self._recent_threshold_scores(nb, tier_clause=tier_clause, limit=250)
            sample_size = len(rows)
            if len(rows) < 20:
                nb.record_threshold_calibration(
                    context=context,
                    tier_clause=tier_clause,
                    floor=floor,
                    percentile=percentile,
                    selected_threshold=floor,
                    fallback_threshold=floor,
                    sample_size=sample_size,
                    labeled_size=0,
                    metrics={"mode": "floor_fallback"},
                    metadata={"reason": "insufficient_rows"},
                )
                return floor

            scores = np.array(
                [float(row["composite_score"]) for row in rows],
                dtype=np.float64,
            )
            pct_value = float(np.percentile(scores, percentile))
            fallback_threshold = max(pct_value, floor)

            if context == "auto_investigate_screening":
                reward_fn = self._selection_insight_reward_investigate
            elif context == "auto_validate_investigation":
                reward_fn = self._selection_insight_reward_validate
            else:
                nb.record_threshold_calibration(
                    context=context,
                    tier_clause=tier_clause,
                    floor=floor,
                    percentile=percentile,
                    selected_threshold=fallback_threshold,
                    fallback_threshold=fallback_threshold,
                    sample_size=sample_size,
                    labeled_size=0,
                    metrics={"mode": "fallback"},
                    metadata={"reason": "unsupported_context"},
                )
                return fallback_threshold

            labeled: List[Tuple[float, float]] = []
            for row in rows:
                reward = reward_fn(row)
                if reward is not None:
                    labeled.append((float(row["composite_score"]), float(reward)))
            positive_count = sum(1 for _, reward in labeled if reward >= 0.55)
            negative_count = sum(1 for _, reward in labeled if reward <= 0.45)
            if len(labeled) < 24:
                nb.record_threshold_calibration(
                    context=context,
                    tier_clause=tier_clause,
                    floor=floor,
                    percentile=percentile,
                    selected_threshold=fallback_threshold,
                    fallback_threshold=fallback_threshold,
                    sample_size=sample_size,
                    labeled_size=len(labeled),
                    positive_count=positive_count,
                    negative_count=negative_count,
                    metrics={"mode": "fallback"},
                    metadata={"reason": "insufficient_labeled_rows"},
                )
                return fallback_threshold

            candidate_thresholds = np.unique(
                np.quantile(
                    np.array([score for score, _ in labeled], dtype=np.float64),
                    np.linspace(0.10, 0.95, 32),
                )
            )

            best_threshold = fallback_threshold
            best_objective = float("-inf")
            best_metrics: Dict[str, Any] = {
                "mode": "fallback",
                "promoted_count": 0,
                "held_count": len(labeled),
            }
            for threshold in candidate_thresholds:
                promoted = [reward for score, reward in labeled if score >= threshold]
                held = [reward for score, reward in labeled if score < threshold]
                if len(promoted) < 5:
                    continue
                tp = sum(1 for reward in promoted if reward >= 0.55)
                fp = sum(1 for reward in promoted if reward <= 0.45)
                fn = sum(1 for reward in held if reward >= 0.55)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = (
                    (2.0 * precision * recall / (precision + recall))
                    if (precision + recall)
                    else 0.0
                )
                avg_reward = sum(promoted) / max(len(promoted), 1)
                rejection_quality = (
                    sum(1 for reward in held if reward <= 0.45) / max(len(held), 1)
                    if held
                    else 0.0
                )
                objective = 0.55 * f1 + 0.30 * avg_reward + 0.15 * rejection_quality
                if objective > best_objective:
                    best_objective = objective
                    best_threshold = float(threshold)
                    best_metrics = {
                        "mode": "adaptive",
                        "precision": round(precision, 6),
                        "recall": round(recall, 6),
                        "f1": round(f1, 6),
                        "avg_reward": round(avg_reward, 6),
                        "rejection_quality": round(rejection_quality, 6),
                        "promoted_count": len(promoted),
                        "held_count": len(held),
                        "selected_quantile": round(
                            float(
                                np.mean(
                                    np.array(
                                        [score <= threshold for score, _ in labeled],
                                        dtype=np.float64,
                                    )
                                )
                            ),
                            6,
                        ),
                    }

            selected_threshold = max(float(best_threshold), float(floor))
            nb.record_threshold_calibration(
                context=context,
                tier_clause=tier_clause,
                floor=floor,
                percentile=percentile,
                selected_threshold=selected_threshold,
                fallback_threshold=fallback_threshold,
                sample_size=sample_size,
                labeled_size=len(labeled),
                positive_count=positive_count,
                negative_count=negative_count,
                objective=best_objective if math.isfinite(best_objective) else None,
                metrics=best_metrics,
                metadata={"candidate_threshold_count": len(candidate_thresholds)},
            )
            return selected_threshold
        except (ImportError, sqlite3.OperationalError, ValueError, TypeError) as e:
            logger.warning(
                "Adaptive threshold calibration failed: %s, using floor %.1f",
                e,
                floor,
            )
            return floor

    def _adaptive_screening_threshold(
        self, nb: LabNotebook, config: RunConfig, floor: float
    ) -> float:
        threshold = self._calibrated_promotion_threshold(
            nb,
            tier_clause="l.tier = 'screening'",
            floor=floor,
            percentile=float(config.screening_promotion_percentile),
            context="auto_investigate_screening",
        )
        logger.info(
            "Adaptive screening threshold: floor=%.1f, using=%.1f",
            floor,
            threshold,
        )
        return threshold

    def _adaptive_investigation_threshold(
        self, nb: LabNotebook, config: RunConfig, floor: float
    ) -> float:
        threshold = self._calibrated_promotion_threshold(
            nb,
            tier_clause=(
                "l.tier IN ('investigation', 'investigation_failed', "
                "'investigation_fingerprint_incomplete')"
            ),
            floor=floor,
            percentile=float(config.investigation_promotion_percentile),
            context="auto_validate_investigation",
        )
        logger.info(
            "Adaptive investigation threshold: floor=%.1f, using=%.1f",
            floor,
            threshold,
        )
        return threshold
