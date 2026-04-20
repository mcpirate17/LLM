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
import sqlite3
from typing import Any, Dict, List

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
        qualified = screening_candidates_above_threshold(
            top, _cs_map, _screening_threshold
        )
        if not qualified:
            return 0

        top_by_id = {row["result_id"]: row for row in qualified if row.get("result_id")}
        selected_ids = build_selected_screening_ids(
            qualified, top_by_id, limit=config.auto_investigate_top_n
        )
        if not selected_ids:
            return 0

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
    ) -> None:
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
        except (ValueError, sqlite3.OperationalError) as error:
            logger.debug("%s: %s", failure_log_label, error)

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
    ) -> None:
        if stage == "investigation":
            self._pending_investigation = {
                "result_ids": result_ids,
                "config": config,
                "hypothesis": (
                    f"Auto-investigation: testing robustness of top "
                    f"{len(result_ids)} screening survivors with "
                    f"{config.n_training_programs} training programs each."
                ),
            }
            evidence_pack = self._safe_build_evidence_pack(
                nb,
                recommendation={"mode": "investigation"},
                decision_type="auto_investigate",
            )
            self._pending_investigation["evidence_pack"] = evidence_pack
            self._emit_event(
                "auto_investigate_queued",
                {
                    "result_ids": result_ids,
                    "n_candidates": len(result_ids),
                    "reason": f"{survivor_count} S1 survivors with loss_ratio < 0.5",
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
                    metadata={"result_ids": result_ids, "evidence_pack": evidence_pack},
                )
            )
            return

        self._pending_validation = {
            "result_ids": result_ids,
            "config": config,
            "hypothesis": (
                f"Auto-validation: publication-grade testing of "
                f"{len(result_ids)} robust investigation survivors."
            ),
        }
        evidence_pack = self._safe_build_evidence_pack(
            nb,
            recommendation={"mode": "validation"},
            decision_type="auto_validate",
        )
        self._pending_validation["evidence_pack"] = evidence_pack
        self._emit_event(
            "auto_validate_queued",
            {
                "result_ids": result_ids,
                "n_candidates": len(result_ids),
                "blocked_incomplete_fingerprint": blocked_incomplete_fingerprint,
                "reason": f"{qualifying_count} candidates passed fingerprint + novelty + "
                f"robustness >= {config.auto_validate_min_robustness} gates",
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
                metadata={"result_ids": result_ids, "evidence_pack": evidence_pack},
            )
        )

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
        self._record_selection_decision(
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
            if not score_row:
                continue
            reward = score_row.get("base_score", 0.0)
            nb.update_selection_family_stats(
                score_row.get("family", "Unknown"),
                reward=float(reward),
            )

        # Leaderboard entries are created at S1-pass time in dashboard.py
        # via _upsert_screening_entry(). No need to duplicate here.
        self._queue_pending_followup(
            nb=nb,
            stage="investigation",
            result_ids=candidate_ids,
            config=config,
            survivor_count=s1_count,
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

        result_ids_all = [r.get("result_id") for r in strong if r.get("result_id")]
        graph_meta = graph_meta_by_result_id(nb, result_ids_all)

        prepared_candidates = prepare_validation_candidates(strong, graph_meta)

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
        self._record_selection_decision(
            nb,
            decision_payload=decision_payload,
            candidate_ids=candidate_ids,
            supporting_insight_ids=selection.get("supporting_insight_ids") or [],
            source_experiment_id=results.get("experiment_id"),
            failure_log_label="Auto-validate selection logging failed",
        )

        for rid in candidate_ids:
            score_row = scored_by_id.get(rid)
            if not score_row:
                continue
            nb.update_selection_family_stats(
                score_row.get("family", "Unknown"),
                reward=float(score_row.get("base_score", 0.0)),
            )

        self._queue_pending_followup(
            nb=nb,
            stage="validation",
            result_ids=candidate_ids,
            config=config,
            blocked_incomplete_fingerprint=blocked_incomplete_fingerprint,
            qualifying_count=len(strong),
        )

    @staticmethod
    def _adaptive_screening_threshold(
        nb: LabNotebook, config: RunConfig, floor: float
    ) -> float:
        """Compute adaptive screening threshold from recent population.

        Uses the configured percentile of the last 200 screening composite
        scores, floored at the fixed threshold to prevent promoting garbage
        in sparse populations.
        """
        try:
            import numpy as np

            rows = nb.conn.execute(
                f"""SELECT l.composite_score FROM leaderboard l
                   WHERE l.tier = 'screening'
                     AND l.composite_score IS NOT NULL
                     AND COALESCE(l.is_reference, 0) = 0
                     AND {sql_trusted_clause(table_alias="l")}
                   ORDER BY l.rowid DESC LIMIT 200"""
            ).fetchall()
            if len(rows) < 20:
                logger.info(
                    "Adaptive screening: only %d scores (need 20), using floor %.1f",
                    len(rows),
                    floor,
                )
                return floor
            scores = np.array([float(r[0]) for r in rows])
            pct_value = float(
                np.percentile(scores, config.screening_promotion_percentile)
            )
            threshold = max(pct_value, floor)
            logger.info(
                "Adaptive screening threshold: percentile(%.0f)=%.1f, "
                "floor=%.1f, using=%.1f (n=%d)",
                config.screening_promotion_percentile,
                pct_value,
                floor,
                threshold,
                len(rows),
            )
            return threshold
        except (ImportError, sqlite3.OperationalError, ValueError) as e:
            logger.warning(
                "Adaptive screening threshold failed: %s, using floor %.1f", e, floor
            )
            return floor

    @staticmethod
    def _adaptive_investigation_threshold(
        nb: LabNotebook, config: RunConfig, floor: float
    ) -> float:
        """Compute adaptive investigation threshold from recent population.

        Uses the configured percentile of the last 200 investigation composite
        scores, floored at the fixed threshold.
        """
        try:
            import numpy as np

            rows = nb.conn.execute(
                f"""SELECT l.composite_score FROM leaderboard l
                   WHERE l.tier IN ('investigation', 'investigation_failed')
                     AND l.composite_score IS NOT NULL
                     AND COALESCE(l.is_reference, 0) = 0
                     AND {sql_trusted_clause(table_alias="l")}
                   ORDER BY l.rowid DESC LIMIT 200"""
            ).fetchall()
            if len(rows) < 20:
                logger.info(
                    "Adaptive investigation: only %d scores (need 20), using floor %.1f",
                    len(rows),
                    floor,
                )
                return floor
            scores = np.array([float(r[0]) for r in rows])
            pct_value = float(
                np.percentile(scores, config.investigation_promotion_percentile)
            )
            threshold = max(pct_value, floor)
            logger.info(
                "Adaptive investigation threshold: percentile(%.0f)=%.1f, "
                "floor=%.1f, using=%.1f (n=%d)",
                config.investigation_promotion_percentile,
                pct_value,
                floor,
                threshold,
                len(rows),
            )
            return threshold
        except (ImportError, sqlite3.OperationalError, ValueError) as e:
            logger.warning(
                "Adaptive investigation threshold failed: %s, using floor %.1f",
                e,
                floor,
            )
            return floor
