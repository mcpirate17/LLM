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

import logging
import time
import uuid
from typing import Any, Dict, List

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
    VALIDATION_BEST_LR_HARD,
)
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

    def _auto_escalate_screening(
        self, results: Dict, config: RunConfig, nb: LabNotebook
    ) -> None:
        if not config.auto_investigate:
            return
        s1_count = results.get("stage1_passed", 0)
        if s1_count < config.auto_investigate_min_survivors:
            return

        exp_id = results.get("experiment_id")
        if exp_id:
            rows = nb.conn.execute(
                """SELECT * FROM program_results
                   WHERE experiment_id = ? AND stage1_passed = 1
                   ORDER BY loss_ratio ASC NULLS LAST
                   LIMIT ?""",
                (exp_id, config.auto_investigate_top_n),
            ).fetchall()
            top = [dict(r) for r in rows]
        else:
            top = nb.get_top_programs(
                config.auto_investigate_top_n, sort_by="loss_ratio"
            )

        try:
            global_rows = nb.conn.execute(
                """SELECT pr.* FROM leaderboard l
                   JOIN program_results pr ON l.result_id = pr.result_id
                   WHERE l.tier = 'screening' AND l.screening_passed = 1
                     AND COALESCE(l.is_reference, 0) = 0
                     AND l.investigation_loss_ratio IS NULL
                     AND (l.tags IS NULL OR l.tags NOT LIKE '%provisional_random_tokens%')
                   ORDER BY l.composite_score DESC
                   LIMIT ?""",
                (config.auto_investigate_top_n,),
            ).fetchall()
            seen = {p.get("result_id") for p in top}
            for r in global_rows:
                d = dict(r)
                if d.get("result_id") not in seen:
                    top.append(d)
                    seen.add(d.get("result_id"))
            if global_rows:
                logger.info(
                    "Auto-escalate: global sweep found %d leaderboard candidates",
                    len(global_rows),
                )
        except Exception as e:
            logger.warning("Auto-escalate global sweep failed: %s", e)

        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(top)
            top = [p for p in top if p.get("graph_fingerprint") not in investigated_fps]
            skipped = before - len(top)
            if skipped:
                logger.info(
                    "Auto-escalate: skipped %d already-investigated archs", skipped
                )

        # v7 screening → investigation threshold: see thresholds.py for calibration.
        _screening_floor = V7_SCREENING_THRESHOLD
        _screening_threshold = _screening_floor
        if config.adaptive_thresholds_enabled:
            _screening_threshold = self._adaptive_screening_threshold(
                nb, config, _screening_floor
            )
        try:
            before = len(top)
            qualified = []
            for p in top:
                rid = p.get("result_id")
                if not rid:
                    continue
                lb_row = nb.conn.execute(
                    "SELECT composite_score FROM leaderboard WHERE result_id = ?",
                    (rid,),
                ).fetchone()
                cs = float(lb_row[0]) if lb_row and lb_row[0] else 0.0
                if cs >= _screening_threshold:
                    qualified.append(p)
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
        except Exception as e:
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
        candidate_ids: List[str] = []
        for item in ranked:
            row = next(
                (p for p in top if p.get("result_id") == item["result_id"]), None
            )
            if row is None:
                continue
            if not row.get("stage1_passed"):
                continue
            candidate_ids.append(item["result_id"])
            if len(candidate_ids) >= config.auto_investigate_top_n:
                break

        if len(candidate_ids) < config.auto_investigate_min_survivors:
            return
        selected_rows = [p for p in top if p.get("result_id") in candidate_ids]
        decision_payload = {
            "decision_id": str(uuid.uuid4())[:12],
            "timestamp": time.time(),
            "context": "auto_investigate_screening",
            "experiment_id": exp_id,
            "candidate_pool_summary": selection.get("summary", {}),
            "score_breakdown": selection.get("scored", []),
            "policy": selection.get("policy", {}),
            "reason": selection.get("reason", ""),
            "chosen_experiments": [
                {
                    "result_id": rid,
                    "family": (row := scored_by_id.get(rid, {})).get("family"),
                    "score": row.get("score"),
                }
                for rid in candidate_ids
            ],
            "trigger": None,
        }
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
            supporting_insight_ids = selection.get("supporting_insight_ids") or []
            if supporting_insight_ids:
                nb.record_selection_insight_trial(
                    decision_id=decision_id,
                    context=decision_payload["context"],
                    insight_ids=supporting_insight_ids,
                    chosen_result_ids=candidate_ids,
                    source_experiment_id=exp_id,
                )
        except Exception as sel_err:
            logger.debug("Auto-investigate selection logging failed: %s", sel_err)

        if config.auto_go_no_go and config.enable_campaigns:
            approved_ids = []
            for p in selected_rows:
                if p["result_id"] not in candidate_ids:
                    continue
                try:
                    existing_decisions = nb.get_decisions(
                        campaign_id=self._active_campaign_id
                    )
                    already_decided = any(
                        p["result_id"] in (d.get("evidence_ids") or [])
                        for d in existing_decisions
                    )
                    if already_decided:
                        approved_ids.append(p["result_id"])
                        continue

                    go_context = build_go_no_go_context(
                        candidate=p,
                        campaign_criteria=(
                            nb.get_campaign(self._active_campaign_id or "") or {}
                        ).get("success_criteria", ""),
                    )
                    decision = self.aria.generate_go_no_go(
                        subject=f"Promote {p['result_id'][:8]} to investigation",
                        evidence=f"loss_ratio={p.get('loss_ratio', '?')}, "
                        f"novelty={p.get('novelty_score', '?')}",
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
                        subject=f"Promote {p['result_id'][:8]} to investigation",
                        rationale=decision["rationale"],
                        evidence_ids=[p["result_id"]],
                        alternatives=[{"considered": decision.get("alternatives", "")}],
                        evidence_pack=evidence_pack,
                    )
                    self._emit_event(
                        "decision_recorded",
                        {
                            "decision_type": decision["decision"],
                            "subject": p["result_id"][:8],
                            "rationale": decision["rationale"][:200],
                            "evidence_pack": evidence_pack,
                        },
                    )
                    if decision["decision"] in ("go", "pivot"):
                        approved_ids.append(p["result_id"])
                except Exception as e:
                    logger.debug("Go/no-go failed for %s: %s", p["result_id"], e)
                    approved_ids.append(p["result_id"])

            candidate_ids = approved_ids if approved_ids else candidate_ids
            selected_rows = [
                p for p in selected_rows if p.get("result_id") in candidate_ids
            ]

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

        self._pending_investigation = {
            "result_ids": candidate_ids,
            "config": config,
            "hypothesis": (
                f"Auto-investigation: testing robustness of top "
                f"{len(candidate_ids)} screening survivors with "
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
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "reason": f"{s1_count} S1 survivors with loss_ratio < 0.5",
                "evidence_pack": evidence_pack,
            },
        )

        nb.add_entry(
            ExperimentEntry(
                entry_type="decision",
                title="Auto-Investigation Triggered",
                content=(
                    f"Automatically queuing investigation for {len(candidate_ids)} "
                    f"top performers. Criteria: {s1_count} S1 survivors."
                ),
                metadata={"result_ids": candidate_ids, "evidence_pack": evidence_pack},
            )
        )

        try:
            sparse_wins = [p for p in top if (p.get("sparsity_ratio") or 0) > 0.3]
            dense_wins = [p for p in top if (p.get("sparsity_ratio") or 0) <= 0.3]
            if sparse_wins and dense_wins:
                avg_sparse_loss = sum(
                    p.get("loss_ratio", 1.0) for p in sparse_wins
                ) / len(sparse_wins)
                avg_dense_loss = sum(
                    p.get("loss_ratio", 1.0) for p in dense_wins
                ) / len(dense_wins)
                if avg_sparse_loss < avg_dense_loss * 0.95:
                    delta = 0.1
                    old_bias = config.grammar_config.structured_sparsity_bias
                    config.grammar_config.update_bias(delta)
                    nb.log_learning_event(
                        event_type="grammar_adjustment",
                        description=f"Boosted structured_sparsity_bias by {delta} due to sparse dominance.",
                        old_weights={"bias": old_bias},
                        new_weights={
                            "bias": config.grammar_config.structured_sparsity_bias
                        },
                        evidence=f"avg_sparse_loss={avg_sparse_loss:.4f}, avg_dense_loss={avg_dense_loss:.4f}",
                    )
        except Exception as z7_err:
            logger.debug("Z7 learning logic failed: %s", z7_err)

    def _auto_escalate_investigation(
        self, results: Dict, config: RunConfig, nb: LabNotebook
    ) -> None:
        if not config.auto_validate:
            return

        inv_results = results.get("investigation_results", [])
        inv_ids = [r.get("result_id") for r in inv_results if r.get("result_id")]
        novelty_meta: Dict[str, Dict[str, Any]] = {}
        if inv_ids:
            placeholders = ",".join("?" for _ in inv_ids)
            rows = nb.conn.execute(
                f"""SELECT result_id, novelty_valid_for_promotion, cka_source,
                       fingerprint_json
                    FROM program_results
                    WHERE result_id IN ({placeholders})""",
                tuple(inv_ids),
            ).fetchall()
            for row in rows:
                meta_dict = dict(row)
                # Extract fingerprint_completed_post_investigation from JSON
                fp_json_str = meta_dict.pop("fingerprint_json", None)
                if fp_json_str:
                    try:
                        import json as _json

                        fp_data = _json.loads(fp_json_str)
                        meta_dict["fingerprint_completed_post_investigation"] = bool(
                            fp_data.get("fingerprint_completed_post_investigation")
                        )
                    except (ValueError, TypeError):
                        meta_dict["fingerprint_completed_post_investigation"] = False
                else:
                    meta_dict["fingerprint_completed_post_investigation"] = False
                novelty_meta[row["result_id"]] = meta_dict

        # v7 investigation → validation threshold: see thresholds.py for calibration.
        _inv_floor = max(
            config.auto_validate_min_composite_score,
            V7_INVESTIGATION_THRESHOLD,
        )
        min_score = _inv_floor
        if config.adaptive_thresholds_enabled:
            min_score = self._adaptive_investigation_threshold(nb, config, _inv_floor)

        inv_id_list = [r.get("result_id") for r in inv_results if r.get("result_id")]
        composite_scores: Dict[str, float] = {}
        replication_info: Dict[str, Dict[str, Any]] = {}
        if inv_id_list:
            ph = ",".join("?" for _ in inv_id_list)
            score_rows = nb.conn.execute(
                f"""SELECT result_id, composite_score,
                       replication_n, replication_loss_std
                    FROM leaderboard WHERE result_id IN ({ph})""",
                tuple(inv_id_list),
            ).fetchall()
            composite_scores = {
                row["result_id"]: float(row["composite_score"] or 0)
                for row in score_rows
            }
            replication_info = {
                row["result_id"]: {
                    "n": int(row["replication_n"] or 1),
                    "loss_std": float(row["replication_loss_std"] or 0),
                }
                for row in score_rows
            }

        strong = []
        blocked_incomplete_fingerprint = 0
        for r in inv_results:
            rid = r.get("result_id")
            meta = novelty_meta.get(rid or "", {})

            # Hard gate 1: fingerprint must be completed post-investigation.
            # Without converged-model CKA, we cannot assess true novelty.
            if not bool(meta.get("fingerprint_completed_post_investigation")):
                blocked_incomplete_fingerprint += 1
                logger.warning(
                    "escalation_blocked_fingerprint_incomplete: "
                    "result_id=%s cka_source=%s",
                    (rid or "?")[:12],
                    meta.get("cka_source", "unknown"),
                )
                continue

            # Hard gate 2: novelty_valid_for_promotion must be True.
            # This means CKA was artifact-backed and non-degenerate.
            # No code path (including empirical override) bypasses this.
            if not bool(meta.get("novelty_valid_for_promotion")):
                logger.info(
                    "escalation_blocked_novelty_invalid: "
                    "result_id=%s reason=%s cka_source=%s",
                    (rid or "?")[:12],
                    meta.get("novelty_validity_reason", "unknown"),
                    meta.get("cka_source", "unknown"),
                )
                continue

            # novelty_valid_for_promotion=True is the binary gate.
            # No numeric novelty_score threshold is applied — the score
            # is source-dependent and a threshold is not meaningful.
            # DEPRECATED: auto_validate_min_novelty_confidence replaced
            # by novelty_valid_for_promotion binary gate. The numeric
            # threshold was not meaningful when novelty source varies
            # between structural-only and full CKA+behavioral blend.

            candidate_score = composite_scores.get(rid, 0.0)
            # Confidence-gated threshold: require margin above threshold
            # proportional to measurement uncertainty. With n=1, require
            # 10% margin; with n>=2 and known std, use std-based margin.
            repl = replication_info.get(rid, {"n": 1, "loss_std": 0})
            repl_n = repl["n"]
            if min_score > 0:
                if repl_n <= 1:
                    # Single run: require 10% margin above threshold
                    effective_threshold = min_score * 1.10
                elif repl["loss_std"] > 0 and repl_n >= 2:
                    # Multiple runs: use score uncertainty margin
                    # Approximate composite score SE from loss_ratio SE
                    # Composite ~ 100*(1-lr)^1.6, so d(score)/d(lr) ~ 160
                    # SE(score) ≈ 160 * SE(lr) = 160 * std/sqrt(n)
                    import math

                    se_score = 160.0 * repl["loss_std"] / math.sqrt(repl_n)
                    # Require lower bound of ~90% CI to exceed threshold
                    effective_threshold = min_score + 1.28 * se_score
                else:
                    effective_threshold = min_score
                if candidate_score < effective_threshold:
                    logger.info(
                        "Auto-validate: %s rejected (score %.1f < threshold %.1f, "
                        "base=%.1f, n=%d, loss_std=%.4f)",
                        (rid or "?")[:12],
                        candidate_score,
                        effective_threshold,
                        min_score,
                        repl_n,
                        repl["loss_std"],
                    )
                    continue
            # Sanity check: RAW loss_ratio should be <= 1.5.
            # Values > 1.5 suggest NORM was accidentally stored as RAW.
            _raw_lr = r.get("best_loss_ratio")
            if _raw_lr is not None and float(_raw_lr) > 1.5:
                logger.warning(
                    "loss_ratio_sanity_check: result_id=%s best_loss_ratio=%.4f > 1.5 "
                    "— possible NORM/RAW confusion. Skipping candidate.",
                    (rid or "?")[:12],
                    float(_raw_lr),
                )
                continue

            baseline_loss_ratio = r.get("baseline_loss_ratio")
            baseline_gate_passed = (
                baseline_loss_ratio is not None
                and float(baseline_loss_ratio) < config.auto_validate_max_baseline_ratio
            )
            empirical_override = self._meets_empirical_validation_override(
                r,
                candidate_score,
                min_score,
            )
            if (
                r.get("robustness", 0) >= config.auto_validate_min_robustness
                and (r.get("best_loss_ratio") or 1.0) < VALIDATION_BEST_LR_HARD
                and (baseline_gate_passed or empirical_override)
                and not r.get("brittle_risk", False)
                and (
                    r.get("loss_ratio_multiplier") is None
                    or r.get("loss_ratio_multiplier")
                    <= config.investigation_max_loss_ratio_multiplier
                )
            ):
                strong.append(r)

        if blocked_incomplete_fingerprint:
            logger.info(
                "Auto-validate: blocked %d candidates with incomplete fingerprints",
                blocked_incomplete_fingerprint,
            )

        if not strong:
            return

        result_ids_all = [r.get("result_id") for r in strong if r.get("result_id")]
        graph_meta: Dict[str, Dict[str, Any]] = {}
        if result_ids_all:
            placeholders = ",".join("?" for _ in result_ids_all)
            rows = nb.conn.execute(
                f"""SELECT result_id, graph_json, routing_mode
                    FROM program_results
                    WHERE result_id IN ({placeholders})""",
                tuple(result_ids_all),
            ).fetchall()
            graph_meta = {row["result_id"]: dict(row) for row in rows}

        prepared_candidates: List[Dict[str, Any]] = []
        for r in strong:
            rid = r.get("result_id")
            if not rid:
                continue
            meta = graph_meta.get(rid, {})
            prepared_candidates.append(
                {
                    "result_id": rid,
                    "graph_json": meta.get("graph_json"),
                    "routing_mode": meta.get("routing_mode"),
                    "loss_ratio": r.get("best_loss_ratio"),
                    "baseline_loss_ratio": r.get("baseline_loss_ratio"),
                    "novelty_score": r.get("novelty_confidence"),
                    "throughput_tok_s": r.get("throughput_tok_s"),
                    "flops_per_token": r.get("flops_per_token"),
                    "peak_memory_mb": r.get("peak_memory_mb"),
                    "stage0_passed": 1,
                    "stage05_passed": 1,
                    "stage1_passed": 1,
                    "stability_score": r.get("robustness"),
                    "has_nan_grad": 0,
                    "has_zero_grad": 0,
                }
            )

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
        decision_payload = {
            "decision_id": str(uuid.uuid4())[:12],
            "timestamp": time.time(),
            "context": "auto_validate_investigation",
            "experiment_id": results.get("experiment_id"),
            "candidate_pool_summary": selection.get("summary", {}),
            "score_breakdown": selection.get("scored", []),
            "policy": selection.get("policy", {}),
            "reason": selection.get("reason", ""),
            "chosen_experiments": [
                {
                    "result_id": rid,
                    "family": (row := scored_by_id.get(rid, {})).get("family"),
                    "score": row.get("score"),
                }
                for rid in candidate_ids
            ],
            "trigger": None,
        }
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
            supporting_insight_ids = selection.get("supporting_insight_ids") or []
            if supporting_insight_ids:
                nb.record_selection_insight_trial(
                    decision_id=decision_id,
                    context=decision_payload["context"],
                    insight_ids=supporting_insight_ids,
                    chosen_result_ids=candidate_ids,
                    source_experiment_id=str(results.get("experiment_id") or ""),
                )
        except Exception as sel_err:
            logger.debug("Auto-validate selection logging failed: %s", sel_err)

        for rid in candidate_ids:
            score_row = scored_by_id.get(rid)
            if not score_row:
                continue
            nb.update_selection_family_stats(
                score_row.get("family", "Unknown"),
                reward=float(score_row.get("base_score", 0.0)),
            )

        self._pending_validation = {
            "result_ids": candidate_ids,
            "config": config,
            "hypothesis": (
                f"Auto-validation: publication-grade testing of "
                f"{len(candidate_ids)} robust investigation survivors."
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
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "blocked_incomplete_fingerprint": blocked_incomplete_fingerprint,
                "reason": f"{len(strong)} candidates passed fingerprint + novelty + "
                f"robustness >= {config.auto_validate_min_robustness} gates",
                "evidence_pack": evidence_pack,
            },
        )

        nb.add_entry(
            ExperimentEntry(
                entry_type="decision",
                title="Auto-Validation Triggered",
                content=(
                    f"Automatically queuing validation for {len(candidate_ids)} "
                    f"robust investigation survivors."
                ),
                metadata={"result_ids": candidate_ids, "evidence_pack": evidence_pack},
            )
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
                """SELECT l.composite_score FROM leaderboard l
                   WHERE l.tier = 'screening'
                     AND l.composite_score IS NOT NULL
                     AND COALESCE(l.is_reference, 0) = 0
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
        except Exception as e:
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
                """SELECT l.composite_score FROM leaderboard l
                   WHERE l.tier IN ('investigation', 'investigation_failed')
                     AND l.composite_score IS NOT NULL
                     AND COALESCE(l.is_reference, 0) = 0
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
        except Exception as e:
            logger.warning(
                "Adaptive investigation threshold failed: %s, using floor %.1f",
                e,
                floor,
            )
            return floor
