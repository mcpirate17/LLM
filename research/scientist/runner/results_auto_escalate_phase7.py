"""Phase 7 split helpers for results._auto_escalate."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List

from ..evidence import validate_selection_decision_log
from ..llm.context import build_go_no_go_context
from ..leaderboard import LeaderboardManager
from ..notebook import ExperimentEntry, LabNotebook
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ResultsAutoEscalatePhase7Mixin:
    """Branch helpers extracted from _auto_escalate orchestration."""

    def _auto_escalate_screening(self, results: Dict, config: RunConfig, nb: LabNotebook) -> None:
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
            top = nb.get_top_programs(config.auto_investigate_top_n, sort_by="loss_ratio")

        try:
            global_rows = nb.conn.execute(
                """SELECT pr.* FROM leaderboard l
                   JOIN program_results pr ON l.result_id = pr.result_id
                   WHERE l.tier = 'screening' AND l.screening_passed = 1
                     AND COALESCE(l.is_reference, 0) = 0
                     AND l.investigation_loss_ratio IS NULL
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
                logger.info("Auto-escalate: global sweep found %d leaderboard candidates", len(global_rows))
        except Exception as e:
            logger.warning("Auto-escalate global sweep failed: %s", e)

        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(top)
            top = [p for p in top if p.get("graph_fingerprint") not in investigated_fps]
            skipped = before - len(top)
            if skipped:
                logger.info("Auto-escalate: skipped %d already-investigated archs", skipped)

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
            row = next((p for p in top if p.get("result_id") == item["result_id"]), None)
            if row is None:
                continue
            if not row.get("stage1_passed"):
                continue
            if row.get("loss_ratio") is not None and float(row.get("loss_ratio")) >= 0.50:
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
                    "family": scored_by_id.get(rid, {}).get("family"),
                    "score": scored_by_id.get(rid, {}).get("score"),
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
                    existing_decisions = nb.get_decisions(campaign_id=self._active_campaign_id)
                    already_decided = any(
                        p["result_id"] in (d.get("evidence_ids") or [])
                        for d in existing_decisions
                    )
                    if already_decided:
                        approved_ids.append(p["result_id"])
                        continue

                    go_context = build_go_no_go_context(
                        candidate=p,
                        campaign_criteria=(nb.get_campaign(self._active_campaign_id or "") or {}).get(
                            "success_criteria", ""
                        ),
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
            selected_rows = [p for p in selected_rows if p.get("result_id") in candidate_ids]

        for rid in candidate_ids:
            score_row = scored_by_id.get(rid)
            if not score_row:
                continue
            reward = score_row.get("base_score", 0.0)
            nb.update_selection_family_stats(
                score_row.get("family", "Unknown"),
                reward=float(reward),
            )

        existing_lb = {e["result_id"]: e["tier"] for e in nb.get_leaderboard(limit=500)}
        for p in selected_rows:
            if p["result_id"] in candidate_ids:
                if p["result_id"] in existing_lb and existing_lb[p["result_id"]] in (
                    "screening",
                    "investigation",
                    "validation",
                ):
                    continue
                # Compute efficiency multiple from screening metrics
                eff_mult = LeaderboardManager.compute_efficiency_multiple(
                    loss_ratio=p.get("loss_ratio"),
                    param_count=p.get("param_count"),
                    flops_forward=p.get("flops_forward"),
                    throughput_tok_s=p.get("throughput_tok_s"),
                    peak_memory_mb=p.get("peak_memory_mb"),
                    forward_time_ms=p.get("forward_time_ms"),
                )
                eff_geomean = eff_mult["geomean"] if eff_mult else None
                nb.upsert_leaderboard(
                    result_id=p["result_id"],
                    model_source=p.get("model_source") or "graph_synthesis",
                    architecture_desc=p.get("graph_fingerprint", "")[:40],
                    screening_loss_ratio=p.get("loss_ratio"),
                    screening_novelty=p.get("novelty_score"),
                    screening_passed=True,
                    tier="screening",
                    novelty_confidence=p.get("novelty_confidence"),
                    fp_jacobian_spectral_norm=p.get("fp_jacobian_spectral_norm"),
                    scaling_param_efficiency=eff_geomean,
                    efficiency_multiple=eff_geomean,
                    routing_savings_ratio=p.get("routing_savings_ratio"),
                    activation_sparsity_score=p.get("activation_sparsity_score"),
                    depth_savings_ratio=p.get("depth_savings_ratio"),
                    compression_ratio=p.get("compression_ratio"),
                )

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
                avg_sparse_loss = sum(p.get("loss_ratio", 1.0) for p in sparse_wins) / len(sparse_wins)
                avg_dense_loss = sum(p.get("loss_ratio", 1.0) for p in dense_wins) / len(dense_wins)
                if avg_sparse_loss < avg_dense_loss * 0.95:
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
        except Exception as z7_err:
            logger.debug("Z7 learning logic failed: %s", z7_err)

    def _auto_escalate_investigation(self, results: Dict, config: RunConfig, nb: LabNotebook) -> None:
        if not config.auto_validate:
            return

        inv_results = results.get("investigation_results", [])
        inv_ids = [r.get("result_id") for r in inv_results if r.get("result_id")]
        novelty_meta: Dict[str, Dict[str, Any]] = {}
        if inv_ids:
            placeholders = ",".join("?" for _ in inv_ids)
            rows = nb.conn.execute(
                f"""SELECT result_id, novelty_valid_for_promotion, cka_source
                    FROM program_results
                    WHERE result_id IN ({placeholders})""",
                tuple(inv_ids),
            ).fetchall()
            novelty_meta = {row["result_id"]: dict(row) for row in rows}

        min_score = config.auto_validate_min_composite_score
        if min_score <= 0:
            # Use best reference loss_ratio as a tier-neutral gate instead of
            # composite_score (which includes a tier confidence discount).
            best_ref_lr_row = nb.conn.execute(
                "SELECT MIN(screening_loss_ratio) FROM leaderboard"
                " WHERE COALESCE(is_reference, 0) = 1"
                " AND screening_loss_ratio IS NOT NULL"
            ).fetchone()
            best_ref_lr = float(best_ref_lr_row[0]) if best_ref_lr_row and best_ref_lr_row[0] else None
            # Convert to equivalent composite floor: same formula as scoring
            # but without tier discount, so investigation candidates compare fairly.
            min_score = 100.0 * max(0, 1.0 - best_ref_lr) * 0.85 if best_ref_lr is not None else 0.0

        inv_id_list = [r.get("result_id") for r in inv_results if r.get("result_id")]
        composite_scores: Dict[str, float] = {}
        if inv_id_list:
            ph = ",".join("?" for _ in inv_id_list)
            score_rows = nb.conn.execute(
                f"SELECT result_id, composite_score FROM leaderboard WHERE result_id IN ({ph})",
                tuple(inv_id_list),
            ).fetchall()
            composite_scores = {row["result_id"]: float(row["composite_score"] or 0) for row in score_rows}

        strong = []
        for r in inv_results:
            rid = r.get("result_id")
            meta = novelty_meta.get(rid or "", {})
            if not meta:
                novelty_valid = True
            else:
                novelty_valid = bool(meta.get("novelty_valid_for_promotion"))
                if not novelty_valid and meta.get("cka_source") == "artifact":
                    novelty_valid = True
            if not novelty_valid and config.allow_heuristic_novelty_promotion:
                novelty_valid = bool(str(config.heuristic_novelty_justification or "").strip())

            candidate_score = composite_scores.get(rid, 0.0)
            if min_score > 0 and candidate_score < min_score:
                logger.info(
                    "Auto-validate: %s rejected (score %.1f < min %.1f)",
                    (rid or "?")[:12],
                    candidate_score,
                    min_score,
                )
                continue
            if (
                r.get("robustness", 0) >= config.auto_validate_min_robustness
                and (r.get("best_loss_ratio") or 1.0) < 0.25
                and r.get("baseline_loss_ratio") is not None
                and r.get("baseline_loss_ratio") < config.auto_validate_max_baseline_ratio
                and r.get("novelty_confidence") is not None
                and r.get("novelty_confidence") >= config.auto_validate_min_novelty_confidence
                and novelty_valid
                and not r.get("brittle_risk", False)
                and (
                    r.get("loss_ratio_multiplier") is None
                    or r.get("loss_ratio_multiplier") <= config.investigation_max_loss_ratio_multiplier
                )
            ):
                strong.append(r)

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
        candidate_ids = [item["result_id"] for item in ranked[: config.auto_validate_top_n]]
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
                    "family": scored_by_id.get(rid, {}).get("family"),
                    "score": scored_by_id.get(rid, {}).get("score"),
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
                "reason": f"{len(strong)} candidates with robustness >= "
                f"{config.auto_validate_min_robustness}",
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
