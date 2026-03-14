"""Continuous investigation methods (pre-inv gate + inline investigation), split from continuous.py."""

from __future__ import annotations

import gc
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import torch

from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...training.training_program import synthesize_training_program_batch
from ...training.checkpointing import CheckpointManager
from ..notebook import LabNotebook, ExperimentEntry
from ._helpers import (
    _record_investigation_result,
    _submit_benchmark_eval,
)
from ..evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from ..llm.context_experiment import (
    build_investigation_context,
    build_mode_selection_context,
)
from ..llm.context_hypothesis import build_hypothesis_context
from ..shared_utils import resolve_device

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress


class _ContinuousInvestigationMixin:
    """Pre-investigation gate and inline investigation execution."""

    __slots__ = ()

    def _pre_inv_probe(self, config: RunConfig, nb: LabNotebook,
                       result_id: str) -> Optional[float]:
        """Stage C: single-seed probe at reduced step count.

        Runs 1 training program at probe_steps_fraction of investigation_steps.
        Returns loss_ratio or None on failure.
        """
        try:
            details = nb.get_program_details([result_id])
            if not details or not details[0]:
                return None
            source = details[0]
            graph_json = source.get("graph_json")
            if not graph_json:
                return None

            probe_config = RunConfig.from_dict(config.to_dict())
            probe_config.stage1_steps = max(
                50, int(config.investigation_steps * config.pre_inv_probe_steps_fraction))
            probe_config.stage1_batch_size = config.investigation_batch_size
            probe_config.n_programs = 1

            dev = resolve_device(config.device)
            dev_str = str(dev)

            from research.synthesis.compiler import compile_model
            model = compile_model(graph_json, probe_config, device=dev)
            if model is None:
                return None

            from research.evaluator import evaluate_stage1
            result = evaluate_stage1(model, probe_config, device=dev)
            lr = result.get("loss_ratio") if result else None
            return float(lr) if lr is not None else None
        except Exception as e:
            logger.warning("Pre-inv probe failed for %s: %s", result_id[:8], e)
            return None

    def _pre_investigation_gate(self, config: RunConfig, nb: LabNotebook,
                                leaderboard: list) -> List[str]:
        """Orchestrate three-stage pre-investigation gate.

        Stage A: SQL hard reject (numerical health, stability, gradient path)
        Stage B: Composite readiness score, rank and take top-N
        Stage C: Optional single-seed probe

        Returns filtered, ranked result_ids ready for investigation.
        Falls back to legacy behavior when pre_inv_gate_enabled=False.
        """
        if not config.pre_inv_gate_enabled:
            # Legacy behavior: filter by loss_ratio threshold only
            investigated_fps = nb.get_investigated_fingerprints()
            candidates = [
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold
                and "provisional_random_tokens" not in (e.get("tags") or "")
            ]
            if investigated_fps:
                candidates = [
                    c for c in candidates
                    if c.get("graph_fingerprint", c.get("architecture_desc", ""))
                    not in investigated_fps
                ]
            return [c["result_id"] for c in candidates[:config.auto_investigate_top_n]
                    if c.get("result_id")]

        # ── Stage A: Hard reject via SQL ──
        # Uses composite_score as primary gate (not loss_ratio) — models
        # with strong efficiency/novelty/stability deserve investigation
        # even if loss is only moderate.
        eligible = nb.get_investigation_eligible(
            max_lr=config.pre_inv_max_lr,
            min_stability=config.pre_inv_min_stability,
            min_spectral_norm=config.pre_inv_min_spectral_norm,
            max_spectral_norm=config.pre_inv_max_spectral_norm,
            min_improvement_rate=config.pre_inv_min_improvement_rate,
            ref_lr_ceiling=self._reference_margin_ceiling(config, nb),
        )

        # Filter out already-investigated fingerprints
        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(eligible)
            eligible = [e for e in eligible
                        if e.get("graph_fingerprint") not in investigated_fps]
            skipped = before - len(eligible)
            if skipped:
                logger.info("Pre-inv gate: skipped %d already-investigated candidates", skipped)

        if not eligible:
            logger.info("Pre-inv gate Stage A: no eligible candidates")
            return []

        logger.info("Pre-inv gate Stage A: %d candidates pass hard filters", len(eligible))

        # ── Stage B: Composite score + rank ──
        ref_lr = self._get_reference_baseline_lr(nb)
        for row in eligible:
            base = LabNotebook.compute_pre_investigation_score(
                row, best_ref_lr=ref_lr)
            # Judgment boost: up to +15% for high-confidence candidates
            j = row.get("judgment_score")
            if j is not None and isinstance(j, (int, float)) and j > 0.5:
                base *= 1.0 + 0.15 * min(1.0, (j - 0.5) * 2.0)
            row["_pre_inv_score"] = base

        eligible.sort(key=lambda r: r.get("_pre_inv_score", 0), reverse=True)
        top_n = eligible[:config.pre_inv_top_n]

        # Persist scores to leaderboard
        for row in eligible:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET pre_inv_score = ? WHERE result_id = ?",
                    (row["_pre_inv_score"], row["result_id"]),
                )
            except Exception:
                pass
        try:
            nb.conn.commit()
        except Exception:
            pass

        logger.info("Pre-inv gate Stage B: top %d scored [%s]",
                     len(top_n),
                     ", ".join(f"{r['result_id'][:8]}={r['_pre_inv_score']:.1f}"
                               for r in top_n))

        # ── Stage C: Optional probe ──
        if config.pre_inv_probe_enabled:
            probed = []
            for row in top_n:
                probe_lr = self._pre_inv_probe(config, nb, row["result_id"])
                if probe_lr is not None and probe_lr > config.pre_inv_probe_max_lr:
                    logger.info("Pre-inv probe rejected %s (lr=%.3f > %.3f)",
                                row["result_id"][:8], probe_lr,
                                config.pre_inv_probe_max_lr)
                    continue
                probed.append(row)
            top_n = probed

        result_ids = [r["result_id"] for r in top_n if r.get("result_id")]

        # ── Stage D: Recipe re-roll for screened_out frontier models ──
        # Models that failed investigation (robustness < 0.5) but have
        # frontier-competitive real-token quality deserve reinvestigation
        # with fresh training programs before being permanently buried.
        reinvest_ids = self._get_reinvestigation_candidates(nb, exclude=set(result_ids))
        if reinvest_ids:
            logger.info(
                "Pre-inv gate Stage D: %d screened_out frontier models queued for recipe re-roll",
                len(reinvest_ids),
            )
            result_ids.extend(reinvest_ids)

        return result_ids

    _MAX_REINVESTIGATION_ATTEMPTS = 2

    def _get_reinvestigation_candidates(
        self, nb: LabNotebook, exclude: set, limit: int = 3,
    ) -> List[str]:
        """Find screened_out models with WikiText quality above the investigation tier.

        These are architectures that failed robustness (typically 1/3 training
        programs passed) but demonstrably generalise on real tokens.  They get
        reinvestigated with fresh training programs — same architecture, new
        recipe — before any architectural mutation is considered.

        Capped at ``_MAX_REINVESTIGATION_ATTEMPTS`` per model to prevent
        infinite re-roll loops.
        """
        max_attempts = self._MAX_REINVESTIGATION_ATTEMPTS
        try:
            rows = nb.conn.execute("""
                SELECT l.result_id, l.wikitext_score, l.investigation_robustness,
                       COALESCE(l.reinvestigation_count, 0) AS reinvest_count
                FROM leaderboard l
                WHERE l.tier = 'screened_out'
                  AND l.wikitext_score IS NOT NULL
                  AND l.wikitext_score > (
                      SELECT COALESCE(MAX(l2.wikitext_score), 0)
                      FROM leaderboard l2
                      WHERE l2.tier = 'investigation'
                  )
                  AND COALESCE(l.investigation_robustness, 0) < 0.5
                  AND COALESCE(l.reinvestigation_count, 0) < ?
                ORDER BY l.wikitext_score DESC
                LIMIT ?
            """, (max_attempts, limit + len(exclude))).fetchall()
        except Exception as e:
            logger.debug("Reinvestigation query failed: %s", e)
            return []

        candidates = [
            r["result_id"] for r in rows
            if r["result_id"] and r["result_id"] not in exclude
        ][:limit]

        # Increment reinvestigation count for selected candidates
        for rid in candidates:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET reinvestigation_count = COALESCE(reinvestigation_count, 0) + 1 "
                    "WHERE result_id = ?",
                    (rid,),
                )
            except Exception:
                pass
            logger.info(
                "  Recipe re-roll candidate: %s (wikitext_score above investigation tier)",
                rid[:8],
            )
        if candidates:
            try:
                nb.conn.commit()
            except Exception:
                pass

        return candidates

    def _reference_margin_ceiling(self, config: RunConfig, nb: LabNotebook) -> Optional[float]:
        """Convert the reference margin knob into a concrete Stage-A LR ceiling."""
        best_ref_lr = self._get_reference_baseline_lr(nb)
        if best_ref_lr is None:
            return None
        margin = max(0.1, float(config.pre_inv_reference_margin or 1.0))
        return float(best_ref_lr) * margin

    def _run_inline_investigation(self, config: RunConfig, nb: LabNotebook,
                                   leaderboard: list, n_experiments: int,
                                   limit_str: str, mode_reasoning: str):
        """Execute investigation phase inline (not threaded) for continuous mode."""
        # Use pre-investigation gate for candidate selection
        result_ids = self._pre_investigation_gate(config, nb, leaderboard)
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        # Build context for hypothesis formulation
        inv_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        inv_map = {d.get("result_id"): d for d in inv_details if d.get("result_id")}
        inv_context = build_investigation_context(inv_details, leaderboard)
        hypothesis = self.aria.formulate_investigation_hypothesis(
            context=inv_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
            created_by="inline_investigation",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="investigating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=(f"[{limit_str}|investigation] "
                              f"Studying {len(result_ids)} candidates"),
            )

        self._emit_event("investigation_started", {
            "experiment_id": exp_id,
            "n_candidates": len(result_ids),
        })

        self._live_training_context = {"exp_id": exp_id, "phase": "investigation"}
        try:
            # ── Inline investigation logic (from _run_investigation_thread) ──
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "investigation_results": [],
            }

            dev = resolve_device(config.device)
            dev_str = str(dev)

            inv_config = RunConfig.from_dict(config.to_dict())
            inv_config.stage1_steps = config.investigation_steps
            inv_config.stage1_batch_size = config.investigation_batch_size

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                # Cost check mid-investigation
                if config.max_cost_dollars > 0 and self.aria.total_cost >= config.max_cost_dollars:
                    logger.info("Cost limit reached during investigation")
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = inv_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Generate training programs (queue-level scheduling telemetry)
                training_programs, tp_sched = synthesize_training_program_batch(
                    n_programs=config.n_training_programs,
                    n_steps=config.investigation_steps,
                    max_seq_len=config.max_seq_len,
                    seed_offset=prog_idx * 1000,
                )
                results.setdefault("training_program_scheduling", []).append({
                    "result_id": source_result_id,
                    **tp_sched,
                })

                # Test each (model x training_program) pair
                tp_results = []
                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    try:
                        model = self._build_model_from_source(
                            model_source,
                            arch_spec_json_str,
                            graph_json_str,
                            config,
                            seq_len_override=config.max_seq_len,
                        )
                        if model is None:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("investigation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "training_program": tp_i + 1,
                        "total_programs": len(training_programs),
                        "status": f"training with {tp.name}",
                    })

                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation"),
                    )
                    tp_results.append({
                        "training_program": tp.name,
                        "passed": tp_result.get("passed", False),
                        "loss_ratio": tp_result.get("loss_ratio"),
                        "final_loss": tp_result.get("final_loss"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Skip candidates where no training program could reconstruct the model
                if not tp_results:
                    logger.debug(
                        f"Investigation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {len(training_programs)} programs"
                    )
                    continue

                # Compute robustness
                n_passed = sum(1 for r in tp_results if r.get("passed"))
                robustness = n_passed / max(len(tp_results), 1)
                best_tp = min(
                    (r for r in tp_results if r.get("loss_ratio") is not None),
                    key=lambda r: r["loss_ratio"],
                    default=None,
                )
                best_lr = best_tp["loss_ratio"] if best_tp else None
                screening_lr = source.get("loss_ratio")
                lr_multiplier = self._investigation_loss_multiplier(screening_lr, best_lr)
                brittle_risk = (
                    lr_multiplier is not None
                    and lr_multiplier > float(config.investigation_max_loss_ratio_multiplier)
                )

                if n_passed > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                investigation_entry = {
                    "result_id": source_result_id,
                    "robustness": robustness,
                    "best_loss_ratio": best_lr,
                    "screening_loss_ratio": screening_lr,
                    "baseline_loss_ratio": source.get("baseline_loss_ratio"),
                    "novelty_confidence": source.get("novelty_confidence"),
                    "loss_ratio_multiplier": lr_multiplier,
                    "brittle_risk": brittle_risk,
                    "n_programs_passed": n_passed,
                    "n_programs_tested": len(tp_results),
                    "best_training_program": best_tp.get("training_program") if best_tp else None,
                    "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
                    "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
                }
                results["investigation_results"].append(investigation_entry)

                if best_lr and (results["best_loss_ratio"] is None
                                or best_lr < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = best_lr
                source_novelty = source.get("novelty_score")
                if source_novelty is not None and (
                    results["best_novelty_score"] is None
                    or source_novelty > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = source_novelty

                # Update leaderboard
                best_tp_json = None
                if best_tp and best_tp.get("training_program"):
                    for tp in training_programs:
                        if tp.name == best_tp["training_program"]:
                            best_tp_json = json.dumps(tp.to_dict())
                            break

                # Brittle risk override: if the investigation LR is good on
                # its own merits (< 0.3), don't let the screening→investigation
                # multiplier veto promotion.  Prevents false positives when
                # screening LR was unrealistically low (e.g. lucky seed).
                # Gate: pass investigation if loss quality is good enough.
                # Robustness is tracked as a ranking signal (robustness_grade)
                # but no longer a hard gate — models with 1/3 pass rate but
                # strong real-token quality should still proceed.
                investigation_passed = (
                    (best_lr or 1.0) < 0.5
                    and (not brittle_risk
                         or (best_lr is not None and best_lr < 0.3))
                )

                # Submit benchmark evals to background thread so the
                # investigation loop can proceed to the next candidate.
                if n_passed > 0:
                    _submit_benchmark_eval(
                        nb=nb,
                        exp_id=exp_id,
                        source_result_id=source_result_id,
                        source=source,
                        model_source=model_source,
                        graph_json_str=graph_json_str,
                        arch_spec_json_str=arch_spec_json_str,
                        n_passed=n_passed,
                        best_lr=best_lr,
                        best_tp_json=best_tp_json,
                        robustness=robustness,
                        investigation_passed=investigation_passed,
                        config=config,
                        dev=dev,
                        cached_json_load=self._cached_json_load,
                    )
                else:
                    _record_investigation_result(
                        nb=nb,
                        exp_id=exp_id,
                        source_result_id=source_result_id,
                        source=source,
                        model_source=model_source,
                        graph_json_str=graph_json_str,
                        arch_spec_json_str=arch_spec_json_str,
                        n_passed=n_passed,
                        best_lr=best_lr,
                        best_tp_json=best_tp_json,
                        robustness=robustness,
                        investigation_passed=investigation_passed,
                        inv_wikitext_ppl=None,
                        inv_wikitext_score=None,
                        inv_tinystories_ppl=None,
                        inv_tinystories_score=None,
                    )

            # Complete experiment with LLM analysis
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id, results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            nb.flush_writes()
            # Auto-escalate to validation if strong candidates found
            self._auto_escalate(results, config, nb, phase="investigation")

            # Knowledge extraction after investigation
            self._maybe_extract_knowledge(config, nb, n_experiments)

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "results": results,
                "summary": summary,
            })

        except Exception as e:
            logger.warning(f"Inline investigation failed: {e}")
            nb.fail_experiment(exp_id, str(e))
            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "error": str(e),
            })
        finally:
            self._live_training_context = None
