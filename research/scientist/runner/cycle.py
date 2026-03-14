"""
Experiment Runner

The autonomous experiment execution engine. Aria uses this to:
1. Generate batches of synthesized programs
2. Evaluate them through the funnel
3. Record results in the lab notebook
4. Analyze patterns and formulate new hypotheses
5. Adjust strategy based on outcomes

Supports background execution controlled from the dashboard.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..notebook import LabNotebook, ExperimentEntry
from ...healer.core import HealerTaskSpec

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig

class _CycleMixin:
    """Main experiment cycle, proactive repair, healer integration."""

    def run_aria_cycle(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        t_start: float,
    ) -> Dict[str, Any]:
        """Run one continuous research cycle (plan -> run -> analyze -> summarize)."""
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=n_experiments,
            selected_mode=None,
            note=f"Planning cycle {n_experiments}.",
        )

        # Get digest from distiller if available
        _digest = None
        _distiller = getattr(self, "_knowledge_distiller", None)
        if _distiller is not None:
            try:
                _digest = _distiller.get_digest()
            except Exception:
                pass

        mode_rec = self._select_next_mode(config, nb, n_experiments, digest=_digest)
        selected_mode = mode_rec.get("mode", "synthesis")
        mode_reasoning = mode_rec.get("reasoning", "")
        mode_confidence = mode_rec.get("confidence", 0)
        mode_config_overrides = mode_rec.get("config") or {}

        # Extract op_weights and structured_sparsity_bias from mode config
        # (these aren't RunConfig attrs, so _config_with_overrides would ignore them)
        if "op_weights" in mode_config_overrides:
            mode_op_weights = mode_config_overrides.pop("op_weights")
            if isinstance(mode_op_weights, dict):
                self._op_weights_overrides.update({
                    str(k): max(0.01, min(10.0, float(v)))
                    for k, v in mode_op_weights.items()
                    if isinstance(v, (int, float))
                })
                logger.info("Mode recommendation applied op_weights: %s", mode_op_weights)
        if "structured_sparsity_bias" in mode_config_overrides:
            bias = mode_config_overrides.pop("structured_sparsity_bias")
            if isinstance(bias, (int, float)):
                # Store as grammar weight override so _build_grammar_config picks it up
                self._structured_sparsity_bias_override = max(0.0, min(1.0, float(bias)))
                logger.info("Mode recommendation applied structured_sparsity_bias: %.2f", bias)

        cycle_config, override_report = self._config_with_overrides(config, mode_config_overrides)

        if override_report.get("applied"):
            self._emit_event("mode_config_applied", {
                "mode": selected_mode,
                "applied": override_report.get("applied"),
                "ignored": override_report.get("ignored", {}),
                "experiment_number": n_experiments,
            })

        prescreen_mode = "continuous"
        if selected_mode == "evolution":
            prescreen_mode = "evolve"
        elif selected_mode == "novelty":
            prescreen_mode = "novelty"

        cycle_config, cycle_prescreen = self.prescreen_run_config(
            cycle_config,
            mode=prescreen_mode,
            auto_harden=True,
        )
        if cycle_prescreen.get("issue_count", 0) > 0:
            self._emit_event("cycle_prescreen", {
                "mode": selected_mode,
                "report": cycle_prescreen,
                "experiment_number": n_experiments,
            })

        effective_max_time_minutes = self._effective_max_time_minutes(config)

        self._emit_event("mode_selected", {
            "mode": selected_mode,
            "reasoning": mode_reasoning,
            "confidence": mode_confidence,
            "experiment_number": n_experiments,
        })
        logger.info(
            "Cycle %d: mode=%s (confidence=%.2f) — %s",
            n_experiments, selected_mode, mode_confidence,
            mode_reasoning[:120] if mode_reasoning else "no reasoning",
        )

        limit_info = []
        if config.max_experiments > 0:
            limit_info.append(f"exp {n_experiments}/{config.max_experiments}")
        if effective_max_time_minutes > 0:
            elapsed_min = (time.time() - t_start) / 60
            limit_info.append(f"{elapsed_min:.0f}/{effective_max_time_minutes}min")
        if config.max_cost_dollars > 0:
            limit_info.append(f"${self.aria.total_cost:.2f}/${config.max_cost_dollars:.2f}")
        limit_str = " | ".join(limit_info) if limit_info else f"exp {n_experiments}"

        # Check for stale screening candidates that beat references
        # Only override if we're past the exploration-first window (first 3 cycles)
        # and the selected mode isn't already investigation/validation
        if (config.auto_investigate
                and selected_mode not in ("investigation", "validation")
                and n_experiments >= 3):
            stale_ids = self._check_stale_screening_candidates(nb, config)
            if stale_ids:
                self._pending_investigation = {
                    "result_ids": stale_ids,
                    "config": config,
                    "hypothesis": f"Priority investigation: {len(stale_ids)} screening models beat reference baselines but are uninvestigated.",
                }

        pending_inv = getattr(self, "_pending_investigation", None)
        pending_val = getattr(self, "_pending_validation", None)

        # Don't let auto-escalation override synthesis in the first 3 cycles —
        # Aria needs to explore before she can make informed investigation decisions
        if pending_inv and selected_mode != "investigation" and n_experiments >= 3:
            selected_mode = "investigation"
            mode_reasoning = pending_inv.get("hypothesis", "Auto-investigation")
            mode_confidence = 0.9
            self._pending_investigation = None
            self._emit_event("mode_selected", {
                "mode": "investigation",
                "reasoning": "Auto-escalation: S1 survivors qualify for investigation",
                "confidence": 0.9,
                "experiment_number": n_experiments,
            })
        elif pending_val and selected_mode != "validation" and n_experiments >= 3:
            selected_mode = "validation"
            mode_reasoning = pending_val.get("hypothesis", "Auto-validation")
            mode_confidence = 0.9
            self._pending_validation = None
            self._emit_event("mode_selected", {
                "mode": "validation",
                "reasoning": "Auto-escalation: investigation survivors qualify for validation",
                "confidence": 0.9,
                "experiment_number": n_experiments,
            })

        before_progress = self.progress.to_dict()
        cycle_error: Optional[str] = None

        # Per-experiment watchdog: abort if a single cycle exceeds this limit.
        # Modes like novelty/evolution can hang in generation loops; this is the
        # last-resort safety net.  The watchdog sets _stop_event, which the inner
        # loops check, causing a graceful exit rather than a hard kill.
        max_cycle_seconds = int(
            getattr(config, "max_cycle_seconds", 0) or 0
        ) or 1800  # default 30 min
        _watchdog_fired = False

        def _cycle_watchdog():
            nonlocal _watchdog_fired
            _watchdog_fired = True
            logger.error(
                "WATCHDOG: Cycle %d (%s) exceeded %ds — setting stop event",
                n_experiments, selected_mode, max_cycle_seconds,
            )
            self._emit_event("cycle_watchdog", {
                "experiment_number": n_experiments,
                "mode": selected_mode,
                "timeout_seconds": max_cycle_seconds,
            })
            self._stop_event.set()

        watchdog = threading.Timer(max_cycle_seconds, _cycle_watchdog)
        watchdog.daemon = True
        watchdog.start()

        try:
            self._set_aria_cycle_phase(
                "running",
                continuous_active=True,
                cycle_index=n_experiments,
                selected_mode=selected_mode,
                note=f"Running {selected_mode} cycle {n_experiments}.",
            )
            logger.info(
                "Cycle %d: starting %s run [%s] (watchdog=%ds)",
                n_experiments, selected_mode, limit_str, max_cycle_seconds,
            )
            if selected_mode in ("investigation", "validation"):
                self._run_continuous_phase(
                    selected_mode, cycle_config, nb, n_experiments,
                    limit_str, mode_reasoning)
            elif selected_mode == "evolution":
                self._run_continuous_evolution(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            elif selected_mode == "novelty":
                self._run_continuous_novelty(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            elif selected_mode == "refinement":
                self._run_continuous_refinement(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            else:
                self._run_continuous_synthesis(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
        except Exception as e:
            cycle_error = str(e)
            logger.warning("Cycle %d FAILED (%s): %s", n_experiments, selected_mode, e)
            failed_exp_id = self._fail_active_cycle_experiment(
                nb,
                cycle_error,
                expected_mode=selected_mode,
            )
            self._emit_event("experiment_failed", {
                "experiment_number": n_experiments,
                "mode": selected_mode,
                "experiment_id": failed_exp_id,
                "error": cycle_error,
            })
        finally:
            watchdog.cancel()
            # If watchdog fired, clear stop event so continuous mode can proceed
            # to the next cycle (the current cycle is already aborted).
            if _watchdog_fired:
                self._stop_event.clear()
                cycle_error = cycle_error or f"Watchdog timeout ({max_cycle_seconds}s)"
                logger.warning(
                    "Cycle %d: watchdog fired, stop event cleared for next cycle",
                    n_experiments,
                )
            if not self._stop_event.is_set():
                self._set_aria_cycle_phase(
                    "analyzing",
                    continuous_active=True,
                    cycle_index=n_experiments,
                    selected_mode=selected_mode,
                    note="Analyzing outcomes and preparing next recommendation.",
                )

        after_progress = self.progress.to_dict()
        summary = self._build_aria_cycle_summary(
            cycle_index=n_experiments,
            selected_mode=selected_mode,
            mode_reasoning=mode_reasoning,
            mode_confidence=mode_confidence,
            before_progress=before_progress,
            after_progress=after_progress,
            error=cycle_error,
        )
        try:
            nb.add_entry(ExperimentEntry(
                entry_type="live_feed",
                experiment_id=after_progress.get("experiment_id") or None,
                title=f"Aria cycle {n_experiments}: {selected_mode}",
                content=(
                    f"Cycle {n_experiments} completed in {selected_mode} mode "
                    f"with ΔS1={summary.get('delta_stage1_survivors', 0)}"
                ),
                metadata={
                    "live_feed_type": "aria_cycle",
                    "payload": summary,
                },
            ))
        except Exception as e:
            logger.debug("Failed to persist aria_cycle live-feed entry: %s", e)
        with self._lock:
            self._last_cycle_summary = summary
            self._aria_cycle_history.append(summary)
            if len(self._aria_cycle_history) > 50:
                self._aria_cycle_history = self._aria_cycle_history[-50:]
        switch_guardrails = self._evaluate_switch_epic_guardrails(config, nb, n_experiments)
        summary["switch_epic_guardrails"] = switch_guardrails
        if switch_guardrails.get("should_switch_epic"):
            self._emit_event("switch_epic_recommended", switch_guardrails)
            try:
                nb.log_learning_event(
                    "switch_epic_recommended",
                    f"Switch-epic criteria met at cycle {n_experiments}",
                    evidence=json.dumps(switch_guardrails, sort_keys=True),
                )
            except Exception:
                pass
        self._emit_event("aria_cycle_completed", summary)

        delta_s1 = summary.get("delta_stage1_survivors", 0)
        after = summary.get("after", {})
        s0 = after.get("stage0_passed", 0)
        s05 = after.get("stage05_passed", 0)
        s1 = after.get("stage1_passed", 0)
        best_loss = after.get("best_loss_ratio")
        loss_str = f", best loss={best_loss:.4f}" if best_loss else ""
        logger.info(
            "Cycle %d done: S0=%d S0.5=%d S1=%d (ΔS1=%+d)%s",
            n_experiments, s0, s05, s1, delta_s1, loss_str,
        )

        # PROACTIVE: Auto-repair on cycle failure
        if cycle_error:
            self._proactive_cycle_repair(cycle_error, selected_mode, n_experiments, nb)

        # PROACTIVE: Detect stagnation and auto-adjust
        self._proactive_stagnation_check(n_experiments, delta_s1, nb)

        # PROACTIVE: Detect recurring error patterns and auto-fix
        self._proactive_recurring_error_fix(n_experiments, nb)

        # PROACTIVE: Spawn agent to investigate persistent S1 stagnation
        self._proactive_stagnation_agent(n_experiments, nb)
        self._maybe_trigger_integrity_healer(
            nb=nb,
            experiment_id=after_progress.get("experiment_id"),
        )

        return summary

    def _proactive_cycle_repair(self, error: str, mode: str,
                                cycle_index: int, nb: LabNotebook):
        """Spawn a code agent to fix cycle failures autonomously."""
        try:
            from ..code_agent import _spawn_code_agent_task, _should_autospawn_self_repair
            if not _should_autospawn_self_repair(error):
                return
            # Rate-limit: don't spawn repair agents more than once per 3 minutes
            now = time.time()
            last = getattr(self, "_last_cycle_repair_spawn", 0)
            if now - last < 180:
                return
            self._last_cycle_repair_spawn = now

            # Build targeted goal based on error type
            import re as _re
            goal_parts = [f"Cycle {cycle_index} ({mode}) failed with: {error[:500]}."]
            # Extract file references from traceback
            file_refs = _re.findall(r'File "([^"]+)", line (\d+)', error)
            if file_refs:
                goal_parts.append(
                    f"Key locations: {', '.join(f'{f}:{l}' for f, l in file_refs[:5])}."
                )
            # Detect error class for targeted advice
            if "ImportError" in error or "ModuleNotFoundError" in error:
                goal_parts.append("This is an import error — check module paths and __init__.py files.")
            elif "CUDA" in error or "RuntimeError" in error:
                goal_parts.append("This may be a CUDA/tensor error — check device placement and tensor shapes.")
            elif "TypeError" in error or "AttributeError" in error:
                goal_parts.append("This is a type/attribute error — check function signatures and object interfaces.")
            goal_parts.append(
                "Investigate root cause. Apply minimal safe fix. "
                "Use local Ollama model if available."
            )
            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=" ".join(goal_parts),
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            logger.info("Proactive repair agent spawned: %s for cycle %d error", task_id, cycle_index)
            nb.log_learning_event(
                "proactive_repair",
                f"Auto-spawned repair agent {task_id} for cycle {cycle_index} failure: {error[:200]}",
                task_id=task_id,
            )
        except Exception as e:
            logger.debug("Proactive cycle repair failed to spawn: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="repeated_exception",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Cycle failure in mode={mode}: {error[:240]}",
            reproduction_steps=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
            trigger_payload={"mode": mode, "cycle_index": cycle_index, "error": error[:1000]},
        )

    def _proactive_stagnation_check(self, cycle_index: int, delta_s1: int,
                                    nb: LabNotebook):
        """Detect stagnation (consecutive zero-survivor cycles) and auto-adjust."""
        history = getattr(self, "_aria_cycle_history", [])
        if len(history) < 3:
            return

        # Check last 3 cycles for zero survivors
        recent = history[-3:]
        consecutive_zero = all(
            (h.get("delta_stage1_survivors", 0) == 0 and
             (h.get("after") or {}).get("stage1_passed", 0) == 0)
            for h in recent
        )
        if not consecutive_zero:
            return

        # Already applied anti-stagnation recently?
        last_anti = getattr(self, "_last_anti_stagnation_cycle", -10)
        if cycle_index - last_anti < 3:
            return
        self._last_anti_stagnation_cycle = cycle_index

        # Apply anti-stagnation: diversify grammar, reduce depth, try different source
        adjustments = {
            "max_depth": max(4, getattr(self, "_grammar_weight_overrides", {}).get("max_depth", 6) - 2),
            "max_ops": max(4, getattr(self, "_grammar_weight_overrides", {}).get("max_ops", 8) - 2),
        }
        self._last_chat_config_overrides = {
            **(self._last_chat_config_overrides or {}),
            **adjustments,
        }
        # Reset any extreme grammar weights
        if hasattr(self, "_grammar_weight_overrides"):
            for k in list(self._grammar_weight_overrides.keys()):
                v = self._grammar_weight_overrides[k]
                if isinstance(v, (int, float)) and (v > 6.0 or v < 0.3):
                    self._grammar_weight_overrides[k] = 1.0

        nb.log_learning_event(
            "anti_stagnation",
            f"3 consecutive zero-S1 cycles detected at cycle {cycle_index}. "
            f"Auto-reduced depth/ops ({adjustments}), reset extreme grammar weights.",
            adjustments=adjustments,
        )
        logger.info(
            "Anti-stagnation triggered at cycle %d: %s",
            cycle_index, adjustments,
        )

    def _proactive_stagnation_agent(self, cycle_index: int, nb: LabNotebook):
        """Spawn a code agent to investigate persistent S1 stagnation."""
        history = getattr(self, "_aria_cycle_history", [])
        window = 5
        if len(history) < window:
            return

        recent = history[-window:]
        all_zero = all(
            h.get("delta_stage1_survivors", 0) == 0
            for h in recent
        )
        if not all_zero:
            return

        # Rate-limit: skip if recently spawned
        if cycle_index - self._last_stagnation_agent_cycle < 5:
            return

        try:
            from ..code_agent import _spawn_code_agent_task
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            neg = analytics.negative_results_synthesis()
            failed_ops = [op["op_name"] for op in neg.get("failed_ops", [])[:10]]
            weights = analytics.compute_grammar_weights() or {}
            trajectory = analytics.learning_trajectory()
            s1_slope = trajectory.get("s1_slope", 0) if trajectory else 0

            recent_s1 = [
                (h.get("after") or {}).get("stage1_passed", 0)
                for h in recent
            ]

            goal = (
                f"STAGNATION: 0 new S1 survivors for {window} consecutive cycles "
                f"(S1 counts: {recent_s1}, trend slope={s1_slope:.4f}). "
                f"Current grammar weights: {json.dumps(weights, indent=None)}. "
                f"Failed ops (0% S1 rate, >=5 uses): {failed_ops}. "
                f"Read these files and propose a targeted fix:\n"
                f"- synthesis/grammar.py: GrammarConfig.category_weights defaults and generate_layer_graph() — "
                f"are failing categories over-weighted?\n"
                f"- scientist/analytics.py: _compute_weights_from_stats() noise guard and weight formula — "
                f"is the learning signal being suppressed?\n"
                f"- morphological_box.py: dimension constraints in ArchSpec — "
                f"is the config search space too narrow?\n"
                f"- evaluator.py: stage1 pass criteria — is the bar set unreachably high?\n"
                f"Identify the single most likely bottleneck and make a minimal, targeted change."
            )

            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=goal,
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            self._last_stagnation_agent_cycle = cycle_index

            nb.log_learning_event(
                "proactive_stagnation_agent",
                f"Spawned agent {task_id} for S1 stagnation ({window} flat cycles)",
                task_id=task_id, s1_slope=s1_slope,
                failed_ops=failed_ops, recent_s1=recent_s1,
            )
            logger.info(
                "Stagnation agent spawned at cycle %d (task %s, slope=%.4f)",
                cycle_index, task_id, s1_slope,
            )
        except Exception as e:
            logger.debug("Stagnation agent spawn failed: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="plateau",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Persistent stagnation at cycle {cycle_index}",
            reproduction_steps=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
            trigger_payload={"cycle_index": cycle_index, "window": window},
        )

    def _proactive_recurring_error_fix(self, cycle_index: int, nb: LabNotebook):
        """Detect recurring error patterns across cycles and spawn a fix agent.

        If the same error class (first line of traceback) appears 2+ times
        in the last 5 cycles, spawn a code agent to fix the root cause.
        """
        history = getattr(self, "_aria_cycle_history", [])
        if len(history) < 2:
            return

        # Collect errors from recent cycles
        recent = history[-5:]
        errors = []
        for h in recent:
            err = h.get("error")
            if err:
                # Normalize: take the error class/first line
                first_line = str(err).split("\n")[0].strip()[:200]
                errors.append(first_line)

        if len(errors) < 2:
            return

        # Find repeated error patterns (same first 80 chars)
        from collections import Counter
        error_keys = [e[:80] for e in errors]
        counts = Counter(error_keys)
        repeated = [(k, c) for k, c in counts.items() if c >= 2]
        if not repeated:
            return

        # Rate-limit: don't spawn more than 1 recurring-error agent per 10 minutes
        now = time.time()
        last = getattr(self, "_last_recurring_error_agent", 0)
        if now - last < 600:
            return
        self._last_recurring_error_agent = now

        worst_error = max(repeated, key=lambda x: x[1])
        try:
            from ..code_agent import _spawn_code_agent_task
            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=(
                    f"RECURRING ERROR (seen {worst_error[1]}x in last 5 cycles): "
                    f"{worst_error[0]}. "
                    "This error keeps happening. Find the ROOT CAUSE in "
                    "scientist/runner.py, synthesis/, eval/, or training/ and fix it. "
                    "Use local Ollama model if available."
                ),
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            nb.log_learning_event(
                "proactive_recurring_error_fix",
                f"Spawned agent {task_id} for recurring error ({worst_error[1]}x): "
                f"{worst_error[0][:200]}",
                task_id=task_id,
                error_pattern=worst_error[0],
                occurrences=worst_error[1],
            )
            logger.info(
                "Recurring error agent spawned: %s for pattern seen %dx: %s",
                task_id, worst_error[1], worst_error[0][:100],
            )
        except Exception as e:
            logger.debug("Failed to spawn recurring error fix agent: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="repeated_exception",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Recurring error pattern {worst_error[0][:200]}",
            reproduction_steps=["python -m pytest tests/test_integration.py -k \"error\" -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_integration.py -k \"error\" -x --tb=short"],
            trigger_payload={"error_pattern": worst_error[0], "occurrences": worst_error[1]},
        )

    def _healer_signature_seen(self, error: str, scope: str) -> bool:
        """Return True if this error was already sent to the healer recently."""
        sig = hashlib.md5((error[:200] + scope[:100]).encode()).hexdigest()[:12]
        now = time.time()
        # Expire old signatures (5-minute TTL)
        self._recent_healer_signatures = {
            k: v for k, v in self._recent_healer_signatures.items()
            if now - v < 300
        }
        if sig in self._recent_healer_signatures:
            return True
        self._recent_healer_signatures[sig] = now
        return False

    def _invoke_code_healer(
        self,
        nb: LabNotebook,
        trigger_type: str,
        experiment_id: Optional[str],
        scope: str,
        reproduction_steps: Optional[List[str]] = None,
        acceptance_tests: Optional[List[str]] = None,
        trigger_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run Code Healer state machine and log output to SQLite."""
        if self._healer is None:
            return None
        if self._healer_signature_seen(
            (trigger_payload or {}).get("error", scope),
            scope,
        ):
            return None
        repro = reproduction_steps or []
        tests = acceptance_tests or ["python -m pytest tests -k integration -x --tb=short"]
        command_timeout_seconds = self._resolve_healer_timeout_seconds(nb, experiment_id)
        try:
            result = self._healer.open_and_run(
                HealerTaskSpec(
                    experiment_id=experiment_id,
                    trigger_type=trigger_type,
                    scope=scope,
                    reproduction_steps=repro,
                    acceptance_tests=tests,
                    trigger_payload={
                        **(trigger_payload or {}),
                        "command_timeout_seconds": command_timeout_seconds,
                    },
                    command_timeout_seconds=command_timeout_seconds,
                )
            )
            nb.log_learning_event(
                "code_healer_invoked",
                f"CodeHealer handled trigger={trigger_type} state={result.get('state')}",
                trigger_type=trigger_type,
                healer_result=result,
            )
            # If heal succeeded, schedule retry on next continuous cycle
            if result and result.get("verification_ok"):
                self._pending_heal_retry = {
                    "trigger_type": trigger_type,
                    "experiment_id": experiment_id,
                    "scope": scope,
                }
            return result
        except Exception as e:
            nb.log_learning_event(
                "code_healer_failed",
                f"CodeHealer failed for trigger={trigger_type}: {e}",
                trigger_type=trigger_type,
                error=str(e),
            )
            return None

    def _resolve_healer_timeout_seconds(
        self,
        nb: LabNotebook,
        experiment_id: Optional[str],
    ) -> int:
        """Resolve healer command timeout from the experiment config when available."""
        default_timeout = 180
        if not experiment_id:
            return default_timeout
        try:
            row = nb.conn.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None or not row["config_json"]:
                return default_timeout
            config_dict = json.loads(row["config_json"])
            config = RunConfig.from_dict(config_dict if isinstance(config_dict, dict) else {})
            return max(1, int(getattr(config, "max_agent_seconds", default_timeout) or default_timeout))
        except Exception:
            return default_timeout

    def _maybe_trigger_integrity_healer(self, nb: LabNotebook, experiment_id: Optional[str]) -> None:
        """Run integrity checks periodically and invoke healer on failures."""
        now = time.time()
        if now - self._last_healer_integrity_check < 300:
            return
        self._last_healer_integrity_check = now
        try:
            from ...tools.novelty_integrity_check import run_integrity_check
            report = run_integrity_check(nb, calibrate_if_missing=False, runs=4)
            if report.get("ok"):
                return
            self._invoke_code_healer(
                nb=nb,
                trigger_type="integrity_failure",
                experiment_id=experiment_id,
                scope="Novelty integrity check failure.",
                reproduction_steps=[
                    f"python tools/novelty_integrity_check.py --db {shlex.quote(str(nb.db_path))}"
                ],
                acceptance_tests=[
                    f"python tools/novelty_integrity_check.py --db {shlex.quote(str(nb.db_path))}"
                ],
                trigger_payload=report,
            )
        except Exception as e:
            logger.debug("Integrity healer check failed to execute: %s", e)

    @staticmethod
    def _config_with_overrides(
        base_config: RunConfig,
        overrides: Dict[str, Any],
    ) -> Tuple[RunConfig, Dict[str, Dict[str, Any]]]:
        """Apply allowed per-cycle mode overrides to a cloned RunConfig.

        User-supplied maximizing fields (max_depth, max_ops, grammar_split_prob,
        three_way_split_prob, residual_prob) are treated as floors — Aria and
        mode selectors can raise them but never lower them.
        """
        effective = RunConfig.from_dict(base_config.to_dict())
        applied: Dict[str, Any] = {}
        ignored: Dict[str, Any] = {}

        # Fields where the user's value is a floor (override can only raise)
        _FLOOR_FIELDS = frozenset({
            "max_depth", "max_ops", "min_depth",
            "grammar_split_prob", "three_way_split_prob",
            "residual_prob", "max_params_ratio", "min_splits",
        })

        for key, value in (overrides or {}).items():
            if not hasattr(effective, key):
                ignored[key] = value
                continue
            if key in _FLOOR_FIELDS and isinstance(value, (int, float)):
                user_val = getattr(base_config, key)
                if isinstance(user_val, (int, float)):
                    value = type(user_val)(max(user_val, value))
            setattr(effective, key, value)
            applied[key] = value

        return effective, {"applied": applied, "ignored": ignored}

    def _evaluate_switch_epic_guardrails(
        self,
        config: RunConfig,
        nb: LabNotebook,
        cycle_index: int,
    ) -> Dict[str, Any]:
        """Evaluate explicit criteria for switching to a new epic/strategy."""
        confidence_min = float(getattr(config, "switch_epic_breakthrough_confidence_min", 0.75) or 0.75)
        stagnation_cycles = max(3, int(getattr(config, "switch_epic_stagnation_cycles", 6) or 6))

        breakthroughs = nb.get_leaderboard(
            tier="breakthrough", limit=5, sort_by="composite_score", include_references=False
        )
        qualified_breakthroughs = []
        for row in breakthroughs:
            conf = float(row.get("novelty_confidence") or 0.0)
            if conf >= confidence_min:
                qualified_breakthroughs.append({
                    "result_id": row.get("result_id"),
                    "composite_score": float(row.get("composite_score") or 0.0),
                    "novelty_confidence": conf,
                })

        recent = list(self._aria_cycle_history[-stagnation_cycles:])
        if recent:
            recent = recent + [dict(self._last_cycle_summary or {})] if self._last_cycle_summary else recent
            recent = recent[-stagnation_cycles:]
        stagnated = bool(recent) and all(int(c.get("delta_stage1_survivors") or 0) == 0 for c in recent)

        should_switch = bool(qualified_breakthroughs) or stagnated
        reasons = []
        if qualified_breakthroughs:
            reasons.append("decision_ready_breakthrough_detected")
        if stagnated:
            reasons.append("stagnation_without_gate_advancement")

        return {
            "should_switch_epic": should_switch,
            "reasons": reasons,
            "criteria": {
                "breakthrough_confidence_min": confidence_min,
                "stagnation_cycles": stagnation_cycles,
            },
            "signals": {
                "qualified_breakthrough_count": len(qualified_breakthroughs),
                "qualified_breakthroughs": qualified_breakthroughs,
                "stagnated_recent_cycles": stagnated,
                "recent_cycle_count": len(recent),
            },
            "evaluated_at_cycle": int(cycle_index),
        }

    def _fail_active_cycle_experiment(
        self,
        nb: LabNotebook,
        error: str,
        expected_mode: Optional[str] = None,
    ) -> Optional[str]:
        """Fail the currently tracked cycle experiment to avoid stale `running` rows."""
        active_exp_id = self.progress.experiment_id
        if not active_exp_id:
            return None

        try:
            row = nb.conn.execute(
                "SELECT experiment_type, status FROM experiments WHERE experiment_id = ?",
                (active_exp_id,),
            ).fetchone()
            if not row or row["status"] != "running":
                return None
            if expected_mode and row["experiment_type"] != expected_mode:
                logger.debug(
                    "Skip failing experiment %s: type mismatch (%s != %s)",
                    active_exp_id,
                    row["experiment_type"],
                    expected_mode,
                )
                return None

            nb.fail_experiment(active_exp_id, error)
            with self._lock:
                if self._progress.experiment_id == active_exp_id:
                    self._progress.status = "failed"
                    self._progress.error = error
                    self._progress.aria_message = self.aria.react_to_failure(error)
            return active_exp_id
        except Exception as finalize_error:
            logger.warning(
                "Failed to mark active experiment %s as failed: %s",
                active_exp_id,
                finalize_error,
            )
            return None

    def _recover_stale_experiments_on_startup(self) -> None:
        """Clean up stale running experiments from previous crashed processes."""
        try:
            nb = self._make_notebook()
            cleaned = nb.cleanup_stale_experiments(timeout_minutes=60)
            if cleaned > 0:
                logger.info("Recovered %d stale experiments from previous crash", cleaned)
            nb.close()
        except Exception as e:
            logger.debug("Startup stale-experiment recovery failed: %s", e)
