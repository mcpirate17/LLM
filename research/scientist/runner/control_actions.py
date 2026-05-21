"""Control mixin: stop, events, chat actions, database maintenance."""

from __future__ import annotations

import os
import queue
import time
from typing import Any, Dict, Optional


from ..native.telemetry import reset_native_runner_telemetry
from ..notebook import LabNotebook
from ..runtime_events import (
    LIFECYCLE_EVENT_TYPES,
    get_runtime_event_services,
    publish_lifecycle_event,
    publish_runtime_event,
)
from ..runtime_events.publishers import publish_live_feed_event

from ._types import RunConfig

_LIVE_LOSS_CURVE_MAX_POINTS = 20000
_TRAINING_STEP_SSE_EVERY = 10

import logging

logger = logging.getLogger(__name__)

_PERSISTED_LIVE_FEED_EVENTS = {
    "experiment_started",
    "experiment_completed",
    "experiment_failed",
    "experiment_stopping",
    "scale_up_started",
    "scale_up_progress",
    "scale_up_completed",
    "champion_confirmation_started",
    "champion_probe_progress",
    "investigation_started",
    "investigation_progress",
    "investigation_training_complete",
    "investigation_completed",
    "validation_started",
    "validation_progress",
    "validation_completed",
    "breakthrough_detected",
    "auto_investigate_queued",
    "auto_validate_queued",
    "auto_scale_up_queued",
}


class _ControlActionsMixin:
    """Stop/events/chat-action methods for ExperimentRunner."""

    __slots__ = ()

    def _persist_live_feed_event(self, event_type: str, data: Dict[str, Any]):
        """Persist selected lifecycle events for feed replay in the dashboard."""
        if event_type not in _PERSISTED_LIVE_FEED_EVENTS:
            return
        try:
            publish_live_feed_event(
                notebook_path=self.notebook_path,
                event_type=event_type,
                data=data,
            )
        except Exception as exc:
            logger.debug("Failed to persist live-feed event %s: %s", event_type, exc)

    def stop(self):
        """Stop the current experiment gracefully."""
        self._stop_event.set()
        self.aria.state.mood = "contemplative"
        with self._lock:
            self._aria_cycle_paused = False
        self._set_aria_cycle_phase(
            "stopping",
            continuous_active=self.is_running,
            note="Stop requested; wrapping up current work.",
        )
        with self._lock:
            self._progress.status = "stopped"
            self._progress.aria_message = "Stopping... wrapping up current evaluation."

            # Z17: Clear global native-runner counters immediately on stop
            reset_native_runner_telemetry()

        if bool(self._aria_cycle_status.get("continuous_active")):
            try:
                publish_runtime_event(
                    notebook_path=self.notebook_path,
                    event_type="continuous_session_stopping",
                    producer="runner.control_actions",
                    run_id=str(
                        getattr(self.progress, "experiment_id", "") or ""
                    ).strip()
                    or None,
                    payload={
                        "mode": "continuous",
                        "status": "stopping",
                    },
                )
            except Exception:
                logger.warning(
                    "Runtime continuous session stop publish failed",
                    exc_info=True,
                )

        self._emit_event("experiment_stopping", {})

    # ── Routing Benchmark Harness (Track C) ──

    def get_events(self, timeout: float = 30.0):
        """Generator yielding events for SSE streaming."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                event = self._event_queue.get(timeout=1.0)
                yield event
            except queue.Empty:
                # Send keepalive
                yield {"type": "keepalive", "data": {}, "timestamp": time.time()}

    # ── Start / Stop ──

    def _emit_event(self, event_type: str, data: Dict):
        """Push an event for SSE consumers."""
        # training_step can emit at very high frequency; throttle SSE pressure so
        # structural live-feed events (program_evaluated/validation_progress/etc.)
        # are not dropped when the event queue is saturated.
        should_enqueue = True
        if event_type == "training_step":
            step = int(data.get("step") or 0)
            total_steps = int(data.get("total_steps") or 0)
            should_enqueue = (
                step <= 1
                or (step % _TRAINING_STEP_SSE_EVERY == 0)
                or (total_steps > 0 and step >= total_steps)
            )

        payload = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }

        try:
            if should_enqueue:
                self._event_queue.put_nowait(payload)
        except queue.Full:
            if event_type != "training_step":
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(payload)
        self._persist_live_feed_event(event_type, data)
        self._publish_runtime_lifecycle_event(event_type, data)
        # Buffer training_step events for REST retrieval (dashboard chart restore).
        # Keep a deep enough history so the dashboard can reconstruct near-full
        # curves for long validation/investigation runs.
        if event_type == "training_step":
            curve = self._live_loss_curve
            exp_id = data.get("experiment_id", "")
            if curve and curve[0].get("experiment_id") != exp_id:
                curve.clear()
            curve.append(data)
            if len(curve) > _LIVE_LOSS_CURVE_MAX_POINTS:
                del curve[: len(curve) - _LIVE_LOSS_CURVE_MAX_POINTS]

    def _publish_runtime_lifecycle_event(
        self, event_type: str, data: Dict[str, Any]
    ) -> None:
        if event_type not in LIFECYCLE_EVENT_TYPES:
            return
        run_id = str(data.get("experiment_id") or "").strip() or None
        if run_id is None:
            return
        try:
            current = get_runtime_event_services(self.notebook_path).registry.get(
                run_id
            )
            if current is not None and current.last_event.event_type == event_type:
                return
        except Exception:
            logger.debug(
                "Runtime lifecycle dedupe probe failed for %s (%s)",
                event_type,
                run_id,
                exc_info=True,
            )
        try:
            publish_lifecycle_event(
                notebook_path=self.notebook_path,
                event_type=event_type,
                producer="runner.control_actions",
                run_id=run_id,
                payload=data,
            )
        except Exception:
            logger.warning(
                "Runtime lifecycle publish failed for %s (%s)",
                event_type,
                run_id,
                exc_info=True,
            )

    def execute_chat_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an action dispatched from Aria's chat response.

        Supported types: adjust_config, adjust_grammar, start_experiment, edit_file.
        """
        action_type = str(action.get("type") or "").strip()

        if action_type == "adjust_config":
            changes = action.get("changes") or {}
            if not isinstance(changes, dict) or not changes:
                return {"status": "error", "error": "No changes provided"}
            # Apply via _config_with_overrides on a fresh default config
            base = RunConfig()
            effective, report = self._config_with_overrides(base, changes)
            # Store as the new defaults for future experiments
            self._last_chat_config_overrides = changes
            self._log_learning_event_compat(
                nb,
                "chat_config_adjusted",
                f"Aria adjusted config: {report.get('applied', {})}",
                changes=report.get("applied", {}),
                ignored=report.get("ignored", {}),
            )
            return {
                "status": "applied",
                "changes": report.get("applied", {}),
                "ignored": report.get("ignored", {}),
            }

        elif action_type == "adjust_grammar":
            weights = action.get("weights") or {}
            if not isinstance(weights, dict) or not weights:
                return {"status": "error", "error": "No weights provided"}
            # Validate values are numeric
            clean_weights = {}
            for k, v in weights.items():
                try:
                    clean_weights[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            if not clean_weights:
                return {"status": "error", "error": "No valid numeric weights"}
            self._grammar_weight_overrides.update(clean_weights)
            self._log_learning_event_compat(
                nb,
                "chat_grammar_adjusted",
                f"Aria adjusted grammar weights: {clean_weights}",
                weights=clean_weights,
                all_overrides=dict(self._grammar_weight_overrides),
            )
            return {"status": "applied", "weights": clean_weights}

        elif action_type == "start_experiment":
            if self.is_running:
                return {"status": "busy", "error": "An experiment is already running"}
            mode = str(action.get("mode") or "synthesis").strip().lower()
            config_overrides = action.get("config") or {}
            config = RunConfig()
            if isinstance(config_overrides, dict):
                for k, v in config_overrides.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
            try:
                if mode in {
                    "sparse_morph",
                    "sparse_morphology",
                    "sparse_morphological",
                }:
                    config.model_source = "morphological_box"
                    config.morph_focus_sparse = True
                    config.n_programs = max(120, int(config.n_programs))
                    config.n_layers = max(1, min(int(config.n_layers), 4))
                    config.max_depth = max(2, min(int(config.max_depth), 6))
                    config.max_ops = max(4, min(int(config.max_ops), 10))
                    exp_id = self.start_experiment(config)
                if mode == "evolution":
                    exp_id = self.start_evolution(config)
                elif mode == "novelty":
                    exp_id = self.start_novelty_search(config)
                elif mode in {
                    "sparse_morph",
                    "sparse_morphology",
                    "sparse_morphological",
                }:
                    pass
                else:
                    exp_id = self.start_experiment(config)
                return {"status": "started", "experiment_id": exp_id, "mode": mode}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif action_type == "edit_file":
            return self._execute_edit_file_action(action, nb)

        elif action_type == "maintain_database":
            return self._execute_maintain_database_action(action, nb)

        else:
            return {"status": "error", "error": f"Unknown action type: {action_type}"}

    def _execute_edit_file_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an edit_file action with safety rails."""
        import py_compile
        import shutil

        path = str(action.get("path") or "").strip()
        search = str(action.get("search") or "")
        replace = str(action.get("replace") or "")
        description = str(action.get("description") or "Chat-initiated edit")

        # Safety: reject path traversal
        if ".." in path:
            return {"status": "error", "error": "Path traversal (..) not allowed"}

        # Safety: allow edits only within known project subpaths
        allowed_prefixes = (
            "research/",
            "scientist/",
            "synthesis/",
            "eval/",
            "search/",
            "training/",
            "dashboard/",
            "tests/",
            "tools/",
            "mathspaces/",
        )
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            return {
                "status": "error",
                "error": "Path must be under research/ or a known project folder",
            }

        # Safety: only .py and .js files
        if not (path.endswith(".py") or path.endswith(".js")):
            return {"status": "error", "error": "Only .py and .js files can be edited"}

        # Resolve to absolute path.
        # project_root is typically <repo>/research when running from the package layout.
        # If the incoming path already starts with research/, resolve from repo root;
        # otherwise resolve from project_root directly.
        # __file__ is runner/control_actions.py; go up 3 levels to reach research/
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        repo_root = os.path.dirname(project_root)
        if path.startswith("research/"):
            abs_path = os.path.normpath(os.path.join(repo_root, path))
        else:
            abs_path = os.path.normpath(os.path.join(project_root, path))

        # Double-check resolved path is under project
        if not abs_path.startswith(project_root):
            return {
                "status": "error",
                "error": "Resolved path escapes project directory",
            }

        if not os.path.isfile(abs_path):
            return {"status": "error", "error": f"File not found: {path}"}

        # Read current content
        with open(abs_path, "r") as f:
            content = f.read()

        if search not in content:
            return {"status": "error", "error": "Search string not found in file"}

        # Create backup
        timestamp = int(time.time())
        backup_path = f"{abs_path}.bak.{timestamp}"
        shutil.copy2(abs_path, backup_path)

        # Apply edit
        new_content = content.replace(search, replace, 1)
        with open(abs_path, "w") as f:
            f.write(new_content)

        # Syntax check for .py files
        if path.endswith(".py"):
            try:
                py_compile.compile(abs_path, doraise=True)
            except py_compile.PyCompileError as e:
                # Restore backup
                shutil.copy2(backup_path, abs_path)
                os.remove(backup_path)
                return {
                    "status": "error",
                    "error": f"Syntax error after edit, reverted: {e}",
                }

        # Log to notebook
        self._log_learning_event_compat(
            nb,
            "chat_file_edited",
            f"Aria edited {path}: {description}",
            path=path,
            backup=backup_path,
            description=description,
        )

        return {
            "status": "applied",
            "path": path,
            "backup": backup_path,
            "description": description,
        }

    # ── Database Maintenance Actions ──────────────────────────────────────

    def _execute_maintain_database_action(
        self,
        action: Dict[str, Any],
        nb: LabNotebook,
    ) -> Dict[str, Any]:
        """Execute a database maintenance operation.

        Allowed operations:
          purge_empty_experiments  — delete failed experiments with no results
          purge_junk_programs      — delete S0 failures with no error classification
          reset_op_stats           — reset op_success_rates for specific ops
          clear_toxic_signatures   — remove failure_signatures for specific ops
          vacuum                   — reclaim disk space
          backfill_failure_signatures — one-time backfill from existing results
        """
        operation = str(action.get("operation") or "").strip()
        if operation not in self._MAINTENANCE_OPS:
            return {
                "status": "error",
                "error": f"Unknown maintenance operation: {operation}. "
                f"Allowed: {', '.join(sorted(self._MAINTENANCE_OPS))}",
            }

        try:
            if operation == "purge_empty_experiments":
                n = nb.purge_empty_experiments()
                self._log_learning_event_compat(
                    nb,
                    "maintenance_purge_experiments",
                    f"Aria purged {n} empty failed experiments",
                )
                return {"status": "applied", "deleted_experiments": n}

            elif operation == "purge_junk_programs":
                # Route through notebook helper so dependent rows are cleaned too.
                result = nb.purge_junk_programs(dry_run=False)
                n = int(result.get("deleted", 0) or 0)
                self._log_learning_event_compat(
                    nb,
                    "maintenance_purge_junk",
                    f"Aria purged {n} junk S0 failure records",
                )
                return {"status": "applied", "deleted_programs": n}

            elif operation == "reset_op_stats":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {
                        "status": "error",
                        "error": "Provide 'ops' list of op names to reset",
                    }
                op_names = [str(o).strip() for o in ops if str(o).strip()]
                if not op_names:
                    return {"status": "error", "error": "No valid op names provided"}
                placeholders = ",".join("?" * len(op_names))
                cur = nb.conn.execute(
                    f"DELETE FROM op_success_rates WHERE op_name IN ({placeholders})",
                    op_names,
                )
                n = cur.rowcount
                nb._maybe_commit()
                self._log_learning_event_compat(
                    nb,
                    "maintenance_reset_op_stats",
                    f"Aria reset op stats for {op_names} ({n} rows)",
                    ops=op_names,
                )
                return {"status": "applied", "ops_reset": op_names, "rows_deleted": n}

            elif operation == "clear_toxic_signatures":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {
                        "status": "error",
                        "error": "Provide 'ops' list of op names to clear signatures for",
                    }
                total = 0
                for op in ops:
                    op = str(op).strip()
                    if not op:
                        continue
                    cur = nb.conn.execute(
                        "DELETE FROM failure_signatures WHERE signature LIKE ?",
                        (f"%{op}%",),
                    )
                    total += cur.rowcount
                nb._maybe_commit()
                self._log_learning_event_compat(
                    nb,
                    "maintenance_clear_toxic",
                    f"Aria cleared {total} toxic signatures for {ops}",
                    ops=[str(o).strip() for o in ops],
                )
                return {
                    "status": "applied",
                    "signatures_deleted": total,
                    "ops": [str(o).strip() for o in ops],
                }

            elif operation == "vacuum":
                nb.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # VACUUM requires isolation_level=None and its own connection —
                # cannot go through the shared NativeConnectionWrapper.
                import sqlite3

                vac_conn = sqlite3.connect(nb.db_path, isolation_level=None)
                vac_conn.execute("VACUUM")
                vac_conn.close()
                self._log_learning_event_compat(
                    nb,
                    "maintenance_vacuum",
                    "Aria ran VACUUM to reclaim disk space",
                )
                return {"status": "applied", "operation": "vacuum"}

            elif operation == "backfill_failure_signatures":
                n = nb.backfill_failure_signatures()
                return {"status": "applied", "signatures_created": n}

        except Exception as e:
            logger.warning("Maintenance action %s failed: %s", operation, e)
            return {"status": "error", "error": str(e)[:200]}

        return {"status": "error", "error": "Unreachable"}

    @property
    def last_recommendation(self) -> Optional[Dict]:
        """Last auto-generated recommendation after experiment completion."""
        with self._lock:
            rec = self._last_recommendation
            # Clear after reading so dashboard only shows it once
            self._last_recommendation = None
            return rec
