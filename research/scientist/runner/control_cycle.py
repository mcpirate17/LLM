"""Control mixin: Aria cycle phase management."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import logging
logger = logging.getLogger(__name__)


class _ControlCycleMixin:
    """Aria continuous-cycle phase tracking for ExperimentRunner."""

    __slots__ = ()

    @staticmethod
    def _aria_phase_label(phase: str) -> str:
        labels = {
            "idle": "Idle",
            "planning": "Planning",
            "running": "Running",
            "analyzing": "Analyzing",
            "paused": "Paused",
            "stopping": "Stopping",
            "completed": "Completed",
            "failed": "Failed",
        }
        return labels.get(phase, phase.replace("_", " ").title())

    def _set_aria_cycle_phase(
        self,
        phase: str,
        *,
        cycle_index: Optional[int] = None,
        selected_mode: Optional[str] = None,
        note: Optional[str] = None,
        continuous_active: Optional[bool] = None,
        emit_event: bool = True,
    ) -> None:
        """Track Aria's continuous cycle phase for observability APIs/UI."""
        with self._lock:
            payload: Dict[str, Any] = {
                "phase": str(phase or "idle"),
                "phase_label": self._aria_phase_label(str(phase or "idle")),
                "last_transition_ts": time.time(),
            }
            if cycle_index is not None:
                payload["cycle_index"] = int(cycle_index)
            if selected_mode is not None:
                payload["selected_mode"] = str(selected_mode)
            if note is not None:
                payload["last_note"] = str(note)
            if continuous_active is not None:
                payload["continuous_active"] = bool(continuous_active)
            if phase == "running" and selected_mode is not None:
                payload["last_completed_mode"] = None
            if phase in {"analyzing", "completed", "failed"} and selected_mode is not None:
                payload["last_completed_mode"] = str(selected_mode)

            self._aria_cycle_status.update(payload)
            snapshot = dict(self._aria_cycle_status)

        if emit_event:
            self._emit_event("aria_cycle_phase", snapshot)

    def get_aria_cycle_status(self) -> Dict[str, Any]:
        """Return latest Aria cycle status for dashboard/API polling."""
        with self._lock:
            cycle = dict(self._aria_cycle_status)
            progress = self._progress.to_dict()
            last_cycle = dict(self._last_cycle_summary) if self._last_cycle_summary else None
            cycle_history = [dict(item) for item in self._aria_cycle_history[-10:]]
            cycle_paused = bool(self._aria_cycle_paused)
        cycle["is_running"] = self.is_running
        cycle["progress_status"] = progress.get("status")
        cycle["aria_message"] = progress.get("aria_message")
        cycle["experiment_id"] = progress.get("experiment_id")
        cycle["last_cycle_summary"] = last_cycle
        cycle["cycle_history"] = cycle_history
        cycle["cycle_paused"] = cycle_paused
        return cycle

    def pause_aria_cycle(self) -> Dict[str, Any]:
        """Pause continuous cycle progression between experiment iterations."""
        with self._lock:
            self._aria_cycle_paused = True
            running = self.is_running
        note = (
            "Pause requested; pausing before the next cycle."
            if running
            else "Cycle is paused. Start continuous mode to resume execution."
        )
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=running,
            note=note,
        )
        self._emit_event("aria_cycle_paused", {"note": note})
        return self.get_aria_cycle_status()

    def resume_aria_cycle(self) -> Dict[str, Any]:
        """Resume continuous cycle progression."""
        with self._lock:
            self._aria_cycle_paused = False
            running = self.is_running
            cycle_index = int(self._aria_cycle_status.get("cycle_index") or 0)
        self._set_aria_cycle_phase(
            "planning" if running else "idle",
            continuous_active=running,
            cycle_index=cycle_index,
            note="Cycle resumed." if running else "Cycle resumed and awaiting start.",
        )
        self._emit_event("aria_cycle_resumed", {"running": running})
        return self.get_aria_cycle_status()

    def _wait_for_cycle_resume(self, cycle_index: int) -> None:
        """Block between cycles while paused, unless stop is requested."""
        with self._lock:
            paused = bool(self._aria_cycle_paused)
        if not paused:
            return
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=True,
            cycle_index=cycle_index,
            note="Cycle paused; waiting for resume.",
        )
        while not self._stop_event.is_set():
            with self._lock:
                paused = bool(self._aria_cycle_paused)
            if not paused:
                break
            time.sleep(0.5)

    def _build_aria_cycle_summary(
        self,
        *,
        cycle_index: int,
        selected_mode: str,
        mode_reasoning: str,
        mode_confidence: Optional[float],
        before_progress: Dict[str, Any],
        after_progress: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a compact cycle summary payload for SSE/UI/chat consumers."""
        before_total = int(before_progress.get("total_programs") or 0)
        after_total = int(after_progress.get("total_programs") or 0)
        before_s1 = int(before_progress.get("stage1_passed") or 0)
        after_s1 = int(after_progress.get("stage1_passed") or 0)

        summary = {
            "cycle_index": int(cycle_index),
            "mode": str(selected_mode or "synthesis"),
            "reasoning": str(mode_reasoning or ""),
            "confidence": float(mode_confidence or 0.0),
            "status": "failed" if error else "completed",
            "programs_total": after_total,
            "stage1_survivors": after_s1,
            "delta_programs": max(0, after_total - before_total),
            "delta_stage1_survivors": max(0, after_s1 - before_s1),
            "aria_message": str(after_progress.get("aria_message") or ""),
            "timestamp": time.time(),
            "before": dict(before_progress or {}),
            "after": dict(after_progress or {}),
        }
        if error:
            summary["error"] = str(error)
        return summary
