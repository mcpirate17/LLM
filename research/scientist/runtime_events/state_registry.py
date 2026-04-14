from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .schema import LIFECYCLE_EVENT_TYPES, RuntimeEvent
from .state_machine import LifecycleStateMachine

logger = logging.getLogger(__name__)


@dataclass
class RuntimeLifecycleState:
    run_id: str
    last_event: RuntimeEvent

    @property
    def status(self) -> str:
        event_type = self.last_event.event_type
        if event_type == "experiment_started":
            return "running"
        if event_type == "experiment_completed":
            return "completed"
        if event_type in {"experiment_failed", "experiment_start_failed"}:
            return "failed"
        return "pending"


class RuntimeLifecycleRegistry:
    """Tracks latest accepted lifecycle event per run in memory."""

    def __init__(self) -> None:
        self._machine = LifecycleStateMachine()
        self._states: Dict[str, RuntimeLifecycleState] = {}
        self.spool_unhealthy = False
        self.projector_unhealthy = False

    def consume(self, event: RuntimeEvent) -> None:
        if event.event_type not in LIFECYCLE_EVENT_TYPES or not event.run_id:
            return
        current = self._states.get(event.run_id)
        prev_status = current.status if current is not None else None
        next_event = self._machine.transition(
            current.last_event if current is not None else None,
            event,
        )
        self._states[event.run_id] = RuntimeLifecycleState(
            run_id=event.run_id,
            last_event=next_event,
        )
        new_status = self._states[event.run_id].status
        if new_status != prev_status:
            logger.info(
                "Registry state change: run_id=%s %s -> %s (event=%s producer=%s)",
                event.run_id,
                prev_status or "none",
                new_status,
                event.event_type,
                event.producer,
            )

    def get(self, run_id: str) -> Optional[RuntimeLifecycleState]:
        return self._states.get(run_id)

    def active_run_id(self) -> Optional[str]:
        running = [
            state.run_id for state in self._states.values() if state.status == "running"
        ]
        return sorted(running)[-1] if running else None
