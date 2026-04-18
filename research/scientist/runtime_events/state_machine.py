from __future__ import annotations

import logging
from typing import Optional

from .schema import RuntimeEvent

logger = logging.getLogger(__name__)

_ALLOWED_TRANSITIONS = {
    None: {"experiment_start_requested", "experiment_started"},
    "experiment_start_requested": {
        "experiment_started",
        "experiment_start_failed",
    },
    "experiment_started": {
        "experiment_completed",
        "experiment_failed",
    },
    "experiment_failed": {"experiment_started"},
    "experiment_start_failed": set(),
    "experiment_completed": set(),
}

_TERMINAL_TYPES = frozenset(
    {"experiment_start_failed", "experiment_completed", "experiment_failed"}
)


class LifecycleConflictError(RuntimeError):
    """Raised when a lifecycle event conflicts with the accepted state."""


def is_terminal_conflict(current_type: Optional[str], new_type: str) -> bool:
    return current_type in _TERMINAL_TYPES and new_type in _TERMINAL_TYPES


class LifecycleStateMachine:
    """Enforces lifecycle ordering and duplicate/conflict policy."""

    def is_valid_transition(self, current_type: Optional[str], new_type: str) -> bool:
        """Check if a transition is allowed without requiring full events."""
        if new_type == current_type:
            return True  # duplicate — caller decides handling
        allowed = _ALLOWED_TRANSITIONS.get(current_type, set())
        return new_type in allowed

    def transition(
        self,
        current_event: Optional[RuntimeEvent],
        new_event: RuntimeEvent,
    ) -> RuntimeEvent:
        current_type = current_event.event_type if current_event is not None else None
        if new_event.event_type == current_type:
            if current_event is not None and current_event.payload != new_event.payload:
                raise LifecycleConflictError(
                    f"Conflicting duplicate lifecycle event {new_event.event_type}"
                )
            return current_event or new_event

        allowed = _ALLOWED_TRANSITIONS.get(current_type, set())
        if current_type in _TERMINAL_TYPES and new_event.event_type not in allowed:
            log = logger.debug if is_terminal_conflict(current_type, new_event.event_type) else logger.warning
            log(
                "Rejected lifecycle event: run_id=%s already terminal at %s, cannot accept %s",
                new_event.run_id,
                current_type,
                new_event.event_type,
            )
            raise LifecycleConflictError(
                f"Lifecycle already terminal at {current_type}; cannot accept {new_event.event_type}"
            )

        if new_event.event_type not in allowed:
            logger.warning(
                "Rejected lifecycle transition: run_id=%s %r -> %r",
                new_event.run_id,
                current_type,
                new_event.event_type,
            )
            raise LifecycleConflictError(
                f"Invalid lifecycle transition {current_type!r} -> {new_event.event_type!r}"
            )

        return new_event
