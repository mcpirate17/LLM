"""
Aria Autonomy Engine — Autonomous decision-making with configurable trust.

Trust levels control what Aria does without asking:
- FULL: Aria does everything (grammar tuning, kill/restart, promote, pivot strategy)
- SUPERVISED: Aria acts but flags decisions for review (user can undo within 5 min)
- ADVISORY: Aria recommends but waits for approval (current behavior)
"""

from __future__ import annotations

import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrustLevel(enum.Enum):
    """How much latitude Aria has to act without user approval."""
    FULL = "full"
    SUPERVISED = "supervised"
    ADVISORY = "advisory"


class ActionBehavior(enum.Enum):
    """Per-decision-type behavior at the current trust level."""
    AUTO = "auto"        # Execute immediately, log in activity feed
    NOTIFY = "notify"    # Execute immediately, show prominent card to user
    ASK = "ask"          # Queue as pending, wait for user approval


class DecisionType(enum.Enum):
    """Types of autonomous decisions Aria can make."""
    GRAMMAR_WEIGHT_ADJUSTMENT = "grammar_weight_adjustment"
    KILL_STALLED_EXPERIMENT = "kill_stalled_experiment"
    RESTART_WITH_NEW_PARAMS = "restart_with_new_params"
    PROMOTE_TO_VALIDATION = "promote_to_validation"
    PROMOTE_TO_BREAKTHROUGH = "promote_to_breakthrough"
    PIVOT_SEARCH_STRATEGY = "pivot_search_strategy"
    EXPORT_FOR_PUBLICATION = "export_for_publication"


# Default behavior matrix per trust level
_DEFAULT_BEHAVIORS: Dict[TrustLevel, Dict[DecisionType, ActionBehavior]] = {
    TrustLevel.FULL: {
        DecisionType.GRAMMAR_WEIGHT_ADJUSTMENT: ActionBehavior.AUTO,
        DecisionType.KILL_STALLED_EXPERIMENT: ActionBehavior.AUTO,
        DecisionType.RESTART_WITH_NEW_PARAMS: ActionBehavior.AUTO,
        DecisionType.PROMOTE_TO_VALIDATION: ActionBehavior.AUTO,
        DecisionType.PROMOTE_TO_BREAKTHROUGH: ActionBehavior.NOTIFY,
        DecisionType.PIVOT_SEARCH_STRATEGY: ActionBehavior.NOTIFY,
        DecisionType.EXPORT_FOR_PUBLICATION: ActionBehavior.ASK,
    },
    TrustLevel.SUPERVISED: {
        DecisionType.GRAMMAR_WEIGHT_ADJUSTMENT: ActionBehavior.AUTO,
        DecisionType.KILL_STALLED_EXPERIMENT: ActionBehavior.NOTIFY,
        DecisionType.RESTART_WITH_NEW_PARAMS: ActionBehavior.NOTIFY,
        DecisionType.PROMOTE_TO_VALIDATION: ActionBehavior.NOTIFY,
        DecisionType.PROMOTE_TO_BREAKTHROUGH: ActionBehavior.ASK,
        DecisionType.PIVOT_SEARCH_STRATEGY: ActionBehavior.ASK,
        DecisionType.EXPORT_FOR_PUBLICATION: ActionBehavior.ASK,
    },
    TrustLevel.ADVISORY: {
        DecisionType.GRAMMAR_WEIGHT_ADJUSTMENT: ActionBehavior.AUTO,
        DecisionType.KILL_STALLED_EXPERIMENT: ActionBehavior.ASK,
        DecisionType.RESTART_WITH_NEW_PARAMS: ActionBehavior.ASK,
        DecisionType.PROMOTE_TO_VALIDATION: ActionBehavior.ASK,
        DecisionType.PROMOTE_TO_BREAKTHROUGH: ActionBehavior.ASK,
        DecisionType.PIVOT_SEARCH_STRATEGY: ActionBehavior.ASK,
        DecisionType.EXPORT_FOR_PUBLICATION: ActionBehavior.ASK,
    },
}

UNDO_WINDOW_SECONDS = 300  # 5 minutes


@dataclass
class AutonomousAction:
    """A single autonomous decision that Aria made or is proposing."""
    action_id: str
    decision_type: str
    behavior: str  # auto / notify / ask
    title: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending / executed / undone / dismissed / expired
    created_at: float = 0.0
    executed_at: Optional[float] = None
    undo_snapshot: Optional[Dict[str, Any]] = None
    experiment_id: Optional[str] = None
    result_id: Optional[str] = None

    @property
    def undoable(self) -> bool:
        """Can this action still be undone?"""
        if self.status != "executed" or self.executed_at is None:
            return False
        if self.undo_snapshot is None:
            return False
        return (time.time() - self.executed_at) < UNDO_WINDOW_SECONDS

    @property
    def undo_remaining_seconds(self) -> int:
        if not self.undoable or self.executed_at is None:
            return 0
        return max(0, int(UNDO_WINDOW_SECONDS - (time.time() - self.executed_at)))

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "action_id": self.action_id,
            "decision_type": self.decision_type,
            "behavior": self.behavior,
            "title": self.title,
            "summary": self.summary,
            "detail": self.detail,
            "status": self.status,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "undoable": self.undoable,
            "undo_remaining_seconds": self.undo_remaining_seconds,
            "experiment_id": self.experiment_id,
            "result_id": self.result_id,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AutonomousAction":
        return cls(
            action_id=d.get("action_id", ""),
            decision_type=d.get("decision_type", ""),
            behavior=d.get("behavior", "ask"),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            detail=d.get("detail") or {},
            status=d.get("status", "pending"),
            created_at=d.get("created_at", 0.0),
            executed_at=d.get("executed_at"),
            undo_snapshot=d.get("undo_snapshot"),
            experiment_id=d.get("experiment_id"),
            result_id=d.get("result_id"),
        )


class AriaAutonomy:
    """Manages Aria's autonomous operation loop and decision dispatch."""

    def __init__(self, notebook=None):
        self._trust_level = TrustLevel.SUPERVISED
        self._overrides: Dict[str, str] = {}  # decision_type -> behavior override
        self._actions: List[AutonomousAction] = []
        self._undo_handlers: Dict[str, Callable] = {}
        self._notebook = notebook
        self._max_stored_actions = 100

    # ── Trust level management ──────────────────────────────────────

    @property
    def trust_level(self) -> TrustLevel:
        return self._trust_level

    @trust_level.setter
    def trust_level(self, value: TrustLevel) -> None:
        old = self._trust_level
        self._trust_level = value
        if old != value:
            logger.info("Aria trust level changed: %s -> %s", old.value, value.value)
            self._log_to_notebook(
                "trust_level_change",
                f"Trust level changed from {old.value} to {value.value}",
            )

    def get_behavior(self, decision_type: DecisionType) -> ActionBehavior:
        """Get the effective behavior for a decision type."""
        # Check per-type overrides first
        override = self._overrides.get(decision_type.value)
        if override:
            try:
                return ActionBehavior(override)
            except ValueError:
                pass
        # Fall back to trust-level defaults
        behaviors = _DEFAULT_BEHAVIORS.get(self._trust_level, {})
        return behaviors.get(decision_type, ActionBehavior.ASK)

    def set_override(self, decision_type: str, behavior: str) -> None:
        """Set a per-decision-type behavior override."""
        # Validate
        DecisionType(decision_type)
        ActionBehavior(behavior)
        self._overrides[decision_type] = behavior

    def clear_override(self, decision_type: str) -> None:
        self._overrides.pop(decision_type, None)

    # ── Decision dispatch ───────────────────────────────────────────

    def propose(
        self,
        decision_type: DecisionType,
        title: str,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        execute_fn: Optional[Callable[[], Optional[Dict]]] = None,
        undo_fn: Optional[Callable[[], None]] = None,
        undo_snapshot: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        result_id: Optional[str] = None,
    ) -> AutonomousAction:
        """Propose an autonomous decision. Executes or queues based on trust level.

        Args:
            decision_type: The type of decision
            title: Human-readable title
            summary: One-line plain-English summary
            detail: Extra data for the action card
            execute_fn: Callable to actually perform the action (returns optional undo snapshot)
            undo_fn: Callable to reverse the action (if undoable)
            undo_snapshot: Pre-computed snapshot for undo (alternative to undo_fn return)
            experiment_id: Associated experiment (if any)
            result_id: Associated program result (if any)

        Returns:
            The created AutonomousAction
        """
        behavior = self.get_behavior(decision_type)

        action = AutonomousAction(
            action_id=str(uuid.uuid4())[:12],
            decision_type=decision_type.value,
            behavior=behavior.value,
            title=title,
            summary=summary,
            detail=detail or {},
            status="pending",
            created_at=time.time(),
            experiment_id=experiment_id,
            result_id=result_id,
        )

        if behavior in (ActionBehavior.AUTO, ActionBehavior.NOTIFY):
            # Execute immediately
            snapshot = None
            if execute_fn:
                try:
                    snapshot = execute_fn()
                except Exception as e:
                    logger.error("Autonomous action %s failed: %s", action.action_id, e)
                    action.status = "failed"
                    action.detail["error"] = str(e)
                    self._store_action(action)
                    return action

            action.status = "executed"
            action.executed_at = time.time()
            action.undo_snapshot = snapshot or undo_snapshot

            if undo_fn:
                self._undo_handlers[action.action_id] = undo_fn

            self._log_to_notebook(
                f"autonomous_{decision_type.value}",
                f"[{behavior.value.upper()}] {title}: {summary}",
                detail=detail,
            )
        # else: ASK — stays pending, user must approve

        self._store_action(action)
        return action

    def approve(self, action_id: str, execute_fn: Optional[Callable] = None) -> Optional[AutonomousAction]:
        """User approves a pending action."""
        action = self._find_action(action_id)
        if not action or action.status != "pending":
            return None

        if execute_fn:
            try:
                snapshot = execute_fn()
                action.undo_snapshot = snapshot
            except Exception as e:
                logger.error("Approved action %s failed: %s", action_id, e)
                action.status = "failed"
                action.detail["error"] = str(e)
                return action

        action.status = "executed"
        action.executed_at = time.time()

        self._log_to_notebook(
            f"approved_{action.decision_type}",
            f"User approved: {action.title}",
        )
        return action

    def dismiss(self, action_id: str) -> Optional[AutonomousAction]:
        """User dismisses a pending or notified action."""
        action = self._find_action(action_id)
        if not action:
            return None
        if action.status in ("pending", "executed"):
            action.status = "dismissed"
            self._log_to_notebook(
                f"dismissed_{action.decision_type}",
                f"User dismissed: {action.title}",
            )
        return action

    def undo(self, action_id: str) -> Optional[AutonomousAction]:
        """Undo a recently executed action (within 5 minute window)."""
        action = self._find_action(action_id)
        if not action:
            return None
        if not action.undoable:
            return None

        undo_fn = self._undo_handlers.get(action_id)
        if undo_fn:
            try:
                undo_fn()
            except Exception as e:
                logger.error("Undo for action %s failed: %s", action_id, e)
                action.detail["undo_error"] = str(e)
                return action

        action.status = "undone"
        self._undo_handlers.pop(action_id, None)

        self._log_to_notebook(
            f"undone_{action.decision_type}",
            f"User undid: {action.title}",
        )
        return action

    # ── Query interface ─────────────────────────────────────────────

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get actions that need user attention (pending + recently executed with undo)."""
        result = []
        for action in self._actions:
            if action.status == "pending":
                result.append(action.to_dict())
            elif action.status == "executed" and action.behavior == "notify":
                result.append(action.to_dict())
            elif action.undoable:
                result.append(action.to_dict())
        return result

    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent autonomous decisions and their outcomes."""
        recent = sorted(self._actions, key=lambda a: a.created_at, reverse=True)
        return [a.to_dict() for a in recent[:limit]]

    def get_config(self) -> Dict[str, Any]:
        """Get current autonomy configuration."""
        behaviors = {}
        for dt in DecisionType:
            effective = self.get_behavior(dt)
            behaviors[dt.value] = {
                "behavior": effective.value,
                "overridden": dt.value in self._overrides,
            }
        return {
            "trust_level": self._trust_level.value,
            "decisions": behaviors,
            "undo_window_seconds": UNDO_WINDOW_SECONDS,
            "overrides": dict(self._overrides),
        }

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update autonomy configuration from API request body."""
        if "trust_level" in config:
            try:
                self.trust_level = TrustLevel(config["trust_level"])
            except ValueError:
                pass

        if "overrides" in config and isinstance(config["overrides"], dict):
            for dt_str, behavior_str in config["overrides"].items():
                try:
                    self.set_override(dt_str, behavior_str)
                except (ValueError, KeyError):
                    pass

        if "clear_overrides" in config and isinstance(config["clear_overrides"], list):
            for dt_str in config["clear_overrides"]:
                self.clear_override(dt_str)

        return self.get_config()

    # ── Internal helpers ────────────────────────────────────────────

    def _store_action(self, action: AutonomousAction) -> None:
        self._actions.append(action)
        # Trim old actions
        if len(self._actions) > self._max_stored_actions:
            self._actions = self._actions[-self._max_stored_actions:]

    def _find_action(self, action_id: str) -> Optional[AutonomousAction]:
        for action in self._actions:
            if action.action_id == action_id:
                return action
        return None

    def _log_to_notebook(self, event_type: str, description: str, detail: Optional[Dict] = None) -> None:
        if not self._notebook:
            return
        try:
            evidence = json.dumps(detail) if detail else None
            self._notebook.log_learning_event(
                event_type=event_type,
                description=description,
                evidence=evidence,
            )
        except Exception as e:
            logger.debug("Failed to log autonomy event to notebook: %s", e)
