from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .projectors.lifecycle_projector import ProjectorStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectorWorkerHealth:
    running: bool
    iterations: int
    last_run_at: Optional[float]
    last_error: Optional[str]
    degraded: bool
    last_applied_count: int


class ProjectorWorker:
    """Runs a projector loop in a background thread."""

    def __init__(
        self,
        replay_once: Callable[[], ProjectorStatus],
        *,
        interval_seconds: float = 1.0,
        name: str = "runtime-event-projector",
    ) -> None:
        self._replay_once = replay_once
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._name = name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._iterations = 0
        self._last_run_at: Optional[float] = None
        self._last_error: Optional[str] = None
        self._degraded = False
        self._last_applied_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run_once(self) -> ProjectorStatus:
        status = self._replay_once()
        self._iterations += 1
        self._last_run_at = time.time()
        self._degraded = bool(status.degraded)
        self._last_applied_count = int(status.applied_count)
        if not status.degraded:
            self._last_error = None
        return status

    def health_snapshot(self) -> ProjectorWorkerHealth:
        return ProjectorWorkerHealth(
            running=self._thread is not None and self._thread.is_alive(),
            iterations=self._iterations,
            last_run_at=self._last_run_at,
            last_error=self._last_error,
            degraded=self._degraded,
            last_applied_count=self._last_applied_count,
        )

    def _run(self) -> None:
        logger.info(
            "Projector worker started: name=%s interval=%.2fs",
            self._name,
            self._interval_seconds,
        )
        was_degraded = False
        while not self._stop.is_set():
            try:
                status = self.run_once()
                if status.degraded and not was_degraded:
                    logger.warning(
                        "Projector entered degraded mode: name=%s", self._name
                    )
                    self._last_error = "projector_degraded"
                elif was_degraded and not status.degraded:
                    logger.info(
                        "Projector recovered from degraded mode: name=%s", self._name
                    )
                was_degraded = status.degraded
                if status.applied_count > 0:
                    logger.debug(
                        "Projector applied %d event(s): name=%s last_event=%s",
                        status.applied_count,
                        self._name,
                        status.last_event_id,
                    )
            except Exception as exc:
                self._iterations += 1
                self._last_run_at = time.time()
                self._degraded = True
                self._last_error = f"{type(exc).__name__}: {exc}"
                if not was_degraded:
                    logger.error(
                        "Projector worker error: name=%s %s: %s",
                        self._name,
                        type(exc).__name__,
                        exc,
                    )
                was_degraded = True
            self._stop.wait(self._interval_seconds)
        logger.info(
            "Projector worker stopped: name=%s iterations=%d",
            self._name,
            self._iterations,
        )
