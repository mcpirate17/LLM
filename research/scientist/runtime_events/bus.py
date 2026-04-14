from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from .schema import RuntimeEvent
from .spool import NdjsonEventSpool, SpoolOffset

logger = logging.getLogger(__name__)

Subscriber = Callable[[RuntimeEvent], None]


@dataclass(frozen=True)
class PublishResult:
    event: RuntimeEvent
    spool_offset: Optional[SpoolOffset]
    subscriber_count: int
    subscriber_failures: int


@dataclass(frozen=True)
class SubscriberFailure:
    subscriber_name: str
    event_type: str
    error_type: str
    message: str
    failed_at: float


@dataclass(frozen=True)
class BusHealthSnapshot:
    publish_count: int
    last_event_id: Optional[str]
    last_event_type: Optional[str]
    last_publish_at: Optional[float]
    last_spool_error: Optional[str]
    subscriber_failure_count: int
    last_subscriber_failure: Optional[SubscriberFailure]


@dataclass
class _Subscription:
    subscriber: Subscriber
    event_types: Optional[frozenset[str]]

    @property
    def name(self) -> str:
        return getattr(self.subscriber, "__name__", self.subscriber.__class__.__name__)

    def matches(self, event_type: str) -> bool:
        return self.event_types is None or event_type in self.event_types


class RuntimeEventBus:
    """In-process event bus with optional durable spool append before fan-out."""

    def __init__(self, *, spool: NdjsonEventSpool | None = None) -> None:
        self._spool = spool
        self._subscribers: List[_Subscription] = []
        self._publish_count = 0
        self._last_event_id: Optional[str] = None
        self._last_event_type: Optional[str] = None
        self._last_publish_at: Optional[float] = None
        self._last_spool_error: Optional[str] = None
        self._subscriber_failures: List[SubscriberFailure] = []
        self._max_subscriber_failures: int = 1000

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(_Subscription(subscriber=subscriber, event_types=None))

    def subscribe_to(
        self, event_types: str | Iterable[str], subscriber: Subscriber
    ) -> None:
        if isinstance(event_types, str):
            event_types = [event_types]
        self._subscribers.append(
            _Subscription(
                subscriber=subscriber,
                event_types=frozenset(str(event_type) for event_type in event_types),
            )
        )

    def publish(self, event: RuntimeEvent) -> PublishResult:
        offset = None
        if self._spool is not None:
            try:
                offset = self._spool.append(event)
                self._last_spool_error = None
            except OSError as exc:
                self._last_spool_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Spool append failed: event_type=%s run_id=%s error=%s",
                    event.event_type,
                    event.run_id,
                    exc,
                )
                raise

        if event.is_lifecycle():
            logger.info(
                "Published lifecycle event: type=%s run_id=%s producer=%s durability=%s event_id=%s",
                event.event_type,
                event.run_id,
                event.producer,
                event.durability,
                event.event_id[:12],
            )
        else:
            logger.debug(
                "Published event: type=%s run_id=%s producer=%s",
                event.event_type,
                event.run_id,
                event.producer,
            )

        matched_subscribers = [
            subscription
            for subscription in list(self._subscribers)
            if subscription.matches(event.event_type)
        ]
        subscriber_failures = 0
        for subscription in matched_subscribers:
            try:
                subscription.subscriber(event)
            except Exception as exc:
                subscriber_failures += 1
                self._subscriber_failures.append(
                    SubscriberFailure(
                        subscriber_name=subscription.name,
                        event_type=event.event_type,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        failed_at=time.time(),
                    )
                )
                if len(self._subscriber_failures) > self._max_subscriber_failures:
                    self._subscriber_failures = self._subscriber_failures[
                        -self._max_subscriber_failures :
                    ]
                logger.warning(
                    "Subscriber %s failed on %s: %s: %s",
                    subscription.name,
                    event.event_type,
                    type(exc).__name__,
                    exc,
                )
        self._publish_count += 1
        self._last_event_id = event.event_id
        self._last_event_type = event.event_type
        self._last_publish_at = time.time()
        return PublishResult(
            event=event,
            spool_offset=offset,
            subscriber_count=len(matched_subscribers),
            subscriber_failures=subscriber_failures,
        )

    def health_snapshot(self) -> BusHealthSnapshot:
        return BusHealthSnapshot(
            publish_count=self._publish_count,
            last_event_id=self._last_event_id,
            last_event_type=self._last_event_type,
            last_publish_at=self._last_publish_at,
            last_spool_error=self._last_spool_error,
            subscriber_failure_count=len(self._subscriber_failures),
            last_subscriber_failure=(
                self._subscriber_failures[-1] if self._subscriber_failures else None
            ),
        )
