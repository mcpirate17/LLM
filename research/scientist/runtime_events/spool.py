from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .schema import RuntimeEvent, RuntimeEventDurability

logger = logging.getLogger(__name__)

_SEGMENT_PREFIX = "segment-"
_SEGMENT_SUFFIX = ".ndjson"


@dataclass(frozen=True)
class SpoolOffset:
    segment: str
    line_number: int


@dataclass(frozen=True)
class SpoolRecord:
    offset: SpoolOffset
    event: RuntimeEvent


class NdjsonEventSpool:
    """Append-only local event spool with simple segment handling."""

    def __init__(
        self, root: str | Path, *, active_segment: Optional[str] = None
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_segment = active_segment or self._discover_or_create_segment_name()
        self._append_line_count: int = self._line_count(self.active_path)
        self._write_lock = threading.Lock()
        logger.info(
            "Spool initialized: root=%s segment=%s existing_lines=%d",
            self.root,
            self.active_segment,
            self._append_line_count,
        )

    @property
    def active_path(self) -> Path:
        return self.root / self.active_segment

    def append(self, event: RuntimeEvent) -> SpoolOffset:
        self.root.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._write_lock:
            with self.active_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                if event.durability == RuntimeEventDurability.CRITICAL:
                    os.fsync(handle.fileno())
            self._append_line_count += 1
            return SpoolOffset(
                segment=self.active_segment,
                line_number=self._append_line_count,
            )

    def replay(self, *, after: Optional[SpoolOffset] = None) -> Iterator[SpoolRecord]:
        started = after is None
        for path in self._segment_paths():
            line_number = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line_number += 1
                    if not line.strip():
                        continue
                    offset = SpoolOffset(segment=path.name, line_number=line_number)
                    if not started:
                        if (
                            path.name == after.segment
                            and line_number <= after.line_number
                        ):
                            continue
                        if path.name == after.segment or path.name > after.segment:
                            started = True
                        else:
                            continue
                    yield SpoolRecord(
                        offset=offset,
                        event=RuntimeEvent.from_dict(json.loads(line)),
                    )

    def _segment_paths(self) -> list[Path]:
        return sorted(self.root.glob(f"{_SEGMENT_PREFIX}*{_SEGMENT_SUFFIX}"))

    def _discover_or_create_segment_name(self) -> str:
        existing = self._segment_paths()
        if existing:
            return existing[-1].name
        return f"{_SEGMENT_PREFIX}000001{_SEGMENT_SUFFIX}"

    @staticmethod
    def _line_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
