"""Append-only component_fab ledger with replayable in-memory rollup."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO

from component_fab.state.schema_versions import (
    LEDGER_GRADE_SCHEMA_VERSION,
    LEDGER_PROMOTION_SCHEMA_VERSION,
    with_schema_version,
)

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = _REPO / "component_fab" / "catalog" / "ledger.jsonl"


def iter_jsonl_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL file, skipping blank/corrupt lines."""
    path = Path(path)
    if not path.exists():
        return
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(record, dict):
                yield record
    if skipped:
        logger.debug("skipped %d corrupt lines in %s", skipped, path)


def iter_rotated_jsonl_paths(
    path: Path | str, *, include_active: bool = True
) -> Iterator[Path]:
    """Yield integer-suffix JSONL rotations oldest-first, then active file."""
    base = Path(path)
    rotations: list[tuple[int, int, Path]] = []
    prefix = base.name + "."
    if base.parent.exists():
        for child in base.parent.iterdir():
            if not child.name.startswith(prefix):
                continue
            suffix = child.name[len(prefix) :]
            try:
                rotations.append((child.stat().st_mtime_ns, int(suffix), child))
            except ValueError:
                continue
    for _, _, child in sorted(rotations):
        yield child
    if include_active:
        yield base


def latest_by_key(
    records: Iterable[dict[str, Any]], key_field: str
) -> dict[str, dict[str, Any]]:
    """Last record per ``str(record[key_field])``; missing keys are skipped."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record.get(key_field)
        if key:
            latest[str(key)] = record
    return latest


def write_json_report(
    payload: dict[str, Any] | list[Any],
    output_path: Path | str,
    *,
    default: Callable[[Any], Any] | None = None,
) -> Path:
    """Shared analyzer-report writer: mkdir -> stable JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default),
        encoding="utf-8",
    )
    return out


def read_last_grades_and_statuses(
    path: Path | str = DEFAULT_LEDGER_PATH,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Replay a ledger JSONL; return last grade and promotion status per id."""
    last_grade: dict[str, dict[str, Any]] = {}
    last_status: dict[str, str] = {}
    for record in iter_jsonl_records(path):
        pid = record.get("proposal_id")
        if not pid:
            continue
        event = record.get("event")
        if event == "grade":
            last_grade[str(pid)] = record
        elif event == "promote":
            status = str(record.get("status") or "")
            if status:
                last_status[str(pid)] = status
    return last_grade, last_status


PROMOTION_PENDING = "pending"
PROMOTION_PROMOTED = "promoted"
PROMOTION_REJECTED = "rejected"
_VALID_STATUSES = frozenset({PROMOTION_PENDING, PROMOTION_PROMOTED, PROMOTION_REJECTED})


class JsonlWriter:
    """Append-mode JSONL writer with one open file handle per instance."""

    __slots__ = ("path", "buffering", "_handle")

    def __init__(self, path: Path | str, *, buffering: int = 1 << 20) -> None:
        self.path = Path(path)
        self.buffering = buffering
        self._handle: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open(
                "a", encoding="utf-8", buffering=self.buffering
            )

    def write(self, record: Any) -> None:
        if self._handle is None:
            self._ensure_open()
        assert self._handle is not None
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def _prune_rotations(base_path: Path, keep: int = 3) -> int:
    """Remove old integer-suffix rotations, keeping newest first."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    prefix = base_path.name + "."
    rotations = [
        child
        for child in base_path.parent.glob(f"{base_path.name}.*")
        if child.name[len(prefix) :].isdigit()
    ]
    rotations.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    removed = 0
    for stale in rotations[keep:]:
        stale.unlink(missing_ok=True)
        removed += 1
    return removed


@dataclass(slots=True)
class LedgerEntry:
    proposal_id: str
    name: str
    category: str
    synthesis_kind: str
    composite_history: list[float] = field(default_factory=list)
    cycles_seen: list[int] = field(default_factory=list)
    metadata_history: list[dict[str, Any]] = field(default_factory=list)
    smoke_pass_count: int = 0
    learned_signal_count: int = 0
    promotion_status: str = PROMOTION_PENDING
    first_seen_iso: str = ""
    last_seen_iso: str = ""

    def best_composite(self) -> float:
        return max(self.composite_history, default=0.0)

    def mean_composite(self, window: int | None = None) -> float:
        if window is not None and window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        history = self.composite_history if window is None else self.composite_history[-window:]
        return sum(history) / len(history) if history else 0.0


class Ledger:
    """JSONL-backed proposal ledger with an in-memory rollup."""

    def __init__(
        self,
        path: Path | str = DEFAULT_LEDGER_PATH,
        *,
        include_rotated: bool = False,
    ) -> None:
        self.path = Path(path)
        self.entries: dict[str, LedgerEntry] = {}
        self._writer: JsonlWriter | None = None
        if include_rotated:
            self._replay_rotated()
        if self.path.exists():
            self._replay(self.path)

    def _replay(self, path: Path) -> None:
        for record in iter_jsonl_records(path):
            self._apply_record(record)

    def _replay_rotated(self) -> None:
        for child in iter_rotated_jsonl_paths(self.path, include_active=False):
            self._replay(child)

    def _apply_record(self, record: dict[str, Any]) -> None:
        if not record.get("proposal_id"):
            return
        event = record.get("event")
        if event == "grade":
            self._apply_grade(record)
        elif event == "promote":
            self._apply_promotion(record)
        elif event == "deep_probe":
            self._apply_deep_probe(record)

    def _apply_deep_probe(self, record: dict[str, Any]) -> None:
        proposal_id = str(record["proposal_id"])
        entry = self.entries.get(proposal_id)
        if entry is None:
            return
        # Fold the outcome into the metadata history so the surrogate can see it.
        meta = {
            "deep_probe_beats_frontier": bool(record.get("beats_frontier")),
            "deep_probe_mean_delta": float(record.get("mean_delta") or 0.0),
            **(record.get("metadata") or {}),
        }
        entry.metadata_history.append(meta)

    def _apply_grade(self, record: dict[str, Any]) -> None:
        proposal_id = str(record["proposal_id"])
        entry = self.entries.get(proposal_id)
        if entry is None:
            entry = LedgerEntry(
                proposal_id=proposal_id,
                name=str(record.get("name") or ""),
                category=str(record.get("category") or ""),
                synthesis_kind=str(record.get("synthesis_kind") or ""),
                first_seen_iso=str(record.get("timestamp") or ""),
            )
            self.entries[proposal_id] = entry
        entry.composite_history.append(float(record.get("composite_score") or 0.0))
        entry.cycles_seen.append(int(record.get("cycle") or 0))
        entry.metadata_history.append(dict(record.get("metadata") or {}))
        entry.last_seen_iso = str(record.get("timestamp") or entry.last_seen_iso)
        if record.get("smoke_pass"):
            entry.smoke_pass_count += 1
        if record.get("learned_signal"):
            entry.learned_signal_count += 1

    def _apply_promotion(self, record: dict[str, Any]) -> None:
        proposal_id = str(record["proposal_id"])
        entry = self.entries.get(proposal_id)
        if entry is None:
            return
        status = str(record.get("status") or PROMOTION_PENDING)
        if status in _VALID_STATUSES:
            entry.promotion_status = status

    def has_seen(self, proposal_id: str) -> bool:
        return proposal_id in self.entries

    def record_grade(
        self,
        proposal_id: str,
        *,
        name: str,
        category: str,
        synthesis_kind: str,
        cycle: int,
        composite_score: float,
        smoke_pass: bool,
        learned_signal: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = with_schema_version(
            {
                "event": "grade",
                "proposal_id": proposal_id,
                "name": name,
                "category": category,
                "synthesis_kind": synthesis_kind,
                "cycle": cycle,
                "composite_score": composite_score,
                "smoke_pass": smoke_pass,
                "learned_signal": learned_signal,
                "metadata": dict(metadata or {}),
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            LEDGER_GRADE_SCHEMA_VERSION,
        )
        self._apply_record(record)
        self._append(record)

    def record_promotion(self, proposal_id: str, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"unknown promotion status: {status}")
        record = with_schema_version(
            {
                "event": "promote",
                "proposal_id": proposal_id,
                "status": status,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            LEDGER_PROMOTION_SCHEMA_VERSION,
        )
        self._apply_record(record)
        self._append(record)

    def record_deep_probe(
        self,
        proposal_id: str,
        *,
        beats_frontier: bool,
        mean_delta: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the high-fidelity outcome of a deep frontier bake-off."""
        record = {
            "event": "deep_probe",
            "proposal_id": proposal_id,
            "beats_frontier": beats_frontier,
            "mean_delta": float(mean_delta),
            "metadata": dict(metadata or {}),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        # deep_probe doesn't change the basic LedgerEntry history (which is
        # nano-cycle focused), but it is replayed into the entries' metadata
        # by _apply_record.
        self._apply_record(record)
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = JsonlWriter(self.path)
        self._writer.write(record)

    def rotate_if_oversized(self, max_bytes: int = 1_048_576) -> Path | None:
        if not self.path.exists() or self.path.stat().st_size < max_bytes:
            return None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        prefix = self.path.name + "."
        existing = [
            int(child.name[len(prefix) :])
            for child in self.path.parent.glob(f"{self.path.name}.*")
            if child.name[len(prefix) :].isdigit()
        ]
        candidate = self.path.with_suffix(
            self.path.suffix + f".{max(existing, default=0) + 1}"
        )
        self.path.rename(candidate)
        self.path.touch()
        _prune_rotations(self.path)
        return candidate

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def all_entries(self) -> Iterable[LedgerEntry]:
        return self.entries.values()

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self.entries.values()]


def resolve_proposal_id(ledger: Ledger, needle: str) -> LedgerEntry:
    """Exact-then-unique-prefix proposal-id lookup over the ledger rollup."""
    entry = ledger.entries.get(needle)
    if entry is not None:
        return entry
    matches = [pid for pid in ledger.entries if pid.startswith(needle)]
    if len(matches) == 1:
        return ledger.entries[matches[0]]
    if matches:
        raise ValueError(
            f"proposal id prefix {needle!r} is ambiguous ({len(matches)} matches)"
        )
    raise ValueError(f"proposal id {needle!r} not found in ledger")
