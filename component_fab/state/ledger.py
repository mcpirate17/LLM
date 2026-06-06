"""Cross-cycle ledger — tracks every proposal seen across autonomous runs.

Backed by a single JSONL append log + an in-memory rollup. Each proposal
gets one ``LedgerEntry`` keyed by ``proposal_id`` with:
- score history across cycles
- promotion status (pending / promoted / rejected)
- timestamps + last-seen cycle

Append-only on disk so cycle history is auditable. The in-memory state
is rebuilt by replaying the log on startup.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = _REPO / "component_fab" / "catalog" / "ledger.jsonl"

PROMOTION_PENDING = "pending"
PROMOTION_PROMOTED = "promoted"
PROMOTION_REJECTED = "rejected"
_VALID_STATUSES = frozenset({PROMOTION_PENDING, PROMOTION_PROMOTED, PROMOTION_REJECTED})


class JsonlWriter:
    """Append-mode JSONL writer that keeps a single file handle open across writes.

    Replaces the ``open('a') -> write -> close()`` per-record pattern. For
    ~hundreds of writes/cycle (e.g. one per fab spec), the open+close cost
    dominates wall-clock; this class amortizes the open cost to once per
    ``JsonlWriter`` instance.

    Thread-safety: NOT safe for concurrent use. Use one writer per thread.
    The fab autonomous loop is single-threaded.

    Behavior change vs the prior open/close pattern: the file handle is held
    open across writes. ``flush()`` is called after every record so external
    readers (``tail -f``) and same-process ``Path.read_text()`` see writes
    immediately — but no ``fsync`` is forced, matching the prior durability
    (process crash mid-cycle can still lose the last buffered line).
    """

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
        """Encode ``record`` as one JSONL line and append it to the file.

        Flushes after every write so external readers see data immediately
        (the prior open/close pattern also flushed on close). With
        ``buffering=1<<20`` and typical small records, the flush is a no-op
        at the syscall level once the in-process buffer fills.
        """
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
    """Delete old ``base_path.N`` integer-suffix rotations, keeping newest first."""
    if keep < 0:
        raise ValueError("keep must be non-negative")

    rotations: list[Path] = []
    prefix = base_path.name + "."
    for child in base_path.parent.glob(f"{base_path.name}.*"):
        suffix = child.name[len(prefix) :]
        if suffix.isdigit():
            rotations.append(child)

    rotations.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    deleted = 0
    for stale in rotations[keep:]:
        stale.unlink(missing_ok=True)
        deleted += 1
    return deleted


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


class Ledger:
    """JSONL-backed proposal ledger with in-memory rollup."""

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
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._apply_record(record)

    def _replay_rotated(self) -> None:
        """Replay rotated audit-trail files in numeric order (.1, .2, ...).

        Rotations are an audit trail by default and not loaded into the
        in-memory rollup. The CLI tooling that wants the full historical
        view opts in via ``include_rotated=True``.
        """
        parent = self.path.parent
        prefix = self.path.name + "."
        rotated = []
        for child in parent.iterdir():
            if not child.name.startswith(prefix):
                continue
            suffix = child.name[len(prefix) :]
            try:
                rotated.append((int(suffix), child))
            except ValueError:
                continue
        rotated.sort()
        for _, child in rotated:
            self._replay(child)

    def _apply_record(self, record: dict[str, Any]) -> None:
        proposal_id = record.get("proposal_id")
        if not proposal_id:
            return
        if record.get("event") == "grade":
            self._apply_grade(record)
        elif record.get("event") == "promote":
            self._apply_promotion(record)

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
        record = {
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
        }
        self._apply_record(record)
        self._append(record)

    def record_promotion(self, proposal_id: str, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"unknown promotion status: {status}")
        record = {
            "event": "promote",
            "proposal_id": proposal_id,
            "status": status,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        self._apply_record(record)
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = JsonlWriter(self.path)
        self._writer.write(record)

    def rotate_if_oversized(self, max_bytes: int = 1_048_576) -> Path | None:
        """Rotate the active JSONL when it exceeds ``max_bytes``.

        Renames ``path`` to ``path.<N>`` (lowest unused integer) and starts
        a fresh active log. In-memory rollup is preserved; the rotated file
        remains on disk as the audit trail. Returns the rotated path, or
        ``None`` if no rotation was needed.
        """
        if not self.path.exists() or self.path.stat().st_size < max_bytes:
            return None
        # Drop the cached handle — the on-disk inode is about to be renamed
        # out from under it. A fresh writer will open on the next _append.
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        index = 1
        while True:
            candidate = self.path.with_suffix(self.path.suffix + f".{index}")
            if not candidate.exists():
                break
            index += 1
        self.path.rename(candidate)
        self.path.touch()
        _prune_rotations(self.path)
        return candidate

    def close(self) -> None:
        """Flush and release the JSONL file handle.

        Safe to call multiple times. The handle is also released when the
        process exits; this is the explicit-cleanup entry point.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def stale_proposals(
        self, max_age_cycles: int = 3, current_cycle: int = 0
    ) -> list[str]:
        """Return proposals not seen in the last ``max_age_cycles`` cycles."""
        out: list[str] = []
        for proposal_id, entry in self.entries.items():
            if not entry.cycles_seen:
                continue
            if current_cycle - max(entry.cycles_seen) > max_age_cycles:
                out.append(proposal_id)
        return out

    def candidates_with_pass_streak(
        self, min_cycles: int, min_score: float
    ) -> list[LedgerEntry]:
        """Entries whose last ``min_cycles`` composite scores all hit ``min_score``."""
        out: list[LedgerEntry] = []
        for entry in self.entries.values():
            if len(entry.composite_history) < min_cycles:
                continue
            recent = entry.composite_history[-min_cycles:]
            if all(score >= min_score for score in recent):
                out.append(entry)
        return out

    def all_entries(self) -> Iterable[LedgerEntry]:
        return self.entries.values()

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self.entries.values()]
