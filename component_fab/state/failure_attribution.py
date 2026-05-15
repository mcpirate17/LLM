"""Gate-level failure attribution for the fab ledger.

Pure read-only analyzer. Walks ``ledger.jsonl`` and answers:

- Which gate killed each proposal? (first-killer, from ``metadata.eliminated_by``).
- What's the kill rate per gate, computed against the population that
  reached it under the canonical gate order?
- Which gates are over-eager (kill > ``over_eager_threshold`` of what they see)?
- Which rejected candidates are "anchor pool" material — eliminated by
  late gates but still scoring high on composite or upstream metrics?

The canonical gate order matches ``validator/capability.py``:
    smoke -> s05_causality_stability -> erf_density -> nano_bind -> ar_*

Output feeds two future loops:
- proposer: anchor pool (rejected-but-promising) re-enters as seed.
- gate calibration: over-eager gates flagged for threshold review.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = _REPO / "component_fab" / "catalog" / "ledger.jsonl"
DEFAULT_OUTPUT_PATH = _REPO / "component_fab" / "catalog" / "failure_attribution.json"

CANONICAL_GATE_ORDER: tuple[str, ...] = (
    "smoke",
    "s05_causality_stability",
    "erf_density",
    "nano_bind",
    "ar_easy",
    "ar_medium",
    "ar_hard",
)
SURVIVED = "survived"


@dataclass(slots=True)
class GateStats:
    gate: str
    killed: int
    reached: int
    kill_rate: float
    over_eager: bool
    # ERF-only: how many kills were at the per-position structural floor.
    # When this fraction dominates, the gate is correct and the generator
    # is producing per-position-only architectures — flip the diagnosis.
    killed_at_floor: int = 0
    generator_floor_bunched: bool = False


@dataclass(slots=True)
class AnchorCandidate:
    proposal_id: str
    name: str
    eliminated_by: str
    composite_score: float
    erf_density: float | None
    nb_max_accuracy: float | None
    math_knobs: tuple[str, ...]
    cycle: int


@dataclass(slots=True)
class FailureReport:
    total_graded: int
    total_promoted: int
    total_rejected: int
    total_pending: int
    gate_stats: list[GateStats] = field(default_factory=list)
    over_eager_gates: list[str] = field(default_factory=list)
    anchor_pool: list[AnchorCandidate] = field(default_factory=list)


def _read_ledger(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    last_grade: dict[str, dict[str, Any]] = {}
    last_status: dict[str, str] = {}
    if not path.exists():
        return last_grade, last_status
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
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


def _gate_from_record(grade: dict[str, Any]) -> str:
    meta = grade.get("metadata") or {}
    e = meta.get("eliminated_by")
    if isinstance(e, str) and e:
        return e
    if grade.get("smoke_pass") is False:
        return "smoke"
    return SURVIVED


def _gate_index(gate: str, order: tuple[str, ...]) -> int:
    """Return position in canonical order; unknown gates go after the last gate."""
    try:
        return order.index(gate)
    except ValueError:
        return len(order)


def _count_outcomes(
    last_grade: dict[str, dict[str, Any]],
    last_status: dict[str, str],
) -> tuple[int, int, int, int, dict[str, int], list[float]]:
    """Return (total, promoted, rejected, pending, killed_by, erf_killed_values)."""
    killed_by: dict[str, int] = defaultdict(int)
    erf_killed_values: list[float] = []
    total = promoted = rejected = pending = 0
    for pid, grade in last_grade.items():
        total += 1
        status = last_status.get(pid, "pending")
        if status == "promoted":
            promoted += 1
        elif status == "rejected":
            rejected += 1
        else:
            pending += 1
        gate = _gate_from_record(grade)
        killed_by[gate] += 1
        if gate == "erf_density":
            erf = (grade.get("metadata") or {}).get("erf_density")
            if isinstance(erf, (int, float)):
                erf_killed_values.append(float(erf))
    return total, promoted, rejected, pending, dict(killed_by), erf_killed_values


def _build_gate_stats(
    total: int,
    killed_by: dict[str, int],
    gate_order: tuple[str, ...],
    over_eager_threshold: float,
    min_n_for_over_eager: int,
    erf_killed_values: list[float] | None = None,
    erf_floor_tolerance: float = 0.001,
    erf_seq_len: int = 32,
    erf_floor_bunch_fraction: float = 0.5,
) -> list[GateStats]:
    """Compute per-gate kill rate against the canonical-order reached count.

    For the ERF gate specifically: if a majority of kills sit at the
    per-position structural floor (~1/seq_len), the gate is doing its job
    and the upstream generator is producing per-position-only modules —
    do not flag ``over_eager``.
    """
    erf_floor = 1.0 / max(1, erf_seq_len)
    floor_kills = 0
    if erf_killed_values:
        floor_kills = sum(
            1 for v in erf_killed_values if abs(v - erf_floor) <= erf_floor_tolerance
        )
    stats: list[GateStats] = []
    for gate in gate_order:
        killed = killed_by.get(gate, 0)
        gate_pos = _gate_index(gate, gate_order)
        killed_earlier = sum(
            count
            for other_gate, count in killed_by.items()
            if other_gate not in (SURVIVED, gate)
            and _gate_index(other_gate, gate_order) < gate_pos
        )
        reached = max(0, total - killed_earlier)
        rate = (killed / reached) if reached else 0.0
        over_eager = (rate >= over_eager_threshold) and (
            reached >= min_n_for_over_eager
        )
        killed_at_floor = 0
        generator_floor_bunched = False
        if gate == "erf_density":
            killed_at_floor = floor_kills
            if killed and (floor_kills / killed) >= erf_floor_bunch_fraction:
                generator_floor_bunched = True
                # Gate-correct case: don't flag the gate as over-eager.
                over_eager = False
        stats.append(
            GateStats(
                gate=gate,
                killed=killed,
                reached=reached,
                kill_rate=rate,
                over_eager=over_eager,
                killed_at_floor=killed_at_floor,
                generator_floor_bunched=generator_floor_bunched,
            )
        )
    return stats


def _candidate_from_grade(
    pid: str,
    grade: dict[str, Any],
    *,
    anchor_min_composite: float,
    anchor_min_erf: float,
) -> AnchorCandidate | None:
    meta = grade.get("metadata") or {}
    e = meta.get("eliminated_by")
    if not isinstance(e, str) or not e:
        return None
    composite = float(grade.get("composite_score") or 0.0)
    erf = meta.get("erf_density")
    erf_f = float(erf) if isinstance(erf, (int, float)) else None
    if composite < anchor_min_composite and (erf_f or 0.0) < anchor_min_erf:
        return None
    nb = meta.get("nb_max_accuracy")
    return AnchorCandidate(
        proposal_id=pid,
        name=str(grade.get("name") or ""),
        eliminated_by=e,
        composite_score=composite,
        erf_density=erf_f,
        nb_max_accuracy=float(nb) if isinstance(nb, (int, float)) else None,
        math_knobs=tuple(str(k) for k in (meta.get("math_knobs") or [])),
        cycle=int(grade.get("cycle") or 0),
    )


def _build_anchor_pool(
    last_grade: dict[str, dict[str, Any]],
    last_status: dict[str, str],
    *,
    anchor_min_composite: float,
    anchor_min_erf: float,
    anchor_pool_size: int,
) -> list[AnchorCandidate]:
    """Rejected-but-promising proposals, ranked by composite then ERF then NB."""
    candidates: list[AnchorCandidate] = []
    for pid, grade in last_grade.items():
        if last_status.get(pid, "pending") != "rejected":
            continue
        cand = _candidate_from_grade(
            pid,
            grade,
            anchor_min_composite=anchor_min_composite,
            anchor_min_erf=anchor_min_erf,
        )
        if cand is not None:
            candidates.append(cand)
    candidates.sort(
        key=lambda c: (
            -c.composite_score,
            -(c.erf_density or 0.0),
            -(c.nb_max_accuracy or 0.0),
            c.proposal_id,
        )
    )
    return candidates[:anchor_pool_size]


def compute_failure_attribution(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    *,
    over_eager_threshold: float = 0.85,
    min_n_for_over_eager: int = 20,
    anchor_min_composite: float = 0.4,
    anchor_min_erf: float = 0.10,
    anchor_pool_size: int = 25,
    gate_order: tuple[str, ...] = CANONICAL_GATE_ORDER,
) -> FailureReport:
    last_grade, last_status = _read_ledger(Path(ledger_path))
    total, promoted, rejected, pending, killed_by, erf_killed_values = _count_outcomes(
        last_grade, last_status
    )
    gate_stats = _build_gate_stats(
        total,
        killed_by,
        gate_order,
        over_eager_threshold,
        min_n_for_over_eager,
        erf_killed_values=erf_killed_values,
    )
    anchor_pool = _build_anchor_pool(
        last_grade,
        last_status,
        anchor_min_composite=anchor_min_composite,
        anchor_min_erf=anchor_min_erf,
        anchor_pool_size=anchor_pool_size,
    )
    return FailureReport(
        total_graded=total,
        total_promoted=promoted,
        total_rejected=rejected,
        total_pending=pending,
        gate_stats=gate_stats,
        over_eager_gates=[g.gate for g in gate_stats if g.over_eager],
        anchor_pool=anchor_pool,
    )


def write_failure_attribution(
    report: FailureReport,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_graded": report.total_graded,
        "total_promoted": report.total_promoted,
        "total_rejected": report.total_rejected,
        "total_pending": report.total_pending,
        "over_eager_gates": list(report.over_eager_gates),
        "gate_stats": [asdict(g) for g in report.gate_stats],
        "anchor_pool": [asdict(c) for c in report.anchor_pool],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_failure_attribution(
    path: Path | str = DEFAULT_OUTPUT_PATH,
) -> FailureReport | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate_stats = [
        GateStats(
            gate=str(g.get("gate") or ""),
            killed=int(g.get("killed") or 0),
            reached=int(g.get("reached") or 0),
            kill_rate=float(g.get("kill_rate") or 0.0),
            over_eager=bool(g.get("over_eager")),
            killed_at_floor=int(g.get("killed_at_floor") or 0),
            generator_floor_bunched=bool(g.get("generator_floor_bunched")),
        )
        for g in (data.get("gate_stats") or [])
    ]
    anchor_pool = [
        AnchorCandidate(
            proposal_id=str(c.get("proposal_id") or ""),
            name=str(c.get("name") or ""),
            eliminated_by=str(c.get("eliminated_by") or ""),
            composite_score=float(c.get("composite_score") or 0.0),
            erf_density=(
                float(c["erf_density"]) if c.get("erf_density") is not None else None
            ),
            nb_max_accuracy=(
                float(c["nb_max_accuracy"])
                if c.get("nb_max_accuracy") is not None
                else None
            ),
            math_knobs=tuple(str(k) for k in (c.get("math_knobs") or [])),
            cycle=int(c.get("cycle") or 0),
        )
        for c in (data.get("anchor_pool") or [])
    ]
    return FailureReport(
        total_graded=int(data.get("total_graded") or 0),
        total_promoted=int(data.get("total_promoted") or 0),
        total_rejected=int(data.get("total_rejected") or 0),
        total_pending=int(data.get("total_pending") or 0),
        gate_stats=gate_stats,
        over_eager_gates=list(data.get("over_eager_gates") or []),
        anchor_pool=anchor_pool,
    )
