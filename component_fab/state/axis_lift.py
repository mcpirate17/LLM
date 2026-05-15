"""Beta-Binomial shrunken pass-rate per (axis, value) across the fab ledger.

Pure read-only analyzer. Replays ``ledger.jsonl`` and emits, for every
axis tracked in proposal metadata, a shrunken posterior pass-rate per
value plus a multiplicative lift over the global promotion rate.

The proposer can sample knob values weighted by these lifts to
concentrate exploration on axes the data says lift. New / unobserved
values appear in the dict with `n=0` and shrink toward the global mean
(so the sampler never excludes a value outright on zero evidence).

Axes mined per entry:
  - ``math_knob``      from ``metadata.math_knobs`` (list[str])
  - ``math_knob_pair`` ordered pair from the same list (len 2+)
  - ``synthesis_kind`` from the grade record top-level
  - ``category``       from the grade record top-level
  - ``anchor_op``      parsed from ``metadata.anchor_witness`` when present

Outcome label = ``promotion_status == promoted`` after replaying the
full ledger. Falls back to ``learned_signal`` if the entry never
received a promotion record (still in-flight).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = _REPO / "component_fab" / "catalog" / "ledger.jsonl"
DEFAULT_OUTPUT_PATH = _REPO / "component_fab" / "catalog" / "axis_lift.json"

PROMOTED = "promoted"


@dataclass(slots=True)
class ValueStats:
    value: str
    n: int = 0
    k_promoted: int = 0
    k_learned: int = 0
    pass_rate_raw: float = 0.0
    pass_rate_shrunk: float = 0.0
    lift: float = 1.0


@dataclass(slots=True)
class AxisLiftReport:
    global_promoted: int
    global_total: int
    global_pass_rate: float
    prior_strength: float
    min_n: int
    by_axis: dict[str, list[ValueStats]] = field(default_factory=dict)

    def weights_for(self, axis: str) -> dict[str, float]:
        """Sampling weights normalized to sum=1 over the axis's values."""
        stats = self.by_axis.get(axis, [])
        if not stats:
            return {}
        total = sum(max(s.lift, 1e-6) for s in stats)
        if total <= 0:
            return {s.value: 1.0 / len(stats) for s in stats}
        return {s.value: max(s.lift, 1e-6) / total for s in stats}


def _read_grades_and_promotions(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Replay the ledger; return (last_grade_per_id, last_status_per_id)."""
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


def _emit_axes(grade: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield (axis_name, value) pairs for the grade record."""
    meta = grade.get("metadata") or {}
    knobs = list(meta.get("math_knobs") or [])
    for knob in knobs:
        if isinstance(knob, str) and knob:
            yield ("math_knob", knob)
    for combo in combinations(sorted(set(k for k in knobs if isinstance(k, str))), 2):
        yield ("math_knob_pair", "+".join(combo))
    sk = grade.get("synthesis_kind")
    if isinstance(sk, str) and sk:
        yield ("synthesis_kind", sk)
    cat = grade.get("category")
    if isinstance(cat, str) and cat:
        yield ("category", cat)
    anchor = meta.get("anchor_witness")
    if isinstance(anchor, str) and anchor:
        yield ("anchor_op", anchor)


def compute_axis_lift(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    *,
    prior_strength: float = 5.0,
    min_n: int = 2,
) -> AxisLiftReport:
    """Replay the ledger and return shrunken pass-rates per (axis, value)."""
    path = Path(ledger_path)
    last_grade, last_status = _read_grades_and_promotions(path)

    total = 0
    promoted = 0
    raw_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "k_prom": 0, "k_learn": 0})
    )

    for pid, grade in last_grade.items():
        total += 1
        is_promoted = last_status.get(pid) == PROMOTED
        is_learned = bool(grade.get("learned_signal"))
        if is_promoted:
            promoted += 1
        seen_axes: set[tuple[str, str]] = set()
        for axis, value in _emit_axes(grade):
            if (axis, value) in seen_axes:
                continue
            seen_axes.add((axis, value))
            bucket = raw_counts[axis][value]
            bucket["n"] += 1
            if is_promoted:
                bucket["k_prom"] += 1
            if is_learned:
                bucket["k_learn"] += 1

    global_p = (promoted / total) if total else 0.0

    by_axis: dict[str, list[ValueStats]] = {}
    for axis, value_map in raw_counts.items():
        rows: list[ValueStats] = []
        for value, c in value_map.items():
            n = c["n"]
            k = c["k_prom"]
            raw = (k / n) if n else 0.0
            shrunk = (k + global_p * prior_strength) / (n + prior_strength)
            denom = global_p if global_p > 0 else 1e-6
            lift = shrunk / denom
            rows.append(
                ValueStats(
                    value=value,
                    n=n,
                    k_promoted=k,
                    k_learned=c["k_learn"],
                    pass_rate_raw=raw,
                    pass_rate_shrunk=shrunk,
                    lift=lift,
                )
            )
        rows.sort(key=lambda r: (-r.lift, -r.n, r.value))
        by_axis[axis] = [r for r in rows if r.n >= min_n] or rows[:0]

    return AxisLiftReport(
        global_promoted=promoted,
        global_total=total,
        global_pass_rate=global_p,
        prior_strength=prior_strength,
        min_n=min_n,
        by_axis=by_axis,
    )


def write_axis_lift(
    report: AxisLiftReport,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_promoted": report.global_promoted,
        "global_total": report.global_total,
        "global_pass_rate": report.global_pass_rate,
        "prior_strength": report.prior_strength,
        "min_n": report.min_n,
        "by_axis": {
            axis: [asdict(row) for row in rows] for axis, rows in report.by_axis.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_axis_lift(path: Path | str = DEFAULT_OUTPUT_PATH) -> AxisLiftReport | None:
    """Best-effort loader for the proposer side. Returns ``None`` on absent / corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    by_axis: dict[str, list[ValueStats]] = {}
    for axis, rows in (data.get("by_axis") or {}).items():
        by_axis[axis] = [
            ValueStats(
                value=str(r.get("value") or ""),
                n=int(r.get("n") or 0),
                k_promoted=int(r.get("k_promoted") or 0),
                k_learned=int(r.get("k_learned") or 0),
                pass_rate_raw=float(r.get("pass_rate_raw") or 0.0),
                pass_rate_shrunk=float(r.get("pass_rate_shrunk") or 0.0),
                lift=float(r.get("lift") or 1.0),
            )
            for r in rows
        ]
    return AxisLiftReport(
        global_promoted=int(data.get("global_promoted") or 0),
        global_total=int(data.get("global_total") or 0),
        global_pass_rate=float(data.get("global_pass_rate") or 0.0),
        prior_strength=float(data.get("prior_strength") or 5.0),
        min_n=int(data.get("min_n") or 2),
        by_axis=by_axis,
    )
