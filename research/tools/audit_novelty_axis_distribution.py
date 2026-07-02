"""Audit: does the geometric-novelty MAP-Elites axis's far bin conflate
meaningfully different distances-from-softmax?

Lane G3 (`research/notes/nm_verification_split_plan_2026-07-02.md`). The
axis (`research.synthesis.novelty_distance.novelty_behavior_axis`) is a coarse
3-bin design: ``<0.75`` inside the softmax basin, ``[0.75, 1.75)`` adjacent,
``>=1.75`` far. In the 8000-iter `deep_registry_discovery` run
(`research/reports/autonomous_run_2026-07-02/deep_registry_discovery.log`)
every one of the 30 elites landed in the far bin, so the axis currently does
zero selection work among far-from-softmax elites.

This script does NOT re-run discovery (CPU-only analysis, no GPU, no wiring
changes). It recomputes each elite's RAW standardized novelty distance by
reading the raw physics descriptors the run already printed
(`perm_equivariance`, `shift_equivariance`, `scale_homogeneity`,
`energy_gain`, `spectral_radius` — the leaderboard columns) straight back
through the same measured basin (`softmax_basin_signatures`) and the same
distance function (`geometric_novelty`) the run used, so the recomputed
distance is exact, not an approximation. As a correctness check, the bin each
recomputed distance falls into is compared against the last coordinate of the
elite's persisted 5-tuple niche key (the run's own novelty bin) and must
match exactly (`novelty_aware_axes()` appends the novelty axis last).

Usage::

    python -m research.tools.audit_novelty_axis_distribution
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from research.synthesis.novelty_distance import (
    DESCRIPTOR_SCALE,
    geometric_novelty,
    novelty_behavior_axis,
    softmax_basin_signatures,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = (
    _REPO_ROOT
    / "research/reports/autonomous_run_2026-07-02/deep_registry_discovery.log"
)
_DEFAULT_OUT = _REPO_ROOT / "research/reports/novelty_axis_audit_2026-07-02.json"

# The run that produced the log: `python -m research.tools.run_open_discovery`
# defaults (dim=32, novelty_aware=True) — confirmed against
# `research/tools/run_open_discovery.py`'s leaderboard print format, which
# emits exactly these 5 raw descriptor columns in this order.
_RUN_DIM = 32

_HEADER_RE = re.compile(r"evaluated=(\d+)\s+inserted=(\d+)\s+niches=(\d+)/(\d+)")
_ROW_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(([^)]+)\)\s+(\S+)\s*$"
)


@dataclass(frozen=True, slots=True)
class EliteRow:
    """One parsed leaderboard row from a `run_open_discovery` log."""

    fitness: float
    perm_equivariance: float
    shift_equivariance: float
    scale_homogeneity: float
    energy_gain: float
    spectral_radius: float
    niche: tuple[int, ...]
    program: str

    def descriptors(self) -> dict[str, float]:
        return {
            "perm_equivariance": self.perm_equivariance,
            "shift_equivariance": self.shift_equivariance,
            "scale_homogeneity": self.scale_homogeneity,
            "energy_gain": self.energy_gain,
            "spectral_radius": self.spectral_radius,
        }


def parse_discovery_log(path: Path) -> tuple[int, list[EliteRow]]:
    """Parse a `run_open_discovery` leaderboard log.

    Returns ``(reported_niche_count, rows)``. Fails loud if the header is
    missing, a row is unparseable, or the parsed row count does not match the
    header's reported niche count (a silent partial parse would corrupt every
    downstream stat).
    """
    text = path.read_text()
    header_match = _HEADER_RE.search(text)
    if header_match is None:
        raise ValueError(f"{path}: no `evaluated=... niches=N/M` header found")
    reported_niches = int(header_match.group(3))

    rows: list[EliteRow] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("evaluated=", "fit", "---")):
            continue
        match = _ROW_RE.match(line)
        if match is None:
            continue
        fit, perm, shift, scale, egain, specr, niche_str, program = match.groups()
        niche = tuple(int(x.strip()) for x in niche_str.split(","))
        rows.append(
            EliteRow(
                fitness=float(fit),
                perm_equivariance=float(perm),
                shift_equivariance=float(shift),
                scale_homogeneity=float(scale),
                energy_gain=float(egain),
                spectral_radius=float(specr),
                niche=niche,
                program=program,
            )
        )
    if len(rows) != reported_niches:
        raise ValueError(
            f"{path}: header reports {reported_niches} niches but parsed "
            f"{len(rows)} leaderboard rows — parser/log mismatch, not "
            f"silently proceeding with a partial table"
        )
    return reported_niches, rows


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (matches `statistics.quantiles` n=100 grid)."""
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def spread_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot compute spread stats over zero values")
    s = sorted(values)
    return {
        "n": len(s),
        "min": s[0],
        "q25": _quantile(s, 0.25),
        "median": statistics.median(s),
        "q75": _quantile(s, 0.75),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


def histogram(values: list[float], n_bins: int = 10) -> list[dict[str, float]]:
    """Fixed-width histogram over ``[min(values), max(values)]``."""
    if not values:
        raise ValueError("cannot histogram zero values")
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1)
        counts[idx] += 1
    return [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": counts[i]}
        for i in range(n_bins)
    ]


def log_histogram(values: list[float], n_bins: int = 10) -> list[dict[str, float]]:
    """Fixed-width histogram in log10 space — appropriate for a heavy-tailed
    novelty-distance distribution where a handful of extreme outliers would
    otherwise swallow every linear bin."""
    if any(v <= 0 for v in values):
        raise ValueError("log_histogram requires strictly positive values")
    logs = [math.log10(v) for v in values]
    lo, hi = min(logs), max(logs)
    if lo == hi:
        return [{"lo": 10**lo, "hi": 10**hi, "count": len(values)}]
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for lv in logs:
        idx = min(int((lv - lo) / width), n_bins - 1)
        counts[idx] += 1
    return [
        {
            "lo": 10 ** (lo + i * width),
            "hi": 10 ** (lo + (i + 1) * width),
            "count": counts[i],
        }
        for i in range(n_bins)
    ]


def audit(log_path: Path, run_dim: int) -> dict:
    reported_niches, rows = parse_discovery_log(log_path)
    basins = softmax_basin_signatures(dim=run_dim)
    axis = novelty_behavior_axis()

    per_elite = []
    bin_mismatches = []
    for row in rows:
        descriptors = row.descriptors()
        raw_distance = geometric_novelty(descriptors, basins=basins)
        recomputed_bin = axis.bin_of(raw_distance)
        logged_bin = row.niche[-1]
        if recomputed_bin != logged_bin:
            bin_mismatches.append(
                {
                    "program": row.program,
                    "niche": row.niche,
                    "recomputed_bin": recomputed_bin,
                    "logged_bin": logged_bin,
                    "raw_distance": raw_distance,
                }
            )
        per_elite.append(
            {
                "program": row.program,
                "fitness": row.fitness,
                "niche": list(row.niche),
                "current_novelty_bin": recomputed_bin,
                "raw_novelty_distance": raw_distance,
                "descriptors": descriptors,
            }
        )

    if bin_mismatches:
        raise RuntimeError(
            "recomputed novelty bin disagrees with the persisted niche's last "
            f"coordinate for {len(bin_mismatches)} elite(s) — the recomputation "
            f"is not reproducing the run's own axis; first mismatch: "
            f"{bin_mismatches[0]}"
        )

    all_distances = [e["raw_novelty_distance"] for e in per_elite]
    edges = axis.edges
    by_bin: dict[int, list[float]] = {}
    for e in per_elite:
        by_bin.setdefault(e["current_novelty_bin"], []).append(
            e["raw_novelty_distance"]
        )

    return {
        "log_path": str(log_path),
        "run_dim": run_dim,
        "reported_niches": reported_niches,
        "n_elites": len(per_elite),
        "current_bin_edges": list(edges),
        "descriptor_scale": dict(DESCRIPTOR_SCALE),
        "overall_spread": spread_stats(all_distances),
        "spread_by_current_bin": {
            str(b): spread_stats(vals) for b, vals in sorted(by_bin.items())
        },
        "histogram_linear": histogram(all_distances, n_bins=10),
        "histogram_log10": log_histogram(all_distances, n_bins=10),
        "per_elite": per_elite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=_DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--dim", type=int, default=_RUN_DIM)
    args = parser.parse_args()

    result = audit(args.log, args.dim)

    bin2 = result["spread_by_current_bin"].get("2")
    print(f"parsed {result['n_elites']} elites from {args.log}")
    print(f"current bin edges: {result['current_bin_edges']}")
    print("overall spread:", result["overall_spread"])
    for b, s in result["spread_by_current_bin"].items():
        print(
            f"  bin {b}: n={s['n']} min={s['min']:.3f} q25={s['q25']:.3f} "
            f"median={s['median']:.3f} q75={s['q75']:.3f} max={s['max']:.3f}"
        )
    if bin2 is not None:
        spread_ratio = bin2["max"] / bin2["min"] if bin2["min"] > 0 else float("inf")
        print(f"bin 2 (far) max/min ratio: {spread_ratio:.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
