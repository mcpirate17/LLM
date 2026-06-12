"""Fidelity ladder — does the nano tier predict higher fidelity? (WS-7).

The project's recurring failure: nano grades (dim32/seq32/60steps) don't reliably
predict what survives at scale, so "wins" keep evaporating. This module measures
that directly. Candidates are scored at rungs:
  - R0: current nano (dim32 / seq32 / 60 steps)
  - R1: dim128 / seq256 / 500 steps
  - R2: a real ARIA child run on WikiText-103 (requires the runtime — populated by
        the evidence backflow, not this analyzer)
For every metric scored at two rungs across enough candidates, we compute the
Spearman rank correlation between rungs. A metric whose R0→R1 correlation is weak
is one the nano tier cannot be trusted on — flagged for weight demotion (it feeds
the WS-4 objective weights and the WS-3 surrogate).

Pure store + analyzer. Rung scores accumulate in ``catalog/fidelity_scores.jsonl``
(one record per candidate per rung); the R1 producer is ``tools/run_fidelity.py``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ._stats import spearman
from .ledger import JsonlWriter, iter_jsonl_records, write_json_report

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = _REPO / "component_fab" / "catalog" / "fidelity_scores.jsonl"
DEFAULT_OUTPUT_PATH = _REPO / "component_fab" / "catalog" / "fidelity_report.json"

RUNGS: tuple[str, ...] = ("R0", "R1", "R2")
WEAK_SPEARMAN = 0.30  # below this, the lower rung does not predict the higher one
DEFAULT_MIN_PAIRS = 8  # fewer paired candidates than this → correlation untrustworthy


@dataclass(slots=True)
class RungScore:
    proposal_id: str
    rung: str
    metrics: dict[str, float]


@dataclass(slots=True)
class MetricFidelity:
    metric: str
    n_pairs: int
    spearman: float | None
    weak: bool  # True when correlation is computable and below WEAK_SPEARMAN
    demote_recommended: bool


@dataclass(slots=True)
class FidelityReport:
    rung_a: str
    rung_b: str
    n_candidates_paired: int
    metrics: list[MetricFidelity] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def append_rung_scores(
    scores: Sequence[RungScore], store_path: Path | str = DEFAULT_STORE_PATH
) -> Path:
    unknown = [s.rung for s in scores if s.rung not in RUNGS]
    if unknown:
        raise ValueError(f"unknown rung {unknown[0]!r}; expected one of {RUNGS}")
    with JsonlWriter(store_path) as writer:
        for score in scores:
            writer.write(
                {
                    "proposal_id": score.proposal_id,
                    "rung": score.rung,
                    "metrics": score.metrics,
                }
            )
    return Path(store_path)


def read_rung_scores(store_path: Path | str = DEFAULT_STORE_PATH) -> list[RungScore]:
    return [
        RungScore(
            proposal_id=str(record["proposal_id"]),
            rung=str(record["rung"]),
            metrics={k: float(v) for k, v in record["metrics"].items()},
        )
        for record in iter_jsonl_records(store_path)
        if record.get("proposal_id")
        and record.get("rung")
        and isinstance(record.get("metrics"), dict)
    ]


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
def _latest_per_candidate(
    records: Sequence[RungScore], rung: str
) -> dict[str, dict[str, float]]:
    """Latest metrics dict per candidate at ``rung`` (later records win)."""
    return {rec.proposal_id: rec.metrics for rec in records if rec.rung == rung}


def compute_fidelity_report(
    records: Sequence[RungScore],
    *,
    rung_a: str = "R0",
    rung_b: str = "R1",
    weak_threshold: float = WEAK_SPEARMAN,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> FidelityReport:
    """Per-metric Spearman between two rungs over candidates scored at both."""
    a = _latest_per_candidate(records, rung_a)
    b = _latest_per_candidate(records, rung_b)
    paired_ids = sorted(set(a) & set(b))
    metric_pairs: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for pid in paired_ids:
        for metric in set(a[pid]) & set(b[pid]):
            xa, xb = metric_pairs[metric]
            xa.append(a[pid][metric])
            xb.append(b[pid][metric])

    metrics: list[MetricFidelity] = []
    for metric in sorted(metric_pairs):
        xa, xb = metric_pairs[metric]
        n = len(xa)
        if n < min_pairs:
            metrics.append(
                MetricFidelity(
                    metric=metric,
                    n_pairs=n,
                    spearman=None,
                    weak=False,
                    demote_recommended=False,
                )
            )
            continue
        rho = spearman(np.array(xa, dtype=float), np.array(xb, dtype=float))
        weak = rho < weak_threshold
        metrics.append(
            MetricFidelity(
                metric=metric,
                n_pairs=n,
                spearman=round(rho, 4),
                weak=weak,
                demote_recommended=weak,
            )
        )
    findings = _fidelity_findings(metrics, rung_a, rung_b, len(paired_ids), min_pairs)
    return FidelityReport(
        rung_a=rung_a,
        rung_b=rung_b,
        n_candidates_paired=len(paired_ids),
        metrics=metrics,
        findings=findings,
    )


def _fidelity_findings(
    metrics: list[MetricFidelity],
    rung_a: str,
    rung_b: str,
    n_paired: int,
    min_pairs: int,
) -> list[str]:
    findings: list[str] = []
    computable = [m for m in metrics if m.spearman is not None]
    if not computable:
        findings.append(
            f"Not enough candidates scored at both {rung_a} and {rung_b} "
            f"(need >= {min_pairs} per metric; have {n_paired} paired). Run the "
            f"{rung_b} producer on more candidates before trusting these numbers."
        )
        return findings
    weak = [m for m in computable if m.weak]
    strong = [m for m in computable if not m.weak]
    for m in weak:
        findings.append(
            f"WEAK FIDELITY: '{m.metric}' {rung_a}->{rung_b} Spearman={m.spearman:.3f} "
            f"(n={m.n_pairs}) — nano does not predict it. Demote its {rung_a} weight "
            f"in the composite/objectives; do not gate promotion on it at {rung_a}."
        )
    for m in strong:
        findings.append(
            f"TRUSTWORTHY: '{m.metric}' {rung_a}->{rung_b} Spearman={m.spearman:.3f} "
            f"(n={m.n_pairs}) — nano tracks the higher rung; keep its weight."
        )
    return findings


def demoted_metrics(report: FidelityReport) -> list[str]:
    """Metrics whose lower-rung weight WS-3/WS-4 should demote."""
    return [m.metric for m in report.metrics if m.demote_recommended]


def write_fidelity_report(
    report: FidelityReport, output_path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    payload = {
        "rung_a": report.rung_a,
        "rung_b": report.rung_b,
        "n_candidates_paired": report.n_candidates_paired,
        "metrics": [asdict(m) for m in report.metrics],
        "findings": report.findings,
    }
    return write_json_report(payload, output_path)
