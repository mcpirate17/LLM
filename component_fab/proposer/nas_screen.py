"""Best-effort NAS pipeline screening for fab proposal specs.

The NAS pipeline scores graph candidates, while component_fab emits single
component specs. This adapter builds a small proxy graph from codegen-relevant
axes and asks the existing CPU/NAS oracle to score that proxy. It is a cheap
screen only; Tier-2 remains the downstream evidence gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from component_fab.proposer.measured_screen import (
    LONG_RANGE_THRESHOLD,
    MAX_CAUSALITY_VIOLATION,
    measured_screen_for_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NasScreenResult:
    proposal_id: str
    available: bool
    gate_pass: bool
    downstream_gate_pass: bool
    rank_score: float
    source: str
    reason: str = ""
    calibration: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def _measured_calibration() -> dict[str, Any]:
    """Operating-point context for the measured-descriptor binding screen.

    The 2026-06-03 audit retired the NAS oracle proxy-graph gate
    (anti-predictive, OOD on the 3-op stub); this screen reads the
    position-Jacobian of the REAL fab module instead.
    """

    return {
        "screen": "measured_descriptors",
        "gate_axis": "long_range_reach",
        "gate_threshold": LONG_RANGE_THRESHOLD,
        "operating_point": "long_range_reach>=0.01 keeps 99.3% of "
        "induction-capable graphs, prunes ~55% of incapable (n=1102)",
        "note": "filters non-binders (MLP-class long_range_reach~0); not a "
        "fine-grained ranker within one architecture family",
    }


def score_spec_with_nas(
    spec: ProposalSpec, scorer: Any | None = None
) -> NasScreenResult:
    """Screen one spec on the MEASURED graph properties of its real module.

    ``scorer`` is an optional ``MeasuredDescriptorExtractor`` reused across specs.
    Replaces the former NAS oracle proxy-graph path (anti-predictive, see
    ``measured_screen``); same ``NasScreenResult`` contract so callers are
    unchanged.
    """

    ms = measured_screen_for_spec(spec, extractor=scorer)
    return NasScreenResult(
        proposal_id=spec.proposal_id,
        available=ms.available,
        gate_pass=ms.binds_likely,
        downstream_gate_pass=ms.causality_violation <= MAX_CAUSALITY_VIOLATION,
        rank_score=ms.rank_score,
        source="measured_descriptors" if ms.available else "unavailable",
        reason=ms.reason,
        calibration=_measured_calibration(),
        raw=ms.descriptors,
    )


def score_specs_with_nas(
    specs: Iterable[ProposalSpec],
    *,
    enabled: bool = True,
) -> dict[str, NasScreenResult]:
    if not enabled:
        return {}
    extractor: Any | None = None
    try:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor

        extractor = MeasuredDescriptorExtractor(n_seeds=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("measured-descriptor extractor unavailable: %s", exc)
    return {
        spec.proposal_id: score_spec_with_nas(spec, scorer=extractor) for spec in specs
    }


def nas_score_multiplier(result: NasScreenResult | None) -> float:
    if result is None or not result.available:
        return 1.0
    if not result.gate_pass:
        return 0.55
    if not result.downstream_gate_pass:
        return 0.70
    if result.rank_score >= 1.25:
        return 1.08
    if result.rank_score >= 1.0:
        return 1.03
    return 1.0
