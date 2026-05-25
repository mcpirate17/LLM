"""Label-free MEASURED-descriptor rescue of GBM-prescreener-dropped candidates.

The GBM prescreener (`execution_experiment_phase3._partition_prescreener_candidates`) drops
candidates whose predicted ``p_pass`` is below the floor. The GBM is label-fit and systematically
skeptical of novel mechanisms — it scored the confirmed STDP-attention winner ``e656938e`` at
0.291, which would silently kill it before any compute. This re-admits dropped candidates that the
closed-book MEASURED filter flags as structurally induction-capable.

The signal is read off the graph's *actual computation* at random init (position-Jacobian,
``MeasuredDescriptorExtractor``) — no training, no capability labels, no op names. Validated in
``project_measured_descriptors``: prospective ROC 0.911 > the label-trained screener 0.768, with
0 capable candidates wrongly skipped.

Design invariants (so this is safe to leave wired):
- **Default OFF** (``ARIA_MEASURED_RESCUE`` unset) → ``measured_rescue_config`` returns ``None`` →
  byte-identical to current behavior.
- **Additive**: only re-admits from the skip pile → recall can only rise, never fall.
- **Affirmative-only** (NOT fail-open): a graph is rescued iff ``descriptors()`` succeeds AND its
  ``long_range_reach >= tau``; un-measurable junk is never rescued.
- **Bounded cost**: at most ``probe_budget`` graphs probed (~0.4s each) and ``max_rescue`` re-admitted
  per batch; isolated try/except so a rescue failure yields zero rescues, never breaks the gate.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TAU = (
    0.01  # validated operating point (n=1102): keeps 99.3% of induction-capable
)
_DEFAULT_MAX_RESCUE = 8
_DEFAULT_PROBE_BUDGET = 64


@dataclass(frozen=True)
class MeasuredRescueConfig:
    tau: float
    max_rescue: int
    probe_budget: int
    device: Optional[str]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def measured_rescue_config(
    device: Optional[str] = None,
) -> Optional[MeasuredRescueConfig]:
    """Read rescue config from env; return ``None`` when disabled (the default) → no-op gate."""
    if os.environ.get("ARIA_MEASURED_RESCUE") != "1":
        return None
    return MeasuredRescueConfig(
        tau=_env_float("ARIA_MEASURED_RESCUE_TAU", _DEFAULT_TAU),
        max_rescue=max(0, _env_int("ARIA_MEASURED_RESCUE_MAX", _DEFAULT_MAX_RESCUE)),
        probe_budget=max(
            0, _env_int("ARIA_MEASURED_RESCUE_PROBE_BUDGET", _DEFAULT_PROBE_BUDGET)
        ),
        device=device,
    )


def rescue_skipped_candidates(
    skipped: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]],
    cfg: MeasuredRescueConfig,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Re-admit GBM-dropped candidates the measured filter flags structurally capable.

    ``skipped``: ``(graph, graph_dict, skip_metrics)`` tuples the prescreener would drop.
    Returns ``(rescued_graphs, records)``. Probes at most ``cfg.probe_budget`` graphs and
    re-admits at most ``cfg.max_rescue``. Affirmative-only: never fail-open.
    """
    if cfg.max_rescue <= 0 or cfg.probe_budget <= 0 or not skipped:
        return [], []
    try:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor
    except (
        Exception
    ) as exc:  # isolation — a missing/broken extractor must not break the gate
        logger.debug("measured rescue unavailable (import failed): %s", exc)
        return [], []
    try:
        mdx = MeasuredDescriptorExtractor(device=cfg.device, n_seeds=1)
    except Exception as exc:
        logger.debug("measured rescue extractor init failed: %s", exc)
        return [], []

    rescued: List[Any] = []
    records: List[Dict[str, Any]] = []
    probed = 0
    for graph, graph_dict, skip_metrics in skipped:
        if len(rescued) >= cfg.max_rescue or probed >= cfg.probe_budget:
            break
        probed += 1
        try:
            d = mdx.descriptors(json.dumps(graph_dict, separators=(",", ":")))
        except Exception as exc:
            logger.debug("measured rescue probe failed: %s", exc)
            d = None
        if d is None:  # affirmative-only: do not rescue what we could not measure
            continue
        reach = float(d.get("long_range_reach", 0.0))
        if reach < cfg.tau:
            continue
        rescued.append(graph)
        fp = graph.fingerprint() if hasattr(graph, "fingerprint") else ""
        records.append(
            {
                "graph_fingerprint": fp,
                "measured_long_range_reach": round(reach, 6),
                "measured_content_dependence": round(
                    float(d.get("content_dependence", 0.0)), 6
                ),
                "predicted_p_s1": round(
                    float(skip_metrics.get("predicted_p_s1", 0.0)), 6
                ),
            }
        )

    if records:
        logger.info(
            "measured rescue: re-admitted %d GBM-dropped candidate(s) (probed %d/%d, tau=%.3f) "
            "flagged structurally induction-capable",
            len(rescued),
            probed,
            len(skipped),
            cfg.tau,
        )
    return rescued, records
