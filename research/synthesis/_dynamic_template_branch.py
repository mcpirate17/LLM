"""Dynamic-template branch for the per-block generation loop.

Extracted from ``grammar.generate_layer_graph`` (2026-05-11) to keep that
module under the 1250-line guardrail. Encapsulates:

  1. Eligibility — whether the dynamic-template pool can be sampled at
     this t_idx, including the ``routing_mandatory`` bypass (only candidates
     whose chain contains a routing op qualify under the mandate).
  2. Sampling — bounded evidence-weighted draw from the eligible pool.
  3. Apply + rollback — calls ``apply_dynamic_template_candidate``;
     on failure restores the graph to its pre-attempt snapshot.
  4. Telemetry — appends a record to ``graph.metadata["dynamic_template_attempts"]``
     for every attempt (success or rollback), so a post-hoc audit can
     distinguish tried-and-failed from never-sampled.

Caller contract: pass the snapshot ids/metadata captured *before* this
attempt. Returns ``(trial_current, dynamic_used)`` where ``dynamic_used``
is True iff a dynamic candidate was successfully applied. When False,
the caller should fall back to the static ``apply_template`` path.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

from .dynamic_template_registry import (
    apply_dynamic_template_candidate,
    choose_dynamic_template_candidate,
)
from .graph import ComputationGraph
from ._context_op_sets import _GATING_OPS


def maybe_apply_dynamic_template(
    *,
    graph: ComputationGraph,
    current: int,
    rng: random.Random,
    runtime: Any,
    config: Any,
    t_idx: int,
    prev_next_id: int,
    prev_output_id: int,
    prev_metadata: Mapping[str, Any],
) -> tuple[Optional[int], bool]:
    """Try a dynamic candidate for this block; return (trial_tail, used).

    ``used=False`` means the dynamic branch was skipped or rolled back —
    the caller is responsible for invoking the static template path.
    """
    if not runtime.dynamic_template_candidates:
        return None, False
    if config.forced_template:
        return None, False

    dynamic_prob = max(0.0, min(1.0, float(config.dynamic_template_candidate_prob)))
    if dynamic_prob <= 0.0 or rng.random() >= dynamic_prob:
        return None, False

    # routing_mandatory at t_idx=0 normally forces the static routing
    # allowlist; we permit the dynamic branch only when the chosen
    # candidate's chain contains a routing op, preserving the mandate.
    routing_allowlist_active = bool(
        config.routing_mandatory and t_idx == 0 and not config.forced_template
    )

    eligible = runtime.dynamic_template_candidates
    if routing_allowlist_active:
        eligible = tuple(
            c for c in eligible if any(op in _GATING_OPS for op in c.chain)
        )
    if not eligible:
        return None, False

    candidate = choose_dynamic_template_candidate(
        rng,
        eligible,
        strength=config.dynamic_template_candidate_strength,
    )

    attempt: dict[str, Any] = {
        "template_id": candidate.template_id,
        "chain": list(candidate.chain),
        "t_idx": int(t_idx),
    }
    try:
        trial = apply_dynamic_template_candidate(graph, current, rng, candidate)
    except Exception as exc:  # noqa: BLE001 — rollback is intentional
        attempt["status"] = "rolled_back"
        attempt["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        _rollback(graph, prev_next_id, prev_output_id, prev_metadata)
        graph.metadata.setdefault("dynamic_template_attempts", []).append(attempt)
        return None, False

    attempt["status"] = "ok"
    graph.metadata.setdefault("dynamic_template_attempts", []).append(attempt)
    return trial, True


def _rollback(
    graph: ComputationGraph,
    prev_next_id: int,
    prev_output_id: int,
    prev_metadata: Mapping[str, Any],
) -> None:
    for nid in range(prev_next_id, graph._next_id):
        graph.nodes.pop(nid, None)
    graph._next_id = prev_next_id
    graph._output_node_id = prev_output_id
    graph.metadata = dict(prev_metadata)
    graph._cache.clear()
