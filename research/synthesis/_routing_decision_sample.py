"""Routing-decision sampling with optional advisory priors.

Extracted from ``_template_helpers.py`` (2026-05-11) to keep that file
under the 1250-line guardrail. Owns the per-(template, decision_key,
value) advisory weighting that lives behind ``sample_routing_choice``.

The advisory artifact itself is loaded by
``research.synthesis.routing_decision_priors`` from
``research/artifacts/routing_decision_priors/latest.json``. Per-graph
state is attached via ``graph._routing_decision_prior_state`` by
``grammar.generate_layer_graph`` when ``use_routing_decision_priors=True``.
This module is the consumer.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from .graph import ComputationGraph
from .routing_decision_priors import (
    routing_decision_prior_for,
    routing_decision_prior_weight,
)


def _json_safe_value(value: Any) -> Any:
    """Coerce a routing-choice value to a JSON-serialisable scalar.

    Duplicates ``_template_helpers._json_safe_value`` (3 lines) so this
    module has no upward import from ``_template_helpers``. Acceptable
    duplication: the function is leaf-pure and unlikely to drift.
    """
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _coerce_prior_strength(value: Any) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return 1.0
    if strength != strength:  # NaN
        return 1.0
    return max(0.0, min(4.0, strength))


def sample_routing_choice_with_prior(
    rng: random.Random,
    choices: Sequence[Any],
    *,
    graph: ComputationGraph,
    template_name: str,
    decision_key: str,
) -> tuple[Any, dict[str, Any] | None]:
    """Sample a routing choice with optional advisory prior weighting.

    Returns ``(value, attribution)``. ``attribution`` is ``None`` when no
    prior was applied — caller falls back to neutral telemetry.
    """
    prior_state = getattr(graph, "_routing_decision_prior_state", None)
    if not prior_state:
        return rng.choice(list(choices)), None
    prior = prior_state.get("prior") if isinstance(prior_state, dict) else None
    if not isinstance(prior, dict) or not prior.get("loaded"):
        return rng.choice(list(choices)), None
    strength = _coerce_prior_strength(prior_state.get("strength", 1.0))
    if strength <= 0.0:
        return rng.choice(list(choices)), None

    weighted: list[tuple[Any, float, float, int]] = []
    matched = 0
    for choice in choices:
        record = routing_decision_prior_for(prior, template_name, decision_key, choice)
        advisory_weight = (
            routing_decision_prior_weight(
                prior, template_name, decision_key, choice, default=1.0
            )
            if record
            else 1.0
        )
        effective_weight = 1.0 + ((advisory_weight - 1.0) * strength)
        effective_weight = max(0.05, min(4.0, float(effective_weight)))
        support = int(record.get("n") or 0) if record else 0
        if record is not None:
            matched += 1
        weighted.append((choice, effective_weight, float(advisory_weight), support))

    if matched == 0 or all(abs(row[1] - 1.0) < 1e-12 for row in weighted):
        return rng.choice(list(choices)), None

    total = sum(row[1] for row in weighted)
    threshold = rng.random() * total
    running = 0.0
    selected = weighted[-1]
    for row in weighted:
        running += row[1]
        if threshold <= running:
            selected = row
            break

    value = selected[0]
    attribution = _routing_prior_attribution(
        prior=prior,
        weighted=weighted,
        selected=value,
        template_name=template_name,
        decision_key=decision_key,
        strength=strength,
    )
    return value, attribution


def _routing_prior_attribution(
    *,
    prior: dict[str, Any],
    weighted: Sequence[tuple[Any, float, float, int]],
    selected: Any,
    template_name: str,
    decision_key: str,
    strength: float,
) -> dict[str, Any]:
    total = sum(row[1] for row in weighted) or 1.0
    selected_key = _json_safe_value(selected)
    top_choices = sorted(weighted, key=lambda row: row[1], reverse=True)[:5]
    return {
        "version": prior.get("version"),
        "template_name": str(template_name),
        "decision_key": str(decision_key),
        "strength": float(strength),
        "selected_effective_weight": next(
            float(row[1])
            for row in weighted
            if _json_safe_value(row[0]) == selected_key
        ),
        "selected_probability": next(
            float(row[1]) / total
            for row in weighted
            if _json_safe_value(row[0]) == selected_key
        ),
        "matched_choices": sum(1 for row in weighted if row[3] > 0),
        "choice_count": len(weighted),
        "top_choices": [
            {
                "value": _json_safe_value(choice),
                "effective_weight": round(float(effective), 6),
                "advisory_weight": round(float(advisory), 6),
                "probability": round(float(effective) / total, 6),
                "support": int(support),
            }
            for choice, effective, advisory, support in top_choices
        ],
    }
