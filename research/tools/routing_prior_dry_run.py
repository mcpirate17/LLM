#!/usr/bin/env python
"""Report how routing-decision priors would reweight observed choice sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.synthesis.routing_decision_priors import (
    DEFAULT_ROUTING_DECISION_PRIOR_PATH,
    load_routing_decision_priors,
)


def build_routing_prior_dry_run(
    path: str | Path = DEFAULT_ROUTING_DECISION_PRIOR_PATH,
    *,
    strength: float = 1.0,
    max_groups: int = 80,
) -> dict[str, Any]:
    """Return uniform-vs-prior probabilities for artifact decision groups."""

    prior = load_routing_decision_priors(path)
    if not prior.get("loaded"):
        return {
            "loaded": False,
            "load_reason": prior.get("load_reason"),
            "path": str(path),
            "groups": [],
        }

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in prior.get("records") or []:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("template_name") or ""),
            str(record.get("decision_key") or ""),
        )
        if not key[0] or not key[1]:
            continue
        groups.setdefault(key, []).append(record)

    clean_strength = max(0.0, min(4.0, float(strength)))
    reports: list[dict[str, Any]] = []
    for (template_name, decision_key), records in groups.items():
        rows = []
        for record in records:
            advisory = _safe_float(record.get("advisory_weight"), 1.0)
            effective = max(0.05, min(4.0, 1.0 + ((advisory - 1.0) * clean_strength)))
            rows.append(
                {
                    "value": record.get("value"),
                    "n": int(record.get("n") or 0),
                    "advisory_weight": round(advisory, 6),
                    "effective_weight": round(effective, 6),
                }
            )
        total = sum(float(row["effective_weight"]) for row in rows) or 1.0
        uniform = 1.0 / max(1, len(rows))
        for row in rows:
            prior_probability = float(row["effective_weight"]) / total
            row["uniform_probability"] = round(uniform, 6)
            row["prior_probability"] = round(prior_probability, 6)
            row["probability_delta"] = round(prior_probability - uniform, 6)
        rows.sort(key=lambda row: abs(float(row["probability_delta"])), reverse=True)
        reports.append(
            {
                "template_name": template_name,
                "decision_key": decision_key,
                "choice_count": len(rows),
                "max_abs_probability_delta": (
                    abs(float(rows[0]["probability_delta"])) if rows else 0.0
                ),
                "choices": rows,
            }
        )

    reports.sort(
        key=lambda row: abs(float(row["max_abs_probability_delta"])), reverse=True
    )
    return {
        "loaded": True,
        "path": str(path),
        "version": prior.get("version"),
        "strength": clean_strength,
        "group_count": len(reports),
        "groups": reports[: max(1, int(max_groups))],
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if converted == converted else default


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Routing Prior Dry Run",
        "",
        f"- Loaded: `{report.get('loaded')}`",
        f"- Version: `{report.get('version') or ''}`",
        f"- Strength: `{report.get('strength', '')}`",
        f"- Groups: {report.get('group_count', 0)}",
        "",
        "| template | decision | value | n | uniform | prior | delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in report.get("groups") or []:
        for row in group.get("choices") or []:
            lines.append(
                "| {template} | {decision} | `{value}` | {n} | {uniform:.3f} | {prior:.3f} | {delta:+.3f} |".format(
                    template=group.get("template_name") or "",
                    decision=group.get("decision_key") or "",
                    value=json.dumps(row.get("value"), sort_keys=True),
                    n=int(row.get("n") or 0),
                    uniform=float(row.get("uniform_probability") or 0.0),
                    prior=float(row.get("prior_probability") or 0.0),
                    delta=float(row.get("probability_delta") or 0.0),
                )
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default=str(DEFAULT_ROUTING_DECISION_PRIOR_PATH))
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-groups", type=int, default=80)
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args(argv)

    report = build_routing_prior_dry_run(
        args.prior,
        strength=args.strength,
        max_groups=args.max_groups,
    )
    if args.format == "md":
        print(_format_markdown(report), end="")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
