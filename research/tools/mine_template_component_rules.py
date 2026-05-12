"""Mine component-rule evidence from the static template corpus.

The database miner observes historical outcomes. This companion samples the
current hand-built template registry and records what structures templates
actually emit, so future component rules can be derived from template behavior
instead of copied into more hardcoded call sites.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.synthesis.component_rules import (
    component_role_counts,
    estimated_chain_lowered_op_count,
    validate_component_op_chain,
)
from research.synthesis.graph import ComputationGraph
from research.synthesis.op_roles import OpRole, get_role
from research.synthesis.templates import TEMPLATES, apply_template


DEFAULT_OUTPUT_DIR = Path("research/reports")
RECURSION_OPS = frozenset(
    {
        "fixed_point_iter",
        "mixture_of_recursions",
        "depth_gated_transform",
        "score_depth_blend",
        "depth_weighted_proj",
    }
)


def mine_template_component_rules(
    *,
    template_names: Iterable[str] | None = None,
    seeds_per_template: int = 2,
    model_dim: int = 64,
    min_window_ops: int = 8,
) -> dict[str, Any]:
    """Sample templates and return emitted op/window evidence."""
    # guardrail: allow-complexity - offline read-only corpus sampler, not runtime.
    names = tuple(template_names) if template_names is not None else tuple(TEMPLATES)
    summaries: list[dict[str, Any]] = []
    windows: Counter[tuple[str, ...]] = Counter()
    failures: list[dict[str, Any]] = []

    for template_name in names:
        if template_name not in TEMPLATES:
            failures.append({"template": template_name, "error": "unknown_template"})
            continue
        emitted: list[tuple[str, ...]] = []
        for seed in range(max(1, int(seeds_per_template))):
            try:
                ops = _sample_template_ops(
                    template_name,
                    seed=seed,
                    model_dim=int(model_dim),
                )
            except Exception as exc:  # noqa: BLE001 - mining should keep going
                failures.append(
                    {
                        "template": template_name,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
                continue
            if not ops:
                continue
            emitted.append(ops)
            for window in _windows(ops, size=int(min_window_ops)):
                windows[window] += 1
        if emitted:
            summaries.append(_template_summary(template_name, emitted))

    return {
        "schema_version": "template_component_rule_mining_v1",
        "created_at": time.time(),
        "model_dim": int(model_dim),
        "seeds_per_template": int(seeds_per_template),
        "min_window_ops": int(min_window_ops),
        "templates_requested": len(names),
        "templates_sampled": len(summaries),
        "failures": failures[:120],
        "template_summaries": summaries,
        "candidate_windows": [
            _window_summary(window, count) for window, count in windows.most_common(160)
        ],
    }


def _sample_template_ops(
    template_name: str, *, seed: int, model_dim: int
) -> tuple[str, ...]:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    tail = apply_template(
        graph, input_id, random.Random(seed), template_name=template_name
    )
    graph.set_output(tail)
    return _topological_ops(graph)


def _topological_ops(graph: ComputationGraph) -> tuple[str, ...]:
    ops: list[str] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.is_input or node.is_output:
            continue
        if node.op_name not in {"input", "output"}:
            ops.append(str(node.op_name))
    return tuple(ops)


def _windows(ops: Sequence[str], *, size: int) -> Iterable[tuple[str, ...]]:
    window_size = max(1, int(size))
    if len(ops) < window_size:
        return ()
    return (
        tuple(ops[idx : idx + window_size]) for idx in range(len(ops) - window_size + 1)
    )


def _template_summary(
    template_name: str, emitted: list[tuple[str, ...]]
) -> dict[str, Any]:
    merged = tuple(op for ops in emitted for op in ops)
    role_counts = component_role_counts(merged)
    unique_ops = sorted(set(merged))
    return {
        "template": template_name,
        "samples": len(emitted),
        "min_ops": min(len(ops) for ops in emitted),
        "max_ops": max(len(ops) for ops in emitted),
        "unique_ops": unique_ops,
        "role_counts": role_counts,
        "mixer_ops": sorted(op for op in set(merged) if get_role(op) is OpRole.MIX),
        "route_ops": sorted(op for op in set(merged) if get_role(op) is OpRole.ROUTE),
        "recursion_ops": sorted(op for op in set(merged) if op in RECURSION_OPS),
    }


def _window_summary(window: tuple[str, ...], count: int) -> dict[str, Any]:
    role_counts = component_role_counts(window)
    return {
        "pattern": list(window),
        "count": int(count),
        "lowered_op_count": estimated_chain_lowered_op_count(window),
        "role_counts": role_counts,
        "mixer_count": int(role_counts.get(OpRole.MIX.value, 0)),
        "route_count": int(role_counts.get(OpRole.ROUTE.value, 0)),
        "has_recursion_signal": any(op in RECURSION_OPS for op in window),
        "violations": list(validate_component_op_chain(window)),
    }


def _write_report(report: dict[str, Any], output: str | Path | None) -> Path:
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = (
            DEFAULT_OUTPUT_DIR / f"template_component_rule_mining_{stamp}.json"
        )
    else:
        output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", nargs="*", default=None)
    parser.add_argument("--seeds-per-template", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--min-window-ops", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = mine_template_component_rules(
        template_names=args.templates,
        seeds_per_template=args.seeds_per_template,
        model_dim=args.model_dim,
        min_window_ops=args.min_window_ops,
    )
    path = _write_report(report, args.output)
    print(
        "template_component_rule_mining "
        f"sampled={report['templates_sampled']}/{report['templates_requested']} "
        f"windows={len(report['candidate_windows'])} "
        f"failures={len(report['failures'])} "
        f"report={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
