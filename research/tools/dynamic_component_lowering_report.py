"""Report dynamic component quality by lowering family.

Read-only analysis tool. It compares historical ``program_results`` rows that
used dynamic components with the current descriptor artifact, so sampling
changes can distinguish screened evidence from merely validated candidates.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


DEFAULT_DB = Path("research/runs.db")
DEFAULT_CANDIDATES = Path(
    "research/data/synthesis_candidates/dynamic_component_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("research/reports")


def build_dynamic_component_lowering_report(
    *,
    db_path: str | Path = DEFAULT_DB,
    candidate_path: str | Path = DEFAULT_CANDIDATES,
    output_path: str | Path | None = None,
    limit: int = 10000,
) -> dict[str, Any]:
    """Build and optionally write a dynamic-lowering quality report."""
    db = Path(db_path)
    rows = _fetch_dynamic_rows(db, limit=max(1, int(limit)))
    historical, attempts = _summarize_historical_rows(db, rows)
    candidates = _summarize_candidate_artifact(Path(candidate_path))
    report = {
        "schema_version": "dynamic_component_lowering_report_v1",
        "created_at": time.time(),
        "db_path": str(db),
        "candidate_path": str(candidate_path),
        "limit": int(limit),
        "historical": historical,
        "candidate_artifact": candidates,
        "dynamic_attempts": attempts,
        "selection_recommendations": _selection_recommendations(
            historical.get("by_lowering", []),
            candidates.get("by_lowering", []),
        ),
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _fetch_dynamic_rows(db: Path, *, limit: int) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    table = _program_results_read_table(conn)
    return list(
        conn.execute(
            f"""
            SELECT result_id, graph_json, stage1_passed, loss_ratio, timestamp
            FROM {table}
            WHERE COALESCE(graph_json, '') NOT IN ('', '{{}}')
              AND (
                graph_json LIKE '%dynamic_components_used%'
                OR graph_json LIKE '%dynamic_templates_used%'
                OR graph_json LIKE '%dynamic_template_attempts%'
              )
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
    )


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='program_results_compat' LIMIT 1"
    ).fetchone()
    return "program_results_compat" if row else "program_results"


def _summarize_historical_rows(
    db: Path,
    rows: list[sqlite3.Row],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lowering_stats: dict[str, dict[str, Any]] = defaultdict(_new_lowering_stats)
    component_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        _new_component_stats
    )
    attempt_counts = Counter()
    parsed_rows = 0
    dynamic_rows = 0

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for row in rows:
        graph = _parse_graph(conn, db, row["graph_json"])
        if not graph:
            continue
        parsed_rows += 1
        metadata = (
            graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
        )
        components = _dynamic_component_records(metadata)
        attempts = metadata.get("dynamic_template_attempts")
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, Mapping):
                    attempt_counts[str(attempt.get("status") or "unknown")] += 1
        if not components:
            continue
        dynamic_rows += 1
        stage1 = bool(row["stage1_passed"])
        loss_ratio = _safe_float(row["loss_ratio"])
        result_id = str(row["result_id"])
        for component in components:
            lowering = _component_lowering(component)
            component_id = _component_id(component)
            _update_lowering_stats(
                lowering_stats[lowering],
                stage1=stage1,
                loss_ratio=loss_ratio,
                result_id=result_id,
            )
            _update_component_stats(
                component_stats[(lowering, component_id)],
                component_id=component_id,
                stage1=stage1,
                loss_ratio=loss_ratio,
                result_id=result_id,
            )

    by_lowering = [
        _finalize_lowering(name, stats) for name, stats in lowering_stats.items()
    ]
    by_lowering.sort(key=lambda item: (-item["n_components"], item["lowering"]))
    best_components = [
        _finalize_component(lowering, stats)
        for (lowering, _component_id), stats in component_stats.items()
    ]
    best_components.sort(
        key=lambda item: (
            -(item["n_stage1"] or 0),
            item["mean_loss_ratio"] if item["mean_loss_ratio"] is not None else 999.0,
            item["component_id"],
        )
    )
    return (
        {
            "rows_scanned": len(rows),
            "rows_parsed": parsed_rows,
            "dynamic_rows": dynamic_rows,
            "by_lowering": by_lowering,
            "best_components": best_components[:20],
        },
        {
            "total": int(sum(attempt_counts.values())),
            "by_status": dict(attempt_counts),
            "rollback_count": int(attempt_counts.get("rolled_back", 0)),
        },
    )


def _parse_graph(
    conn: sqlite3.Connection,
    db: Path,
    graph_json: Any,
) -> dict[str, Any] | None:
    try:
        payload = resolve_graph_json_value(conn, db, graph_json)
        graph = json.loads(payload)
    except Exception:
        return None
    return graph if isinstance(graph, dict) else None


def _dynamic_component_records(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    current = [
        item
        for item in _metadata_list(metadata, "dynamic_components_used")
        if isinstance(item, Mapping)
    ]
    if current:
        return current
    return [
        item
        for item in _metadata_list(metadata, "dynamic_templates_used")
        if isinstance(item, Mapping)
    ]


def _metadata_list(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata.get(key)
    return value if isinstance(value, list) else []


def _component_lowering(component: Mapping[str, Any]) -> str:
    descriptor = component.get("component_descriptor")
    if not isinstance(descriptor, Mapping):
        descriptor = {}
    return str(component.get("lowering") or descriptor.get("lowering") or "unknown")


def _component_id(component: Mapping[str, Any]) -> str:
    descriptor = component.get("component_descriptor")
    if not isinstance(descriptor, Mapping):
        descriptor = {}
    return str(
        component.get("component_id")
        or descriptor.get("component_id")
        or component.get("template_id")
        or "unknown"
    )


def _new_lowering_stats() -> dict[str, Any]:
    return {"n": 0, "n_stage1": 0, "losses": [], "result_ids": set()}


def _new_component_stats() -> dict[str, Any]:
    return {
        "component_id": "",
        "n": 0,
        "n_stage1": 0,
        "losses": [],
        "result_ids": set(),
    }


def _update_lowering_stats(
    stats: dict[str, Any],
    *,
    stage1: bool,
    loss_ratio: float | None,
    result_id: str,
) -> None:
    stats["n"] += 1
    stats["n_stage1"] += int(stage1)
    stats["result_ids"].add(result_id)
    if loss_ratio is not None:
        stats["losses"].append(loss_ratio)


def _update_component_stats(
    stats: dict[str, Any],
    *,
    component_id: str,
    stage1: bool,
    loss_ratio: float | None,
    result_id: str,
) -> None:
    stats["component_id"] = component_id
    _update_lowering_stats(
        stats,
        stage1=stage1,
        loss_ratio=loss_ratio,
        result_id=result_id,
    )


def _finalize_lowering(lowering: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(stats.get("losses") or [])
    n = int(stats.get("n") or 0)
    n_stage1 = int(stats.get("n_stage1") or 0)
    return {
        "lowering": lowering,
        "n_components": n,
        "n_graphs": len(stats.get("result_ids") or ()),
        "n_stage1": n_stage1,
        "stage1_rate": n_stage1 / n if n else 0.0,
        "mean_loss_ratio": sum(losses) / len(losses) if losses else None,
        "best_loss_ratio": min(losses) if losses else None,
    }


def _finalize_component(lowering: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    out = _finalize_lowering(lowering, stats)
    out["component_id"] = str(stats.get("component_id") or "unknown")
    return out


def _summarize_candidate_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(path), "ready_count": 0, "by_lowering": []}
    ready = payload.get("ready_for_registration")
    rows = ready if isinstance(ready, list) else []
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "validated": 0, "scores": [], "components": []}
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        descriptor = row.get("component_descriptor")
        if not isinstance(descriptor, Mapping):
            descriptor = {}
        lowering = str(descriptor.get("lowering") or "unknown")
        bucket = stats[lowering]
        bucket["n"] += 1
        validation = row.get("validation")
        if isinstance(validation, Mapping) and validation.get("backward_passed"):
            bucket["validated"] += 1
        score = _safe_float(row.get("promotion_score"))
        if score is not None:
            bucket["scores"].append(score)
        bucket["components"].append(str(descriptor.get("component_id") or "unknown"))

    by_lowering = []
    for lowering, bucket in stats.items():
        scores = bucket["scores"]
        by_lowering.append(
            {
                "lowering": lowering,
                "ready_count": int(bucket["n"]),
                "backward_validated_count": int(bucket["validated"]),
                "mean_promotion_score": sum(scores) / len(scores) if scores else None,
                "max_promotion_score": max(scores) if scores else None,
                "example_components": bucket["components"][:5],
            }
        )
    by_lowering.sort(key=lambda item: (-item["ready_count"], item["lowering"]))
    return {"path": str(path), "ready_count": len(rows), "by_lowering": by_lowering}


def _selection_recommendations(
    historical_rows: list[Mapping[str, Any]],
    candidate_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    historical = {str(row["lowering"]): row for row in historical_rows}
    recommendations = []
    for row in candidate_rows:
        lowering = str(row["lowering"])
        hist = historical.get(lowering)
        n_components = int(hist.get("n_components") or 0) if hist else 0
        stage1_rate = float(hist.get("stage1_rate") or 0.0) if hist else None
        multiplier = _recommended_multiplier(
            lowering,
            n_components=n_components,
            stage1_rate=stage1_rate,
        )
        recommendations.append(
            {
                "lowering": lowering,
                "ready_count": int(row.get("ready_count") or 0),
                "historical_components": n_components,
                "historical_stage1_rate": stage1_rate,
                "recommended_selection_multiplier": multiplier,
                "confidence": "historical" if n_components >= 8 else "prior",
            }
        )
    recommendations.sort(key=lambda item: item["lowering"])
    return recommendations


def _recommended_multiplier(
    lowering: str,
    *,
    n_components: int,
    stage1_rate: float | None,
) -> float:
    if n_components >= 8 and stage1_rate is not None:
        if stage1_rate >= 0.75:
            return 1.10
        if stage1_rate <= 0.40:
            return 0.75
        return 1.0
    priors = {
        "trunk_sidecar_merge_v1": 1.10,
        "mixer_sidecar_restore_v1": 1.10,
        "router_lane_blend_v1": 0.75,
        "rmsnorm_chain_with_binary_skip": 1.0,
    }
    return priors.get(lowering, 1.0)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    output = args.output
    if not output:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output = str(DEFAULT_OUTPUT_DIR / f"dynamic_component_lowering_{stamp}.json")
    report = build_dynamic_component_lowering_report(
        db_path=args.db,
        candidate_path=args.candidates,
        output_path=output,
        limit=args.limit,
    )
    historical = report["historical"]
    candidates = report["candidate_artifact"]
    attempts = report["dynamic_attempts"]
    print(
        "dynamic_component_lowering_report "
        f"historical_rows={historical['dynamic_rows']} "
        f"ready={candidates['ready_count']} "
        f"attempts={attempts['total']} "
        f"rollbacks={attempts['rollback_count']} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
