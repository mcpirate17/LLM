"""Build runnable refinement queues from label-triage signals.

Usage:
    python -m research.tools.export_label_refinement_queue
    python -m research.tools.export_label_refinement_queue --top-ambiguous 12 --top-near-miss 24
    python -m research.tools.export_label_refinement_queue --triage-json research/reports/label_triage_20260403_100810.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from research.scientist.intelligence.predictor import load_runtime_ensemble
from research.synthesis.graph_features import (
    enrich_with_op_stats,
    extract_graph_features,
    load_op_stats,
)

_DEFAULT_DB = Path("research/lab_notebook.db")
_DEFAULT_REPORT_DIR = Path("research/reports")
_DEFAULT_PROFILING_DB = "research/profiling/component_profiles.db"


def _graph_signature(graph_json: Any) -> Optional[str]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(graph_json, dict):
        return None
    try:
        canonical = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _extract_ops_templates(graph_json: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph_json.get("nodes") or {}
    ops = sorted(
        {
            node.get("op_name", "")
            for node in nodes.values()
            if node.get("op_name", "") and node.get("op_name", "") != "input"
        }
    )
    metadata = graph_json.get("metadata") or {}
    return {
        "ops": ops,
        "templates_used": list(metadata.get("templates_used") or []),
        "motifs_used": list(metadata.get("motifs_used") or []),
    }


def _load_triage_rows(db_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    rows = conn.execute(
        """SELECT result_id, timestamp, graph_json, stage0_passed, stage05_passed,
                  stage1_passed, loss_ratio, wikitext_perplexity
           FROM program_results
           WHERE graph_json IS NOT NULL"""
    ).fetchall()
    conn.close()

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        graph_raw = row["graph_json"]
        try:
            graph_json = (
                json.loads(graph_raw) if isinstance(graph_raw, str) else graph_raw
            )
        except (json.JSONDecodeError, TypeError):
            continue
        signature = _graph_signature(graph_json)
        if signature is None:
            continue
        parsed.append(
            {
                "result_id": row["result_id"],
                "timestamp": row["timestamp"],
                "graph_json": graph_json,
                "signature": signature,
                "stage0_passed": bool(row["stage0_passed"]),
                "stage05_passed": bool(row["stage05_passed"]),
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
                "wikitext_perplexity": row["wikitext_perplexity"],
            }
        )
    return parsed


def build_report(
    db_path: Path,
    top_ambiguous: int,
    top_near_miss: int,
    profiling_db: str,
) -> Dict[str, Any]:
    rows = _load_triage_rows(db_path)
    op_stats_cache = load_op_stats(str(db_path))
    ensemble = load_runtime_ensemble(profiling_db=profiling_db)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["signature"]].append(row)

    ambiguous_groups: List[Dict[str, Any]] = []
    near_misses: List[Dict[str, Any]] = []

    for signature, items in groups.items():
        s1_vals = [1 if it["stage1_passed"] else 0 for it in items]
        s1_rate = sum(s1_vals) / len(s1_vals)
        sample_graph = items[0]["graph_json"]
        graph_meta = _extract_ops_templates(sample_graph)
        feats = extract_graph_features(sample_graph)
        ensemble_score = None
        gbm_score = None
        graph_score = None
        induction_score = None
        induction_auc = None
        if feats:
            enrich_with_op_stats(feats, graph_meta["ops"], preloaded=op_stats_cache)
            if ensemble.gbm is not None and ensemble.gbm.is_fitted():
                gbm_score = float(ensemble.gbm.predict_gate(feats))
            if ensemble.is_fitted():
                ensemble_score = float(
                    ensemble.predict_gate(sample_graph, graph_features=feats)
                )
                induction_score = float(
                    ensemble.predict_induction_learner_prob(
                        sample_graph, graph_features=feats
                    )
                )
                induction_auc = float(
                    ensemble.predict_induction_auc(sample_graph, graph_features=feats)
                )
        if ensemble.graph_pred is not None and ensemble.graph_pred.is_fitted():
            graph_score = float(ensemble.graph_pred.predict_gate(sample_graph))

        if len(items) > 1 and 0.0 < s1_rate < 1.0:
            ambiguous_groups.append(
                {
                    "signature": signature,
                    "n_rows": len(items),
                    "s1_rate": s1_rate,
                    "stage05_rate": sum(
                        1 if it["stage05_passed"] else 0 for it in items
                    )
                    / len(items),
                    "stage0_rate": sum(1 if it["stage0_passed"] else 0 for it in items)
                    / len(items),
                    "ensemble_score": ensemble_score,
                    "gbm_score": gbm_score,
                    "graph_score": graph_score,
                    "predicted_induction_learner": induction_score,
                    "predicted_induction_auc": induction_auc,
                    "ops": graph_meta["ops"],
                    "templates_used": graph_meta["templates_used"],
                    "motifs_used": graph_meta["motifs_used"],
                    "result_ids": [it["result_id"] for it in items],
                }
            )

        for item in items:
            if item["stage1_passed"] or not item["stage05_passed"]:
                continue
            near_misses.append(
                {
                    "result_id": item["result_id"],
                    "timestamp": item["timestamp"],
                    "signature": signature,
                    "ensemble_score": ensemble_score,
                    "gbm_score": gbm_score,
                    "graph_score": graph_score,
                    "predicted_induction_learner": induction_score,
                    "predicted_induction_auc": induction_auc,
                    "loss_ratio": item["loss_ratio"],
                    "wikitext_perplexity": item["wikitext_perplexity"],
                    "dup_group_size": len(items),
                    "dup_group_s1_rate": s1_rate if len(items) > 1 else 0.0,
                    "ops": graph_meta["ops"],
                    "templates_used": graph_meta["templates_used"],
                    "motifs_used": graph_meta["motifs_used"],
                }
            )

    ambiguous_groups.sort(
        key=lambda item: (
            -(item["ensemble_score"] if item["ensemble_score"] is not None else -1.0),
            -item["s1_rate"],
            -item["n_rows"],
        )
    )
    near_misses.sort(
        key=lambda item: (
            -(item["ensemble_score"] if item["ensemble_score"] is not None else -1.0),
            -(item["gbm_score"] if item["gbm_score"] is not None else -1.0),
            -(item["graph_score"] if item["graph_score"] is not None else -1.0),
            -item["dup_group_s1_rate"],
        )
    )

    return {
        "summary": {
            "n_rows": len(rows),
            "n_unique_graphs": len(groups),
            "n_ambiguous_groups": len(ambiguous_groups),
            "n_stage05_near_misses": len(near_misses),
            "ensemble_loaded": bool(ensemble.is_fitted()),
        },
        "ambiguous_groups": ambiguous_groups[:top_ambiguous],
        "near_misses": near_misses[:top_near_miss],
    }


def _load_triage_report(
    db_path: Path,
    triage_json: Optional[Path],
    top_ambiguous: int,
    top_near_miss: int,
    profiling_db: str,
) -> Dict[str, Any]:
    if triage_json is not None:
        return json.loads(triage_json.read_text())
    return build_report(
        db_path=db_path,
        top_ambiguous=top_ambiguous,
        top_near_miss=top_near_miss,
        profiling_db=profiling_db,
    )


def _fetch_result_rows(
    db_path: Path,
    result_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    ids = [str(rid).strip() for rid in result_ids if str(rid).strip()]
    if not ids:
        return {}
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT result_id, experiment_id, graph_fingerprint, stage0_passed,
                   stage05_passed, stage1_passed, loss_ratio, timestamp
            FROM program_results
            WHERE result_id IN ({placeholders})""",
        tuple(ids),
    ).fetchall()
    conn.close()
    return {str(row["result_id"]): dict(row) for row in rows}


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_priority(row: Dict[str, Any]) -> Tuple[float, ...]:
    loss_ratio = row.get("loss_ratio")
    loss_term = float(loss_ratio) if loss_ratio is not None else float("inf")
    return (
        1.0 if bool(row.get("stage1_passed")) else 0.0,
        1.0 if bool(row.get("stage05_passed")) else 0.0,
        -loss_term,
        _safe_float(row.get("timestamp"), 0.0),
    )


def _select_ambiguous_sources(
    group: Dict[str, Any],
    row_by_id: Dict[str, Dict[str, Any]],
    max_sources: int,
) -> List[Dict[str, Any]]:
    rows = [
        row_by_id[rid]
        for rid in group.get("result_ids", [])
        if rid in row_by_id and row_by_id[rid].get("graph_fingerprint")
    ]
    rows.sort(key=_source_priority, reverse=True)
    return rows[: max(1, int(max_sources))]


def _score_priority(item: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        _safe_float(item.get("ensemble_score")),
        _safe_float(item.get("gbm_score")),
        _safe_float(item.get("graph_score")),
        -_safe_float(item.get("dup_group_s1_rate"), 0.0),
        -_safe_float(item.get("loss_ratio"), float("inf")),
    )


def _dedupe_near_misses(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_signature: Dict[str, Dict[str, Any]] = {}
    for item in items:
        signature = str(item.get("signature") or "").strip()
        if not signature:
            continue
        current = best_by_signature.get(signature)
        if current is None or _score_priority(item) > _score_priority(current):
            best_by_signature[signature] = item
    deduped = list(best_by_signature.values())
    deduped.sort(key=_score_priority, reverse=True)
    return deduped


def _chunked(
    items: Sequence[Dict[str, Any]], batch_size: int
) -> List[List[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _build_payload(
    result_ids: Sequence[str],
    *,
    hypothesis: str,
    refine_intent: str,
    n_programs: int,
    refine_mutations_per_source: int,
    refine_pool_multiplier: int,
) -> Dict[str, Any]:
    return {
        "mode": "refine_fingerprint",
        "result_ids": list(result_ids),
        "n_programs": int(n_programs),
        "model_source": "fingerprint_refine",
        "refine_intent": str(refine_intent),
        "refine_mutations_per_source": int(refine_mutations_per_source),
        "refine_pool_multiplier": int(refine_pool_multiplier),
        "mutation_rate": 0.85,
        "preflight_override": True,
        "enforce_preflight": True,
        "exploratory": True,
        "hypothesis": hypothesis,
    }


def _build_ambiguous_batches(
    groups: Sequence[Dict[str, Any]],
    row_by_id: Dict[str, Dict[str, Any]],
    *,
    max_sources_per_group: int,
    batch_size: int,
    n_programs: int,
    refine_mutations_per_source: int,
    refine_pool_multiplier: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for group in groups:
        sources = _select_ambiguous_sources(group, row_by_id, max_sources_per_group)
        if not sources:
            continue
        selected.append(
            {
                "signature": group["signature"],
                "n_rows": group["n_rows"],
                "s1_rate": group["s1_rate"],
                "ensemble_score": group.get("ensemble_score"),
                "gbm_score": group.get("gbm_score"),
                "graph_score": group.get("graph_score"),
                "templates_used": group.get("templates_used", []),
                "motifs_used": group.get("motifs_used", []),
                "ops": group.get("ops", []),
                "source_result_ids": [row["result_id"] for row in sources],
                "source_graph_fingerprints": [
                    row["graph_fingerprint"] for row in sources
                ],
            }
        )

    batches: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(_chunked(selected, batch_size), start=1):
        result_ids: List[str] = []
        seen: set[str] = set()
        for item in chunk:
            for rid in item["source_result_ids"]:
                if rid not in seen:
                    seen.add(rid)
                    result_ids.append(rid)
        if not result_ids:
            continue
        hypothesis = (
            "Label-triage refinement for ambiguous exact-graph duplicate groups "
            f"(batch {idx})"
        )
        batches.append(
            {
                "batch_id": f"ambiguous_refine_{idx:02d}",
                "queue_type": "ambiguous_duplicate_groups",
                "n_groups": len(chunk),
                "n_sources": len(result_ids),
                "source_signatures": [item["signature"] for item in chunk],
                "sources": chunk,
                "launch_payload": _build_payload(
                    result_ids,
                    hypothesis=hypothesis,
                    refine_intent="recommended",
                    n_programs=n_programs,
                    refine_mutations_per_source=refine_mutations_per_source,
                    refine_pool_multiplier=refine_pool_multiplier,
                ),
            }
        )
    return batches


def _build_near_miss_batches(
    items: Sequence[Dict[str, Any]],
    row_by_id: Dict[str, Dict[str, Any]],
    *,
    batch_size: int,
    n_programs: int,
    refine_mutations_per_source: int,
    refine_pool_multiplier: int,
) -> List[Dict[str, Any]]:
    deduped = _dedupe_near_misses(items)
    selected: List[Dict[str, Any]] = []
    for item in deduped:
        rid = str(item.get("result_id") or "").strip()
        row = row_by_id.get(rid)
        if not row or not row.get("graph_fingerprint"):
            continue
        selected.append(
            {
                "result_id": rid,
                "signature": item["signature"],
                "ensemble_score": item.get("ensemble_score"),
                "gbm_score": item.get("gbm_score"),
                "graph_score": item.get("graph_score"),
                "loss_ratio": item.get("loss_ratio"),
                "dup_group_size": item.get("dup_group_size"),
                "dup_group_s1_rate": item.get("dup_group_s1_rate"),
                "graph_fingerprint": row["graph_fingerprint"],
                "templates_used": item.get("templates_used", []),
                "motifs_used": item.get("motifs_used", []),
                "ops": item.get("ops", []),
            }
        )

    batches: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(_chunked(selected, batch_size), start=1):
        result_ids = [item["result_id"] for item in chunk]
        if not result_ids:
            continue
        hypothesis = (
            "Label-triage refinement for high-confidence stage05 near misses "
            f"(batch {idx})"
        )
        batches.append(
            {
                "batch_id": f"near_miss_refine_{idx:02d}",
                "queue_type": "stage05_near_misses",
                "n_sources": len(result_ids),
                "source_signatures": [item["signature"] for item in chunk],
                "sources": chunk,
                "launch_payload": _build_payload(
                    result_ids,
                    hypothesis=hypothesis,
                    refine_intent="quality",
                    n_programs=n_programs,
                    refine_mutations_per_source=refine_mutations_per_source,
                    refine_pool_multiplier=refine_pool_multiplier,
                ),
            }
        )
    return batches


def build_refinement_queue(
    *,
    db_path: Path,
    profiling_db: str,
    triage_json: Optional[Path],
    top_ambiguous: int,
    top_near_miss: int,
    ambiguous_batch_size: int,
    near_miss_batch_size: int,
    ambiguous_programs: int,
    near_miss_programs: int,
    ambiguous_sources_per_group: int,
    refine_mutations_per_source: int,
    refine_pool_multiplier: int,
) -> Dict[str, Any]:
    triage = _load_triage_report(
        db_path=db_path,
        triage_json=triage_json,
        top_ambiguous=top_ambiguous,
        top_near_miss=top_near_miss,
        profiling_db=profiling_db,
    )
    all_result_ids = set()
    for group in triage.get("ambiguous_groups", []):
        all_result_ids.update(group.get("result_ids", []))
    for item in triage.get("near_misses", []):
        rid = item.get("result_id")
        if rid:
            all_result_ids.add(rid)

    row_by_id = _fetch_result_rows(db_path, sorted(all_result_ids))
    ambiguous_batches = _build_ambiguous_batches(
        triage.get("ambiguous_groups", []),
        row_by_id,
        max_sources_per_group=ambiguous_sources_per_group,
        batch_size=ambiguous_batch_size,
        n_programs=ambiguous_programs,
        refine_mutations_per_source=refine_mutations_per_source,
        refine_pool_multiplier=refine_pool_multiplier,
    )
    near_miss_batches = _build_near_miss_batches(
        triage.get("near_misses", []),
        row_by_id,
        batch_size=near_miss_batch_size,
        n_programs=near_miss_programs,
        refine_mutations_per_source=refine_mutations_per_source,
        refine_pool_multiplier=refine_pool_multiplier,
    )
    return {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "db_path": str(db_path),
        "triage_summary": triage.get("summary", {}),
        "queue_summary": {
            "ambiguous_batches": len(ambiguous_batches),
            "near_miss_batches": len(near_miss_batches),
            "ambiguous_sources": sum(batch["n_sources"] for batch in ambiguous_batches),
            "near_miss_sources": sum(batch["n_sources"] for batch in near_miss_batches),
        },
        "ambiguous_refinement_batches": ambiguous_batches,
        "near_miss_refinement_batches": near_miss_batches,
    }


def _format_md(queue: Dict[str, Any]) -> str:
    summary = queue.get("queue_summary", {})
    triage = queue.get("triage_summary", {})
    lines = [
        "# Label Refinement Queue",
        "",
        f"- Generated: {queue.get('generated_at')}",
        f"- Rows scanned: {triage.get('n_rows')}",
        f"- Unique exact graphs: {triage.get('n_unique_graphs')}",
        f"- Ambiguous groups in triage: {triage.get('n_ambiguous_groups')}",
        f"- Stage05 near misses in triage: {triage.get('n_stage05_near_misses')}",
        f"- Ambiguous refinement batches: {summary.get('ambiguous_batches')}",
        f"- Near-miss refinement batches: {summary.get('near_miss_batches')}",
        "",
        "## Ambiguous Duplicate Group Batches",
        "",
    ]
    ambiguous = queue.get("ambiguous_refinement_batches", [])
    if not ambiguous:
        lines.append("None.")
    else:
        for batch in ambiguous:
            lines.extend(
                [
                    f"### {batch['batch_id']}",
                    f"- sources: {batch['n_sources']}",
                    f"- groups: {batch['n_groups']}",
                    f"- signatures: {[sig[:12] for sig in batch['source_signatures']]}",
                    f"- result_ids: {batch['launch_payload']['result_ids']}",
                    f"- intent: {batch['launch_payload']['refine_intent']}",
                    f"- n_programs: {batch['launch_payload']['n_programs']}",
                    "",
                ]
            )
    lines.extend(["## Stage05 Near-Miss Batches", ""])
    near_miss = queue.get("near_miss_refinement_batches", [])
    if not near_miss:
        lines.append("None.")
    else:
        for batch in near_miss:
            lines.extend(
                [
                    f"### {batch['batch_id']}",
                    f"- sources: {batch['n_sources']}",
                    f"- signatures: {[sig[:12] for sig in batch['source_signatures']]}",
                    f"- result_ids: {batch['launch_payload']['result_ids']}",
                    f"- intent: {batch['launch_payload']['refine_intent']}",
                    f"- n_programs: {batch['launch_payload']['n_programs']}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export runnable refinement queues from label-triage signals"
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--profiling-db", type=str, default=_DEFAULT_PROFILING_DB)
    parser.add_argument("--triage-json", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--top-ambiguous", type=int, default=12)
    parser.add_argument("--top-near-miss", type=int, default=24)
    parser.add_argument("--ambiguous-batch-size", type=int, default=4)
    parser.add_argument("--near-miss-batch-size", type=int, default=8)
    parser.add_argument("--ambiguous-programs", type=int, default=48)
    parser.add_argument("--near-miss-programs", type=int, default=64)
    parser.add_argument("--ambiguous-sources-per-group", type=int, default=2)
    parser.add_argument("--refine-mutations-per-source", type=int, default=4)
    parser.add_argument("--refine-pool-multiplier", type=int, default=3)
    args = parser.parse_args()

    queue = build_refinement_queue(
        db_path=args.db,
        profiling_db=args.profiling_db,
        triage_json=args.triage_json,
        top_ambiguous=args.top_ambiguous,
        top_near_miss=args.top_near_miss,
        ambiguous_batch_size=args.ambiguous_batch_size,
        near_miss_batch_size=args.near_miss_batch_size,
        ambiguous_programs=args.ambiguous_programs,
        near_miss_programs=args.near_miss_programs,
        ambiguous_sources_per_group=args.ambiguous_sources_per_group,
        refine_mutations_per_source=args.refine_mutations_per_source,
        refine_pool_multiplier=args.refine_pool_multiplier,
    )

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(queue["generated_at"])
    json_path = report_dir / f"label_refinement_queue_{stamp}.json"
    md_path = report_dir / f"label_refinement_queue_{stamp}.md"

    json_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_format_md(queue))

    print(json_path)
    print(md_path)
    print(json.dumps(queue["queue_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
