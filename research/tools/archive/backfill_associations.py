#!/usr/bin/env python3
"""Backfill experiment/fingerprint lineage associations across Research and Designer."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from research.tools.backfill_common import add_common_backfill_args, default_lab_notebook_path, ensure_db_exists

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "aria_designer") not in sys.path:
    sys.path.insert(0, str(ROOT / "aria_designer"))

from research.scientist.notebook import LabNotebook


@dataclass(slots=True)
class Stats:
    designer_workflows_seen: int = 0
    designer_lineage_rows_written: int = 0
    designer_program_rows_inserted: int = 0
    designer_program_rows_skipped: int = 0
    designer_parent_ids_backfilled: int = 0
    refinement_links_fixed: int = 0
    refinement_links_unresolved: int = 0
    refinement_placeholders_created: int = 0
    result_lineage_rows_backfilled: int = 0
    experiments_synced: int = 0
    experiments_without_programs_annotated: int = 0
    experiments_quarantined_invalid: int = 0


def _parse_iso_ts(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).timestamp()
    except Exception:
        return None


def _ensure_designer_experiment(nb: LabNotebook) -> None:
    row = nb.conn.execute(
        "SELECT 1 FROM experiments WHERE experiment_id = 'designer_edits'"
    ).fetchone()
    if row:
        return
    now = time.time()
    nb.conn.execute(
        """INSERT INTO experiments
           (experiment_id, timestamp, experiment_type, status, config_json, started_at, completed_at)
           VALUES ('designer_edits', ?, 'designer', 'completed', '{}', ?, ?)""",
        (now, now, now),
    )
    nb.conn.commit()


def _ensure_lineage_backfill_experiment(nb: LabNotebook) -> None:
    row = nb.conn.execute(
        "SELECT 1 FROM experiments WHERE experiment_id = 'historical_lineage_backfill'"
    ).fetchone()
    if row:
        return
    now = time.time()
    nb.conn.execute(
        """INSERT INTO experiments
           (experiment_id, timestamp, experiment_type, status, config_json, started_at, completed_at)
           VALUES ('historical_lineage_backfill', ?, 'lineage_backfill', 'completed', '{}', ?, ?)""",
        (now, now, now),
    )
    nb.conn.commit()


def _convert_workflow_to_graph_json(workflow: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    try:
        from runtime.bridge import workflow_to_graph as _w2g
        from research.synthesis.serializer import graph_to_json
    except Exception:
        return None, None

    try:
        model_dim = int((workflow.get("metadata") or {}).get("model_dim") or 256)
    except Exception:
        model_dim = 256

    try:
        try:
            out = _w2g(workflow, model_dim=model_dim, return_id_map=True)
        except TypeError:
            out = _w2g(workflow, model_dim=model_dim)
        graph = out[0] if isinstance(out, tuple) else out
        return graph.fingerprint(), graph_to_json(graph)
    except Exception:
        return None, None


def _backfill_designer(nb: LabNotebook, designer_db: Path, stats: Stats, dry_run: bool) -> None:
    if not designer_db.exists():
        return

    _ensure_designer_experiment(nb)
    conn = sqlite3.connect(str(designer_db))
    conn.row_factory = sqlite3.Row
    try:
        parent_rows = conn.execute(
            "SELECT id, version FROM workflows WHERE version > 1 AND (parent_id IS NULL OR TRIM(parent_id) = '')"
        ).fetchall()
        if not dry_run:
            for prow in parent_rows:
                wid = str(prow["id"])
                ver = int(prow["version"] or 1)
                conn.execute(
                    "UPDATE workflows SET parent_id = ? WHERE id = ?",
                    (f"{wid}@v{max(1, ver - 1)}", wid),
                )
            conn.commit()
        stats.designer_parent_ids_backfilled += len(parent_rows)

        rows = conn.execute(
            "SELECT id, version, graph_json, updated_at FROM workflows ORDER BY updated_at DESC"
        ).fetchall()
        for row in rows:
            stats.designer_workflows_seen += 1
            workflow_id = str(row["id"])
            version = int(row["version"] or 1)
            workflow_json = row["graph_json"]
            if not workflow_json:
                continue
            try:
                workflow = json.loads(workflow_json)
            except Exception:
                continue

            meta = workflow.get("metadata") or {}
            fp = str(meta.get("graph_fingerprint") or "").strip()
            if not fp:
                continue

            created_at = _parse_iso_ts(row["updated_at"])
            run_id = f"hist_{workflow_id}_v{version}"
            if not dry_run:
                nb.save_designer_run_lineage(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    workflow_version=version,
                    graph_fingerprint=fp,
                    status="saved",
                    source="aria_designer_backfill",
                    total_time_ms=0.0,
                    metrics={
                        "node_count": len(workflow.get("nodes") or []),
                        "edge_count": len(workflow.get("edges") or []),
                    },
                    payload={"workflow_id": workflow_id, "version": version},
                    created_at=created_at,
                )
            stats.designer_lineage_rows_written += 1

            exists = nb.conn.execute(
                "SELECT 1 FROM program_results WHERE graph_fingerprint = ? LIMIT 1",
                (fp,),
            ).fetchone()
            if exists:
                stats.designer_program_rows_skipped += 1
                continue

            canonical_fp, graph_json = _convert_workflow_to_graph_json(workflow)
            if not graph_json:
                stats.designer_program_rows_skipped += 1
                continue
            graph_fp = canonical_fp or fp

            loss_ratio = meta.get("loss_ratio")
            novelty_score = meta.get("novelty_score")
            try:
                loss_ratio = float(loss_ratio) if loss_ratio is not None else 1.0
            except Exception:
                loss_ratio = 1.0
            try:
                novelty_score = float(novelty_score) if novelty_score is not None else None
            except Exception:
                novelty_score = None

            if not dry_run:
                result_id = nb.record_program_result(
                    experiment_id="designer_edits",
                    graph_fingerprint=graph_fp,
                    graph_json=graph_json,
                    model_source="designer_edit",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=loss_ratio < 1.0,
                    loss_ratio=loss_ratio,
                    novelty_score=novelty_score,
                )
                if result_id:
                    stats.designer_program_rows_inserted += 1
            else:
                stats.designer_program_rows_inserted += 1
    finally:
        conn.close()


def _find_parent_result_id(
    nb: LabNotebook, parent_fp: str, child_ts: float
) -> Optional[str]:
    if not parent_fp:
        return None
    row = nb.conn.execute(
        """SELECT result_id FROM program_results
           WHERE graph_fingerprint = ? AND timestamp <= ?
           ORDER BY timestamp DESC
           LIMIT 1""",
        (parent_fp, child_ts),
    ).fetchone()
    if row:
        return str(row["result_id"])
    row = nb.conn.execute(
        """SELECT result_id FROM program_results
           WHERE graph_fingerprint = ?
           ORDER BY ABS(timestamp - ?) ASC
           LIMIT 1""",
        (parent_fp, child_ts),
    ).fetchone()
    return str(row["result_id"]) if row else None


def _repair_refinement_links(nb: LabNotebook, stats: Stats, dry_run: bool) -> None:
    rows = nb.conn.execute(
        """
        SELECT child.result_id, child.timestamp, child.graph_json
        FROM program_results child
        WHERE json_extract(child.graph_json, '$.metadata.refinement.source_result_id') IS NOT NULL
          AND TRIM(json_extract(child.graph_json, '$.metadata.refinement.source_result_id')) != ''
          AND NOT EXISTS (
                SELECT 1 FROM program_results src
                WHERE src.result_id = json_extract(child.graph_json, '$.metadata.refinement.source_result_id')
          )
        """
    ).fetchall()

    for row in rows:
        result_id = str(row["result_id"])
        raw_graph = row["graph_json"]
        if not raw_graph:
            stats.refinement_links_unresolved += 1
            continue
        try:
            graph = json.loads(raw_graph)
        except Exception:
            stats.refinement_links_unresolved += 1
            continue

        metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
        refinement = metadata.get("refinement") if isinstance(metadata.get("refinement"), dict) else {}
        lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}
        child_ts = float(row["timestamp"] or 0.0)

        parent_fp = str(lineage.get("parent") or "").strip()
        if not parent_fp:
            parent_fp = str(refinement.get("seed_fingerprint") or "").strip()
        resolved_parent = _find_parent_result_id(nb, parent_fp, child_ts)
        if not resolved_parent:
            source_result_id = str(refinement.get("source_result_id") or "").strip()
            if source_result_id and parent_fp:
                if not dry_run:
                    _ensure_lineage_backfill_experiment(nb)
                    placeholder_graph = {
                        "nodes": [],
                        "metadata": {
                            "lineage_backfill": True,
                            "source": "historical_association_repair",
                            "expected_child_result_id": result_id,
                        },
                    }
                    inserted = nb.record_program_result(
                        experiment_id="historical_lineage_backfill",
                        graph_fingerprint=parent_fp,
                        graph_json=json.dumps(placeholder_graph),
                        result_id=source_result_id,
                        model_source="lineage_backfill",
                    )
                    if inserted:
                        stats.refinement_placeholders_created += 1
                        continue
                else:
                    stats.refinement_placeholders_created += 1
                    continue
            stats.refinement_links_unresolved += 1
            continue

        refinement["source_result_id"] = resolved_parent
        metadata["refinement"] = refinement
        graph["metadata"] = metadata
        if not dry_run:
            nb.conn.execute(
                "UPDATE program_results SET graph_json = ? WHERE result_id = ?",
                (json.dumps(graph), result_id),
            )
        stats.refinement_links_fixed += 1

    if not dry_run:
        nb.conn.commit()


def _sync_experiment_summaries(nb: LabNotebook, stats: Stats, dry_run: bool) -> None:
    rows = nb.conn.execute("SELECT experiment_id FROM experiments").fetchall()
    for row in rows:
        exp_id = str(row["experiment_id"])
        if dry_run:
            stats.experiments_synced += 1
            continue
        nb.sync_experiment_counters(exp_id)
        stats.experiments_synced += 1


def _annotate_experiments_without_programs(nb: LabNotebook, stats: Stats, dry_run: bool) -> None:
    rows = nb.conn.execute(
        """
        SELECT e.experiment_id, e.status, e.results_json
        FROM experiments e
        LEFT JOIN program_results p ON p.experiment_id = e.experiment_id
        WHERE p.result_id IS NULL
        GROUP BY e.experiment_id
        """
    ).fetchall()

    for row in rows:
        exp_id = str(row["experiment_id"])
        existing = row["results_json"]
        payload: Dict[str, Any] = {}
        if existing is not None:
            try:
                decoded = nb._decompress(existing)
                if isinstance(decoded, dict):
                    payload = dict(decoded)
            except Exception:
                payload = {}

        integrity = payload.get("integrity") if isinstance(payload.get("integrity"), dict) else {}
        integrity["association_state"] = "missing_program_results"
        integrity["decision_policy"] = "exclude_or_downrank"
        integrity["annotated_by"] = "backfill_associations"
        integrity["annotated_at_unix"] = time.time()
        payload["integrity"] = integrity
        payload.setdefault("failure_reason", "missing_program_results")
        payload.setdefault(
            "integrity_note",
            "Experiment has no persisted program_results rows; exclude from ranking decisions.",
        )

        aria_summary = (
            "INTEGRITY FLAG: missing program_results associations; excluded from ranking decisions."
        )

        if not dry_run:
            new_status = "invalid" if str(row["status"] or "").strip().lower() == "completed" else row["status"]
            nb.conn.execute(
                """
                UPDATE experiments
                SET results_json = ?, aria_summary = COALESCE(aria_summary, ?), status = COALESCE(?, status)
                WHERE experiment_id = ?
                """,
                (nb._compress(payload), aria_summary, new_status, exp_id),
            )
            if new_status == "invalid":
                stats.experiments_quarantined_invalid += 1
        else:
            if str(row["status"] or "").strip().lower() == "completed":
                stats.experiments_quarantined_invalid += 1
        stats.experiments_without_programs_annotated += 1

    if not dry_run:
        nb.conn.commit()


def run_backfill(research_db: Path, designer_db: Path, dry_run: bool = False) -> Stats:
    stats = Stats()
    nb = LabNotebook(str(research_db))
    try:
        _backfill_designer(nb, designer_db, stats, dry_run)
        nb.flush_writes()
        _repair_refinement_links(nb, stats, dry_run)
        if not dry_run:
            stats.result_lineage_rows_backfilled = int(nb.rebuild_result_lineage_index() or 0)
        _sync_experiment_summaries(nb, stats, dry_run)
        _annotate_experiments_without_programs(nb, stats, dry_run)
        nb.flush_writes()
    finally:
        nb.close()
    return stats


def _default_research_db() -> Path:
    return Path(default_lab_notebook_path())


def _default_designer_db() -> Path:
    return ROOT / "aria_designer" / "api" / "aria_designer.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill experiment/fingerprint lineage associations.")
    parser.add_argument("--research-db", type=Path, default=Path(default_lab_notebook_path()))
    parser.add_argument("--designer-db", type=Path, default=_default_designer_db())
    add_common_backfill_args(parser, include_db=False, include_dry_run=True)
    args = parser.parse_args()

    stats = run_backfill(
        research_db=Path(ensure_db_exists(str(args.research_db.expanduser()))),
        designer_db=args.designer_db.expanduser(),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(stats.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
