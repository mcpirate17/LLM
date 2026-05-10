#!/usr/bin/env python3
"""Run focused single-op deletion ablations for top leaderboard graphs.

The exhaustive champion ablation tests many replacement operators.  This tool
is intentionally narrower: for each selected parent graph, delete one reachable
non-input node at a time by bypassing it to its first parent, reject duplicate
or non-compiling children, then run the surviving children through the canonical
ablation/evidence path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.causal_attribution import (  # noqa: E402
    CausalAblationCandidate,
    run_ablation_suite,
)
from research.scientist.causal_deletion_ablation import (  # noqa: E402
    DeletionChild,
    build_deletion_children as _shared_build_deletion_children,
    delete_node_by_bypass as _shared_delete_node_by_bypass,
    make_deletion_candidate,
)
from research.scientist.construction_priors import (  # noqa: E402
    assess_local_edit_prior,
    get_active_construction_prior,
)
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value  # noqa: E402
from research.scientist.runner import ExperimentRunner  # noqa: E402
from research.scientist.runner._types import RunConfig  # noqa: E402
from research.synthesis.graph import ComputationGraph  # noqa: E402
from research.synthesis.serializer import graph_from_json  # noqa: E402
from research.tools.champion_exhaustive_ablation import (  # noqa: E402
    ensure_ablation_metric_completeness,
)


DB_PATH = PROJECT_ROOT / "research/runs.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
GOOGLE_BACKUP_ROOT = Path("/home/tim/GoogleDrive/Backups/LLM_Research")
LOGGER = logging.getLogger("focused_op_deletion_ablation")


@dataclasses.dataclass(slots=True)
class ParentCandidate:
    result_id: str
    experiment_id: str
    fingerprint: str
    loss_ratio: float | None
    composite_score: float | None
    graph: ComputationGraph
    config: RunConfig


@dataclasses.dataclass(slots=True)
class ParentPlan:
    parent: ParentCandidate
    config: RunConfig
    children: list[DeletionChild]
    rejected: list[dict[str, Any]]
    metric_audit: dict[str, int]


def log(message: str) -> None:
    LOGGER.info(message)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def configure_logging(log_file: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        Path(log_file) if log_file else RUNTIME_DIR / "focused_op_deletion_ablation.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path, encoding="utf-8"),
        ],
    )
    return path


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _table_columns(nb: LabNotebook, table: str) -> set[str]:
    return {str(row["name"]) for row in nb.conn.execute(f"PRAGMA table_info({table})")}


def select_top_parents(
    nb: LabNotebook,
    *,
    top_k: int,
    include_references: bool,
    rank_offset: int = 0,
    parent_result_ids: set[str] | None = None,
) -> list[ParentCandidate]:
    leaderboard_cols = _table_columns(nb, "leaderboard")
    reference_clause = ""
    if not include_references and "is_reference" in leaderboard_cols:
        reference_clause = "AND COALESCE(l.is_reference, 0) = 0"
    parent_filter = ""
    params: list[Any] = []
    if parent_result_ids:
        placeholders = ",".join("?" for _ in parent_result_ids)
        parent_filter = f"AND pr.result_id IN ({placeholders})"
        params.extend(sorted(parent_result_ids))
    params.extend([max(1, int(top_k)), max(0, int(rank_offset))])
    rows = nb.conn.execute(
        f"""
        SELECT pr.result_id, pr.experiment_id, pr.graph_fingerprint, pr.graph_json,
               pr.loss_ratio, e.config_json, l.composite_score
        FROM leaderboard l
        JOIN program_results_compat pr ON pr.result_id = l.result_id
        LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE COALESCE(pr.stage1_passed, 0) = 1
          AND TRIM(COALESCE(pr.graph_json, '')) <> ''
          AND l.composite_score IS NOT NULL
          AND pr.wikitext_perplexity IS NOT NULL
          AND pr.hellaswag_acc IS NOT NULL
          AND pr.blimp_overall_accuracy IS NOT NULL
          AND pr.induction_screening_auc IS NOT NULL
          AND pr.binding_screening_auc IS NOT NULL
          AND pr.binding_screening_composite IS NOT NULL
          AND pr.ar_legacy_auc IS NOT NULL
          {reference_clause}
          {parent_filter}
        ORDER BY l.composite_score DESC, pr.loss_ratio ASC
        LIMIT ?
        OFFSET ?
        """,
        tuple(params),
    ).fetchall()
    parents: list[ParentCandidate] = []
    for row in rows:
        config_payload = json.loads(row["config_json"] or "{}")
        config = RunConfig.from_dict(
            config_payload if isinstance(config_payload, dict) else {}
        )
        graph_json = resolve_graph_json_value(nb.conn, nb.db_path, row["graph_json"])
        graph = graph_from_json(graph_json)
        parents.append(
            ParentCandidate(
                result_id=str(row["result_id"]),
                experiment_id=str(row["experiment_id"]),
                fingerprint=str(row["graph_fingerprint"] or graph.fingerprint()),
                loss_ratio=_optional_float(row["loss_ratio"]),
                composite_score=_optional_float(row["composite_score"]),
                graph=graph,
                config=config,
            )
        )
    return parents


def prepare_config(parent_config: RunConfig, *, device: str | None) -> RunConfig:
    config = parent_config.copy()
    config.mode = "single"
    config.continuous = False
    config.enable_causal_ablation = True
    config.auto_investigate = False
    config.auto_validate = False
    config.auto_scale_up = False
    if device:
        config.device = device
    return config


# Re-exported from the shared module so existing call sites keep working.
delete_node_by_bypass = _shared_delete_node_by_bypass


def build_deletion_children(
    parent: ParentCandidate,
    config: RunConfig,
    *,
    global_seen: set[str],
    rule_keys: set[str] | None = None,
) -> tuple[list[DeletionChild], list[dict[str, Any]]]:
    """Driver wrapper — preserves the original full-sweep semantics.

    The driver does NOT skip SCAFFOLD_OPS (its purpose is exhaustive op
    deletion, including load-bearing scaffolding to confirm what is and
    isn't necessary).  The investigation auto-ablation hook calls the
    shared helper with ``skip_scaffold=True`` to keep its compute budget
    finite.
    """
    children, rejected = _shared_build_deletion_children(
        graph=parent.graph,
        parent_fingerprint=parent.fingerprint,
        max_ops=config.max_ops,
        max_depth=config.max_depth,
        min_splits=config.min_splits,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        global_seen=global_seen,
        skip_scaffold=False,
        rule_keys=rule_keys,
    )
    # The shared module annotates ``meta`` with parent_fingerprint; the
    # driver historically also tagged parent_result_id on every rejection
    # for the audit JSON.  Patch it in here so the on-disk format is
    # unchanged.
    for meta in rejected:
        meta.setdefault("parent_result_id", parent.result_id)
    return children, rejected


def make_candidate(
    parent: ParentCandidate,
    child: DeletionChild,
    *,
    active_prior: dict[str, Any] | None = None,
) -> CausalAblationCandidate:
    return make_deletion_candidate(
        child=child,
        parent_experiment_id=parent.experiment_id,
        parent_result_id=parent.result_id,
        parent_fingerprint=parent.fingerprint,
        parent_loss_ratio=parent.loss_ratio,
        parent_graph=parent.graph,
        parent_composite_score=parent.composite_score,
        active_prior=active_prior,
    )


def child_status(
    child: DeletionChild,
    *,
    active_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_key = f"{child.node_id}:{child.op_name}"
    return {
        "node_id": child.node_id,
        "op_name": child.op_name,
        "bypass_input_id": child.bypass_input_id,
        "fingerprint": child.fingerprint,
        "pruned_nodes": child.pruned_nodes,
        "prior_assessment": assess_local_edit_prior(
            active_prior,
            rule_type="node_delete",
            rule_key=rule_key,
        ),
    }


def child_meta(
    parent: ParentCandidate,
    child: DeletionChild,
    *,
    active_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_key = f"{child.node_id}:{child.op_name}"
    return {
        "campaign": "focused_op_deletion_ablation",
        "parent_result_id": parent.result_id,
        "parent_fingerprint": parent.fingerprint,
        "node_id": child.node_id,
        "deleted_op": child.op_name,
        "bypass_input_id": child.bypass_input_id,
        "pruned_nodes": child.pruned_nodes,
        "prior_assessment": assess_local_edit_prior(
            active_prior,
            rule_type="node_delete",
            rule_key=rule_key,
        ),
    }


def make_backups(db_path: Path, *, dry_run: bool) -> dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_focused_op_deletion_ablation_{ts}"
    local_dir = PROJECT_ROOT / "research/db_backups" / backup_name
    google_dir = GOOGLE_BACKUP_ROOT / backup_name
    local_target = local_dir / db_path.name
    google_target = google_dir / db_path.name
    if dry_run:
        return {"local": str(local_target), "google_drive": str(google_target)}
    local_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, local_target)
    google_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, google_target)
    return {"local": str(local_target), "google_drive": str(google_target)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument(
        "--parent-result-id",
        action="append",
        default=[],
        help="Restrict planning/runs to selected parent result_id values.",
    )
    parser.add_argument(
        "--rule-key",
        action="append",
        default=[],
        help="Restrict deletion planning/runs to selected node_id:op_name keys.",
    )
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_file)
    db_path = Path(args.db)
    status_path = RUNTIME_DIR / "focused_op_deletion_ablation_status.json"
    nb = LabNotebook(str(db_path), use_native=False)
    try:
        active_prior = get_active_construction_prior(nb)
        parents = select_top_parents(
            nb,
            top_k=max(1, int(args.top_k)),
            include_references=bool(args.include_references),
            rank_offset=max(0, int(args.rank_offset)),
            parent_result_ids=set(args.parent_result_id or []) or None,
        )
        if not parents:
            raise SystemExit("no eligible parent graphs found")
        global_seen: set[str] = set()
        execution_plans: list[ParentPlan] = []
        total_children = 0
        total_rejected = 0
        for parent in parents:
            metric_audit = ensure_ablation_metric_completeness(
                nb, parent_result_id=parent.result_id
            )
            config = prepare_config(parent.config, device=args.device)
            children, rejected = build_deletion_children(
                parent,
                config,
                global_seen=global_seen,
                rule_keys=set(args.rule_key or []) or None,
            )
            total_children += len(children)
            total_rejected += len(rejected)
            execution_plans.append(
                ParentPlan(
                    parent=parent,
                    config=config,
                    children=children,
                    rejected=rejected,
                    metric_audit=metric_audit,
                )
            )

        plan_payload = [
            {
                "parent": {
                    "result_id": plan.parent.result_id,
                    "experiment_id": plan.parent.experiment_id,
                    "fingerprint": plan.parent.fingerprint,
                    "loss_ratio": plan.parent.loss_ratio,
                    "composite_score": plan.parent.composite_score,
                },
                "children": [
                    child_status(child, active_prior=active_prior)
                    for child in plan.children
                ],
                "rejected": plan.rejected,
                "metric_audit": plan.metric_audit,
            }
            for plan in execution_plans
        ]

        status = {
            "created_at": time.time(),
            "log_path": str(log_path),
            "top_k": max(1, int(args.top_k)),
            "rank_offset": max(0, int(args.rank_offset)),
            "parent_result_ids": list(args.parent_result_id or []),
            "rule_keys": list(args.rule_key or []),
            "include_references": bool(args.include_references),
            "parent_count": len(parents),
            "planned_children": total_children,
            "rejected_children": total_rejected,
            "plan": plan_payload,
            "dry_run": bool(args.dry_run),
            "audit_only": bool(args.audit_only),
        }
        json_dump(status_path, status)
        log(
            "planned focused op deletion ablation "
            f"parents={len(parents)} children={total_children} "
            f"rejected={total_rejected} status={status_path}"
        )
        if args.dry_run or args.audit_only:
            log("audit/dry run complete; no database writes or child training launched")
            return 0
        if total_children <= 0:
            log("no compile-valid deletion children to run")
            return 1
        backup_paths = {} if args.no_backup else make_backups(db_path, dry_run=False)
        if backup_paths:
            log(f"database backups created: {backup_paths}")

        runner = ExperimentRunner(str(db_path))
        results: list[dict[str, Any]] = []
        for parent_plan in execution_plans:
            for child in parent_plan.children:
                result = run_ablation_suite(
                    nb=nb,
                    runner=runner,
                    config=parent_plan.config,
                    candidate=make_candidate(
                        parent_plan.parent,
                        child,
                        active_prior=active_prior,
                    ),
                    graphs=[child.graph],
                    child_meta_by_fingerprint={
                        child.fingerprint: child_meta(
                            parent_plan.parent,
                            child,
                            active_prior=active_prior,
                        )
                    },
                    campaign="focused_op_deletion_ablation",
                    extra_evidence_fields={
                        "focused_delete": child_status(
                            child,
                            active_prior=active_prior,
                        )
                    },
                    exclude_failed_observations=True,
                )
                if result is not None:
                    results.append(result)
                    status["latest_results"] = results[-20:]
                    status["completed_children"] = len(results)
                    status["updated_at"] = time.time()
                    json_dump(status_path, status)
                    log(
                        f"recorded node_delete {parent_plan.parent.result_id} "
                        f"{child.node_id}:{child.op_name} outcome={result['outcome']} "
                        f"effect={result['effect_size']}"
                    )
        log(f"focused op deletion ablation complete results={len(results)}")
        return 0
    finally:
        try:
            nb.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
