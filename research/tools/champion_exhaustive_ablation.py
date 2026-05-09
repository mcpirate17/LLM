#!/usr/bin/env python3
"""Run exhaustive child ablations for one leaderboard champion.

This tool is intentionally separate from the dashboard-triggered wrapper flow:
it never rewrites the parent graph/result, and it records only child
counterfactuals plus causal evidence/provenance in the existing notebook tables.
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
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.causal_attribution import (  # noqa: E402
    CausalAblationCandidate,
    run_ablation_suite,
)
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value  # noqa: E402
from research.scientist.runner import ExperimentRunner  # noqa: E402
from research.scientist.runner._helpers_metrics import (  # noqa: E402
    _rebuild_graph_with_overrides,
)
from research.scientist.runner._types import RunConfig  # noqa: E402
from research.synthesis.primitives import PrimitiveOp, get_primitive, list_primitives  # noqa: E402
from research.synthesis.serializer import graph_from_json  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/runs.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
GOOGLE_BACKUP_ROOT = Path("/home/tim/GoogleDrive/Backups/LLM_Research")
DEFAULT_TARGET_RESULT_ID = "574271ca-f37"
LOGGER = logging.getLogger("champion_exhaustive_ablation")
SCAFFOLD_PRIORITY = {
    "identity": 0,
    "rmsnorm": 1,
    "layernorm": 2,
    "linear_proj": 3,
    "gelu": 4,
    "silu": 5,
    "relu": 6,
    "tanh": 7,
    "sigmoid": 8,
    "add": 9,
    "mul": 10,
}

_REQUIRED_PARENT_METRICS = (
    "wikitext_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "ar_legacy_auc",
)


def ensure_ablation_metric_completeness(
    nb: LabNotebook,
    *,
    parent_result_id: str,
) -> dict[str, int]:
    """Fail before ablation if the parent S1 row is missing required metrics.

    Several ablation entry points import this helper. Keeping it here gives
    them one shared parent-data quality check instead of each runner silently
    accepting partial parents.
    """
    row = nb.conn.execute(
        f"""SELECT stage1_passed, {", ".join(_REQUIRED_PARENT_METRICS)}
            FROM program_results
            WHERE result_id = ?""",
        (parent_result_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"ablation parent not found: {parent_result_id}")
    if not bool(row["stage1_passed"]):
        raise RuntimeError(f"ablation parent is not S1-passed: {parent_result_id}")
    missing = [name for name in _REQUIRED_PARENT_METRICS if row[name] is None]
    if missing:
        raise RuntimeError(
            f"ablation parent {parent_result_id} missing required S1 metrics: {missing}"
        )
    return {
        "required": len(_REQUIRED_PARENT_METRICS),
        "present": len(_REQUIRED_PARENT_METRICS) - len(missing),
        "missing": len(missing),
    }


@dataclasses.dataclass(slots=True)
class ParentProgram:
    result_id: str
    experiment_id: str
    fingerprint: str
    loss_ratio: float | None
    graph: Any
    config: RunConfig


@dataclasses.dataclass(slots=True)
class PlannedSuite:
    candidate: CausalAblationCandidate
    graphs: list[Any]
    child_meta_by_fingerprint: dict[str, dict[str, Any]]


def log(message: str) -> None:
    LOGGER.info(message)


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_parent(nb: LabNotebook, result_id: str) -> ParentProgram:
    row = nb.conn.execute(
        """SELECT pr.result_id, pr.experiment_id, pr.graph_fingerprint,
                  pr.graph_json, pr.loss_ratio, e.config_json
           FROM program_results pr
           LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id
           WHERE pr.result_id = ?""",
        (result_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"parent result not found: {result_id}")
    graph_json = resolve_graph_json_value(nb.conn, nb.db_path, row["graph_json"])
    if not graph_json.strip():
        raise SystemExit(f"parent result has no graph_json: {result_id}")
    graph = graph_from_json(graph_json)
    config_payload = json.loads(row["config_json"] or "{}")
    config = RunConfig.from_dict(
        config_payload if isinstance(config_payload, dict) else {}
    )
    return ParentProgram(
        result_id=str(row["result_id"]),
        experiment_id=str(row["experiment_id"]),
        fingerprint=str(row["graph_fingerprint"] or graph.fingerprint()),
        loss_ratio=_optional_float(row["loss_ratio"]),
        graph=graph,
        config=config,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prepare_config(
    parent_config: RunConfig,
    *,
    device: str | None,
    stage1_steps: int | None,
    max_ops_margin: int,
) -> RunConfig:
    config = parent_config.copy()
    config.mode = "single"
    config.continuous = False
    config.enable_causal_ablation = True
    config.auto_investigate = False
    config.auto_validate = False
    config.auto_scale_up = False
    config.max_ops = max(
        int(config.max_ops or 1), _parent_op_count_hint(config) + max_ops_margin
    )
    if device:
        config.device = device
    if stage1_steps is not None:
        config.stage1_steps = max(1, int(stage1_steps))
    return config


def _parent_op_count_hint(config: RunConfig) -> int:
    return max(1, int(getattr(config, "max_ops", 24) or 24))


def primitive_signature_map() -> dict[tuple[int, str], list[PrimitiveOp]]:
    grouped: dict[tuple[int, str], list[PrimitiveOp]] = {}
    for op in list_primitives():
        if op.name in {"input", "graph_input"}:
            continue
        grouped.setdefault((int(op.n_inputs), str(op.shape_rule)), []).append(op)
    for signature, ops in grouped.items():
        grouped[signature] = sorted(
            ops,
            key=lambda op: (
                op.name not in SCAFFOLD_PRIORITY,
                SCAFFOLD_PRIORITY.get(op.name, 1000),
                op.category.value,
                op.name,
            ),
        )
    return grouped


def replacement_ops(
    node_op: str,
    signature_map: Mapping[tuple[int, str], list[PrimitiveOp]],
    *,
    max_replacements_per_node: int,
) -> list[PrimitiveOp]:
    primitive = get_primitive(node_op)
    same_signature = [
        op
        for op in signature_map.get(
            (int(primitive.n_inputs), str(primitive.shape_rule)), []
        )
        if op.name != node_op
    ]
    if max_replacements_per_node > 0:
        return same_signature[:max_replacements_per_node]
    return same_signature


def build_node_suites(
    parent: ParentProgram,
    *,
    max_replacements_per_node: int,
    max_children: int,
) -> list[PlannedSuite]:
    signature_map = primitive_signature_map()
    suites: list[PlannedSuite] = []
    total_children = 0
    parent_fp = parent.graph.fingerprint()
    for node_id in parent.graph.topological_order():
        if max_children > 0 and total_children >= max_children:
            break
        node = parent.graph.nodes[node_id]
        if node.is_input:
            continue
        try:
            primitive = get_primitive(node.op_name)
        except (KeyError, ValueError) as exc:
            log(
                f"skip node={node_id} op={node.op_name}: primitive lookup failed: {exc}"
            )
            continue

        replacements = replacement_ops(
            node.op_name,
            signature_map,
            max_replacements_per_node=max_replacements_per_node,
        )
        graphs: list[Any] = []
        meta_by_fp: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for replacement in replacements:
            if max_children > 0 and total_children >= max_children:
                break
            rebuilt = _rebuild_graph_with_overrides(
                parent.graph,
                {
                    node_id: {
                        "op_name": replacement.name,
                        "config": dict(node.config or {}),
                    }
                },
            )
            if rebuilt is None:
                continue
            try:
                fp = rebuilt.fingerprint()
            except (RuntimeError, ValueError):
                continue
            if fp == parent_fp or fp in seen:
                continue
            seen.add(fp)
            graphs.append(rebuilt)
            total_children += 1
            meta_by_fp[fp] = {
                "campaign": "champion_exhaustive_node_ablation",
                "parent_result_id": parent.result_id,
                "parent_fingerprint": parent.fingerprint,
                "node_id": int(node_id),
                "original_op": node.op_name,
                "replacement_op": replacement.name,
                "original_category": primitive.category.value,
                "replacement_category": replacement.category.value,
                "signature": {
                    "n_inputs": int(primitive.n_inputs),
                    "shape_rule": str(primitive.shape_rule),
                },
                "input_ids": list(node.input_ids),
                "output_shape": getattr(
                    node.output_shape, "__dict__", str(node.output_shape)
                ),
            }

        if not graphs:
            continue
        rule_key = f"{node_id}:{node.op_name}"
        context = {
            "node_id": int(node_id),
            "original_op": node.op_name,
            "original_category": primitive.category.value,
            "signature": {
                "n_inputs": int(primitive.n_inputs),
                "shape_rule": str(primitive.shape_rule),
            },
            "children_planned": len(graphs),
            "replacement_ops": sorted(
                {meta["replacement_op"] for meta in meta_by_fp.values()}
            ),
        }
        candidate = CausalAblationCandidate(
            parent_experiment_id=parent.experiment_id,
            parent_result_id=parent.result_id,
            parent_fingerprint=parent.fingerprint,
            parent_loss_ratio=parent.loss_ratio,
            graph=parent.graph,
            rule_type="node_ablation",
            rule_key=rule_key,
            hypothesis=f"node_ablation:{rule_key}",
            context=context,
        )
        suites.append(
            PlannedSuite(
                candidate=candidate,
                graphs=graphs,
                child_meta_by_fingerprint=meta_by_fp,
            )
        )
    return suites


def existing_evidence_rules(
    nb: LabNotebook, parent_result_id: str
) -> set[tuple[str, str]]:
    rows = nb.conn.execute(
        """SELECT rule_type, rule_key
           FROM causal_rule_evidence
           WHERE parent_result_id = ?""",
        (parent_result_id,),
    ).fetchall()
    return {(str(row["rule_type"]), str(row["rule_key"])) for row in rows}


def run_suite(
    *,
    runner: ExperimentRunner,
    nb: LabNotebook,
    config: RunConfig,
    suite: PlannedSuite,
) -> dict[str, Any] | None:
    """Thin wrapper — defers to the canonical run_ablation_suite."""
    return run_ablation_suite(
        nb=nb,
        runner=runner,
        config=config,
        candidate=suite.candidate,
        graphs=suite.graphs,
        child_meta_by_fingerprint=suite.child_meta_by_fingerprint,
        campaign="champion_exhaustive_node_ablation",
    )


def make_backups(db_path: Path, *, dry_run: bool) -> dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_champion_exhaustive_ablation_{ts}"
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


def inventory_payload(
    parent: ParentProgram, suites: list[PlannedSuite]
) -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "parent": {
            "result_id": parent.result_id,
            "experiment_id": parent.experiment_id,
            "fingerprint": parent.fingerprint,
            "loss_ratio": parent.loss_ratio,
            "non_input_nodes": sum(
                1 for node in parent.graph.nodes.values() if not node.is_input
            ),
        },
        "suite_count": len(suites),
        "child_count": sum(len(suite.graphs) for suite in suites),
        "suites": [
            {
                "rule_type": suite.candidate.rule_type,
                "rule_key": suite.candidate.rule_key,
                "children": len(suite.graphs),
                "context": dict(suite.candidate.context),
            }
            for suite in suites
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-result-id", default=DEFAULT_TARGET_RESULT_ID)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument(
        "--max-replacements-per-node",
        type=int,
        default=0,
        help="0 means every same-signature replacement primitive.",
    )
    parser.add_argument("--max-children", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Only use for dry operational audits; DB-writing runs should back up.",
    )
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def configure_logging(log_file: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        Path(log_file) if log_file else RUNTIME_DIR / "champion_exhaustive_ablation.log"
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


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_file)
    db_path = Path(args.db)
    status_path = (
        RUNTIME_DIR
        / f"champion_{args.target_result_id}_exhaustive_ablation_status.json"
    )
    inventory_path = (
        RUNTIME_DIR / f"champion_{args.target_result_id}_ablation_inventory.json"
    )

    nb = LabNotebook(str(db_path), use_native=False)
    try:
        parent = load_parent(nb, args.target_result_id)
        config = prepare_config(
            parent.config,
            device=args.device,
            stage1_steps=args.stage1_steps,
            max_ops_margin=4,
        )
        config.max_ops = max(config.max_ops, len(parent.graph.nodes) + 4)
        suites = build_node_suites(
            parent,
            max_replacements_per_node=max(0, int(args.max_replacements_per_node)),
            max_children=max(0, int(args.max_children)),
        )
        inv = inventory_payload(parent, suites)
        json_dump(inventory_path, inv)
        log(
            "planned champion exhaustive ablation "
            f"parent={parent.result_id} fp={parent.fingerprint} "
            f"suites={inv['suite_count']} children={inv['child_count']} "
            f"inventory={inventory_path}"
        )
        if args.dry_run:
            log("dry run complete; no database writes or child training launched")
            return 0
        if inv["child_count"] <= 0:
            log("no child graphs generated")
            return 1
        backup_paths = {} if args.no_backup else make_backups(db_path, dry_run=False)
        if backup_paths:
            log(f"database backups created: {backup_paths}")

        existing = existing_evidence_rules(nb, parent.result_id)
        runner = ExperimentRunner(str(db_path))
        completed: list[dict[str, Any]] = []
        skipped_existing = 0
        started_at = time.time()
        for index, suite in enumerate(suites, start=1):
            rule_key = (suite.candidate.rule_type, suite.candidate.rule_key)
            if rule_key in existing and not args.rerun_existing:
                skipped_existing += 1
                log(f"skip existing evidence {rule_key[0]}:{rule_key[1]}")
                continue
            log(
                f"running suite {index}/{len(suites)} "
                f"{suite.candidate.rule_type}:{suite.candidate.rule_key} "
                f"children={len(suite.graphs)}"
            )
            result = run_suite(runner=runner, nb=nb, config=config, suite=suite)
            if result is not None:
                completed.append(result)
                log(
                    f"recorded evidence={result['evidence_id']} "
                    f"outcome={result['outcome']} effect={result['effect_size']} "
                    f"stage1={result['ablation_stage1_pass_count']}/{result['ablation_total']}"
                )
            json_dump(
                status_path,
                {
                    "parent_result_id": parent.result_id,
                    "parent_fingerprint": parent.fingerprint,
                    "started_at": started_at,
                    "updated_at": time.time(),
                    "inventory_path": str(inventory_path),
                    "log_path": str(log_path),
                    "planned_suites": len(suites),
                    "planned_children": inv["child_count"],
                    "completed_suites": len(completed),
                    "skipped_existing_suites": skipped_existing,
                    "latest_results": completed[-20:],
                },
            )
        log(
            "champion exhaustive ablation complete "
            f"completed_suites={len(completed)} skipped_existing={skipped_existing}"
        )
        return 0
    finally:
        try:
            nb.close()
        except Exception:  # noqa: BLE001 - close should never mask campaign result
            pass


if __name__ == "__main__":
    raise SystemExit(main())
