#!/usr/bin/env python3
"""Run proper component-knockout ablations for the top leaderboard graphs.

This intentionally does not use the loss-only causal ablation summary path.
Each accepted child graph is recorded as a normal ``program_results`` row:

* S1 phase: full stacked screening replay at the configured S1 budget.
* Investigation phase: 2500-step investigation follow-up with v2 probes.

The causal evidence rows are metadata/index rows only; the full metric payloads
live in ``program_results``.
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

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.orchestrator.executor import JobResult  # noqa: E402
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.runner import ExperimentRunner  # noqa: E402
from research.scientist.runner._types import RunConfig  # noqa: E402
from research.scientist.runner.execution_screening import (  # noqa: E402
    INITIAL_LOSS_THRESHOLD,
    _record_screening_failure,
)
from research.scientist.runner._helpers import clear_gpu_memory  # noqa: E402
from research.scientist.shared_utils import resolve_device  # noqa: E402
from research.synthesis.compiler import compile_model  # noqa: E402
from research.synthesis.serializer import graph_to_json  # noqa: E402
from research.tools.champion_exhaustive_ablation import (  # noqa: E402
    ensure_ablation_metric_completeness,
)
from research.tools.focused_op_deletion_ablation import (  # noqa: E402
    DeletionChild,
    ParentCandidate,
    delete_node_by_bypass,
    json_dump,
    prepare_config,
    select_top_parents,
)
from research.training.loss_ops import next_token_cross_entropy  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/lab_notebook.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
GOOGLE_BACKUP_ROOT = Path("/home/tim/GoogleDrive/Backups/LLM_Research")
LOGGER = logging.getLogger("proper_component_knockout_ablation")

S1_METRIC_COLUMNS = (
    "loss_ratio",
    "wikitext_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "ar_auc",
    "fp_jacobian_erf_density",
    "fp_icld_velocity",
    "fp_logit_margin_delta",
)

INVESTIGATION_METRIC_COLUMNS = (
    "loss_ratio",
    "wikitext_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "ar_auc",
    "induction_v2_investigation_auc",
    "induction_v2_investigation_max_gap_acc",
    "induction_v2_investigation_status",
    "induction_v2_investigation_protocol_version",
    "binding_v2_investigation_auc",
    "binding_v2_investigation_max_distance_acc",
    "binding_v2_investigation_status",
    "binding_v2_investigation_protocol_version",
    "fp_jacobian_erf_density",
    "fp_icld_velocity",
    "fp_logit_margin_delta",
)


@dataclasses.dataclass(slots=True)
class ParentPlan:
    parent: ParentCandidate
    config: RunConfig
    children: list[DeletionChild]
    rejected: list[dict[str, Any]]
    metric_audit: dict[str, int]


def configure_logging(log_file: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        Path(log_file)
        if log_file
        else RUNTIME_DIR / "proper_component_knockout_ablation.log"
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


def log(message: str) -> None:
    LOGGER.info(message)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_row_by_experiment_and_fingerprint(
    nb: LabNotebook, *, experiment_id: str, fingerprint: str
) -> dict[str, Any] | None:
    row = nb.conn.execute(
        """SELECT * FROM program_results
           WHERE experiment_id = ? AND graph_fingerprint = ?
           ORDER BY timestamp DESC LIMIT 1""",
        (experiment_id, fingerprint),
    ).fetchone()
    return dict(row) if row else None


def _row_by_result_id(nb: LabNotebook, result_id: str) -> dict[str, Any] | None:
    row = nb.conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    return dict(row) if row else None


def _existing_fingerprints(nb: LabNotebook, fingerprints: list[str]) -> set[str]:
    if not fingerprints:
        return set()
    placeholders = ",".join("?" for _ in fingerprints)
    rows = nb.conn.execute(
        f"SELECT DISTINCT graph_fingerprint FROM program_results "
        f"WHERE graph_fingerprint IN ({placeholders})",
        tuple(fingerprints),
    ).fetchall()
    return {str(row["graph_fingerprint"]) for row in rows}


def _existing_knockout_s1_fingerprints(
    nb: LabNotebook, fingerprints: list[str]
) -> set[str]:
    if not fingerprints:
        return set()
    placeholders = ",".join("?" for _ in fingerprints)
    rows = nb.conn.execute(
        f"""SELECT DISTINCT graph_fingerprint FROM program_results
            WHERE graph_fingerprint IN ({placeholders})
              AND intentional_rerun_reason = 'proper_component_knockout_s1'""",
        tuple(fingerprints),
    ).fetchall()
    return {str(row["graph_fingerprint"]) for row in rows}


def _candidate_bypass_children(
    parent: ParentCandidate,
    config: RunConfig,
    node_id: int,
    *,
    global_seen: set[str],
) -> tuple[DeletionChild | None, dict[str, Any]]:
    node = parent.graph.nodes[node_id]
    attempts: list[dict[str, Any]] = []
    original_inputs = list(node.input_ids)
    if not original_inputs:
        return None, {
            "reason": "no_bypass_input",
            "node_id": node_id,
            "op_name": node.op_name,
        }

    for bypass_input_id in original_inputs:
        trial_graph = parent.graph.copy()
        trial_node = trial_graph.nodes[node_id]
        trial_node.input_ids = [
            int(bypass_input_id),
            *[int(i) for i in original_inputs if int(i) != int(bypass_input_id)],
        ]
        child, meta = delete_node_by_bypass(trial_graph, node_id)
        meta.update(
            {
                "parent_result_id": parent.result_id,
                "parent_fingerprint": parent.fingerprint,
                "node_id": node_id,
                "op_name": node.op_name,
                "attempted_bypass_input_id": int(bypass_input_id),
            }
        )
        if child is None:
            attempts.append(meta)
            continue
        try:
            compile_model(
                [child] * int(config.n_layers),
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            fingerprint = child.fingerprint()
        except (RuntimeError, ValueError, TypeError) as exc:
            meta["reason"] = "compile_failed"
            meta["error"] = str(exc)
            attempts.append(meta)
            continue
        if fingerprint == parent.fingerprint:
            meta["reason"] = "duplicate_parent_fingerprint"
            meta["fingerprint"] = fingerprint
            attempts.append(meta)
            continue
        if fingerprint in global_seen:
            meta["reason"] = "duplicate_planned_fingerprint"
            meta["fingerprint"] = fingerprint
            attempts.append(meta)
            continue
        global_seen.add(fingerprint)
        return (
            DeletionChild(
                graph=child,
                node_id=node_id,
                op_name=node.op_name,
                bypass_input_id=int(meta["bypass_input_id"]),
                fingerprint=fingerprint,
                pruned_nodes=int(meta.get("pruned_nodes") or 0),
            ),
            meta,
        )
    return None, {
        "reason": "no_valid_bypass",
        "node_id": node_id,
        "op_name": node.op_name,
        "attempts": attempts,
    }


def build_component_children(
    nb: LabNotebook,
    parent: ParentCandidate,
    config: RunConfig,
    *,
    global_seen: set[str],
    allow_existing_knockout_s1: bool = False,
) -> tuple[list[DeletionChild], list[dict[str, Any]]]:
    children: list[DeletionChild] = []
    rejected: list[dict[str, Any]] = []
    reachable = parent.graph.get_reachable_nodes()
    for node_id in parent.graph.topological_order():
        node = parent.graph.nodes[node_id]
        if node.is_input or node_id not in reachable:
            continue
        child, meta = _candidate_bypass_children(
            parent,
            config,
            node_id,
            global_seen=global_seen,
        )
        if child is None:
            rejected.append(meta)
            continue
        children.append(child)

    existing = _existing_fingerprints(nb, [child.fingerprint for child in children])
    allowed_existing = (
        _existing_knockout_s1_fingerprints(
            nb, [child.fingerprint for child in children]
        )
        if allow_existing_knockout_s1
        else set()
    )
    if existing:
        kept: list[DeletionChild] = []
        for child in children:
            if (
                child.fingerprint in existing
                and child.fingerprint not in allowed_existing
            ):
                rejected.append(
                    {
                        "reason": "duplicate_existing_fingerprint",
                        "node_id": child.node_id,
                        "op_name": child.op_name,
                        "fingerprint": child.fingerprint,
                    }
                )
            else:
                kept.append(child)
        children = kept
    return children, rejected


def child_status(child: DeletionChild) -> dict[str, Any]:
    return {
        "node_id": child.node_id,
        "op_name": child.op_name,
        "bypass_input_id": child.bypass_input_id,
        "fingerprint": child.fingerprint,
        "pruned_nodes": child.pruned_nodes,
    }


def make_backups(db_path: Path, *, dry_run: bool) -> dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_proper_component_knockout_ablation_{ts}"
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


def _metric_snapshot(
    row: dict[str, Any] | None, columns: tuple[str, ...]
) -> dict[str, Any]:
    if row is None:
        return {}
    return {column: row.get(column) for column in columns}


def _metric_deltas(
    parent: dict[str, Any] | None,
    child: dict[str, Any] | None,
    columns: tuple[str, ...],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for column in columns:
        p = _optional_float(parent.get(column) if parent else None)
        c = _optional_float(child.get(column) if child else None)
        deltas[column] = None if p is None or c is None else c - p
    return deltas


def _missing_columns(row: dict[str, Any], columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if row.get(column) is None]


def _assert_evidence_row_complete(
    *,
    child: DeletionChild,
    child_row: dict[str, Any] | None,
    phase: str,
) -> None:
    if child_row is None:
        raise RuntimeError(
            "Refusing to record "
            f"{phase} evidence for {child.node_id}:{child.op_name}: "
            "no durable program_results row exists."
        )
    if not child_row.get("stage1_passed"):
        return
    columns = S1_METRIC_COLUMNS if phase == "s1" else INVESTIGATION_METRIC_COLUMNS
    missing = _missing_columns(child_row, columns)
    if missing:
        raise RuntimeError(
            "Refusing to record "
            f"{phase} evidence for {child.node_id}:{child.op_name} "
            f"({child.fingerprint[:12]}): passed row is missing {missing}."
        )


def record_phase_evidence(
    nb: LabNotebook,
    *,
    parent: ParentCandidate,
    child: DeletionChild,
    child_row: dict[str, Any] | None,
    phase: str,
    phase_experiment_id: str,
    parent_row: dict[str, Any] | None,
) -> str:
    _assert_evidence_row_complete(child=child, child_row=child_row, phase=phase)
    columns = S1_METRIC_COLUMNS if phase == "s1" else INVESTIGATION_METRIC_COLUMNS
    payload = {
        "campaign": "proper_component_knockout_ablation",
        "phase": phase,
        "phase_experiment_id": phase_experiment_id,
        "parent_result_id": parent.result_id,
        "parent_experiment_id": parent.experiment_id,
        "parent_fingerprint": parent.fingerprint,
        "child": child_status(child),
        "child_result_id": child_row.get("result_id") if child_row else None,
        "child_stage1_passed": bool(child_row.get("stage1_passed"))
        if child_row
        else False,
        "parent_metrics": _metric_snapshot(parent_row, columns),
        "child_metrics": _metric_snapshot(child_row, columns),
        "metric_deltas_child_minus_parent": _metric_deltas(
            parent_row, child_row, columns
        ),
    }
    original_loss = _optional_float(
        parent_row.get("loss_ratio") if parent_row else None
    )
    child_loss = _optional_float(child_row.get("loss_ratio") if child_row else None)
    effect = (
        None
        if original_loss is None or child_loss is None
        else child_loss - original_loss
    )
    evidence = {
        "parent_experiment_id": parent.experiment_id,
        "parent_result_id": parent.result_id,
        "parent_fingerprint": parent.fingerprint,
        "ablation_experiment_id": phase_experiment_id,
        "rule_type": f"node_delete_{phase}",
        "rule_key": f"{child.node_id}:{child.op_name}",
        "rule_context": json.dumps(child_status(child), sort_keys=True),
        "original_loss_ratio": original_loss,
        "ablation_best_loss_ratio": child_loss,
        "effect_size": effect,
        "original_stage1_passed": 1,
        "ablation_stage1_pass_count": 1
        if child_row and child_row.get("stage1_passed")
        else 0,
        "ablation_total": 1 if child_row else 0,
        "outcome": f"measured_{phase}",
        "confidence": 0.0,
        "evidence_json": json.dumps(payload, sort_keys=True),
    }
    evidence_id = nb.record_causal_rule_evidence(evidence)
    nb.flush_writes()
    return evidence_id


def _record_latest_result(
    nb: LabNotebook,
    *,
    experiment_id: str,
    child: DeletionChild,
) -> dict[str, Any] | None:
    nb.flush_writes()
    return _latest_row_by_experiment_and_fingerprint(
        nb, experiment_id=experiment_id, fingerprint=child.fingerprint
    )


def _collect_latest_rows_for_fingerprints(
    nb: LabNotebook,
    *,
    experiment_id: str,
    fingerprints: list[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for fingerprint in fingerprints:
        row = _latest_row_by_experiment_and_fingerprint(
            nb,
            experiment_id=experiment_id,
            fingerprint=fingerprint,
        )
        if row:
            rows[fingerprint] = row
    return rows


def _wait_for_complete_phase_rows(
    nb: LabNotebook,
    *,
    experiment_id: str,
    children: list[DeletionChild],
    phase: str,
    timeout_s: float = 1800.0,
    poll_s: float = 5.0,
) -> dict[str, dict[str, Any]]:
    fingerprints = [child.fingerprint for child in children]
    deadline = time.time() + max(0.0, float(timeout_s))
    last_status = ""
    while True:
        nb.flush_writes()
        rows = _collect_latest_rows_for_fingerprints(
            nb,
            experiment_id=experiment_id,
            fingerprints=fingerprints,
        )
        missing_fps = [fp for fp in fingerprints if fp not in rows]
        incomplete_passed: dict[str, list[str]] = {}
        for child in children:
            row = rows.get(child.fingerprint)
            if not row or not row.get("stage1_passed"):
                continue
            columns = (
                S1_METRIC_COLUMNS if phase == "s1" else INVESTIGATION_METRIC_COLUMNS
            )
            missing = _missing_columns(row, columns)
            if missing:
                incomplete_passed[child.fingerprint] = missing
        if not missing_fps and not incomplete_passed:
            return rows

        status = (
            f"{phase} rows {len(rows)}/{len(fingerprints)} "
            f"missing={len(missing_fps)} incomplete_passed={len(incomplete_passed)}"
        )
        if status != last_status:
            log(f"waiting for durable {status}")
            last_status = status
        if time.time() >= deadline:
            missing_detail = []
            for child in children:
                if child.fingerprint in missing_fps:
                    missing_detail.append(
                        f"{child.node_id}:{child.op_name}:{child.fingerprint[:12]}"
                    )
            incomplete_detail = {
                fp[:12]: missing for fp, missing in incomplete_passed.items()
            }
            raise RuntimeError(
                f"Timed out waiting for complete {phase} rows for {experiment_id}: "
                f"missing={missing_detail}, incomplete_passed={incomplete_detail}"
            )
        time.sleep(max(0.5, float(poll_s)))


def run_s1_phase(
    *,
    nb: LabNotebook,
    runner: ExperimentRunner,
    parent_plan: ParentPlan,
    status: dict[str, Any],
    status_path: Path,
) -> tuple[str, dict[str, dict[str, Any]]]:
    config = parent_plan.config.copy()
    config.model_source = "ablation"
    config.auto_investigate = False
    config.auto_validate = False
    config.auto_scale_up = False
    config.enable_causal_ablation = False
    config.gbm_prescreener_enabled = False
    runner._ensure_math_spaces()
    config, _ = runner.prescreen_run_config(config, mode="single", auto_harden=True)
    exp_config = config.to_dict()
    exp_config["campaign"] = "proper_component_knockout_ablation"
    exp_config["phase"] = "s1"
    exp_config["parent_result_id"] = parent_plan.parent.result_id
    exp_config["child_fingerprints"] = [
        child.fingerprint for child in parent_plan.children
    ]
    exp_id = nb.start_experiment(
        "ablation",
        exp_config,
        hypothesis=(
            "Proper component knockout S1: delete one reachable op at a time "
            f"from {parent_plan.parent.result_id} and run the full S1 screen."
        ),
    )
    dev = resolve_device(config.device)
    dev_str = str(dev)
    results: dict[str, Any] = {
        "total": len(parent_plan.children),
        "stage0_passed": 0,
        "stage05_passed": 0,
        "rapid_screening_killed": 0,
        "rapid_screening_kill_reasons": {},
        "stage1_passed": 0,
        "best_loss_ratio": None,
        "best_novelty_score": None,
        "survivors": [],
        "funnel_counts": {
            "raw_generated": len(parent_plan.children),
            "post_batch_dedup": len(parent_plan.children),
            "judgment_filtered": 0,
            "post_judgment": len(parent_plan.children),
            "screening_considered": len(parent_plan.children),
            "dropped_stage0": 0,
            "dropped_stage05": 0,
            "dropped_s075_high_init": 0,
            "rapid_screen_attempted": 0,
            "dropped_rapid_screening": 0,
            "stage1_queued": 0,
            "stage1_completed": 0,
            "stage1_survived": 0,
            "persisted_rows": 0,
            "dropped_persistence_quality_gate": 0,
        },
    }
    child_rows: dict[str, dict[str, Any]] = {}
    runner._live_training_context = {
        "exp_id": exp_id,
        "phase": "proper_component_knockout_s1",
    }
    try:
        for index, child in enumerate(parent_plan.children):
            graph = child.graph
            program_metrics: dict[str, Any] = {
                "model_source": "ablation",
                "intentional_rerun_reason": "proper_component_knockout_s1",
                "source_result_id": parent_plan.parent.result_id,
                "source_graph_fingerprint": parent_plan.parent.fingerprint,
                "knockout_parent_result_id": parent_plan.parent.result_id,
                "knockout_node_id": child.node_id,
                "knockout_op_name": child.op_name,
                "knockout_bypass_input_id": child.bypass_input_id,
                "knockout_phase": "s1",
            }
            try:
                phase_vocab = (
                    config.qualifying_vocab_size
                    if config.progressive_screening
                    and config.vocab_size > config.qualifying_vocab_size
                    else config.vocab_size
                )
                model = compile_model(
                    [graph] * int(config.n_layers),
                    vocab_size=phase_vocab,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = runner._safe_eval_for_stage(
                    model,
                    stage_tag="proper_component_knockout_s1",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=phase_vocab,
                    device=dev_str,
                    timeout_seconds=30,
                )
                program_metrics.update(runner._extract_sandbox_metrics(sandbox_result))
                program_metrics["param_count"] = sandbox_result.param_count
                s0_passed = bool(sandbox_result.passed)
                s05_passed = bool(
                    s0_passed
                    and sandbox_result.stability_score
                    >= config.stage05_stability_threshold
                    and sandbox_result.causality_passed
                )
                if s0_passed:
                    results["stage0_passed"] += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                if not s0_passed or not s05_passed:
                    key = "dropped_stage0" if not s0_passed else "dropped_stage05"
                    results["funnel_counts"][key] += 1
                    _record_screening_failure(
                        nb=nb,
                        exp_id=exp_id,
                        graph=graph,
                        stage0_passed=s0_passed,
                        stage05_passed=s05_passed,
                        error_type=sandbox_result.error_type or "screening_gate_failed",
                        error_message=(sandbox_result.error or "")[:240] or None,
                        stage_at_death="stage0" if not s0_passed else "stage05",
                        stability_score=sandbox_result.stability_score,
                        extra_metrics=program_metrics,
                    )
                    child_row = _record_latest_result(
                        nb, experiment_id=exp_id, child=child
                    )
                    if child_row:
                        child_rows[child.fingerprint] = child_row
                    continue

                try:
                    s075_dev = torch.device(dev_str)
                    model.train()
                    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                    ids = torch.randint(0, phase_vocab, (4, 64), device=s075_dev)
                    with torch.amp.autocast(
                        device_type=s075_dev.type,
                        dtype=torch.bfloat16,
                        enabled=s075_dev.type == "cuda",
                    ):
                        logits = model(ids)
                        loss = next_token_cross_entropy(logits, ids, logits.size(-1))
                    initial_loss = float(loss.item())
                    program_metrics["s075_initial_loss"] = initial_loss
                    opt.zero_grad(set_to_none=True)
                    del opt
                    if (
                        not torch.isnan(torch.tensor(initial_loss))
                        and not torch.isinf(torch.tensor(initial_loss))
                        and initial_loss > INITIAL_LOSS_THRESHOLD
                    ):
                        results["funnel_counts"]["dropped_s075_high_init"] += 1
                        _record_screening_failure(
                            nb=nb,
                            exp_id=exp_id,
                            graph=graph,
                            stage0_passed=True,
                            stage05_passed=True,
                            error_type="high_initial_loss",
                            error_message=(
                                f"initial_loss={initial_loss:.4f} > "
                                f"{INITIAL_LOSS_THRESHOLD:.4f}"
                            ),
                            stage_at_death="stage075",
                            stability_score=sandbox_result.stability_score,
                            extra_metrics=program_metrics,
                        )
                        child_row = _record_latest_result(
                            nb, experiment_id=exp_id, child=child
                        )
                        if child_row:
                            child_rows[child.fingerprint] = child_row
                        continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "S0.75 probe skipped for %s: %s", child.fingerprint[:12], exc
                    )

                from research.eval.screening_rapid import RapidScreeningCheck

                rapid = RapidScreeningCheck()
                results["funnel_counts"]["rapid_screen_attempted"] += 1
                rapid_result = rapid.run(
                    model,
                    vocab_size=phase_vocab,
                    seq_len=min(128, config.max_seq_len),
                    batch_size=2,
                    device=dev_str,
                )
                program_metrics["rapid_screening_passed"] = rapid_result.passed
                program_metrics["rapid_screening_elapsed_ms"] = rapid_result.elapsed_ms
                if not rapid_result.passed:
                    results["rapid_screening_killed"] += 1
                    results["funnel_counts"]["dropped_rapid_screening"] += 1
                    reason = rapid_result.kill_reason or "unknown"
                    results["rapid_screening_kill_reasons"][reason] = (
                        results["rapid_screening_kill_reasons"].get(reason, 0) + 1
                    )
                    _record_screening_failure(
                        nb=nb,
                        exp_id=exp_id,
                        graph=graph,
                        stage0_passed=True,
                        stage05_passed=True,
                        error_type="rapid_screening_error",
                        error_message=reason[:240],
                        stage_at_death="rapid_screening",
                        stability_score=sandbox_result.stability_score,
                        extra_metrics=program_metrics,
                    )
                    child_row = _record_latest_result(
                        nb, experiment_id=exp_id, child=child
                    )
                    if child_row:
                        child_rows[child.fingerprint] = child_row
                    continue

                if (
                    config.progressive_screening
                    and config.vocab_size > config.qualifying_vocab_size
                ):
                    del model
                    clear_gpu_memory()
                    model = compile_model(
                        [graph] * int(config.n_layers),
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )
                    program_metrics["progressive_phase2_compiled"] = True

                results["funnel_counts"]["stage1_queued"] += 1
                s1_result = runner._micro_train(
                    model,
                    config,
                    dev,
                    seed=runner._stable_seed(
                        exp_id,
                        parent_plan.parent.result_id,
                        child.node_id,
                        child.fingerprint,
                        "proper_component_knockout_s1",
                    ),
                    graph_json=graph_to_json(graph),
                )
                jr = JobResult(
                    index=index,
                    s1_result=s1_result,
                    payload={"metrics": program_metrics, "graph": graph},
                    telemetry={},
                )
                runner._record_orchestrator_result(jr, nb, exp_id, results, config)
                child_row = _record_latest_result(nb, experiment_id=exp_id, child=child)
                if child_row:
                    child_rows[child.fingerprint] = child_row
            except Exception as exc:  # noqa: BLE001
                program_metrics["error_type"] = "proper_component_knockout_s1_error"
                program_metrics["error_message"] = str(exc)[:240]
                _record_screening_failure(
                    nb=nb,
                    exp_id=exp_id,
                    graph=graph,
                    stage0_passed=False,
                    stage05_passed=False,
                    error_type="proper_component_knockout_s1_error",
                    error_message=str(exc)[:240],
                    stage_at_death="stage0",
                    stability_score=None,
                    extra_metrics=program_metrics,
                )
                child_row = _record_latest_result(nb, experiment_id=exp_id, child=child)
                if child_row:
                    child_rows[child.fingerprint] = child_row
            finally:
                clear_gpu_memory()
            status["running"] = {
                "phase": "s1",
                "parent_result_id": parent_plan.parent.result_id,
                "completed": index + 1,
                "total": len(parent_plan.children),
            }
            json_dump(status_path, status)
            log(
                f"S1 knockout {parent_plan.parent.result_id} "
                f"{index + 1}/{len(parent_plan.children)} "
                f"{child.node_id}:{child.op_name}"
            )
    finally:
        runner._live_training_context = None

    nb.complete_experiment(
        experiment_id=exp_id,
        results=results,
        aria_summary=(
            "Proper component knockout S1 complete: "
            f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1"
        ),
    )
    nb.flush_writes()
    return exp_id, child_rows


def run_investigation_phase(
    *,
    nb: LabNotebook,
    runner: ExperimentRunner,
    parent_plan: ParentPlan,
    s1_child_rows: dict[str, dict[str, Any]],
    investigation_steps: int,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    source_result_ids = [
        str(s1_child_rows[child.fingerprint]["result_id"])
        for child in parent_plan.children
        if child.fingerprint in s1_child_rows
        and s1_child_rows[child.fingerprint].get("result_id")
    ]
    if not source_result_ids:
        return None, {}
    config = parent_plan.config.copy()
    config.model_source = "ablation"
    config.n_training_programs = 1
    config.investigation_steps = int(investigation_steps)
    # The user asked for investigation-length evidence, not early-stop probes.
    # `_build_investigation_config` scales these screening-calibrated values by
    # investigation_steps / stage1_steps, so setting them to the current S1
    # budget forces the derived investigation config to wait ~investigation_steps
    # before early stopping can trigger.
    config.early_stop_min_steps = int(config.stage1_steps)
    config.early_stop_patience = int(config.stage1_steps)
    config.validation_n_seeds = 1
    config.auto_validate = False
    config.auto_scale_up = False
    config.auto_investigate = False
    config.gbm_prescreener_enabled = False
    hypothesis = (
        "Proper component knockout investigation: rerun each single-node "
        f"deletion from {parent_plan.parent.result_id} for "
        f"{int(investigation_steps)} steps with investigation probes."
    )
    exp_id = runner.start_investigation(
        source_result_ids,
        config,
        hypothesis=hypothesis,
        exploratory=True,
        force=True,
    )
    thread = getattr(runner, "_thread", None)
    if thread is not None:
        thread.join()
    nb.flush_writes()
    expected_children = [
        child for child in parent_plan.children if child.fingerprint in s1_child_rows
    ]
    rows = _wait_for_complete_phase_rows(
        nb,
        experiment_id=exp_id,
        children=expected_children,
        phase="investigation",
    )
    return exp_id, rows


def build_plans(
    *,
    nb: LabNotebook,
    top_k: int,
    include_references: bool,
    device: str | None,
    allow_existing_knockout_s1: bool = False,
    require_existing_knockout_s1: bool = False,
    parent_result_ids: set[str] | None = None,
) -> list[ParentPlan]:
    parents = select_top_parents(
        nb,
        top_k=max(1, int(top_k)),
        include_references=bool(include_references),
    )
    plans: list[ParentPlan] = []
    global_seen: set[str] = set()
    for parent in parents:
        if parent_result_ids and parent.result_id not in parent_result_ids:
            continue
        metric_audit = ensure_ablation_metric_completeness(
            nb, parent_result_id=parent.result_id
        )
        config = prepare_config(parent.config, device=device)
        children, rejected = build_component_children(
            nb,
            parent,
            config,
            global_seen=global_seen,
            allow_existing_knockout_s1=allow_existing_knockout_s1,
        )
        if require_existing_knockout_s1:
            existing_s1 = _existing_knockout_s1_fingerprints(
                nb, [child.fingerprint for child in children]
            )
            kept_children: list[DeletionChild] = []
            for child in children:
                if child.fingerprint in existing_s1:
                    kept_children.append(child)
                else:
                    rejected.append(
                        {
                            "reason": "no_existing_knockout_s1_row",
                            "node_id": child.node_id,
                            "op_name": child.op_name,
                            "fingerprint": child.fingerprint,
                        }
                    )
            children = kept_children
        plans.append(
            ParentPlan(
                parent=parent,
                config=config,
                children=children,
                rejected=rejected,
                metric_audit=metric_audit,
            )
        )
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--parent-result-id",
        action="append",
        default=[],
        help=(
            "Restrict planning/runs to this parent result_id. May be passed "
            "multiple times; the parent must be within --top-k selection."
        ),
    )
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--investigation-steps", type=int, default=2500)
    parser.add_argument("--s1-only", action="store_true")
    parser.add_argument("--investigation-only", action="store_true")
    parser.add_argument(
        "--rerun-existing-knockouts",
        action="store_true",
        help=(
            "Allow child fingerprints already produced by this proper knockout "
            "campaign to be measured again. Other existing fingerprints remain "
            "duplicate rejects."
        ),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_file)
    db_path = Path(args.db)
    status_path = RUNTIME_DIR / "proper_component_knockout_ablation_status.json"
    nb = LabNotebook(str(db_path), use_native=False)
    try:
        plans = build_plans(
            nb=nb,
            top_k=max(1, int(args.top_k)),
            include_references=bool(args.include_references),
            device=args.device,
            allow_existing_knockout_s1=bool(
                args.investigation_only or args.rerun_existing_knockouts
            ),
            require_existing_knockout_s1=bool(args.investigation_only),
            parent_result_ids=set(args.parent_result_id or []) or None,
        )
        if args.parent_result_id and not plans:
            requested = ", ".join(args.parent_result_id)
            raise RuntimeError(
                f"No selected parent matched --parent-result-id {requested}; "
                "increase --top-k or verify the parent result_id."
            )
        status: dict[str, Any] = {
            "created_at": time.time(),
            "log_path": str(log_path),
            "db_path": str(db_path),
            "top_k": max(1, int(args.top_k)),
            "parent_result_ids": list(args.parent_result_id or []),
            "investigation_steps": int(args.investigation_steps),
            "audit_only": bool(args.audit_only),
            "dry_run": bool(args.dry_run),
            "s1_only": bool(args.s1_only),
            "investigation_only": bool(args.investigation_only),
            "plans": [
                {
                    "parent": {
                        "result_id": plan.parent.result_id,
                        "experiment_id": plan.parent.experiment_id,
                        "fingerprint": plan.parent.fingerprint,
                        "loss_ratio": plan.parent.loss_ratio,
                        "composite_score": plan.parent.composite_score,
                    },
                    "accepted_children": [
                        child_status(child) for child in plan.children
                    ],
                    "rejected_children": plan.rejected,
                    "metric_audit": plan.metric_audit,
                }
                for plan in plans
            ],
        }
        status["parent_count"] = len(plans)
        status["planned_children"] = sum(len(plan.children) for plan in plans)
        status["rejected_children"] = sum(len(plan.rejected) for plan in plans)
        status["planned_program_rows"] = status["planned_children"] * (
            1 if args.s1_only or args.investigation_only else 2
        )
        json_dump(status_path, status)
        log(
            "planned proper component knockout "
            f"parents={status['parent_count']} children={status['planned_children']} "
            f"rejected={status['rejected_children']} status={status_path}"
        )
        if args.audit_only or args.dry_run:
            log("audit/dry run complete; no database writes or child training launched")
            return 0
        if status["planned_children"] <= 0:
            log("no compile-valid unique knockout children to run")
            return 1
        if not args.no_backup:
            status["backup_paths"] = make_backups(db_path, dry_run=False)
            json_dump(status_path, status)
            log(f"database backups created: {status['backup_paths']}")

        runner = ExperimentRunner(str(db_path))
        parent_rows = {
            plan.parent.result_id: _row_by_result_id(nb, plan.parent.result_id)
            for plan in plans
        }
        for plan in plans:
            if not plan.children:
                continue
            s1_exp_id: str | None = None
            s1_rows: dict[str, dict[str, Any]] = {}
            if not args.investigation_only:
                s1_exp_id, s1_rows = run_s1_phase(
                    nb=nb,
                    runner=runner,
                    parent_plan=plan,
                    status=status,
                    status_path=status_path,
                )
                for child in plan.children:
                    child_row = s1_rows.get(child.fingerprint)
                    evidence_id = record_phase_evidence(
                        nb,
                        parent=plan.parent,
                        child=child,
                        child_row=child_row,
                        phase="s1",
                        phase_experiment_id=s1_exp_id,
                        parent_row=parent_rows.get(plan.parent.result_id),
                    )
                    status.setdefault("s1_evidence_ids", []).append(evidence_id)
            else:
                for child in plan.children:
                    row = nb.conn.execute(
                        """SELECT * FROM program_results
                           WHERE graph_fingerprint = ?
                             AND intentional_rerun_reason = 'proper_component_knockout_s1'
                           ORDER BY timestamp DESC LIMIT 1""",
                        (child.fingerprint,),
                    ).fetchone()
                    if row:
                        s1_rows[child.fingerprint] = dict(row)
                if s1_rows:
                    phase_ids = {
                        str(row.get("experiment_id") or "")
                        for row in s1_rows.values()
                        if row.get("experiment_id")
                    }
                    s1_phase_id = ",".join(sorted(phase_ids))
                    for child in plan.children:
                        child_row = s1_rows.get(child.fingerprint)
                        if child_row is None:
                            continue
                        evidence_id = record_phase_evidence(
                            nb,
                            parent=plan.parent,
                            child=child,
                            child_row=child_row,
                            phase="s1",
                            phase_experiment_id=s1_phase_id,
                            parent_row=parent_rows.get(plan.parent.result_id),
                        )
                        status.setdefault("s1_evidence_ids", []).append(evidence_id)

            if not args.s1_only:
                inv_exp_id, inv_rows = run_investigation_phase(
                    nb=nb,
                    runner=runner,
                    parent_plan=plan,
                    s1_child_rows=s1_rows,
                    investigation_steps=int(args.investigation_steps),
                )
                if inv_exp_id:
                    for child in plan.children:
                        child_row = inv_rows.get(child.fingerprint)
                        evidence_id = record_phase_evidence(
                            nb,
                            parent=plan.parent,
                            child=child,
                            child_row=child_row,
                            phase="investigation",
                            phase_experiment_id=inv_exp_id,
                            parent_row=parent_rows.get(plan.parent.result_id),
                        )
                        status.setdefault("investigation_evidence_ids", []).append(
                            evidence_id
                        )
            status["updated_at"] = time.time()
            json_dump(status_path, status)

        final_audit = {
            plan.parent.result_id: ensure_ablation_metric_completeness(
                nb,
                parent_result_id=plan.parent.result_id,
            )
            for plan in plans
        }
        status["final_metric_audit"] = final_audit
        status["completed_at"] = time.time()
        json_dump(status_path, status)
        log(f"proper component knockout complete final_metric_audit={final_audit}")
        return 0
    finally:
        try:
            nb.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
