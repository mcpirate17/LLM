#!/usr/bin/env python3
"""Backpopulate missing screening/probe metrics onto existing backfill rows.

Rebuilds models from stored ``program_results.graph_json`` and replays only the
applicable screening stages for rows that already reached those stages.
Updates are written in place to the existing ``result_id``; no duplicate rows
are inserted.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import gc
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from research.eval.screening_rapid import RapidScreeningCheck
from research.eval.binding_curriculum import (
    CURRICULUM_BINDING_PROTOCOL_VERSION,
    CURRICULUM_BINDING_DISTANCES,
    CURRICULUM_BINDING_EVAL_SCREENING,
    curriculum_binding_range_profile,
)
from research.eval.binding_range import binding_range_profile
from research.eval.hellaswag_eval import screening_hellaswag_eval
from research.eval.native_induction import (
    induction_result_metadata,
    induction_score_gold,
)
from research.scientist.notebook import LabNotebook
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner.shared import get_shared_runner
from research.scientist.runner._helpers import (
    screening_probe_fields,
    screening_wikitext_fields,
)
from research.scientist.shared_utils import resolve_device
from research.synthesis.serializer import graph_from_json
from research.tools._candidate_selection import fetch_latest_unique_fingerprint_rows
from research.tools._fingerprint_selection import dedupe_records_by_fingerprint
from research.tools.backfill import store_probe_results

DB_PATH = Path("research/runs.db")
REPORT_PATH = Path("research/reports/backpopulate_screening_metrics.tsv")
DEFAULT_BATCH_COMMIT = 10
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_WORKER_TIMEOUT_SECONDS = None
DEFAULT_POST_TRAIN_STABILITY_RUNS = 1
DEFAULT_SELECTION_SLICE = "backfill"
RAPID_REQUIRED_FIELDS = (
    "rapid_screening_passed",
    "rapid_screening_elapsed_ms",
    "rapid_screening_steps_completed",
    "rapid_screening_max_steps",
)
POST_REQUIRED_FIELDS = (
    "wikitext_perplexity",
    "hellaswag_acc",
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
)

POST_TARGET_ALIASES = {
    "full": "full",
    "one": "hellaswag",
    "hellaswag": "hellaswag",
    "two": "binding",
    "binding": "binding",
    "induction": "induction",
    "ar": "ar",
    "blimp": "blimp",
    "ncd": "ncd",
    "all": "all",
}


def _parse_optional_int(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def _release_model(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        gc.collect()
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()


def _normalize_post_target(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in POST_TARGET_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unsupported --post-train-target={raw!r}; "
            "use one, two, all, full, hellaswag, binding, induction, ar, blimp, or ncd."
        )
    return POST_TARGET_ALIASES[key]


def _target_post_fields(target: str) -> tuple[str, ...]:
    if target == "hellaswag":
        return ("hellaswag_acc",)
    if target == "induction":
        return ("induction_screening_auc",)
    if target == "binding":
        return ("binding_screening_auc",)
    if target == "ar":
        return ("ar_legacy_auc",)
    if target == "blimp":
        return ("blimp_overall_accuracy",)
    if target == "ncd":
        return ("ncd_score",)
    if target == "all":
        return (
            "hellaswag_acc",
            "induction_screening_auc",
            "binding_screening_auc",
            "binding_screening_composite",
        )
    return POST_REQUIRED_FIELDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backpopulate missing screening/probe metrics in-place"
    )
    add = parser.add_argument
    add("--db", type=Path, default=DB_PATH)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--limit", type=int, default=0)
    add("--result-id", action="append", default=[])
    add("--from-report", type=Path)
    add("--force", action="store_true")
    add(
        "--selection-slice",
        choices=("backfill", "trusted_candidates", "nonref_unique_fingerprints"),
        default=DEFAULT_SELECTION_SLICE,
        help=(
            "Which row cohort to scan when --result-id/--from-report are not used. "
            "'backfill' preserves legacy behavior; 'trusted_candidates' targets "
            "candidate_grade + candidate_comparable rows; "
            "'nonref_unique_fingerprints' targets the latest row for each "
            "non-reference graph fingerprint."
        ),
    )
    add(
        "--balance-by-family",
        action="store_true",
        help="When selecting without explicit result ids, interleave graph families "
        "so dense/sparse/routing/moe coverage grows more evenly.",
    )
    add("--skip-rapid", action="store_true")
    add("--skip-post-train", action="store_true")
    add(
        "--post-train-target",
        type=_normalize_post_target,
        default="full",
        help="Which post-train metric family to backfill: "
        "'one'/'hellaswag', 'two'/'binding' (induction+binding AUCs), "
        "'induction', 'all' (hellaswag+induction+binding probes), "
        "or 'full' (legacy full post-train, including wikitext).",
    )
    add(
        "--allow-insufficient-learning-metrics",
        action="store_true",
        help="For backpopulate only, keep post-train screening/probe metrics even "
        "when CUDA replay fails only due to the validation-loss generalization gate.",
    )
    add("--batch-commit", type=int, default=DEFAULT_BATCH_COMMIT)
    add(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help="Stop the run after this many row-level failures in a row to avoid "
        "burning long CUDA batches on catastrophic tool/runtime failures.",
    )
    add(
        "--worker-timeout-seconds",
        type=_parse_optional_int,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="Hard timeout for a single isolated replay worker. Use 'none' or 'null' for no timeout.",
    )
    add("--report", type=Path, default=REPORT_PATH)
    add(
        "--post-train-stability-runs",
        type=int,
        default=DEFAULT_POST_TRAIN_STABILITY_RUNS,
        help="Repeat post-train CUDA replay this many times and fail closed when "
        "key metrics drift beyond tolerance.",
    )
    add(
        "--stability-wikitext-rel-tol",
        type=float,
        default=0.10,
        help="Maximum allowed relative drift for wikitext_perplexity.",
    )
    add(
        "--stability-hellaswag-abs-tol",
        type=float,
        default=0.05,
        help="Maximum allowed absolute drift for hellaswag_acc.",
    )
    add(
        "--stability-probe-abs-tol",
        type=float,
        default=0.01,
        help="Maximum allowed absolute drift for induction/binding probe metrics.",
    )
    add("--dry-run", action="store_true")
    add("--isolate-subprocess", action=argparse.BooleanOptionalAction, default=True)
    add(
        "--fallback-device",
        default="none",
        help="Disabled. CUDA failures are treated as unrecovered; keep this as 'none'.",
    )
    add("--worker-payload", type=Path, help=argparse.SUPPRESS)
    add("--worker-output", type=Path, help=argparse.SUPPRESS)
    add("--audit-prefix", default="", help=argparse.SUPPRESS)
    add("--audit-experiment-id", default="", help=argparse.SUPPRESS)
    add("--audit-source-script", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    return bool(int(_row_value(row, key) or 0))


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row.keys():
        return row[key]
    return None


def _needs_rapid(row: Mapping[str, Any], force: bool) -> bool:
    if not _truthy(row, "stage0_passed") or not _truthy(row, "stage05_passed"):
        return False
    if force:
        return True
    return any(
        row[key] is None
        for key in (
            "rapid_screening_passed",
            "rapid_screening_elapsed_ms",
            "rapid_screening_steps_completed",
            "rapid_screening_max_steps",
        )
    )


def _needs_post_train(
    row: Mapping[str, Any], force: bool, target_fields: Sequence[str]
) -> bool:
    if not _truthy(row, "stage0_passed") or not _truthy(row, "stage05_passed"):
        return False
    if _supports_compile_only_post_target(target_fields):
        if force:
            return True
        return any(_row_value(row, key) is None for key in target_fields)
    if not _has_replayable_train_budget(row):
        return False
    if force:
        return True
    return any(_row_value(row, key) is None for key in target_fields)


def _is_reference_row(row: Mapping[str, Any]) -> bool:
    trust_label = str(_row_value(row, "trust_label") or "").strip().lower()
    comparability_label = (
        str(_row_value(row, "comparability_label") or "").strip().lower()
    )
    result_id = str(_row_value(row, "result_id") or "").strip().lower()
    return (
        trust_label == "reference"
        or comparability_label == "reference_comparable"
        or result_id.startswith("ref_")
    )


def _has_replayable_train_budget(row: Mapping[str, Any]) -> bool:
    if _row_value(row, "n_train_steps") is not None:
        return True
    if _row_value(row, "train_budget_steps") is not None:
        return True
    return _truthy(row, "stage1_passed") and _is_reference_row(row)


def _supports_compile_only_post_target(target_fields: Sequence[str]) -> bool:
    fields = set(target_fields)
    compile_only_fields = {
        "hellaswag_acc",
        "induction_screening_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "ar_legacy_auc",
        "blimp_overall_accuracy",
        "ncd_score",
    }
    return bool(fields) and fields.issubset(compile_only_fields)


def _candidate_result_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = [str(rid).strip() for rid in args.result_id if str(rid).strip()]
    if args.from_report:
        lines = args.from_report.read_text(encoding="utf-8").splitlines()
        if lines:
            header = lines[0].split("\t")
            try:
                idx = header.index("result_id")
            except ValueError:
                idx = 0
            for line in lines[1:]:
                parts = line.split("\t")
                if idx < len(parts) and parts[idx].strip():
                    ids.append(parts[idx].strip())
    seen = set()
    ordered: List[str] = []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


def _fetch_rows(
    conn: sqlite3.Connection,
    result_ids: Sequence[str],
    limit: int,
    force: bool,
    selection_slice: str,
    balance_by_family: bool,
    target_post_fields: Sequence[str],
) -> List[sqlite3.Row]:
    compile_only = _supports_compile_only_post_target(target_post_fields)
    base = """
        SELECT
            pr.result_id,
            pr.experiment_id,
            pr.graph_fingerprint,
            pr.graph_json,
            pr.stage0_passed,
            pr.stage05_passed,
            pr.stage1_passed,
            pr.n_train_steps,
            pr.train_budget_steps,
            pr.rapid_screening_passed,
            pr.rapid_screening_elapsed_ms,
            pr.rapid_screening_steps_completed,
            pr.rapid_screening_max_steps,
            pr.wikitext_perplexity,
            pr.hellaswag_acc,
            pr.induction_screening_auc,
            pr.binding_screening_auc,
            pr.binding_screening_composite,
            pr.ar_legacy_auc,
            pr.blimp_overall_accuracy,
            pr.ncd_score,
            pr.trust_label,
            pr.comparability_label,
            pr.data_provenance_json,
            e.config_json,
            e.timestamp
        FROM program_results_compat pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
          AND pr.graph_json <> '{}'
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
    """
    params: List[Any] = []
    if result_ids:
        placeholders = ",".join("?" for _ in result_ids)
        base += f" AND pr.result_id IN ({placeholders})"
        params.extend(result_ids)
    else:
        if selection_slice == "trusted_candidates":
            base += (
                " AND pr.trust_label = 'candidate_grade'"
                " AND pr.comparability_label = 'candidate_comparable'"
                " AND pr.stage1_passed = 1"
            )
        elif selection_slice == "nonref_unique_fingerprints":
            base += (
                " AND COALESCE(pr.trust_label, '') <> 'reference'"
                " AND TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''"
            )
        else:
            base += " AND e.experiment_type = 'backfill'"
    if not result_ids and not force:
        if selection_slice == "trusted_candidates":
            missing_target_sql = " OR ".join(
                f"pr.{field} IS NULL" for field in target_post_fields
            )
            budget_clause = (
                "1=1"
                if compile_only
                else """
                pr.n_train_steps IS NOT NULL OR
                pr.train_budget_steps IS NOT NULL OR
                (
                  pr.stage1_passed = 1 AND (
                    COALESCE(pr.trust_label, '') = 'reference' OR
                    COALESCE(pr.comparability_label, '') = 'reference_comparable' OR
                    pr.result_id LIKE 'ref\\_%' ESCAPE '\\'
                  )
                )
            """
            )
            base += f"""
              AND ({budget_clause})
              AND ({missing_target_sql})
            """
        else:
            budget_clause = (
                "1=1"
                if compile_only
                else """
                    pr.n_train_steps IS NOT NULL OR
                    pr.train_budget_steps IS NOT NULL OR
                    (
                      pr.stage1_passed = 1 AND (
                        COALESCE(pr.trust_label, '') = 'reference' OR
                        COALESCE(pr.comparability_label, '') = 'reference_comparable' OR
                        pr.result_id LIKE 'ref\\_%' ESCAPE '\\'
                      )
                    )
            """
            )
            base += f"""
              AND (
                pr.rapid_screening_passed IS NULL OR
                pr.rapid_screening_elapsed_ms IS NULL OR
                pr.rapid_screening_steps_completed IS NULL OR
                pr.rapid_screening_max_steps IS NULL OR
                (
                  ({budget_clause}) AND (
                    pr.wikitext_perplexity IS NULL OR
                    pr.hellaswag_acc IS NULL OR
                    pr.induction_screening_auc IS NULL OR
                    pr.binding_screening_auc IS NULL OR
                    pr.binding_screening_composite IS NULL
                  )
                )
              )
            """
    if not result_ids and selection_slice == "nonref_unique_fingerprints":
        extra_where = base.split("WHERE", 1)[1]
        rows = fetch_latest_unique_fingerprint_rows(
            conn,
            select_sql="""
                pr.result_id,
                pr.experiment_id,
                pr.graph_fingerprint,
                pr.graph_json,
                pr.stage0_passed,
                pr.stage05_passed,
                pr.stage1_passed,
                pr.n_train_steps,
                pr.train_budget_steps,
                pr.rapid_screening_passed,
                pr.rapid_screening_elapsed_ms,
                pr.rapid_screening_steps_completed,
                pr.rapid_screening_max_steps,
                pr.wikitext_perplexity,
                pr.hellaswag_acc,
                pr.induction_screening_auc,
                pr.binding_screening_auc,
                pr.binding_screening_composite,
                pr.ar_legacy_auc,
                pr.blimp_overall_accuracy,
                pr.ncd_score,
                pr.trust_label,
                pr.comparability_label,
                pr.data_provenance_json,
                e.config_json,
                e.timestamp
            """,
            extra_where_sql=" AND " + extra_where,
            params=params,
            include_leaderboard=False,
        )
    else:
        base += " ORDER BY e.timestamp ASC, pr.result_id ASC"
        rows = list(conn.execute(base, tuple(params)).fetchall())
    if not result_ids and balance_by_family:
        rows = _interleave_rows_by_family(rows)
    if limit > 0:
        rows = rows[: int(limit)]
    return rows


def _dedupe_rows_by_fingerprint_keep_latest(
    rows: Sequence[sqlite3.Row],
) -> List[sqlite3.Row]:
    return dedupe_records_by_fingerprint(rows)


def _graph_family_from_row(row: sqlite3.Row) -> str:
    raw = row["data_provenance_json"] if "data_provenance_json" in row.keys() else None
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            graph = payload.get("graph") or {}
            if isinstance(graph, dict):
                family = str(graph.get("graph_family") or "").strip().lower()
                if family:
                    return family
    return "unknown"


def _interleave_rows_by_family(rows: Sequence[sqlite3.Row]) -> List[sqlite3.Row]:
    buckets: dict[str, list[sqlite3.Row]] = {}
    family_order: list[str] = []
    for row in rows:
        family = _graph_family_from_row(row)
        if family not in buckets:
            buckets[family] = collections.deque()
            family_order.append(family)
        buckets[family].append(row)
    interleaved: List[sqlite3.Row] = []
    while True:
        progressed = False
        for family in family_order:
            bucket = buckets[family]
            if bucket:
                interleaved.append(bucket.popleft())
                progressed = True
        if not progressed:
            break
    return interleaved


def _build_run_config(row: sqlite3.Row, device: str) -> RunConfig:
    config = RunConfig(device=device, model_source="backpopulate_screening_metrics")
    try:
        raw = json.loads(row["config_json"] or "{}")
    except json.JSONDecodeError:
        raw = {}
    valid = {f.name for f in dataclasses.fields(RunConfig)}
    for key, value in raw.items():
        if key in valid:
            setattr(config, key, value)
    config.device = device
    budget_steps = (
        row["train_budget_steps"] or row["n_train_steps"] or config.stage1_steps
    )
    config.stage1_steps = int(budget_steps)
    config.collect_training_curve = False
    config.enable_perf_tracing = False
    config.enable_stage09_cheap_train_gate = False
    config.skip_post_s1_fingerprint = True
    config.skip_post_s1_triage = True
    if not bool(getattr(config, "binding_probe_offload_source_model", False)):
        config.binding_probe_offload_source_model = False
    if int(getattr(config, "binding_probe_eval_batch_size", 0) or 0) <= 0:
        config.binding_probe_eval_batch_size = 16 if device.startswith("cuda") else 32
    config.screening_probe_seed = ExperimentRunner._stable_seed(
        row["result_id"],
        row["graph_fingerprint"],
        "backpopulate_screening_probe",
    )
    return config


def _row_to_payload(
    row: sqlite3.Row,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    payload = {key: row[key] for key in row.keys()}
    if conn is not None and "graph_json" in payload:
        payload["graph_json"] = resolve_graph_json_value(
            conn,
            db_path,
            payload["graph_json"],
        )
    return payload


@contextlib.contextmanager
def _deterministic_compile_seed(device: str, seed: int):
    dev = resolve_device(device)
    cuda_devices: list[int] = []
    if dev.type == "cuda" and torch.cuda.is_available():
        cuda_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        cuda_devices = [int(cuda_idx)]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def _run_rapid(
    graph_json: str,
    config: RunConfig,
    device: str,
    result_id: str,
) -> Dict[str, Any]:
    graph = graph_from_json(graph_json)
    phase1_vocab = (
        config.qualifying_vocab_size
        if config.progressive_screening
        and config.vocab_size > config.qualifying_vocab_size
        else config.vocab_size
    )
    compile_seed = ExperimentRunner._stable_seed(
        result_id, graph.fingerprint(), "backpopulate_rapid_compile"
    )
    with _deterministic_compile_seed(device, compile_seed):
        model = compile_model(
            [graph] * int(config.n_layers),
            vocab_size=phase1_vocab,
            max_seq_len=config.max_seq_len,
        )
    dev_str = str(resolve_device(device))
    rapid = RapidScreeningCheck()
    try:
        result = rapid.run(
            model,
            vocab_size=phase1_vocab,
            seq_len=min(128, int(config.max_seq_len)),
            batch_size=2,
            device=dev_str,
        )
        metrics = result.metrics or {}
        updates: Dict[str, Any] = {
            "rapid_screening_passed": int(bool(result.passed)),
            "rapid_screening_elapsed_ms": result.elapsed_ms,
            "rapid_screening_steps_completed": metrics.get("steps_completed"),
            "rapid_screening_max_steps": rapid.max_steps,
            "rapid_screening_gpu_minutes_saved": result.gpu_minutes_saved,
            "rapid_screening_metrics": metrics,
        }
        if result.degraded:
            updates["rapid_screening_degraded"] = 1
            updates["rapid_screening_degraded_reasons"] = result.degraded_reasons
        if result.kill_reason:
            updates["rapid_screening_kill_reason"] = result.kill_reason
        if result.kill_step is not None:
            updates["rapid_screening_kill_step"] = result.kill_step
        if result.kill_metric is not None:
            updates["rapid_screening_kill_metric"] = result.kill_metric
        for step, col in (
            (10, "screening_loss_10"),
            (25, "screening_loss_25"),
            (50, "screening_loss_50"),
        ):
            key = f"loss_at_{step}"
            if key not in metrics and len(metrics.get("losses", [])) >= step:
                metrics[key] = metrics["losses"][step - 1]
            if metrics.get(key) is not None:
                updates[col] = metrics[key]
        return screening_probe_fields(updates)
    finally:
        _release_model(model)


def _run_post_train(
    runner: ExperimentRunner,
    graph_json: str,
    config: RunConfig,
    device: str,
    result_id: str,
    allow_insufficient_learning_metrics: bool = False,
) -> Dict[str, Any]:
    graph = graph_from_json(graph_json)
    compile_seed = ExperimentRunner._stable_seed(
        result_id, graph.fingerprint(), "backpopulate_post_compile"
    )
    with _deterministic_compile_seed(device, compile_seed):
        model = compile_model(
            [graph] * int(config.n_layers),
            vocab_size=int(config.vocab_size),
            max_seq_len=config.max_seq_len,
        )
    dev = resolve_device(device)
    try:
        s1_result = runner._micro_train(
            model,
            config,
            dev,
            seed=runner._stable_seed(result_id, graph.fingerprint(), "backpopulate"),
            graph_json=graph_json,
        )
        if s1_result.get("smoke_test_failure"):
            raise RuntimeError(
                f"smoke_test_failure: {s1_result.get('smoke_test_failure')}"
            )
        tolerate_gate_failure = (
            allow_insufficient_learning_metrics
            and s1_result.get("error")
            and s1_result.get("error_type") == "insufficient_learning"
        )
        if s1_result.get("error") and not tolerate_gate_failure:
            detail_parts = [str(s1_result.get("error"))]
            if s1_result.get("error_type"):
                detail_parts.append(f"type={s1_result.get('error_type')}")
            if s1_result.get("failure_op"):
                detail_parts.append(f"op={s1_result.get('failure_op')}")
            raise RuntimeError("micro_train_failed: " + " | ".join(detail_parts))
        updates: Dict[str, Any] = {}
        updates.update(screening_wikitext_fields(s1_result))
        updates.update(screening_probe_fields(s1_result))
        if s1_result.get("hellaswag_acc") is not None:
            updates["hellaswag_acc"] = s1_result.get("hellaswag_acc")
        if s1_result.get("hellaswag_status") is not None:
            updates["hellaswag_status"] = s1_result.get("hellaswag_status")
        if s1_result.get("hellaswag_n_examples") is not None:
            updates["hellaswag_n_examples"] = s1_result.get("hellaswag_n_examples")
        if tolerate_gate_failure:
            updates.update(
                _recover_hellaswag_after_gate_failure(
                    model=model,
                    config=config,
                    device=str(dev),
                )
            )
        updates["train_budget_steps"] = int(config.stage1_steps)
        return updates
    finally:
        _release_model(model)


def _run_compile_only_post_eval(
    graph_json: str,
    config: RunConfig,
    device: str,
    result_id: str,
    target_post_fields: Sequence[str],
) -> Dict[str, Any]:
    graph = graph_from_json(graph_json)
    compile_seed = ExperimentRunner._stable_seed(
        result_id, graph.fingerprint(), "backpopulate_compile_only_eval"
    )
    with _deterministic_compile_seed(device, compile_seed):
        model = compile_model(
            [graph] * int(config.n_layers),
            vocab_size=int(config.vocab_size),
            max_seq_len=config.max_seq_len,
        )
    dev = resolve_device(device)
    model = model.to(dev)
    target_set = set(target_post_fields)
    updates: Dict[str, Any] = {}
    try:
        if "hellaswag_acc" in target_set:
            hs = screening_hellaswag_eval(model, int(config.vocab_size), str(dev))
            if hs.get("hellaswag_status") == "all_failed":
                updates["hellaswag_acc"] = None
                updates["screening_hellaswag_correct"] = None
                updates["screening_hellaswag_total"] = None
            elif hs.get("hellaswag_acc") is not None:
                updates["hellaswag_acc"] = hs.get("hellaswag_acc")
            if hs.get("hellaswag_status") is not None:
                updates["hellaswag_status"] = hs.get("hellaswag_status")
            if hs.get("hellaswag_metric_version") is not None:
                updates["hellaswag_metric_version"] = hs.get("hellaswag_metric_version")
            if hs.get("hellaswag_tokenizer_mode") is not None:
                updates["hellaswag_tokenizer_mode"] = hs.get("hellaswag_tokenizer_mode")
            if hs.get("hellaswag_tiktoken_encoding") is not None:
                updates["hellaswag_tiktoken_encoding"] = hs.get(
                    "hellaswag_tiktoken_encoding"
                )
            if hs.get("hellaswag_total") is not None:
                updates["hellaswag_n_examples"] = hs.get("hellaswag_total")
                updates["screening_hellaswag_correct"] = hs.get("hellaswag_correct")
                updates["screening_hellaswag_total"] = hs.get("hellaswag_total")
            if hs.get("elapsed_ms") is not None:
                updates["screening_hellaswag_elapsed_ms"] = hs.get("elapsed_ms")

        if (
            "induction_screening_auc" in target_set
            or "binding_screening_composite" in target_set
        ):
            ind = induction_score_gold(
                model,
                device=str(dev),
                seed=getattr(config, "screening_probe_seed", None),
            )
            updates.update(induction_result_metadata(ind))

        if (
            "binding_screening_auc" in target_set
            or "binding_screening_composite" in target_set
        ):
            zero = binding_range_profile(
                model,
                distances=CURRICULUM_BINDING_DISTANCES,
                n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                device=str(dev),
                seed=getattr(config, "screening_probe_seed", None),
            )
            br = curriculum_binding_range_profile(
                model,
                distances=CURRICULUM_BINDING_DISTANCES,
                n_train_steps=400,
                n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                train_batch_size=max(
                    1,
                    int(
                        getattr(
                            config,
                            "binding_probe_train_batch_size",
                            16,
                        )
                        or 16
                    ),
                ),
                eval_batch_size=max(
                    1,
                    int(
                        getattr(
                            config,
                            "binding_probe_eval_batch_size",
                            32,
                        )
                        or 32
                    ),
                ),
                device=str(dev),
                seed=getattr(config, "screening_probe_seed", None),
                offload_source_model=bool(
                    getattr(config, "binding_probe_offload_source_model", False)
                ),
            )
            updates.update(
                {
                    "binding_screening_auc": zero.auc,
                    "binding_distance_accuracies": zero.distance_accuracies,
                    "binding_screening_eval_examples": CURRICULUM_BINDING_EVAL_SCREENING,
                    "binding_probe_distances": list(CURRICULUM_BINDING_DISTANCES),
                    "binding_screening_elapsed_ms": zero.elapsed_ms,
                    "binding_curriculum_auc": br.auc,
                    "binding_distance_accuracies_curriculum": br.distance_accuracies,
                    "binding_curriculum_steps": br.train_steps,
                    "binding_curriculum_elapsed_ms": br.elapsed_ms,
                    "binding_curriculum_protocol_version": CURRICULUM_BINDING_PROTOCOL_VERSION,
                }
            )

        if "ar_legacy_auc" in target_set:
            from research.eval.associative_recall import associative_recall_score

            ar = associative_recall_score(
                model,
                n_pairs=20,
                n_eval=200,
                n_train_steps=500,
                batch_size=16,
                device=str(dev),
            )
            updates["ar_legacy_auc"] = ar.auc
            updates["ar_legacy_final_acc"] = ar.final_acc
            updates["ar_legacy_timed_out"] = int(ar.timed_out)
            updates["ar_legacy_above_chance"] = int(ar.above_chance)

        if "blimp_overall_accuracy" in target_set:
            from research.eval.blimp_eval import evaluate_blimp

            blimp = evaluate_blimp(
                model, int(config.vocab_size), str(dev), n_per_subtask=50
            )
            updates["blimp_overall_accuracy"] = blimp.overall_accuracy
            updates["blimp_subtask_accuracies_json"] = json.dumps(
                blimp.subtask_accuracies
            )
            updates["blimp_n_subtasks"] = blimp.n_subtasks
            updates["blimp_status"] = blimp.status

        if "ncd_score" in target_set:
            from research.eval.ncd import compute_graph_ncd

            ncd_result = compute_graph_ncd(graph_json)
            updates["ncd_score"] = ncd_result.get("ncd_score")
            updates["ncd_description_length_per_param"] = ncd_result.get(
                "description_length_per_param"
            )

        # screening_probe_fields is a whitelist — pass probe fields through
        # it, then re-merge BLiMP/NCD fields that aren't in the whitelist.
        extra = {}
        for k in (
            "blimp_overall_accuracy",
            "blimp_subtask_accuracies_json",
            "blimp_n_subtasks",
            "blimp_status",
            "ncd_score",
            "ncd_description_length_per_param",
        ):
            if k in updates:
                extra[k] = updates[k]
        result = screening_probe_fields(updates)
        result.update(extra)
        return result
    finally:
        _release_model(model)


def _recover_hellaswag_after_gate_failure(
    *,
    model: torch.nn.Module,
    config: RunConfig,
    device: str,
) -> Dict[str, Any]:
    """Recover HellaSwag for tolerated insufficient-learning gate failures.

    ``_micro_train`` only runs post-S1 probes when ``result["passed"]`` is true.
    Backpopulate explicitly opts into keeping HellaSwag metrics for certain
    replay-only generalization gate failures, so rerun the requested HellaSwag
    probe on the trained model before returning an empty update set.
    """
    if getattr(config, "skip_screening_hellaswag", False):
        return {}

    from research.eval.hellaswag_eval import screening_hellaswag_eval

    hs = screening_hellaswag_eval(model, config.vocab_size, device)
    updates: Dict[str, Any] = {}
    if hs.get("hellaswag_status") == "all_failed":
        updates["hellaswag_acc"] = None
    elif hs.get("hellaswag_acc") is not None:
        updates["hellaswag_acc"] = hs.get("hellaswag_acc")
    if hs.get("hellaswag_status") is not None:
        updates["hellaswag_status"] = hs.get("hellaswag_status")
    if hs.get("hellaswag_metric_version") is not None:
        updates["hellaswag_metric_version"] = hs.get("hellaswag_metric_version")
    if hs.get("hellaswag_tokenizer_mode") is not None:
        updates["hellaswag_tokenizer_mode"] = hs.get("hellaswag_tokenizer_mode")
    if hs.get("hellaswag_tiktoken_encoding") is not None:
        updates["hellaswag_tiktoken_encoding"] = hs.get("hellaswag_tiktoken_encoding")
    if hs.get("hellaswag_total") is not None:
        updates["hellaswag_n_examples"] = hs.get("hellaswag_total")
    return updates


def _select_updates(
    row: sqlite3.Row, updates: Dict[str, Any], force: bool
) -> Dict[str, Any]:
    clearable_null_fields = set()
    if updates.get("hellaswag_status") == "all_failed":
        # Force-clearing stale HellaSwag values keeps old 0.0 rows from surviving
        # a rerun that never scored a valid example.
        clearable_null_fields.update(
            {
                "hellaswag_acc",
                "screening_hellaswag_correct",
                "screening_hellaswag_total",
            }
        )
    if force:
        return {
            k: v
            for k, v in updates.items()
            if v is not None or k in clearable_null_fields
        }
    selected: Dict[str, Any] = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key not in row.keys() or row[key] is None:
            selected[key] = value
    return selected


def _merge_binding_screening_composite_from_existing(
    row: Mapping[str, Any], updates: Dict[str, Any], force: bool
) -> None:
    binding_screening_auc = updates.get("binding_screening_auc")
    induction_screening_auc = updates.get("induction_screening_auc")
    if induction_screening_auc is None:
        induction_screening_auc = row.get("induction_screening_auc")
    if binding_screening_auc is None:
        binding_screening_auc = row.get("binding_screening_auc")
    if induction_screening_auc is None or binding_screening_auc is None:
        return
    if (
        force
        or row.get("binding_screening_composite") is None
        or "binding_screening_auc" in updates
        or "induction_screening_auc" in updates
    ):
        updates["binding_screening_composite"] = round(
            0.3 * float(induction_screening_auc) + 0.3 * float(binding_screening_auc), 4
        )


def _missing_required_fields(
    row: Dict[str, Any],
    updates: Dict[str, Any],
    force: bool,
    rapid_needed: bool,
    post_needed: bool,
    target_post_fields: Sequence[str],
) -> List[str]:
    missing: List[str] = []
    if rapid_needed:
        for key in RAPID_REQUIRED_FIELDS:
            if force or row.get(key) is None:
                if updates.get(key) is None:
                    missing.append(key)
    if post_needed:
        for key in target_post_fields:
            if (
                key == "hellaswag_acc"
                and updates.get("hellaswag_status") == "all_failed"
            ):
                continue
            if force or row.get(key) is None:
                if updates.get(key) is None:
                    missing.append(key)
    return missing


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_post_train_stability(
    runs: Sequence[Dict[str, Any]],
    compare_keys: Sequence[str],
    *,
    wikitext_rel_tol: float,
    hellaswag_abs_tol: float,
    probe_abs_tol: float,
) -> None:
    if len(runs) <= 1:
        return
    for key in compare_keys:
        values = [_float_or_none(run.get(key)) for run in runs]
        values = [v for v in values if v is not None]
        if len(values) <= 1:
            continue
        lo = min(values)
        hi = max(values)
        if key == "wikitext_perplexity":
            baseline = max(abs(sum(values) / len(values)), 1.0)
            drift = (hi - lo) / baseline
            if drift > wikitext_rel_tol:
                raise RuntimeError(
                    f"unstable_post_train_replay: {key} drift={drift:.4f} "
                    f"range=[{lo:.6g},{hi:.6g}]"
                )
        elif key == "hellaswag_acc":
            drift = hi - lo
            if drift > hellaswag_abs_tol:
                raise RuntimeError(
                    f"unstable_post_train_replay: {key} drift={drift:.4f} "
                    f"range=[{lo:.6g},{hi:.6g}]"
                )
        else:
            drift = hi - lo
            if drift > probe_abs_tol:
                raise RuntimeError(
                    f"unstable_post_train_replay: {key} drift={drift:.4f} "
                    f"range=[{lo:.6g},{hi:.6g}]"
                )


def _evaluate_row_payload(
    payload: Dict[str, Any],
    device: str,
    force: bool,
    skip_rapid: bool,
    skip_post_train: bool,
    post_train_stability_runs: int,
    stability_wikitext_rel_tol: float,
    stability_hellaswag_abs_tol: float,
    stability_probe_abs_tol: float,
    allow_insufficient_learning_metrics: bool,
    post_train_target: str,
    selection_slice: str = DEFAULT_SELECTION_SLICE,
) -> Dict[str, Any]:
    row = payload
    graph_json = str(row["graph_json"])
    target_post_fields = _target_post_fields(post_train_target)
    rapid_needed = (
        selection_slice != "trusted_candidates"
        and (not skip_rapid)
        and _needs_rapid(row, force)
    )
    post_needed = (not skip_post_train) and _needs_post_train(
        row, force, target_post_fields
    )
    updates: Dict[str, Any] = {}
    config = _build_run_config(row, device)
    # Replay only the missing post-train metrics for this row.
    target_post_field_set = set(target_post_fields)
    config.skip_screening_wikitext = (
        "wikitext_perplexity" not in target_post_field_set
        or (not force and row.get("wikitext_perplexity") is not None)
    )
    config.skip_screening_hellaswag = "hellaswag_acc" not in target_post_field_set or (
        not force and row.get("hellaswag_acc") is not None
    )
    config.skip_induction_probe = (
        "induction_screening_auc" not in target_post_field_set
        or (not force and row.get("induction_screening_auc") is not None)
    )
    config.skip_binding_probe = (
        "binding_screening_auc" not in target_post_field_set
        or (not force and row.get("binding_screening_auc") is not None)
    )
    config.skip_binding_probes = (
        config.skip_induction_probe and config.skip_binding_probe
    )
    if rapid_needed:
        updates.update(
            _run_rapid(
                graph_json,
                config,
                device,
                str(row["result_id"]),
            )
        )
    if post_needed and _supports_compile_only_post_target(target_post_fields):
        updates.update(
            _run_compile_only_post_eval(
                graph_json,
                config,
                device,
                str(row["result_id"]),
                target_post_fields,
            )
        )
        _merge_binding_screening_composite_from_existing(row, updates, force)
    elif post_needed:
        post_runs: List[Dict[str, Any]] = []
        runner = get_shared_runner(str(DB_PATH))
        for _ in range(max(1, int(post_train_stability_runs))):
            post_runs.append(
                _run_post_train(
                    runner,
                    graph_json,
                    config,
                    device,
                    str(row["result_id"]),
                    allow_insufficient_learning_metrics=allow_insufficient_learning_metrics,
                )
            )
        compare_keys = [
            key for key in target_post_fields if force or row.get(key) is None
        ]
        _check_post_train_stability(
            post_runs,
            compare_keys,
            wikitext_rel_tol=stability_wikitext_rel_tol,
            hellaswag_abs_tol=stability_hellaswag_abs_tol,
            probe_abs_tol=stability_probe_abs_tol,
        )
        updates.update(post_runs[-1])
        _merge_binding_screening_composite_from_existing(row, updates, force)
    updates = _select_updates(row, updates, force)
    missing_required = _missing_required_fields(
        row=row,
        updates=updates,
        force=force,
        rapid_needed=rapid_needed,
        post_needed=post_needed,
        target_post_fields=target_post_fields,
    )
    if missing_required:
        raise RuntimeError(
            "required metrics still missing after CUDA replay: "
            + ",".join(missing_required)
        )
    return {
        "rapid_needed": int(rapid_needed),
        "post_needed": int(post_needed),
        "updates": updates,
    }


def _run_worker_subprocess(
    conn: sqlite3.Connection, row: sqlite3.Row, args: argparse.Namespace
) -> Dict[str, Any]:
    payload = _row_to_payload(row, conn=conn, db_path=args.db)
    with tempfile.TemporaryDirectory(prefix="backpopulate_screening_") as tmpdir:
        payload_path = Path(tmpdir) / "payload.json"
        output_path = Path(tmpdir) / "output.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        cmd = [
            sys.executable,
            "-m",
            "research.tools.backpopulate_screening_metrics",
            "--device",
            str(args.device),
            "--selection-slice",
            str(args.selection_slice),
            "--worker-payload",
            str(payload_path),
            "--worker-output",
            str(output_path),
            "--post-train-stability-runs",
            str(args.post_train_stability_runs),
            "--stability-wikitext-rel-tol",
            str(args.stability_wikitext_rel_tol),
            "--stability-hellaswag-abs-tol",
            str(args.stability_hellaswag_abs_tol),
            "--stability-probe-abs-tol",
            str(args.stability_probe_abs_tol),
            "--post-train-target",
            str(args.post_train_target),
        ]
        if args.force:
            cmd.append("--force")
        if args.skip_rapid:
            cmd.append("--skip-rapid")
        if args.skip_post_train:
            cmd.append("--skip-post-train")
        if args.balance_by_family:
            cmd.append("--balance-by-family")
        if args.allow_insufficient_learning_metrics:
            cmd.append("--allow-insufficient-learning-metrics")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=(
                    None
                    if args.worker_timeout_seconds is None
                    else max(1, int(args.worker_timeout_seconds))
                ),
            )
            worker_output = {}
            if output_path.exists():
                worker_output = json.loads(output_path.read_text(encoding="utf-8"))
            if not output_path.exists():
                raise RuntimeError(
                    f"worker produced no output (exit={proc.returncode})"
                )
            if not bool(worker_output.get("ok", 0)):
                raise RuntimeError(
                    str(worker_output.get("error") or f"worker_exit_{proc.returncode}")
                )
            if proc.returncode != 0:
                raise RuntimeError(f"worker_exit_{proc.returncode}")
            return worker_output
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"worker_timeout_after_{int(args.worker_timeout_seconds)}s"
            ) from exc


def _write_report(report_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "result_id",
        "graph_fingerprint",
        "rapid_replayed",
        "post_train_replayed",
        "source_device",
        "updated_fields",
        "status",
        "error",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            clean = {
                key: str(row.get(key, ""))
                .replace("\t", " ")
                .replace("\n", " ")
                .replace("\r", " ")
                for key in headers
            }
            f.write("\t".join(clean[h] for h in headers) + "\n")


def _backpopulate_provenance_context(
    args: argparse.Namespace, device: str
) -> Dict[str, Any]:
    return {
        "kind": "screening_metric_backfill",
        "prefix": str(getattr(args, "audit_prefix", "") or ""),
        "experiment_id": str(getattr(args, "audit_experiment_id", "") or ""),
        "source_script": str(getattr(args, "audit_source_script", "") or ""),
        "post_train_target": str(args.post_train_target),
        "allow_insufficient_learning_metrics": bool(
            args.allow_insufficient_learning_metrics
        ),
        "post_train_stability_runs": int(args.post_train_stability_runs),
        "worker_timeout_seconds": args.worker_timeout_seconds,
        "device": str(device),
        "updated_at": round(time.time(), 3),
    }


def _apply_row_updates(
    nb: LabNotebook,
    *,
    result_id: str,
    updates: Dict[str, Any],
    provenance_context: Dict[str, Any],
) -> None:
    """Apply one row's updates inside a short-lived transaction."""
    if not updates:
        return
    with nb.batch():
        store_probe_results(
            nb,
            result_id,
            updates,
            write_leaderboard=True,
            provenance_context=provenance_context,
        )


def _print_backpopulate_summary(
    *,
    processed: int,
    total_rows: int,
    updated: int,
    updated_cuda: int,
    report_path: Path,
    elapsed: float,
    interrupted: bool = False,
) -> None:
    prefix = "Interrupted after" if interrupted else "Processed"
    print(
        f"{prefix} {processed}/{total_rows} rows, updated {updated} "
        f"(cuda={updated_cuda}), report={report_path}, elapsed={elapsed:.1f}s"
    )


def _run_worker_mode(args: argparse.Namespace) -> None:
    """Standalone worker: process one payload file and write the result."""
    payload = json.loads(args.worker_payload.read_text(encoding="utf-8"))
    try:
        result = _evaluate_row_payload(
            payload=payload,
            device=args.device,
            force=args.force,
            skip_rapid=args.skip_rapid,
            skip_post_train=args.skip_post_train,
            post_train_stability_runs=args.post_train_stability_runs,
            stability_wikitext_rel_tol=args.stability_wikitext_rel_tol,
            stability_hellaswag_abs_tol=args.stability_hellaswag_abs_tol,
            stability_probe_abs_tol=args.stability_probe_abs_tol,
            allow_insufficient_learning_metrics=args.allow_insufficient_learning_metrics,
            post_train_target=args.post_train_target,
            selection_slice=args.selection_slice,
        )
        result["ok"] = 1
    except Exception as exc:  # noqa: BLE001
        result = {"ok": 0, "error": str(exc)}
    args.worker_output.write_text(json.dumps(result), encoding="utf-8")


def _evaluate_one_row(
    conn: sqlite3.Connection, row: sqlite3.Row, args: argparse.Namespace
) -> tuple[int, int, Dict[str, Any]]:
    """Run a row through worker (in-process or subprocess). Returns (rapid_needed, post_needed, updates)."""
    if args.isolate_subprocess:
        worker = _run_worker_subprocess(conn, row, args)
    else:
        worker = _evaluate_row_payload(
            payload=_row_to_payload(row, conn=conn, db_path=args.db),
            device=args.device,
            force=args.force,
            skip_rapid=args.skip_rapid,
            skip_post_train=args.skip_post_train,
            post_train_stability_runs=args.post_train_stability_runs,
            stability_wikitext_rel_tol=args.stability_wikitext_rel_tol,
            stability_hellaswag_abs_tol=args.stability_hellaswag_abs_tol,
            stability_probe_abs_tol=args.stability_probe_abs_tol,
            allow_insufficient_learning_metrics=args.allow_insufficient_learning_metrics,
            post_train_target=args.post_train_target,
            selection_slice=args.selection_slice,
        )
    return (
        int(worker.get("rapid_needed") or 0),
        int(worker.get("post_needed") or 0),
        dict(worker.get("updates") or {}),
    )


def _process_row(
    nb: LabNotebook,
    row: sqlite3.Row,
    args: argparse.Namespace,
    target_post_fields: Sequence[str],
) -> Dict[str, Any]:
    """Process one row and return a report entry. Status reflects what happened."""
    rapid_needed = (not args.skip_rapid) and _needs_rapid(row, args.force)
    post_needed = (not args.skip_post_train) and _needs_post_train(
        row, args.force, target_post_fields
    )
    status = "skipped"
    err = ""
    updates: Dict[str, Any] = {}
    source_device = str(args.device)
    try:
        rapid_needed, post_needed, updates = _evaluate_one_row(nb.conn, row, args)
        if updates and not args.dry_run:
            # Keep the write transaction short. Holding nb.batch() open
            # across the expensive CUDA replay loop starves other writers.
            _apply_row_updates(
                nb,
                result_id=str(row["result_id"]),
                updates=updates,
                provenance_context=_backpopulate_provenance_context(
                    args, source_device
                ),
            )
            status = "updated"
        elif updates:
            status = "would_update"
        else:
            status = "no_missing_fields"
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        status = "error"
    return {
        "result_id": row["result_id"],
        "graph_fingerprint": row["graph_fingerprint"],
        "rapid_replayed": rapid_needed,
        "post_train_replayed": post_needed,
        "source_device": source_device,
        "updated_fields": ",".join(sorted(updates.keys())),
        "status": status,
        "error": err[:240],
        "n_updates": len(updates),
    }


def main() -> None:
    args = _parse_args()
    if args.worker_payload and args.worker_output:
        _run_worker_mode(args)
        return

    nb = LabNotebook(str(args.db))
    nb.conn.row_factory = sqlite3.Row
    result_ids = _candidate_result_ids(args)
    target_post_fields = _target_post_fields(args.post_train_target)
    rows = _fetch_rows(
        nb.conn,
        result_ids,
        args.limit,
        args.force,
        args.selection_slice,
        args.balance_by_family,
        target_post_fields,
    )
    if not rows:
        print("No candidate rows found.")
        return

    processed = 0
    updated = 0
    updated_cuda = 0
    consecutive_failures = 0
    report_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    batch_size = max(1, int(args.batch_commit))

    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            chunk_report_rows: List[Dict[str, Any]] = []
            stop_error: str | None = None
            for row in chunk:
                processed += 1
                entry = _process_row(nb, row, args, target_post_fields)
                if entry["status"] == "updated":
                    updated += 1
                    updated_cuda += 1
                if entry["status"] == "error":
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                chunk_report_rows.append(entry)
                print(
                    f"[{processed}/{len(rows)}] {row['result_id']} "
                    f"rapid={int(entry['rapid_replayed'])} post={int(entry['post_train_replayed'])} "
                    f"source={entry['source_device']} status={entry['status']} "
                    f"fields={entry['n_updates']}",
                    flush=True,
                )
                if int(
                    args.max_consecutive_failures
                ) > 0 and consecutive_failures >= int(args.max_consecutive_failures):
                    stop_error = (
                        "Stopping backpopulate run after "
                        f"{consecutive_failures} consecutive row failures. "
                        "This likely indicates a catastrophic tool/runtime issue; "
                        f"see report {args.report} for the exact failing rows."
                    )
                    break
            report_rows.extend(chunk_report_rows)
            _write_report(args.report, report_rows)
            if stop_error:
                raise RuntimeError(stop_error)

        _write_report(args.report, report_rows)
        _print_backpopulate_summary(
            processed=processed,
            total_rows=len(rows),
            updated=updated,
            updated_cuda=updated_cuda,
            report_path=args.report,
            elapsed=time.time() - t0,
        )
    except KeyboardInterrupt:
        _write_report(args.report, report_rows)
        _print_backpopulate_summary(
            processed=processed,
            total_rows=len(rows),
            updated=updated,
            updated_cuda=updated_cuda,
            report_path=args.report,
            elapsed=time.time() - t0,
            interrupted=True,
        )
        print("Keyboard interrupt received. Partial results were preserved.")
        return
    finally:
        nb.close()


if __name__ == "__main__":
    main()
