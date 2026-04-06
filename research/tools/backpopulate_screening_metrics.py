#!/usr/bin/env python3
"""Backpopulate missing screening/probe metrics onto existing backfill rows.

Rebuilds models from stored ``program_results.graph_json`` and replays only the
applicable screening stages for rows that already reached those stages.
Updates are written in place to the existing ``result_id``; no duplicate rows
are inserted.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
from research.scientist.notebook import LabNotebook
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner._helpers import (
    screening_probe_fields,
    screening_wikitext_fields,
)
from research.scientist.shared_utils import resolve_device
from research.synthesis.serializer import graph_from_json
from research.tools.backfill import store_probe_results

DB_PATH = Path("research/lab_notebook.db")
REPORT_PATH = Path("research/reports/backpopulate_screening_metrics.tsv")
DEFAULT_BATCH_COMMIT = 10
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_WORKER_TIMEOUT_SECONDS = None
DEFAULT_POST_TRAIN_STABILITY_RUNS = 1
RAPID_REQUIRED_FIELDS = (
    "rapid_screening_passed",
    "rapid_screening_elapsed_ms",
    "rapid_screening_steps_completed",
    "rapid_screening_max_steps",
)
POST_REQUIRED_FIELDS = (
    "wikitext_perplexity",
    "hellaswag_acc",
    "induction_auc",
    "binding_auc",
    "binding_composite",
)

POST_TARGET_ALIASES = {
    "full": "full",
    "one": "hellaswag",
    "hellaswag": "hellaswag",
    "two": "binding",
    "binding": "binding",
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


def _normalize_post_target(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in POST_TARGET_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unsupported --post-train-target={raw!r}; use one, two, all, full, hellaswag, or binding."
        )
    return POST_TARGET_ALIASES[key]


def _target_post_fields(target: str) -> tuple[str, ...]:
    if target == "hellaswag":
        return ("hellaswag_acc",)
    if target == "binding":
        return ("induction_auc", "binding_auc", "binding_composite")
    if target == "all":
        return ("hellaswag_acc", "induction_auc", "binding_auc", "binding_composite")
    return POST_REQUIRED_FIELDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backpopulate missing screening/probe metrics in-place"
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result-id", action="append", default=[])
    parser.add_argument("--from-report", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-rapid", action="store_true")
    parser.add_argument("--skip-post-train", action="store_true")
    parser.add_argument(
        "--post-train-target",
        type=_normalize_post_target,
        default="full",
        help=(
            "Which post-train metric family to backfill: "
            "'one'/'hellaswag', 'two'/'binding' (induction+binding AUCs), "
            "'all' (hellaswag+binding probes), or 'full' (legacy full post-train, including wikitext)."
        ),
    )
    parser.add_argument(
        "--allow-insufficient-learning-metrics",
        action="store_true",
        help=(
            "For backpopulate only, keep post-train screening/probe metrics even "
            "when CUDA replay fails only due to the validation-loss generalization gate."
        ),
    )
    parser.add_argument("--batch-commit", type=int, default=DEFAULT_BATCH_COMMIT)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=(
            "Stop the run after this many row-level failures in a row to avoid "
            "burning long CUDA batches on catastrophic tool/runtime failures."
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=_parse_optional_int,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="Hard timeout for a single isolated replay worker. Use 'none' or 'null' for no timeout.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--post-train-stability-runs",
        type=int,
        default=DEFAULT_POST_TRAIN_STABILITY_RUNS,
        help=(
            "Repeat post-train CUDA replay this many times and fail closed when "
            "key metrics drift beyond tolerance."
        ),
    )
    parser.add_argument(
        "--stability-wikitext-rel-tol",
        type=float,
        default=0.10,
        help="Maximum allowed relative drift for wikitext_perplexity.",
    )
    parser.add_argument(
        "--stability-hellaswag-abs-tol",
        type=float,
        default=0.05,
        help="Maximum allowed absolute drift for hellaswag_acc.",
    )
    parser.add_argument(
        "--stability-probe-abs-tol",
        type=float,
        default=0.01,
        help=("Maximum allowed absolute drift for induction/binding probe metrics."),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--isolate-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback-device",
        default="none",
        help="Disabled. CUDA failures are treated as unrecovered; keep this as 'none'.",
    )
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    return bool(int(row[key] or 0))


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
    if row["n_train_steps"] is None:
        return False
    if force:
        return True
    return any(row[key] is None for key in target_fields)


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
) -> List[sqlite3.Row]:
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
            pr.induction_auc,
            pr.binding_auc,
            pr.binding_composite,
            e.config_json,
            e.timestamp
        FROM program_results pr
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
        base += " AND e.experiment_type = 'backfill'"
    if not result_ids and not force:
        base += """
          AND (
            pr.rapid_screening_passed IS NULL OR
            pr.rapid_screening_elapsed_ms IS NULL OR
            pr.rapid_screening_steps_completed IS NULL OR
            pr.rapid_screening_max_steps IS NULL OR
            (
              pr.n_train_steps IS NOT NULL AND (
                pr.wikitext_perplexity IS NULL OR
                pr.hellaswag_acc IS NULL OR
                pr.induction_auc IS NULL OR
                pr.binding_auc IS NULL OR
                pr.binding_composite IS NULL
              )
            )
          )
        """
    base += " ORDER BY e.timestamp ASC, pr.result_id ASC"
    if limit > 0:
        base += f" LIMIT {int(limit)}"
    return conn.execute(base, tuple(params)).fetchall()


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
    config.screening_probe_seed = ExperimentRunner._stable_seed(
        row["result_id"],
        row["graph_fingerprint"],
        "backpopulate_screening_probe",
    )
    return config


def _row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


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
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


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
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


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
    if hs.get("hellaswag_acc") is not None:
        updates["hellaswag_acc"] = hs.get("hellaswag_acc")
    if hs.get("hellaswag_status") is not None:
        updates["hellaswag_status"] = hs.get("hellaswag_status")
    if hs.get("hellaswag_total") is not None:
        updates["hellaswag_n_examples"] = hs.get("hellaswag_total")
    return updates


def _select_updates(
    row: sqlite3.Row, updates: Dict[str, Any], force: bool
) -> Dict[str, Any]:
    if force:
        return {k: v for k, v in updates.items() if v is not None}
    selected: Dict[str, Any] = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key not in row.keys() or row[key] is None:
            selected[key] = value
    return selected


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
) -> Dict[str, Any]:
    row = payload
    target_post_fields = _target_post_fields(post_train_target)
    rapid_needed = (not skip_rapid) and _needs_rapid(row, force)
    post_needed = (not skip_post_train) and _needs_post_train(
        row, force, target_post_fields
    )
    updates: Dict[str, Any] = {}
    config = _build_run_config(row, device)
    # Replay only the missing post-train metrics for this row.
    target_post_field_set = set(target_post_fields)
    config.skip_screening_wikitext = (
        "wikitext_perplexity" not in target_post_field_set
        or row.get("wikitext_perplexity") is not None
    )
    config.skip_screening_hellaswag = (
        "hellaswag_acc" not in target_post_field_set
        or row.get("hellaswag_acc") is not None
    )
    target_probe_fields = {"induction_auc", "binding_auc", "binding_composite"}
    config.skip_binding_probes = not bool(
        target_probe_fields & target_post_field_set
    ) or all(row.get(key) is not None for key in target_probe_fields)
    if rapid_needed:
        updates.update(
            _run_rapid(
                str(row["graph_json"]),
                config,
                device,
                str(row["result_id"]),
            )
        )
    if post_needed:
        post_runs: List[Dict[str, Any]] = []
        for _ in range(max(1, int(post_train_stability_runs))):
            runner = ExperimentRunner(notebook_path=str(DB_PATH))
            try:
                post_runs.append(
                    _run_post_train(
                        runner,
                        str(row["graph_json"]),
                        config,
                        device,
                        str(row["result_id"]),
                        allow_insufficient_learning_metrics=allow_insufficient_learning_metrics,
                    )
                )
            finally:
                close_runner = getattr(runner, "close", None)
                if callable(close_runner):
                    close_runner()
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
    row: sqlite3.Row, args: argparse.Namespace
) -> Dict[str, Any]:
    payload = _row_to_payload(row)
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


def main() -> None:
    args = _parse_args()
    if args.worker_payload and args.worker_output:
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
            )
            result["ok"] = 1
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": 0,
                "error": str(exc),
            }
        args.worker_output.write_text(json.dumps(result), encoding="utf-8")
        return

    nb = LabNotebook(str(args.db))
    nb.conn.row_factory = sqlite3.Row
    result_ids = _candidate_result_ids(args)
    rows = _fetch_rows(nb.conn, result_ids, args.limit, args.force)
    if not rows:
        print("No candidate rows found.")
        return

    processed = 0
    updated = 0
    updated_cuda = 0
    consecutive_failures = 0
    report_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    target_post_fields = _target_post_fields(args.post_train_target)

    for start in range(0, len(rows), max(1, int(args.batch_commit))):
        chunk = rows[start : start + max(1, int(args.batch_commit))]
        chunk_report_rows: List[Dict[str, Any]] = []
        stop_error: str | None = None
        with nb.batch():
            for row in chunk:
                processed += 1
                rapid_needed = (not args.skip_rapid) and _needs_rapid(row, args.force)
                post_needed = (not args.skip_post_train) and _needs_post_train(
                    row, args.force, target_post_fields
                )
                status = "skipped"
                err = ""
                updates: Dict[str, Any] = {}
                source_device = str(args.device)
                try:
                    if args.isolate_subprocess:
                        worker = _run_worker_subprocess(row, args)
                        rapid_needed = int(worker.get("rapid_needed") or 0)
                        post_needed = int(worker.get("post_needed") or 0)
                        updates = dict(worker.get("updates") or {})
                    else:
                        worker = _evaluate_row_payload(
                            payload=_row_to_payload(row),
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
                        )
                        rapid_needed = int(worker.get("rapid_needed") or 0)
                        post_needed = int(worker.get("post_needed") or 0)
                        updates = dict(worker.get("updates") or {})
                    if updates and not args.dry_run:
                        store_probe_results(
                            nb,
                            str(row["result_id"]),
                            updates,
                            write_leaderboard=True,
                        )
                        updated += 1
                        updated_cuda += 1
                        status = "updated"
                    elif updates:
                        status = "would_update"
                    else:
                        status = "no_missing_fields"
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                    status = "error"
                if status == "error":
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                chunk_report_rows.append(
                    {
                        "result_id": row["result_id"],
                        "graph_fingerprint": row["graph_fingerprint"],
                        "rapid_replayed": rapid_needed,
                        "post_train_replayed": post_needed,
                        "source_device": source_device,
                        "updated_fields": ",".join(sorted(updates.keys())),
                        "status": status,
                        "error": err[:240],
                    }
                )
                print(
                    f"[{processed}/{len(rows)}] {row['result_id']} "
                    f"rapid={int(rapid_needed)} post={int(post_needed)} "
                    f"source={source_device} status={status} fields={len(updates)}",
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
    elapsed = time.time() - t0
    print(
        f"Processed {processed} rows, updated {updated} "
        f"(cuda={updated_cuda}), "
        f"report={args.report}, elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
