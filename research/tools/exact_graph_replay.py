"""Exact-graph screening replay for targeted label cleanup.

Replays stored graph_json rows through the standard screening funnel without
mutation so ambiguous duplicate groups can be re-measured directly.

Usage:
    python -m research.tools.exact_graph_replay --result-id abc123 --result-id def456
    python -m research.tools.exact_graph_replay --triage-json research/reports/label_triage_20260403_100810.json --ambiguous-index 1 --repeat-per-source 2
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from research.defaults import RUNS_DB
from research.orchestrator.executor import JobResult
from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner.execution_screening import (
    INITIAL_LOSS_THRESHOLD,
    _record_screening_failure,
)
from research.scientist.runner._helpers import clear_gpu_memory, graph_routing_ops
from research.scientist.shared_utils import resolve_device
from research.training.loss_ops import next_token_cross_entropy
from research.synthesis.serializer import graph_from_json
from research.tools._db_maintenance import connect_readonly
from research.scientist.native_runner import (
    compile_model_native_first as compile_model,
)
from research.scientist.notebook.graph_artifacts import (
    is_nonempty_graph_json,
    resolve_graph_json_value,
)

_DEFAULT_DB = Path(RUNS_DB)


def _load_ambiguous_result_ids(triage_json: Path, ambiguous_index: int) -> List[str]:
    payload = json.loads(triage_json.read_text())
    groups = payload.get("ambiguous_groups") or []
    index = max(1, int(ambiguous_index)) - 1
    if index >= len(groups):
        raise ValueError(
            f"Ambiguous index {ambiguous_index} out of range for {len(groups)} groups"
        )
    return [
        str(rid) for rid in groups[index].get("result_ids") or [] if str(rid).strip()
    ]


def _fetch_source_rows(
    db_path: Path, result_ids: Sequence[str]
) -> List[Dict[str, Any]]:
    ids = [str(rid).strip() for rid in result_ids if str(rid).strip()]
    if not ids:
        return []
    conn = connect_readonly(db_path)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT result_id, graph_json, graph_fingerprint, loss_ratio, stage1_passed,
                   stage05_passed, timestamp
            FROM program_results
            WHERE result_id IN ({placeholders})
              AND TRIM(COALESCE(graph_json, '')) <> ''
              AND graph_json <> '{{}}'""",
        tuple(ids),
    ).fetchall()
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        if not is_nonempty_graph_json(row["graph_json"]):
            continue
        payload = dict(row)
        payload["graph_json"] = resolve_graph_json_value(
            conn, db_path, row["graph_json"]
        )
        graph_json = payload["graph_json"].strip()
        if not graph_json or graph_json == "{}":
            continue
        try:
            if not isinstance(json.loads(graph_json), dict):
                continue
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        resolved_rows.append(payload)
    conn.close()
    row_by_id = {str(row["result_id"]): row for row in resolved_rows}
    ordered = [row_by_id[rid] for rid in ids if rid in row_by_id]
    deduped: List[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for row in ordered:
        fingerprint = str(row.get("graph_fingerprint") or "").strip()
        dedup_key = fingerprint or f"result:{str(row.get('result_id') or '').strip()}"
        if dedup_key in seen_fingerprints:
            continue
        seen_fingerprints.add(dedup_key)
        deduped.append(row)
    return deduped


def _expand_replays(
    rows: Sequence[Dict[str, Any]], repeat_per_source: int
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        for replay_idx in range(max(1, int(repeat_per_source))):
            expanded.append(
                {
                    **row,
                    "replay_index": replay_idx,
                }
            )
    return expanded


def _build_config(device: str, repeat_count: int) -> RunConfig:
    return RunConfig(
        device=device,
        n_programs=max(1, int(repeat_count)),
        model_source="exact_graph_replay",
        persist_screening_failures=True,
        gbm_prescreener_enabled=False,
        progressive_screening=True,
    )


def _apply_fast_replay_budget(config: RunConfig) -> None:
    config.stage1_steps = min(int(config.stage1_steps), 80)
    config.stage1_batch_size = min(int(config.stage1_batch_size), 4)
    config.max_seq_len = min(int(config.max_seq_len), 128)
    config.stage1_compute_val_loss = False
    config.stage1_compute_discovery_loss = False
    config.stage1_val_batches = 0
    config.stage1_discovery_batches = 0


def _evaluate_exact_replay(
    runner: ExperimentRunner,
    nb: LabNotebook,
    exp_id: str,
    config: RunConfig,
    replay_rows: Sequence[Dict[str, Any]],
    verbose: bool = False,
    independent_sample: bool = False,
    candidate_confirmation: bool = False,
) -> Dict[str, Any]:
    dev = resolve_device(config.device)
    dev_str = str(dev)
    results: Dict[str, Any] = {
        "total": len(replay_rows),
        "stage0_passed": 0,
        "stage05_passed": 0,
        "rapid_screening_killed": 0,
        "rapid_screening_kill_reasons": {},
        "stage1_passed": 0,
        "best_loss_ratio": None,
        "funnel_counts": {
            "raw_generated": len(replay_rows),
            "post_batch_dedup": len(replay_rows),
            "judgment_filtered": 0,
            "post_judgment": len(replay_rows),
            "screening_considered": len(replay_rows),
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
        "replayed_result_ids": [row["result_id"] for row in replay_rows],
        "replay_unique_sources": len({row["result_id"] for row in replay_rows}),
        "_s0_op_counts": {},
    }

    runner._live_training_context = {"exp_id": exp_id, "phase": "exact_graph_replay"}
    for i, row in enumerate(replay_rows):
        graph = graph_from_json(str(row["graph_json"]))
        runner._update_progress(
            status="evaluating",
            current_program=i + 1,
            total_programs=len(replay_rows),
            current_fingerprint=graph.fingerprint()[:10],
            aria_message=f"{runner.aria.NAME}: Replaying {graph.fingerprint()[:8]}...",
        )
        program_metrics: Dict[str, Any] = {
            "model_source": "exact_graph_replay",
            "source_result_id": row["result_id"],
            "source_graph_fingerprint": row.get("graph_fingerprint"),
            "replay_index": row.get("replay_index", 0),
            "source_loss_ratio": row.get("loss_ratio"),
            "candidate_confirmation": bool(candidate_confirmation),
            # When True, _persist_program_row writes a NEW program_results
            # row (independent sample for CV math).  When False / absent,
            # it patches the source row in place (legacy fix-incomplete-
            # data behavior).  See dashboard_orchestrator
            # ._persist_program_row for the gate.
            "intentional_independent_sample": bool(independent_sample),
        }
        # Bypass the duplicate-fingerprint guard when we explicitly want
        # an independent sample.  Without this, record_program_result
        # raises DuplicateFingerprintError because a row already exists
        # for the source graph_fingerprint.  Stamping the reason gives
        # auditable provenance for the new row.
        if independent_sample:
            program_metrics["intentional_rerun_reason"] = (
                "exact_graph_replay_independent_sample"
            )
        results["funnel_counts"]["stage0_attempted"] = (
            int(results["funnel_counts"].get("stage0_attempted", 0)) + 1
        )
        try:
            if verbose:
                print(
                    f"[{i + 1}/{len(replay_rows)}] replay {row['result_id']} "
                    f"fp={(row.get('graph_fingerprint') or '')[:12]}",
                    flush=True,
                )
            layer_graphs = [graph] * config.n_layers
            phase1_vocab = (
                config.qualifying_vocab_size
                if config.progressive_screening
                and config.vocab_size > config.qualifying_vocab_size
                else config.vocab_size
            )
            model = compile_model(
                layer_graphs,
                vocab_size=phase1_vocab,
                max_seq_len=config.max_seq_len,
            )
            sandbox_result = runner._safe_eval_for_stage(
                model,
                stage_tag="candidate_screening",
                batch_size=2,
                seq_len=min(128, config.max_seq_len),
                vocab_size=phase1_vocab,
                device=dev_str,
                timeout_seconds=30,
            )
            program_metrics.update(runner._extract_sandbox_metrics(sandbox_result))
            program_metrics["param_count"] = sandbox_result.param_count
            s0_passed = bool(sandbox_result.passed)
            s05_passed = bool(
                sandbox_result.stability_score >= config.stage05_stability_threshold
                and sandbox_result.causality_passed
            )
            for node in graph.nodes.values():
                if not node.is_input and node.op_name:
                    counts = results["_s0_op_counts"].setdefault(
                        node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0}
                    )
                    counts["n_used"] += 1
                    if s0_passed:
                        counts["n_s0"] += 1
                    if s05_passed:
                        counts["n_s05"] += 1
            if s0_passed:
                results["stage0_passed"] += 1
            if s05_passed:
                results["stage05_passed"] += 1
            if not s0_passed or not s05_passed:
                if verbose:
                    print(
                        f"  fail early: s0={int(s0_passed)} s05={int(s05_passed)}",
                        flush=True,
                    )
                if not s0_passed:
                    results["funnel_counts"]["dropped_stage0"] += 1
                else:
                    results["funnel_counts"]["dropped_stage05"] += 1
                _record_screening_failure(
                    nb=nb,
                    exp_id=exp_id,
                    graph=graph,
                    source_result_id=row["result_id"],
                    stage0_passed=s0_passed,
                    stage05_passed=s05_passed,
                    error_type=sandbox_result.error_type or "unknown",
                    error_message=(sandbox_result.error or "")[:240] or None,
                    stage_at_death="stage0" if not s0_passed else "stage05",
                    stability_score=sandbox_result.stability_score,
                )
                continue

            try:
                s075_dev = torch.device(dev_str)
                model.train()
                s075_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                s075_ids = torch.randint(0, phase1_vocab, (4, 64), device=s075_dev)
                with torch.amp.autocast(
                    device_type=s075_dev.type,
                    dtype=torch.bfloat16,
                    enabled=(s075_dev.type == "cuda"),
                ):
                    s075_logits = model(s075_ids)
                    s075_loss = next_token_cross_entropy(
                        s075_logits, s075_ids, s075_logits.size(-1)
                    )
                initial_loss = float(s075_loss.item())
                program_metrics["s075_initial_loss"] = initial_loss
                if (
                    not torch.isnan(torch.tensor(initial_loss))
                    and not torch.isinf(torch.tensor(initial_loss))
                    and initial_loss > INITIAL_LOSS_THRESHOLD
                ):
                    if verbose:
                        print(
                            f"  fail s075: initial_loss={initial_loss:.2f}",
                            flush=True,
                        )
                    results["funnel_counts"]["dropped_s075_high_init"] += 1
                    _record_screening_failure(
                        nb=nb,
                        exp_id=exp_id,
                        graph=graph,
                        source_result_id=row["result_id"],
                        stage0_passed=True,
                        stage05_passed=True,
                        error_type="high_initial_loss",
                        error_message=(
                            f"initial_loss={initial_loss:.4f} > "
                            f"{INITIAL_LOSS_THRESHOLD:.4f}"
                        ),
                        stage_at_death="stage075",
                        stability_score=sandbox_result.stability_score,
                    )
                    del s075_opt
                    continue
                s075_opt.zero_grad(set_to_none=True)
                del s075_opt
            except Exception:
                pass

            from research.eval.screening_rapid import RapidScreeningCheck

            rapid = RapidScreeningCheck()
            results["funnel_counts"]["rapid_screen_attempted"] += 1
            rapid_result = rapid.run(
                model,
                vocab_size=phase1_vocab,
                seq_len=min(128, config.max_seq_len),
                batch_size=2,
                device=dev_str,
            )
            program_metrics["rapid_screening_passed"] = rapid_result.passed
            program_metrics["rapid_screening_elapsed_ms"] = rapid_result.elapsed_ms
            if not rapid_result.passed:
                if verbose:
                    print(
                        f"  fail rapid: {rapid_result.kill_reason or 'unknown'}",
                        flush=True,
                    )
                results["rapid_screening_killed"] += 1
                results["funnel_counts"]["dropped_rapid_screening"] += 1
                kill_reason = rapid_result.kill_reason or "unknown"
                results["rapid_screening_kill_reasons"][kill_reason] = (
                    results["rapid_screening_kill_reasons"].get(kill_reason, 0) + 1
                )
                _record_screening_failure(
                    nb=nb,
                    exp_id=exp_id,
                    graph=graph,
                    source_result_id=row["result_id"],
                    stage0_passed=True,
                    stage05_passed=True,
                    error_type="rapid_screening_error",
                    error_message=kill_reason[:240],
                    stage_at_death="rapid_screening",
                    stability_score=sandbox_result.stability_score,
                )
                continue

            if (
                config.progressive_screening
                and config.vocab_size > config.qualifying_vocab_size
            ):
                del model
                clear_gpu_memory()
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                program_metrics["progressive_phase2_compiled"] = True

            routing_ops = graph_routing_ops(graph)
            if routing_ops:
                program_metrics["routing_fast_lane_applied"] = 1

            results["funnel_counts"]["stage1_queued"] += 1
            if verbose:
                print(
                    "  stage1 micro-train "
                    f"(steps={config.stage1_steps}, batch={config.stage1_batch_size})",
                    flush=True,
                )
            s1_result = runner._micro_train(
                model,
                config,
                dev,
                seed=runner._stable_seed(
                    exp_id,
                    row["result_id"],
                    row.get("replay_index", 0),
                    "exact_graph_replay",
                ),
            )
            jr = JobResult(
                index=i,
                s1_result=s1_result,
                payload={"metrics": program_metrics, "graph": graph},
                telemetry={},
            )
            runner._record_orchestrator_result(jr, nb, exp_id, results, config)
            if verbose:
                print(
                    "  done: "
                    f"passed={int(bool(s1_result.get('passed', False)))} "
                    f"loss_ratio={s1_result.get('loss_ratio')}",
                    flush=True,
                )
        except Exception as exc:
            if verbose:
                print(f"  error: {exc}", flush=True)
            program_metrics["error_type"] = "exact_replay_error"
            program_metrics["error_message"] = str(exc)[:240]
            _record_screening_failure(
                nb=nb,
                exp_id=exp_id,
                graph=graph,
                source_result_id=row["result_id"],
                stage0_passed=False,
                stage05_passed=False,
                error_type="exact_replay_error",
                error_message=str(exc)[:240],
                stage_at_death="stage0",
                stability_score=None,
            )
        finally:
            clear_gpu_memory()

    runner._live_training_context = None
    results["elapsed_seconds"] = 0.0
    return results


def run_exact_replay(
    *,
    db_path: Path,
    result_ids: Sequence[str],
    repeat_per_source: int,
    device: str,
    hypothesis: str,
    fast: bool = False,
    verbose: bool = False,
    independent_sample: bool = False,
    candidate_confirmation: bool = False,
    stage1_steps: int | None = None,
) -> str:
    rows = _fetch_source_rows(db_path, result_ids)
    if not rows:
        raise ValueError("No replayable source rows found for the requested result_ids")
    replay_rows = _expand_replays(rows, repeat_per_source)

    runner = ExperimentRunner(str(db_path))
    config = _build_config(device=device, repeat_count=len(replay_rows))
    if stage1_steps is not None:
        config.stage1_steps = max(1, int(stage1_steps))
    if fast:
        _apply_fast_replay_budget(config)
    config, _ = runner.prescreen_run_config(config, mode="single", auto_harden=True)
    runner._ensure_math_spaces()

    nb = LabNotebook(str(db_path))
    exp_config = config.to_dict()
    exp_config["source_result_ids"] = [
        str(row.get("result_id") or "").strip()
        for row in rows
        if str(row.get("result_id") or "").strip()
    ]
    exp_config["source_graph_fingerprints"] = [
        str(row.get("graph_fingerprint") or "").strip()
        for row in rows
        if str(row.get("graph_fingerprint") or "").strip()
    ]
    exp_id = nb.start_experiment(
        "exact_graph_replay",
        exp_config,
        hypothesis=hypothesis,
    )
    try:
        if verbose:
            print(
                f"experiment {exp_id}: replaying {len(replay_rows)} rows",
                flush=True,
            )
        results = _evaluate_exact_replay(
            runner,
            nb,
            exp_id,
            config,
            replay_rows,
            verbose=verbose,
            independent_sample=independent_sample,
            candidate_confirmation=candidate_confirmation,
        )
        results["elapsed_seconds"] = float(results.get("elapsed_seconds") or 0.0)
        nb.complete_experiment(
            experiment_id=exp_id,
            results=results,
            aria_summary=(
                "Exact graph replay complete: "
                f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1"
            ),
        )
        s0_op_counts = results.pop("_s0_op_counts", None)
        with nb.batch():
            if s0_op_counts:
                nb.merge_op_failure_counts(s0_op_counts)
            else:
                nb.update_op_success_rates(exp_id)
            nb.strip_graph_json_for_failures(exp_id)
            nb.update_failure_signatures(exp_id)
    except Exception:
        nb.fail_experiment(exp_id, "exact_graph_replay_failed")
        raise
    finally:
        nb.close()
    return exp_id


def start_exact_replay_async(
    *,
    db_path: Path,
    result_ids: Sequence[str],
    repeat_per_source: int,
    device: str,
    hypothesis: str,
    fast: bool = False,
    verbose: bool = False,
    independent_sample: bool = False,
    candidate_confirmation: bool = False,
    stage1_steps: int | None = None,
) -> str:
    """Launch exact replay in a background thread and return the experiment id."""
    rows = _fetch_source_rows(db_path, result_ids)
    if not rows:
        raise ValueError("No replayable source rows found for the requested result_ids")
    replay_rows = _expand_replays(rows, repeat_per_source)

    runner = ExperimentRunner(str(db_path))
    config = _build_config(device=device, repeat_count=len(replay_rows))
    if stage1_steps is not None:
        config.stage1_steps = max(1, int(stage1_steps))
    if fast:
        _apply_fast_replay_budget(config)
    config, _ = runner.prescreen_run_config(config, mode="single", auto_harden=True)
    runner._ensure_math_spaces()

    init_nb = LabNotebook(str(db_path))
    exp_config = config.to_dict()
    exp_config["source_result_ids"] = [
        str(row.get("result_id") or "").strip()
        for row in rows
        if str(row.get("result_id") or "").strip()
    ]
    exp_config["source_graph_fingerprints"] = [
        str(row.get("graph_fingerprint") or "").strip()
        for row in rows
        if str(row.get("graph_fingerprint") or "").strip()
    ]
    exp_id = init_nb.start_experiment(
        "exact_graph_replay",
        exp_config,
        hypothesis=hypothesis,
    )
    init_nb.close()

    def _worker() -> None:
        nb = LabNotebook(str(db_path))
        try:
            results = _evaluate_exact_replay(
                runner,
                nb,
                exp_id,
                config,
                replay_rows,
                verbose=verbose,
                independent_sample=independent_sample,
                candidate_confirmation=candidate_confirmation,
            )
            results["elapsed_seconds"] = float(results.get("elapsed_seconds") or 0.0)
            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=(
                    "Exact graph replay complete: "
                    f"{results.get('stage1_passed', 0)}/{results.get('total', 0)} S1"
                ),
            )
            s0_op_counts = results.pop("_s0_op_counts", None)
            with nb.batch():
                if s0_op_counts:
                    nb.merge_op_failure_counts(s0_op_counts)
                else:
                    nb.update_op_success_rates(exp_id)
                nb.strip_graph_json_for_failures(exp_id)
                nb.update_failure_signatures(exp_id)
        except Exception:
            nb.fail_experiment(exp_id, "exact_graph_replay_failed")
            raise
        finally:
            nb.close()

    thread = threading.Thread(
        target=_worker,
        name=f"exact_graph_replay:{exp_id[:8]}",
        daemon=True,
    )
    thread.start()
    return exp_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay exact stored graphs through screening"
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--result-id", action="append", default=[])
    parser.add_argument("--triage-json", type=Path, default=None)
    parser.add_argument("--ambiguous-index", type=int, default=1)
    parser.add_argument("--repeat-per-source", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_ids = list(args.result_id or [])
    if args.triage_json is not None:
        result_ids.extend(
            _load_ambiguous_result_ids(args.triage_json, args.ambiguous_index)
        )
    if not result_ids:
        raise SystemExit("Provide --result-id or --triage-json/--ambiguous-index")
    hypothesis = (
        "Exact graph replay for label cleanup: "
        f"{len(result_ids)} source graphs, repeat_per_source={int(args.repeat_per_source)}"
    )
    exp_id = run_exact_replay(
        db_path=args.db,
        result_ids=result_ids,
        repeat_per_source=args.repeat_per_source,
        device=args.device,
        hypothesis=hypothesis,
        fast=bool(args.fast),
        verbose=bool(args.verbose),
    )
    print(exp_id)


if __name__ == "__main__":
    main()
