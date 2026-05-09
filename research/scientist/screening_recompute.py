from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

import torch

from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.notebook import LabNotebook
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.notebook.leaderboard_maintenance import (
    sync_fingerprint_leaderboard,
)
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner._helpers import (
    screening_probe_fields,
    screening_wikitext_fields,
)
from research.scientist.runner.execution_triage import run_triage
from research.scientist.runner.shared import get_shared_runner
from research.scientist.shared_utils import resolve_device
from research.synthesis.serializer import graph_from_json
from research.tools.backfill import (
    _fingerprint_one,
    prefetch_program_results,
    rescore_entry,
    store_probe_results,
)
from research.tools.backpopulate_screening_metrics import (
    _deterministic_compile_seed,
    _recover_hellaswag_after_gate_failure,
    _run_rapid,
)

SCREENING_LOSS_KEYS = (
    "initial_loss",
    "final_loss",
    "loss_ratio",
    "loss_improvement_rate",
    "validation_loss",
    "validation_loss_ratio",
    "generalization_gap",
    "discovery_loss",
    "discovery_loss_ratio",
)


def load_program_row(nb: LabNotebook, result_id: str) -> Dict[str, Any] | None:
    row = nb.conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?",
        (str(result_id),),
    ).fetchone()
    if not row:
        return None
    payload = dict(row)
    if "graph_json" in payload:
        payload["graph_json"] = resolve_graph_json_value(
            nb.conn,
            nb.db_path,
            payload["graph_json"],
        )
    return payload


def load_run_config(nb: LabNotebook, program: Dict[str, Any]) -> RunConfig:
    exp_id = program.get("experiment_id")
    config_json = None
    if exp_id:
        exp_row = nb.conn.execute(
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        if exp_row:
            config_json = exp_row["config_json"]

    config_dict = json.loads(config_json) if config_json else {}
    valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = RunConfig(**filtered)

    graph_json = program.get("graph_json")
    if graph_json:
        graph = graph_from_json(graph_json)
        graph_dim = getattr(graph, "model_dim", None)
        if graph_dim and config.model_dim != graph_dim:
            config.model_dim = int(graph_dim)
    return config


def _loss_updates_from_training(s1_result: Dict[str, Any]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for key in SCREENING_LOSS_KEYS:
        value = s1_result.get(key)
        if value is not None:
            updates[key] = value
    return updates


def _run_full_post_train(
    *,
    notebook_path: Path,
    program: Dict[str, Any],
    config: RunConfig,
    device: str,
    allow_insufficient_learning_metrics: bool,
) -> Dict[str, Any]:
    graph_json = str(program.get("graph_json") or "")
    result_id = str(program.get("result_id") or "")
    graph = graph_from_json(graph_json)
    compile_seed = ExperimentRunner._stable_seed(
        result_id, graph.fingerprint(), "program_detail_full_recompute_compile"
    )
    with _deterministic_compile_seed(device, compile_seed):
        try:
            model = compile_model(
                [graph] * int(config.n_layers),
                vocab_size=int(config.vocab_size),
                max_seq_len=config.max_seq_len,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(
                exc, torch.cuda.OutOfMemoryError
            ):
                # Native compilation OOM — fall back to Python IR executor
                import logging

                logging.getLogger(__name__).warning(
                    "Native compile OOM for %s, falling back to IR executor: %s",
                    result_id,
                    exc,
                )
                torch.cuda.empty_cache()
                from research.synthesis.compiler import (
                    compile_model as compile_model_ir,
                )

                model = compile_model_ir(
                    [graph] * int(config.n_layers),
                    vocab_size=int(config.vocab_size),
                    max_seq_len=config.max_seq_len,
                )
            else:
                raise

    # Process-wide singleton: amortizes CodeHealer init, baseline transformer,
    # SSE handler, and corpus batcher across all calls in a batch loop. Lives
    # for process lifetime; cleaned up via atexit in runner.shared.
    runner = get_shared_runner(str(notebook_path))
    dev = resolve_device(device)
    try:
        s1_result = runner._micro_train(
            model,
            config,
            dev,
            seed=runner._stable_seed(
                result_id, graph.fingerprint(), "program_detail_full_recompute"
            ),
            graph_json=graph_json,
        )
        updates: Dict[str, Any] = {}
        updates.update(_loss_updates_from_training(s1_result))
        updates.update(screening_wikitext_fields(s1_result))
        updates.update(screening_probe_fields(s1_result))
        if s1_result.get("hellaswag_acc") is not None:
            updates["hellaswag_acc"] = s1_result.get("hellaswag_acc")
        if s1_result.get("hellaswag_status") is not None:
            updates["hellaswag_status"] = s1_result.get("hellaswag_status")
        if s1_result.get("hellaswag_n_examples") is not None:
            updates["hellaswag_n_examples"] = s1_result.get("hellaswag_n_examples")
        if s1_result.get("hellaswag_metric_version") is not None:
            updates["hellaswag_metric_version"] = s1_result.get(
                "hellaswag_metric_version"
            )
        if s1_result.get("hellaswag_tokenizer_mode") is not None:
            updates["hellaswag_tokenizer_mode"] = s1_result.get(
                "hellaswag_tokenizer_mode"
            )
        if s1_result.get("hellaswag_tiktoken_encoding") is not None:
            updates["hellaswag_tiktoken_encoding"] = s1_result.get(
                "hellaswag_tiktoken_encoding"
            )

        tolerate_gate_failure = (
            allow_insufficient_learning_metrics
            and s1_result.get("error")
            and s1_result.get("error_type") == "insufficient_learning"
        )
        if tolerate_gate_failure:
            updates.update(
                _recover_hellaswag_after_gate_failure(
                    model=model,
                    config=config,
                    device=str(dev),
                )
            )
            # Also recover BLiMP and binding probes on gate failure
            # (these are zero-shot / lightweight and don't need S1 pass)
            try:
                from research.eval.blimp_eval import evaluate_blimp

                blimp = evaluate_blimp(
                    model,
                    vocab_size=int(config.vocab_size),
                    device=str(dev),
                    n_per_subtask=50,
                    timeout_s=120,
                )
                updates["blimp_overall_accuracy"] = blimp.overall_accuracy
                updates["blimp_status"] = "ok"
            except Exception:
                pass
            try:
                from research.eval.binding_pipeline import run_screening_binding_probes

                bp = run_screening_binding_probes(model, device=str(dev))
                updates.update(screening_probe_fields(bp))
            except Exception:
                pass
        elif s1_result.get("error"):
            raise RuntimeError(
                f"micro_train_failed: {s1_result.get('error')} "
                f"type={s1_result.get('error_type')}"
            )

        triage = run_triage(model, graph, s1_result, config.model_dim)
        if triage:
            updates.update(triage)
        updates["train_budget_steps"] = int(config.stage1_steps)
        return updates
    finally:
        # Do NOT close runner — it's a process-wide singleton owned by
        # runner.shared and torn down via atexit.
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def recompute_screening_metrics(
    *,
    nb: LabNotebook,
    notebook_path: Path,
    result_id: str,
    device: str = "cpu",
    include_rapid: bool = True,
    include_fingerprint: bool = True,
    include_post_train: bool = True,
    allow_insufficient_learning_metrics: bool = True,
    provenance_source: str = "screening_recompute",
) -> Dict[str, Any]:
    program = load_program_row(nb, result_id)
    if not program:
        raise ValueError(f"Program not found: {result_id}")
    if not program.get("graph_json"):
        raise ValueError(f"No graph_json for result_id={result_id}")

    updates: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    config = load_run_config(nb, program)

    if include_rapid:
        try:
            updates.update(
                _run_rapid(
                    str(program["graph_json"]),
                    config,
                    device,
                    str(result_id),
                )
            )
        except Exception as exc:
            errors["rapid"] = str(exc)

    if include_fingerprint:
        try:
            updates.update(
                _fingerprint_one(
                    str(result_id),
                    str(program["graph_json"]),
                    device,
                )
            )
        except Exception as exc:
            errors["fingerprint"] = str(exc)

    if include_post_train:
        try:
            updates.update(
                _run_full_post_train(
                    notebook_path=notebook_path,
                    program=program,
                    config=config,
                    device=device,
                    allow_insufficient_learning_metrics=allow_insufficient_learning_metrics,
                )
            )
        except Exception as exc:
            errors["post_train"] = str(exc)

    if updates:
        store_probe_results(
            nb=nb,
            result_id=str(result_id),
            updates=updates,
            write_leaderboard=True,
            provenance_context={
                "kind": "screening_recompute",
                "source": provenance_source,
                "device": str(device),
                "include_rapid": bool(include_rapid),
                "include_fingerprint": bool(include_fingerprint),
                "include_post_train": bool(include_post_train),
            },
        )
        sync_fingerprint_leaderboard(nb, str(result_id))
        fingerprint_row = nb.conn.execute(
            "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
            (str(result_id),),
        ).fetchone()
        graph_fingerprint = (
            str(fingerprint_row["graph_fingerprint"])
            if fingerprint_row and fingerprint_row["graph_fingerprint"]
            else ""
        )
        leaderboard_rows = []
        if graph_fingerprint:
            leaderboard_rows = nb.conn.execute(
                """
                SELECT l.entry_id, l.result_id, l.is_reference
                FROM leaderboard l
                JOIN program_results pr ON pr.result_id = l.result_id
                WHERE pr.graph_fingerprint = ?
                """,
                (graph_fingerprint,),
            ).fetchall()
        else:
            entry = nb.conn.execute(
                "SELECT entry_id, result_id, is_reference FROM leaderboard WHERE result_id = ?",
                (str(result_id),),
            ).fetchone()
            if entry:
                leaderboard_rows = [entry]

        if leaderboard_rows:
            pr_cache = prefetch_program_results(
                nb.conn,
                [str(row["result_id"]) for row in leaderboard_rows],
            )
            pr_cache[str(result_id)] = load_program_row(nb, str(result_id)) or {}
            for entry in leaderboard_rows:
                entry_result_id = str(entry["result_id"])
                extra_updates = updates if entry_result_id == str(result_id) else None
                rescore_entry(
                    nb,
                    str(entry["entry_id"]),
                    entry_result_id,
                    bool(entry["is_reference"]),
                    pr_cache,
                    pr_updates=extra_updates,
                )
        nb.conn.commit()

    return {
        "status": "ok" if updates else "noop",
        "result_id": str(result_id),
        "mode": "full_screening_recompute",
        "updates": updates,
        "errors": errors,
    }
