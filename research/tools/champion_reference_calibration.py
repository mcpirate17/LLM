"""Run known-good GPT-2 controls through champion-scale training.

The tool is intentionally narrow: it trains reference GPT-2 layer stacks with
the same micro-train and champion hard-probe code used by dashboard scale-up,
persists rows/curves/checkpoints, and writes a JSONL trace for observability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from research.defaults import RUNTIME_DIR_ABS
from research.scientist.json_utils import json_safe
from research.scientist.native_runner import compile_model_native_first
from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner._helpers import (
    clear_gpu_memory,
    program_result_kwargs_from_s1,
)
from research.scientist.runner.execution_champion_confirmation import (
    ChampionConfirmationEvaluator,
)
from research.synthesis.reference_architectures import build_reference, list_references
from research.synthesis.serializer import graph_to_json
from research.training.checkpointing import CheckpointManager

LOGGER = logging.getLogger(__name__)
RUNTIME_DIR = RUNTIME_DIR_ABS / "champion_reference_calibration"


def calibration_fingerprint(
    layer_fingerprint: str,
    *,
    arch: str,
    layers: int,
    steps: int,
    model_dim: int,
    seq_len: int,
    batch_size: int,
) -> str:
    """Stable run key for a calibration variant.

    This is deliberately not the persisted graph_fingerprint.  Champion
    calibration varies layer count, step budget, and batch settings, but those
    are experiment/run dimensions under the same parent graph.
    """
    payload = "|".join(
        [
            str(layer_fingerprint),
            str(arch),
            f"layers={int(layers)}",
            f"steps={int(steps)}",
            f"d={int(model_dim)}",
            f"seq={int(seq_len)}",
            f"batch={int(batch_size)}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{arch}_control_{digest}"


def resolve_reference_parent_fingerprint(
    nb: LabNotebook,
    *,
    arch: str,
    reference_name: str,
    layer_fingerprint: str,
) -> str:
    """Return the canonical parent graph fingerprint for a reference family."""
    row = nb.conn.execute(
        """
        SELECT COALESCE(NULLIF(l.graph_fingerprint, ''), pr.graph_fingerprint) AS fp
        FROM leaderboard l
        LEFT JOIN program_results_compat pr ON pr.result_id = l.result_id
        WHERE COALESCE(l.is_reference, 0) = 1
          AND COALESCE(l.model_source, '') != 'reference_calibration'
          AND (
                LOWER(COALESCE(l.reference_name, '')) = LOWER(?)
             OR COALESCE(l.tags, '') LIKE ?
             OR COALESCE(l.graph_fingerprint, '') = ?
             OR COALESCE(pr.graph_fingerprint, '') = ?
          )
        ORDER BY
          CASE
            WHEN COALESCE(l.tags, '') LIKE ? THEN 0
            WHEN LOWER(COALESCE(l.reference_name, '')) = LOWER(?) THEN 1
            ELSE 2
          END,
          l.timestamp DESC
        LIMIT 1
        """,
        (
            reference_name,
            f"%reference,{arch},%",
            layer_fingerprint,
            layer_fingerprint,
            f"%reference,{arch},%",
            reference_name,
        ),
    ).fetchone()
    fp = str((row["fp"] if row else "") or "").strip()
    return fp or layer_fingerprint


def calibration_milestones(total_steps: int, requested: Iterable[int]) -> list[int]:
    total = max(1, int(total_steps))
    milestones = {int(step) for step in requested if 0 < int(step) <= total}
    milestones.add(total)
    return sorted(milestones)


def calibration_floor_checkpoint_milestones(
    total_steps: int, interval: int
) -> list[int]:
    step_interval = int(interval)
    total = max(1, int(total_steps))
    if step_interval <= 0:
        return []
    return list(range(step_interval, total + 1, step_interval))


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        "|".join(str(p) for p in parts).encode("utf-8")
    ).hexdigest()[:8]
    return f"{prefix}{digest}"[:12]


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event_type": event_type,
            "payload": json_safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _configure_run(
    args: argparse.Namespace,
    *,
    layers: int,
    steps: int,
) -> RunConfig:
    cfg = RunConfig()
    cfg.mode = "confirmation"
    cfg.model_source = "reference_calibration"
    cfg.n_layers = int(layers)
    cfg.model_dim = int(args.model_dim)
    cfg.vocab_size = int(args.vocab_size)
    cfg.max_seq_len = int(args.seq_len)
    cfg.scale_up_seq_len = int(args.seq_len)
    cfg.scale_up_batch_size = int(args.batch_size)
    cfg.scale_up_steps = int(steps)
    cfg.stage1_steps = int(steps)
    cfg.stage1_batch_size = int(args.batch_size)
    cfg.stage1_lr = float(args.lr)
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(args.checkpoint_dir)
    cfg.data_mode = str(args.data_mode)
    cfg.corpus_path = str(args.corpus_path)
    cfg.collect_training_curve = True
    cfg.enable_cuda_graphs = bool(args.enable_cuda_graphs)
    cfg.profile_disable_inflight_checks = bool(args.disable_inflight_checks)
    cfg.early_stop_min_steps = int(steps) + 1
    cfg.early_stop_patience = int(steps) + 1
    cfg.skip_ar_probe = True
    cfg.skip_binding_probes = True
    cfg.skip_induction_probe = True
    cfg.skip_screening_hellaswag = True
    cfg.skip_screening_blimp = True
    return cfg


def _insert_training_curve_unconditionally(
    nb: LabNotebook, result_id: str, curve: list[dict[str, Any]] | None
) -> None:
    if not result_id or not curve:
        return
    nb.conn.executemany(
        """INSERT OR REPLACE INTO training_curves
           (result_id, step, loss, grad_norm, step_time_ms)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (
                result_id,
                int(d.get("step", i) or i),
                d.get("loss"),
                d.get("grad_norm"),
                d.get("step_time_ms"),
            )
            for i, d in enumerate(curve)
        ],
    )
    nb.conn.commit()


def _leaderboard_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "wikitext_perplexity",
        "wikitext_score",
        "wikitext_pre_perplexity",
        "wikitext_ppl_improvement",
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
        "screening_wikitext_variant",
        "screening_wikitext_elapsed_ms",
        "hellaswag_acc",
        "hellaswag_metric_version",
        "hellaswag_tokenizer_mode",
        "hellaswag_tiktoken_encoding",
        "blimp_overall_accuracy",
        "blimp_n_subtasks",
        "blimp_status",
        "ar_legacy_auc",
        "induction_screening_auc",
        "binding_screening_auc",
        "binding_screening_composite",
        "induction_intermediate_auc",
        "induction_intermediate_max_gap_acc",
        "induction_intermediate_protocol_version",
        "binding_intermediate_auc",
        "binding_intermediate_max_distance_acc",
        "binding_intermediate_protocol_version",
        "activation_sparsity_score",
        "dead_neuron_ratio",
        "quant_int8_retention",
        "quant_quality_per_byte",
        "robustness_noise_score",
        "robustness_long_ctx_score",
        "robustness_long_ctx_scaling_score",
        "robustness_long_ctx_assoc_score",
        "robustness_long_ctx_passkey_score",
        "robustness_long_ctx_multi_hop_score",
        "robustness_long_ctx_retrieval_aggregate",
        "robustness_long_ctx_combined_score",
        "max_viable_seq_len",
        "fp_jacobian_erf_density",
        "fp_jacobian_erf_variance",
        "fp_jacobian_erf_decay_slope",
        "fp_icld_velocity",
        "fp_logit_margin_velocity",
        "fp_id_collapse_rate",
        "champion_floor_protocol_version",
        "champion_steps_to_floor",
        "champion_floor_loss",
        "champion_floor_ppl",
        "champion_floor_loss_std",
        "champion_plateau_detected_step",
        "champion_plateau_window",
        "champion_baseline_result_id",
        "champion_baseline_layers",
        "champion_baseline_protocol_version",
        "champion_steps_to_floor_score",
        "champion_floor_quality_score",
        "champion_floor_stability_score",
        "champion_induction_validation_score",
        "champion_binding_long_context_score",
        "champion_ar_validation_score",
        "champion_tiny_model_score",
        "champion_tiny_model_protocol_version",
        "champion_hard_failure_reason",
        "induction_validation_auc",
        "induction_validation_max_gap_acc",
        "induction_validation_gap_accuracy_cv",
        "induction_validation_gap_accuracies_json",
        "induction_validation_steps_trained",
        "induction_validation_status",
        "induction_validation_elapsed_ms",
        "induction_validation_protocol_version",
        "ar_validation_metric_version",
        "ar_validation_final_acc",
        "ar_validation_held_pair_acc",
        "ar_validation_held_class_acc",
        "ar_validation_learning_curve_json",
        "ar_validation_steps_to_floor",
        "ar_validation_rank_score",
        "ar_validation_status",
        "ar_validation_elapsed_ms",
    )
    return {key: metrics[key] for key in keys if metrics.get(key) is not None}


def _run_probe_policy(
    runner: ExperimentRunner,
    *,
    graph,
    checkpoint_paths: dict[int, Path],
    cfg: RunConfig,
    dev: torch.device,
    exp_id: str,
    result_id: str,
    event_writer: EventWriter,
    policy: str,
    required_steps: Iterable[int] = (),
) -> dict[str, Any]:
    if policy == "none":
        return {}
    evaluator = ChampionConfirmationEvaluator(runner)
    steps = sorted(checkpoint_paths)
    if policy == "final":
        steps = sorted({steps[-1], *[int(step) for step in required_steps]})
    snapshots = []
    metrics: dict[str, Any] = {}
    for step in steps:
        path = checkpoint_paths[step]
        if not path.exists():
            event_writer.write(
                "reference_probe_missing_checkpoint",
                {"result_id": result_id, "step": step, "path": str(path)},
            )
            continue
        event_writer.write(
            "reference_probe_start",
            {"result_id": result_id, "step": step, "path": str(path)},
        )
        snapshot = evaluator._scale_up_eval_checkpoint_snapshot(
            graph=graph,
            checkpoint_path=path,
            step=step,
            config=cfg,
            dev=dev,
            dev_str=str(dev),
            exp_id=exp_id,
            source_result_id=result_id,
            prog_idx=0,
        )
        snapshots.append(snapshot)
        event_writer.write(
            "reference_probe_done",
            {
                "result_id": result_id,
                "step": step,
                "status": snapshot.get("status"),
                "induction_intermediate": snapshot.get("induction_intermediate_auc"),
                "binding_intermediate": snapshot.get("binding_intermediate_auc"),
            },
        )
    evaluator._scale_up_apply_champion_id_collapse(snapshots)
    completed = [
        item
        for item in snapshots
        if item.get("status") in (None, "ok", "partial")
        and int(item.get("step") or 0) > 0
    ]
    if completed:
        final_snapshot = max(completed, key=lambda item: int(item["step"]))
        evaluator._scale_up_apply_champion_snapshot(metrics, final_snapshot)
    for item in snapshots:
        item.pop("_id_hidden_snapshot", None)
    metrics["external_benchmarks_json"] = json.dumps(
        json_safe({"reference_calibration_milestones": snapshots}),
        sort_keys=True,
    )
    return metrics


def _checkpoint_paths_for_result(
    checkpoints: CheckpointManager,
    *,
    exp_id: str,
    result_id: str,
    milestone_steps: list[int],
) -> dict[int, Path]:
    return {
        step: checkpoints._artifact_dir(exp_id)
        / f"{result_id}_tp0_reference_calibration_step{step}.pt"
        for step in milestone_steps
    }


def _train_reference_model(
    runner: ExperimentRunner,
    *,
    graph,
    cfg: RunConfig,
    dev: torch.device,
    graph_json: dict[str, Any],
    checkpoints: CheckpointManager,
    exp_id: str,
    result_id: str,
    reference_name: str,
    milestone_steps: list[int],
    args: argparse.Namespace,
    layers: int,
    steps: int,
) -> dict[str, Any]:
    model = compile_model_native_first(
        [graph] * int(layers),
        vocab_size=int(cfg.vocab_size),
        max_seq_len=int(cfg.scale_up_seq_len),
    ).to(dev)
    base_ctx = {"exp_id": exp_id, "phase": "reference_calibration"}
    runner._live_training_context = {
        **base_ctx,
        "source_result_id": result_id,
        "candidate_index": 1,
        "total_candidates": 1,
        "training_program_index": 1,
        "total_training_programs": 1,
        "training_program_label": reference_name,
        "run_kind": "reference_calibration",
        "checkpoint_manager": checkpoints,
        "checkpoint_phase": "validation",
        "checkpoint_candidate_idx": 0,
        "checkpoint_seed_idx": 0,
        "checkpoint_interval_steps": 0,
        "checkpoint_artifact_interval_steps": int(
            getattr(args, "floor_checkpoint_interval", 0) or 0
        ),
        "checkpoint_milestone_steps": milestone_steps,
        "checkpoint_resume_state": None,
    }
    try:
        return runner._micro_train(
            model,
            cfg,
            dev,
            seed=int(args.seed) + layers * 1000 + steps,
            graph_json=graph_json,
        )
    finally:
        runner._live_training_context = base_ctx
        del model
        clear_gpu_memory()


def _reference_extra(
    *,
    probe_metrics: dict[str, Any],
    arch: str,
    layers: int,
    steps: int,
    milestone_steps: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        **probe_metrics,
        "result_cohort": "reference_calibration",
        "trust_label": "reference_control",
        "comparability_label": "reference_control",
        "evaluation_protocol_version": "champion_reference_calibration_v1",
        "data_provenance_json": json.dumps(
            json_safe(
                {
                    "tool": "research.tools.champion_reference_calibration",
                    "arch": arch,
                    "layers": layers,
                    "steps": steps,
                    "probe_policy": args.probe_policy,
                    "milestones": milestone_steps,
                }
            ),
            sort_keys=True,
        ),
    }


def _record_reference_program_result(
    *,
    nb: LabNotebook,
    exp_id: str,
    result_id: str,
    parent_fp: str,
    graph_json: dict[str, Any],
    s1: dict[str, Any],
    extra: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    kwargs = program_result_kwargs_from_s1(
        s1,
        model_source="reference_calibration",
        extra=extra,
    )
    stored_result_id = nb.record_program_result(
        experiment_id=exp_id,
        result_id=result_id,
        graph_fingerprint=parent_fp,
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=bool(s1.get("passed")),
        bypass_quality_gate=True,
        intentional_rerun_reason="reference_calibration",
        **kwargs,
    )
    return stored_result_id, kwargs


def _upsert_reference_leaderboard(
    nb: LabNotebook,
    *,
    result_id: str,
    parent_fp: str,
    arch: str,
    reference_name: str,
    layers: int,
    steps: int,
    s1: dict[str, Any],
    kwargs: dict[str, Any],
    probe_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    loss_ratio = s1.get("loss_ratio")
    nb.upsert_leaderboard(
        result_id=result_id,
        model_source="reference_calibration",
        architecture_desc=(
            f"{reference_name} reference, {layers} layers, {steps} steps"
        ),
        tier="validation" if bool(s1.get("passed")) else "screening",
        tags=f"reference_calibration,{arch},champion_control",
        notes=(
            f"Champion-scale {reference_name} control; probe_policy={args.probe_policy}"
        ),
        is_reference=True,
        reference_name=reference_name,
        screening_loss_ratio=loss_ratio,
        screening_passed=True,
        validation_loss_ratio=loss_ratio,
        validation_passed=bool(s1.get("passed")),
        graph_fingerprint=parent_fp,
        eval_budget_steps=int(steps),
        evaluation_stage="reference_calibration",
        result_cohort="reference_calibration",
        trust_label="reference_control",
        comparability_label="reference_control",
        evaluation_protocol_version="champion_reference_calibration_v1",
        **_leaderboard_metrics({**kwargs, **probe_metrics}),
    )


def _reference_summary(
    *,
    result_id: str,
    arch: str,
    reference_name: str,
    layers: int,
    steps: int,
    s1: dict[str, Any],
    probe_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "result_id": result_id,
        "arch": arch,
        "reference_name": reference_name,
        "layers": int(layers),
        "steps": int(steps),
        "passed": bool(s1.get("passed")),
        "final_loss": s1.get("final_loss"),
        "loss_ratio": s1.get("loss_ratio"),
        "min_loss": s1.get("min_loss"),
        "early_stopped": bool(s1.get("early_stopped")),
        "probe_policy": args.probe_policy,
        "induction_intermediate": probe_metrics.get("induction_intermediate_auc"),
        "binding_intermediate": probe_metrics.get("binding_intermediate_auc"),
        "wikitext_perplexity": probe_metrics.get("wikitext_perplexity"),
    }


def _write_reference_train_start(
    event_writer: EventWriter,
    *,
    exp_id: str,
    result_id: str,
    arch: str,
    reference_name: str,
    layers: int,
    steps: int,
    milestone_steps: list[int],
    dev: torch.device,
) -> None:
    event_writer.write(
        "reference_train_start",
        {
            "experiment_id": exp_id,
            "result_id": result_id,
            "arch": arch,
            "reference_name": reference_name,
            "layers": layers,
            "steps": steps,
            "milestones": milestone_steps,
            "device": str(dev),
        },
    )


def _run_reference_probes(
    runner: ExperimentRunner,
    *,
    graph,
    checkpoint_paths: dict[int, Path],
    cfg: RunConfig,
    dev: torch.device,
    exp_id: str,
    result_id: str,
    event_writer: EventWriter,
    args: argparse.Namespace,
    required_steps: Iterable[int] = (),
) -> dict[str, Any]:
    return _run_probe_policy(
        runner,
        graph=graph,
        checkpoint_paths=checkpoint_paths,
        cfg=cfg,
        dev=dev,
        exp_id=exp_id,
        result_id=result_id,
        event_writer=event_writer,
        policy=str(args.probe_policy),
        required_steps=required_steps,
    )


def _build_reference_run_context(
    nb: LabNotebook,
    args: argparse.Namespace,
    *,
    arch: str,
    layers: int,
    steps: int,
    exp_id: str,
) -> dict[str, Any]:
    cfg = _configure_run(args, layers=layers, steps=steps)
    requested = torch.device(str(cfg.device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        if not bool(getattr(args, "allow_cpu", False)):
            raise RuntimeError("champion reference calibration requires CUDA")
        dev = torch.device("cpu")
    else:
        dev = requested
    if dev.type == "cpu" and not bool(getattr(args, "allow_cpu", False)):
        raise RuntimeError("champion reference calibration requires an accelerator")
    cfg.device = str(dev)
    graph = build_reference(arch, d_model=cfg.model_dim)
    ref_meta = {row["key"]: row for row in list_references()}[arch]
    reference_name = str(ref_meta["name"])
    graph_json = graph_to_json(graph)
    layer_fp = graph.fingerprint()
    parent_fp = resolve_reference_parent_fingerprint(
        nb,
        arch=arch,
        reference_name=reference_name,
        layer_fingerprint=layer_fp,
    )
    run_fp = calibration_fingerprint(
        layer_fp,
        arch=arch,
        layers=layers,
        steps=steps,
        model_dim=cfg.model_dim,
        seq_len=cfg.scale_up_seq_len,
        batch_size=cfg.scale_up_batch_size,
    )
    result_id = _stable_id(f"{arch[:6]}cal", run_fp)
    checkpoints = CheckpointManager(str(cfg.checkpoint_dir))
    milestone_steps = calibration_milestones(
        steps,
        [
            *args.probe_milestones,
            *calibration_floor_checkpoint_milestones(
                steps,
                int(getattr(args, "floor_checkpoint_interval", 0) or 0),
            ),
        ],
    )
    return {
        "cfg": cfg,
        "dev": dev,
        "graph": graph,
        "graph_json": graph_json,
        "parent_fp": parent_fp,
        "run_fp": run_fp,
        "result_id": result_id,
        "arch": arch,
        "reference_name": f"{reference_name} control {layers}L {steps // 1000}K",
        "reference_family_name": reference_name,
        "checkpoints": checkpoints,
        "milestone_steps": milestone_steps,
        "checkpoint_paths": _checkpoint_paths_for_result(
            checkpoints,
            exp_id=exp_id,
            result_id=result_id,
            milestone_steps=milestone_steps,
        ),
    }


def run_one(
    *,
    args: argparse.Namespace,
    nb: LabNotebook,
    runner: ExperimentRunner,
    event_writer: EventWriter,
    exp_id: str,
    arch: str,
    layers: int,
    steps: int,
) -> dict[str, Any]:
    ctx = _build_reference_run_context(
        nb,
        args,
        arch=arch,
        layers=layers,
        steps=steps,
        exp_id=exp_id,
    )
    _write_reference_train_start(
        event_writer,
        exp_id=exp_id,
        result_id=ctx["result_id"],
        arch=ctx["arch"],
        reference_name=ctx["reference_name"],
        layers=layers,
        steps=steps,
        milestone_steps=ctx["milestone_steps"],
        dev=ctx["dev"],
    )
    s1 = _train_reference_model(
        runner,
        graph=ctx["graph"],
        cfg=ctx["cfg"],
        dev=ctx["dev"],
        graph_json=ctx["graph_json"],
        checkpoints=ctx["checkpoints"],
        exp_id=exp_id,
        result_id=ctx["result_id"],
        reference_name=ctx["reference_name"],
        milestone_steps=ctx["milestone_steps"],
        args=args,
        layers=layers,
        steps=steps,
    )
    floor_probe_steps: list[int] = []
    try:
        from research.eval.champion_floor_metrics import extract_champion_floor_metrics

        floor_metrics = extract_champion_floor_metrics(s1.get("training_curve") or [])
        floor_step = floor_metrics.champion_steps_to_floor
        interval = int(getattr(args, "floor_checkpoint_interval", 0) or 0)
        if floor_step is not None and interval > 0:
            rounded = int(round(float(floor_step) / float(interval)) * interval)
            rounded = max(interval, min(int(steps), rounded))
            floor_probe_steps.append(rounded)
    except (ImportError, RuntimeError, ValueError, TypeError):
        floor_probe_steps = []
    probe_metrics = _run_reference_probes(
        runner,
        graph=ctx["graph"],
        checkpoint_paths=ctx["checkpoint_paths"],
        cfg=ctx["cfg"],
        dev=ctx["dev"],
        exp_id=exp_id,
        result_id=ctx["result_id"],
        event_writer=event_writer,
        args=args,
        required_steps=floor_probe_steps,
    )
    extra = _reference_extra(
        probe_metrics=probe_metrics,
        arch=ctx["arch"],
        layers=layers,
        steps=steps,
        milestone_steps=ctx["milestone_steps"],
        args=args,
    )
    result_id, kwargs = _record_reference_program_result(
        nb=nb,
        exp_id=exp_id,
        result_id=ctx["result_id"],
        parent_fp=ctx["parent_fp"],
        graph_json=ctx["graph_json"],
        s1=s1,
        extra=extra,
    )
    _insert_training_curve_unconditionally(
        nb, result_id, s1.get("training_curve") or []
    )
    _upsert_reference_leaderboard(
        nb,
        result_id=result_id,
        parent_fp=ctx["parent_fp"],
        arch=ctx["arch"],
        reference_name=ctx["reference_family_name"],
        layers=layers,
        steps=steps,
        s1=s1,
        kwargs=kwargs,
        probe_metrics=probe_metrics,
        args=args,
    )
    summary = _reference_summary(
        result_id=result_id,
        arch=ctx["arch"],
        reference_name=ctx["reference_family_name"],
        layers=layers,
        steps=steps,
        s1=s1,
        probe_metrics=probe_metrics,
        args=args,
    )
    event_writer.write("reference_train_done", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs",
        nargs="+",
        default=["gpt2"],
        choices=[row["key"] for row in list_references()],
        help="Reference architecture keys to calibrate.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--steps", type=int, nargs="+", default=[40000])
    parser.add_argument(
        "--probe-policy", choices=("none", "final", "milestones"), default="final"
    )
    parser.add_argument(
        "--probe-milestones", type=int, nargs="+", default=[10000, 20000, 40000]
    )
    parser.add_argument(
        "--floor-checkpoint-interval",
        type=int,
        default=1000,
        help=(
            "Save regular artifacts so the nearest floor-entry checkpoint can be "
            "evaluated after floor extraction. Set 0 to disable."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for tiny local debugging only.",
    )
    parser.add_argument("--data-mode", default="corpus")
    parser.add_argument(
        "--corpus-path",
        default="/home/tim/Projects/LLM/research/corpus/wikitext103_train.npy",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--notebook", default="research/runs.db")
    parser.add_argument("--event-log", default="")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--enable-cuda-graphs", action="store_true")
    parser.add_argument("--disable-inflight-checks", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    event_path = (
        Path(args.event_log)
        if args.event_log
        else RUNTIME_DIR / f"run_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    event_writer = EventWriter(event_path)
    runner = ExperimentRunner(args.notebook)
    original_emit = getattr(runner, "_emit_event", None)

    def mirror_emit(event_type: str, payload: dict[str, Any]) -> None:
        event_writer.write(event_type, payload)
        if callable(original_emit):
            original_emit(event_type, payload)

    runner._emit_event = mirror_emit
    summaries = []
    batch_id = f"batch_{time.strftime('%Y%m%dT%H%M%S')}"
    with LabNotebook(args.notebook) as nb:
        for arch in args.archs:
            for layers in args.layers:
                for steps in args.steps:
                    exp_id = nb.start_experiment(
                        "reference_calibration",
                        config={
                            "tool": "champion_reference_calibration",
                            "batch_id": batch_id,
                            "arch": str(arch),
                            "layers": int(layers),
                            "steps": int(steps),
                            "probe_policy": args.probe_policy,
                        },
                        hypothesis=(
                            "Known-good reference controls calibrate "
                            "champion-scale probes."
                        ),
                        research_question=(
                            "Do known reference architectures produce sane "
                            "champion probe and scoring baselines under the "
                            "current runner?"
                        ),
                    )
                    try:
                        summary = run_one(
                            args=args,
                            nb=nb,
                            runner=runner,
                            event_writer=event_writer,
                            exp_id=exp_id,
                            arch=str(arch),
                            layers=int(layers),
                            steps=int(steps),
                        )
                        summaries.append(summary)
                        nb.complete_experiment(
                            exp_id,
                            {
                                "total": 1,
                                "stage0_passed": 1,
                                "stage05_passed": 1,
                                "stage1_passed": int(bool(summary.get("passed"))),
                                "best_loss_ratio": summary.get("loss_ratio"),
                                "best_novelty_score": None,
                                "summary": summary,
                                "batch_id": batch_id,
                            },
                            aria_summary="Reference calibration completed.",
                            aria_mood="focused",
                        )
                    except Exception:
                        nb.fail_experiment(
                            exp_id,
                            error="reference calibration failed",
                            results={
                                "total": len(summaries),
                                "summaries": summaries,
                                "batch_id": batch_id,
                            },
                        )
                        raise
    summary_path = RUNTIME_DIR / f"summary_{batch_id}.json"
    summary_path.write_text(
        json.dumps(
            json_safe({"batch_id": batch_id, "summaries": summaries}),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Reference calibration complete: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
