#!/usr/bin/env python
"""Backfill champion test columns and save reference-model test results."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.champion_floor_metrics import lookup_gpt2_champion_baseline
from research.eval.ar_validation import ARValidationConfig, run_ar_validation
from research.scientist.leaderboard_scoring import (
    CHAMPION_INDUCTION_V3_PROTOCOLS,
    compute_champion_tiny_model_score_v1,
)
from research.scientist.native_runner import compile_model_native_first
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.notebook._shared import _PROGRAM_RESULTS_NEW_COLUMNS
from research.scientist.runner._helpers import clear_gpu_memory
from research.scientist.shared_utils import coerce_finite_float as _finite
from research.synthesis.serializer import graph_from_json
from research.tools.check_backup_freshness import main as check_backup_freshness_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/runs.db"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/_investigation_artifacts"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "research/runtime/champion_reference_tests"

REFERENCE_TARGETS = {
    "gpt2cal490d5": {"label": "GPT-2 control 4L 40K", "layers": 4, "kind": "gpt2"},
    "gpt2cal87a29": {"label": "GPT-2 control 6L 40K", "layers": 6, "kind": "gpt2"},
    "ref_mamba_76ff10cd": {"label": "Mamba", "layers": 6, "kind": "frontier"},
    "ref_rwkv_61754c8e": {"label": "RWKV", "layers": 6, "kind": "frontier"},
    "ref_retrieval_augmented_ab5cf5ae": {
        "label": "Retrieval-Augmented",
        "layers": 6,
        "kind": "frontier",
    },
}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_program_result_columns(conn: sqlite3.Connection) -> list[str]:
    existing = _table_columns(conn, "program_results")
    added: list[str] = []
    for name, col_type in _PROGRAM_RESULTS_NEW_COLUMNS.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE program_results ADD COLUMN {name} {col_type}")
        added.append(name)
    return added


def _select_rows(
    conn: sqlite3.Connection,
    db_path: Path,
) -> dict[str, dict[str, Any]]:
    columns = _table_columns(conn, "program_results")
    wanted = [
        "result_id",
        "experiment_id",
        "graph_json",
        "final_loss",
        "wikitext_perplexity",
        "induction_intermediate_auc",
        "induction_intermediate_max_gap_acc",
        "induction_intermediate_gap_accuracies_json",
        "induction_intermediate_steps_trained",
        "induction_intermediate_status",
        "induction_intermediate_elapsed_ms",
        "induction_validation_auc",
        "induction_validation_max_gap_acc",
        "induction_validation_gap_accuracy_cv",
        "induction_validation_gap_accuracies_json",
        "induction_validation_steps_trained",
        "induction_validation_status",
        "induction_validation_elapsed_ms",
        "induction_validation_protocol_version",
        "binding_intermediate_auc",
        "binding_intermediate_max_distance_acc",
        "binding_intermediate_distance_accuracies_json",
        "binding_intermediate_train_steps",
        "binding_intermediate_status",
        "binding_intermediate_elapsed_ms",
        "binding_intermediate_protocol_version",
        "robustness_long_ctx_combined_score",
        "ar_validation_metric_version",
        "ar_validation_final_acc",
        "ar_validation_held_pair_acc",
        "ar_validation_held_class_acc",
        "ar_validation_learning_curve_json",
        "ar_validation_steps_to_floor",
        "ar_validation_rank_score",
        "ar_validation_status",
        "ar_validation_elapsed_ms",
    ]
    selected = [col for col in wanted if col in columns]
    ids = list(REFERENCE_TARGETS)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM program_results_compat "
        f"WHERE result_id IN ({placeholders})",
        ids,
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        if "graph_json" in payload:
            payload["graph_json"] = resolve_graph_json_value(
                conn,
                db_path,
                payload["graph_json"],
            )
        out[str(row["result_id"])] = payload
    return out


def _checkpoint_path(row: dict[str, Any], checkpoint_root: Path) -> Path | None:
    result_id = str(row.get("result_id") or "")
    exp_id = str(row.get("experiment_id") or "")
    if not result_id or not exp_id:
        return None
    candidates = sorted((checkpoint_root / exp_id).glob(f"{result_id}_*step*.pt"))
    return candidates[-1] if candidates else None


def _load_checkpoint_model(
    row: dict[str, Any],
    *,
    layers: int,
    checkpoint_path: Path,
    device: str,
):
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_state = state["model_state_dict"]
    embed = model_state.get("embed.weight")
    vocab_size = int(embed.shape[0]) if hasattr(embed, "shape") else 32_000
    graph = graph_from_json(str(row["graph_json"]))
    model = compile_model_native_first(
        [graph] * int(layers),
        vocab_size=vocab_size,
        max_seq_len=128,
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _materialize_gpt2_floor(row: dict[str, Any], layers: int) -> dict[str, Any]:
    baseline = lookup_gpt2_champion_baseline(layers)
    return {
        "champion_floor_protocol_version": baseline.protocol_version,
        "champion_steps_to_floor": baseline.champion_steps_to_floor,
        "champion_floor_loss": baseline.champion_floor_loss,
        "champion_floor_ppl": baseline.champion_floor_ppl,
        "champion_floor_loss_std": baseline.champion_floor_loss_std,
        "champion_plateau_detected_step": baseline.champion_plateau_detected_step,
        "champion_plateau_window": baseline.champion_plateau_window,
        "champion_baseline_result_id": baseline.result_id,
        "champion_baseline_layers": baseline.layers,
        "champion_baseline_protocol_version": baseline.protocol_version,
        "champion_baseline_steps_to_floor": baseline.champion_steps_to_floor,
        "champion_baseline_floor_ppl": baseline.champion_floor_ppl,
        "champion_baseline_floor_loss_std": baseline.champion_floor_loss_std,
    }


def _existing_champion_probe_fields(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "induction_validation_auc",
        "induction_validation_max_gap_acc",
        "induction_validation_gap_accuracy_cv",
        "induction_validation_gap_accuracies_json",
        "induction_validation_steps_trained",
        "induction_validation_status",
        "induction_validation_elapsed_ms",
        "induction_validation_protocol_version",
        "binding_intermediate_auc",
        "binding_intermediate_max_distance_acc",
        "binding_intermediate_distance_accuracies_json",
        "binding_intermediate_train_steps",
        "binding_intermediate_status",
        "binding_intermediate_elapsed_ms",
        "binding_intermediate_protocol_version",
    )
    fields = {key: row.get(key) for key in keys if row.get(key) is not None}
    protocol = str(fields.get("induction_validation_protocol_version") or "").strip()
    if protocol and protocol not in CHAMPION_INDUCTION_V3_PROTOCOLS:
        fields["induction_validation_auc"] = None
        fields["induction_validation_max_gap_acc"] = None
        fields["induction_validation_gap_accuracy_cv"] = None
        fields["induction_validation_status"] = f"invalid_protocol:{protocol}"
    elif not protocol:
        fields.setdefault("induction_validation_status", "missing_not_run")
    return fields


def _champion_probe_fields_from_model(model, *, device: str, induction_steps: int):
    from research.eval.binding_intermediate_probe import (
        run_binding_intermediate,
    )
    from research.eval.induction_validation_probe import (
        run_induction_validation_champion,
    )

    metrics: dict[str, Any] = {}
    induction = run_induction_validation_champion(
        model,
        device=device,
        n_train_steps=int(induction_steps),
    ).to_dict()
    if "induction_validation_gap_accuracies" in induction:
        induction["induction_validation_gap_accuracies_json"] = json.dumps(
            induction.pop("induction_validation_gap_accuracies"),
            sort_keys=True,
        )
    metrics.update(induction)

    binding = run_binding_intermediate(model, device=device).to_dict()
    if "binding_intermediate_distance_accuracies" in binding:
        binding["binding_intermediate_distance_accuracies_json"] = json.dumps(
            binding.pop("binding_intermediate_distance_accuracies"),
            sort_keys=True,
        )
    metrics.update(binding)
    return metrics


def _run_champion_probes_if_requested(
    row: dict[str, Any],
    *,
    target: dict[str, Any],
    checkpoint_root: Path,
    device: str,
    induction_steps: int,
    force: bool,
    run_probe: bool,
    allow_cpu: bool,
) -> tuple[dict[str, Any], str | None]:
    existing = _existing_champion_probe_fields(row)
    protocol = str(existing.get("induction_validation_protocol_version") or "").strip()
    has_valid_induction = (
        protocol in CHAMPION_INDUCTION_V3_PROTOCOLS
        and _finite(existing.get("induction_validation_auc")) is not None
    )
    has_binding = _finite(existing.get("binding_intermediate_auc")) is not None
    if has_valid_induction and has_binding and not force:
        return existing, None
    if not run_probe:
        return existing, None
    if torch.device(device).type == "cpu" and not allow_cpu:
        existing["induction_validation_status"] = "missing_accelerator"
        existing["binding_intermediate_status"] = "missing_accelerator"
        return existing, None
    path = _checkpoint_path(row, checkpoint_root)
    if path is None:
        existing["induction_validation_status"] = "missing_checkpoint"
        existing["binding_intermediate_status"] = "missing_checkpoint"
        return existing, None
    model = _load_checkpoint_model(
        row,
        layers=int(target["layers"]),
        checkpoint_path=path,
        device=device,
    )
    try:
        return (
            _champion_probe_fields_from_model(
                model,
                device=device,
                induction_steps=int(induction_steps),
            ),
            str(path),
        )
    finally:
        del model
        clear_gpu_memory()


def _run_ar_validation_if_requested(
    row: dict[str, Any],
    *,
    target: dict[str, Any],
    checkpoint_root: Path,
    device: str,
    train_steps: int,
    timeout_s: float,
    force: bool,
    run_probe: bool,
    allow_cpu: bool,
) -> tuple[dict[str, Any], str | None]:
    if row.get("ar_validation_status") == "ok" and not force:
        return {k: row.get(k) for k in row if k.startswith("ar_validation_")}, None
    if not run_probe:
        return {"ar_validation_status": "missing_not_run"}, None
    if torch.device(device).type == "cpu" and not allow_cpu:
        return {"ar_validation_status": "missing_accelerator"}, None
    path = _checkpoint_path(row, checkpoint_root)
    if path is None:
        return {"ar_validation_status": "missing_checkpoint"}, None
    model = _load_checkpoint_model(
        row,
        layers=int(target["layers"]),
        checkpoint_path=path,
        device=device,
    )
    try:
        result = run_ar_validation(
            model,
            cfg=ARValidationConfig(train_steps=train_steps, timeout_s=timeout_s),
            device=device,
        )
        return result.to_dict(), str(path)
    finally:
        del model
        clear_gpu_memory()


def _champion_score_fields(
    metrics: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    score = compute_champion_tiny_model_score_v1(
        champion_checkpoint_available=metrics.get("champion_checkpoint_available"),
        champion_steps_to_floor=metrics.get("champion_steps_to_floor"),
        champion_baseline_steps_to_floor=metrics.get(
            "champion_baseline_steps_to_floor"
        ),
        champion_floor_ppl=metrics.get("champion_floor_ppl"),
        champion_baseline_floor_ppl=metrics.get("champion_baseline_floor_ppl"),
        champion_floor_loss_std=metrics.get("champion_floor_loss_std"),
        champion_baseline_floor_loss_std=metrics.get(
            "champion_baseline_floor_loss_std"
        ),
        induction_validation_auc=metrics.get("induction_validation_auc"),
        induction_validation_gap_accuracy_cv=metrics.get(
            "induction_validation_gap_accuracy_cv"
        ),
        binding_intermediate_auc=metrics.get("binding_intermediate_auc")
        if metrics.get("binding_intermediate_auc") is not None
        else row.get("binding_intermediate_auc"),
        robustness_long_ctx_combined_score=row.get(
            "robustness_long_ctx_combined_score"
        ),
        champion_baseline_long_ctx_combined_score=metrics.get(
            "champion_baseline_long_ctx_combined_score"
        ),
        ar_validation_held_pair_acc=metrics.get("ar_validation_held_pair_acc"),
        ar_validation_held_class_acc=metrics.get("ar_validation_held_class_acc"),
        ar_validation_steps_to_floor=metrics.get("ar_validation_steps_to_floor"),
        champion_baseline_ar_validation_steps_to_floor=metrics.get(
            "champion_baseline_ar_validation_steps_to_floor"
        ),
    )
    return {
        "champion_steps_to_floor_score": score["steps_to_floor"],
        "champion_floor_quality_score": score["floor_quality"],
        "champion_floor_stability_score": score["floor_stability"],
        "champion_induction_validation_score": score["induction_validation"],
        "champion_binding_long_context_score": score["binding_long_context"],
        "champion_ar_validation_score": score["ar_validation"],
        "champion_tiny_model_score": score["total"],
        "champion_tiny_model_protocol_version": score["protocol_version"],
        "champion_hard_failure_reason": score["hard_failure_reason"],
    }


def _update_row(
    conn: sqlite3.Connection, result_id: str, values: dict[str, Any]
) -> None:
    columns = _table_columns(conn, "program_results")
    items = [(k, v) for k, v in values.items() if k in columns]
    if not items:
        return
    set_clause = ", ".join(f"{key} = ?" for key, _value in items)
    params = [value for _key, value in items]
    params.append(result_id)
    conn.execute(
        f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
        params,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.write:
        rc = check_backup_freshness_main([])
        if rc != 0:
            raise SystemExit(rc)
    if args.write:
        conn = sqlite3.connect(str(args.db), timeout=30.0)
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    report: dict[str, Any] = {
        "generated_at": time.time(),
        "db": str(args.db),
        "write": bool(args.write),
        "rows": [],
    }
    try:
        added = ensure_program_result_columns(conn) if args.write else []
        rows = _select_rows(conn, Path(args.db))
        long_ctx_vals = [
            _finite(row.get("robustness_long_ctx_combined_score"))
            for rid, row in rows.items()
            if rid.startswith("gpt2cal")
        ]
        long_ctx_base = statistics.fmean(v for v in long_ctx_vals if v is not None)
        ar_validation_steps: list[float] = []
        pending: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

        for result_id, target in REFERENCE_TARGETS.items():
            row = rows.get(result_id)
            if not row:
                report["rows"].append({"result_id": result_id, "status": "missing_row"})
                continue
            metrics: dict[str, Any] = {
                "champion_checkpoint_available": _checkpoint_path(
                    row, Path(args.checkpoint_root)
                )
                is not None,
                "champion_baseline_long_ctx_combined_score": long_ctx_base,
            }
            if target["kind"] == "gpt2":
                metrics.update(_materialize_gpt2_floor(row, int(target["layers"])))
            else:
                metrics.update(
                    {
                        "champion_floor_protocol_version": "missing_training_curve",
                        "champion_hard_failure_reason": "missing_training_curve",
                    }
                )
            champion_probes, champion_artifact_path = _run_champion_probes_if_requested(
                row,
                target=target,
                checkpoint_root=Path(args.checkpoint_root),
                device=str(args.device),
                induction_steps=int(args.champion_induction_steps),
                force=bool(args.force_champion_probes),
                run_probe=bool(args.run_champion_probes),
                allow_cpu=bool(args.allow_cpu),
            )
            metrics.update(champion_probes)
            ar_validation, artifact_path = _run_ar_validation_if_requested(
                row,
                target=target,
                checkpoint_root=Path(args.checkpoint_root),
                device=str(args.device),
                train_steps=int(args.ar_validation_train_steps),
                timeout_s=float(args.ar_validation_timeout_s),
                force=bool(args.force_ar_validation),
                run_probe=bool(args.run_ar_validation),
                allow_cpu=bool(args.allow_cpu),
            )
            metrics.update(ar_validation)
            step = _finite(metrics.get("ar_validation_steps_to_floor"))
            if step is not None:
                ar_validation_steps.append(step)
            pending.append(
                (
                    result_id,
                    target,
                    row,
                    metrics
                    | {"artifact_path": artifact_path or champion_artifact_path},
                )
            )

        baseline_ar_validation_steps = (
            statistics.fmean(ar_validation_steps) if ar_validation_steps else None
        )
        for result_id, target, row, metrics in pending:
            metrics["champion_baseline_ar_validation_steps_to_floor"] = (
                baseline_ar_validation_steps
            )
            metrics.update(_champion_score_fields(metrics, row))
            report["rows"].append(
                {
                    "result_id": result_id,
                    "label": target["label"],
                    "artifact_path": metrics.pop("artifact_path", None),
                    "metrics": metrics,
                }
            )
            if args.write:
                _update_row(conn, result_id, metrics)
        if args.write:
            conn.commit()
        report["columns_added"] = added
    finally:
        conn.close()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--run-champion-probes",
        action="store_true",
        help="Run actual champion induction v3 and binding v2 probes from checkpoints.",
    )
    parser.add_argument("--force-champion-probes", action="store_true")
    parser.add_argument(
        "--champion-induction-steps",
        type=int,
        choices=(2000, 5000, 10000),
        default=2000,
    )
    parser.add_argument("--small-ar-train-steps", type=int, default=5_000)
    parser.add_argument("--small-ar-timeout-s", type=float, default=900.0)
    parser.add_argument("--run-small-ar", action="store_true")
    parser.add_argument("--force-small-ar", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow champion-only probes to run on CPU for tiny local debugging only.",
    )
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"champion_reference_tests_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
