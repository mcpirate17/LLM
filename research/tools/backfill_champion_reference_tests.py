#!/usr/bin/env python
"""Backfill champion test columns and save reference-model test results."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.champion_floor_metrics import lookup_gpt2_champion_baseline
from research.eval.small_ar_champion import SmallARChampionConfig, run_small_ar_champion
from research.scientist.leaderboard_scoring import compute_champion_tiny_model_score_v1
from research.scientist.native_runner import compile_model_native_first
from research.scientist.notebook._shared import _PROGRAM_RESULTS_NEW_COLUMNS
from research.synthesis.serializer import graph_from_json
from research.tools.check_backup_freshness import main as check_backup_freshness_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/lab_notebook.db"
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


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _gap_cv(value: Any) -> float | None:
    vals = [_finite(v) for v in _json_dict(value).values()]
    nums = [v for v in vals if v is not None]
    if len(nums) < 2:
        return None
    mean = statistics.fmean(nums)
    return statistics.pstdev(nums) / mean if mean > 0.0 else None


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


def _select_rows(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    columns = _table_columns(conn, "program_results")
    wanted = [
        "result_id",
        "experiment_id",
        "graph_json",
        "final_loss",
        "wikitext_perplexity",
        "induction_v2_investigation_auc",
        "induction_v2_investigation_max_gap_acc",
        "induction_v2_investigation_gap_accuracies_json",
        "induction_v2_investigation_steps_trained",
        "induction_v2_investigation_status",
        "induction_v2_investigation_elapsed_ms",
        "binding_v2_investigation_auc",
        "robustness_long_ctx_combined_score",
        "small_ar_champion_metric_version",
        "small_ar_champion_final_acc",
        "small_ar_champion_held_pair_match_acc",
        "small_ar_champion_held_class_acc",
        "small_ar_champion_learning_curve_json",
        "small_ar_champion_steps_to_floor",
        "small_ar_champion_score",
        "small_ar_champion_status",
        "small_ar_champion_elapsed_ms",
    ]
    selected = [col for col in wanted if col in columns]
    ids = list(REFERENCE_TARGETS)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM program_results "
        f"WHERE result_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {str(row["result_id"]): dict(row) for row in rows}


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


def _materialize_induction_v3(row: dict[str, Any]) -> dict[str, Any]:
    steps = int(row.get("induction_v2_investigation_steps_trained") or 0)
    if steps != 5_000:
        return {
            "induction_v3_status": "missing_champion_budget",
            "induction_v3_protocol_version": None,
        }
    return {
        "induction_v3_auc": row.get("induction_v2_investigation_auc"),
        "induction_v3_max_gap_acc": row.get("induction_v2_investigation_max_gap_acc"),
        "induction_v3_gap_accuracy_cv": _gap_cv(
            row.get("induction_v2_investigation_gap_accuracies_json")
        ),
        "induction_v3_gap_accuracies_json": row.get(
            "induction_v2_investigation_gap_accuracies_json"
        ),
        "induction_v3_steps_trained": steps,
        "induction_v3_status": row.get("induction_v2_investigation_status") or "ok",
        "induction_v3_elapsed_ms": row.get("induction_v2_investigation_elapsed_ms"),
        "induction_v3_protocol_version": "induction_v3_head_counterfactual_5k",
    }


def _run_small_ar_if_requested(
    row: dict[str, Any],
    *,
    target: dict[str, Any],
    checkpoint_root: Path,
    device: str,
    train_steps: int,
    timeout_s: float,
    force: bool,
    run_probe: bool,
) -> tuple[dict[str, Any], str | None]:
    if row.get("small_ar_champion_status") == "ok" and not force:
        return {k: row.get(k) for k in row if k.startswith("small_ar_champion_")}, None
    if not run_probe:
        return {"small_ar_champion_status": "missing_not_run"}, None
    path = _checkpoint_path(row, checkpoint_root)
    if path is None:
        return {"small_ar_champion_status": "missing_checkpoint"}, None
    model = _load_checkpoint_model(
        row,
        layers=int(target["layers"]),
        checkpoint_path=path,
        device=device,
    )
    try:
        result = run_small_ar_champion(
            model,
            cfg=SmallARChampionConfig(train_steps=train_steps, timeout_s=timeout_s),
            device=device,
        )
        return result.to_dict(), str(path)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


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
        induction_v3_auc=metrics.get("induction_v3_auc"),
        induction_v3_gap_accuracy_cv=metrics.get("induction_v3_gap_accuracy_cv"),
        binding_v2_investigation_auc=row.get("binding_v2_investigation_auc"),
        robustness_long_ctx_combined_score=row.get(
            "robustness_long_ctx_combined_score"
        ),
        champion_baseline_long_ctx_combined_score=metrics.get(
            "champion_baseline_long_ctx_combined_score"
        ),
        small_ar_champion_held_pair_match_acc=metrics.get(
            "small_ar_champion_held_pair_match_acc"
        ),
        small_ar_champion_held_class_acc=metrics.get(
            "small_ar_champion_held_class_acc"
        ),
        small_ar_champion_steps_to_floor=metrics.get(
            "small_ar_champion_steps_to_floor"
        ),
        champion_baseline_small_ar_steps_to_floor=metrics.get(
            "champion_baseline_small_ar_steps_to_floor"
        ),
    )
    return {
        "champion_steps_to_floor_score": score["steps_to_floor"],
        "champion_floor_quality_score": score["floor_quality"],
        "champion_floor_stability_score": score["floor_stability"],
        "champion_induction_v3_score": score["induction_v3"],
        "champion_binding_long_context_score": score["binding_long_context"],
        "champion_small_ar_score": score["small_ar"],
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
        rows = _select_rows(conn)
        long_ctx_vals = [
            _finite(row.get("robustness_long_ctx_combined_score"))
            for rid, row in rows.items()
            if rid.startswith("gpt2cal")
        ]
        long_ctx_base = statistics.fmean(v for v in long_ctx_vals if v is not None)
        small_ar_steps: list[float] = []
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
            metrics.update(_materialize_induction_v3(row))
            small_ar, artifact_path = _run_small_ar_if_requested(
                row,
                target=target,
                checkpoint_root=Path(args.checkpoint_root),
                device=str(args.device),
                train_steps=int(args.small_ar_train_steps),
                timeout_s=float(args.small_ar_timeout_s),
                force=bool(args.force_small_ar),
                run_probe=bool(args.run_small_ar),
            )
            metrics.update(small_ar)
            step = _finite(metrics.get("small_ar_champion_steps_to_floor"))
            if step is not None:
                small_ar_steps.append(step)
            pending.append(
                (result_id, target, row, metrics | {"artifact_path": artifact_path})
            )

        baseline_small_ar_steps = (
            statistics.fmean(small_ar_steps) if small_ar_steps else None
        )
        for result_id, target, row, metrics in pending:
            metrics["champion_baseline_small_ar_steps_to_floor"] = (
                baseline_small_ar_steps
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
    parser.add_argument("--small-ar-train-steps", type=int, default=5_000)
    parser.add_argument("--small-ar-timeout-s", type=float, default=900.0)
    parser.add_argument("--run-small-ar", action="store_true")
    parser.add_argument("--force-small-ar", action="store_true")
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
