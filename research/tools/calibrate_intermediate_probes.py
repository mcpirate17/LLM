#!/usr/bin/env python
"""Read-only calibration harness for intermediate binding probes.

Runs ``ar_intermediate_probe`` and/or ``binding_multislot_probe`` over saved model
checkpoints and writes CSV/JSONL artifacts under
``research/runtime/intermediate_probe_calibration``. The notebook DB is opened
read-only and is used only to recover ``graph_json`` for checkpoint
reconstruction when a target spec does not provide it directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from research.eval.ar_intermediate_probe import (
    ARIntermediateConfig,
    ar_intermediate_probe,
)
from research.eval.binding_multislot_probe import (
    BindingMultislotConfig,
    binding_multislot_probe,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools.run_bim_scale_experiment import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/runs.db"
DEFAULT_OUT_DIR = PROJECT_ROOT / "research/runtime/intermediate_probe_calibration"
CALIBRATION_PROTOCOL_VERSION = "intermediate_probe_calibration_v1"

BASE_FIELDS = [
    "run_id",
    "created_unix",
    "target_name",
    "model",
    "branch",
    "checkpoint_path",
    "checkpoint_step",
    "probe",
    "seed",
    "status",
    "elapsed_ms",
    "wall_seconds",
    "error",
]
MEDIUM_FIELDS = [
    "ar_intermediate_metric_version",
    "ar_intermediate_train_pair_acc",
    "ar_intermediate_held_pair_acc",
    "ar_intermediate_held_class_acc",
    "ar_intermediate_pair_chance_acc",
    "ar_intermediate_class_chance_acc",
    "ar_intermediate_held_pair_lift",
    "ar_intermediate_held_class_lift",
    "ar_intermediate_early_held_pair_acc",
    "ar_intermediate_final_held_pair_acc",
    "ar_intermediate_best_held_pair_acc",
    "ar_intermediate_improvement",
    "ar_intermediate_slope_per_100_steps",
    "ar_intermediate_auc",
    "ar_intermediate_auc_lift",
    "ar_intermediate_steps_to_threshold",
    "ar_intermediate_diagnostic_score",
    "ar_intermediate_steps_trained",
    "ar_intermediate_status",
    "ar_intermediate_elapsed_ms",
    "ar_intermediate_error",
]
MULTI_BLANK_FIELDS = [
    "binding_multislot_metric_version",
    "binding_multislot_train_slot_acc",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_entity_class_acc",
    "binding_multislot_two_plus_slots_acc",
    "binding_multislot_all_slots_acc",
    "binding_multislot_mixed_query_acc",
    "binding_multislot_mixed_two_plus_slots_acc",
    "binding_multislot_mixed_all_slots_acc",
    "binding_multislot_slot_chance_acc",
    "binding_multislot_class_chance_acc",
    "binding_multislot_two_plus_slots_chance_acc",
    "binding_multislot_all_slots_chance_acc",
    "binding_multislot_held_slot_lift",
    "binding_multislot_held_class_lift",
    "binding_multislot_two_plus_slots_lift",
    "binding_multislot_all_slots_lift",
    "binding_multislot_mixed_query_lift",
    "binding_multislot_mixed_two_plus_slots_lift",
    "binding_multislot_mixed_all_slots_lift",
    "binding_multislot_early_slot_acc",
    "binding_multislot_final_slot_acc",
    "binding_multislot_best_slot_acc",
    "binding_multislot_improvement",
    "binding_multislot_slope_per_100_steps",
    "binding_multislot_auc",
    "binding_multislot_auc_lift",
    "binding_multislot_steps_to_threshold",
    "binding_multislot_diagnostic_score",
    "binding_multislot_steps_trained",
    "binding_multislot_status",
    "binding_multislot_elapsed_ms",
    "binding_multislot_error",
]
CSV_FIELDS = BASE_FIELDS + MEDIUM_FIELDS + MULTI_BLANK_FIELDS


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    name: str
    checkpoint_path: Path
    fingerprint: str = ""
    branch: str = ""
    model: str = ""
    result_id: str = ""
    graph_json: str | None = None
    d_model: int = 1024
    core_dim: int = 256
    n_layers: int = 12
    vocab_size: int = 100_277
    seq_len: int = 256
    scaled_shell: bool = True


def _bim_known_targets() -> list[CalibrationTarget]:
    root = PROJECT_ROOT / "research/runtime/bim_scale_experiment"
    base10 = root / "bim_scale_20260508T130328Z/checkpoints"
    return [
        CalibrationTarget(
            name="f86a6903_lm_continue_10k",
            model="f86a6903",
            branch="lm_continue_10k",
            fingerprint="f86a6903d32c4ab6",
            checkpoint_path=base10 / "f86a6903d32c4ab6/lm_continue/step_010000.pt",
        ),
        CalibrationTarget(
            name="f86a6903_capability_mix_10k",
            model="f86a6903",
            branch="capability_mix_10k",
            fingerprint="f86a6903d32c4ab6",
            checkpoint_path=base10 / "f86a6903d32c4ab6/capability_mix/step_010000.pt",
        ),
        CalibrationTarget(
            name="ce47c80b_lm_continue_10k",
            model="ce47c80b",
            branch="lm_continue_10k",
            fingerprint="ce47c80b4d581606",
            checkpoint_path=base10 / "ce47c80b4d581606/lm_continue/step_010000.pt",
        ),
        CalibrationTarget(
            name="ce47c80b_capability_mix_10k",
            model="ce47c80b",
            branch="capability_mix_10k",
            fingerprint="ce47c80b4d581606",
            checkpoint_path=base10 / "ce47c80b4d581606/capability_mix/step_010000.pt",
        ),
        CalibrationTarget(
            name="f86a6903_lm_continue_long_120k",
            model="f86a6903",
            branch="lm_continue_long_120k",
            fingerprint="f86a6903d32c4ab6",
            checkpoint_path=root
            / "bim_f86a_long_20260508T151428Z/checkpoints"
            / "f86a6903d32c4ab6/lm_continue_long/step_120000.pt",
        ),
    ]


def _target_from_dict(raw: dict[str, Any]) -> CalibrationTarget:
    path = raw.get("checkpoint_path") or raw.get("path")
    if not path:
        raise ValueError("target requires checkpoint_path")
    name = str(raw.get("name") or Path(path).stem)
    return CalibrationTarget(
        name=name,
        checkpoint_path=Path(path),
        fingerprint=str(raw.get("fingerprint") or raw.get("graph_fingerprint") or ""),
        branch=str(raw.get("branch") or ""),
        model=str(raw.get("model") or ""),
        result_id=str(raw.get("result_id") or ""),
        graph_json=raw.get("graph_json"),
        d_model=int(raw.get("d_model") or 1024),
        core_dim=int(raw.get("core_dim") or 256),
        n_layers=int(raw.get("n_layers") or 12),
        vocab_size=int(raw.get("vocab_size") or 100_277),
        seq_len=int(raw.get("seq_len") or 256),
        scaled_shell=bool(raw.get("scaled_shell", True)),
    )


def load_targets(
    *,
    preset: str,
    targets_json: Path | None,
    include_missing: bool,
) -> list[CalibrationTarget]:
    targets: list[CalibrationTarget] = []
    if preset == "bim-known":
        targets.extend(_bim_known_targets())
    elif preset != "none":
        raise ValueError(f"unknown preset: {preset}")

    if targets_json is not None:
        payload = json.loads(targets_json.read_text(encoding="utf-8"))
        raw_targets = payload.get("targets") if isinstance(payload, dict) else payload
        if not isinstance(raw_targets, list):
            raise ValueError("targets JSON must be a list or {'targets': [...]}")
        targets.extend(_target_from_dict(item) for item in raw_targets)

    if include_missing:
        return targets
    return [target for target in targets if target.checkpoint_path.exists()]


def connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def lookup_graph_json(
    conn: sqlite3.Connection,
    db_path_or_target: Path | CalibrationTarget,
    target: CalibrationTarget | None = None,
) -> str:
    if target is None:
        db_path = DEFAULT_DB
        target = db_path_or_target
    else:
        db_path = Path(db_path_or_target)
    if not isinstance(target, CalibrationTarget):
        raise TypeError("target must be a CalibrationTarget")
    if target.graph_json:
        return str(target.graph_json)
    clauses: list[str] = []
    params: list[str] = []
    if target.result_id:
        clauses.append("result_id = ?")
        params.append(target.result_id)
    if target.fingerprint:
        clauses.append("graph_fingerprint = ?")
        params.append(target.fingerprint)
    if not clauses:
        raise ValueError(
            f"target {target.name} needs graph_json, result_id, or fingerprint"
        )
    row = conn.execute(
        f"""
        SELECT graph_json
        FROM program_results
        WHERE ({" OR ".join(clauses)})
          AND TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{{}}'
        ORDER BY LENGTH(graph_json) DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        raise ValueError(f"no graph_json found for target {target.name}")
    return resolve_graph_json_value(conn, db_path, row["graph_json"])


def _state_dict_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
    raise ValueError(
        "checkpoint does not contain model_state/model_state_dict/state_dict"
    )


def _checkpoint_step(payload: Any) -> int | None:
    if isinstance(payload, dict) and payload.get("step") is not None:
        return int(payload["step"])
    return None


def build_checkpoint_model(
    target: CalibrationTarget,
    *,
    graph_json: str,
    checkpoint_payload: Any,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(
        graph_json,
        int(target.d_model),
        int(target.n_layers),
        int(target.vocab_size),
        int(target.seq_len),
        core_dim=int(target.core_dim),
        scaled_shell=bool(target.scaled_shell),
    ).to(device)
    model.load_state_dict(_state_dict_from_payload(checkpoint_payload), strict=False)
    model.eval()
    return model


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    return {}


def result_row(
    *,
    run_id: str,
    created_unix: float,
    target: CalibrationTarget,
    checkpoint_step: int | None,
    probe: str,
    seed: int,
    wall_seconds: float,
    result: Any | None = None,
    status: str | None = None,
    error: str = "",
) -> dict[str, Any]:
    payload = _result_payload(result) if result is not None else {}
    row = {
        "run_id": run_id,
        "created_unix": created_unix,
        "target_name": target.name,
        "model": target.model,
        "branch": target.branch,
        "checkpoint_path": str(target.checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "probe": probe,
        "seed": int(seed),
        "status": status or payload.get(f"{probe}_status") or "ok",
        "elapsed_ms": payload.get(f"{probe}_elapsed_ms"),
        "wall_seconds": round(float(wall_seconds), 3),
        "error": error or str(payload.get(f"{probe}_error") or ""),
    }
    row.update(payload)
    return row


def append_artifact_rows(
    *,
    csv_path: Path,
    jsonl_path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with (
        csv_path.open("a", newline="", encoding="utf-8") as csv_handle,
        jsonl_path.open(
            "a",
            encoding="utf-8",
        ) as jsonl_handle,
    ):
        writer = csv.DictWriter(
            csv_handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        if not csv_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            jsonl_handle.write(json.dumps(row, sort_keys=True) + "\n")
        csv_handle.flush()
        jsonl_handle.flush()


def _probe_configs(
    *,
    seed: int,
    medium_timeout_s: float | None,
    multi_timeout_s: float | None,
) -> tuple[ARIntermediateConfig, BindingMultislotConfig]:
    medium = ARIntermediateConfig(seed=int(seed), copy_model=True)
    multi = BindingMultislotConfig(seed=int(seed), copy_model=True)
    if medium_timeout_s is not None:
        medium = replace(medium, timeout_s=float(medium_timeout_s))
    if multi_timeout_s is not None:
        multi = replace(multi, timeout_s=float(multi_timeout_s))
    return medium, multi


def run_target(
    target: CalibrationTarget,
    *,
    conn: sqlite3.Connection,
    db_path: Path,
    device: torch.device,
    run_id: str,
    created_unix: float,
    seeds: tuple[int, ...],
    probes: tuple[str, ...],
    medium_timeout_s: float | None,
    multi_timeout_s: float | None,
) -> list[dict[str, Any]]:
    graph_json = lookup_graph_json(conn, db_path, target)
    payload = torch.load(target.checkpoint_path, map_location=device)
    checkpoint_step = _checkpoint_step(payload)
    model = build_checkpoint_model(
        target,
        graph_json=graph_json,
        checkpoint_payload=payload,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            medium_cfg, multi_cfg = _probe_configs(
                seed=seed,
                medium_timeout_s=medium_timeout_s,
                multi_timeout_s=multi_timeout_s,
            )
            if "medium" in probes:
                t0 = time.perf_counter()
                try:
                    result = ar_intermediate_probe(
                        model, cfg=medium_cfg, device=str(device)
                    )
                    rows.append(
                        result_row(
                            run_id=run_id,
                            created_unix=created_unix,
                            target=target,
                            checkpoint_step=checkpoint_step,
                            probe="ar_intermediate",
                            seed=seed,
                            wall_seconds=time.perf_counter() - t0,
                            result=result,
                            status=result.status,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        result_row(
                            run_id=run_id,
                            created_unix=created_unix,
                            target=target,
                            checkpoint_step=checkpoint_step,
                            probe="ar_intermediate",
                            seed=seed,
                            wall_seconds=time.perf_counter() - t0,
                            status="exception",
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}",
                        )
                    )
            if "multi_blank" in probes:
                t0 = time.perf_counter()
                try:
                    result = binding_multislot_probe(
                        model,
                        cfg=multi_cfg,
                        device=str(device),
                    )
                    rows.append(
                        result_row(
                            run_id=run_id,
                            created_unix=created_unix,
                            target=target,
                            checkpoint_step=checkpoint_step,
                            probe="binding_multislot",
                            seed=seed,
                            wall_seconds=time.perf_counter() - t0,
                            result=result,
                            status=result.status,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        result_row(
                            run_id=run_id,
                            created_unix=created_unix,
                            target=target,
                            checkpoint_step=checkpoint_step,
                            probe="binding_multislot",
                            seed=seed,
                            wall_seconds=time.perf_counter() - t0,
                            status="exception",
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}",
                        )
                    )
    finally:
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _parse_probes(raw: str) -> tuple[str, ...]:
    aliases = {
        "both": ("medium", "multi_blank"),
        "medium": ("medium",),
        "ar_intermediate": ("medium",),
        "multi_blank": ("multi_blank",),
        "multi": ("multi_blank",),
        "binding_multislot": ("multi_blank",),
    }
    if str(raw).strip() == "both":
        return aliases["both"]
    out: list[str] = []
    for part in str(raw).split(","):
        key = part.strip()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown probe: {key}")
        out.extend(aliases[key])
    return tuple(dict.fromkeys(out))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--preset", choices=("bim-known", "none"), default="bim-known")
    parser.add_argument("--targets-json", type=Path, default=None)
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--probes", default="both")
    parser.add_argument("--medium-timeout-s", type=float, default=None)
    parser.add_argument("--multi-timeout-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = load_targets(
        preset=args.preset,
        targets_json=args.targets_json,
        include_missing=bool(args.include_missing),
    )
    seeds = _parse_seeds(args.seeds)
    probes = _parse_probes(args.probes)
    run_id = time.strftime("intermediate_probe_calibration_%Y%m%dT%H%M%S")
    csv_path = args.out_csv or (args.out_dir / f"{run_id}.csv")
    jsonl_path = args.out_jsonl or (args.out_dir / f"{run_id}.jsonl")
    created = round(time.time(), 3)
    print(
        json.dumps(
            {
                "event": "selected",
                "run_id": run_id,
                "protocol_version": CALIBRATION_PROTOCOL_VERSION,
                "targets": len(targets),
                "seeds": list(seeds),
                "probes": list(probes),
                "device": args.device,
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for target in targets:
        print(
            json.dumps(
                {
                    "event": "target",
                    "name": target.name,
                    "model": target.model,
                    "branch": target.branch,
                    "fingerprint": target.fingerprint,
                    "checkpoint_path": str(target.checkpoint_path),
                    "exists": target.checkpoint_path.exists(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.dry_run:
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda_unavailable")

    conn = connect_ro(args.db)
    try:
        for target in targets:
            t0 = time.perf_counter()
            print(
                json.dumps(
                    {"event": "target_start", "name": target.name},
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                rows = run_target(
                    target,
                    conn=conn,
                    db_path=args.db,
                    device=device,
                    run_id=run_id,
                    created_unix=created,
                    seeds=seeds,
                    probes=probes,
                    medium_timeout_s=args.medium_timeout_s,
                    multi_timeout_s=args.multi_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                rows = [
                    {
                        "run_id": run_id,
                        "created_unix": created,
                        "target_name": target.name,
                        "model": target.model,
                        "branch": target.branch,
                        "checkpoint_path": str(target.checkpoint_path),
                        "probe": "target_load",
                        "status": "exception",
                        "wall_seconds": round(time.perf_counter() - t0, 3),
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}",
                    }
                ]
            append_artifact_rows(csv_path=csv_path, jsonl_path=jsonl_path, rows=rows)
            print(
                json.dumps(
                    {
                        "event": "target_done",
                        "name": target.name,
                        "rows": len(rows),
                        "elapsed_s": round(time.perf_counter() - t0, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        conn.close()
    print(
        json.dumps(
            {"event": "complete", "csv": str(csv_path), "jsonl": str(jsonl_path)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
