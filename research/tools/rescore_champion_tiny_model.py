#!/usr/bin/env python
"""Dry-run-first champion tiny-model backfill and rescore.

Reads saved training curves and persisted probe metrics for the GPT-2
calibration controls plus the current Mamba/champion artifact target. By
default this is read-only and prints a compact comparison table. Passing
``--write`` persists derived champion fields after a fresh-backup check.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from research.tools.check_backup_freshness import main as check_backup_freshness_main
from research.tools.db_health import backup_sqlite_db


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/lab_notebook.db"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/_investigation_artifacts"
GPT2_EXPERIMENT_ID = "ba70eb86-3f8"
GPT2_TARGETS = {
    "gpt2cal490d5": ("GPT-2 4L", 4),
    "gpt2cal87a29": ("GPT-2 6L", 6),
}
DEFAULT_MAMBA_RESULT_ID = "574271ca-f37"
FLOOR_PROTOCOL_VERSION = "champion_floor_v1_window500"
SCORE_PROTOCOL_VERSION = "champion_tiny_model_score_v1"


@dataclass(frozen=True)
class CurveMetrics:
    steps_to_floor: int | None
    floor_loss: float | None
    floor_ppl: float | None
    floor_loss_std: float | None
    plateau_detected_step: int | None
    plateau_window: int
    plateau_found: bool


@dataclass(frozen=True)
class TargetRow:
    result_id: str
    label: str
    layers: int | None
    experiment_id: str | None
    metrics: dict[str, Any]
    artifact_paths: tuple[str, ...]


@dataclass(frozen=True)
class ScoreRow:
    target: TargetRow
    curve: CurveMetrics
    scores: dict[str, float]
    total_score: float
    metric_sources: dict[str, str]
    hard_failure_reason: str | None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _finite_float(value: Any) -> float | None:
    try:
        out = None if value is None else float(value)
    except (TypeError, ValueError):
        return None
    if out is None or not math.isfinite(out):
        return None
    return out


def _safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        return math.exp(min(float(value), 700.0))
    except (OverflowError, ValueError):
        return None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def load_training_curve(
    conn: sqlite3.Connection, result_id: str
) -> list[tuple[int, float]]:
    rows = conn.execute(
        """
        SELECT step, loss
        FROM training_curves
        WHERE result_id = ? AND loss IS NOT NULL
        ORDER BY step
        """,
        (result_id,),
    ).fetchall()
    curve: list[tuple[int, float]] = []
    for step, loss in rows:
        loss_f = _finite_float(loss)
        if loss_f is not None:
            curve.append((int(step), loss_f))
    return curve


def extract_floor_metrics(
    curve: Iterable[tuple[int, float]],
    *,
    plateau_window: int = 500,
) -> CurveMetrics:
    points = sorted((int(step), float(loss)) for step, loss in curve)
    if not points:
        return CurveMetrics(None, None, None, None, None, plateau_window, False)

    losses = [loss for _, loss in points]
    best_idx = min(range(len(points)), key=lambda i: losses[i])
    plateau_start_idx: int | None = None
    plateau_detected_step: int | None = None

    for start_idx, (start_step, start_loss) in enumerate(points):
        end_idx = None
        min_end_step = start_step + plateau_window
        for idx in range(start_idx + 1, len(points)):
            if points[idx][0] >= min_end_step:
                end_idx = idx
                break
        if end_idx is None:
            break
        end_loss = points[end_idx][1]
        improvement = start_loss - end_loss
        threshold = max(0.02, abs(start_loss) * 0.005)
        if improvement <= threshold:
            plateau_start_idx = start_idx
            plateau_detected_step = points[end_idx][0]
            break

    if plateau_start_idx is None:
        floor_loss = losses[best_idx]
        return CurveMetrics(
            steps_to_floor=points[best_idx][0],
            floor_loss=floor_loss,
            floor_ppl=_safe_exp(floor_loss),
            floor_loss_std=None,
            plateau_detected_step=None,
            plateau_window=plateau_window,
            plateau_found=False,
        )

    post_losses = losses[plateau_start_idx:]
    floor_loss = min(post_losses)
    floor_std = statistics.pstdev(post_losses) if len(post_losses) > 1 else 0.0
    floor_band = floor_loss + max(0.03, floor_std)
    floor_step = points[plateau_start_idx][0]
    for step, loss in points[plateau_start_idx:]:
        if loss <= floor_band:
            floor_step = step
            break
    return CurveMetrics(
        steps_to_floor=floor_step,
        floor_loss=floor_loss,
        floor_ppl=_safe_exp(floor_loss),
        floor_loss_std=floor_std,
        plateau_detected_step=plateau_detected_step,
        plateau_window=plateau_window,
        plateau_found=True,
    )


def _json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _cv_from_json(value: Any) -> float | None:
    vals = [_finite_float(v) for v in _json_dict(value).values()]
    nums = [v for v in vals if v is not None]
    if len(nums) < 2:
        return None
    mean = statistics.fmean(nums)
    if mean <= 0:
        return None
    return statistics.pstdev(nums) / mean


def _select_existing_columns(conn: sqlite3.Connection) -> list[str]:
    available = _table_columns(conn, "program_results")
    wanted = [
        "result_id",
        "experiment_id",
        "final_loss",
        "min_loss",
        "wikitext_perplexity",
        "avg_step_time_ms",
        "total_train_time_ms",
        "n_train_steps",
        "induction_v3_auc",
        "induction_v3_gap_accuracy_cv",
        "induction_v3_protocol_version",
        "induction_v2_investigation_auc",
        "induction_v2_investigation_gap_accuracies_json",
        "induction_v2_investigation_protocol_version",
        "binding_v2_investigation_auc",
        "robustness_long_ctx_combined_score",
        "small_ar_champion_held_pair_match_acc",
        "small_ar_champion_held_class_acc",
        "small_ar_champion_steps_to_floor",
        "small_ar_champion_status",
        "small_ar_champion_metric_version",
        "nano_ar_inv_held_pair_match_acc",
        "nano_ar_inv_held_class_acc",
        "nano_ar_inv_score",
        "nano_ar_inv_status",
    ]
    return [col for col in wanted if col in available]


def _fetch_metric_rows(
    conn: sqlite3.Connection, result_ids: list[str]
) -> dict[str, dict[str, Any]]:
    if not result_ids:
        return {}
    columns = _select_existing_columns(conn)
    placeholders = ",".join("?" for _ in result_ids)
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM program_results WHERE result_id IN ({placeholders})",
        result_ids,
    ).fetchall()
    return {str(row["result_id"]): dict(row) for row in rows}


def _artifact_paths(
    result_id: str,
    experiment_id: str | None,
    *,
    checkpoint_root: Path,
) -> tuple[str, ...]:
    paths: list[Path] = []
    if checkpoint_root.exists():
        search_roots = [checkpoint_root / str(experiment_id)] if experiment_id else []
        search_roots.append(checkpoint_root)
        seen: set[Path] = set()
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.glob(f"**/{result_id}*"):
                if path.is_file() and path not in seen:
                    paths.append(path)
                    seen.add(path)
    return tuple(
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
        for path in sorted(paths)
    )


def load_targets(
    conn: sqlite3.Connection,
    *,
    mamba_result_id: str = DEFAULT_MAMBA_RESULT_ID,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> list[TargetRow]:
    result_ids = [*GPT2_TARGETS.keys(), mamba_result_id]
    rows = _fetch_metric_rows(conn, result_ids)
    targets: list[TargetRow] = []
    for result_id, (label, layers) in GPT2_TARGETS.items():
        row = rows.get(result_id)
        if row:
            artifacts = _artifact_paths(
                result_id, row.get("experiment_id"), checkpoint_root=checkpoint_root
            )
            targets.append(
                TargetRow(
                    result_id, label, layers, row.get("experiment_id"), row, artifacts
                )
            )

    row = rows.get(mamba_result_id)
    if row:
        artifacts = _artifact_paths(
            mamba_result_id, row.get("experiment_id"), checkpoint_root=checkpoint_root
        )
        if artifacts or load_training_curve(conn, mamba_result_id):
            targets.append(
                TargetRow(
                    mamba_result_id,
                    "Mamba champion",
                    None,
                    row.get("experiment_id"),
                    row,
                    artifacts,
                )
            )
    return targets


def _mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(nums) if nums else None


def _baseline(
    curves: dict[str, CurveMetrics], rows: list[TargetRow]
) -> dict[str, float | None]:
    gpt_ids = [row.result_id for row in rows if row.result_id in GPT2_TARGETS]
    return {
        "steps_to_floor": _mean(
            curves[rid].steps_to_floor for rid in gpt_ids if rid in curves
        ),
        "floor_loss": _mean(curves[rid].floor_loss for rid in gpt_ids if rid in curves),
        "floor_ppl": _mean(curves[rid].floor_ppl for rid in gpt_ids if rid in curves),
        "floor_loss_std": _mean(
            curves[rid].floor_loss_std for rid in gpt_ids if rid in curves
        ),
        "long_ctx": _mean(
            _finite_float(row.metrics.get("robustness_long_ctx_combined_score"))
            for row in rows
            if row.result_id in GPT2_TARGETS
        ),
    }


def _induction_score(metrics: dict[str, Any]) -> tuple[float, str]:
    auc = _finite_float(metrics.get("induction_v3_auc"))
    cv = _finite_float(metrics.get("induction_v3_gap_accuracy_cv"))
    source = str(metrics.get("induction_v3_protocol_version") or "induction_v3")
    if auc is None:
        auc = _finite_float(metrics.get("induction_v2_investigation_auc"))
        cv = _cv_from_json(
            metrics.get("induction_v2_investigation_gap_accuracies_json")
        )
        source = str(
            metrics.get("induction_v2_investigation_protocol_version")
            or "fallback_induction_v2"
        )
    if auc is None:
        return 0.0, "missing"
    cv_term = 1.0 if cv is None else _clamp(1.0 - cv)
    return _clamp((auc - 0.20) / 0.75) * 8.0 + cv_term * 2.0, source


def _small_ar_score(
    metrics: dict[str, Any], baseline_steps: float | None
) -> tuple[float, str]:
    pair = _finite_float(metrics.get("small_ar_champion_held_pair_match_acc"))
    held_class = _finite_float(metrics.get("small_ar_champion_held_class_acc"))
    steps = _finite_float(metrics.get("small_ar_champion_steps_to_floor"))
    source = str(metrics.get("small_ar_champion_metric_version") or "small_ar_champion")
    if pair is None and held_class is None:
        pair = _finite_float(metrics.get("nano_ar_inv_held_pair_match_acc"))
        held_class = _finite_float(metrics.get("nano_ar_inv_held_class_acc"))
        source = "fallback_nano_ar_inv"
    if pair is None and held_class is None:
        return 0.0, "missing"
    speed = 0.0
    if baseline_steps and steps is not None:
        speed = _clamp((baseline_steps - steps) / baseline_steps)
    return 6.0 * _clamp(pair or 0.0) + 2.0 * _clamp(
        held_class or 0.0
    ) + 2.0 * speed, source


def compute_score_rows(
    conn: sqlite3.Connection, targets: list[TargetRow]
) -> list[ScoreRow]:
    curves = {
        row.result_id: extract_floor_metrics(load_training_curve(conn, row.result_id))
        for row in targets
    }
    base = _baseline(curves, targets)
    score_rows: list[ScoreRow] = []
    for target in targets:
        curve = curves[target.result_id]
        hard_failure_reason = None
        if curve.floor_loss is None:
            hard_failure_reason = "missing_training_curve"
        elif curve.floor_loss > 100 or not math.isfinite(curve.floor_loss):
            hard_failure_reason = "divergent_or_corrupt_floor_loss"

        base_steps = base.get("steps_to_floor")
        base_ppl = base.get("floor_ppl")
        base_loss = base.get("floor_loss")
        base_std = base.get("floor_loss_std")
        speed_score = (
            _clamp((base_steps - curve.steps_to_floor) / base_steps) * 10.0
            if base_steps and curve.steps_to_floor is not None
            else 0.0
        )
        if (
            base_ppl
            and curve.floor_ppl
            and math.isfinite(base_ppl)
            and math.isfinite(curve.floor_ppl)
        ):
            floor_quality_score = _clamp((base_ppl - curve.floor_ppl) / base_ppl) * 10.0
        elif base_loss and curve.floor_loss is not None:
            floor_quality_score = (
                _clamp((base_loss - curve.floor_loss) / base_loss) * 10.0
            )
        else:
            floor_quality_score = 0.0
        stability_score = (
            _clamp(1.0 - (curve.floor_loss_std or 0.0) / base_std) * 5.0
            if base_std and curve.floor_loss_std is not None
            else 0.0
        )
        induction_score, induction_source = _induction_score(target.metrics)
        small_ar_score, small_ar_source = _small_ar_score(target.metrics, base_steps)
        binding_auc = (
            _finite_float(target.metrics.get("binding_v2_investigation_auc")) or 0.0
        )
        long_ctx = (
            _finite_float(target.metrics.get("robustness_long_ctx_combined_score"))
            or 0.0
        )
        base_long_ctx = base.get("long_ctx") or 0.0
        binding_long_score = 3.0 * _clamp(binding_auc)
        if base_long_ctx > 0:
            binding_long_score += 2.0 * _clamp(long_ctx / base_long_ctx)

        scores = {
            "speed": speed_score,
            "floor_quality": floor_quality_score,
            "stability": stability_score,
            "induction_v3": induction_score,
            "binding_long_context": binding_long_score,
            "small_ar": small_ar_score,
        }
        score_rows.append(
            ScoreRow(
                target=target,
                curve=curve,
                scores=scores,
                total_score=sum(scores.values()),
                metric_sources={
                    "induction_v3": induction_source,
                    "small_ar": small_ar_source,
                },
                hard_failure_reason=hard_failure_reason,
            )
        )
    return score_rows


CHAMPION_COLUMNS = {
    "champion_floor_protocol_version": "TEXT",
    "champion_steps_to_floor": "REAL",
    "champion_floor_loss": "REAL",
    "champion_floor_ppl": "REAL",
    "champion_floor_loss_std": "REAL",
    "champion_plateau_detected_step": "REAL",
    "champion_plateau_window": "REAL",
    "champion_baseline_result_id": "TEXT",
    "champion_baseline_layers": "TEXT",
    "champion_baseline_protocol_version": "TEXT",
    "champion_steps_to_floor_score": "REAL",
    "champion_floor_quality_score": "REAL",
    "champion_floor_stability_score": "REAL",
    "champion_induction_v3_score": "REAL",
    "champion_binding_long_context_score": "REAL",
    "champion_small_ar_score": "REAL",
    "champion_tiny_model_score": "REAL",
    "champion_tiny_model_protocol_version": "TEXT",
    "champion_hard_failure_reason": "TEXT",
}


def _ensure_champion_columns(conn: sqlite3.Connection) -> None:
    existing = _table_columns(conn, "program_results")
    for column, col_type in CHAMPION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE program_results ADD COLUMN {column} {col_type}")


def write_score_rows(conn: sqlite3.Connection, rows: list[ScoreRow]) -> None:
    _ensure_champion_columns(conn)
    for row in rows:
        conn.execute(
            """
            UPDATE program_results
            SET champion_floor_protocol_version = ?,
                champion_steps_to_floor = ?,
                champion_floor_loss = ?,
                champion_floor_ppl = ?,
                champion_floor_loss_std = ?,
                champion_plateau_detected_step = ?,
                champion_plateau_window = ?,
                champion_baseline_result_id = ?,
                champion_baseline_layers = ?,
                champion_baseline_protocol_version = ?,
                champion_steps_to_floor_score = ?,
                champion_floor_quality_score = ?,
                champion_floor_stability_score = ?,
                champion_induction_v3_score = ?,
                champion_binding_long_context_score = ?,
                champion_small_ar_score = ?,
                champion_tiny_model_score = ?,
                champion_tiny_model_protocol_version = ?,
                champion_hard_failure_reason = ?
            WHERE result_id = ?
            """,
            (
                FLOOR_PROTOCOL_VERSION,
                row.curve.steps_to_floor,
                row.curve.floor_loss,
                row.curve.floor_ppl,
                row.curve.floor_loss_std,
                row.curve.plateau_detected_step,
                row.curve.plateau_window,
                ",".join(GPT2_TARGETS),
                ",".join(str(layers) for _, layers in GPT2_TARGETS.values()),
                FLOOR_PROTOCOL_VERSION,
                row.scores["speed"],
                row.scores["floor_quality"],
                row.scores["stability"],
                row.scores["induction_v3"],
                row.scores["binding_long_context"],
                row.scores["small_ar"],
                row.total_score,
                SCORE_PROTOCOL_VERSION,
                row.hard_failure_reason,
                row.target.result_id,
            ),
        )
    conn.commit()


def _fmt(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def print_comparison(rows: list[ScoreRow], out: TextIO) -> None:
    print("Champion tiny-model dry-run comparison", file=out)
    print(
        "target              result_id     steps floor_loss speed floor_q stable induction_v3 small_ar total artifacts",
        file=out,
    )
    for row in rows:
        print(
            f"{row.target.label[:18]:18} "
            f"{row.target.result_id:12} "
            f"{_fmt(row.curve.steps_to_floor, 0):>5} "
            f"{_fmt(row.curve.floor_loss, 3):>10} "
            f"{_fmt(row.scores['speed']):>5} "
            f"{_fmt(row.scores['floor_quality']):>7} "
            f"{_fmt(row.scores['stability']):>6} "
            f"{_fmt(row.scores['induction_v3']):>12} "
            f"{_fmt(row.scores['small_ar']):>8} "
            f"{_fmt(row.total_score):>5} "
            f"{len(row.target.artifact_paths):>9}",
            file=out,
        )
    fallback_notes = sorted(
        {
            f"{row.target.result_id}: induction={row.metric_sources['induction_v3']}, small_ar={row.metric_sources['small_ar']}"
            for row in rows
            if "fallback" in row.metric_sources["induction_v3"]
            or "fallback" in row.metric_sources["small_ar"]
            or row.metric_sources["induction_v3"] == "missing"
            or row.metric_sources["small_ar"] == "missing"
        }
    )
    if fallback_notes:
        print("metric_sources: " + "; ".join(fallback_notes), file=out)


def _json_payload(rows: list[ScoreRow]) -> dict[str, Any]:
    return {
        "protocol_version": SCORE_PROTOCOL_VERSION,
        "floor_protocol_version": FLOOR_PROTOCOL_VERSION,
        "rows": [
            {
                "result_id": row.target.result_id,
                "label": row.target.label,
                "experiment_id": row.target.experiment_id,
                "layers": row.target.layers,
                "curve": row.curve.__dict__,
                "scores": row.scores,
                "total_score": row.total_score,
                "metric_sources": row.metric_sources,
                "hard_failure_reason": row.hard_failure_reason,
                "artifact_paths": row.target.artifact_paths,
            }
            for row in rows
        ],
    }


def run(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    db_path = Path(args.db)
    checkpoint_root = Path(args.checkpoint_root)
    if not args.write:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        if args.make_backup:
            backup_path = backup_sqlite_db(db_path, suffix="pre_champion_tiny_rescore")
            print(f"backup={backup_path}", file=out)
        else:
            rc = check_backup_freshness_main([])
            if rc != 0:
                return rc
        conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        targets = load_targets(
            conn,
            mamba_result_id=str(args.mamba_result_id),
            checkpoint_root=checkpoint_root,
        )
        rows = compute_score_rows(conn, targets)
        if args.write:
            write_score_rows(conn, rows)
        if args.json:
            print(json.dumps(_json_payload(rows), indent=2, sort_keys=True), file=out)
        else:
            print_comparison(rows, out)
            print("mode=" + ("WRITE" if args.write else "DRY-RUN"), file=out)
        return 0
    finally:
        conn.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--mamba-result-id", default=DEFAULT_MAMBA_RESULT_ID)
    parser.add_argument("--write", action="store_true", help="Persist champion fields.")
    parser.add_argument(
        "--make-backup",
        action="store_true",
        help="With --write, create a fresh SQLite backup instead of requiring an existing fresh backup.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of the compact table."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
