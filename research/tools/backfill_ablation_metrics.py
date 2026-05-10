#!/usr/bin/env python3
"""Backfill missing post-S1 metrics on existing ablation program rows.

The historical ablation runner persisted S0/S0.5/S1 and loss values but threw
away post-S1 screening/probe metrics that `_micro_train` already computed. This
tool replays affected child graphs and merges the missing metrics back into the
same program_results rows. It does not create leaderboard rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from research.scientist.native_runner import compile_model_native_first as compile_model  # noqa: E402
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value  # noqa: E402
from research.scientist.runner import ExperimentRunner  # noqa: E402
from research.scientist.runner._helpers import (  # noqa: E402
    program_result_kwargs_from_s1,
)
from research.scientist.runner._types import RunConfig  # noqa: E402
from research.scientist.runtime_events import publish_runtime_event  # noqa: E402
from research.synthesis.serializer import graph_from_json  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/runs.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
GOOGLE_BACKUP_ROOT = Path("/home/tim/GoogleDrive/Backups/LLM_Research")
LOGGER = logging.getLogger("ablation_metric_backfill")


# Sweep mode (no --result-id): only ablation experiments are safe to backfill
# in bulk. Targeted mode (--result-id supplied): caller has pinned a specific
# row, so we drop the experiment_type filter — the ID itself is the safety.
_BULK_SWEEP_TYPE_FILTER = "e.experiment_type = 'ablation' AND"
_INCOMPLETE_METRICS_PREDICATE = """
    COALESCE(pr.stage1_passed, 0) = 1
    AND pr.graph_json IS NOT NULL
    AND TRIM(CAST(pr.graph_json AS TEXT)) <> ''
    AND (
        pr.hellaswag_acc IS NULL
        OR pr.blimp_overall_accuracy IS NULL
        OR pr.induction_screening_auc IS NULL
        OR pr.binding_screening_auc IS NULL
        OR pr.binding_screening_composite IS NULL
        OR pr.ar_legacy_auc IS NULL
        OR pr.wikitext_perplexity IS NULL
        OR pr.wikitext_score IS NULL
        OR pr.fp_jacobian_erf_density IS NULL
        OR pr.fp_icld_delta_loss IS NULL
        OR pr.fp_logit_margin_delta IS NULL
    )
"""
CORE_MISSING_WHERE = f"{_BULK_SWEEP_TYPE_FILTER} {_INCOMPLETE_METRICS_PREDICATE}"


def configure_logging(log_file: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(log_file) if log_file else RUNTIME_DIR / "ablation_metric_backfill.log"
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


def make_backups(db_path: Path, *, dry_run: bool) -> dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_ablation_metric_backfill_{ts}"
    local_target = PROJECT_ROOT / "research/db_backups" / backup_name / db_path.name
    google_target = GOOGLE_BACKUP_ROOT / backup_name / db_path.name
    if not dry_run:
        local_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, local_target)
        google_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, google_target)
    return {"local": str(local_target), "google_drive": str(google_target)}


def select_rows(
    nb: LabNotebook, *, limit: int, result_id: str = ""
) -> list[dict[str, Any]]:
    params: list[Any] = []
    if result_id:
        # Targeted: trust the explicit ID, skip the ablation-only safety.
        where = _INCOMPLETE_METRICS_PREDICATE + " AND pr.result_id = ?"
        params.append(result_id)
    else:
        where = CORE_MISSING_WHERE
    sql = f"""
        SELECT pr.*, e.config_json AS experiment_config_json
        FROM program_results_compat pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE {where}
        ORDER BY pr.timestamp ASC, pr.result_id ASC
    """
    if limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = [dict(row) for row in nb.conn.execute(sql, tuple(params)).fetchall()]
    for row in rows:
        row["graph_json"] = resolve_graph_json_value(
            nb.conn,
            nb.db_path,
            row.get("graph_json"),
        )
    return rows


def count_missing(nb: LabNotebook) -> dict[str, int]:
    row = nb.conn.execute(
        f"""
        SELECT COUNT(*) AS rows,
               SUM(CASE WHEN COALESCE(pr.stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                   AS s1_rows
        FROM program_results_compat pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE {CORE_MISSING_WHERE}
        """
    ).fetchone()
    return {"rows": int(row["rows"] or 0), "s1_rows": int(row["s1_rows"] or 0)}


def config_for_row(row: Mapping[str, Any], *, device: str | None) -> RunConfig:
    try:
        payload = json.loads(row.get("experiment_config_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        payload = {}
    cfg = RunConfig.from_dict(payload if isinstance(payload, dict) else {})
    cfg.mode = "single"
    cfg.continuous = False
    cfg.enable_causal_ablation = False
    cfg.skip_screening_wikitext = False
    cfg.skip_screening_hellaswag = False
    cfg.skip_screening_blimp = False
    cfg.skip_binding_probes = False
    cfg.skip_induction_probe = False
    cfg.skip_binding_probe = False
    cfg.skip_ar_probe = False
    cfg.skip_post_s1_fingerprint = False
    cfg.skip_post_s1_triage = False
    cfg.profile_disable_post_eval = False
    if device:
        cfg.device = device
    return cfg


def metric_patch_from_s1(s1: Mapping[str, Any]) -> dict[str, Any]:
    """Build a merge_program_result_patch payload from a fresh _micro_train s1.

    Delegates the entire metric-bundle reconstruction to
    program_result_kwargs_from_s1 so this tool can never drift from the
    runner's persistence path. Adds backfill-only provenance labels on top.
    """
    patch = program_result_kwargs_from_s1(
        dict(s1),
        model_source="ablation",
        extra={
            "trust_label": "ablation_metric_backfill_replay",
            "comparability_label": "reconstructed_init_variant",
            "evaluation_protocol_version": "ablation_metric_backfill_v1",
        },
    )
    return {k: v for k, v in patch.items() if v is not None}


def publish_progress(
    db_path: Path,
    run_id: str,
    *,
    index: int,
    total: int,
    result_id: str,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    data = {
        "index": index,
        "total": total,
        "result_id": result_id,
        "status": status,
        **dict(payload),
    }
    publish_runtime_event(
        notebook_path=db_path,
        event_type="ablation_metric_backfill_progress",
        producer="tools.backfill_ablation_metrics",
        run_id=run_id,
        payload=data,
    )


def backfill_row(
    *,
    runner: ExperimentRunner,
    nb: LabNotebook,
    row: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    result_id = str(row["result_id"])
    graph_json = str(row["graph_json"])
    graph = graph_from_json(graph_json)
    cfg = config_for_row(row, device=str(device))
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len
    ).to(device)
    s1 = runner._micro_train(
        model,
        cfg,
        device,
        seed=runner._stable_seed(result_id, "ablation_metric_backfill"),
        graph_json=graph_json,
    )
    if not bool(s1.get("passed")):
        return {
            "patched": False,
            "passed": False,
            "error_type": s1.get("error_type"),
            "error": s1.get("error"),
        }
    patch = metric_patch_from_s1(s1)
    changed = nb.merge_program_result_patch(
        result_id=result_id,
        graph_fingerprint=str(row["graph_fingerprint"] or graph.fingerprint()),
        graph_json=graph_json,
        clear_failure_if_stage1=True,
        relabel_backfill_if_orphan=False,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        **patch,
    )
    nb.flush_writes()
    return {
        "patched": bool(changed),
        "passed": True,
        "loss_ratio_replay": s1.get("loss_ratio"),
        "hellaswag_acc": patch.get("hellaswag_acc"),
        "blimp_overall_accuracy": patch.get("blimp_overall_accuracy"),
        "induction_screening_auc": patch.get("induction_screening_auc"),
        "binding_screening_auc": patch.get("binding_screening_auc"),
        "binding_screening_composite": patch.get("binding_screening_composite"),
        "ar_legacy_auc": patch.get("ar_legacy_auc"),
        "wikitext_perplexity": patch.get("wikitext_perplexity"),
        "field_count": len(patch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 means all missing rows")
    parser.add_argument("--result-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    log_path = configure_logging(args.log_file)
    nb = LabNotebook(str(db_path), use_native=False)
    run_id = ""
    try:
        before = count_missing(nb)
        rows = select_rows(nb, limit=max(0, int(args.limit)), result_id=args.result_id)
        LOGGER.info(
            "ablation metric backfill planned rows=%d missing_before=%s",
            len(rows),
            before,
        )
        if args.dry_run:
            return 0
        if not rows:
            LOGGER.info("no rows require backfill")
            return 0
        backup_paths = {} if args.no_backup else make_backups(db_path, dry_run=False)
        LOGGER.info("database backups created: %s", backup_paths)
        run_id = nb.start_experiment(
            "ablation_metric_backfill",
            {
                "limit": int(args.limit),
                "result_id": args.result_id,
                "planned_rows": len(rows),
                "log_path": str(log_path),
                "backup_paths": backup_paths,
            },
            hypothesis="Backfill missing post-S1 screening/probe metrics for ablation child rows.",
            hypothesis_metadata={"source": "tools.backfill_ablation_metrics"},
        )
        publish_runtime_event(
            notebook_path=db_path,
            event_type="ablation_metric_backfill_started",
            producer="tools.backfill_ablation_metrics",
            run_id=run_id,
            payload={"planned_rows": len(rows), "missing_before": before},
        )
        runner = ExperimentRunner(str(db_path))
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        patched = 0
        failed = 0
        started = time.time()
        for index, row in enumerate(rows, start=1):
            rid = str(row["result_id"])
            LOGGER.info("backfill %d/%d result=%s", index, len(rows), rid)
            try:
                result = backfill_row(runner=runner, nb=nb, row=row, device=device)
            except Exception as exc:  # noqa: BLE001 - operational backfill must continue
                failed += 1
                result = {"patched": False, "passed": False, "error": repr(exc)}
                LOGGER.exception("backfill failed result=%s", rid)
            else:
                if result.get("patched"):
                    patched += 1
                if not result.get("passed"):
                    failed += 1
            LOGGER.info("backfill result=%s summary=%s", rid, result)
            publish_progress(
                db_path,
                run_id,
                index=index,
                total=len(rows),
                result_id=rid,
                status="patched" if result.get("patched") else "no_change_or_failed",
                payload=result,
            )
        after = count_missing(nb)
        results = {
            "total": len(rows),
            "stage0_passed": patched,
            "stage05_passed": patched,
            "stage1_passed": patched,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "planned": len(rows),
            "patched": patched,
            "failed": failed,
            "missing_before": before,
            "missing_after": after,
            "elapsed_seconds": time.time() - started,
        }
        nb.complete_experiment(
            run_id,
            results=results,
            aria_summary=(
                f"Ablation metric backfill patched {patched}/{len(rows)} rows; "
                f"{failed} failed."
            ),
        )
        LOGGER.info("ablation metric backfill complete: %s", results)
        return 0 if failed == 0 else 2
    except Exception as exc:
        if run_id:
            try:
                nb.fail_experiment(run_id, repr(exc))
            except Exception:
                pass
        raise
    finally:
        try:
            nb.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
