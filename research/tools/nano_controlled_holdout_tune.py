"""Tune the held-out nano-HellaSwag probe across vocab/step grids.

Read-only calibration harness: loads saved graph JSON from the lab notebook,
does the same quick base-train used by the controlled-language tuners, then
sweeps ``nano_controlled_holdout_probe`` over a (active_vocab × train_steps) grid.

Does NOT write DB rows.  Output is a single JSON report under
``research/reports/`` per run.

Usage:
    python -m research.tools.nano_controlled_holdout_tune \
        --targets ec7025d7-338 8d087a16-692 903157e5-219
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.nano_controlled_holdout_probe import nano_controlled_holdout_probe
from research.eval.utils import micro_train_loop
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

VOCAB_SIZE = 50257
DEFAULT_TARGETS: tuple[str, ...] = (
    "ec7025d7-338",
    "8d087a16-692",
    "903157e5-219",
)
# (active_vocab_size, n_train_steps) — chosen to span the same calibration
# window codex calibrated for ``controlled_sentence_probe`` plus a tougher
# upper end to stress generalisation, not memorisation, of the held-out
# buckets.
DEFAULT_CONFIGS: tuple[tuple[int, int], ...] = (
    (256, 40),
    (512, 100),
    (1000, 100),
    (1000, 300),
)
DEFAULT_HOLD_OUT_FRAC = 0.25
DEFAULT_N_CLASSES = 4
DEFAULT_N_EVAL_PER_BUCKET = 24


def _parse_config(value: str) -> tuple[int, int]:
    try:
        vocab, steps = value.split(":", 1)
        return int(vocab), int(steps)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("configs must look like VOCAB:STEPS") from exc


def _load_target(db: Path, target: str) -> dict[str, Any]:
    conn = connect_readonly(db)
    try:
        row = conn.execute(
            """
            SELECT l.entry_id, l.result_id, l.tier, l.composite_score,
                   l.induction_auc, l.binding_composite,
                   l.induction_v2_investigation_auc,
                   l.binding_v2_investigation_auc,
                   pr.graph_json, pr.graph_fingerprint,
                   pr.controlled_lang_s05_sa_score,
                   pr.controlled_lang_s10_sa_score,
                   pr.controlled_lang_inv_sa_score,
                   pgf.template_name
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id
            WHERE l.entry_id = ? OR l.result_id = ? OR pr.result_id = ?
            ORDER BY l.composite_score DESC
            LIMIT 1
            """,
            (target, target, target),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else {}


def _train_base(
    graph_json_str: str,
    *,
    base_steps: int,
    device: str,
) -> torch.nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph]).to(device)
    batches = [torch.randint(0, VOCAB_SIZE, (4, 128), device=device) for _ in range(8)]
    micro_train_loop(model, batches, vocab_size=VOCAB_SIZE, n_steps=base_steps, lr=3e-4)
    return model


def _run_target(
    row: dict[str, Any],
    *,
    configs: tuple[tuple[int, int], ...],
    base_steps: int,
    n_eval_per_bucket: int,
    hold_out_frac: float,
    n_classes: int,
    device: str,
    timeout_s: float,
    seeds: tuple[int, ...] = (42,),
) -> list[dict[str, Any]]:
    model = _train_base(row["graph_json"], base_steps=base_steps, device=device)
    try:
        out: list[dict[str, Any]] = []
        for vocab, steps in configs:
            for seed in seeds:
                t0 = time.perf_counter()
                result = nano_controlled_holdout_probe(
                    model,
                    active_vocab_size=vocab,
                    n_train_steps=steps,
                    n_eval_per_bucket=n_eval_per_bucket,
                    hold_out_frac=hold_out_frac,
                    n_classes=n_classes,
                    device=device,
                    timeout_s=timeout_s,
                    tokenizer="tiktoken",
                    tiktoken_encoding="gpt2",
                    seed=int(seed),
                )
                payload = result.to_dict()
                payload["config_vocab"] = vocab
                payload["config_steps"] = steps
                payload["seed"] = int(seed)
                payload["wall_seconds"] = round(time.perf_counter() - t0, 3)
                out.append(payload)
        return out
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def _build_report(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    configs: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for row in rows:
        logger.info(
            "running %s (%s)", row["result_id"], row.get("template_name") or "?"
        )
        try:
            results = _run_target(
                row,
                configs=configs,
                base_steps=int(args.base_train_steps),
                n_eval_per_bucket=int(args.n_eval_per_bucket),
                hold_out_frac=float(args.hold_out_frac),
                n_classes=int(args.n_classes),
                device=str(args.device),
                timeout_s=float(args.timeout_s),
                seeds=tuple(int(s) for s in args.seeds),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("target %s failed", row.get("result_id"))
            results = [{"status": "error", "error": str(exc)}]
        report.append(
            {
                "target": {
                    k: row.get(k)
                    for k in (
                        "entry_id",
                        "result_id",
                        "tier",
                        "composite_score",
                        "template_name",
                        "induction_auc",
                        "binding_composite",
                        "induction_v2_investigation_auc",
                        "binding_v2_investigation_auc",
                        "controlled_lang_s05_sa_score",
                        "controlled_lang_s10_sa_score",
                        "controlled_lang_inv_sa_score",
                    )
                },
                "results": results,
            }
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=Path("research/lab_notebook.db"), type=Path)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        help="Probe config as VOCAB:STEPS. Can be repeated.",
    )
    parser.add_argument("--base-train-steps", type=int, default=750)
    parser.add_argument(
        "--n-eval-per-bucket", type=int, default=DEFAULT_N_EVAL_PER_BUCKET
    )
    parser.add_argument("--hold-out-frac", type=float, default=DEFAULT_HOLD_OUT_FRAC)
    parser.add_argument("--n-classes", type=int, default=DEFAULT_N_CLASSES)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="One or more probe seeds.  Each seed reshuffles held-out splits and rng.",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"research/reports/nano_controlled_holdout_tune_{int(time.time())}.json"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configs = tuple(args.config or DEFAULT_CONFIGS)
    rows = [_load_target(args.db, target) for target in args.targets]
    rows = [row for row in rows if row]
    if not rows:
        logger.error("no targets found")
        return 1

    report = _build_report(rows, args=args, configs=configs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
