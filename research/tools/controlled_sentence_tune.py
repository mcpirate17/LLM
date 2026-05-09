"""Tune the controlled sentence diagnostic on selected leaderboard rows.

Read-only calibration harness.  It compiles saved graph JSON from the notebook,
does the same quick base train used by language-control backfills, then runs
``controlled_sentence_probe`` across a vocab-size × train-step grid.  This is a
language-shape and step-curve diagnostic; held-out slot/binding composition is
owned by ``nano_blimp_v3``.

Usage:
    python -m research.tools.controlled_sentence_tune \
        --targets ec7025d7-338 8d087a16-692 903157e5-219
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.controlled_sentence_probe import (
    CONTROLLED_SENTENCE_PROBE_ROLE,
    build_sentence_probe_corpus,
    controlled_sentence_probe,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools._db_maintenance import connect_readonly
from research.tools._tuning_train import train_compiled_graph_base

logger = logging.getLogger(__name__)

VOCAB_SIZE = 50257
DEFAULT_TARGETS = ("ec7025d7-338", "8d087a16-692", "903157e5-219")
DEFAULT_CONFIGS = ((256, 40), (512, 100), (1000, 300))
DEFAULT_SEEDS = (42,)
_CURVE_METRICS = (
    "controlled_sentence_score",
    "controlled_sentence_nano_hellaswag_acc",
    "controlled_sentence_nano_blimp_order_acc",
    "controlled_sentence_nano_blimp_binding_acc",
)


def _corpus_summary(
    *, active_vocab_size: int, n_eval_items: int, seed: int
) -> dict[str, Any]:
    corpus = build_sentence_probe_corpus(
        active_vocab_size=active_vocab_size,
        vocab_size=VOCAB_SIZE,
        tokenizer="tiktoken",
        tiktoken_encoding="gpt2",
        n_eval_items=n_eval_items,
        seed=seed,
    )
    sentence_shape_sample = [
        {
            "prefix": item.prefix,
            "correct": item.correct,
            "distractors": list(item.distractors),
            "good": item.good_sentence,
            "bad_order": item.bad_order_sentence,
            "bad_binding": item.bad_binding_sentence,
            "source": item.source,
        }
        for item in corpus.eval_items[:4]
    ]
    return {
        "probe_role": CONTROLLED_SENTENCE_PROBE_ROLE,
        "source_counts": corpus.source_counts,
        "vocab_sample": list(corpus.vocabulary[:40]),
        "train_sample": list(corpus.train_sentences[:8]),
        "sentence_shape_sample": sentence_shape_sample,
        "hellaswag_sample": sentence_shape_sample,
        "blimp_pair_sample": [
            {"good": good, "bad": bad} for good, bad in corpus.blimp_pairs[:4]
        ],
    }


def _parse_config(value: str) -> tuple[int, int]:
    try:
        vocab, steps = value.split(":", 1)
        return int(vocab), int(steps)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("configs must look like VOCAB:STEPS") from exc


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _curve_auc(points: list[tuple[int, float]]) -> float:
    """Trapezoid AUC normalized by the step span."""
    if not points:
        return 0.0
    ordered = sorted((int(step), float(value)) for step, value in points)
    if len(ordered) == 1:
        return ordered[0][1]
    step_min = ordered[0][0]
    step_max = ordered[-1][0]
    if step_max <= step_min:
        return _mean([value for _step, value in ordered])
    area = 0.0
    for (s0, v0), (s1, v1) in zip(ordered, ordered[1:]):
        area += (s1 - s0) * ((v0 + v1) / 2.0)
    return area / (step_max - step_min)


def _step_curve_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("controlled_sentence_status") != "ok":
            continue
        grouped[(int(row["config_vocab"]), int(row["config_seed"]))].append(row)

    summaries = []
    for (vocab, seed), rows in sorted(grouped.items()):
        by_step = {int(row["config_steps"]): row for row in rows}
        if not by_step:
            continue
        baseline_step = min(by_step)
        final_step = max(by_step)
        baseline = by_step[baseline_step]
        final = by_step[final_step]
        item: dict[str, Any] = {
            "config_vocab": vocab,
            "config_seed": seed,
            "baseline_step": baseline_step,
            "final_step": final_step,
        }
        for metric in _CURVE_METRICS:
            points = [
                (int(row["config_steps"]), float(row.get(metric, 0.0))) for row in rows
            ]
            item[f"{metric}_baseline"] = round(float(baseline.get(metric, 0.0)), 4)
            item[f"{metric}_final"] = round(float(final.get(metric, 0.0)), 4)
            item[f"{metric}_delta"] = round(
                float(final.get(metric, 0.0)) - float(baseline.get(metric, 0.0)),
                4,
            )
            item[f"{metric}_auc"] = round(_curve_auc(points), 4)
        summaries.append(item)
    return summaries


def _load_target(db: Path, target: str) -> dict[str, Any]:
    conn = connect_readonly(db)
    try:
        row = conn.execute(
            """
            SELECT l.entry_id, l.result_id, l.tier, l.composite_score,
                   l.induction_screening_auc, l.binding_screening_composite,
                   l.induction_intermediate_auc,
                   l.binding_intermediate_auc,
                   pr.graph_json, pr.graph_fingerprint,
                   pr.language_control_s05_sentence_assoc_score,
                   pr.language_control_s10_sentence_assoc_score,
                   pr.language_control_investigation_sentence_assoc_score,
                   pr.language_control_s05_binding_order_acc,
                   pr.language_control_s10_binding_order_acc,
                   pr.language_control_investigation_binding_order_acc,
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
        if row is None:
            return {}
        payload = dict(row)
        payload["graph_json"] = resolve_graph_json_value(
            conn, db, payload["graph_json"]
        )
        return payload
    finally:
        conn.close()


def _train_base(
    graph_json_str: str, *, base_steps: int, device: str
) -> torch.nn.Module:
    return train_compiled_graph_base(
        graph_json_str,
        base_steps=base_steps,
        device=device,
        vocab_size=VOCAB_SIZE,
    )


def _run_target(
    row: dict[str, Any],
    *,
    configs: tuple[tuple[int, int], ...],
    base_steps: int,
    device: str,
    n_eval_items: int,
    timeout_s: float,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    model = _train_base(row["graph_json"], base_steps=base_steps, device=device)
    try:
        out = []
        for seed in seeds:
            for vocab, steps in configs:
                t0 = time.perf_counter()
                result = controlled_sentence_probe(
                    model,
                    active_vocab_size=vocab,
                    n_train_steps=steps,
                    n_eval_items=n_eval_items,
                    device=device,
                    timeout_s=timeout_s,
                    tokenizer="tiktoken",
                    tiktoken_encoding="gpt2",
                    seed=seed,
                )
                payload = result.to_dict()
                payload["config_vocab"] = vocab
                payload["config_steps"] = steps
                payload["config_seed"] = seed
                payload["wall_seconds"] = round(time.perf_counter() - t0, 3)
                payload["corpus_summary"] = _corpus_summary(
                    active_vocab_size=vocab,
                    n_eval_items=n_eval_items,
                    seed=seed,
                )
                out.append(payload)
        return out
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=Path("research/runs.db"), type=Path)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        help="Probe config as VOCAB:STEPS. Can be repeated.",
    )
    parser.add_argument("--base-train-steps", type=int, default=750)
    parser.add_argument("--n-eval-items", type=int, default=64)
    parser.add_argument("--seed", action="append", type=int, help="Probe seed.")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"research/reports/controlled_sentence_tune_{int(time.time())}.json"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configs = tuple(args.config or DEFAULT_CONFIGS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    rows = [_load_target(args.db, target) for target in args.targets]
    rows = [row for row in rows if row]
    if not rows:
        logger.error("no targets found")
        return 1

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
                device=str(args.device),
                n_eval_items=int(args.n_eval_items),
                timeout_s=float(args.timeout_s),
                seeds=seeds,
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
                        "induction_screening_auc",
                        "binding_screening_composite",
                        "induction_intermediate_auc",
                        "binding_intermediate_auc",
                        "language_control_s05_sentence_assoc_score",
                        "language_control_s10_sentence_assoc_score",
                        "language_control_investigation_sentence_assoc_score",
                        "language_control_s05_binding_order_acc",
                        "language_control_s10_binding_order_acc",
                        "language_control_investigation_binding_order_acc",
                    )
                },
                "results": results,
                "step_curve_summary": _step_curve_summary(results),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
