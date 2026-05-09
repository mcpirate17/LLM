"""Build an offline investigation queue from existing empirical evidence.

This intentionally does not enable ML gating or learned grammar influence.  It
uses persisted GBM rank heads only as an advisory tie-breaker over candidates
that already have recorded evidence in the notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from research.defaults import RUNS_DB
from research.scientist.intelligence.ml_corpus import (
    load_screening_predictor_corpus_rows,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.intelligence.predictor_artifacts import STATE_DIR
from research.scientist.intelligence.predictor_gbm import GBMPredictor
from research.synthesis.graph_features import (
    enrich_with_op_stats,
    extract_graph_features_bundle,
    load_op_stats,
)
from research.synthesis.grammar_support import (
    DBOpWeightCache,
    DBTemplateWeightCache,
    _capability_score,
)

REPORT_DIR = Path("research/reports")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _quality_from_loss(loss_ratio: Any) -> float:
    loss = _float_or_none(loss_ratio)
    if loss is None:
        return 0.35
    return _bounded(1.0 - (loss / 1.25), 0.0, 1.0)


def _quality_from_ppl(ppl: Any) -> float:
    value = _float_or_none(ppl)
    if value is None:
        return 0.0
    return _bounded(math.exp(-max(value, 0.0) / 25.0), 0.0, 1.0)


def _quality_from_composite(composite: Any) -> float:
    value = _float_or_none(composite)
    if value is None:
        return 0.0
    return _bounded(value / 100.0, 0.0, 1.0)


def _quality_from_predicted_ppl(ppl: Any) -> float:
    value = _float_or_none(ppl)
    if value is None or value >= 1e5:
        return 0.0
    return _quality_from_ppl(value)


def _quality_from_predicted_composite(composite: Any) -> float:
    value = _float_or_none(composite)
    if value is None or value <= -1e5 or value >= 1e5:
        return 0.0
    return _quality_from_composite(value)


def _missing_signals(row: Dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if row.get("induction_intermediate_auc_best") is None:
        missing.append("induction_intermediate_auc")
    if row.get("binding_intermediate_auc_best") is None:
        missing.append("binding_intermediate_auc")
    if row.get("validation_loss_ratio_best") is None:
        missing.append("validation_loss_ratio")
    if row.get("wikitext_perplexity_best") is None:
        missing.append("wikitext_perplexity")
    return missing


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _parse_graph_metadata(graph_json: str) -> tuple[list[str], list[str], list[str]]:
    try:
        graph = json.loads(graph_json)
    except (TypeError, json.JSONDecodeError):
        return [], [], []
    metadata = graph.get("metadata") if isinstance(graph, dict) else {}
    templates = [
        str(x)
        for x in (metadata.get("templates_used") if isinstance(metadata, dict) else [])
        or []
        if str(x)
    ]
    motifs = [
        str(x)
        for x in (metadata.get("motifs_used") if isinstance(metadata, dict) else [])
        or []
        if str(x)
    ]
    ops: list[str] = []
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
    node_iter = nodes.values() if isinstance(nodes, dict) else nodes
    for node in node_iter:
        if not isinstance(node, dict):
            continue
        op_name = str(node.get("op_name") or "")
        if op_name and op_name != "input":
            ops.append(op_name)
    return templates, motifs, ops


def _build_feature_dict(
    graph_json: str,
    row: Dict[str, Any],
    op_stats_cache: Dict[str, Any],
) -> Dict[str, float]:
    graph = json.loads(graph_json)
    feats, ops = extract_graph_features_bundle(graph)
    for op in ops:
        if op:
            feats[f"op_{op}"] = feats.get(f"op_{op}", 0.0) + 1.0
    enrich_with_op_stats(feats, ops, preloaded=op_stats_cache)
    for post_key in (
        "hellaswag_acc_best",
        "induction_screening_auc_best",
        "ar_legacy_auc_best",
        "blimp_overall_accuracy_best",
        "binding_screening_composite_best",
        "induction_intermediate_auc_best",
        "binding_intermediate_auc_best",
        "validation_loss_ratio_best",
        "rapid_screening_passed_best",
        "initial_loss_best",
        "mean_grad_norm_best",
        "max_grad_norm_best",
        "grad_norm_std_best",
    ):
        value = row.get(post_key)
        feats[post_key] = float(value) if value is not None else float("nan")
    return feats


def _load_best_result_metadata(db_path: str) -> Dict[str, Dict[str, Any]]:
    from research.scientist.intelligence.ml_corpus import _graph_fingerprint

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT result_id, experiment_id, graph_json, graph_fingerprint,
               model_source, result_cohort, timestamp, stage1_passed, loss_ratio,
               validation_loss_ratio, novelty_score
        FROM program_results
        WHERE TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{}'
        """
    ).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        graph_json = resolve_graph_json_value(conn, db_path, row["graph_json"])
        canonical = _graph_fingerprint(graph_json)
        loss = _float_or_none(row["loss_ratio"])
        rank = (
            0 if bool(row["stage1_passed"]) else 1,
            loss is None,
            loss if loss is not None else float("inf"),
            -float(row["timestamp"] or 0.0),
        )
        current = out.get(canonical)
        if current is None or rank < current["_rank"]:
            out[canonical] = {
                "_rank": rank,
                "result_id": str(row["result_id"] or ""),
                "experiment_id": str(row["experiment_id"] or ""),
                "graph_fingerprint": str(row["graph_fingerprint"] or canonical),
                "model_source": str(row["model_source"] or ""),
                "result_cohort": str(row["result_cohort"] or ""),
                "timestamp": float(row["timestamp"] or 0.0),
                "novelty_score": _float_or_none(row["novelty_score"]),
                "validation_loss_ratio": _float_or_none(row["validation_loss_ratio"]),
            }
    conn.close()
    for value in out.values():
        value.pop("_rank", None)
    return out


def build_queue(
    db_path: str,
    *,
    limit: int = 100,
    include_investigated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_screening_predictor_corpus_rows(db_path, validate=False)
    gbm = GBMPredictor.load(STATE_DIR)
    op_stats_cache = load_op_stats(db_path)
    db_template_weights = DBTemplateWeightCache(ttl=0.0).get(db_path) or {}
    db_op_weights = DBOpWeightCache(ttl=0.0).get(db_path) or {}
    result_meta = _load_best_result_metadata(db_path)

    candidates: list[dict[str, Any]] = []
    skipped = {
        "not_stage1": 0,
        "already_investigated": 0,
        "feature_error": 0,
    }
    for row in rows:
        if not bool(row.get("stage1_any_passed")):
            skipped["not_stage1"] += 1
            continue
        if (
            not include_investigated
            and row.get("induction_intermediate_auc_best") is not None
            and row.get("binding_intermediate_auc_best") is not None
        ):
            skipped["already_investigated"] += 1
            continue

        graph_json = str(row.get("graph_json") or "")
        canonical = str(row.get("canonical_fingerprint") or "")
        try:
            feats = _build_feature_dict(graph_json, row, op_stats_cache)
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            skipped["feature_error"] += 1
            continue

        templates, motifs, ops = _parse_graph_metadata(graph_json)
        table_template = _mean(
            min(max(float(db_template_weights.get(name, 1.0)), 0.0), 20.0) / 10.0
            for name in templates
        )
        table_op = _mean(
            min(max(float(db_op_weights.get(name, 1.0)), 0.0), 4.5) / 4.5
            for name in ops
        )
        table_score = _bounded((0.65 * table_template) + (0.35 * table_op))

        capability = _bounded(
            _capability_score(
                row.get("induction_screening_auc_best"),
                row.get("binding_screening_auc_best"),
                row.get("binding_screening_composite_best"),
                row.get("ar_legacy_auc_best"),
                row.get("hellaswag_acc_best"),
                row.get("blimp_overall_accuracy_best"),
                row.get("induction_intermediate_auc_best"),
                row.get("binding_intermediate_auc_best"),
                0.0,
            )
            / 1.5,
            0.0,
            1.0,
        )
        observed_quality = _bounded(
            (0.40 * _quality_from_loss(row.get("loss_ratio_best")))
            + (0.20 * _quality_from_ppl(row.get("wikitext_perplexity_best")))
            + (0.20 * _quality_from_composite(row.get("composite_score_best")))
            + (0.20 * capability)
        )

        predicted_ppl = gbm.predict_rank_ppl(feats)
        predicted_composite = gbm.predict_rank_composite(feats)
        predicted_quality = _bounded(
            (
                _quality_from_predicted_ppl(predicted_ppl)
                + _quality_from_predicted_composite(predicted_composite)
            )
            / 2.0
        )

        novelty = _bounded(
            _float_or_none((result_meta.get(canonical) or {}).get("novelty_score"))
            or 0.0
        )
        missing_signals = _missing_signals(row)
        missing_v2 = int(row.get("induction_intermediate_auc_best") is None) + int(
            row.get("binding_intermediate_auc_best") is None
        )
        missing_validation = int(row.get("validation_loss_ratio_best") is None)
        info_gap = _bounded((0.40 * missing_v2) + (0.20 * missing_validation), 0.0, 1.0)

        final_score = _bounded(
            (0.42 * observed_quality)
            + (0.23 * table_score)
            + (0.20 * predicted_quality)
            + (0.10 * novelty)
            + (0.05 * info_gap)
        )
        meta = result_meta.get(canonical) or {}
        candidates.append(
            {
                "rank_score": final_score,
                "observed_quality": observed_quality,
                "table_score": table_score,
                "gbm_rank_quality": predicted_quality,
                "novelty_score": novelty,
                "info_gap": info_gap,
                "canonical_fingerprint": canonical,
                "result_id": meta.get("result_id", ""),
                "experiment_id": meta.get("experiment_id", ""),
                "graph_fingerprint": meta.get("graph_fingerprint", canonical),
                "model_source": meta.get("model_source", ""),
                "result_cohort": meta.get("result_cohort", ""),
                "stage1_pass_rate": float(row.get("stage1_pass_rate") or 0.0),
                "n_rows": int(row.get("n_rows") or 0),
                "loss_ratio_best": _float_or_none(row.get("loss_ratio_best")),
                "validation_loss_ratio_best": _float_or_none(
                    row.get("validation_loss_ratio_best")
                ),
                "wikitext_perplexity_best": _float_or_none(
                    row.get("wikitext_perplexity_best")
                ),
                "composite_score_best": _float_or_none(row.get("composite_score_best")),
                "induction_screening_auc_best": _float_or_none(
                    row.get("induction_screening_auc_best")
                ),
                "binding_screening_composite_best": _float_or_none(
                    row.get("binding_screening_composite_best")
                ),
                "induction_intermediate_auc_best": _float_or_none(
                    row.get("induction_intermediate_auc_best")
                ),
                "binding_intermediate_auc_best": _float_or_none(
                    row.get("binding_intermediate_auc_best")
                ),
                "predicted_ppl": _float_or_none(predicted_ppl),
                "predicted_composite": _float_or_none(predicted_composite),
                "templates": templates,
                "motifs": motifs[:12],
                "ops": sorted(set(ops)),
                "missing_signals": missing_signals,
                "recommended_next_step": (
                    "run_v2_investigation"
                    if missing_v2
                    else "rerun_validation_or_replay"
                ),
            }
        )

    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": db_path,
        "limit": int(limit),
        "include_investigated": bool(include_investigated),
        "n_source_rows": len(rows),
        "n_ranked_candidates": len(candidates),
        "skipped": skipped,
        "uses_screening_ensemble_gate": False,
        "uses_learned_generation_influence": False,
        "scoring_formula": (
            "0.42*observed_quality + 0.23*table_score + "
            "0.20*gbm_rank_quality + 0.10*novelty + 0.05*info_gap"
        ),
    }
    return candidates[:limit], summary


def write_reports(
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_prefix: Path,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    json_path.write_text(
        json.dumps({"summary": summary, "candidates": candidates}, indent=2),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return json_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=RUNS_DB)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-investigated", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output path without suffix. Defaults to research/reports/investigation_queue_YYYY-MM-DD.",
    )
    args = parser.parse_args()

    candidates, summary = build_queue(
        args.db,
        limit=max(1, int(args.limit)),
        include_investigated=bool(args.include_investigated),
    )
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prefix = REPORT_DIR / f"investigation_queue_{stamp}"
    json_path, jsonl_path = write_reports(candidates, summary, output_prefix=prefix)
    print(f"Wrote {len(candidates)} candidates")
    print(f"JSON:  {json_path}")
    print(f"JSONL: {jsonl_path}")
    for idx, row in enumerate(candidates[:10], 1):
        print(
            f"{idx:2d}. score={row['rank_score']:.3f} "
            f"rid={row.get('result_id') or '-'} "
            f"loss={row.get('loss_ratio_best')} "
            f"step={row.get('recommended_next_step')} "
            f"templates={','.join(row.get('templates') or [])}"
        )


if __name__ == "__main__":
    main()
