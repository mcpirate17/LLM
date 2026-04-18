from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research.scientist.intelligence.graph_segments import (
    build_feature_matrices,
    evaluate_feature_families,
    extract_graph_segments,
    load_stage05_native_segment_corpus,
    summarize_binary_fragment_associations,
)

DEFAULT_DB = Path("research/lab_notebook.db")
DEFAULT_OUT_DIR = Path("research/reports/graph_segments")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph segment fingerprint datasets and offline model comparisons."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_graph_table(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "canonical_fingerprint",
                "n_rows",
                "latest_timestamp",
                "stage1_any_passed",
                "stage1_pass_rate",
                "loss_ratio_best",
                "wikitext_perplexity_best",
                "binding_auc",
                "induction_auc",
                "hellaswag_acc",
                "binding_positive",
                "induction_positive",
                "hellaswag_positive",
                "all_three_positive",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.canonical_fingerprint,
                    row.n_rows,
                    row.latest_timestamp,
                    int(row.stage1_any_passed),
                    row.stage1_pass_rate,
                    row.loss_ratio_best,
                    row.wikitext_perplexity_best,
                    row.binding_auc,
                    row.induction_auc,
                    row.hellaswag_acc,
                    int(row.binding_positive),
                    int(row.induction_positive),
                    int(row.hellaswag_positive),
                    int(row.all_three_positive),
                ]
            )


def _write_incidence_table(path: Path, rows, extractions) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["canonical_fingerprint", "fragment_id", "path_len", "count", "present"]
        )
        for row, extraction in zip(rows, extractions):
            for fragment_id, count in sorted(extraction.count_map.items()):
                path_len = int(fragment_id.split(":", 1)[0].replace("seg_p", ""))
                writer.writerow(
                    [row.canonical_fingerprint, fragment_id, path_len, int(count), 1]
                )


def _write_dictionary_table(path: Path, extractions) -> None:
    support_graphs: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for extraction in extractions:
        for fragment_id in extraction.presence_set:
            support_graphs[fragment_id] = support_graphs.get(fragment_id, 0) + 1
        for fragment_id, count in extraction.count_map.items():
            total_counts[fragment_id] = total_counts.get(fragment_id, 0) + int(count)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["fragment_id", "path_len", "support_graphs", "support_total_count"]
        )
        for fragment_id in sorted(total_counts):
            writer.writerow(
                [
                    fragment_id,
                    int(fragment_id.split(":", 1)[0].replace("seg_p", "")),
                    int(support_graphs.get(fragment_id, 0)),
                    int(total_counts.get(fragment_id, 0)),
                ]
            )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_stage05_native_segment_corpus(args.db_path)
    _write_graph_table(out_dir / "graph_table.tsv", rows)

    extractions = [extract_graph_segments(row.graph_json) for row in rows]
    _write_incidence_table(out_dir / "segment_incidence.tsv", rows, extractions)
    _write_dictionary_table(out_dir / "fragment_dictionary.tsv", extractions)

    support_associations = {}
    for target_name in (
        "stage1_any_passed",
        "binding_positive",
        "induction_positive",
        "hellaswag_positive",
        "all_three_positive",
    ):
        associations = summarize_binary_fragment_associations(
            rows,
            extractions,
            target_name=target_name,
            min_support=args.min_support,
        )
        support_associations[target_name] = [
            {
                "fragment_id": item.fragment_id,
                "path_len": item.path_len,
                "support_graphs": item.support_graphs,
                "support_total_count": item.support_total_count,
                "present_rate": item.present_rate,
                "absent_rate": item.absent_rate,
                "rate_lift": item.rate_lift,
                "posterior_mean": item.posterior_mean,
                "posterior_low": item.posterior_low,
                "posterior_high": item.posterior_high,
            }
            for item in associations[:200]
        ]
    _write_json(out_dir / "fragment_associations.json", support_associations)

    model_report = evaluate_feature_families(
        rows,
        min_support=args.min_support,
        seed=args.seed,
    )
    _write_json(out_dir / "model_comparison.json", model_report)

    _, _, baseline_names, fragment_names, _ = build_feature_matrices(
        rows,
        min_support=args.min_support,
    )
    summary = {
        "db_path": str(Path(args.db_path)),
        "out_dir": str(out_dir),
        "n_graphs": len(rows),
        "n_baseline_features": len(baseline_names),
        "n_fragment_features": len(fragment_names),
        "min_support": int(args.min_support),
    }
    _write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
