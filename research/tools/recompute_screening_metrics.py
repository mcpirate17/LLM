#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from research.scientist.notebook import LabNotebook
from research.scientist.screening_recompute import recompute_screening_metrics
from research.tools._db_maintenance import connect_readonly


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute full screening metrics in place for existing program rows."
    )
    parser.add_argument("--db", type=Path, default=Path("research/runs.db"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--result-id", action="append", default=[])
    parser.add_argument("--fingerprint", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--skip-rapid", action="store_true")
    parser.add_argument("--skip-fingerprint", action="store_true")
    parser.add_argument("--skip-post-train", action="store_true")
    parser.add_argument(
        "--allow-insufficient-learning-metrics",
        action="store_true",
    )
    return parser.parse_args()


def _select_result_ids(args: argparse.Namespace) -> List[str]:
    con = connect_readonly(args.db)
    try:
        if args.result_id:
            return [str(rid) for rid in args.result_id if str(rid).strip()]

        if args.fingerprint:
            result_ids: List[str] = []
            for fp in args.fingerprint:
                row = con.execute(
                    """
                    SELECT result_id
                    FROM program_results_compat
                    WHERE graph_fingerprint = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (str(fp),),
                ).fetchone()
                if row:
                    result_ids.append(str(row["result_id"]))
            return result_ids

        clauses = ["TRIM(COALESCE(graph_json, '')) <> ''", "graph_json <> '{}'"]
        if args.stage1_only:
            clauses.append("COALESCE(stage1_passed, 0) = 1")
        if args.missing_only:
            clauses.append(
                "("
                "fp_jacobian_spectral_norm IS NULL OR "
                "fp_interaction_locality IS NULL OR "
                "activation_sparsity_score IS NULL OR "
                "fp_isotropy IS NULL OR "
                "fp_rank_ratio IS NULL OR "
                "fp_sensitivity_uniformity IS NULL OR "
                "hellaswag_acc IS NULL OR "
                "induction_screening_auc IS NULL OR "
                "binding_screening_auc IS NULL OR "
                "discovery_loss_ratio IS NULL OR "
                "validation_loss_ratio IS NULL"
                ")"
            )
        limit_clause = f" LIMIT {int(args.limit)}" if int(args.limit) > 0 else ""
        rows = con.execute(
            f"""
            SELECT result_id
            FROM program_results_compat
            WHERE {" AND ".join(clauses)}
            ORDER BY rowid DESC
            {limit_clause}
            """
        ).fetchall()
        return [str(row["result_id"]) for row in rows]
    finally:
        con.close()


def main() -> None:
    args = _parse_args()
    result_ids = _select_result_ids(args)
    nb = LabNotebook(str(args.db))
    try:
        print(f"recompute_screening_metrics: {len(result_ids)} rows on {args.device}")
        for idx, result_id in enumerate(result_ids, start=1):
            payload = recompute_screening_metrics(
                nb=nb,
                notebook_path=args.db,
                result_id=result_id,
                device=str(args.device),
                include_rapid=not bool(args.skip_rapid),
                include_fingerprint=not bool(args.skip_fingerprint),
                include_post_train=not bool(args.skip_post_train),
                allow_insufficient_learning_metrics=bool(
                    args.allow_insufficient_learning_metrics
                ),
                provenance_source="recompute_screening_metrics_cli",
            )
            print(
                f"[{idx}/{len(result_ids)}] {result_id} "
                f"status={payload.get('status')} "
                f"updates={len(payload.get('updates') or {})} "
                f"errors={','.join(sorted((payload.get('errors') or {}).keys())) or '-'}"
            )
    finally:
        nb.close()


if __name__ == "__main__":
    main()
