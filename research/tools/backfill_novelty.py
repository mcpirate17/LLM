#!/usr/bin/env python3
"""Backfill novelty scores for S1 survivors missing them.

Recompiles each model from graph_json, runs behavioral fingerprinting on GPU,
computes full novelty_score (structural + behavioral), and updates
program_results + leaderboard.

Usage:
    python -m research.tools.backfill_novelty [--db PATH] [--device DEVICE]
        [--batch-size 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional


def _fetch_candidates(
    nb,
    *,
    include_all: bool = False,
    recalculate_top: bool = False,
    limit: Optional[int] = None,
    leaderboard_only: bool = False,
) -> List[Dict]:
    """Return candidate program rows for novelty backfill/recalculation."""
    if recalculate_top:
        query = [
            "SELECT pr.result_id, pr.graph_json, pr.graph_fingerprint, pr.param_count,",
            "       pr.novelty_score, l.composite_score",
            "FROM program_results pr",
            "LEFT JOIN leaderboard l ON l.result_id = pr.result_id",
            "WHERE pr.graph_json IS NOT NULL AND pr.graph_json != ''",
            "  AND pr.novelty_score IS NOT NULL",
        ]
        if not include_all:
            query.append("  AND pr.stage1_passed = 1")
        if leaderboard_only:
            query.append("  AND l.result_id IS NOT NULL")
        query.append("ORDER BY COALESCE(l.composite_score, 0) DESC, pr.timestamp DESC")
        params: List[object] = []
    else:
        query = [
            "SELECT result_id, graph_json, graph_fingerprint, param_count, novelty_score",
            "FROM program_results",
            "WHERE (novelty_score IS NULL OR novelty_score = 0)",
            "  AND graph_json IS NOT NULL AND graph_json != ''",
        ]
        if not include_all:
            query.append("  AND stage1_passed = 1")
        query.append("ORDER BY timestamp DESC")
        params = []
    if limit is not None:
        query.append("LIMIT ?")
        params.append(int(limit))
    rows = nb.conn.execute("\n".join(query), tuple(params)).fetchall()
    return [dict(row) for row in rows]


def backfill_novelty(
    db_path: str,
    device: str = "cuda",
    batch_size: int = 50,
    dry_run: bool = False,
    verbose: bool = True,
    include_all: bool = False,
):
    """Compute and store novelty scores for S1 survivors that lack them."""
    import torch

    from ..eval.fingerprint import BehavioralFingerprint, compute_fingerprint
    from ..eval.metrics import novelty_score
    from ..scientist.notebook import LabNotebook
    from ..synthesis.compiler import compile_model
    from ..synthesis.graph import ComputationGraph
    from ..synthesis.serializer import graph_from_json
    from ..mathspaces.registry import register_all_mathspaces

    # Ensure all exotic/mathspace primitives are registered
    register_all_mathspaces()

    nb = LabNotebook(db_path)

    candidates = _fetch_candidates(nb, include_all=include_all)

    if verbose:
        print(f"Novelty backfill")
        print(f"  DB: {db_path}")
        print(f"  Device: {device}")
        scope = "all fingerprints" if include_all else "S1 survivors"
        print(f"  Candidates: {len(candidates)} {scope} missing novelty")
        print()

    if not candidates:
        print("Nothing to backfill.")
        nb.close()
        return

    if dry_run:
        print("DRY RUN — would compute novelty for:")
        for c in candidates[:20]:
            fp = (c["graph_fingerprint"] or "??")[:12]
            print(f"  {fp} params={c.get('param_count', '?')}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
        nb.close()
        return

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {"scored": 0, "structural_only": 0, "failed": 0}

    for i, c in enumerate(candidates):
        rid = c["result_id"]
        fp_str = (c["graph_fingerprint"] or "??")[:12]
        graph_json = c["graph_json"]

        try:
            # 1. Parse graph
            graph = graph_from_json(graph_json)
            if graph is None:
                raise ValueError("graph_from_json returned None")

            # 2. Try to compile and fingerprint behaviorally
            behavioral_fp = None
            try:
                n_layers = 1
                model = compile_model(
                    [graph] * n_layers,
                    vocab_size=32000,
                    max_seq_len=128,
                ).to(dev)
                model.eval()

                behavioral_fp = compute_fingerprint(
                    model,
                    seq_len=min(64, 128),
                    model_dim=graph.model_dim if hasattr(graph, "model_dim") else 256,
                    vocab_size=32000,
                    device=str(dev),
                )

                # Clean up GPU memory
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e_fp:
                if verbose and i < 5:
                    print(f"  [{i+1}] {fp_str}: behavioral fingerprint failed ({e_fp}), using structural only")

            # 3. Compute novelty score
            nov = novelty_score(graph, fingerprint=behavioral_fp)

            # Cast to native Python float to avoid numpy.float32 blob storage in SQLite
            n_score = float(nov.overall_novelty)
            s_nov = float(nov.structural_novelty)
            b_nov = float(nov.behavioral_novelty)
            confidence = float(nov.novelty_confidence)

            # 4. Update program_results
            update_fields = {
                "novelty_score": n_score,
                "structural_novelty": s_nov,
                "behavioral_novelty": b_nov,
                "most_similar_to": nov.most_similar_to,
                "novelty_confidence": confidence,
                "novelty_raw_score": nov.raw_novelty,
                "novelty_z_score": nov.novelty_z_score,
                "novelty_reference_version": nov.novelty_reference_version,
                "novelty_valid_for_promotion": int(nov.novelty_valid_for_promotion),
                "novelty_validity_reason": nov.novelty_validity_reason,
                "novelty_requires_justification": int(
                    getattr(nov, "novelty_requires_justification", False)
                ),
                "cka_source": (
                    "artifact" if behavioral_fp and getattr(behavioral_fp, "cka_source", None) == "artifact"
                    else "heuristic" if behavioral_fp
                    else "structural_only"
                ),
            }
            if behavioral_fp is not None:
                try:
                    update_fields["fp_jacobian_spectral_norm"] = float(
                        getattr(behavioral_fp, "jacobian_spectral_norm", 0.0) or 0.0
                    )
                except Exception:
                    pass

            set_clauses = ", ".join(f"{k} = ?" for k in update_fields)
            values = list(update_fields.values()) + [rid]
            nb.conn.execute(
                f"UPDATE program_results SET {set_clauses} WHERE result_id = ?",
                values,
            )

            # 5. Update leaderboard if entry exists
            lb_row = nb.conn.execute(
                "SELECT entry_id, screening_novelty FROM leaderboard WHERE result_id = ?",
                (rid,),
            ).fetchone()
            if lb_row:
                entry_id = lb_row["entry_id"]
                old_nov = lb_row["screening_novelty"]
                # Update screening_novelty and recompute composite_score
                nb.conn.execute(
                    "UPDATE leaderboard SET screening_novelty = ? WHERE entry_id = ?",
                    (n_score, entry_id),
                )
                # Recompute composite via promote_to_tier (preserves current tier)
                lb_full = nb.conn.execute(
                    "SELECT * FROM leaderboard WHERE entry_id = ?",
                    (entry_id,),
                ).fetchone()
                if lb_full:
                    d = dict(lb_full)
                    pr_conf = nb.conn.execute(
                        "SELECT novelty_confidence FROM program_results WHERE result_id = ?",
                        (rid,),
                    ).fetchone()
                    nov_conf = pr_conf["novelty_confidence"] if pr_conf else None
                    composite = nb.compute_composite_score(
                        screening_lr=d.get("screening_loss_ratio"),
                        screening_nov=n_score,
                        inv_lr=d.get("investigation_loss_ratio"),
                        inv_robust=d.get("investigation_robustness"),
                        val_lr=d.get("validation_loss_ratio"),
                        val_baseline=d.get("validation_baseline_ratio"),
                        val_std=d.get("validation_multi_seed_std"),
                        novelty_confidence=nov_conf,
                        scaling_param_efficiency=d.get("scaling_param_efficiency"),
                    )
                    nb.conn.execute(
                        "UPDATE leaderboard SET composite_score = ? WHERE entry_id = ?",
                        (composite, entry_id),
                    )

            if behavioral_fp is not None:
                results["scored"] += 1
            else:
                results["structural_only"] += 1

            if verbose and (i < 10 or (i + 1) % 25 == 0 or i == len(candidates) - 1):
                kind = "full" if behavioral_fp else "struct"
                print(
                    f"  [{i+1}/{len(candidates)}] {fp_str}: "
                    f"novelty={n_score:.3f} (s={s_nov:.3f} b={b_nov:.3f}) "
                    f"conf={confidence:.2f} [{kind}]"
                )

            # Commit in batches
            if (i + 1) % batch_size == 0:
                nb.conn.commit()

        except Exception as e:
            results["failed"] += 1
            if verbose and (i < 10 or results["failed"] <= 5):
                print(f"  [{i+1}/{len(candidates)}] {fp_str}: FAILED {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    nb.conn.commit()
    nb.close()

    if verbose:
        print()
        total = results["scored"] + results["structural_only"] + results["failed"]
        print(f"Done. Processed {total}/{len(candidates)}:")
        print(f"  Full (behavioral + structural): {results['scored']}")
        print(f"  Structural only: {results['structural_only']}")
        print(f"  Failed: {results['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill novelty scores for S1 survivors"
    )
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab_notebook.db",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Commit every N entries (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be scored"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Backfill novelty for all fingerprints (not just S1 survivors)",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    backfill_novelty(
        args.db,
        device=args.device,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        include_all=args.all,
    )


if __name__ == "__main__":
    main()
