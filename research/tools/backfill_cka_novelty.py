"""Backfill CKA fingerprints and novelty scores for program_results.

Recompiles each graph from graph_json, runs a forward pass, and computes
the full behavioral fingerprint including CKA similarity scores. Updates
program_results rows that have NULL or degenerate CKA.

Each model is fingerprinted in a subprocess to isolate segfaults from
C-kernel ops that crash on certain graph topologies.

Usage:
    python -m research.tools.backfill_cka_novelty                 # all missing
    python -m research.tools.backfill_cka_novelty --limit 50      # first 50
    python -m research.tools.backfill_cka_novelty --dry-run       # preview
    python -m research.tools.backfill_cka_novelty --force         # re-run all
    python -m research.tools.backfill_cka_novelty --device cpu    # force CPU
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time

from research.tools._script_audit import (
    build_metric_backfill_context,
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)
from research.tools.backfill import store_probe_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def _fingerprint_one(result_id: str, graph_json_str: str, device: str):
    """Run in subprocess: compile model, compute fingerprint, return updates dict."""
    os.environ["ARIA_DISABLE_NATIVE_CKA"] = "1"

    from research.eval.fingerprint import compute_fingerprint
    from research.synthesis.compiler import compile_model
    from research.synthesis.serializer import graph_from_json

    graph = graph_from_json(graph_json_str)
    model_dim = getattr(graph, "model_dim", 256)

    model = compile_model([graph], vocab_size=256, max_seq_len=128)

    fp = compute_fingerprint(
        model,
        seq_len=64,
        model_dim=model_dim,
        vocab_size=256,
        device=device,
        n_probes=32,
        include_cka=True,
        include_behavioral_probes=True,
    )

    updates = {}
    if fp.cka_vs_transformer is not None:
        updates["fp_cka_vs_transformer"] = fp.cka_vs_transformer
        updates["fp_cka_vs_ssm"] = fp.cka_vs_ssm
        updates["fp_cka_vs_conv"] = fp.cka_vs_conv
    updates["cka_source"] = getattr(fp, "cka_source", None)
    updates["cka_artifact_version"] = getattr(fp, "cka_artifact_version", None)
    updates["cka_probe_protocol_hash"] = getattr(fp, "cka_probe_protocol_hash", None)
    updates["cka_reference_quality"] = (
        1 if getattr(fp, "cka_reference_quality", False) else 0
    )
    updates["novelty_score"] = fp.novelty_score
    updates["novelty_valid_for_promotion"] = 1 if fp.novelty_valid_for_promotion else 0
    updates["novelty_validity_reason"] = fp.novelty_validity_reason
    updates["novelty_reference_version"] = getattr(
        fp, "novelty_reference_version", None
    )
    updates["novelty_scoring_policy_version"] = "backfill_cka_v2"

    for attr, col in [
        ("interaction_locality", "fp_interaction_locality"),
        ("interaction_sparsity", "fp_interaction_sparsity"),
        ("interaction_symmetry", "fp_interaction_symmetry"),
        ("interaction_hierarchy", "fp_interaction_hierarchy"),
        ("intrinsic_dim", "fp_intrinsic_dim"),
        ("isotropy", "fp_isotropy"),
        ("rank_ratio", "fp_rank_ratio"),
        ("jacobian_spectral_norm", "fp_jacobian_spectral_norm"),
        ("jacobian_effective_rank", "fp_jacobian_effective_rank"),
        ("sensitivity_uniformity", "fp_sensitivity_uniformity"),
    ]:
        val = getattr(fp, attr, None)
        if val is not None:
            updates[col] = float(val)

    return updates


def _worker(args_tuple):
    """Subprocess wrapper that catches all exceptions."""
    result_id, graph_json_str, device = args_tuple
    try:
        updates = _fingerprint_one(result_id, graph_json_str, device)
        return (result_id, updates, None)
    except Exception as e:
        return (result_id, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill CKA fingerprints and novelty scores"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max entries to process (0=all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if CKA already populated",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Per-model timeout in seconds"
    )
    args = parser.parse_args()

    import torch

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    nb, exp_id = start_script_experiment(
        db_path="research/lab_notebook.db",
        experiment_type="cka_novelty_backfill",
        config={
            "limit": args.limit,
            "force": bool(args.force),
            "device": device,
            "timeout": args.timeout,
        },
        source_script="backfill_cka_novelty",
        hypothesis="Backfill CKA and novelty fingerprints",
    )
    conn = nb.conn
    provenance_context = build_metric_backfill_context(
        kind="cka_novelty_backfill",
        source_script="backfill_cka_novelty",
        experiment_id=exp_id,
        device=device,
        limit=args.limit,
        force=bool(args.force),
        timeout=args.timeout,
    )

    where = "WHERE stage1_passed = 1 AND graph_json IS NOT NULL"
    if not args.force:
        where += (
            " AND (fp_cka_vs_transformer IS NULL"
            " OR novelty_validity_reason LIKE '%degenerate%')"
        )

    limit_clause = f" LIMIT {args.limit}" if args.limit > 0 else ""

    rows = conn.execute(
        f"SELECT result_id, graph_json, loss_ratio "
        f"FROM program_results {where} "
        f"ORDER BY loss_ratio DESC{limit_clause}"
    ).fetchall()

    logger.info(
        "Found %d entries to backfill on %s%s",
        len(rows),
        device,
        " (dry run)" if args.dry_run else "",
    )

    updated = 0
    failed = 0
    crashed = 0
    degenerate = 0
    t0 = time.time()

    ctx = mp.get_context("spawn")

    if not rows:
        complete_script_experiment(
            nb,
            exp_id,
            results={"total": 0, "updated": 0, "failed": 0, "crashed": 0},
            summary="CKA backfill found no candidates",
        )
        nb.close()
        return

    if args.dry_run:
        fail_script_experiment(
            nb,
            exp_id,
            error="Dry-run invocation does not write results",
            results={"total": len(rows), "updated": 0, "dry_run": True},
        )
        nb.close()
        return

    try:
        for i, row in enumerate(rows):
            result_id = row["result_id"]
            graph_json_str = row["graph_json"]

            if not graph_json_str:
                continue

            # Run in subprocess to isolate segfaults
            pool = ctx.Pool(1)
            try:
                async_result = pool.apply_async(
                    _worker, ((result_id, graph_json_str, device),)
                )
                result_id_out, updates, error = async_result.get(timeout=args.timeout)

                if error:
                    failed += 1
                    logger.warning(
                        "  [%d/%d] %s FAILED: %s",
                        i + 1,
                        len(rows),
                        result_id[:12],
                        error,
                    )
                    continue

                is_degen = updates.get(
                    "novelty_validity_reason", ""
                ) and "degenerate" in updates.get("novelty_validity_reason", "")
                if is_degen:
                    degenerate += 1

                store_probe_results(
                    nb,
                    result_id_out,
                    updates,
                    write_leaderboard=False,
                    provenance_context=provenance_context,
                )
                updated += 1

                if (i + 1) % 10 == 0:
                    conn.commit()

            except mp.TimeoutError:
                failed += 1
                logger.warning(
                    "  [%d/%d] %s TIMEOUT (%ds)",
                    i + 1,
                    len(rows),
                    result_id[:12],
                    args.timeout,
                )
            except Exception as e:
                crashed += 1
                logger.warning(
                    "  [%d/%d] %s CRASHED: %s",
                    i + 1,
                    len(rows),
                    result_id[:12],
                    e,
                )
            finally:
                pool.terminate()
                pool.join()

            if (i + 1) % 50 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    "  Progress: %d/%d (ok=%d, fail=%d, crash=%d, degen=%d) "
                    "%.1f/s, ETA %.0fs",
                    i + 1,
                    len(rows),
                    updated,
                    failed,
                    crashed,
                    degenerate,
                    rate,
                    (len(rows) - i - 1) / max(rate, 0.01),
                )
    except KeyboardInterrupt:
        fail_script_experiment(
            nb,
            exp_id,
            error="KeyboardInterrupt",
            results={
                "total": len(rows),
                "updated": updated,
                "failed": failed,
                "crashed": crashed,
                "degenerate": degenerate,
            },
        )
        nb.close()
        raise
    except Exception as exc:
        fail_script_experiment(
            nb,
            exp_id,
            error=str(exc),
            results={
                "total": len(rows),
                "updated": updated,
                "failed": failed,
                "crashed": crashed,
                "degenerate": degenerate,
            },
        )
        nb.close()
        raise

    conn.commit()

    elapsed = time.time() - t0
    logger.info(
        "Done: %d updated, %d failed, %d crashed, %d degenerate in %.1fs",
        updated,
        failed,
        crashed,
        degenerate,
        elapsed,
    )
    complete_script_experiment(
        nb,
        exp_id,
        results={
            "total": len(rows),
            "updated": updated,
            "failed": failed,
            "crashed": crashed,
            "degenerate": degenerate,
            "elapsed_s": round(elapsed, 3),
        },
        summary=(
            f"CKA/novelty backfill: updated={updated} failed={failed} "
            f"crashed={crashed} degenerate={degenerate}"
        ),
    )
    nb.close()


if __name__ == "__main__":
    main()
