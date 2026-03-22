"""Backfill robustness/efficiency metrics for leaderboard entries missing them.

Rebuilds each model from stored graph_json, runs:
  - Fingerprint + novelty score
  - FLOPs/param efficiency
  - Noise sensitivity
  - Quantization INT8 retention
  - Init sensitivity (multi-seed)
  - Baseline comparison

Usage:
    python -m research.tools.backfill_metrics [--limit 50] [--tier screening] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _ensure_mathspaces():
    """Register mathspace ops so compile_model can handle them."""
    from ..mathspaces.registry import register_all_mathspaces

    register_all_mathspaces()


_ensure_mathspaces()


def backfill_entry(row, device="cpu", n_train_steps=50):
    """Backfill metrics for a single leaderboard entry."""
    from ..synthesis.graph import ComputationGraph
    from ..synthesis.compiler import compile_model
    from ..eval.fingerprint import compute_fingerprint
    from ..eval.metrics import novelty_score
    from ..eval.flops import estimate_flops
    from ..eval.noise_sensitivity import evaluate_noise_sensitivity
    from ..eval.quantization import evaluate_sparse_quant_quality
    from ..eval.baseline import TransformerBaseline
    from ..scientist.notebook import LabNotebook

    result_id = row["result_id"]
    graph_json = row["graph_json"]
    entry_id = row["entry_id"]

    if not graph_json:
        return {"result_id": result_id, "status": "no_graph_json"}

    try:
        graph_data = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return {"result_id": result_id, "status": "invalid_json"}

    try:
        graph = ComputationGraph.from_dict(graph_data)
    except Exception as e:
        return {"result_id": result_id, "status": "graph_parse_failed", "error": str(e)}

    d_model = graph.model_dim or 256
    vocab_size = 32000
    seq_len = 128
    dev = torch.device(device)

    # Build model from graph; try 2-layer stack first, then fall back to 1 layer
    model = None
    compile_err = None
    for layers in ([graph, graph], [graph]):
        try:
            model = compile_model(
                layers, vocab_size=vocab_size, max_seq_len=seq_len
            ).to(dev)
            compile_err = None
            break
        except Exception as e:
            compile_err = e
            model = None
    if model is None:
        return {
            "result_id": result_id,
            "status": "compile_failed",
            "error": str(compile_err),
        }

    # Quick forward sanity check (non-fatal for partial backfill)
    model_usable = True
    try:
        with torch.no_grad():
            test_ids = torch.randint(0, vocab_size, (1, 32), device=dev)
            test_out = model(test_ids)
            if torch.isnan(test_out).any() or torch.isinf(test_out).any():
                model_usable = False
    except Exception:
        model_usable = False

    updates = {}

    # Novelty (prefer structural+behavioral; fall back to structural-only)
    try:
        if model_usable:
            bfp = compute_fingerprint(
                model,
                seq_len=min(seq_len, 64),
                model_dim=d_model,
                vocab_size=vocab_size,
                device=str(dev),
            )
            nm = novelty_score(graph, fingerprint=bfp)
            # Z13: Capture spectral norm for stability scoring
            updates["fp_jacobian_spectral_norm"] = float(bfp.jacobian_spectral_norm)
        else:
            nm = novelty_score(graph)
        updates["screening_novelty"] = float(nm.overall_novelty)
    except Exception:
        pass

    # FLOPs
    try:
        flop_est = estimate_flops(graph, seq_len=seq_len, d_model=d_model)
        updates["param_efficiency"] = float(flop_est.flops_per_param)
    except Exception:
        pass

    # Noise sensitivity
    if model_usable:
        try:
            input_batches = [
                torch.randint(0, vocab_size, (2, seq_len), device=dev) for _ in range(3)
            ]
            noise_result = evaluate_noise_sensitivity(
                model, input_batches, dev, vocab_size=vocab_size
            )
            ns = noise_result.get("noise_sensitivity_score")
            if ns is not None:
                updates["robustness_noise_score"] = float(ns)
        except Exception:
            pass

    # Quantization
    if model_usable:
        try:
            quant_batches = [
                torch.randint(0, vocab_size, (2, seq_len), device=dev) for _ in range(3)
            ]
            quant_result = evaluate_sparse_quant_quality(model, quant_batches, dev)
            if quant_result:
                ret = quant_result.get("full_retention")
                qpb = quant_result.get("quality_per_byte")
                if ret is not None:
                    updates["quant_int8_retention"] = float(ret)
                if qpb is not None:
                    updates["quant_quality_per_byte"] = float(qpb)
        except Exception:
            pass

    # Init sensitivity (2 seeds for speed)
    if model_usable:
        try:
            seed_ratios = []
            for seed in [42, 789]:
                torch.manual_seed(seed)
                seed_model = compile_model(
                    [graph], vocab_size=vocab_size, max_seq_len=seq_len
                ).to(dev)
                optimizer = torch.optim.AdamW(
                    seed_model.parameters(), lr=3e-4, weight_decay=0.01
                )
                losses = []
                for step in range(n_train_steps):
                    ids = torch.randint(0, vocab_size, (2, seq_len), device=dev)
                    logits = seed_model(ids)[:, :-1].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        ids[:, 1:].contiguous().view(-1),
                    )
                    if torch.isnan(loss) or torch.isinf(loss):
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(seed_model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
                if len(losses) >= 2:
                    seed_ratios.append(losses[-1] / losses[0])
                del seed_model
            if len(seed_ratios) == 2:
                updates["init_sensitivity_std"] = float(
                    abs(seed_ratios[0] - seed_ratios[1])
                )
        except Exception:
            pass

    # Baseline comparison
    try:
        loss_ratio = row.get("screening_loss_ratio")
        if loss_ratio is not None:
            baseline = TransformerBaseline()
            baseline_loss = baseline.get_baseline_loss(
                d_model=d_model,
                seq_len=seq_len,
                n_steps=n_train_steps,
                vocab_size=vocab_size,
                device=str(dev),
            )
            if baseline_loss and baseline_loss > 0:
                updates["normalized_baseline_ratio"] = float(loss_ratio) / float(
                    baseline_loss
                )
    except Exception:
        pass

    # Apply updates
    if updates:
        nb = LabNotebook()
        try:
            set_parts = []
            vals = []
            for k, v in updates.items():
                set_parts.append(f"{k} = ?")
                vals.append(v)
            vals.append(entry_id)
            nb.conn.execute(
                f"UPDATE leaderboard SET {', '.join(set_parts)} WHERE entry_id = ?",
                vals,
            )
            nb.conn.commit()
        except Exception as e:
            return {"result_id": result_id, "status": "update_failed", "error": str(e)}

    del model
    status = "backfilled" if model_usable else "backfilled_partial"
    return {
        "result_id": result_id,
        "status": status,
        "metrics_added": list(updates.keys()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill robustness metrics for leaderboard entries"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Max entries to backfill per run"
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=["screening", "investigation", "validation", "all"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-loss",
        type=float,
        default=0.5,
        help="Only backfill entries with loss_ratio below this",
    )
    args = parser.parse_args()

    from ..scientist.notebook import LabNotebook

    nb = LabNotebook()

    tier_filter = "" if args.tier == "all" else f"AND lb.tier = '{args.tier}'"
    query = f"""
        SELECT lb.entry_id, lb.result_id, lb.tier, lb.screening_loss_ratio,
               pr.graph_json
        FROM leaderboard lb
        JOIN program_results pr ON lb.result_id = pr.result_id
        WHERE lb.is_reference = 0
        AND lb.screening_loss_ratio IS NOT NULL
        AND lb.screening_loss_ratio < ?
        AND (lb.quant_int8_retention IS NULL
             OR lb.robustness_noise_score IS NULL
             OR lb.init_sensitivity_std IS NULL
             OR lb.discovery_loss_ratio IS NULL
             OR lb.fp_jacobian_spectral_norm IS NULL)
        AND pr.graph_json IS NOT NULL
        {tier_filter}
        ORDER BY lb.screening_loss_ratio ASC
        LIMIT ?
    """
    rows = nb.conn.execute(query, (args.min_loss, args.limit)).fetchall()
    col_names = [
        desc[0] for desc in nb.conn.execute(query, (args.min_loss, 1)).description
    ]

    log.info(
        "Found %d entries to backfill (limit=%d, tier=%s, max_loss=%.2f)",
        len(rows),
        args.limit,
        args.tier,
        args.min_loss,
    )

    success = 0
    failed = 0
    for i, row_tuple in enumerate(rows):
        row = dict(zip(col_names, row_tuple))
        log.info(
            "[%d/%d] Backfilling %s (tier=%s, loss=%.6f)...",
            i + 1,
            len(rows),
            row["result_id"][:12],
            row["tier"],
            row.get("screening_loss_ratio") or 0,
        )
        try:
            result = backfill_entry(row, device=args.device)
            if result["status"] in ("backfilled", "backfilled_partial"):
                success += 1
                log.info("  OK: added %s", result.get("metrics_added", []))
            else:
                failed += 1
                log.warning(
                    "  SKIP: %s — %s", result["status"], result.get("error", "")
                )
        except Exception:
            failed += 1
            log.error("  ERROR: %s", traceback.format_exc().split("\n")[-2])

    log.info(
        "Done: %d backfilled, %d failed/skipped out of %d total",
        success,
        failed,
        len(rows),
    )
    log.info(
        "Remaining to backfill: %d",
        nb.conn.execute(
            "SELECT count(*) FROM leaderboard WHERE is_reference=0 AND screening_loss_ratio < ? AND quant_int8_retention IS NULL",
            (args.min_loss,),
        ).fetchone()[0],
    )


if __name__ == "__main__":
    main()
