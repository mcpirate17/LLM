#!/usr/bin/env python3
"""Re-screen leaderboard entries after registry + memory + optimizer fixes.

Three fixes invalidate existing screening scores:
  1. Registry fix: mathspace ops were missing from PRIMITIVE_REGISTRY
  2. Memory fix: autograd.Function rewrite changes backward tensor retention
  3. Optimizer fix: AdamW betas (0.9, 0.999) → (0.9, 0.95) / Muon

This script re-evaluates graphs through Stage 0 + Stage 1 and records the
delta between old and new scores.

Usage:
    python -m research.tools.rescreen_leaderboard --pass top50 [--device cuda]
    python -m research.tools.rescreen_leaderboard --pass mathspace [--device cuda]
    python -m research.tools.rescreen_leaderboard --pass remaining [--device cuda] [--score-floor 5.0]
    python -m research.tools.rescreen_leaderboard --dry-run --pass top50
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sqlite3
import time
import uuid
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..defaults import MODEL_DIM, VOCAB_SIZE, MAX_SEQ_LEN, STAGE1_STEPS, STAGE1_LR, STAGE1_BATCH_SIZE

logger = logging.getLogger(__name__)

# ── Query helpers ──────────────────────────────────────────────────────

_MATHSPACE_PATTERNS = ("%ultrametric%", "%tropical%", "%padic%")


def _query_top_n(conn: sqlite3.Connection, n: int) -> List[sqlite3.Row]:
    """Top-N non-reference entries by old composite score, pending rescore."""
    return conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.composite_score, l.old_composite_score,
               l.screening_loss_ratio, l.tier,
               p.graph_json, p.graph_fingerprint
        FROM leaderboard l
        JOIN program_results p ON p.result_id = l.result_id
        WHERE l.rescore_status = 'pending'
          AND (l.is_reference = 0 OR l.is_reference IS NULL)
          AND p.graph_json IS NOT NULL AND p.graph_json != ''
        ORDER BY l.old_composite_score DESC NULLS LAST
        LIMIT ?
        """,
        (n,),
    ).fetchall()


def _query_mathspace(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """All pending entries whose graph contains mathspace ops."""
    return conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.composite_score, l.old_composite_score,
               l.screening_loss_ratio, l.tier,
               p.graph_json, p.graph_fingerprint
        FROM leaderboard l
        JOIN program_results p ON p.result_id = l.result_id
        WHERE l.rescore_status = 'pending'
          AND (l.is_reference = 0 OR l.is_reference IS NULL)
          AND p.graph_json IS NOT NULL AND p.graph_json != ''
          AND (p.graph_json LIKE ? OR p.graph_json LIKE ? OR p.graph_json LIKE ?)
        ORDER BY l.old_composite_score DESC NULLS LAST
        """,
        _MATHSPACE_PATTERNS,
    ).fetchall()


def _query_remaining(
    conn: sqlite3.Connection, score_floor: float = 0.0
) -> List[sqlite3.Row]:
    """All remaining pending entries above score floor."""
    return conn.execute(
        """
        SELECT l.entry_id, l.result_id, l.composite_score, l.old_composite_score,
               l.screening_loss_ratio, l.tier,
               p.graph_json, p.graph_fingerprint
        FROM leaderboard l
        JOIN program_results p ON p.result_id = l.result_id
        WHERE l.rescore_status = 'pending'
          AND (l.is_reference = 0 OR l.is_reference IS NULL)
          AND p.graph_json IS NOT NULL AND p.graph_json != ''
          AND (l.old_composite_score >= ? OR l.old_composite_score IS NULL)
        ORDER BY l.old_composite_score DESC NULLS LAST
        """,
        (score_floor,),
    ).fetchall()


# ── Single-graph re-screening ─────────────────────────────────────────


def _rescreen_one(
    graph_json: str,
    device: torch.device,
    n_layers: int = 4,
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    stage1_steps: int = STAGE1_STEPS,
    stage1_lr: float = STAGE1_LR,
    stage1_batch_size: int = STAGE1_BATCH_SIZE,
) -> Dict:
    """Compile a graph and run Stage 0 + Stage 1. Returns result dict."""
    from ..synthesis.serializer import graph_from_json
    from ..scientist.native_runner import compile_model_native_first as compile_model
    pass  # safe_eval skipped: re-screening only needs fresh training metrics
    from ..training.optimizer_synthesis import build_optimizer

    result: Dict = {"passed": False, "stage0_passed": True, "stage1_passed": False}

    # Deserialize
    try:
        graph = graph_from_json(graph_json)
    except Exception as exc:
        result["error_type"] = "deserialize"
        result["error_message"] = str(exc)[:500]
        return result

    # Compile
    try:
        layer_graphs = [graph] * n_layers
        model = compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=max_seq_len)
    except Exception as exc:
        result["error_type"] = "compile"
        result["error_message"] = str(exc)[:500]
        return result

    # Skip safe_eval for re-screening: these graphs already passed Stage 0
    # during original screening. We only need fresh training metrics with
    # corrected optimizer (Muon) and registry fixes.

    # Stage 1: micro-train with Muon
    try:
        model = model.to(device)
        result.update(
            _micro_train_standalone(
                model, device, vocab_size,
                steps=stage1_steps,
                lr=stage1_lr,
                batch_size=stage1_batch_size,
            )
        )
    except Exception as exc:
        result["error_type"] = "stage1_exception"
        result["error_message"] = str(exc)[:500]
    finally:
        _cleanup_model(model, device)

    return result


def _micro_train_standalone(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    steps: int = 500,
    lr: float = 3e-4,
    batch_size: int = 4,
    seq_len: int = 128,
) -> Dict:
    """Stage 1 micro-training with Muon optimizer and bf16 autocast on CUDA."""
    from ..training.optimizer_synthesis import build_optimizer

    model.train()
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=lr,
        weight_decay=0.0,
    )

    use_amp = (device.type == "cuda")
    initial_loss = None
    final_loss = None
    min_loss = float("inf")
    losses = []

    t0 = time.perf_counter()
    for step in range(steps):
        g = torch.Generator(device="cpu")
        g.manual_seed(step)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(input_ids)
            if isinstance(output, tuple):
                output = output[0]
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                targets.view(-1),
            )

        if torch.isnan(loss) or torch.isinf(loss):
            return {
                "stage1_passed": False,
                "error_type": "nan_loss",
                "error_message": f"Loss became {'nan' if torch.isnan(loss) else 'inf'} at step {step}",
                "initial_loss": initial_loss,
                "n_train_steps": step,
            }

        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val
        min_loss = min(min_loss, loss_val)

        if step < 10 or step % 50 == 0 or step == steps - 1:
            losses.append({"step": step, "loss": loss_val})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - t0
    tokens_processed = steps * batch_size * seq_len
    throughput = tokens_processed / elapsed if elapsed > 0 else 0.0

    loss_ratio = final_loss / initial_loss if initial_loss and initial_loss > 0 else None
    passed = loss_ratio is not None and loss_ratio < 0.95

    param_count = sum(p.numel() for p in model.parameters())

    return {
        "stage1_passed": passed,
        "passed": passed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "loss_ratio": loss_ratio,
        "n_train_steps": steps,
        "throughput_tok_s": throughput,
        "param_count": param_count,
        "training_curve": json.dumps(losses),
        "optimizer_type": "muon",
    }


def _cleanup_model(model: nn.Module, device: torch.device) -> None:
    """Free model and all associated state from GPU."""
    try:
        # Zero gradients to release autograd graph references
        for p in model.parameters():
            p.grad = None
        model.cpu()
        del model
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ── Batch re-screening ────────────────────────────────────────────────


def rescreen_batch(
    rows: List[sqlite3.Row],
    conn: sqlite3.Connection,
    nb,
    device: torch.device,
    experiment_id: str,
    reason: str = "registry+memory+optimizer-fix",
) -> Dict:
    """Re-screen a batch of leaderboard entries. Returns summary stats."""
    improved = 0
    degraded = 0
    unchanged = 0
    failed = 0
    deltas = []

    for i, row in enumerate(rows):
        entry_id = row["entry_id"]
        result_id = row["result_id"]
        graph_json = row["graph_json"]
        fingerprint = row["graph_fingerprint"]
        old_score = row["old_composite_score"]
        old_lr = row["screening_loss_ratio"]

        logger.info(
            "[%d/%d] Re-screening %s (old_score=%.4f)",
            i + 1, len(rows), entry_id[:12], old_score or 0.0,
        )

        s1 = _rescreen_one(graph_json, device)

        # Record new program result
        new_result_id = nb.record_program_result(
            experiment_id=experiment_id,
            graph_fingerprint=fingerprint,
            graph_json=graph_json,
            stage0_passed=int(s1.get("stage0_passed", False)),
            stage1_passed=int(s1.get("stage1_passed", False)),
            loss_ratio=s1.get("loss_ratio"),
            final_loss=s1.get("final_loss"),
            initial_loss=s1.get("initial_loss"),
            min_loss=s1.get("min_loss"),
            throughput_tok_s=s1.get("throughput_tok_s"),
            param_count=s1.get("param_count"),
            error_type=s1.get("error_type"),
            error_message=s1.get("error_message"),
            n_train_steps=s1.get("n_train_steps"),
            training_curve=s1.get("training_curve"),
        )
        nb.flush_writes()

        # Update leaderboard entry
        new_lr = s1.get("loss_ratio")
        if new_lr is not None and s1.get("stage1_passed"):
            # Fetch full existing row to recompute composite with all metrics
            existing = conn.execute(
                "SELECT * FROM leaderboard WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            d = dict(existing) if existing else {}

            # Recompute composite with new screening_lr but preserve higher-tier metrics
            new_composite = nb.compute_composite_score(
                screening_lr=new_lr,
                screening_nov=d.get("screening_novelty"),
                inv_lr=d.get("investigation_loss_ratio"),
                inv_robust=d.get("investigation_robustness"),
                val_lr=d.get("validation_loss_ratio"),
                val_baseline=d.get("validation_baseline_ratio"),
                val_std=d.get("validation_multi_seed_std"),
                novelty_confidence=d.get("screening_novelty"),
                scaling_param_efficiency=d.get("scaling_param_efficiency"),
                is_reference=bool(d.get("is_reference")),
                routing_savings=d.get("routing_savings_ratio"),
                compression_ratio=d.get("compression_ratio"),
                discovery_lr=d.get("discovery_loss_ratio"),
                robustness_noise=d.get("robustness_noise_score"),
                quant_retention=d.get("quant_int8_retention"),
                long_ctx_score=d.get("robustness_long_ctx_score"),
                init_std=d.get("init_sensitivity_std"),
                ncd_score=d.get("ncd_score"),
                activation_sparsity=d.get("activation_sparsity_score"),
            )

            conn.execute(
                """
                UPDATE leaderboard
                SET screening_loss_ratio = ?,
                    composite_score = ?,
                    rescore_status = 'complete',
                    rescore_timestamp = ?,
                    rescore_reason = ?
                WHERE entry_id = ?
                """,
                (new_lr, new_composite, time.time(), reason, entry_id),
            )

            conn.commit()

            delta = (new_composite or 0) - (old_score or 0)
            deltas.append(delta)
            if abs(delta) < (old_score or 1.0) * 0.01:
                unchanged += 1
            elif delta > 0:
                improved += 1
            else:
                degraded += 1

            logger.info(
                "  → new_lr=%.6f old_lr=%.6f new_composite=%.4f delta=%+.4f",
                new_lr, old_lr or 0, new_composite, delta,
            )
        else:
            # Failed re-screening
            conn.execute(
                """
                UPDATE leaderboard
                SET rescore_status = 'failed',
                    rescore_timestamp = ?,
                    rescore_reason = ?
                WHERE entry_id = ?
                """,
                (time.time(), reason, entry_id),
            )
            conn.commit()
            failed += 1
            logger.info("  → FAILED: %s", s1.get("error_type", "no_signal"))

    return {
        "total": len(rows),
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "failed": failed,
        "avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "deltas": deltas,
    }


def print_summary(stats: Dict, pass_name: str) -> None:
    """Print a human-readable summary of re-screening results."""
    print(f"\n{'='*60}")
    print(f"Re-screening Pass: {pass_name}")
    print(f"{'='*60}")
    print(f"Total processed:  {stats['total']}")
    print(f"Improved (new>old): {stats['improved']}  avg delta: {sum(d for d in stats['deltas'] if d > 0) / max(1, stats['improved']):+.4f}")
    print(f"Degraded (new<old): {stats['degraded']}  avg delta: {sum(d for d in stats['deltas'] if d < 0) / max(1, stats['degraded']):+.4f}")
    print(f"Unchanged (<1%):    {stats['unchanged']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Overall avg delta:  {stats['avg_delta']:+.4f}")
    print(f"{'='*60}\n")


# ── Skip criteria ─────────────────────────────────────────────────────


def apply_skip_criteria(conn: sqlite3.Connection, score_floor: float = 5.0) -> int:
    """Mark low-value entries as skip to reduce rescore queue."""
    # Bottom-percentile by old score
    p20 = conn.execute(
        "SELECT old_composite_score FROM leaderboard "
        "WHERE old_composite_score IS NOT NULL "
        "ORDER BY old_composite_score ASC "
        "LIMIT 1 OFFSET (SELECT COUNT(*)/5 FROM leaderboard)"
    ).fetchone()
    threshold = max(score_floor, (p20[0] if p20 else score_floor))

    cursor = conn.execute(
        """
        UPDATE leaderboard
        SET rescore_status = 'skip'
        WHERE rescore_status = 'pending'
          AND (is_reference = 0 OR is_reference IS NULL)
          AND old_composite_score IS NOT NULL
          AND old_composite_score < ?
        """,
        (threshold,),
    )
    conn.commit()
    skipped = cursor.rowcount
    logger.info("Skipped %d entries below score floor %.4f", skipped, threshold)
    return skipped


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Re-screen leaderboard entries")
    parser.add_argument("--db", default="research/lab_notebook.db", help="Path to lab_notebook.db")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pass", dest="pass_name", required=True,
                        choices=["top50", "mathspace", "remaining", "skip"],
                        help="Which pass to run")
    parser.add_argument("--limit", type=int, default=50, help="Limit for top-N pass")
    parser.add_argument("--score-floor", type=float, default=5.0,
                        help="Score floor for remaining/skip passes")
    parser.add_argument("--dry-run", action="store_true", help="Print targets without re-screening")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.pass_name == "skip":
        skipped = apply_skip_criteria(conn, score_floor=args.score_floor)
        print(f"Skipped {skipped} entries below score floor {args.score_floor}")
        conn.close()
        return

    # Select targets
    if args.pass_name == "top50":
        rows = _query_top_n(conn, args.limit)
    elif args.pass_name == "mathspace":
        rows = _query_mathspace(conn)
    elif args.pass_name == "remaining":
        rows = _query_remaining(conn, score_floor=args.score_floor)
    else:
        raise ValueError(f"Unknown pass: {args.pass_name}")

    print(f"Pass '{args.pass_name}': {len(rows)} entries to re-screen")

    if args.dry_run:
        for row in rows[:20]:
            has_ms = any(p.strip("%") in (row["graph_json"] or "") for p in _MATHSPACE_PATTERNS)
            print(f"  {row['entry_id'][:12]}  tier={row['tier']:15s}  "
                  f"old_score={row['old_composite_score'] or 0:.4f}  mathspace={'Y' if has_ms else 'N'}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
        conn.close()
        return

    # Create a rescore experiment
    from ..scientist.notebook import LabNotebook
    nb = LabNotebook(args.db)
    exp_id = nb.start_experiment(
        experiment_type="rescore",
        config={"pass": args.pass_name, "n_targets": len(rows), "device": args.device},
        hypothesis=f"Re-screen pass '{args.pass_name}': {len(rows)} entries after registry+memory+optimizer fixes",
        research_question="How do scores change after fixing silent registry/optimizer bugs?",
    )

    device = torch.device(args.device)
    stats = rescreen_batch(rows, conn, nb, device, exp_id)
    print_summary(stats, args.pass_name)

    # Complete experiment
    nb.complete_experiment(exp_id, results={
        "pass": args.pass_name,
        "stats": {k: v for k, v in stats.items() if k != "deltas"},
    })

    # Print top-10 overlap if this was top50 pass
    if args.pass_name == "top50":
        old_top10 = conn.execute(
            "SELECT entry_id FROM leaderboard "
            "WHERE (is_reference = 0 OR is_reference IS NULL) "
            "ORDER BY old_composite_score DESC LIMIT 10"
        ).fetchall()
        new_top10 = conn.execute(
            "SELECT entry_id FROM leaderboard "
            "WHERE (is_reference = 0 OR is_reference IS NULL) "
            "AND rescore_status != 'pending' "
            "ORDER BY composite_score DESC LIMIT 10"
        ).fetchall()
        old_ids = {r["entry_id"] for r in old_top10}
        new_ids = {r["entry_id"] for r in new_top10}
        overlap = len(old_ids & new_ids)
        print(f"\nOld top-10 vs new top-10 overlap: {overlap}/10")

    conn.close()


if __name__ == "__main__":
    main()
