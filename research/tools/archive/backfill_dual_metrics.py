"""Backfill dual-metric fields (discovery_loss, generalization_gap) for legacy rows.

Usage:
    # Phase 1: Quick SQL backfill (generalization_gap from existing data)
    python -m research.tools.backfill_dual_metrics --phase gap

    # Phase 2: Discovery loss via random-token eval (heavier, batch mode)
    python -m research.tools.backfill_dual_metrics --phase discovery --batch-size 50 --device cpu

    # Phase 3: Re-train with train/val split for proper gap (heaviest)
    python -m research.tools.backfill_dual_metrics --phase retrain --batch-size 20 --device cpu

    # Dry run (no writes):
    python -m research.tools.backfill_dual_metrics --phase gap --dry-run

    # Status report only:
    python -m research.tools.backfill_dual_metrics --status
"""
import argparse
import json
import os
import sys
import time
import dataclasses

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.synthesis.serializer import graph_from_json
from research.scientist.native_runner import compile_model_native_first as compile_model


DB_PATH = os.environ.get(
    "LAB_NOTEBOOK_DB",
    os.path.join(os.path.dirname(__file__), "..", "lab_notebook.db"),
)


def print_status(db: LabNotebook):
    """Print current dual-metric coverage stats."""
    c = db.conn
    base = "FROM program_results WHERE stage1_passed = 1 AND result_id NOT LIKE 'ref_%'"
    total_s1 = c.execute(f"SELECT COUNT(*) {base}").fetchone()[0]
    null_disc = c.execute(
        f"SELECT COUNT(*) {base} AND discovery_loss IS NULL"
    ).fetchone()[0]
    null_gap = c.execute(
        f"SELECT COUNT(*) {base} AND generalization_gap IS NULL"
    ).fetchone()[0]
    has_all = c.execute(
        f"SELECT COUNT(*) {base}"
        " AND discovery_loss IS NOT NULL AND generalization_gap IS NOT NULL"
    ).fetchone()[0]
    meaningful_gap = c.execute(
        f"SELECT COUNT(*) {base}"
        " AND generalization_gap IS NOT NULL AND ABS(generalization_gap) > 0.01"
    ).fetchone()[0]

    print("=== Dual-Metric Backfill Status ===")
    print(f"Stage 1 survivors:        {total_s1}")
    print(f"Missing discovery_loss:   {null_disc} ({null_disc*100//max(total_s1,1)}%)")
    print(f"Missing generalization_gap: {null_gap} ({null_gap*100//max(total_s1,1)}%)")
    print(f"Has all dual fields:      {has_all} ({has_all*100//max(total_s1,1)}%)")
    print(f"Meaningful gap (|gap|>0.01): {meaningful_gap}")


def backfill_gap(db: LabNotebook, dry_run: bool = False):
    """Phase 1: Compute generalization_gap = validation_loss - final_loss for rows that have both."""
    c = db.conn
    # Only backfill where val != final (meaningful gap)
    rows = c.execute(
        "SELECT result_id, final_loss, validation_loss FROM program_results"
        " WHERE stage1_passed = 1 AND generalization_gap IS NULL"
        " AND result_id NOT LIKE 'ref_%'"
        " AND validation_loss IS NOT NULL AND final_loss IS NOT NULL"
    ).fetchall()

    if not rows:
        print("No rows need generalization_gap backfill.")
        return

    meaningful = 0
    trivial = 0
    for result_id, final_loss, val_loss in rows:
        gap = val_loss - final_loss
        if abs(gap) > 0.01:
            meaningful += 1
        else:
            trivial += 1

    print(f"Found {len(rows)} rows to backfill generalization_gap:")
    print(f"  Meaningful (|gap| > 0.01): {meaningful}")
    print(f"  Trivial (val == final):    {trivial}")

    if dry_run:
        print("[DRY RUN] No changes written.")
        return

    updated = 0
    for result_id, final_loss, val_loss in rows:
        gap = val_loss - final_loss
        c.execute(
            "UPDATE program_results SET generalization_gap = ? WHERE result_id = ?",
            (gap, result_id),
        )
        updated += 1

    db.conn.commit()
    print(f"Updated {updated} rows with generalization_gap.")


def backfill_discovery(
    db: LabNotebook,
    batch_size: int = 50,
    device: str = "cpu",
    dry_run: bool = False,
):
    """Phase 2: Compute discovery_loss on random tokens for legacy rows."""
    c = db.conn
    rows = c.execute(
        "SELECT p.result_id, p.graph_json, p.initial_loss, e.config_json"
        " FROM program_results p"
        " JOIN experiments e ON p.experiment_id = e.experiment_id"
        " WHERE p.stage1_passed = 1 AND p.discovery_loss IS NULL"
        " AND p.result_id NOT LIKE 'ref_%'"
        " AND (p.error_type IS NULL OR p.error_type != 'discovery_backfill_shape_mismatch')"
        " AND p.graph_json IS NOT NULL"
        f" LIMIT {batch_size}"
    ).fetchall()

    if not rows:
        print("No rows need discovery_loss backfill.")
        return

    print(f"Processing {len(rows)} rows for discovery_loss backfill...")
    if dry_run:
        print(f"[DRY RUN] Would process {len(rows)} models.")
        return

    dev = torch.device(device)
    success = 0
    failed = 0
    discovery_batches = 2
    discovery_batch_size = 4

    for result_id, graph_json, initial_loss, config_json in rows:
        try:
            config_dict = json.loads(config_json) if config_json else {}
            valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
            filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
            config = RunConfig(**filtered)

            graph = graph_from_json(graph_json)
            # Ensure model_dim matches graph to avoid shape mismatches
            graph_dim = getattr(graph, "model_dim", None)
            if graph_dim and getattr(config, "model_dim", None) != graph_dim:
                config.model_dim = int(graph_dim)
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            model = model.to(dev)
            model.eval()

            # Run random-token forward passes to compute discovery loss
            seq_len = min(128, config.max_seq_len)
            losses = []
            with torch.no_grad():
                for _ in range(discovery_batches):
                    ids = torch.randint(0, config.vocab_size, (discovery_batch_size, seq_len), device=dev)
                    out = model(ids)
                    if hasattr(out, "logits"):
                        logits = out.logits
                    elif isinstance(out, tuple):
                        logits = out[0]
                    else:
                        logits = out
                    # Cross-entropy loss
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    if torch.isfinite(loss):
                        losses.append(loss.item())

            if not losses:
                print(f"  {result_id}: no valid losses from random eval")
                failed += 1
                continue

            disc_loss = sum(losses) / len(losses)
            disc_ratio = disc_loss / max(initial_loss or disc_loss, 1e-6)

            c.execute(
                "UPDATE program_results SET discovery_loss = ?, discovery_loss_ratio = ?"
                " WHERE result_id = ?",
                (disc_loss, disc_ratio, result_id),
            )
            success += 1
            if success % 10 == 0:
                db.conn.commit()
                print(f"  Progress: {success}/{len(rows)} done")

            # Clean up GPU memory
            del model
            if device != "cpu":
                torch.cuda.empty_cache()

        except Exception as e:
            msg = str(e)
            print(f"  {result_id}: FAILED - {msg}")
            # Skip shape-mismatch rows so we don't keep retrying them.
            if "mat1 and mat2 shapes cannot be multiplied" in msg:
                try:
                    c.execute(
                        "UPDATE program_results SET error_type = ? WHERE result_id = ?",
                        ("discovery_backfill_shape_mismatch", result_id),
                    )
                except Exception:
                    pass
            failed += 1

    db.conn.commit()
    print(f"Discovery backfill complete: {success} updated, {failed} failed.")


def backfill_retrain(
    db: LabNotebook,
    batch_size: int = 20,
    device: str = "cpu",
    dry_run: bool = False,
):
    """Phase 3: Re-train with train/val split for proper generalization_gap.

    Only targets rows where gap is trivial (val == final from old rescore).
    """
    c = db.conn
    rows = c.execute(
        "SELECT p.result_id, p.graph_json, e.config_json"
        " FROM program_results p"
        " JOIN experiments e ON p.experiment_id = e.experiment_id"
        " WHERE p.stage1_passed = 1"
        "   AND p.result_id NOT LIKE 'ref_%'"
        "   AND p.generalization_gap IS NOT NULL"
        "   AND ABS(p.generalization_gap) <= 0.01"
        "   AND p.graph_json IS NOT NULL"
        f" LIMIT {batch_size}"
    ).fetchall()

    if not rows:
        print("No rows need retrain-based gap backfill.")
        return

    print(f"Found {len(rows)} rows needing retrain for proper generalization_gap.")
    if dry_run:
        print(f"[DRY RUN] Would retrain {len(rows)} models.")
        return

    runner = ExperimentRunner(DB_PATH)
    dev = torch.device(device)
    success = 0
    failed = 0

    for result_id, graph_json, config_json in rows:
        try:
            config_dict = json.loads(config_json) if config_json else {}
            valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
            filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
            config = RunConfig(**filtered)

            # Force corpus mode with val split enabled
            config.data_mode = "corpus"
            config.corpus_path = os.path.join(
                os.path.dirname(__file__), "..", "micro_corpus.txt"
            )
            config.stage1_compute_val_loss = True
            config.stage1_compute_discovery_loss = True

            graph = graph_from_json(graph_json)
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )

            s1_result = runner._micro_train(model, config, dev, seed=42)
            if not s1_result.get("passed", False):
                print(f"  {result_id}: retrain failed - {s1_result.get('error')}")
                failed += 1
                continue

            updates = {}
            if s1_result.get("validation_loss") is not None:
                updates["validation_loss"] = s1_result["validation_loss"]
            if s1_result.get("validation_loss_ratio") is not None:
                updates["validation_loss_ratio"] = s1_result["validation_loss_ratio"]
            if s1_result.get("generalization_gap") is not None:
                updates["generalization_gap"] = s1_result["generalization_gap"]
            if s1_result.get("discovery_loss") is not None:
                updates["discovery_loss"] = s1_result["discovery_loss"]
            if s1_result.get("discovery_loss_ratio") is not None:
                updates["discovery_loss_ratio"] = s1_result["discovery_loss_ratio"]
            # Also update final_loss and loss_ratio with fresh corpus results
            if s1_result.get("final_loss") is not None:
                updates["final_loss"] = s1_result["final_loss"]
            if s1_result.get("loss_ratio") is not None:
                updates["loss_ratio"] = s1_result["loss_ratio"]

            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                c.execute(
                    f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
                    list(updates.values()) + [result_id],
                )
                success += 1

            if success % 5 == 0:
                db.conn.commit()
                print(f"  Progress: {success}/{len(rows)} done")

            del model
            if device != "cpu":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  {result_id}: FAILED - {e}")
            failed += 1

    db.conn.commit()
    print(f"Retrain backfill complete: {success} updated, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description="Backfill dual-metric fields for legacy rows")
    parser.add_argument("--phase", choices=["gap", "discovery", "retrain"], help="Backfill phase")
    parser.add_argument("--status", action="store_true", help="Print status report only")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Max rows per batch")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--db", default=DB_PATH, help="Path to lab_notebook.db")
    args = parser.parse_args()

    db = LabNotebook(args.db)

    if args.status or not args.phase:
        print_status(db)
        return

    t0 = time.time()
    if args.phase == "gap":
        backfill_gap(db, dry_run=args.dry_run)
    elif args.phase == "discovery":
        backfill_discovery(db, batch_size=args.batch_size, device=args.device, dry_run=args.dry_run)
    elif args.phase == "retrain":
        backfill_retrain(db, batch_size=args.batch_size, device=args.device, dry_run=args.dry_run)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print()
    print_status(db)


if __name__ == "__main__":
    main()
