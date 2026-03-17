"""Backfill missing leaderboard evaluation data from stored graph_json.

Three phases, each progressively heavier:
  Phase 1 (sql)     — SQL-only fixes: generalization_gap, copy loss ratios
  Phase 2 (compile) — Rebuild model, run compile-only evals (no training)
  Phase 3 (train)   — Rebuild model, micro-train + eval (wikitext, tinystories, cross_task)

Usage:
    # Status report
    python -m research.tools.backfill_leaderboard --status

    # Phase 1 (instant)
    python -m research.tools.backfill_leaderboard --phase sql --dry-run
    python -m research.tools.backfill_leaderboard --phase sql

    # Phase 2 (rebuild + forward-only evals)
    python -m research.tools.backfill_leaderboard --phase compile --limit 5 --device cpu
    python -m research.tools.backfill_leaderboard --phase compile --device cuda

    # Phase 3 (rebuild + micro-train evals)
    python -m research.tools.backfill_leaderboard --phase train --limit 10 --device cpu

    # All phases
    python -m research.tools.backfill_leaderboard --phase all --device cpu

    # Target specific entry
    python -m research.tools.backfill_leaderboard --phase compile --entry-id abc123
"""

import argparse
import dataclasses
import json
import os
import sys
import time

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model(graph_json, config):
    """Deserialize graph_json and compile a model."""
    graph = graph_from_json(graph_json)
    graph_dim = getattr(graph, "model_dim", None)
    if graph_dim and getattr(config, "model_dim", None) != graph_dim:
        config.model_dim = int(graph_dim)
    layer_graphs = [graph] * config.n_layers
    model = compile_model(
        layer_graphs,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
    )
    return model


def _parse_config(config_json):
    """Parse a config_json string into a RunConfig."""
    config_dict = json.loads(config_json) if config_json else {}
    valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return RunConfig(**filtered)


def _make_input_batches(vocab_size, device, n_batches=3, batch_size=4, seq_len=128):
    """Generate random token batches for compile-only evals."""
    return [
        torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        for _ in range(n_batches)
    ]


def _fmt_eta(elapsed, done, total):
    """Format remaining time estimate."""
    if done == 0:
        return "?"
    rate = elapsed / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining / 60:.1f}m"
    return f"{remaining / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def print_status(db):
    """Print coverage stats for leaderboard columns."""
    c = db.conn
    total = c.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0]
    if total == 0:
        print("Leaderboard is empty.")
        return

    print(f"=== Leaderboard Backfill Status ({total} entries) ===\n")

    cols = [
        # Phase 1
        ("generalization_gap", "program_results"),
        ("discovery_loss", "program_results"),
        ("discovery_loss_ratio", "program_results"),
        # Phase 2 (compile-only)
        ("efficiency_wall_score", "leaderboard"),
        ("max_viable_seq_len", "leaderboard"),
        ("scaling_regime", "leaderboard"),
        ("activation_sparsity_score", "leaderboard"),
        ("dead_neuron_ratio", "leaderboard"),
        ("routing_collapse_score", "leaderboard"),
        ("routing_savings_ratio", "leaderboard"),
        ("compression_ratio", "leaderboard"),
        # Phase 3 (train)
        ("wikitext_perplexity", "leaderboard"),
        ("wikitext_score", "leaderboard"),
        ("tinystories_perplexity", "leaderboard"),
        ("tinystories_score", "leaderboard"),
        ("cross_task_score", "leaderboard"),
        # Tier-related
        ("validation_loss_ratio", "leaderboard"),
        ("investigation_loss_ratio", "leaderboard"),
    ]

    for col, table in cols:
        try:
            if table == "leaderboard":
                null_count = c.execute(
                    f"SELECT COUNT(*) FROM leaderboard WHERE {col} IS NULL"
                ).fetchone()[0]
            else:
                null_count = c.execute(
                    f"SELECT COUNT(*) FROM leaderboard l JOIN program_results pr"
                    f" ON l.result_id = pr.result_id WHERE pr.{col} IS NULL"
                ).fetchone()[0]
            pct = null_count * 100 // total
            has = total - null_count
            print(f"  {col:40s}  {has:4d}/{total} populated  ({pct}% NULL)")
        except Exception:
            print(f"  {col:40s}  [column missing]")

    print()


# ---------------------------------------------------------------------------
# Phase 1: SQL-only fixes
# ---------------------------------------------------------------------------


def phase_sql(db, dry_run=False):
    """Instant SQL backfills that don't require model rebuild."""
    c = db.conn
    print("=== Phase 1: SQL-only fixes ===\n")

    # 1a. Backfill generalization_gap from existing final_loss + validation_loss
    rows = c.execute(
        "SELECT result_id, final_loss, validation_loss FROM program_results"
        " WHERE generalization_gap IS NULL"
        " AND result_id NOT LIKE 'ref_%'"
        " AND validation_loss IS NOT NULL AND final_loss IS NOT NULL"
    ).fetchall()
    print(f"  generalization_gap: {len(rows)} rows to backfill")
    if not dry_run and rows:
        for result_id, final_loss, val_loss in rows:
            gap = val_loss - final_loss
            c.execute(
                "UPDATE program_results SET generalization_gap = ? WHERE result_id = ?",
                (gap, result_id),
            )
        db.conn.commit()
        print(f"    -> Updated {len(rows)} rows")

    # 1b. Copy investigation_loss_ratio from program_results to leaderboard
    # where leaderboard is NULL but program_results has it
    rows = c.execute(
        "SELECT l.entry_id, pr.loss_ratio"
        " FROM leaderboard l"
        " JOIN program_results pr ON l.result_id = pr.result_id"
        " WHERE l.investigation_loss_ratio IS NULL"
        " AND pr.loss_ratio IS NOT NULL"
    ).fetchall()
    print(f"  investigation_loss_ratio copy: {len(rows)} rows to backfill")
    if not dry_run and rows:
        updated = 0
        for entry_id, loss_ratio in rows:
            c.execute(
                "UPDATE leaderboard SET investigation_loss_ratio = ? WHERE entry_id = ?",
                (loss_ratio, entry_id),
            )
            updated += 1
        db.conn.commit()
        print(f"    -> Updated {updated} rows")

    # 1c. Copy discovery_loss from program_results to ensure consistency
    rows = c.execute(
        "SELECT l.entry_id, pr.discovery_loss_ratio"
        " FROM leaderboard l"
        " JOIN program_results pr ON l.result_id = pr.result_id"
        " WHERE l.screening_loss_ratio IS NULL"
        " AND pr.discovery_loss_ratio IS NOT NULL"
    ).fetchall()
    print(f"  screening_loss_ratio copy: {len(rows)} rows to backfill")
    if not dry_run and rows:
        updated = 0
        for entry_id, ratio in rows:
            c.execute(
                "UPDATE leaderboard SET screening_loss_ratio = ? WHERE entry_id = ?",
                (ratio, entry_id),
            )
            updated += 1
        db.conn.commit()
        print(f"    -> Updated {updated} rows")

    if dry_run:
        print("\n  [DRY RUN] No changes written.")
    print()


# ---------------------------------------------------------------------------
# Phase 2: Compile-only evals (no training)
# ---------------------------------------------------------------------------


def phase_compile(db, device="cpu", limit=0, entry_id=None, dry_run=False):
    """Rebuild models and run forward-only evaluations."""
    from research.eval.efficiency_wall import evaluate_efficiency_wall
    from research.eval.sparsity import evaluate_activation_sparsity
    from research.eval.routing_heatmap import evaluate_routing_heatmap

    c = db.conn
    print("=== Phase 2: Compile-only evals ===\n")

    # Build query — find entries missing ANY compile-only metric
    where = (
        "l.result_id = pr.result_id"
        " AND pr.graph_json IS NOT NULL"
        " AND pr.result_id NOT LIKE 'ref_%'"
        " AND ("
        "   l.efficiency_wall_score IS NULL"
        "   OR l.activation_sparsity_score IS NULL"
        "   OR l.routing_collapse_score IS NULL"
        " )"
    )
    if entry_id:
        where += " AND l.entry_id = ?"
        params = [entry_id]
    else:
        params = []

    query = (
        "SELECT l.entry_id, l.tier, pr.result_id, pr.graph_json, e.config_json"
        " FROM leaderboard l"
        " JOIN program_results pr ON l.result_id = pr.result_id"
        " JOIN experiments e ON pr.experiment_id = e.experiment_id"
        f" WHERE {where}"
    )
    if limit > 0 and not entry_id:
        query += f" LIMIT {limit}"

    rows = c.execute(query, params).fetchall()
    print(f"  Found {len(rows)} entries needing compile-only evals")
    if not rows:
        return
    if dry_run:
        print("  [DRY RUN] Would process these entries.")
        return

    dev = torch.device(device)
    success = 0
    failed = 0
    t0 = time.time()

    for i, (eid, tier, result_id, graph_json, config_json) in enumerate(rows):
        try:
            config = _parse_config(config_json)
            model = _build_model(graph_json, config)
            model = model.to(dev)
            model.eval()

            updates = {}
            seq_len = min(128, config.max_seq_len)
            input_batches = _make_input_batches(
                config.vocab_size, dev, n_batches=3, batch_size=4, seq_len=seq_len
            )

            # Efficiency wall
            try:
                ew = evaluate_efficiency_wall(
                    model,
                    config.vocab_size,
                    dev,
                    seq_lens=(64, 128, 256, 512),
                    batch_size=2,
                    memory_budget_mb=2048,
                )
                if ew.get("efficiency_wall_score") is not None:
                    updates["efficiency_wall_score"] = ew["efficiency_wall_score"]
                if ew.get("max_viable_seq_len") is not None:
                    updates["max_viable_seq_len"] = ew["max_viable_seq_len"]
                if ew.get("scaling_regime") is not None:
                    updates["scaling_regime"] = ew["scaling_regime"]
            except Exception as e:
                print(f"    {eid}: efficiency_wall failed: {e}")

            # Activation sparsity
            try:
                sp = evaluate_activation_sparsity(model, input_batches, dev)
                if sp.get("activation_sparsity_score") is not None:
                    updates["activation_sparsity_score"] = sp[
                        "activation_sparsity_score"
                    ]
                if sp.get("dead_neuron_ratio") is not None:
                    updates["dead_neuron_ratio"] = sp["dead_neuron_ratio"]
            except Exception as e:
                print(f"    {eid}: sparsity failed: {e}")

            # Routing heatmap
            try:
                rh = evaluate_routing_heatmap(model, input_batches, dev)
                if rh.get("routing_collapse_score") is not None:
                    updates["routing_collapse_score"] = rh["routing_collapse_score"]
            except Exception as e:
                print(f"    {eid}: routing_heatmap failed: {e}")

            # Architecture telemetry (routing_savings_ratio, compression_ratio)
            try:
                runner = ExperimentRunner.__new__(ExperimentRunner)
                telemetry = runner._extract_architecture_telemetry(model)
                if telemetry.get("routing_savings_ratio") is not None:
                    updates["routing_savings_ratio"] = telemetry[
                        "routing_savings_ratio"
                    ]
                if telemetry.get("compression_ratio") is not None:
                    updates["compression_ratio"] = telemetry["compression_ratio"]
            except Exception as e:
                print(f"    {eid}: telemetry failed: {e}")

            # Write updates via promote_to_tier
            if updates:
                db.promote_to_tier(eid, tier or "screening", **updates)
                success += 1
            else:
                failed += 1

            del model
            if device != "cpu":
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            eta = _fmt_eta(elapsed, i + 1, len(rows))
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(
                    f"  Progress: {i + 1}/{len(rows)} (ok={success} fail={failed}) ETA: {eta}"
                )

        except Exception as e:
            msg = str(e)
            # Truncate long error messages (e.g. unknown primitive lists)
            if len(msg) > 120:
                msg = msg[:120] + "..."
            print(f"  {eid} ({result_id}): FAILED - {msg}")
            failed += 1

    elapsed = time.time() - t0
    print(f"\nPhase 2 complete: {success} updated, {failed} failed in {elapsed:.1f}s\n")


# ---------------------------------------------------------------------------
# Phase 3: Micro-train evals
# ---------------------------------------------------------------------------


def phase_train(db, device="cpu", limit=0, entry_id=None, dry_run=False):
    """Rebuild models, micro-train, then run training-dependent evals."""
    from research.eval.wikitext_eval import evaluate_wikitext_perplexity
    from research.eval.tinystories_eval import evaluate_tinystories
    from research.eval.cross_task_eval import evaluate_cross_task_robustness

    c = db.conn
    print("=== Phase 3: Micro-train evals ===\n")

    where = (
        "l.result_id = pr.result_id"
        " AND pr.graph_json IS NOT NULL"
        " AND pr.result_id NOT LIKE 'ref_%'"
        " AND ("
        "   l.wikitext_perplexity IS NULL"
        "   OR l.tinystories_perplexity IS NULL"
        "   OR l.cross_task_score IS NULL"
        " )"
    )
    if entry_id:
        where += " AND l.entry_id = ?"
        params = [entry_id]
    else:
        params = []

    query = (
        "SELECT l.entry_id, l.tier, pr.result_id, pr.graph_json, e.config_json,"
        " l.wikitext_perplexity, l.tinystories_perplexity, l.cross_task_score,"
        " pr.discovery_loss"
        " FROM leaderboard l"
        " JOIN program_results pr ON l.result_id = pr.result_id"
        " JOIN experiments e ON pr.experiment_id = e.experiment_id"
        f" WHERE {where}"
    )
    if limit > 0 and not entry_id:
        query += f" LIMIT {limit}"

    rows = c.execute(query, params).fetchall()
    print(f"  Found {len(rows)} entries needing micro-train evals")
    if not rows:
        return
    if dry_run:
        print("  [DRY RUN] Would process these entries.")
        return

    dev = torch.device(device)
    success = 0
    failed = 0
    t0 = time.time()
    n_train_steps = 200
    seq_len = 128

    for i, row in enumerate(rows):
        eid = row[0]
        tier = row[1]
        result_id = row[2]
        graph_json = row[3]
        config_json = row[4]
        existing_wikitext = row[5]
        existing_tinystories = row[6]
        existing_cross_task = row[7]
        existing_discovery_loss = row[8]

        try:
            config = _parse_config(config_json)
            updates = {}
            pr_updates = {}

            # Wikitext perplexity (mutates model, so build fresh)
            if existing_wikitext is None:
                try:
                    model = _build_model(graph_json, config)
                    model = model.to(dev)
                    wk = evaluate_wikitext_perplexity(
                        model,
                        config.vocab_size,
                        dev,
                        n_train_steps=n_train_steps,
                        seq_len=min(seq_len, config.max_seq_len),
                    )
                    if wk.get("wikitext_perplexity") is not None:
                        updates["wikitext_perplexity"] = wk["wikitext_perplexity"]
                    if wk.get("wikitext_score") is not None:
                        updates["wikitext_score"] = wk["wikitext_score"]
                    del model
                except Exception as e:
                    print(f"    {eid}: wikitext failed: {e}")

            # TinyStories (fresh model)
            if existing_tinystories is None:
                try:
                    model = _build_model(graph_json, config)
                    model = model.to(dev)
                    ts = evaluate_tinystories(
                        model,
                        config.vocab_size,
                        dev,
                        n_train_steps=n_train_steps,
                        seq_len=min(seq_len, config.max_seq_len),
                    )
                    if ts.get("tinystories_perplexity") is not None:
                        updates["tinystories_perplexity"] = ts["tinystories_perplexity"]
                    if ts.get("tinystories_score") is not None:
                        updates["tinystories_score"] = ts["tinystories_score"]
                    del model
                except Exception as e:
                    print(f"    {eid}: tinystories failed: {e}")

            # Cross-task robustness (needs model factory)
            if existing_cross_task is None:
                try:

                    def make_model_fn(_gj=graph_json, _cfg=config, _dev=dev):
                        m = _build_model(_gj, _parse_config(config_json))
                        return m.to(_dev)

                    ct = evaluate_cross_task_robustness(
                        make_model_fn,
                        config.vocab_size,
                        dev,
                        n_train_steps=n_train_steps,
                        seq_len=min(seq_len, config.max_seq_len),
                    )
                    if ct.get("cross_task_score") is not None:
                        updates["cross_task_score"] = ct["cross_task_score"]
                except Exception as e:
                    print(f"    {eid}: cross_task failed: {e}")

            # Discovery loss (if missing in program_results)
            if existing_discovery_loss is None:
                try:
                    model = _build_model(graph_json, config)
                    model = model.to(dev)
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for _ in range(2):
                            ids = torch.randint(
                                0,
                                config.vocab_size,
                                (4, min(seq_len, config.max_seq_len)),
                                device=dev,
                            )
                            out = model(ids)
                            logits = (
                                out.logits
                                if hasattr(out, "logits")
                                else (out[0] if isinstance(out, tuple) else out)
                            )
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = ids[:, 1:].contiguous()
                            loss = torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                            if torch.isfinite(loss):
                                losses.append(loss.item())
                    if losses:
                        disc_loss = sum(losses) / len(losses)
                        pr_updates["discovery_loss"] = disc_loss
                    del model
                except Exception as e:
                    print(f"    {eid}: discovery_loss failed: {e}")

            # Write leaderboard updates
            if updates:
                db.promote_to_tier(eid, tier or "screening", **updates)

            # Write program_results updates
            if pr_updates:
                set_clause = ", ".join(f"{k} = ?" for k in pr_updates)
                c.execute(
                    f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
                    list(pr_updates.values()) + [result_id],
                )
                db.conn.commit()

            if updates or pr_updates:
                success += 1
            else:
                failed += 1

            if device != "cpu":
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            eta = _fmt_eta(elapsed, i + 1, len(rows))
            if (i + 1) % 5 == 0 or (i + 1) == len(rows):
                print(
                    f"  Progress: {i + 1}/{len(rows)} (ok={success} fail={failed}) ETA: {eta}"
                )

        except Exception as e:
            msg = str(e)
            if len(msg) > 120:
                msg = msg[:120] + "..."
            print(f"  {eid} ({result_id}): FAILED - {msg}")
            failed += 1

    elapsed = time.time() - t0
    print(f"\nPhase 3 complete: {success} updated, {failed} failed in {elapsed:.1f}s\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing leaderboard evaluation data"
    )
    parser.add_argument(
        "--phase",
        choices=["sql", "compile", "train", "all"],
        help="Which backfill phase to run",
    )
    parser.add_argument("--status", action="store_true", help="Print coverage report")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max entries to process (0=all)"
    )
    parser.add_argument(
        "--entry-id", type=str, default=None, help="Target a specific entry"
    )
    parser.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    parser.add_argument("--db", default=DB_PATH, help="Path to lab_notebook.db")
    args = parser.parse_args()

    db = LabNotebook(args.db)
    # Set busy timeout so we don't fail on "database is locked" when
    # the dashboard or other processes hold the DB open.
    db.conn.execute("PRAGMA busy_timeout = 30000")  # 30s retry

    if args.status or not args.phase:
        print_status(db)
        return

    t0 = time.time()

    if args.phase in ("sql", "all"):
        phase_sql(db, dry_run=args.dry_run)

    if args.phase in ("compile", "all"):
        phase_compile(
            db,
            device=args.device,
            limit=args.limit,
            entry_id=args.entry_id,
            dry_run=args.dry_run,
        )

    if args.phase in ("train", "all"):
        phase_train(
            db,
            device=args.device,
            limit=args.limit,
            entry_id=args.entry_id,
            dry_run=args.dry_run,
        )

    elapsed = time.time() - t0
    print(f"Total elapsed: {elapsed:.1f}s\n")
    print_status(db)


if __name__ == "__main__":
    main()
