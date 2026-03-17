"""Rerun training curves for Stage 1 survivors with missing or short curves.

Usage:
  # Dry run
  python -m research.tools.backfill_training_curves --min-steps 200 --steps 1000 --device cuda --dry-run

  # Rerun curves
  python -m research.tools.backfill_training_curves --min-steps 200 --steps 1000 --device cuda
"""

import argparse
import dataclasses
import json
import os

import torch

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.synthesis.serializer import graph_from_json
from research.scientist.native_runner import compile_model_native_first as compile_model


DB_PATH = os.environ.get(
    "LAB_NOTEBOOK_DB",
    os.path.join(os.path.dirname(__file__), "..", "lab_notebook.db"),
)


def _load_config(config_json: str | None) -> RunConfig:
    config_dict = json.loads(config_json) if config_json else {}
    valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return RunConfig(**filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill training curves for survivors"
    )
    parser.add_argument(
        "--min-steps", type=int, default=200, help="Min curve length to keep"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Steps to use when rerunning"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Max rows per batch")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to lab_notebook.db")
    args = parser.parse_args()

    db = LabNotebook(args.db)
    c = db.conn

    # Build map of existing curve lengths
    curve_counts = {
        r[0]: r[1]
        for r in c.execute(
            "SELECT result_id, COUNT(*) FROM training_curves GROUP BY result_id"
        ).fetchall()
    }

    rows = c.execute(
        "SELECT p.result_id, p.graph_json, e.config_json"
        " FROM program_results p"
        " JOIN experiments e ON p.experiment_id = e.experiment_id"
        " WHERE p.stage1_passed = 1"
        "   AND p.graph_json IS NOT NULL"
        "   AND p.result_id NOT LIKE 'ref_%'"
        " LIMIT ?",
        (args.batch_size,),
    ).fetchall()

    targets = []
    for rid, graph_json, config_json in rows:
        existing = curve_counts.get(rid, 0)
        if existing < args.min_steps:
            targets.append((rid, graph_json, config_json, existing))

    if not targets:
        print("No rows need training curve backfill.")
        return

    print(
        f"Found {len(targets)} rows needing training curve backfill (min_steps={args.min_steps})."
    )
    if args.dry_run:
        for rid, _, _, existing in targets[:10]:
            print(f"  {rid}: existing_steps={existing}")
        print("[DRY RUN] No changes written.")
        return

    runner = ExperimentRunner(args.db)
    dev = torch.device(args.device)
    success = 0
    failed = 0

    for rid, graph_json, config_json, existing in targets:
        try:
            config = _load_config(config_json)
            config.collect_training_curve = True
            config.stage1_steps = max(int(args.steps), int(config.stage1_steps))

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
            s1 = runner._micro_train(model, config, dev, seed=42)
            curve = s1.get("training_curve") or []
            if curve:
                db.store_training_curve(rid, curve)
                success += 1
            else:
                print(f"  {rid}: no curve produced")
                failed += 1

            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  {rid}: FAILED - {e}")
            failed += 1

    print(f"Training curve backfill complete: {success} updated, {failed} failed.")


if __name__ == "__main__":
    main()
