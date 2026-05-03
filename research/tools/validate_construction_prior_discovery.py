#!/usr/bin/env python3
"""Validate active construction priors against candidate generation.

The audit phase compares a normal active-prior grammar against a baseline
grammar with learned/prior signals disabled, then compiles and smoke-tests the
generated candidates. The optional run phase launches a small real screening
experiment through the standard runner path, so full S1 metrics are recorded by
the platform rather than by this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.scientist.construction_priors import get_active_construction_prior  # noqa: E402
from research.scientist.notebook import LabNotebook  # noqa: E402
from research.scientist.runner import ExperimentRunner, RunConfig  # noqa: E402
from research.scientist.runner.execution_screening import (  # noqa: E402
    _make_experiment_results,
)
from research.synthesis.compiler import compile_model  # noqa: E402
from research.synthesis.grammar import batch_generate  # noqa: E402
from research.synthesis.validator import validate_graph  # noqa: E402


RUNTIME_DIR = PROJECT_ROOT / "research/runtime"
DB_PATH = PROJECT_ROOT / "research/lab_notebook.db"


def _config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig()
    cfg.n_programs = max(1, int(args.n_programs))
    cfg.max_ops = max(1, int(args.max_ops))
    cfg.max_depth = max(1, int(args.max_depth))
    cfg.composition_depth = max(1, int(args.composition_depth))
    cfg.stage1_steps = max(1, int(args.stage1_steps))
    cfg.device = str(args.device)
    cfg.model_source = "graph_synthesis"
    cfg.mode = "single"
    cfg.continuous = False
    cfg.auto_investigate = False
    cfg.auto_validate = False
    cfg.auto_scale_up = False
    cfg.enable_causal_ablation = False
    cfg.use_construction_priors = not bool(args.disable_construction_priors)
    return cfg


def _slot_motif_counts(graphs: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for graph in graphs:
        usage = (getattr(graph, "metadata", {}) or {}).get("template_slot_usage") or []
        if not isinstance(usage, list):
            continue
        for item in usage:
            if not isinstance(item, dict):
                continue
            slot = item.get("slot_key_canonical") or item.get("slot_key")
            motif = item.get("selected_motif")
            if slot and motif:
                counts[f"{slot}:{motif}"] += 1
    return counts


def _op_counts(graphs: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for graph in graphs:
        for node in graph.nodes.values():
            if not node.is_input:
                counts[str(node.op_name)] += 1
    return counts


def _compile_smoke(graphs: list[Any], config: RunConfig) -> dict[str, Any]:
    import torch

    compiled = 0
    passed_s0 = 0
    errors: Counter[str] = Counter()
    valid = 0
    invalid = 0
    for graph in graphs:
        validation = validate_graph(
            graph,
            max_ops=max(1, int(config.max_ops)),
            max_depth=max(1, int(config.max_depth)),
            min_splits=config.min_splits,
        )
        if not validation.valid:
            invalid += 1
            for err in validation.errors[:2]:
                errors[f"validation:{err[:120]}"] += 1
            continue
        valid += 1
        try:
            model = compile_model(
                [graph] * max(1, int(config.n_layers)),
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            compiled += 1
            x = torch.randint(0, config.vocab_size, (1, min(16, config.max_seq_len)))
            out = model(x)
            if out is not None:
                passed_s0 += 1
        except Exception as exc:
            errors[type(exc).__name__ + ":" + str(exc)[:160]] += 1
    return {
        "valid": valid,
        "invalid": invalid,
        "compiled": compiled,
        "passed_s0": passed_s0,
        "errors": dict(errors.most_common(10)),
    }


def _audit_generation(
    *,
    runner: ExperimentRunner,
    nb: LabNotebook,
    config: RunConfig,
    sample_n: int,
    use_learned_grammar: bool,
    seed: int,
) -> dict[str, Any]:
    results = _make_experiment_results()
    label = "active_prior" if use_learned_grammar else "baseline_no_learned_prior"
    grammar, _failure_blocklist, _analytics = runner._prepare_grammar_config(
        f"prior_validation_{label}",
        config,
        nb,
        results,
        use_learned_grammar=use_learned_grammar,
    )
    started = time.time()
    generated = batch_generate(
        max(1, int(sample_n)),
        grammar,
        base_seed=int(seed),
    ).graphs
    smoke = _compile_smoke(generated, config)
    slot_counts = _slot_motif_counts(generated)
    op_counts = _op_counts(generated)
    active_prior = get_active_construction_prior(nb)
    local = ((active_prior or {}).get("payload") or {}).get("local_edit_priors") or {}
    multipliers = local.get("slot_motif_multipliers") or {}
    prior_slot_hits = []
    for slot_motif, count in slot_counts.items():
        if ":" not in slot_motif:
            continue
        slot, motif = slot_motif.rsplit(":", 1)
        multiplier = (multipliers.get(slot) or {}).get(motif)
        if multiplier is not None:
            prior_slot_hits.append(
                {
                    "slot_motif": slot_motif,
                    "count": count,
                    "multiplier": float(multiplier),
                }
            )
    prior_slot_hits.sort(
        key=lambda item: (abs(item["multiplier"] - 1.0), item["count"]), reverse=True
    )
    return {
        "label": label,
        "use_learned_grammar": bool(use_learned_grammar),
        "elapsed_sec": round(time.time() - started, 3),
        "generated": len(generated),
        "unique_fingerprints": len({graph.fingerprint() for graph in generated}),
        "compile_smoke": smoke,
        "top_ops": dict(op_counts.most_common(20)),
        "top_slot_motifs": dict(slot_counts.most_common(20)),
        "prior_slot_hits": prior_slot_hits[:20],
        "applied_op_weight_count": len(getattr(grammar, "op_weights", {}) or {}),
        "applied_slot_motif_multiplier_count": sum(
            len(v)
            for v in (
                getattr(grammar, "slot_motif_weight_multipliers", {}) or {}
            ).values()
        ),
    }


def _run_screening_batch(
    *,
    db_path: Path,
    config: RunConfig,
    hypothesis: str,
) -> dict[str, Any]:
    runner = ExperimentRunner(str(db_path))
    exp_id = runner.start_experiment(config, hypothesis=hypothesis, exploratory=True)
    thread = getattr(runner, "_thread", None)
    if thread is not None:
        thread.join()
    nb = LabNotebook(str(db_path), use_native=False)
    try:
        rows = nb.conn.execute(
            """
            SELECT result_id, stage0_passed, stage05_passed, stage1_passed,
                   loss_ratio, wikitext_perplexity, hellaswag_acc,
                   blimp_overall_accuracy, induction_auc, binding_auc,
                   binding_composite, ar_auc, error_type, stage_at_death
            FROM program_results
            WHERE experiment_id = ?
            ORDER BY timestamp DESC
            """,
            (exp_id,),
        ).fetchall()
    finally:
        nb.close()
    required = (
        "wikitext_perplexity",
        "hellaswag_acc",
        "blimp_overall_accuracy",
        "induction_auc",
        "binding_auc",
        "binding_composite",
        "ar_auc",
    )
    s1_rows = [row for row in rows if int(row["stage1_passed"] or 0) == 1]
    return {
        "experiment_id": exp_id,
        "program_rows": len(rows),
        "stage0_passed": sum(1 for row in rows if int(row["stage0_passed"] or 0) == 1),
        "stage05_passed": sum(
            1 for row in rows if int(row["stage05_passed"] or 0) == 1
        ),
        "stage1_passed": len(s1_rows),
        "stage1_missing_required_metrics": sum(
            1 for row in s1_rows if any(row[key] is None for key in required)
        ),
        "best_loss_ratio": min(
            [
                float(row["loss_ratio"])
                for row in s1_rows
                if row["loss_ratio"] is not None
            ],
            default=None,
        ),
        "result_ids": [str(row["result_id"]) for row in rows],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--sample-n", type=int, default=12)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--max-ops", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--composition-depth", type=int, default=3)
    parser.add_argument("--stage1-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=260501)
    parser.add_argument("--run-s1", action="store_true")
    parser.add_argument("--disable-construction-priors", action="store_true")
    parser.add_argument(
        "--output",
        default=str(RUNTIME_DIR / "construction_prior_discovery_validation.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    output = Path(args.output)
    config = _config(args)
    nb = LabNotebook(str(db_path), use_native=False)
    try:
        active_prior = get_active_construction_prior(nb)
        runner = ExperimentRunner(str(db_path))
        audit = {
            "created_at": time.time(),
            "active_prior": {
                "version": (active_prior or {}).get("version"),
                "summary": (active_prior or {}).get("summary") or {},
            },
            "config": config.to_dict(),
            "sample_n": int(args.sample_n),
            "baseline": _audit_generation(
                runner=runner,
                nb=nb,
                config=config,
                sample_n=int(args.sample_n),
                use_learned_grammar=False,
                seed=int(args.seed),
            ),
            "active": _audit_generation(
                runner=runner,
                nb=nb,
                config=config,
                sample_n=int(args.sample_n),
                use_learned_grammar=True,
                seed=int(args.seed),
            ),
        }
    finally:
        nb.close()

    if args.run_s1:
        audit["screening_run"] = _run_screening_batch(
            db_path=db_path,
            config=config,
            hypothesis=(
                "Validate active local ablation construction priors on a small "
                "screening batch; advisory only, no gating."
            ),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "active_prior": audit["active_prior"],
                "screening_run": audit.get("screening_run"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
