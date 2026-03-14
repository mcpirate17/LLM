"""Cache equal-budget WikiText PPL trajectories for reference architectures.

Runs each reference at checkpoints [200, 500, 1000, 2000, 4000] steps and
stores the trajectory in a JSON artifact + leaderboard columns.

This is W1 of the real-token eval action plan — the blocking prerequisite
for trajectory-based escalation triggers and equal-budget frontier comparison.

Usage:
    python -m research.tools.cache_reference_trajectories --device cpu
    python -m research.tools.cache_reference_trajectories --arch gpt2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
from ..defaults import VOCAB_SIZE

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = (200, 500, 1000, 2000, 4000)
ARTIFACT_PATH = Path("research/eval/reference_trajectories.json")


def run_reference_trajectory(
    arch_key: str,
    device: str = "cpu",
    d_model: int = 256,
    n_layers: int = 6,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = 128,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
) -> Dict[str, Any]:
    """Build a reference model and evaluate its WikiText trajectory."""
    from ..synthesis.reference_architectures import REFERENCE_ARCHITECTURES, build_reference
    from ..synthesis.compiler import compile_model
    from ..eval.wikitext_eval import evaluate_wikitext_trajectory

    ref_info = REFERENCE_ARCHITECTURES[arch_key]
    ref_name = ref_info["name"]
    log.info("=== %s: running trajectory at checkpoints %s ===", ref_name, checkpoints)

    layer_graphs = [build_reference(arch_key, d_model) for _ in range(n_layers)]
    model = compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=seq_len)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("  Params: %s", f"{total_params:,}")

    result = evaluate_wikitext_trajectory(
        model, vocab_size, str(dev),
        checkpoints=checkpoints,
        seq_len=seq_len,
    )

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "arch_key": arch_key,
        "reference_name": ref_name,
        "paradigm": ref_info.get("paradigm", ""),
        "param_count": total_params,
        "d_model": d_model,
        "n_layers": n_layers,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        **result,
    }


def save_artifact(results: list[Dict[str, Any]], path: Path) -> None:
    """Write trajectory results to a JSON artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing data if present
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if "trajectories" not in existing:
        existing = {"generated_at": time.time(), "protocol": "trajectory_probe_v1", "trajectories": {}}

    for r in results:
        existing["trajectories"][r["arch_key"]] = r

    existing["generated_at"] = time.time()
    path.write_text(json.dumps(existing, indent=2, default=str))
    log.info("Saved trajectory artifact to %s", path)


def update_leaderboard(results: list[Dict[str, Any]]) -> None:
    """Update leaderboard rows for references with trajectory data."""
    from ..scientist.notebook import LabNotebook

    nb = LabNotebook()
    refs = nb.get_references()
    ref_by_name = {r.get("reference_name"): r for r in refs}

    for r in results:
        ref_row = ref_by_name.get(r["reference_name"])
        if not ref_row:
            log.warning("  %s not found on leaderboard, skipping", r["reference_name"])
            continue

        ckpts = r.get("checkpoints", {})
        best_ckpt = max(ckpts.keys()) if ckpts else None

        update_kwargs: Dict[str, Any] = {
            "evaluation_stage": "VALIDATED",
            "eval_budget_steps": best_ckpt,
            "robustness_grade": "A",
        }
        # Use peak_ppl (best at any checkpoint) as the canonical PPL
        if r.get("peak_ppl") is not None:
            update_kwargs["peak_ppl"] = r["peak_ppl"]
            update_kwargs["wikitext_perplexity"] = r["peak_ppl"]
        if r.get("peak_step") is not None:
            update_kwargs["peak_step"] = r["peak_step"]
        if r.get("steps_to_divergence") is not None:
            update_kwargs["steps_to_divergence"] = r["steps_to_divergence"]
        # ppl_500 for equal-budget comparison
        ppl_500 = ckpts.get(500, {}).get("ppl") if 500 in ckpts else None
        if ppl_500 is not None:
            update_kwargs["ppl_500"] = ppl_500
        # Compute wikitext_score from peak_ppl
        if r.get("peak_ppl") and r["peak_ppl"] > 0:
            import math
            vocab = r.get("vocab_size", 32000)
            ws = max(0.0, math.log(vocab / r["peak_ppl"]) / math.log(vocab))
            update_kwargs["wikitext_score"] = round(ws, 4)
        if r.get("improvement_ratio") is not None:
            update_kwargs["wikitext_ppl_improvement_ratio"] = r["improvement_ratio"]

        entry_id = ref_row.get("entry_id")
        if entry_id:
            sets = []
            params = []
            for col, val in update_kwargs.items():
                sets.append(f"{col} = ?")
                params.append(val)
            params.append(entry_id)
            try:
                nb.conn.execute(
                    f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
                    params,
                )
                nb.conn.commit()
                log.info("  Updated leaderboard for %s (entry_id=%s)", r["reference_name"], entry_id)
            except Exception as e:
                log.warning("  Leaderboard update failed for %s: %s", r["reference_name"], e)

    nb.close()


def main() -> None:
    from ..synthesis.reference_architectures import REFERENCE_ARCHITECTURES

    parser = argparse.ArgumentParser(description="Cache reference WikiText trajectories")
    parser.add_argument(
        "--arch", default="all",
        choices=list(REFERENCE_ARCHITECTURES.keys()) + ["all"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--no-leaderboard", action="store_true",
                        help="Skip leaderboard update, only save JSON artifact")
    args = parser.parse_args()

    arch_keys = list(REFERENCE_ARCHITECTURES.keys()) if args.arch == "all" else [args.arch]

    results = []
    for key in arch_keys:
        try:
            r = run_reference_trajectory(key, device=args.device,
                                         d_model=args.d_model, n_layers=args.n_layers)
            results.append(r)

            # Print summary
            ckpts = r.get("checkpoints", {})
            for step in sorted(ckpts.keys()):
                c = ckpts[step]
                log.info("  %s @ %d steps: PPL=%.1f score=%.3f",
                         r["reference_name"], step,
                         c.get("ppl") or 0, c.get("score") or 0)
        except Exception as e:
            log.error("  %s FAILED: %s", key, e)

    if results:
        save_artifact(results, ARTIFACT_PATH)
        if not args.no_leaderboard:
            update_leaderboard(results)

    log.info("Done. %d/%d references cached.", len(results), len(arch_keys))


if __name__ == "__main__":
    main()
