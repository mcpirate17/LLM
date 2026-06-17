"""Param/width scaling ladder — predictivity calibration for cheap screening.

Goal: find the cheapest model scale whose capability ranking PREDICTS the 40M
outcome, so screening stops relying on the (anti-predictive) nano signal. The
ladder spans two regimes the project already uses:

  * LANE rungs (4K -> ~400K): the mechanism in isolation, scored on the capability
    probes (binding / induction / state-tracking) that DISCRIMINATE at small scale
    (BLiMP is ~chance < 30M, so it carries no signal at the cheap end). This is
    what nano screening does, swept across width.
  * LM rungs (~4M): a full TinyLM (embedding + lane + FFN + head) scored on real
    BLiMP + perplexity. A real BPE vocab makes a <1M-param *full* LM impossible
    (embedding dominates), so full-LM rungs start here.
  * L4 (~30-100M): read from research/data/scale_ladder/softmax_l4_anchor.json +
    runs.db (no re-run — the ground truth already exists).

Run softmax first to establish the reference curve + validate the harness; then
the same call on candidate lanes gives, per metric, Spearman(rung_k, L4) across
candidates — the cheapest predictive rung. CPU-friendly; lane rungs are seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import torch

from component_fab.harness.lm_eval import evaluate_lm
from component_fab.harness.probe_block import short_training_probe
from component_fab.harness.probe_tasks import DEFAULT_PROBE_TASKS
from component_fab.harness.tiny_lm import lane_factory_for_baseline
from component_fab.harness.training_probe import build_tiny_lm

_REPO = Path(__file__).resolve().parents[1]
_ANCHOR = _REPO / "data" / "scale_ladder" / "softmax_l4_anchor.json"

# Lane-rung widths (dim) -> roughly {4K, 16K, 65K, 260K} lane params for attention.
_LANE_DIMS = (32, 64, 128, 256)
# Full-LM rung: (dim, n_blocks) chosen to land near ~4M params with a real vocab.
_LM_RUNGS = ((192, 4),)
_PROBE_STEPS = 600  # fixed across lane rungs to isolate the width effect
_PROBE_SEQ = 64


def _lane_params(lane_factory, dim: int) -> int:
    return sum(p.numel() for p in lane_factory(dim).parameters())


def _probe_capability(lane_factory, dim: int, *, seeds=(0, 1)) -> dict[str, float]:
    """Mean loss-ratio per probe task (higher = learns the task better)."""
    per_task: dict[str, list[float]] = {t.name: [] for t in DEFAULT_PROBE_TASKS}
    for seed in seeds:
        torch.manual_seed(seed)
        lane = lane_factory(dim)
        for task in DEFAULT_PROBE_TASKS:
            r = short_training_probe(
                lane,
                dim=dim,
                seq_len=_PROBE_SEQ,
                n_steps=_PROBE_STEPS,
                seed=seed,
                target_fn=task.target_fn,
            )
            if r.trained_successfully:
                per_task[task.name].append(r.loss_ratio_initial_over_final)
    out = {k: (mean(v) if v else 0.0) for k, v in per_task.items()}
    out["_mean"] = mean([out[t.name] for t in DEFAULT_PROBE_TASKS])
    return out


def _lm_params(lane_factory, dim: int, n_blocks: int) -> int:
    m = build_tiny_lm(
        lane_factory,
        vocab_size=8000,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=128,
        use_position_embedding=True,
        use_ffn=True,
        ffn_mult=4,
    )
    return sum(p.numel() for p in m.parameters())


def run_ladder(lane_name: str, *, device: str = "cpu") -> dict:
    lane_factory = lane_factory_for_baseline(lane_name)
    report: dict = {"lane": lane_name, "lane_rungs": [], "lm_rungs": []}

    print(
        f"\n=== {lane_name}: LANE rungs (capability probes, {_PROBE_STEPS} steps) ==="
    )
    for dim in _LANE_DIMS:
        t0 = time.time()
        params = _lane_params(lane_factory, dim)
        cap = _probe_capability(lane_factory, dim)
        row = {
            "dim": dim,
            "lane_params": params,
            "mean_loss_ratio": round(cap["_mean"], 4),
            "induction": round(cap.get("causal_induction", 0.0), 4),
            "running_parity": round(cap.get("running_parity", 0.0), 4),
            "shifted_copy": round(cap.get("shifted_copy", 0.0), 4),
            "seconds": round(time.time() - t0, 1),
        }
        report["lane_rungs"].append(row)
        print(
            f"  dim{dim:<4} params={params:>8,} mean_lr={row['mean_loss_ratio']:.3f} "
            f"induction={row['induction']:.3f} parity={row['running_parity']:.3f} "
            f"({row['seconds']}s)"
        )

    print(f"\n=== {lane_name}: LM rungs (TinyLM + real BLiMP + ppl) ===")
    for dim, n_blocks in _LM_RUNGS:
        t0 = time.time()
        params = _lm_params(lane_factory, dim, n_blocks)
        # ~5 tok/param @ batch16/seq128, capped to bound CPU wall-clock (the lane
        # rungs carry the discriminating signal; the LM rung is the BLiMP bridge).
        n_steps = min(4000, max(200, (5 * params) // (16 * 128)))
        res = evaluate_lm(
            lane_factory,
            mixer_label=lane_name,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_steps,
            device=device,
        )
        row = {
            "dim": dim,
            "n_blocks": n_blocks,
            "lm_params": params,
            "n_train_steps": n_steps,
            "blimp": round(float(res.blimp_overall_accuracy), 4),
            "post_ppl": round(float(getattr(res, "post_train_ppl", float("nan"))), 2),
            "seconds": round(time.time() - t0, 1),
        }
        report["lm_rungs"].append(row)
        print(
            f"  dim{dim}/{n_blocks}blk params={params:>10,} steps={n_steps} "
            f"BLiMP={row['blimp']:.4f} ppl={row['post_ppl']} ({row['seconds']}s)"
        )

    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lanes", nargs="*", default=["softmax_attention", "gpt2"])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output",
        default=str(_REPO / "reports" / "scale_ladder_softmax.json"),
    )
    args = p.parse_args(argv)

    anchor = json.loads(_ANCHOR.read_text()) if _ANCHOR.exists() else {}
    out = {"L4_anchor": anchor.get("anchors", {}), "ladders": []}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for lane in args.lanes:
        out["ladders"].append(run_ladder(lane, device=args.device))
        out_path.write_text(json.dumps(out, indent=2))

    print(f"\nL4 anchor (existing softmax @ scale): {list(out['L4_anchor'])}")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
