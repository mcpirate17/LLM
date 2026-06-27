"""Nano no-go gate for the HYPERBOLIC addressing lane vs the reciprocal softmax-twin.

The proposal: replace reciprocal (a cosmetic softmax derivative) with hyperbolic
(Lorentz-model) distance scoring — a genuinely non-Euclidean addressing geometry.
This gate asks the only question that justifies it: does hyperbolic MATCH OR BEAT
reciprocal PER PARAMETER on induction (reciprocal's proven strength) and binding?
All three lanes are param-matched single-head attention; only the score geometry
differs. Gate PASS = hyperbolic >= reciprocal on induction AUC (and not worse on
binding). FAIL = it's just a slower softmax variant; don't scale it.

    python -m research.tools.grade_hyperbolic_nano --device cuda --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from component_fab.harness.tiny_lm import TinyLM, TinyLMConfig
from research.eval.binding_multislot_probe import (
    BindingMultislotConfig,
    binding_multislot_probe,
    build_multi_blank_layout,
)
from research.eval.induction_probe import induction_score
from research.tools._scaling_lanes import _build_lane_factory

LANES = ("softmax_attention", "reciprocal_rank_attention", "hyperbolic_attention")


def _grade_one(
    lane: str,
    *,
    dim: int,
    n_blocks: int,
    steps: int,
    seed: int,
    device: str,
    binding_to: float,
) -> dict:
    factory = _build_lane_factory(lane)
    bcfg = BindingMultislotConfig(seed=seed, train_steps=steps, timeout_s=binding_to)
    vocab = max(256, build_multi_blank_layout(bcfg).required_vocab) + 1
    torch.manual_seed(seed)
    model = TinyLM(
        factory,
        TinyLMConfig(vocab_size=vocab, dim=dim, n_blocks=n_blocks, use_ffn=True),
    )
    n_params = sum(p.numel() for p in model.parameters())

    ind = induction_score(model, n_train_steps=steps, device=device, seed=seed)
    bind = binding_multislot_probe(model, cfg=bcfg, device=device).to_dict()
    return {
        "induction_auc": round(float(ind.auc), 4),
        "induction_status": ind.status,
        "binding_all_slots": round(
            float(bind.get("binding_multislot_all_slots_acc", 0.0)), 4
        ),
        "binding_two_plus": round(
            float(bind.get("binding_multislot_two_plus_slots_acc", 0.0)), 4
        ),
        "binding_status": bind.get("binding_multislot_status", "?"),
        "n_params": n_params,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--binding-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="research/reports/hyperbolic_nano_gate.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    t0 = time.perf_counter()
    rows: dict[str, list[dict]] = {lane: [] for lane in LANES}
    for lane in LANES:
        for seed in seeds:
            row = _grade_one(
                lane,
                dim=args.dim,
                n_blocks=args.n_blocks,
                steps=args.steps,
                seed=seed,
                device=args.device,
                binding_to=args.binding_timeout,
            )
            rows[lane].append(row)
            print(f"[{lane} seed={seed}] {row}")

    keys = ("induction_auc", "binding_all_slots", "binding_two_plus")
    summary = {
        lane: {
            **{
                k: round(sum(r[k] for r in rows[lane]) / len(rows[lane]), 4)
                for k in keys
            },
            "n_params": rows[lane][0]["n_params"],
        }
        for lane in LANES
    }
    hyp, recip = summary["hyperbolic_attention"], summary["reciprocal_rank_attention"]
    verdict = {
        "hyperbolic_beats_reciprocal_induction": hyp["induction_auc"]
        >= recip["induction_auc"],
        "induction_delta_vs_reciprocal": round(
            hyp["induction_auc"] - recip["induction_auc"], 4
        ),
        "binding_all_slots_delta_vs_reciprocal": round(
            hyp["binding_all_slots"] - recip["binding_all_slots"], 4
        ),
    }
    report = {
        "config": vars(args),
        "per_seed": rows,
        "summary": summary,
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY (mean over seeds) ===")
    for lane in LANES:
        s = summary[lane]
        print(
            f"{lane:28s} induction_auc={s['induction_auc']:.3f}  "
            f"binding_all_slots={s['binding_all_slots']:.3f}  two_plus={s['binding_two_plus']:.3f}"
        )
    print(f"VERDICT: {verdict}")
    print(f"wrote {args.out}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
