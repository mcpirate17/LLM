"""Scaled discriminator for the HYPERBOLIC addressing lane vs the reciprocal twin.

The nano gate (dim128, 1k steps) tied hyperbolic with reciprocal. This asks the
follow-up: with QUADRUPLED params (dim 128 -> 256) and an escalating step ladder
(>20k), does hyperbolic SEPARATE from reciprocal on induction (their tie axis)?
Keep climbing the ladder while hyperbolic's induction AUC keeps rising. Softmax
included as the floor. All CPU; results stream after every probe.

    python -m research.tools.hyperbolic_param_step_sweep --dim 256 --device cpu
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

LANES = ("hyperbolic_attention", "reciprocal_rank_attention", "softmax_attention")
LADDER = (1000, 2000, 5000, 10000, 20000, 50000)


def _probe(
    lane: str,
    *,
    dim: int,
    n_blocks: int,
    steps: int,
    seed: int,
    device: str,
    timeout: float,
) -> dict:
    factory = _build_lane_factory(lane)
    bcfg = BindingMultislotConfig(seed=seed, train_steps=steps, timeout_s=timeout)
    vocab = max(256, build_multi_blank_layout(bcfg).required_vocab) + 1
    torch.manual_seed(seed)
    model = TinyLM(
        factory,
        TinyLMConfig(vocab_size=vocab, dim=dim, n_blocks=n_blocks, use_ffn=True),
    )
    nonembed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embed" not in n.lower() and "lm_head" not in n.lower()
    )
    t0 = time.perf_counter()
    ind = induction_score(model, n_train_steps=steps, device=device, seed=seed)
    bind = binding_multislot_probe(model, cfg=bcfg, device=device).to_dict()
    return {
        "steps": steps,
        "induction_auc": round(float(ind.auc), 4),
        "induction_status": ind.status,
        "binding_all_slots": round(
            float(bind.get("binding_multislot_all_slots_acc", 0.0)), 4
        ),
        "binding_two_plus": round(
            float(bind.get("binding_multislot_two_plus_slots_acc", 0.0)), 4
        ),
        "nonembed_params": nonembed,
        "probe_seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dim", type=int, default=256, help="quadrupled vs the dim128 nano gate"
    )
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="hyperbolic induction-AUC rise to keep climbing",
    )
    ap.add_argument("--timeout", type=float, default=6000.0)
    ap.add_argument(
        "--out", default="research/reports/hyperbolic_param_step_sweep.json"
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {lane: [] for lane in LANES}
    started = time.perf_counter()

    def flush() -> None:
        out.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "ladder": list(LADDER),
                    "results": results,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                },
                indent=2,
            )
        )

    prev_hyp = -1.0
    for steps in LADDER:
        for lane in LANES:
            row = _probe(
                lane,
                dim=args.dim,
                n_blocks=args.n_blocks,
                steps=steps,
                seed=args.seed,
                device=args.device,
                timeout=args.timeout,
            )
            results[lane].append(row)
            flush()
            print(
                f"[{lane:26s} steps={steps:>6}] induction_auc={row['induction_auc']:.4f} "
                f"binding_all_slots={row['binding_all_slots']:.4f} "
                f"({row['probe_seconds']}s, nonembed={row['nonembed_params'] / 1e6:.2f}M)",
                flush=True,
            )
        hyp = results["hyperbolic_attention"][-1]["induction_auc"]
        recip = results["reciprocal_rank_attention"][-1]["induction_auc"]
        print(
            f"  >> @ {steps} steps: hyperbolic {hyp:.4f} vs reciprocal {recip:.4f}  (delta {hyp - recip:+.4f})",
            flush=True,
        )
        gain = hyp - prev_hyp
        if steps >= 20000 and gain < args.min_gain:
            print(
                f"  hyperbolic plateaued at {steps} (gain {gain:+.4f}); stopping ladder",
                flush=True,
            )
            break
        prev_hyp = hyp

    flush()
    print(f"\nDONE. wrote {out} ({round(time.perf_counter() - started, 1)}s)")


if __name__ == "__main__":
    main()
