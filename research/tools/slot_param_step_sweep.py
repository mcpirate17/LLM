"""Educational params x training-steps scaling study for the slot lanes.

Quadruples the nano active params (default dim 128 -> 256) and sweeps the
binding_multislot probe over an escalating step ladder for BOTH the baseline
slot lane and the DPLR slot lane. Adaptive stop: keep climbing the ladder while
all_slots keeps rising by >= --min-gain; stop a lane once it plateaus. All CPU
so it never touches a live GPU run. Results are written after EVERY probe so a
long run can be collected mid-flight.

    python -m research.tools.slot_param_step_sweep --dim 256 --device cpu
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
from research.tools.grade_slot_dplr_nano import _ablation_factory

# variant key -> human label. "base" = no DPLR levers (baseline slot_table_mh),
# "all" = content-forget + DPLR value + learnable slots (slot_table_mh_dplr).
LANES = (("baseline_slot", "base"), ("dplr_slot", "all"))
LADDER = (1000, 2000, 5000, 10000, 20000, 50000)


def _probe(
    variant: str,
    *,
    dim: int,
    n_blocks: int,
    steps: int,
    seed: int,
    device: str,
    timeout: float,
) -> dict:
    factory = _ablation_factory(variant)
    cfg = BindingMultislotConfig(seed=seed, train_steps=steps, timeout_s=timeout)
    vocab = build_multi_blank_layout(cfg).required_vocab + 1
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
    r = binding_multislot_probe(model, cfg=cfg, device=device).to_dict()
    return {
        "steps": steps,
        "all_slots": round(float(r["binding_multislot_all_slots_acc"]), 4),
        "two_plus": round(float(r["binding_multislot_two_plus_slots_acc"]), 4),
        "held_class": round(float(r["binding_multislot_held_entity_class_acc"]), 4),
        "status": r.get("binding_multislot_status", "?"),
        "nonembed_params": nonembed,
        "probe_seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dim", type=int, default=256, help="quadrupled vs the dim128 nano base"
    )
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.005,
        help="all_slots rise needed to keep climbing",
    )
    ap.add_argument(
        "--timeout", type=float, default=6000.0, help="per-probe wall budget"
    )
    ap.add_argument("--out", default="research/reports/slot_param_step_sweep.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {label: [] for label, _ in LANES}
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

    for label, variant in LANES:
        prev = -1.0
        for steps in LADDER:
            row = _probe(
                variant,
                dim=args.dim,
                n_blocks=args.n_blocks,
                steps=steps,
                seed=args.seed,
                device=args.device,
                timeout=args.timeout,
            )
            results[label].append(row)
            flush()
            print(
                f"[{label}] steps={steps:>6} all_slots={row['all_slots']:.4f} "
                f"two_plus={row['two_plus']:.4f} ({row['status']}, {row['probe_seconds']}s, "
                f"nonembed={row['nonembed_params'] / 1e6:.2f}M)",
                flush=True,
            )
            # Adaptive stop: keep going only while all_slots keeps rising.
            gain = row["all_slots"] - prev
            if steps >= 5000 and gain < args.min_gain:
                print(
                    f"[{label}] plateaued at {steps} (gain {gain:+.4f} < {args.min_gain}); stopping lane",
                    flush=True,
                )
                break
            prev = row["all_slots"]

    flush()
    print(f"\nDONE. wrote {out} ({round(time.perf_counter() - started, 1)}s)")


if __name__ == "__main__":
    main()
